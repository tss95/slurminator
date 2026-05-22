# Status Callback And Metric Display

Slurminator can run without a training callback: scheduler state still moves
jobs through queued, running, and terminal states, and dashboard v4 will still
show submission state, resource settings, and SLURM log tails. What the
callback adds is everything that is internal to a training run: live progress,
scalar metrics, dashboard metric columns, trajectory sparklines, per-run plots,
and tracker links.

The callback writes validated status files under:

```text
$SAVE_PATH/.orchestrator_status[/sweep_<sweep_id>]/status_<job_id>.json
$SAVE_PATH/.orchestrator_status[/sweep_<sweep_id>]/history_<job_id>.jsonl
```

The dashboard ingests these files while jobs are running and projects the latest
values onto the experiment-state YAML. The schema is defined by
`slurminator.schemas.status_schema` and is currently at version 1.1.

## What Dashboard v4 Reads From The Callback

Most v4 features map to a specific callback knob. If a feature is empty in the
dashboard, this table is the first place to check.

| v4 feature                                  | What it needs from the callback                                            |
| ------------------------------------------- | -------------------------------------------------------------------------- |
| Home-table progress, speed, ETA             | `progress` block in `status_<job_id>.json` (lifecycle hooks fire normally) |
| Primary / secondary highlight columns       | `primary_metric` / `secondary_metric` *and* that key emitted into `metrics` |
| Any named metric column                     | An entry in `metric_info` *and* the key emitted into `metrics`             |
| Metric column color                         | `threshold` and `higher_better` on the column's `MetricDisplayCandidate`   |
| Best-so-far value in cell (parenthetical)   | `best_key` on the candidate *and* that best key emitted into `metrics`     |
| Trajectory sparkline in the home table      | Metric value present in `history_<job_id>.jsonl`                           |
| Per-run plot screen                         | Metric values present in the history file                                  |
| Plot x-axis unit (epoch vs step)            | `progress.unit` on the latest history entry / status                       |
| Best-overlay line on the plot               | `best_key` value emitted into history; `higher_better` controls direction  |
| Tracker link buttons (W&B, TensorBoard, …)  | A `_resolve_links()` override that returns the canonical keys              |

The materialization rule is strict: declaring a metric candidate does not put
it into `display.metric_info` until the numeric metric appears in `metrics`.
That keeps the dashboard from showing dead columns during `initializing`.

## Setup Checklist

Decide these before you ship a callback to a long-running cluster:

1. **Primary and secondary metrics.** Which one or two metrics belong in the
   highlight columns of the home table.
2. **Full display set.** Every other metric you want as a sortable, color-coded
   column. Anything not in `metric_info` still ends up in the status file but
   does not get a named column.
3. **History scope.** Which metrics you want to trend in sparklines and the
   per-run plot. The default appends every finite numeric metric; for projects
   with dozens of metrics per epoch you almost always want a narrower set.
4. **Progress unit.** Is the run measured in epochs or steps? Step-counted runs
   need `progress.unit="step"` or v4 plots against the wrong axis.
5. **Tracker link policy.** Which tracker URLs (W&B run, sweep, TensorBoard)
   should surface as link buttons.
6. **Best-key strategy.** Which metrics have a separately tracked best-so-far
   value, and whether the trainer already emits that best key alongside the
   live value.

The rest of this document walks through each item.

## Minimal Integration

Use `OrchestratorStatusCallback` directly when your trainer can call simple
lifecycle hooks:

```python
from slurminator.callbacks.status_callback import OrchestratorStatusCallback
from slurminator.callbacks.status_normalization import MetricDisplayCandidate

status_cb = OrchestratorStatusCallback(
    cfg=cfg,
    primary_metric="val/accuracy",
    secondary_metric="val/loss",
    metric_info={
        "val/accuracy": MetricDisplayCandidate(
            shortform="acc",
            higher_better=True,
            format=".2%",
            threshold=0.90,
            best_key="val/global_best_accuracy",
        ),
        "val/loss": MetricDisplayCandidate(
            shortform="loss",
            higher_better=False,
            format=".4f",
        ),
    },
)

status_cb.on_train_start(trainer)

for epoch in range(num_epochs):
    for batch_idx, batch in enumerate(train_loader):
        train_one_batch(batch)
        status_cb.on_train_batch_end(trainer, batch, batch_idx, logs={})

    train_logs = {"loss": train_loss}
    val_logs = {"accuracy": val_accuracy, "loss": val_loss}
    status_cb.on_epoch_end(trainer, epoch, train_logs, val_logs)

status_cb.on_train_end(trainer)
```

`on_epoch_end()` prefixes metric dictionaries:

- `train_logs={"loss": 0.2}` becomes metric key `train/loss`.
- `val_logs={"accuracy": 0.91}` becomes metric key `val/accuracy`.

Set `primary_metric`, `secondary_metric`, and `metric_info` to these final
metric keys, not the unprefixed log names.

## Identity And Paths

The callback needs a save path, job id, and experiment id. Pass them
explicitly:

```python
status_cb = OrchestratorStatusCallback(
    save_path="/cluster/work/my_project",
    job_id="123456",
    sweep_id="my_sweep",
    experiment_id="cifar10_adamw_s42",
)
```

If omitted, the callback resolves them from environment variables:

- `SAVE_PATH` for the output root.
- `SLURM_JOB_ID`, `PBS_JOBID`, or `JOB_ID` for the job id.
- `ORCHESTRATOR_SWEEP_ID` or `SWEEP_ID` for optional sweep grouping.
- `ORCHESTRATOR_EXPERIMENT_ID` or `EXPERIMENT_ID` for the displayed experiment.

The generic Slurminator launcher sets the scheduler-side variables. Project
launch code should ensure `SAVE_PATH` and, when useful, the orchestrator
experiment identifiers are exported into the job environment.

## Optional Config Context

The `cfg` constructor argument is optional. The base callback stores it as
`self.cfg`, and `on_train_start()` refreshes it from `trainer.cfg` or
`trainer.config` when either attribute exists:

```text
self.cfg = trainer.cfg or trainer.config or constructor_cfg
```

Use `cfg` as project context, not as a required Slurminator schema. Generic
callbacks can pass `cfg=None` and provide identity values through constructor
arguments or environment variables.

## Display Metrics: Primary, Secondary, And The Full Set

The main dashboard table discovers metric columns from, in order:

1. `primary_metric`,
2. `secondary_metric`,
3. every key in `metric_info`.

Columns with no value across the visible rows are dropped to keep the table
compact. Compact summary widgets may still show only primary and secondary
metrics; the full metric set is preserved in `metrics` and rendered in the
main table when values exist.

Declare every metric you want as a named column in `metric_info`, even if it
is the same as `primary_metric` or `secondary_metric`:

```python
status_cb = OrchestratorStatusCallback(
    cfg=cfg,
    primary_metric="val/accuracy",
    secondary_metric="val/loss",
    metric_info={
        "val/accuracy": MetricDisplayCandidate(
            shortform="acc",
            higher_better=True,
            format=".2%",
            threshold=0.90,
        ),
        "val/loss": MetricDisplayCandidate(
            shortform="loss",
            higher_better=False,
            format=".4f",
        ),
        "val/f1": MetricDisplayCandidate(
            shortform="f1",
            higher_better=True,
            format=".3f",
        ),
        "val/auroc": MetricDisplayCandidate(
            shortform="auc",
            higher_better=True,
            format=".3f",
            threshold=0.95,
        ),
    },
)
```

Then emit those metric keys through `on_epoch_end()` or `update_metrics()`:

```python
status_cb.on_epoch_end(
    trainer,
    epoch,
    train_logs={"loss": train_loss},
    val_logs={
        "accuracy": val_accuracy,
        "loss": val_loss,
        "f1": val_f1,
        "auroc": val_auroc,
    },
)
```

Metrics that are emitted but not listed in `metric_info` are still preserved
in the status file and experiment-state YAML. They are not promoted to named
dashboard columns unless you declare display metadata for them.

### Stable Keys And Shortforms

Use stable metric keys. Renaming a metric halfway through a run creates a new
dashboard column instead of updating the old one.

`shortform` is not purely cosmetic: it is also a history-lookup alias. When v4
resolves a sparkline or table cell for a metric, it first looks up the full
key in the history file, and if it does not find it, walks `metric_info`
looking for an entry whose `shortform` matches. Pick shortforms that are
distinctive within a run, and keep them stable across runs that you want to
compare.

## Metric Display Metadata

`MetricDisplayCandidate` fields and their effects:

| Field           | Effect                                                                                          |
| --------------- | ----------------------------------------------------------------------------------------------- |
| `shortform`     | Column header in the home table; also serves as a history-lookup alias.                          |
| `higher_better` | Drives column sort direction, color polarity, sparkline trend color, and plot best-overlay sign. |
| `format`        | Python format spec (e.g. `.2%`, `.4f`, `.3f`), or the literal `"integer"` for rounded integers.  |
| `threshold`     | Cells render green at or beyond the threshold, red on the wrong side. Combined with `higher_better`. |
| `best_key`      | Metric key whose value renders in parentheses next to the current value (see *Best-So-Far*).     |

Format examples that show up frequently:

- `".2%"` → `0.913` displayed as `91.30%`.
- `".4f"` → `0.7234567` displayed as `0.7235`.
- `".3f"` → `0.7234567` displayed as `0.723`.
- `"integer"` → values are rounded to the nearest integer.

If `format` is omitted, the dashboard picks a default numeric format. If
`higher_better` is omitted, the dashboard treats the metric as higher-is-better
for sort and color purposes; set it explicitly for losses, errors, or any
metric where lower is better.

## History And Plots

The history file is what makes the home-table trajectory sparklines and the
per-run plot screen useful. Each line is a `HistoryEntry` JSON object with
`timestamp`, `attempt`, `epoch`, `step`, `unit`, and a `metrics` mapping.

By default, the callback appends every finite numeric metric in
`status.metrics` to the history file. For projects with dozens of metrics per
epoch, that produces noisy plot menus and large history files. Override
`_history_metrics()` to scope the history to the metrics you actually want to
trend:

```python
class ProjectStatusCallback(OrchestratorStatusCallback):
    def _history_metrics(self, status):
        keep = {
            "train/loss",
            "val/loss",
            "val/accuracy",
            "val/global_best_accuracy",
        }
        return {key: value for key, value in status.metrics.items() if key in keep}
```

Two things to remember when picking the history scope:

- A metric needs to be in history for its sparkline to render, even if it is
  already a named column.
- If you want a best-overlay in the per-run plot for a metric, the metric's
  `best_key` value must also be in history.

### Plot Axis Unit

The per-run plot reads `progress.unit` to decide whether to plot against
epochs or steps. Epoch-driven trainers do not need to do anything: the default
progress snapshot emits `unit="epoch"`.

Step-counted runs should override `_build_progress_snapshot()` and emit
`unit="step"`, otherwise the plot will use whatever epoch field is available
and step-only metrics will look stair-stepped or empty:

```python
from slurminator.callbacks.status_callback import OrchestratorStatusCallback
from slurminator.callbacks.status_normalization import GenericProgressSnapshot


class StepProgressStatusCallback(OrchestratorStatusCallback):
    def _build_progress_snapshot(self, trainer, *, epoch=None, epoch_completed=False):
        return GenericProgressSnapshot(
            unit="step",
            current_step=int(trainer.global_step),
            total_steps=int(trainer.max_steps),
            speed_value=self._it_per_sec_ema,
            speed_unit="it/sec",
        )
```

History entries carry their own `unit` field as of schema v1.1, so a run that
genuinely changes axis (rare) does not retroactively rewrite older entries.

### Relaunch And `attempt`

When a job is relaunched and writes to an existing history file, the callback
detects the existing history and increments `attempt` for new entries. The
status file `attempt` also bumps. v4 keeps the full history across attempts;
expect a visible discontinuity at the attempt boundary if the metric jumped
when the run restarted from checkpoint (or from scratch). This is informative,
not a bug.

## Best-So-Far Values

If a metric has a separately tracked best-so-far value, set `best_key` in its
`MetricDisplayCandidate`. The dashboard renders the current value plus the
best value in parentheses, and the per-run plot can overlay the best track.

The contract is two-sided: declaring `best_key` is not enough. The trainer
must also emit that best key into `metrics` (and, if you want a plot overlay,
into history). For example:

```python
status_cb = OrchestratorStatusCallback(
    cfg=cfg,
    primary_metric="val/accuracy",
    metric_info={
        "val/accuracy": MetricDisplayCandidate(
            shortform="acc",
            higher_better=True,
            format=".2%",
            best_key="val/global_best_accuracy",
        ),
    },
)

# ... later, at validation time:
status_cb.update_metrics(
    {
        "val/accuracy": current_accuracy,
        "val/global_best_accuracy": max(best_so_far, current_accuracy),
    },
    trainer=trainer,
)
```

If `best_key` is declared but never appears in `metrics`, the dashboard cell
silently falls back to just the current value. That is the most common reason
the parenthetical does not render.

## Tracker Links

Subclass `_resolve_links()` to attach tracker or artifact links:

```python
class TrackerStatusCallback(OrchestratorStatusCallback):
    def _resolve_links(self, trainer):
        links = {}
        run_url = getattr(getattr(trainer, "tracker", None), "run_url", None)
        if run_url:
            links["tracker_run_url"] = run_url
        return links
```

The dashboard treats link values as opaque strings. Use these canonical keys
so v4 (and external tooling that reads the status file) recognizes them:

| Key                 | Intended target                                           |
| ------------------- | --------------------------------------------------------- |
| `tracker_run_url`   | Generic per-run tracker URL.                              |
| `tracker_sweep_url` | Generic per-sweep tracker URL.                            |
| `wandb_run_url`     | Weights & Biases run URL.                                 |
| `tensorboard_url`   | TensorBoard URL when one is reachable from the dashboard. |

You may attach project-specific keys as well; the dashboard preserves unknown
keys but will not surface them as named link buttons.

## Custom Progress Semantics

The base callback reports epoch progress by default and also records step
counts when it can read `trainer.global_step` and `trainer.max_train_steps`.

Subclass `_build_progress_snapshot()` when your project uses a different
primary axis (see *Plot Axis Unit* above). The package does not assign
semantics to your training loop's epochs, validation intervals, or probe
schedules. Decide those semantics in your project callback, then emit a
normalized progress snapshot.

## Late Or External Metrics

Use `update_metrics()` when a metric is produced outside the epoch hook:

```python
status_cb.update_metrics(
    {
        "test/accuracy": test_accuracy,
        "test/loss": test_loss,
    },
    trainer=trainer,
)
```

This is useful for final test metrics, probe metrics, expensive validation
passes, or framework callbacks that report metrics after the main training
loop. The write goes through the same throttle and history append path as
other metric updates.

## Project-Specific Metric Selection

Use constructor arguments when every run uses the same metric keys. Subclass
the callback when metric selection depends on task type, dataset, model
family, or a project config object.

Override `_resolve_display_candidates()` to set the display metric policy:

```python
from slurminator.callbacks.status_callback import OrchestratorStatusCallback
from slurminator.callbacks.status_normalization import MetricDisplayCandidate


class ProjectStatusCallback(OrchestratorStatusCallback):
    def _resolve_display_candidates(self, trainer) -> None:
        super()._resolve_display_candidates(trainer)

        cfg = getattr(trainer, "cfg", None) or getattr(trainer, "config", None) or self.cfg
        task_type = getattr(cfg, "task_type", "classification")

        if task_type == "classification":
            self._primary_metric = "val/accuracy"
            self._secondary_metric = "val/loss"
            self._metric_info.update(
                {
                    "val/accuracy": MetricDisplayCandidate(
                        shortform="acc", higher_better=True, format=".2%"
                    ),
                    "val/loss": MetricDisplayCandidate(
                        shortform="loss", higher_better=False, format=".4f"
                    ),
                    "val/f1": MetricDisplayCandidate(
                        shortform="f1", higher_better=True, format=".3f"
                    ),
                }
            )
        elif task_type == "forecasting":
            self._primary_metric = "val/mae"
            self._secondary_metric = "val/mse"
            self._metric_info.update(
                {
                    "val/mae": MetricDisplayCandidate(
                        shortform="mae", higher_better=False, format=".4f"
                    ),
                    "val/mse": MetricDisplayCandidate(
                        shortform="mse", higher_better=False, format=".4f"
                    ),
                }
            )
```

The same strict materialization rule applies: declaring a metric candidate
does not write it into `display.metric_info` until the numeric metric appears
in `metrics`. This lets you declare the expected display set at train start
without creating orphan display entries.

## Failure Behavior And State Machine

The callback writes files atomically using a temporary file followed by
`os.replace()`, so readers should never see a partial JSON write.

The live callback status state machine is intentionally small:

```text
initializing -> running -> completed
initializing -> completed
```

Backward transitions are rejected (the callback raises rather than silently
regressing). Scheduler-owned terminal states such as timeout, cancellation,
out-of-memory, or kill are handled by the orchestrator on the login node. The
callback reports the training process's live state and metrics, not the
scheduler's final verdict.

Status writes are throttled by `min_write_interval_seconds` (constructor
argument). The same throttle gates history appends triggered from batch hooks,
so a tight inner loop does not produce a write per step. Epoch boundaries and
explicit `update_metrics()` calls always flush.

## Status Schema v1.1

Schema v1.1 adds:

- `attempt` on `OrchestratorStatus` and `HistoryEntry`. Incremented on
  relaunch against an existing history file.
- `progress.unit` ("epoch" or "step") and the same `unit` field on each
  `HistoryEntry`, so plots can choose the right x-axis without inspecting
  config.

Compatibility is forward-asymmetric: a v1.1 reader accepts v1.0 status files
(missing fields default cleanly), but a v1.0 reader is not expected to accept
v1.1 files. Anyone consuming the JSONL externally should pin to v1.1 readers.

## Troubleshooting

**A metric column shows `—` everywhere.** Either the metric is not yet in
`status.metrics` for any visible run, or it is missing from `metric_info`.
Check that the prefixed key (e.g. `val/accuracy`, not `accuracy`) matches what
the trainer actually emits.

**The trajectory sparkline is empty even though the column has values.** The
metric is not in the history file. Default behavior appends every numeric
metric, so the usual cause is a custom `_history_metrics()` that filters it
out. If the metric only appears via `update_metrics()` between epochs, confirm
the throttle is not eating the write — `min_write_interval_seconds=0.0` is
common during debugging.

**The per-run plot says "No history available" for a running job.** The
history file has not been written yet. The first write happens at the first
batch hook or `update_metrics()` call after `on_train_start()`. For very short
jobs, increase the visibility of metric emission rather than waiting for
batch ticks.

**No best-overlay on the plot, or no parenthetical in the cell.** The
`best_key` is declared but the trainer is not emitting that key into
`metrics`. Add the best value to the same `update_metrics()` call (or
`val_logs`) that emits the live value.

**Plot x-axis says "epoch" for a step-counted run.** `progress.unit` is
defaulting to epoch. Override `_build_progress_snapshot()` to emit
`unit="step"` (see *Plot Axis Unit*).

**The plot shows a discontinuity after a relaunch.** Expected: the run
restarted and `attempt` incremented. v4 keeps all attempts in history. If the
discontinuity is not informative, archive the old history file before
relaunching.

**A column appears for a metric but won't sort the way you want.**
`higher_better` is not set. The dashboard defaults to higher-is-better when
the field is missing; set it explicitly on losses and errors.

**The status file exists but the dashboard shows scheduler state only.** Most
likely `SAVE_PATH` differs between the orchestrator and the job. Confirm the
file path the callback logs at `on_train_start()` matches what the
orchestrator polls for that experiment row.

# Status Callback And Metric Display

Slurminator can run without a training callback: scheduler state still moves
jobs through queued, running, and terminal states. The callback adds live
training progress, scalar metrics, dashboard metric columns, and tracker links.

The callback writes validated status files under:

```text
$SAVE_PATH/.orchestrator_status[/sweep_<sweep_id>]/status_<job_id>.json
```

The dashboard ingests these files while jobs are running and projects the latest
values onto the experiment-state YAML.

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

Set `primary_metric`, `secondary_metric`, and `metric_info` to these final metric
keys, not the unprefixed log names.

## Identity And Paths

The callback needs a save path, job id, and experiment id. You may pass them
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

- `SAVE_PATH`
- `SLURM_JOB_ID`, `PBS_JOBID`, or `JOB_ID`
- `ORCHESTRATOR_SWEEP_ID` or `SWEEP_ID`
- `ORCHESTRATOR_EXPERIMENT_ID` or `EXPERIMENT_ID`

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

If a subclass depends on config fields, resolve them defensively and fail with a
clear project error when they are required:

```python
def _project_cfg(self, trainer):
    return getattr(trainer, "cfg", None) or getattr(trainer, "config", None) or self.cfg
```

## Metric Keys And Display Metadata

Status files contain two related metric sections:

- `metrics`: flat mapping from metric key to finite numeric value.
- `display`: dashboard hints, including `primary_metric`, `secondary_metric`,
  and `metric_info`.

`primary_metric` and `secondary_metric` are not a two-metric limit. They are
highlight references used for sorting, quick summaries, and compact views. The
full display metric set is declared by `metric_info`, and all numeric values are
kept in `metrics`.

Metric values must be JSON numbers. Booleans, strings, nested dictionaries,
lists, NaN, and infinity are ignored or rejected.

Display metadata is strict: a display key is materialized only after the
corresponding numeric metric exists. This means the first `initializing` status
usually has no primary metric column yet. Once `val/accuracy` appears in
`metrics`, the callback can safely materialize:

```json
{
  "metrics": {
    "val/accuracy": 0.91
  },
  "display": {
    "primary_metric": "val/accuracy",
    "metric_info": {
      "val/accuracy": {
        "shortform": "acc",
        "higher_better": true,
        "format": ".2%",
        "threshold": 0.9
      }
    }
  }
}
```

Use stable metric keys. Renaming a metric halfway through a run creates a new
dashboard column instead of updating the old one.

## Showing More Than Two Metrics

Declare every metric you want the main dashboard table to consider in
`metric_info`:

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

The main v3 dashboard table discovers metric columns from:

1. `primary_metric`,
2. `secondary_metric`,
3. every key in `metric_info`.

Columns with no value across the visible rows are dropped to keep the table
compact. Compact summary tables may still show only primary and secondary
metrics; the full metric set is preserved in `metrics` and rendered in the main
table when values exist.

Metrics that are emitted but not listed in `metric_info` are still preserved in
the status file and experiment-state YAML. They are not promoted to named
dashboard columns unless you declare display metadata for them.

If a metric has a separately tracked best-so-far value, set `best_key` in its
`MetricDisplayCandidate`. Dashboard metric cells render the current value plus
the best value in parentheses when that key is present in the status metrics.

## Project-Specific Metric Selection

Use constructor arguments when every run uses the same metric keys. Subclass the
callback when metric selection depends on task type, dataset, model family, or a
project config object.

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
                        shortform="acc",
                        higher_better=True,
                        format=".2%",
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
                }
            )
        elif task_type == "forecasting":
            self._primary_metric = "val/mae"
            self._secondary_metric = "val/mse"
            self._metric_info.update(
                {
                    "val/mae": MetricDisplayCandidate(
                        shortform="mae",
                        higher_better=False,
                        format=".4f",
                    ),
                    "val/mse": MetricDisplayCandidate(
                        shortform="mse",
                        higher_better=False,
                        format=".4f",
                    ),
                }
            )
```

The same strict materialization rule applies: declaring a metric candidate does
not write it into `display.metric_info` until the numeric metric appears in
`metrics`. This lets you declare the expected display set at train start without
creating orphan display entries.

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
passes, or framework callbacks that report metrics after the main training loop.

## Custom Progress Semantics

The base callback reports epoch progress by default and also records step counts
when it can read `trainer.global_step` and `trainer.max_train_steps`.

Subclass `_build_progress_snapshot()` when your project uses a different primary
axis, such as a step-counted run:

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

The package does not assign semantics to your training loop's epochs, validation
intervals, or probe schedules. Decide those semantics in your project callback,
then emit a normalized progress snapshot.

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

The dashboard treats links as opaque strings. Common keys are
`tracker_run_url`, `tracker_sweep_url`, `wandb_run_url`, and `tensorboard_url`.

## Failure Behavior

The callback writes files atomically using a temporary file followed by
`os.replace()`, so readers should never see a partial JSON write.

The live callback status state machine is intentionally small:

```text
initializing -> running -> completed
initializing -> completed
```

Scheduler-owned terminal states such as timeout, cancellation, out-of-memory, or
kill are handled by the orchestrator on the login node. The callback reports the
training process's live state and metrics, not the scheduler's final verdict.

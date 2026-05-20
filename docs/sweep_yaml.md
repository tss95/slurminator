# Sweep And Experiment YAML

Slurminator uses two related YAML formats:

- **Experiment-state YAML** is the runnable file passed with `--yaml`. It has a
  top-level `experiments:` list and is updated by the orchestrator as jobs are
  submitted, polled, retried, and completed. It is also the durable run ledger:
  Slurminator writes scheduler ids, output paths, resource requests, timestamps,
  status, live metrics, and links back into the same file.
- **Custom-sweep YAML** is the generator input passed with `--sweepfile`. It
  describes datasets, cases, seeds, and parameter sweeps. Slurminator expands it
  into an experiment-state YAML under `experiment_lists/`.

Use `--sweepfile` to create a new experiment list. Use `--yaml` to run or resume
an existing experiment list.

## Experiment-State YAML

The minimum runnable state file is:

```yaml
metadata:
  project_git_sha: null
  slurminator_git_sha: null
experiments:
  - experiment_id: smoke_001
    status: pending
    task_type: train
    dataset_name: smoke
    extra_command: "python train.py --config cfg/smoke.yaml --orchestrator"
```

One state file can contain many unrelated experiments. They do not have to come
from the same sweep, dataset, task type, or command style:

```yaml
experiments:
  - experiment_id: image_smoke_resnet
    status: pending
    task_type: classification
    dataset_name: cifar10
    config: configs/cifar10_resnet.yaml

  - experiment_id: tabular_smoke_xgb
    status: pending
    task_type: tabular
    dataset_name: fraud_small
    extra_command: "python train_tabular.py --config configs/fraud_xgb.yaml"

  - experiment_id: text_smoke_transformer
    status: pending
    task_type: language_modeling
    dataset_name: tiny_text
    command: "python train_text.py --dataset tiny_text --max-steps 500"
    resource_overrides:
      gpu_count: 1
      memory_gb: 40
      time_hours: 2
```

This is the simplest way to run several independent canaries under one
dashboard. The orchestrator treats every row independently and uses
`experiment_id` as the stable row key.

Generated experiment-state YAML includes provenance metadata for reproducible
launches. `metadata.project_git_sha` records the project repository `HEAD` when
the ledger was created, and `metadata.slurminator_git_sha` records the
Slurminator repository `HEAD`. Either field may be `null` when the corresponding
directory is not a git checkout, `git` is unavailable, or SHA capture otherwise
fails. On submission, each row gains `git_sha_at_submission` with `project` and
`slurminator` subkeys so resumed or relaunched rows can record the code state
active for that specific submission.

Each row is one schedulable job. Slurminator owns the scheduler fields and may
rewrite them while running:

- `status`: `pending`, `queued`, `running`, `completed`, `failed`, `partial`,
  `timeout`, `out_of_memory`, `cancelled`, or `killed`.
- `job_id`: scheduler job id after submission.
- `hpc_assignment`: cluster selected for the job.
- `queued_timestamp`, `running_timestamp`, `completed_timestamp`: lifecycle
  timestamps when available.
- `output_dir`, `save_path`, `requested_time_hours`, `requested_ram_gb`,
  `requested_gpu_count`: resolved runtime details.
- `git_sha_at_submission`: best-effort project and Slurminator git SHAs captured
  just before submission.
- `metrics`, `display_metric_info`, `links`, and progress fields populated from
  status callbacks.

The `job_id` and `output_dir` fields are enough to recover the corresponding
SLURM stdout/stderr files later (`slurm-<job_id>.out` and
`slurm-<job_id>.err` under the recorded output directory). Status callbacks and
external trackers such as W&B can add richer live metrics and links, but the
experiment-state YAML remains the scheduling record Slurminator manages.

### Generated Run Ledger Example

A generated run typically has this shape:

```text
experiment_lists/
  experiments_20260511_032235.yaml
  outputs/
    experiments_20260511_032235/
      <experiment_id>/
        slurm-<job_id>.out
        slurm-<job_id>.err
```

The top-level `experiments_20260511_032235.yaml` file is the file you pass back
to Slurminator with `--yaml`. It contains the run history and current scheduler
state. A small canary run with that name had four rows and recorded, for each
row:

- the original intent: `experiment_id`, `task_type`, `dataset_name`,
  `sweep_params`, and project metadata such as run name and seed;
- Slurminator state: `status`, `hpc_assignment`, `job_id`, output directory,
  requested time/RAM/GPU count, and lifecycle timestamps;
- live/tracker state: W&B run id/URL, step and epoch counters, metrics, and
  display metadata when the status callback had emitted them.

If a run is cancelled or the orchestrator process exits before a final scheduler
poll, the YAML may show the last known state rather than the final terminal
state. Relaunching with `--yaml experiment_lists/experiments_20260511_032235.yaml`
lets Slurminator refresh rows from scheduler state and the recorded log/status
locations, as long as those artifacts are still visible.

Project-owned fields can also live on the row. The package treats unknown row
fields as data for plugins, command builders, or downstream tools. Common
generic fields are:

- `extra_command` or `command`: full command to run.
- `config` or `config_path`: config path consumed by `SimpleCommandPlugin`.
- `sweep_params`: semicolon-separated override string passed to the command
  builder when configured.
- `resource_overrides`: row-level resource overrides such as `gpu_count`,
  `memory_gb`, or `time_hours`.
- `pinned_hpc`: force a row to one cluster.
- `ensure_dirs`: extra directories to create before submission.
- `metadata`: free-form project metadata.

## Resuming With `--yaml`

`--yaml` is the pick-up-where-you-left-off path. If the orchestrator exits while
jobs are still queued or running, launch it again with the generated
`experiment_lists/experiments_*.yaml` file:

```bash
python -m slurminator \
  --yaml experiment_lists/experiments_20260511_032235.yaml \
  --olivia-limit 4 \
  --n-gpus 1
```

On restart Slurminator reloads the YAML, reconnects to the scheduler, and
continues from the recorded row state:

- `pending` and `partial` rows may be submitted when concurrency is available.
- `queued` and `running` rows are polled by their recorded `job_id`.
- terminal rows such as `completed`, `failed`, `timeout`, `out_of_memory`,
  `cancelled`, and `killed` are left alone.
- if a row is marked `queued` or `running` but has no `job_id`, orphan recovery
  resets it to `pending` so it can be submitted again.

Do not resume by re-running `--sweepfile`; that creates a new experiment-state
file. Resume with the generated `experiment_lists/experiments_*.yaml` file
instead.

You may change concurrency flags when resuming. Be careful not to set the active
cluster limit to zero for a YAML that still has live jobs on that cluster,
because disabled assignments can be reset to `pending`.

While Slurminator is running, it reloads the experiment-state YAML each poll
cycle. That means the file is an operational control surface as well as a record:
if you set a finished or failed row back to `status: pending`, Slurminator will
consider it available for relaunch when concurrency permits. For clean
bookkeeping, also remove stale scheduler fields such as `job_id`, timestamps,
and output paths; otherwise new submission metadata will overwrite the fields it
owns as the row is relaunched.

To intentionally rerun a finished experiment after Slurminator has stopped, make
the same edit and restart with the generated file:
`--yaml experiment_lists/experiments_*.yaml`.

## Custom-Sweep YAML

A generic custom sweep file is a mapping with `custom_sweeps:`:

```yaml
custom_sweeps:
  - experiment_prefix: optimizer_ablation
    task_type: train
    datasets: [cifar10]
    seeds: [42]
    num_epochs: 10
    base_overrides:
      trainer.max_epochs: 10
    sweep_keys:
      optimizer.name: [adamw, sgd]
      optimizer.lr: [0.001, 0.0003]
```

This expands as:

```text
datasets x Cartesian product(sweep_keys) x seeds
```

For the example above, Slurminator creates four rows for `cifar10`: two
optimizer names times two learning rates, each with seed `42`.

Important fields:

- `datasets`: list of datasets to run. Use `dataset_name` for a single dataset.
- `experiment_prefix`: prefix used in generated experiment ids.
- `task_type`: opaque task taxonomy string passed through to the experiment row.
- `base_overrides`: overrides applied to every generated row.
- `sweep_keys`: mapping from override key to a list of values. Values are
  expanded as a Cartesian product.
- `cases`: named variants with explicit override dictionaries.
- `seeds`: exact seed list to generate for every row.
- `num_seeds`: truncate or extend the default seed list to this count. If
  `seeds` is explicitly provided, `num_seeds` must match its length.
- `num_epochs`: default epoch count copied into generated rows when your
  project command uses epoch-based training.
- `config_profile`: project profile metadata copied into generated rows.
- `run_name_prefix`, `run_name_suffix`, and `parameters_prefix`: naming helpers.
- `resume_from`: scalar checkpoint path, or a dataset-scoped mapping.

Override keys may use dot notation (`trainer.max_steps`) or double-underscore
notation (`trainer__max_steps`). The generator normalizes double underscores to
dots.

Sweep override keys are otherwise opaque. Slurminator builds the override string
and passes it to the configured command builder; your training code or project
plugin decides what the keys mean.

## Override Consumption Contract

Custom-sweep generation is intentionally separate from project configuration
loading. Slurminator expands `base_overrides`, `sweep_keys`, and `cases` into a
row-level `sweep_params` string:

```yaml
experiments:
  - experiment_id: optimizer_ablation_cifar10_adamw_lr0p001_s42
    status: pending
    task_type: train
    dataset_name: cifar10
    config: configs/cifar10.yaml
    sweep_params: "optimizer.name=adamw;optimizer.lr=0.001;trainer.max_epochs=10"
```

Slurminator does not mutate your framework config object. One of the command
paths must consume `sweep_params` and apply it inside your training program:

- Explicit row commands can include the override flag themselves:
  `extra_command: "python train.py --config configs/cifar10.yaml --overrides 'optimizer.lr=0.001'"`.
- `SimpleCommandPlugin` can forward the generated string when configured with a
  sweep-params argument, for example
  `--simple-command-sweep-params-arg "--overrides"`.
- A project plugin can implement `build_commands_line()` and append
  `exp["sweep_params"]` to whatever CLI your project already uses.
- A project-specific generator can instead write fully resolved config files and
  put those file paths in `config`/`config_path`.

If none of these paths exists, the sweep still submits jobs, but those jobs run
the base config because no process consumed the generated overrides. Treat
"training entrypoint accepts config overrides" as part of the project contract,
not as something Slurminator can infer.

### Minimal Target-Script Support

Your training script needs a CLI argument that accepts the forwarded
`sweep_params` string and applies it to your own config system. The argument name
is project-defined; it only has to match the command builder. For example, this
Slurminator command:

```bash
python -m slurminator \
  --yaml experiment_lists/small.yaml \
  --simple-command-entrypoint "python train.py" \
  --simple-command-config-arg "--config" \
  --simple-command-sweep-params-arg "--overrides"
```

requires `train.py` to define and consume `--overrides`:

```python
import argparse
from pathlib import Path
from typing import Any

import yaml


def parse_sweep_params(raw: str | None) -> dict[str, Any]:
    """Parse Slurminator's semicolon-separated key=value override string."""
    overrides: dict[str, Any] = {}
    if not raw:
        return overrides
    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid override {item!r}; expected key=value.")
        key, value = item.split("=", 1)
        overrides[key.strip()] = yaml.safe_load(value)
    return overrides


def set_dotted(config: dict[str, Any], key: str, value: Any) -> None:
    """Apply a dotted-path override to a nested dictionary config."""
    target = config
    parts = key.split(".")
    for part in parts[:-1]:
        existing = target.setdefault(part, {})
        if not isinstance(existing, dict):
            raise ValueError(f"Cannot set {key!r}; {part!r} is not a mapping.")
        target = existing
    target[parts[-1]] = value


parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--overrides", default="", help="Slurminator sweep_params string.")
args = parser.parse_args()

config = yaml.safe_load(Path(args.config).read_text())
for key, value in parse_sweep_params(args.overrides).items():
    set_dotted(config, key, value)

train(config)
```

Projects that already use Hydra, OmegaConf, Pydantic settings, or another
configuration layer should apply the same contract through that system instead
of copying this dictionary helper. The important invariant is that every
generated key in `sweep_params` is either accepted and applied, or rejected with
a clear error before the run starts.

## Multiple Sweep Blocks

One custom-sweep YAML can describe several separate experiment families:

```yaml
custom_sweeps:
  - experiment_prefix: optimizer_ablation
    task_type: train
    datasets: [cifar10, fashion_mnist]
    seeds: [42]
    base_overrides:
      trainer.max_epochs: 20
    sweep_keys:
      optimizer.name: [adamw, sgd]
      optimizer.lr: [0.001, 0.0003]

  - experiment_prefix: model_size_canary
    task_type: train
    datasets: [cifar10]
    seeds: [42, 45]
    cases:
      - name: tiny
        overrides:
          model.width: 64
          model.depth: 4
      - name: small
        overrides:
          model.width: 128
          model.depth: 6

  - experiment_prefix: eval_only
    task_type: evaluation
    dataset_name: cifar10
    seeds: [42]
    cases:
      - name: baseline_checkpoint
        resume_from: /path/to/baseline.ckpt
        overrides:
          eval.split: test
```

Each block expands independently. The resulting rows are written into one
experiment-state YAML, so they can be submitted, monitored, paused, and resumed
together.

## Multiple Datasets

Use `datasets` to run one sweep block across several datasets:

```yaml
custom_sweeps:
  - experiment_prefix: backbone_canary
    task_type: train
    datasets: [cifar10, fashion_mnist, mnist]
    seeds: [42, 45]
    sweep_keys:
      model.backbone: [small, base]
```

This expands as:

```text
3 datasets x 2 backbone values x 2 seeds = 12 experiment rows
```

Dataset-scoped values can use a mapping with exact dataset names plus optional
`default` or `*` fallbacks:

```yaml
custom_sweeps:
  - experiment_prefix: resumed_probe
    datasets: [cifar10, fashion_mnist]
    resume_from:
      cifar10: /path/to/cifar10.ckpt
      fashion_mnist: /path/to/fashion_mnist.ckpt
```

If a dataset-scoped mapping omits the current dataset and has no `default` or
`*`, generation fails hard. This avoids silently running a dataset with the
wrong checkpoint or resource assumption.

## Named Cases And Nested Variants

Use `cases` when each experiment variant has a meaningful name or a nested set
of overrides:

```yaml
custom_sweeps:
  - experiment_prefix: augmentation_ablation
    datasets: [cifar10]
    seeds: [42]
    base_overrides:
      trainer.max_epochs: 20
    cases:
      - name: color_only
        overrides:
          augmentation.color_jitter: true
          augmentation.random_crop: false
      - name: crop_only
        overrides:
          augmentation.color_jitter: false
          augmentation.random_crop: true
      - name: combined
        base_overrides:
          augmentation.color_jitter: true
        overrides:
          augmentation.random_crop: true
```

Cases are useful for nested experiments because each case can carry its own
`base_overrides`, `overrides`, `resume_from`, and project-specific extension
fields consumed by your adapter.

Current expansion semantics are deliberately simple:

- If a sweep block has `cases`, Slurminator emits one row per case, dataset, and
  seed.
- If the same block also has `sweep_keys`, Slurminator also emits the Cartesian
  `sweep_keys` rows as sibling variants.
- It does **not** multiply `cases x sweep_keys`. If you need named nested
  variants, write those combinations as explicit `cases`.

## Epochs, Steps, And Progress

Slurminator's sweep generator does not train models itself. Values such as
`trainer.max_epochs`, `trainer.max_steps`, or `scheduler.warmup_steps` are just
override keys passed through to your project command.

Use `num_epochs` when you want generated rows to carry a default epoch horizon:

```yaml
custom_sweeps:
  - experiment_prefix: short_canary
    datasets: [cifar10]
    num_epochs: 5
    sweep_keys:
      optimizer.lr: [0.001, 0.0003]
```

For live dashboard progress, the status callback is the source of truth. It
writes either epoch-based or step-based progress:

```python
from slurminator.callbacks.status_normalization import GenericProgressSnapshot

progress = GenericProgressSnapshot(
    unit="step",
    current_step=global_step,
    total_steps=max_steps,
)
```

Project adapters may add convenience rules for deriving an epoch horizon from
their own training-limit fields, but that is adapter policy. The portable
contract is: sweep YAML passes opaque overrides, and callbacks report actual
progress.

## Practical Workflow

1. Write or update a custom-sweep YAML.
2. Dry-run generation:

   ```bash
   python -m slurminator --sweepfile cfg/sweeps/my_sweep.yaml --olivia-limit 1 --dry-run
   ```

3. Inspect the generated `experiment_lists/<name>.yaml`.
4. Launch the generated list, either directly from the first command or by
   passing the generated file with `--yaml`.
5. If the orchestrator stops or your session disconnects, resume with `--yaml`
   and the same generated `experiment_lists/experiments_*.yaml` file.

For live metric collection and dashboard metric-key configuration, see
[`docs/status_callback.md`](status_callback.md).

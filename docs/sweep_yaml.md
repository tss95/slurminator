# Sweep And Experiment YAML

Slurminator uses two related YAML formats:

- **Experiment-state YAML** is the runnable file passed with `--yaml`. It has a
  top-level `experiments:` list and is updated by the orchestrator as jobs are
  submitted, polled, retried, and completed.
- **Custom-sweep YAML** is the generator input passed with `--sweepfile`. It
  describes datasets, cases, seeds, and parameter sweeps. Slurminator expands it
  into an experiment-state YAML under `experiment_lists/`.

Use `--sweepfile` to create a new experiment list. Use `--yaml` to run or resume
an existing experiment list.

## Experiment-State YAML

The minimum runnable state file is:

```yaml
experiments:
  - experiment_id: smoke_001
    status: pending
    task_type: train
    dataset_name: smoke
    extra_command: "python train.py --config cfg/smoke.yaml --orchestrator"
```

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
- `metrics`, `display_metric_info`, `links`, and progress fields populated from
  status callbacks.

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
jobs are still queued or running, launch it again with the same experiment-state
YAML:

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
file. Resume with the generated `experiment_lists/<name>.yaml` file instead.

You may change concurrency flags when resuming. Be careful not to set the active
cluster limit to zero for a YAML that still has live jobs on that cluster,
because disabled assignments can be reset to `pending`.

To intentionally rerun a finished experiment, copy the YAML or edit that row:
set `status: pending` and remove stale scheduler fields such as `job_id`,
timestamps, and output paths.

## Custom-Sweep YAML

A generic custom sweep file is a mapping with `custom_sweeps:`:

```yaml
custom_sweeps:
  - experiment_prefix: har_loss_ab
    task_type: self_supervised
    datasets: [HAR]
    seeds: [42]
    num_epochs: 10
    base_overrides:
      training_configs.max_train_steps: 1000
      training_configs.pseudo_epoch_steps: 100
    sweep_keys:
      loss.alpha: [0.0, 0.5]
      loss.beta: [0.0, 0.5]
```

This expands as:

```text
datasets x Cartesian product(sweep_keys) x seeds
```

For the example above, Slurminator creates four rows for `HAR`: `(alpha=0,
beta=0)`, `(alpha=0, beta=0.5)`, `(alpha=0.5, beta=0)`, and `(alpha=0.5,
beta=0.5)`, each with seed `42`.

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
- `num_epochs`: default epoch count when no explicit step-budget horizon is
  inferable.
- `config_profile`: project profile metadata copied into generated rows.
- `run_name_prefix`, `run_name_suffix`, and `parameters_prefix`: naming helpers.
- `resume_from`: scalar checkpoint path, or a dataset-scoped mapping.

Override keys may use dot notation (`training_configs.max_train_steps`) or
double-underscore notation (`training_configs__max_train_steps`). The generator
normalizes double underscores to dots.

## Multiple Datasets

Use `datasets` to run one sweep block across several datasets:

```yaml
custom_sweeps:
  - experiment_prefix: backbone_canary
    task_type: train
    datasets: [HAR, FordA, FordB]
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
    datasets: [HAR, FordA]
    checkpoint_probe: true
    resume_from:
      HAR: /path/to/har.ckpt
      FordA: /path/to/forda.ckpt
```

If a dataset-scoped mapping omits the current dataset and has no `default` or
`*`, generation fails hard. This avoids silently running a dataset with the
wrong checkpoint or resource assumption.

## Named Cases And Nested Variants

Use `cases` when each experiment variant has a meaningful name or a nested set
of overrides:

```yaml
custom_sweeps:
  - experiment_prefix: loss_ab
    datasets: [HAR]
    seeds: [42]
    base_overrides:
      training_configs.max_train_steps: 1000
      training_configs.pseudo_epoch_steps: 100
    cases:
      - name: alpha_only
        overrides:
          loss.alpha: 0.5
          loss.beta: 0.0
      - name: beta_only
        overrides:
          loss.alpha: 0.0
          loss.beta: 0.5
      - name: combined
        base_overrides:
          loss.alpha: 0.5
        overrides:
          loss.beta: 0.5
```

Cases are useful for nested experiments because each case can carry its own
`base_overrides`, `overrides`, `resume_from`, `checkpoint_probe`, and
`checkpoint_probe_epoch_offset`.

Current expansion semantics are deliberately simple:

- If a sweep block has `cases`, Slurminator emits one row per case, dataset, and
  seed.
- If the same block also has `sweep_keys`, Slurminator also emits the Cartesian
  `sweep_keys` rows as sibling variants.
- It does **not** multiply `cases x sweep_keys`. If you need named nested
  variants, write those combinations as explicit `cases`.

## Step-Budget Sweeps

For step-budget runs, provide both:

```yaml
training_configs.max_train_steps: 1000
training_configs.pseudo_epoch_steps: 100
```

If `training_configs.num_epochs` is not set, Slurminator infers the epoch
horizon as:

```text
ceil(max_train_steps / pseudo_epoch_steps)
```

If only one of `max_train_steps` and `pseudo_epoch_steps` is provided,
generation fails. This keeps dashboard progress and training horizons aligned.

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
   and the same generated experiment-state file.

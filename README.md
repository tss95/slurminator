# Slurminator

[![CI](https://github.com/tss95/slurminator/actions/workflows/ci.yml/badge.svg)](https://github.com/tss95/slurminator/actions/workflows/ci.yml)

**Author:** Tord Sture Stangeland
**Affiliations:** NORSAR (primary); University of Oslo (PhD affiliation)

Slurminator is a reusable SLURM/HPC experiment orchestrator extracted from a
research workflow.
It turns experiment YAML files into `sbatch` jobs, polls scheduler state, reads
live status files, and renders a terminal dashboard for active sweeps.

The package owns generic orchestration: config loading, SSH/SLURM submission,
status ingestion, timeout handling, dashboard rendering, and quota display.
Project-specific behavior enters through a small plugin interface.

## High-Level Description

Slurminator separates *experiment intent* from *cluster execution*.

1. You write a sweep YAML that describes what should be run: datasets, seeds,
   named cases, and override values.
2. Slurminator expands that sweep into an experiment-state YAML: one row per job,
   with stable ids, resource metadata, status, and generated `sweep_params`.
   This file then becomes the run ledger.
3. For each pending row, Slurminator asks the command builder or project plugin
   to turn that row into a shell command.
4. Slurminator submits that command through SLURM using the configured cluster,
   resource defaults, environment script, and concurrency limits.
5. The training script applies any forwarded config overrides, runs the
   experiment, and can optionally write live status files through the callback.
6. Slurminator polls scheduler state plus live status files, writes SLURM job
   ids, output paths, resource requests, timestamps, metrics, and links back into
   the experiment-state YAML, and renders the dashboard.

The generated file, usually named like
`experiment_lists/experiments_YYYYMMDD_HHMMSS.yaml`, is therefore both the
launch input and the persistent record of what happened. If the process stops,
launch Slurminator again with
`--yaml experiment_lists/experiments_YYYYMMDD_HHMMSS.yaml` to pick up where it
left off, assuming the recorded scheduler jobs, logs, and status files are still
visible from the cluster filesystem. While Slurminator is running, it reloads
the YAML each poll cycle; editing a completed or failed row back to
`status: pending` makes it eligible for relaunch when concurrency is available.

This is useful because large sweeps become small, reviewable YAML artifacts
instead of custom launch scripts. You can distribute or archive the sweep file,
launch it with one Slurminator command, resume or relaunch from the generated
state file, recover SLURM logs through recorded job/output metadata, and keep
cluster-specific details in user config rather than in the experiment
definition. Tools such as W&B, MLflow, TensorBoard, or project-specific trackers
remain complementary; Slurminator manages scheduling and run state while those
tools manage richer experiment analytics.

## Status

This repository is early but functional. The public API may change before a
tagged `0.1.0` release.

## License

Slurminator is released under the MIT License. See [LICENSE](LICENSE).

## Install

Editable install from a sibling checkout:

```bash
python -m pip install -e ../slurminator
```

Install the optional Textual dashboard v4 dependencies:

```bash
python -m pip install -e "../slurminator[v4]"
```

Dashboard v4 requires a modern terminal configuration. When running inside
tmux, use `tmux-256color` or `xterm-256color`, not the legacy
`screen-256color` default. See the
[Dashboard v4 requirements](#dashboard-v4-requirements) section for details.

Install from GitHub:

```bash
python -m pip install "slurminator @ git+https://github.com/tss95/slurminator.git@main"
```

The package exposes both a console script and a module entrypoint:

```bash
slurminator --help
python -m slurminator --help
```

If the console script is not on `PATH`, use `python -m slurminator`.

See [docs/cli.md](docs/cli.md) for the full built-in CLI reference and plugin
extension hooks.

## Quickstart

Create a user config:

```bash
mkdir -p ~/.slurminator_config
cp scripts/templates/hpc_config.example.yaml ~/.slurminator_config/hpc_config.yaml
$EDITOR ~/.slurminator_config/hpc_config.yaml
```

Minimal `hpc_config.yaml`:

```yaml
clusters:
  OLIVIA:
    partition: ACCEL
    account: "my_account"
    hostname: "olivia.example.org"
    username: "my_user"
    repo_path: "/cluster/work/my_user/my_project"
    save_path: "/cluster/work/my_user/my_project"
    data_path: "/cluster/work/my_user/my_project/data"
    environment_setup: "step_0.sh"
    gpu_type: "a100"
    gpu_count: 1
    gpu_gres_name: "a100"
    cpus_per_task: 4
    base_memory_gb: 80
    base_time_hours: 4
```

Create an experiment list:

```yaml
# experiment_lists/small.yaml
experiments:
  - experiment_id: smoke_001
    status: pending
    task_type: train
    dataset_name: smoke
    extra_command: "python train.py --config cfg/smoke.yaml --orchestrator"
```

Dry-run the launch:

```bash
python -m slurminator \
  --yaml experiment_lists/small.yaml \
  --olivia-limit 1 \
  --n-gpus 1 \
  --dry-run
```

Remove `--dry-run` only when you intentionally want to submit jobs.

See [docs/sweep_yaml.md](docs/sweep_yaml.md) for the full experiment-state and
custom-sweep YAML formats, including multiple datasets, named cases, Cartesian
parameter sweeps, and `--yaml` resume semantics.

To avoid repeated command-builder flags, put simple command defaults in the
optional `orchestrator_config.yaml`:

```yaml
command:
  entrypoint: "python train.py"
  config_arg: "--config"
  sweep_params_arg: "--overrides"
```

Then rows with `config:` and optional `sweep_params:` can run without
`--simple-command-*` CLI arguments. Use a plugin only when command construction,
validation, tracker integration, or log interpretation is project-specific.

## Critical Sweep Contract

Slurminator can generate sweep rows, but it does not know how to mutate your
training framework's config. Generated custom sweeps store overrides in each
experiment row as `sweep_params`, for example:

```yaml
sweep_params: "optimizer.lr=0.001;trainer.max_epochs=10"
```

Your command path must forward that string to the target training script, and
the target script must parse and apply it before training starts. If this is
missing, jobs still submit, but they run the base config.

For simple commands, wire the override argument explicitly:

```bash
python -m slurminator \
  --yaml experiment_lists/small.yaml \
  --simple-command-entrypoint "python train.py" \
  --simple-command-config-arg "--config" \
  --simple-command-sweep-params-arg "--overrides" \
  --olivia-limit 1
```

Then `train.py` must define and consume `--overrides`. Projects using Hydra,
OmegaConf, Pydantic settings, or custom config loaders should map Slurminator's
semicolon-separated `key=value` string into that system and fail clearly on
unknown keys. See
[docs/sweep_yaml.md](docs/sweep_yaml.md#minimal-target-script-support) for a
minimal target-script parser.

## User Config

Slurminator loads two user-facing YAML files:

- `hpc_config.yaml` is required and contains cluster connection details,
  resource defaults, project paths, dataset pinning, and per-cluster environment.
- `orchestrator_config.yaml` is optional and contains dashboard/orchestrator
  behavior knobs. Missing values use package defaults.

Config lookup order:

1. Explicit CLI flags: `--hpc-config-file`, `--orchestrator-config-file`.
2. Environment variables: `SLURMINATOR_HPC_CONFIG_FILE`,
   `SLURMINATOR_ORCHESTRATOR_CONFIG_FILE`, and `SLURMINATOR_REPO_ROOT`.
3. `~/.slurminator_config/hpc_config.yaml` and
   `~/.slurminator_config/orchestrator_config.yaml`.
4. Legacy fallback: `~/.slurminator/`.
5. `<repo_root>/user_configs/` when `--repo-root`,
   `SLURMINATOR_REPO_ROOT`, or a plugin default repo root is provided,
   otherwise `./user_configs/`.

`repo_path` inside `hpc_config.yaml` is the cluster-side project path used when
submitting jobs. It is not the same thing as the local repo root used for config
discovery.

Supported cluster identifiers currently match the built-in `HPCType` enum:
`FOX`, `LUMI`, `SAGA`, and `OLIVIA`. Adding a new identifier currently requires
extending the package enum and, optionally, adding a quota provider.

## Commands

Use an existing experiment list:

```bash
python -m slurminator \
  --yaml experiment_lists/small.yaml \
  --olivia-limit 1 \
  --n-gpus 1
```

`--yaml` is also the resume path. Launch Slurminator with the same generated
`experiment_lists/experiments_*.yaml` file to reconnect to queued/running jobs
and continue submitting pending rows.

Generate an experiment list from a custom sweep file:

```bash
python -m slurminator \
  --sweepfile cfg/sweeps/my_sweep.yaml \
  --olivia-limit 4 \
  --n-gpus 1 \
  --job-time-hours 4 \
  --job-ram-gb 80
```

`--sweepfile` creates an experiment-state YAML with row-level `sweep_params`.
Those generated overrides only affect training if the command builder forwards
them and your training entrypoint parses them. See
[Critical Sweep Contract](#critical-sweep-contract).

Override a partition for one run:

```bash
python -m slurminator \
  --yaml experiment_lists/small.yaml \
  --partition-override OLIVIA=accel_long \
  --olivia-limit 1
```

## Command Building

For generic usage, each experiment may provide `extra_command` or `command`.
Alternatively, use `SimpleCommandPlugin` from the CLI:

```bash
python -m slurminator \
  --yaml experiment_lists/small.yaml \
  --simple-command-entrypoint "python train.py" \
  --simple-command-config-arg "--config" \
  --simple-command-sweep-params-arg "--overrides" \
  --olivia-limit 1
```

With `SimpleCommandPlugin`, each experiment row should include `config` or
`config_path`. If rows contain generated `sweep_params`, the
`--simple-command-sweep-params-arg` value must match an argument parsed by your
training script.

The same settings can live under `command:` in `orchestrator_config.yaml`, so
users do not need to repeat them on every launch.

## Project Plugins

Projects can customize parser flags, command construction, validation, tracker
integration, log interpretation, and runtime presentation through a plugin:

```bash
export SLURMINATOR_PLUGIN="my_project.orchestrator:MyOrchestratorPlugin"
python -m slurminator --yaml experiment_lists/small.yaml --olivia-limit 1
```

The value may use `module:ClassName` or `module.ClassName` syntax. If unset,
Slurminator uses the generic default plugin.

Runtime plugin methods:

- `validate_experiment(exp, overrides)`
- `build_commands_line(exp, context)`
- `prepare_remote_runtime(hpc_type, connection_manager)`
- `interpret_log_tail(exp, log_tail, current_status, stage)`
- `annotate_log_tail(exp, log_tail)`

Optional runtime integration hooks:

- `status_projection_options()`
- `parse_sweep_overrides(raw)`
- `is_local_hpc(hpc_type)`
- `dashboard_class()`
- `overview_printer()`

Optional CLI integration hooks:

- `pre_parse_argv(argv)`
- `extend_parser(parser)`
- `prepare_args(args)`
- `configure_from_args(args)`
- `generate_experiment_yaml(args)`
- `run_sweep_mode(args, connection_manager)`
- `launch_guard()`

Most adopters do not need all hooks. Start with explicit `extra_command` rows or
`SimpleCommandPlugin`, then add a project plugin only when the command line or
tracker behavior cannot be expressed as data.

Projects that already use local environment-variable prefixes can set
`SLURMINATOR_ENV_PREFIXES`, for example:

```bash
export SLURMINATOR_ENV_PREFIXES="MYPROJECT,SLURMINATOR"
```

Slurminator always keeps `SLURMINATOR` as a fallback prefix.

## Environment Script

Cluster jobs source the `environment_setup` script from `hpc_config.yaml`
before running the experiment command. Use this script to load modules,
activate containers, set cache paths, and configure tracker credentials.

The examples use `step_0.sh`, but adopters can use any project-local script
name.

## Logging

The Slurminator CLI configures a clickable-path console logger for the
`slurminator` logger. Log lines include a relative `path:line` location so
warnings and errors can be traced back to the source quickly.

Set `SLURMINATOR_LOG_LEVEL=DEBUG` (or `INFO`, `WARNING`, `ERROR`) to control
Slurminator verbosity. If that variable is unset, `LOG_LEVEL` is used as a
fallback before defaulting to `INFO`.

Adopters can reuse the logger setup directly:

```python
from slurminator.logging_config import configure_logging

configure_logging(project_root=".")
```

## Status Files

Jobs can write live status files using Slurminator's callback helpers under:

```text
$SAVE_PATH/.orchestrator_status[/sweep_<sweep_id>]/status_<job_id>.json
```

The target schema lives in `slurminator.schemas.status_schema`. Status files are
optional but improve dashboard progress, metrics, links, and live speed display.
Without them, scheduler state still drives terminal status.

### Wiring The Metrics Callback

To collect live metrics in the dashboard, include
`slurminator.callbacks.status_callback.OrchestratorStatusCallback` in your
training loop, or subclass it for project-specific display metrics and tracker
links.

See [docs/status_callback.md](docs/status_callback.md) for a complete callback
integration example, including full display metric sets, primary/secondary
highlight metrics, and display metadata.

Minimal framework-neutral pattern:

```python
from slurminator.callbacks.status_callback import OrchestratorStatusCallback

status_cb = OrchestratorStatusCallback(cfg=cfg)
status_cb.on_train_start(trainer)

for epoch in range(num_epochs):
    for batch_idx, batch in enumerate(loader):
        logs = train_one_batch(batch)
        status_cb.on_train_batch_end(trainer, batch, batch_idx, logs)
    status_cb.on_epoch_end(trainer, epoch, train_logs, val_logs)

status_cb.on_train_end(trainer)
```

The callback accepts explicit constructor values for `save_path`, `job_id`,
`sweep_id`, `experiment_id`, `primary_metric`, `secondary_metric`, and
`metric_info`. If omitted, it resolves identity from environment variables:

- `SAVE_PATH` for the output root.
- `SLURM_JOB_ID`, `PBS_JOBID`, or `JOB_ID` for the job id.
- `ORCHESTRATOR_SWEEP_ID` or `SWEEP_ID` for optional sweep grouping.
- `ORCHESTRATOR_EXPERIMENT_ID` or `EXPERIMENT_ID` for the displayed experiment.

Only flat, finite numeric metrics are written. Booleans, nested values, NaN, and
infinities are ignored. Display metadata is strict: a metric's display info is
materialized only after that metric key exists in the numeric `metrics` block.

Use `update_metrics(metrics, trainer=trainer)` for late metrics produced outside
the normal epoch hook. Subclasses usually override `_resolve_links()`,
`_resolve_display_candidates()`, and, when needed, `_build_progress_snapshot()`.

## Dashboard Quota Providers

The terminal dashboard can render cluster budget/quota information through
optional quota providers. Slurminator ships with an OLIVIA/Sigma2 provider and
custom providers can be registered for other clusters. See
[`docs/quota_providers.md`](docs/quota_providers.md).

Quota warnings are approximate. The dashboard estimates remaining sweep cost
from literal requested walltime and GPU counts; it does not model site-specific
charging rules such as partition multipliers, full-node charging, or minimum GPU
counts per node. Treat the quota footer as an early warning and sanity check,
not as an authoritative accounting statement.

## Dashboard v4

Dashboard v4 is an optional Textual interface. Install it with
`pip install "slurminator[v4]"`, then launch with `--dashboard-ui v4`. The
default `--dashboard-ui v3` remains Rich-based and does not require Textual.
Phase 4 implementation decisions are tracked in
[`docs/slurminator_ui_v4_phase4_decisions.md`](docs/slurminator_ui_v4_phase4_decisions.md).

### Dashboard v4 Requirements

If you run v4 inside tmux, configure tmux to use `tmux-256color`:

```tmux
set-option -g default-terminal "tmux-256color"
set-option -ga terminal-overrides ",xterm-256color:RGB"
```

Restart tmux with `tmux kill-server && tmux`, or export `TERM=tmux-256color`
inside the current tmux session before launching the dashboard. Verify with
`echo $TERM`; it should not be `screen-256color`.

On Windows, use Windows Terminal or another modern terminal emulator such as
WezTerm rather than the legacy `powershell.exe` console host, which may not
propagate resize events through SSH reliably. See
[`docs/slurminator_ui_v4_phase4_decisions.md#known-terminal-compatibility`](docs/slurminator_ui_v4_phase4_decisions.md#known-terminal-compatibility)
for the full compatibility notes and the `tic` workaround for older systems
without `tmux-256color` terminfo.

## Development

Run package tests:

```bash
pytest -q
```

Run focused CLI tests:

```bash
pytest -q tests/test_cli.py tests/test_orchestrator_plugin.py
```

Lint and format touched files:

```bash
ruff check src tests
black --line-length 120 --skip-string-normalization --skip-magic-trailing-comma src tests
```

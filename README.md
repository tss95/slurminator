# Slurminator

Slurminator is a reusable SLURM/HPC experiment orchestrator extracted from PMT.
It turns experiment YAML files into `sbatch` jobs, polls scheduler state, reads
live status files, and renders a terminal dashboard for active sweeps.

The package owns generic orchestration: config loading, SSH/SLURM submission,
status ingestion, timeout handling, dashboard rendering, and quota display.
Project-specific behavior enters through a small plugin interface.

## Status

This repository is early but functional. PMT is currently the reference adopter,
and the package is still stabilizing around that extraction. The public API may
change before a tagged `0.1.0` release.

## Install

Editable install from a sibling checkout:

```bash
python -m pip install -e ../slurminator
```

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

## Quickstart

Create a user config:

```bash
mkdir -p ~/.slurminator
$EDITOR ~/.slurminator/hpc_config.yaml
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

## User Config

Slurminator loads two user-facing YAML files:

- `hpc_config.yaml` is required and contains cluster connection details,
  resource defaults, project paths, dataset pinning, and per-cluster environment.
- `orchestrator_config.yaml` is optional and contains dashboard/orchestrator
  behavior knobs. Missing values use package defaults.

Config lookup order:

1. Explicit CLI flags: `--hpc-config-file`, `--orchestrator-config-file`.
2. `~/.slurminator/hpc_config.yaml` and
   `~/.slurminator/orchestrator_config.yaml`.
3. `<repo_root>/user_configs/` when `--repo-root` is provided, otherwise
   `./user_configs/`.

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

Generate an experiment list from a custom sweep file:

```bash
python -m slurminator \
  --sweepfile cfg/sweeps/my_sweep.yaml \
  --olivia-limit 4 \
  --n-gpus 1 \
  --job-time-hours 4 \
  --job-ram-gb 80
```

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
  --olivia-limit 1
```

With `SimpleCommandPlugin`, each experiment row should include `config` or
`config_path`.

## Project Plugins

Projects can customize parser flags, command construction, validation, tracker
integration, log interpretation, and class selection through a plugin:

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

Optional CLI integration hooks:

- `pre_parse_argv(argv)`
- `extend_parser(parser)`
- `prepare_args(args)`
- `configure_from_args(args)`
- `generate_experiment_yaml(args)`
- `run_sweep_mode(args, connection_manager)`
- `launch_guard()`
- `orchestrator_cls`
- `connection_manager_cls`

Most adopters do not need all hooks. Start with explicit `extra_command` rows or
`SimpleCommandPlugin`, then add a project plugin only when the command line or
tracker behavior cannot be expressed as data.

## Environment Script

Cluster jobs source the `environment_setup` script from `hpc_config.yaml`
before running the experiment command. Use this script to load modules,
activate containers, set cache paths, and configure tracker credentials.

The default PMT convention is `step_0.sh`, but adopters can use any project-local
script name.

## Status Files

Jobs can write live status files using Slurminator's callback helpers under:

```text
$SAVE_PATH/.orchestrator_status_v2/<sweep_id>/status_<job_id>.json
```

The target schema lives in `slurminator.schemas.status_schema`. Status files are
optional but improve dashboard progress, metrics, links, and live speed display.
Without them, scheduler state still drives terminal status.

## Dashboard Quota Providers

The terminal dashboard can render cluster budget/quota information through
optional quota providers. Slurminator ships with an OLIVIA/Sigma2 provider and
custom providers can be registered for other clusters. See
[`docs/quota_providers.md`](docs/quota_providers.md).

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

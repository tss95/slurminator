# CLI Reference And Extension Hooks

Slurminator exposes both a console script and a module entrypoint:

```bash
slurminator --help
python -m slurminator --help
```

Use `python -m slurminator` when the console script is not on `PATH`.

## Input Selection

Choose exactly one of the normal input modes:

- `--yaml FILE`: run or resume an existing experiment-state YAML.
- `--sweepfile FILE`: generate an experiment-state YAML from a custom-sweep
  YAML, then run it.

Compatibility aliases:

- `--custom-sweep.file FILE`: alias for `--sweepfile`.
- `--run-custom-sweeps`: compatibility flag implied by `--sweepfile`.

Generation helpers:

- `--config-profile NAME`: optional metadata/profile value copied into generated
  rows.
- `--num-epochs N`: optional generated-row epoch count for projects that use
  epoch-based training.

`--yaml` is the resume path. If an orchestrator session exits while jobs are
still active, rerun with the same generated experiment-state YAML rather than
regenerating from `--sweepfile`.

## Resources And Retry Policy

- `--n-gpus N`: GPUs per submitted job. Default: `1`.
- `--job-time-hours HOURS`: override Slurm walltime for submitted jobs.
- `--job-ram-gb GB`: override Slurm memory request.
- `--retry-timeout-with-estimated-time`: relaunch timed-out jobs with walltime
  estimated from observed progress.
- `--timeout-retry-buffer FLOAT`: multiplicative safety factor for timeout
  relaunches. Default: `1.3`.
- `--timeout-retry-max-attempts N`: maximum timeout-based relaunch attempts per
  experiment. Default: `1`.

## Cluster Limits And Partitions

The built-in cluster identifiers currently map to:

- `--fox-limit N`
- `--lumi-limit N`
- `--saga-limit N`
- `--olivia-limit N`

Set only the limit for clusters you want this orchestrator session to use. A
limit of `0` disables that cluster for the session.

Partition overrides:

- `--partition-override HPC=PARTITION`: override one cluster partition for this
  run. May be repeated.
- `--lumi-partition PARTITION`: compatibility alias for
  `--partition-override LUMI=PARTITION`.

## Runtime Behavior

- `--poll-interval SECONDS`: scheduler polling interval. Default: `2`.
- `--dashboard-ui v2|v3|v4`: dashboard implementation. If omitted,
  `orchestrator_config.yaml` may set `dashboard.ui_version`; otherwise the
  package default is `v4`. Use `--dashboard-ui v3` for the legacy Rich
  dashboard.
- `--dry-run`: generate and validate inputs, then exit without launching jobs.
- `--debug`: enable debug mode.
- `--no-prog`: reserved compatibility flag; currently ignored by the generic
  orchestrator.

## Config Discovery

- `--hpc-config-file FILE`: explicit path to `hpc_config.yaml`.
- `--orchestrator-config-file FILE`: explicit path to optional
  `orchestrator_config.yaml`.
- `--repo-root DIR`: repository root used when searching `user_configs/`.

Config lookup order is:

1. explicit CLI config-file flags,
2. `SLURMINATOR_HPC_CONFIG_FILE`, `SLURMINATOR_ORCHESTRATOR_CONFIG_FILE`,
   and `SLURMINATOR_REPO_ROOT`,
3. `~/.slurminator_config/`,
4. legacy fallback `~/.slurminator/`,
5. `<repo_root>/user_configs/` or `./user_configs/`.

When a project plugin implements `default_repo_root()`, Slurminator uses that
as the repo-root fallback before checking `./user_configs/`. `repo_path` in
`hpc_config.yaml` remains the cluster-side execution path and is not used as
the local repo root.

## Generic Command Building

For simple projects, avoid writing a plugin by using `SimpleCommandPlugin` from
the CLI:

```bash
python -m slurminator \
  --yaml experiment_lists/small.yaml \
  --simple-command-entrypoint "python train.py" \
  --simple-command-config-arg "--config" \
  --simple-command-sweep-params-arg "--overrides" \
  --olivia-limit 1
```

With this mode, each experiment row should include `config` or `config_path`.
Rows generated from a custom-sweep file may also include `sweep_params`:

```yaml
experiments:
  - experiment_id: smoke
    status: pending
    task_type: train
    dataset_name: smoke
    config: configs/smoke.yaml
    sweep_params: "optimizer.lr=0.001;trainer.max_epochs=10"
```

The resulting command is equivalent to:

```bash
python train.py --config configs/smoke.yaml --overrides 'optimizer.lr=0.001;trainer.max_epochs=10' --orchestrator
```

Slurminator only forwards the generated override string. Your training
entrypoint must parse the argument named by
`--simple-command-sweep-params-arg` and apply those overrides to its config.
See the sweep YAML guide's target-script example for a minimal parser and
config-application pattern.

Flags:

- `--simple-command-entrypoint COMMAND`: entrypoint such as
  `python train.py`.
- `--simple-command-config-arg ARG`: argument used before the row's `config` or
  `config_path`. Default: `--config`. Use an empty string when the entrypoint
  does not take a config argument.
- `--simple-command-sweep-params-arg ARG`: optional argument used before the
  row's `sweep_params`, such as `--overrides`, `--set`, or `--sweep-params`.
  Leave unset when the row has no generated overrides or when the command reads
  a fully resolved config file instead.

Explicit row-level `extra_command` or `command` always wins over
`SimpleCommandPlugin`.

Projects with richer command rules should implement `build_commands_line()` in
a plugin. The same contract applies there: if a generated row has
`sweep_params`, the plugin must either forward it to the training CLI or replace
it with an equivalent resolved config.

The simple command settings can also be stored in
`orchestrator_config.yaml`, which avoids repeating `--simple-command-*` flags:

```yaml
command:
  entrypoint: "python train.py"
  config_field: "config"
  config_arg: "--config"
  sweep_params_arg: "--overrides"
  extra_args: []
  orchestrator_flag: "--orchestrator"
```

## Plugin Discovery

Projects can extend the CLI by setting:

```bash
export SLURMINATOR_PLUGIN="my_project.orchestrator:MyPlugin"
python -m slurminator --yaml experiment_lists/small.yaml --olivia-limit 1
```

The value may use `module:ClassName` or `module.ClassName` syntax. Slurminator
imports the object, instantiates it when it is a class, and requires at least a
`build_commands_line(exp, context)` method.

If `SLURMINATOR_PLUGIN` is unset, Slurminator uses the generic default plugin.

## CLI Extension Hooks

A plugin may implement any of these optional CLI hooks:

- `pre_parse_argv(argv) -> list[str] | None`: rewrite raw arguments before
  argparse runs. Use this for compatibility aliases or external sweep ids that
  must be converted before parsing.
- `extend_parser(parser) -> parser`: add project-specific flags.
- `prepare_args(args) -> None`: normalize or validate parsed arguments. If not
  provided, Slurminator applies its generic `--sweepfile`/`--yaml`
  normalization. If you implement this hook and still support those generic
  modes, preserve the same validation.
- `default_repo_root() -> str | Path | None`: provide a local repo root used for
  `user_configs/` discovery when no CLI/env repo root is set.
- `generate_experiment_yaml(args) -> str`: generate an experiment-state YAML
  from project-specific flags.
- `run_sweep_mode(args, connection_manager) -> None`: handle an external sweep
  mode and exit without starting the normal orchestrator. In the current CLI
  this is triggered by a plugin-added `args.wandb_sweep` value.
- `launch_guard() -> str | None`: return an error message to block launch, or
  `None` to continue.
- `configure_from_args(args) -> OrchestratorPlugin | None`: configure and return
  the runtime plugin after parsing.

Example:

```python
from shlex import quote

from slurminator.plugins import CommandBuildContext, DefaultOrchestratorPlugin


class MyPlugin(DefaultOrchestratorPlugin):
    def __init__(self) -> None:
        self.project_name = "default"

    def extend_parser(self, parser):
        parser.add_argument("--project-name", default="default")
        parser.add_argument("--train-entrypoint", default="python train.py")
        return parser

    def configure_from_args(self, args):
        self.project_name = args.project_name
        self.train_entrypoint = args.train_entrypoint
        return self

    def build_commands_line(self, exp, context: CommandBuildContext) -> str:
        config = exp.get("config") or exp.get("config_path")
        if not config:
            raise ValueError(f"{exp.get('experiment_id')} is missing config/config_path.")
        sweep_params = exp.get("sweep_params")
        sweep_args = f" --overrides {quote(str(sweep_params))}" if sweep_params else ""
        return (
            f"{self.train_entrypoint} --config {quote(str(config))} "
            f"--project {quote(str(self.project_name))}{sweep_args} --orchestrator"
        )
```

## Runtime Plugin Methods

The core runtime plugin surface is intentionally small:

- `validate_experiment(exp, overrides) -> bool`
- `build_commands_line(exp, context) -> str`
- `prepare_remote_runtime(hpc_type=..., connection_manager=...) -> None`
- `interpret_log_tail(exp=..., log_tail=..., current_status=..., stage=...)`
- `annotate_log_tail(exp=..., log_tail=...) -> None`

Projects that need runtime integration without subclassing the orchestrator can
also implement optional hooks:

- `status_projection_options() -> dict`: customize how status files project into
  experiment rows.
- `parse_sweep_overrides(raw) -> dict`: parse project-specific sweep override
  strings for validation.
- `is_local_hpc(hpc_type) -> bool`: override current-cluster detection.
- `dashboard_class() -> type | None`: provide a project dashboard subclass.
- `overview_printer() -> callable | None`: provide text/debug rendering.

Most adopters start with explicit `extra_command` rows or
`SimpleCommandPlugin`. Add a custom plugin only when the command line, validation
rules, tracker integration, or log interpretation cannot be expressed as data.

Connection-manager environment variables normally use the `SLURMINATOR_` prefix.
If a project already has a private prefix, set `SLURMINATOR_ENV_PREFIXES` before
launching:

```bash
export SLURMINATOR_ENV_PREFIXES="MYPROJECT,SLURMINATOR"
```

The `SLURMINATOR` prefix is kept as a fallback even if omitted.

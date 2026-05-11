"""Generic command-line entry point for Slurminator orchestration."""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from slurminator.base_orchestrator import BaseOrchestrator
from slurminator.config import HPCType, HPC_CONFIGS, OrchestratorSettings, REPO_ROOT_ENV, load_user_config
from slurminator.connection_manager import HPCConnectionConfig, HPCConnectionManager
from slurminator.experiments import CustomSweepConfig, MasterExperimentConfig
from slurminator.experiments.yaml_utils import load_yaml
from slurminator.hpc_orchestrator import HPCOrchestrator
from slurminator.launch_guard import get_orchestrator_gpu_hpc_launch_block_message
from slurminator.logging_config import configure_logging
from slurminator.plugins import DefaultOrchestratorPlugin, OrchestratorPlugin, SimpleCommandPlugin

logger = logging.getLogger("slurminator")

ParserExtender = Callable[[argparse.ArgumentParser], argparse.ArgumentParser]
ArgsPreparer = Callable[[argparse.Namespace], None]
ExperimentGenerator = Callable[[argparse.Namespace], str]
PluginFactory = Callable[[argparse.Namespace], OrchestratorPlugin]
SweepModeRunner = Callable[[argparse.Namespace, Any | None], None]
LaunchGuard = Callable[[], str | None]
PLUGIN_ENV_VAR = "SLURMINATOR_PLUGIN"


def build_base_parser() -> argparse.ArgumentParser:
    """Build Slurminator's generic orchestrator parser."""
    parser = argparse.ArgumentParser(description="Run Slurm/HPC experiment orchestration.", conflict_handler="resolve")
    parser.add_argument("--yaml", type=str, help="Path to an existing experiment YAML file.")
    parser.add_argument(
        "--run-custom-sweeps",
        action="store_true",
        help="Generate experiments from custom sweep specs. Compatibility flag; implied by --sweepfile.",
    )
    parser.add_argument(
        "--sweepfile",
        "--custom-sweep.file",
        dest="custom_sweep_file",
        type=str,
        metavar="FILE",
        help="Path to a custom sweep YAML file. Implies custom-sweep experiment generation.",
    )
    parser.add_argument("--config-profile", type=str, default=None, help="Optional generated-row config profile.")
    parser.add_argument("--num-epochs", type=int, default=None, help="Optional generated-row epoch override.")

    parser.add_argument("--n-gpus", type=int, default=1, help="Number of GPUs per submitted job.")
    parser.add_argument("--job-time-hours", type=int, default=None, help="Override Slurm walltime in hours.")
    parser.add_argument("--job-ram-gb", type=int, default=None, help="Override Slurm memory request in GB.")
    parser.add_argument(
        "--retry-timeout-with-estimated-time",
        dest="retry_timeout_with_estimated_time",
        action="store_true",
        help="Retry timed-out jobs once with walltime estimated from observed progress.",
    )
    parser.add_argument(
        "--timeout-retry-buffer",
        dest="timeout_retry_buffer",
        type=float,
        default=1.3,
        help="Multiplicative timeout retry safety factor.",
    )
    parser.add_argument(
        "--timeout-retry-max-attempts",
        dest="timeout_retry_max_attempts",
        type=int,
        default=1,
        help="Maximum timeout-based relaunch attempts per experiment.",
    )

    for hpc_type in HPCType:
        parser.add_argument(
            f"--{hpc_type.name.lower()}-limit", type=int, default=0, help=f"Concurrency limit for {hpc_type.name}."
        )
    parser.add_argument(
        "--partition-override",
        action="append",
        default=[],
        metavar="HPC=PARTITION",
        help="Override one cluster partition for this orchestrator run. May be repeated.",
    )
    parser.add_argument(
        "--lumi-partition", type=str, default=None, help="Compatibility alias for --partition-override LUMI=PARTITION."
    )
    parser.add_argument("--poll-interval", type=int, default=2, help="Polling interval in seconds.")
    parser.add_argument(
        "--dashboard-ui", dest="dashboard_ui", choices=["v2", "v3"], default="v3", help="Select dashboard UI version."
    )
    parser.add_argument("--dry-run", action="store_true", help="Generate/validate and exit without launching jobs.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    parser.add_argument("--no-prog", action="store_true", help="Reserved compatibility flag; ignored by orchestrator.")

    parser.add_argument("--hpc-config-file", type=str, default=None, help="Path to hpc_config.yaml.")
    parser.add_argument(
        "--orchestrator-config-file", type=str, default=None, help="Optional path to orchestrator_config.yaml."
    )
    parser.add_argument("--repo-root", type=str, default=None, help="Repository root used for user config discovery.")

    parser.add_argument(
        "--simple-command-entrypoint",
        type=str,
        default=None,
        help="Use SimpleCommandPlugin with this entrypoint, e.g. 'python train.py'.",
    )
    parser.add_argument(
        "--simple-command-config-arg",
        type=str,
        default=None,
        help="Config argument used by SimpleCommandPlugin. Use an empty string to disable.",
    )
    parser.add_argument(
        "--simple-command-sweep-params-arg",
        type=str,
        default=None,
        help=(
            "Optional argument used by SimpleCommandPlugin before a generated row's sweep_params value, "
            "for example '--overrides' or '--sweep-params'."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Console-script entry point for the generic Slurminator CLI."""
    configure_logging(Path.cwd())
    run_orchestrator_cli(argv=argv)


def run_orchestrator_cli(
    argv: Sequence[str] | None = None,
    *,
    parser_extender: ParserExtender | None = None,
    args_preparer: ArgsPreparer | None = None,
    experiment_generator: ExperimentGenerator | None = None,
    plugin_factory: PluginFactory | None = None,
    sweep_mode_runner: SweepModeRunner | None = None,
    launch_guard: LaunchGuard | None = get_orchestrator_gpu_hpc_launch_block_message,
    load_configs: bool = True,
    cluster_configs: Mapping[HPCType, Any] | None = None,
    orchestrator_cls: type[HPCOrchestrator] = HPCOrchestrator,
    connection_manager_cls: type[HPCConnectionManager] = HPCConnectionManager,
) -> None:
    """Run orchestration from CLI arguments with optional project hooks."""
    discovered_plugin = None if plugin_factory is not None else discover_plugin()
    raw_argv = list(argv) if argv is not None else None
    if discovered_plugin is not None:
        raw_argv = _call_optional_hook(
            discovered_plugin, "pre_parse_argv", raw_argv if raw_argv is not None else sys.argv[1:]
        )

    parser = build_base_parser()
    if parser_extender is None and discovered_plugin is not None:
        parser_extender = _get_optional_hook(discovered_plugin, "extend_parser")
    if parser_extender is not None:
        parser = parser_extender(parser)
    args = parser.parse_args(raw_argv)

    if args_preparer is None and discovered_plugin is not None:
        args_preparer = _get_optional_hook(discovered_plugin, "prepare_args")
    if args_preparer is not None:
        args_preparer(args)
    else:
        _normalise_generic_generation_args(args)

    loaded_config = None
    if load_configs:
        repo_root = args.repo_root or os.environ.get(REPO_ROOT_ENV) or _repo_root_from_plugin(discovered_plugin)
        loaded_config = load_user_config(
            hpc_config_file=args.hpc_config_file,
            orchestrator_config_file=args.orchestrator_config_file,
            repo_root=repo_root,
        )

    if launch_guard is get_orchestrator_gpu_hpc_launch_block_message and discovered_plugin is not None:
        launch_guard = _get_optional_hook(discovered_plugin, "launch_guard") or launch_guard
    if launch_guard is not None:
        block_msg = launch_guard()
        if block_msg:
            logger.error(block_msg)
            raise SystemExit(2)

    if connection_manager_cls is HPCConnectionManager and discovered_plugin is not None:
        connection_manager_cls = _get_optional_class(
            discovered_plugin, "connection_manager_cls", connection_manager_cls
        )
    if orchestrator_cls is HPCOrchestrator and discovered_plugin is not None:
        orchestrator_cls = _get_optional_class(discovered_plugin, "orchestrator_cls", orchestrator_cls)

    concurrency_limits = build_concurrency_limits(args)
    early_connection_manager = _bootstrap_connection_manager(
        concurrency_limits, cluster_configs=cluster_configs, connection_manager_cls=connection_manager_cls
    )

    if getattr(args, "wandb_sweep", None) is not None:
        if sweep_mode_runner is None and discovered_plugin is not None:
            sweep_mode_runner = _get_optional_hook(discovered_plugin, "run_sweep_mode")
        if sweep_mode_runner is None:
            raise SystemExit("--wandb-sweep was provided, but no sweep-mode runner is configured.")
        sweep_mode_runner(args, early_connection_manager)
        return

    if experiment_generator is None and discovered_plugin is not None:
        experiment_generator = _get_optional_hook(discovered_plugin, "generate_experiment_yaml")
    experiment_file = _resolve_experiment_file(args, experiment_generator=experiment_generator)
    if not Path(experiment_file).exists():
        raise SystemExit(f"Experiment file not found: {experiment_file}")

    if args.dry_run:
        logger.info("Dry run enabled - not launching HPC orchestrator.")
        return

    if plugin_factory is not None:
        plugin = plugin_factory(args)
    elif discovered_plugin is not None:
        plugin = _configured_plugin_from_args(discovered_plugin, args)
    else:
        plugin = _default_plugin_from_args(
            args, orchestrator_settings=loaded_config.orchestrator if loaded_config is not None else None
        )
    partition_overrides = parse_partition_overrides(args)
    orchestrator = orchestrator_cls(
        experiment_file=str(experiment_file),
        concurrency_limits=concurrency_limits,
        poll_interval=args.poll_interval,
        max_gpus_per_job=args.n_gpus,
        time_hours_override=args.job_time_hours,
        memory_gb_override=args.job_ram_gb,
        retry_timeout_with_estimated_time=args.retry_timeout_with_estimated_time,
        timeout_retry_buffer=args.timeout_retry_buffer,
        timeout_retry_max_attempts=args.timeout_retry_max_attempts,
        debug=args.debug,
        dashboard_ui=args.dashboard_ui,
        connection_manager=early_connection_manager,
        plugin=plugin,
        partition_overrides=partition_overrides,
    )
    orchestrator.run()


def discover_plugin(env: Mapping[str, str] | None = None) -> Any | None:
    """Load the plugin declared by ``SLURMINATOR_PLUGIN``, if any."""
    env_map = os.environ if env is None else env
    dotted_path = env_map.get(PLUGIN_ENV_VAR, "").strip()
    if not dotted_path:
        return None

    try:
        plugin_obj = load_object(dotted_path)
    except Exception as exc:
        raise RuntimeError(
            f"Could not import Slurminator plugin from {PLUGIN_ENV_VAR}={dotted_path!r}. "
            "Use 'module:ClassName' or 'module.ClassName', ensure the module is importable, "
            "or unset SLURMINATOR_PLUGIN to use the default generic plugin."
        ) from exc

    plugin = plugin_obj() if isinstance(plugin_obj, type) else plugin_obj
    if not hasattr(plugin, "build_commands_line"):
        raise TypeError(
            f"Object loaded from {PLUGIN_ENV_VAR}={dotted_path!r} is not an orchestrator plugin. "
            "It must provide at least build_commands_line(exp, context), or unset SLURMINATOR_PLUGIN."
        )
    return plugin


def build_concurrency_limits(args: argparse.Namespace) -> dict[HPCType, int]:
    """Return per-cluster concurrency limits from parsed CLI arguments."""
    return {hpc_type: int(getattr(args, f"{hpc_type.name.lower()}_limit", 0) or 0) for hpc_type in HPCType}


def _get_optional_hook(plugin: Any, hook_name: str) -> Callable[..., Any] | None:
    """Return an optional callable plugin hook."""
    hook = getattr(plugin, hook_name, None)
    return hook if callable(hook) else None


def _call_optional_hook(plugin: Any, hook_name: str, default: Any) -> Any:
    """Call an optional plugin hook and return ``default`` when absent."""
    hook = _get_optional_hook(plugin, hook_name)
    if hook is None:
        return default
    result = hook(default)
    return default if result is None else result


def _get_optional_class(plugin: Any, hook_name: str, default: type[Any]) -> type[Any]:
    """Return a class override from a plugin attribute/property/method."""
    value = getattr(plugin, hook_name, None)
    if value is None:
        return default
    if isinstance(value, type):
        return value
    if callable(value):
        resolved = value()
        return resolved if isinstance(resolved, type) else default
    return default


def _repo_root_from_plugin(plugin: Any | None) -> str | None:
    """Return a local repo-root default supplied by a project plugin."""
    if plugin is None:
        return None

    hook = _get_optional_hook(plugin, "default_repo_root")
    if hook is not None:
        repo_root = hook()
    else:
        repo_root = getattr(plugin, "repo_root", None)
        if callable(repo_root):
            repo_root = repo_root()

    if repo_root is None:
        return None
    return str(Path(repo_root).expanduser())


def _configured_plugin_from_args(plugin: Any, args: argparse.Namespace) -> OrchestratorPlugin:
    """Return a discovered plugin configured with parsed CLI args when supported."""
    hook = _get_optional_hook(plugin, "configure_from_args")
    if hook is None:
        return plugin
    configured = hook(args)
    return plugin if configured is None else configured


def parse_partition_overrides(args: argparse.Namespace) -> dict[HPCType, str]:
    """Return per-cluster partition overrides from parsed CLI arguments."""
    raw_overrides = list(getattr(args, "partition_override", []) or [])
    lumi_partition = getattr(args, "lumi_partition", None)
    if lumi_partition:
        raw_overrides.append(f"LUMI={lumi_partition}")

    parsed: dict[HPCType, str] = {}
    for raw in raw_overrides:
        if "=" not in raw:
            raise SystemExit(f"Invalid --partition-override {raw!r}; expected HPC=PARTITION.")
        hpc_name, partition = raw.split("=", 1)
        hpc_name = hpc_name.strip().upper()
        partition = partition.strip()
        if not partition:
            raise SystemExit(f"Invalid --partition-override {raw!r}; partition is empty.")
        try:
            parsed[HPCType[hpc_name]] = partition
        except KeyError as exc:
            known = ", ".join(hpc.name for hpc in HPCType)
            raise SystemExit(f"Unknown HPC in --partition-override {raw!r}; known values: {known}.") from exc
    return parsed


def generate_experiment_yaml_from_flags(args: argparse.Namespace, *, orchestrator_cls: type[BaseOrchestrator]) -> str:
    """Generate an experiment YAML from generic custom-sweep CLI flags."""
    if not getattr(args, "run_custom_sweeps", False):
        raise ValueError("Only --sweepfile custom-sweep generation is supported.")
    if not getattr(args, "custom_sweep_file", None):
        raise ValueError("--sweepfile requires a YAML spec path.")

    master_config = MasterExperimentConfig(
        run_custom_sweeps=True,
        custom_sweeps=_load_custom_sweeps(args.custom_sweep_file),
        num_epochs=args.num_epochs,
        config_profile=args.config_profile,
    )
    generator = orchestrator_cls(master_config)
    generator.generate_all_experiments()
    return generator.generate_experiment_file()


def _normalise_generic_generation_args(args: argparse.Namespace) -> None:
    """Normalize generic custom-sweep aliases."""
    if getattr(args, "custom_sweep_file", None):
        args.run_custom_sweeps = True
    if getattr(args, "yaml", None) and getattr(args, "run_custom_sweeps", False):
        raise SystemExit("--yaml cannot be combined with --sweepfile or --run-custom-sweeps.")


def _resolve_experiment_file(
    args: argparse.Namespace, *, experiment_generator: ExperimentGenerator | None = None
) -> str:
    """Return an experiment YAML path from --yaml or generation flags."""
    experiment_file = getattr(args, "yaml", None)
    if experiment_file:
        logger.info("Using existing experiment file: %s", experiment_file)
        return str(experiment_file)

    if getattr(args, "run_custom_sweeps", False):
        generator = experiment_generator or (
            lambda parsed_args: generate_experiment_yaml_from_flags(parsed_args, orchestrator_cls=BaseOrchestrator)
        )
        logger.info("Generating new experiment list based on provided run flags...")
        generated = generator(args)
        logger.info("Generated experiment file: %s", generated)
        return str(generated)

    raise SystemExit("No action specified. Provide --yaml or --sweepfile.")


def _load_custom_sweeps(path: str) -> list[CustomSweepConfig]:
    """Load generic custom-sweep specs from YAML."""
    data = load_yaml(path)
    sweeps_raw = data.get("custom_sweeps", data)
    if isinstance(sweeps_raw, dict):
        sweeps_raw = sweeps_raw.get("custom_sweeps", [])
    if not isinstance(sweeps_raw, list):
        raise ValueError("custom_sweeps must be a list in the provided YAML file.")
    return [CustomSweepConfig(**entry) for entry in sweeps_raw]


def _default_plugin_from_args(
    args: argparse.Namespace, *, orchestrator_settings: OrchestratorSettings | None = None
) -> OrchestratorPlugin:
    """Return a generic command-building plugin from parsed CLI arguments."""
    command_settings = orchestrator_settings.command if orchestrator_settings is not None else None
    entrypoint = getattr(args, "simple_command_entrypoint", None) or (
        command_settings.entrypoint if command_settings is not None else None
    )
    if entrypoint:
        config_arg = getattr(args, "simple_command_config_arg", None)
        if config_arg is None:
            config_arg = command_settings.config_arg if command_settings is not None else "--config"
        sweep_params_arg = getattr(args, "simple_command_sweep_params_arg", None) or (
            command_settings.sweep_params_arg if command_settings is not None else None
        )
        return SimpleCommandPlugin(
            entrypoint=entrypoint,
            config_field=command_settings.config_field if command_settings is not None else "config",
            config_arg=config_arg or None,
            extra_args=command_settings.extra_args if command_settings is not None else (),
            experiment_args_field=(
                command_settings.experiment_args_field if command_settings is not None else "command_args"
            ),
            sweep_params_arg=sweep_params_arg or None,
            orchestrator_flag=command_settings.orchestrator_flag if command_settings is not None else "--orchestrator",
            multi_experiment_flag=command_settings.multi_experiment_flag if command_settings is not None else None,
        )
    return DefaultOrchestratorPlugin()


def _bootstrap_connection_manager(
    concurrency_limits: Mapping[HPCType, int],
    *,
    cluster_configs: Mapping[HPCType, Any] | None = None,
    connection_manager_cls: type[HPCConnectionManager] = HPCConnectionManager,
) -> HPCConnectionManager | None:
    """Create and warm SSH connections for clusters with non-zero limits."""
    conn_cfgs = _connection_configs_for_limits(concurrency_limits, cluster_configs=cluster_configs)
    if not conn_cfgs:
        return None

    connection_manager = connection_manager_cls(conn_cfgs)
    for hpc in list(conn_cfgs.keys()):
        try:
            is_local = connection_manager.is_local_hpc(hpc)
            connection_manager.connect(hpc, force_remote=not is_local)
        except Exception as exc:
            logger.warning("Early SSH connect to %s failed: %s", hpc.name, exc)
    for hpc in list(conn_cfgs.keys()):
        try:
            is_local = connection_manager.is_local_hpc(hpc)
            connection_manager.run_command(hpc, "true", prefer_remote=not is_local)
        except Exception as exc:
            logger.warning("Early remote noop on %s failed: %s", hpc.name, exc)
    return connection_manager


def _connection_configs_for_limits(
    concurrency_limits: Mapping[HPCType, int], *, cluster_configs: Mapping[HPCType, Any] | None = None
) -> dict[HPCType, HPCConnectionConfig]:
    """Build connection configs, including proxy jump hosts, for active clusters."""
    registry = cluster_configs or HPC_CONFIGS
    conn_cfgs: dict[HPCType, HPCConnectionConfig] = {}
    for hpc_type, cfg in registry.items():
        if int(concurrency_limits.get(hpc_type, 0) or 0) <= 0:
            continue
        conn_cfgs[hpc_type] = _connection_config_from_cluster(cfg)

    jump_hosts_needed: set[HPCType] = set()
    for cfg in conn_cfgs.values():
        if not cfg.proxy_jump:
            continue
        try:
            jump_hpc_type = HPCType[cfg.proxy_jump.upper()]
        except KeyError:
            logger.warning("Unknown jump host type: %s", cfg.proxy_jump)
            continue
        if jump_hpc_type not in conn_cfgs and jump_hpc_type in registry:
            jump_hosts_needed.add(jump_hpc_type)

    for jump_type in jump_hosts_needed:
        conn_cfgs[jump_type] = _connection_config_from_cluster(registry[jump_type])
    return conn_cfgs


def _connection_config_from_cluster(cluster_config: Any) -> HPCConnectionConfig:
    """Build a connection-manager config from a cluster registry entry."""
    return HPCConnectionConfig(
        hostname=cluster_config.hostname,
        username=cluster_config.username,
        port=cluster_config.port,
        use_key=cluster_config.use_key,
        key_path=cluster_config.key_path,
        two_factor=cluster_config.two_factor,
        keep_alive=True,
        keep_alive_interval=30,
        proxy_jump=getattr(cluster_config, "proxy_jump", None),
        proxy_jump_username=getattr(cluster_config, "proxy_jump_username", None),
        proxy_jump_port=getattr(cluster_config, "proxy_jump_port", 22),
        submission_host=getattr(cluster_config, "submission_host", None),
        submission_username=getattr(cluster_config, "submission_username", None),
        submission_port=getattr(cluster_config, "submission_port", None),
        submission_use_key=getattr(cluster_config, "submission_use_key", None),
        submission_key_path=getattr(cluster_config, "submission_key_path", None),
        submission_two_factor=getattr(cluster_config, "submission_two_factor", None),
    )


def load_object(dotted_path: str) -> Any:
    """Load an object from ``module:attr`` or ``module.attr`` dotted syntax."""
    if ":" in dotted_path:
        module_name, attr_name = dotted_path.split(":", 1)
    else:
        module_name, attr_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])

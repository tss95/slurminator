import copy
import logging
import os
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

# NOTE: The Rich-based dashboard has heavy runtime deps. Import it lazily so
# unit tests (and --debug mode) can run without Rich installed.

from slurminator.cli.override_parser import parse_override_list
from slurminator.command_queue import CommandQueueContext, default_command_handlers, process_command_queue
from slurminator.config import HPCType, HPC_CONFIGS, is_current_hpc
from slurminator.experiments import ExperimentStatus
from slurminator.experiment_policy import resolve_extra_remote_dirs, resolve_resource_overrides
from slurminator.hpc_state import is_terminal_status
from slurminator.plugins import CommandBuildContext, DefaultOrchestratorPlugin, OrchestratorPlugin
from slurminator.connection_manager import HPCConnectionManager, HPCConnectionConfig
from slurminator.log_gathering import LogGatheringContext, LogTailReadResult, gather_logs, read_log_tail_incremental
from slurminator.reassignment import ReassignmentContext, maybe_reassign_experiments
from slurminator.scheduler_polling import expand_short, map_state, poll_hpc, update_scheduler_statuses
from slurminator.state_store import ExperimentStateStore, replace_exp_in_list
from slurminator.status_ingest import (
    StatusIngestContext,
    apply_target_status_to_experiment,
    extract_display_metrics,
    force_read_full_history as force_read_full_history_from_status,
    populate_display_metrics,
    update_experiment_config_with_metrics,
    update_running_experiment_info,
)
from slurminator.submission import SubmissionContext, maybe_submit, submit_experiment_universal
from slurminator.timeout_policy import (
    apply_timeout_policy,
    estimate_timeout_retry_hours,
    resolve_progress_fraction,
    resolve_requested_time_hours,
)

# Backward-compatibility alias for tests and any external imports.
_parse_override_list = parse_override_list

logger = logging.getLogger("slurminator")

IsLocalHPC = Callable[[HPCType], bool]
OverviewPrinter = Callable[[list[dict[str, Any]]], None]


def _call_plugin_noarg_hook(plugin: Any, hook_name: str) -> Any | None:
    """Call an optional no-argument plugin hook or return an attribute value."""
    value = getattr(plugin, hook_name, None)
    if value is None:
        return None
    return value() if callable(value) else value


def _get_plugin_callable(plugin: Any, hook_name: str, default: Callable[..., Any]) -> Callable[..., Any]:
    """Return an optional callable plugin hook or ``default``."""
    value = getattr(plugin, hook_name, None)
    return value if callable(value) else default


class HPCOrchestrator:
    """
    HPCOrchestrator that:
      - Reads an 'experiment_file' YAML with multiple experiments.
      - Submits experiments via a universal job script (universal_job.sh).
      - Polls HPC job states (squeue + sacct) to update statuses.
      - Reassigns queued experiments if HPC is overloaded too long.
      - Exits once all experiments are terminal.
      - Optionally retries TIMEOUT jobs with an estimated longer walltime.
    """

    def __init__(
        self,
        experiment_file: str,
        concurrency_limits: Dict[HPCType, int],
        poll_interval: int = 2,
        max_unqueue_seconds: int = 600,
        local_log_dir: str = "local_logs",
        max_gpus_per_job: Optional[int] = None,
        time_hours_override: Optional[int] = None,
        memory_gb_override: Optional[int] = None,
        debug: bool = False,
        dashboard_ui: str = "v3",
        *,
        connection_manager: Optional["HPCConnectionManager"] = None,
        retry_timeout_with_estimated_time: bool = False,
        timeout_retry_buffer: float = 1.3,
        timeout_retry_max_attempts: int = 1,
        plugin: Optional[OrchestratorPlugin] = None,
        runtime_options: Optional[Mapping[str, Any]] = None,
        partition_overrides: Optional[Mapping[HPCType, str | None]] = None,
        projection_options: Optional[Mapping[str, Any]] = None,
        parse_overrides: Optional[Callable[[list[str] | str], dict[str, Any]]] = None,
        is_local_hpc_fn: Optional[IsLocalHPC] = None,
        dashboard_cls: Optional[Type[Any]] = None,
        dashboard_settings: object | None = None,
        overview_printer: Optional[OverviewPrinter] = None,
    ):
        self.experiment_file = Path(experiment_file).resolve()
        self.experiment_dir = self.experiment_file.parent

        self.output_dir = self.experiment_dir / "outputs" / self.experiment_file.stem
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.concurrency_limits = concurrency_limits or {}
        self.submissions_paused = False
        self._dashboard_exit_requested = False
        self._dashboard_snapshot: list[dict[str, Any]] = []
        self.state_store = ExperimentStateStore(self.experiment_file, self.concurrency_limits)
        self.poll_interval = poll_interval
        self.max_unqueue_seconds = max_unqueue_seconds

        self.local_log_dir = Path(local_log_dir).resolve()
        self.local_log_dir.mkdir(parents=True, exist_ok=True)

        self.max_gpus_per_job = max_gpus_per_job
        if time_hours_override is not None and time_hours_override <= 0:
            raise ValueError("time_hours_override must be a positive integer when provided")
        if memory_gb_override is not None and memory_gb_override <= 0:
            raise ValueError("memory_gb_override must be a positive integer when provided")
        if timeout_retry_buffer < 1.0:
            raise ValueError("timeout_retry_buffer must be >= 1.0")
        if timeout_retry_max_attempts < 0:
            raise ValueError("timeout_retry_max_attempts must be >= 0")
        self.time_hours_override = time_hours_override
        self.memory_gb_override = memory_gb_override

        self.debug = debug
        self.dashboard_ui = dashboard_ui
        self.dashboard_settings = dashboard_settings
        self.retry_timeout_with_estimated_time = retry_timeout_with_estimated_time
        self.timeout_retry_buffer = timeout_retry_buffer
        self.timeout_retry_max_attempts = timeout_retry_max_attempts
        self.plugin = plugin or DefaultOrchestratorPlugin()
        self.runtime_options = dict(runtime_options or {})
        self.partition_overrides = {
            hpc_type: str(partition).strip()
            for hpc_type, partition in dict(partition_overrides or {}).items()
            if partition is not None and str(partition).strip()
        }
        plugin_projection_options = _call_plugin_noarg_hook(self.plugin, "status_projection_options")
        self.projection_options = dict(
            projection_options if projection_options is not None else plugin_projection_options or {}
        )
        self.parse_overrides = parse_overrides or _get_plugin_callable(
            self.plugin, "parse_sweep_overrides", parse_override_list
        )
        self.is_local_hpc_fn = is_local_hpc_fn or _get_plugin_callable(self.plugin, "is_local_hpc", is_current_hpc)
        self.dashboard_cls = dashboard_cls or _call_plugin_noarg_hook(self.plugin, "dashboard_class")
        self.overview_printer = overview_printer or _call_plugin_noarg_hook(self.plugin, "overview_printer")

        # Build or reuse HPC connections
        if connection_manager is not None:
            # Reuse the pre-connected manager supplied by caller
            self.connection_manager = connection_manager
        else:
            connection_configs = {}
            for hpc_type, c in HPC_CONFIGS.items():
                # Only include HPCs with concurrency limit > 0
                if self.concurrency_limits.get(hpc_type, 0) > 0:
                    if not c.account:
                        raise ValueError(f"HPC {hpc_type} has no 'account' set in HPC_CONFIGS.")
                    connection_configs[hpc_type] = HPCConnectionConfig(
                        hostname=c.hostname,
                        username=c.username,
                        port=c.port,
                        use_key=c.use_key,
                        key_path=c.key_path,
                        two_factor=c.two_factor,
                        keep_alive=True,
                        keep_alive_interval=int(os.getenv('SLURMINATOR_SSH_KEEPALIVE_INTERVAL', '30')),
                        proxy_jump=getattr(c, 'proxy_jump', None),
                        proxy_jump_username=getattr(c, 'proxy_jump_username', None),
                        proxy_jump_port=getattr(c, 'proxy_jump_port', 22),
                        submission_host=getattr(c, 'submission_host', None),
                        submission_username=getattr(c, 'submission_username', None),
                        submission_port=getattr(c, 'submission_port', None),
                        submission_use_key=getattr(c, 'submission_use_key', None),
                        submission_key_path=getattr(c, 'submission_key_path', None),
                        submission_two_factor=getattr(c, 'submission_two_factor', None),
                    )

            # Automatically include jump hosts needed for proxy connections
            jump_hosts_needed = set()
            for hpc_type, cfg in connection_configs.items():
                if cfg.proxy_jump:
                    try:
                        jump_hpc_type = HPCType[cfg.proxy_jump.upper()]
                        if jump_hpc_type not in connection_configs and jump_hpc_type in HPC_CONFIGS:
                            jump_hosts_needed.add(jump_hpc_type)
                    except KeyError:
                        logger.warning(f"Unknown jump host type: {cfg.proxy_jump}")

            # Add jump host configs
            for jump_type in jump_hosts_needed:
                jump_cfg = HPC_CONFIGS[jump_type]
                connection_configs[jump_type] = HPCConnectionConfig(
                    hostname=jump_cfg.hostname,
                    username=jump_cfg.username,
                    port=jump_cfg.port,
                    use_key=jump_cfg.use_key,
                    key_path=jump_cfg.key_path,
                    two_factor=jump_cfg.two_factor,
                    keep_alive=True,
                    keep_alive_interval=int(os.getenv('SLURMINATOR_SSH_KEEPALIVE_INTERVAL', '30')),
                    proxy_jump=getattr(jump_cfg, 'proxy_jump', None),
                    proxy_jump_username=getattr(jump_cfg, 'proxy_jump_username', None),
                    proxy_jump_port=getattr(jump_cfg, 'proxy_jump_port', 22),
                )

            self.connection_manager = HPCConnectionManager(configs=connection_configs)

        # ---------------------------------------------------------------
        # Detect if the supplied YAML defines a sweep (multiple experiments)
        # ---------------------------------------------------------------
        try:
            data_yaml = self._load_yaml()
            if isinstance(data_yaml, dict):
                self.multi_experiment = len(data_yaml.get("experiments", [])) > 1
            else:
                self.multi_experiment = False
        except Exception:
            # In case of any parsing error, default to single-experiment behaviour
            self.multi_experiment = False
        logger.debug(f"[HPCOrchestrator] multi_experiment set to {self.multi_experiment}")

    def run(self):
        """Main loop: manage experiments and render progress."""
        # Lazily import the Rich dashboard only when we actually need it to avoid
        # hard dependency issues in "--debug" / unit-test scenarios where Rich
        # might not be available.

        # ---------------------------------------------------------------
        # EARLY HPC CONNECTION
        # ---------------------------------------------------------------
        # Establish the SSH connections right away so that any interactive
        # password / 2-FA prompts are shown immediately.  This allows the
        # user to authenticate while the potentially slow pre-flight sanity
        # checks run, reducing overall perceived startup latency.
        try:
            logger.info("=== HPCOrchestrator: connecting to HPCs (early) ... ===")
            self.connection_manager.connect_all()
            logger.info("Early HPC connections established.")
        except Exception:
            logger.error("Early HPC connection failed. Aborting orchestrator startup.")
            raise

        # ------------------------------------------------------------------
        # 0) Sanity-check experiment-level overrides (if any) before doing
        #    any SSH connections or job submissions.  This catches typos in
        #    manual experiment YAMLs early and avoids queuing bad jobs.
        # ------------------------------------------------------------------
        try:
            data = self._load_yaml()

            orchestrator_logger = logging.getLogger("slurminator")
            _orig_level = orchestrator_logger.level
            orchestrator_logger.setLevel(logging.WARNING)

            validated_exps = 0

            for exp in data.get("experiments", []):  # type: ignore[arg-type]
                dataset_val = exp.get("dataset_name")  # type: ignore[call-arg]
                if not dataset_val:
                    continue  # Skip entries without dataset – they will fail later anyway

                # Ensure dataset is a plain string for the config loader
                dataset_str: str = str(dataset_val)

                override_str = exp.get("sweep_params")  # type: ignore[call-arg]
                if not override_str:
                    if self.plugin.validate_experiment(exp, {}):
                        validated_exps += 1
                    continue  # Nothing to validate

                try:
                    overrides = self.parse_overrides(override_str)
                except Exception as e:
                    logger.error(
                        "Failed to parse sweep_params for experiment '%s': %s", exp.get("experiment_id", "<unknown>"), e
                    )
                    raise

                try:
                    if self.plugin.validate_experiment(exp, overrides):
                        validated_exps += 1
                except Exception as e:
                    logger.error(
                        "Config sanity-check failed for experiment '%s' (dataset=%s): %s",
                        exp.get("experiment_id", "<unknown>"),
                        dataset_str,
                        e,
                    )
                    raise

            # Restore log level
            orchestrator_logger.setLevel(_orig_level)

            logger.info(
                "Experiment YAML sanity-check passed – %d experiments with overrides validated.", validated_exps
            )

            # (multi_experiment already computed in __init__)
        except Exception:
            logger.error("Pre-flight sanity-check failed. Aborting orchestrator startup.")
            raise

        try:
            # ------------------------------------------------------------------
            # 1) HPC connections are already established earlier.  We still
            #    perform initial remote-directory sanity work, but avoid a
            #    second (redundant) SSH login attempt.
            # ------------------------------------------------------------------
            logger.info("=== HPCOrchestrator: reusing existing HPC connections. ===")

            # Preflight: verify Slurm tools and repo availability on submission hosts
            self._preflight_test_hpcs()

            # Ensure HPC directories
            for hpc_type in self.connection_manager.configs.keys():
                if self.concurrency_limits.get(hpc_type, 0) > 0:
                    self._ensure_remote_dirs(hpc_type)
                    self.plugin.prepare_remote_runtime(hpc_type=hpc_type, connection_manager=self.connection_manager)

            # Fix potential orphans before we start the dashboard
            self._recover_orphans()

            # ------------------------------------------------------------------
            # 2) Enter UI loop
            # ------------------------------------------------------------------
            if self.debug:
                if self.overview_printer is None:
                    from slurminator.orchestrator_ui import print_overview
                else:
                    print_overview = self.overview_printer

                while True:
                    data = self._load_yaml()
                    exps = data["experiments"]

                    self._process_command_queue(exps)
                    self._update_statuses(exps)
                    self._update_queue_estimates(exps)
                    data["experiments"] = exps
                    self._save_yaml(data)

                    concurrency_used = self._count_concurrency(exps)

                    if not self.submissions_paused:
                        for exp in exps:
                            self._maybe_submit(exp, concurrency_used, data)

                        self._maybe_reassign_experiments(exps, concurrency_used, data)

                    data["experiments"] = exps
                    self._save_yaml(data)
                    self._publish_dashboard_snapshot(exps)

                    print_overview(exps)

                    if self._all_done(exps):
                        logger.info("All experiments terminal => exiting orchestrator.")
                        break

                    time.sleep(self.poll_interval)
            else:
                DashboardCls = self._resolve_dashboard_cls()
                dashboard_kwargs = {"n_recent": 0, "ui_version": self.dashboard_ui}
                if str(self.dashboard_ui).strip().lower() == "v4":
                    dashboard_kwargs["sparkline_thresholds"] = getattr(self.dashboard_settings, "sparkline", None)
                dash = DashboardCls(**dashboard_kwargs)

                with dash.mount(self) as live:
                    while True:
                        data = self._load_yaml()
                        exps = data["experiments"]

                        self._process_command_queue(exps)
                        if self._dashboard_requested_exit(dash):
                            logger.info("Dashboard requested orchestrator exit.")
                            break
                        self._update_statuses(exps)
                        self._update_queue_estimates(exps)
                        data["experiments"] = exps
                        self._save_yaml(data)
                        if self._dashboard_requested_exit(dash):
                            logger.info("Dashboard requested orchestrator exit.")
                            break

                        concurrency_used = self._count_concurrency(exps)

                        if not self.submissions_paused:
                            for exp in exps:
                                self._maybe_submit(exp, concurrency_used, data)

                            self._maybe_reassign_experiments(exps, concurrency_used, data)

                        data["experiments"] = exps
                        self._save_yaml(data)

                        # Ensure all experiments have display metrics populated before rendering
                        populate_display_metrics(exps)
                        self._publish_dashboard_snapshot(exps)
                        live.update(dash.render(exps))

                        if self._dashboard_requested_exit(dash):
                            logger.info("Dashboard requested orchestrator exit.")
                            break

                        if self._all_done(exps):
                            logger.info("All experiments terminal => exiting orchestrator.")
                            # Ensure display metrics are populated for final render
                            populate_display_metrics(exps)
                            self._publish_dashboard_snapshot(exps)
                            live.update(dash.render(exps))
                            break

                        if self._sleep_until_next_poll(dash):
                            logger.info("Dashboard requested orchestrator exit.")
                            break

        except KeyboardInterrupt:
            logger.info("User interrupted HPC Orchestrator.")
        except Exception as e:
            logger.error(f"Orchestrator error: {e}", exc_info=True)
            raise
        finally:
            logger.info("Closing HPC connections.")
            self.connection_manager.close_all()

    def _resolve_dashboard_cls(self):
        """Resolve the concrete dashboard class for the requested UI version."""
        if str(self.dashboard_ui).strip().lower() == "v4":
            from slurminator.dashboard_v4.app import TextualDashboardApp

            return TextualDashboardApp
        if self.dashboard_cls is not None:
            return self.dashboard_cls
        from slurminator.ui_dashboard import TerminalDashboard

        return TerminalDashboard

    def _publish_dashboard_snapshot(self, exps: list[dict[str, Any]]) -> None:
        """Publish a copy of the latest ledger for threaded dashboard readers."""
        self._dashboard_snapshot = copy.deepcopy(exps)

    def _dashboard_requested_exit(self, dashboard: Any) -> bool:
        """Return True when a dashboard requested the orchestrator loop to stop."""
        return bool(
            getattr(self, "_dashboard_exit_requested", False) or getattr(dashboard, "dashboard_exit_requested", False)
        )

    def _sleep_until_next_poll(self, dashboard: Any) -> bool:
        """Sleep until the next poll, waking early for dashboard exit requests."""
        deadline = time.monotonic() + max(float(self.poll_interval), 0.0)
        while True:
            if self._dashboard_requested_exit(dashboard):
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return self._dashboard_requested_exit(dashboard)
            time.sleep(min(remaining, 0.2))

    # -------------------------------------------------------------------------
    # Preflight checks
    # -------------------------------------------------------------------------
    def _preflight_test_hpcs(self) -> None:
        """Run a lightweight sanity test on each enabled HPC via remote path.

        - Confirms sbatch/squeue availability and prints version.
        - Confirms repo path exists and, if a git repo, prints HEAD short SHA.

        Raises RuntimeError if Slurm (sbatch) is missing on any required HPC.
        """
        for hpc_type in self.connection_manager.configs.keys():
            if self.concurrency_limits.get(hpc_type, 0) <= 0:
                continue

            c = HPC_CONFIGS.get(hpc_type)
            repo = c.repo_path if c and c.repo_path else ""

            test_cmd = (
                "set -e; "
                "echo '[preflight]' $(hostname); "
                "if command -v sbatch >/dev/null 2>&1; then sbatch --version | head -n1; "
                "else echo 'ERROR: sbatch not found'; exit 2; fi; "
                "if command -v squeue >/dev/null 2>&1; then (squeue --version 2>/dev/null || squeue -V 2>/dev/null || true); "
                "else echo 'WARN: squeue not found'; fi; "
                f"if [ -n '{repo}' ] && [ -d '{repo}' ]; then echo 'repo:' '{repo}'; "
                f"  if [ -d '{repo}/.git' ]; then git -C '{repo}' rev-parse --short HEAD 2>/dev/null || true; else echo 'WARN: not a git repo'; fi; "
                "else echo 'WARN: repo path missing'; fi;"
            )

            # Always preflight through the submission/remote execution path so
            # we validate the same Slurm environment used for actual submits.
            out, err = self.connection_manager.run_command(hpc_type, test_cmd, prefer_remote=True)
            if out.strip():
                logger.info(f"[Preflight {hpc_type.name}]\n{out.strip()}")
            if err.strip():
                logger.warning(f"[Preflight {hpc_type.name} stderr] \n{err.strip()}")

            if "ERROR: sbatch not found" in out:
                raise RuntimeError(f"Slurm not available on {hpc_type.name} submission host")

            # Fail fast if sbatch exists but is not operational (e.g., missing
            # config/plugins inside container, DNS SRV failures, auth plugin errors).
            if "sbatch:" in err.lower() and "error" in err.lower():
                first = err.strip().splitlines()[0] if err.strip() else "unknown sbatch error"
                raise RuntimeError(f"Slurm preflight failed on {hpc_type.name}: {first}")

        logger.info("Preflight checks passed on all enabled HPCs.")

    # -------------------------------------------------------------------------
    # Orphan fix
    # -------------------------------------------------------------------------
    def _recover_orphans(self):
        """
        If experiment is QUEUED/RUNNING but missing 'job_id', revert => PENDING
        """
        data = self._load_yaml()
        changed = False
        for exp in data["experiments"]:
            st = exp.get("status")
            hpc = exp.get("hpc_assignment")
            # Only recover orphans from HPCs with concurrency limit > 0
            if (
                st in [ExperimentStatus.QUEUED, ExperimentStatus.RUNNING]
                and not exp.get("job_id")
                and hpc
                and self.concurrency_limits.get(hpc, 0) > 0
            ):
                logger.warning(f"Orphan {exp['experiment_id']} => revert => PENDING")
                exp["status"] = ExperimentStatus.PENDING
                changed = True
        if changed:
            self._save_yaml(data)

    # -------------------------------------------------------------------------
    # HPC Status Polling
    # -------------------------------------------------------------------------
    def _update_statuses(self, exps: List[dict]):
        """
        Poll HPC states via squeue/sacct. Update statuses for queued/running exps.
        If a job is terminal, gather logs and apply terminal-status policies.
        """
        update_scheduler_statuses(
            exps,
            connection_manager=self.connection_manager,
            concurrency_limits=self.concurrency_limits,
            gather_logs=self._gather_logs,
        )

        # --- NEW: live metric progress via status files -----------------
        for e in exps:
            if e.get("status") == ExperimentStatus.RUNNING:
                try:
                    self._update_running_experiment_info(e)
                except Exception:
                    pass
            elif is_terminal_status(e.get("status")):
                # Fetch final telemetry if the retained status file is still present.
                # Terminal lifecycle is owned by this experiment YAML / sacct, not by
                # status-file presence.
                try:
                    self._update_running_experiment_info(e)
                except Exception:
                    pass

    def _poll_hpc(self, hpc_type: HPCType, jobids: List[str]) -> Dict[str, str]:
        """
        Return {job_id: HPC textual state} from squeue + sacct.
        """
        return poll_hpc(self.connection_manager, hpc_type, jobids)

    def _expand_short(self, code: str) -> str:
        """
        Convert short Slurm code or sacct state to standard string.
        """
        return expand_short(code)

    def _map_state(self, st: str) -> ExperimentStatus:
        """
        Map HPC textual state to an ExperimentStatus enum.
        """
        return map_state(st)

    @staticmethod
    def _resolve_progress_fraction(exp: dict) -> Optional[float]:
        """Return best-known completion fraction in (0, 1], or None when unavailable."""
        return resolve_progress_fraction(exp)

    def _resolve_requested_time_hours(self, exp: dict, hpc_type: HPCType) -> int:
        """Best-effort resolution of walltime hours used by the latest submission."""
        cluster_cfg = HPC_CONFIGS.get(hpc_type)
        resource_overrides = resolve_resource_overrides(exp, hpc_type=hpc_type, cluster_configs=HPC_CONFIGS)
        return resolve_requested_time_hours(
            exp,
            hpc_type,
            cluster_config=cluster_cfg,
            resource_overrides=resource_overrides,
            global_time_hours_override=self.time_hours_override,
        )

    def _estimate_timeout_retry_hours(self, exp: dict, hpc_type: HPCType) -> Optional[int]:
        """Estimate required walltime after timeout using observed progress + configured buffer."""
        cluster_cfg = HPC_CONFIGS.get(hpc_type)
        resource_overrides = resolve_resource_overrides(exp, hpc_type=hpc_type, cluster_configs=HPC_CONFIGS)
        return estimate_timeout_retry_hours(
            exp,
            hpc_type,
            cluster_config=cluster_cfg,
            resource_overrides=resource_overrides,
            global_time_hours_override=self.time_hours_override,
            timeout_retry_buffer=self.timeout_retry_buffer,
        )

    def _apply_timeout_policy(self, exp: dict, hpc_type: HPCType, reason: str) -> None:
        """Apply timeout handling policy for a finished experiment."""
        cluster_cfg = HPC_CONFIGS.get(hpc_type)
        resource_overrides = resolve_resource_overrides(exp, hpc_type=hpc_type, cluster_configs=HPC_CONFIGS)
        apply_timeout_policy(
            exp,
            hpc_type,
            reason=reason,
            cluster_config=cluster_cfg,
            resource_overrides=resource_overrides,
            global_time_hours_override=self.time_hours_override,
            retry_timeout_with_estimated_time=self.retry_timeout_with_estimated_time,
            timeout_retry_buffer=self.timeout_retry_buffer,
            timeout_retry_max_attempts=self.timeout_retry_max_attempts,
        )

    def _gather_logs(self, exp: dict, job_id: str, hpc_type: HPCType):
        """Inspect Slurm logs and let the configured plugin refine terminal status."""
        context = LogGatheringContext(
            connection_manager=self.connection_manager,
            hpc_configs=HPC_CONFIGS,
            plugin=self.plugin,
            is_local_hpc=self.is_local_hpc,
            global_time_hours_override=self.time_hours_override,
            retry_timeout_with_estimated_time=self.retry_timeout_with_estimated_time,
            timeout_retry_buffer=self.timeout_retry_buffer,
            timeout_retry_max_attempts=self.timeout_retry_max_attempts,
        )
        gather_logs(exp, job_id, hpc_type, context)

    # -------------------------------------------------------------------------
    # Concurrency
    # -------------------------------------------------------------------------
    def _count_concurrency(self, experiments: List[dict]) -> Dict[HPCType, int]:
        """
        Count how many jobs are currently QUEUED or RUNNING on each HPC.
        """
        usage = {h: 0 for h, limit in self.concurrency_limits.items() if limit > 0}
        for e in experiments:
            if e.get("status") in {ExperimentStatus.QUEUED, ExperimentStatus.RUNNING}:
                h = e.get("hpc_assignment")
                if h in usage:
                    usage[h] += 1
                else:
                    logger.warning(f"{e['experiment_id']} queued/running on disabled HPC {h}")
        return usage

    # -------------------------------------------------------------------------
    # Submission
    # -------------------------------------------------------------------------
    def _maybe_submit(self, exp: dict, concurrency_used: Dict[HPCType, int], data: dict):
        """
        If PENDING or PARTIAL => pick HPC if not assigned, check concurrency, do sbatch universal_job.sh ...
        """
        maybe_submit(
            exp,
            concurrency_used,
            data,
            concurrency_limits=self.concurrency_limits,
            hpc_configs=HPC_CONFIGS,
            submit_experiment=self._submit_experiment_universal,
            replace_exp_in_list=self._replace_exp_in_list,
            save_yaml=self._save_yaml,
        )

    def _submit_experiment_universal(self, exp: dict, hpc_type: HPCType) -> Optional[str]:
        context = SubmissionContext(
            experiment_file=self.experiment_file,
            concurrency_limits=self.concurrency_limits,
            hpc_configs=HPC_CONFIGS,
            connection_manager=self.connection_manager,
            build_commands_line=self._build_commands_line,
            is_local_hpc=self.is_local_hpc,
            max_gpus_per_job=self.max_gpus_per_job,
            time_hours_override=self.time_hours_override,
            memory_gb_override=self.memory_gb_override,
            partition_overrides=self.partition_overrides,
        )
        return submit_experiment_universal(exp, hpc_type, context)

    def _build_commands_line(self, exp: dict, gpus: int, hpc_type: HPCType) -> str:
        """
        Build the command string that universal_job.sh will execute.
        Project-specific command semantics are supplied by the orchestrator plugin.
        """
        context = CommandBuildContext(
            gpus=gpus,
            hpc_type=hpc_type,
            multi_experiment=bool(getattr(self, "multi_experiment", False)),
            runtime_options=self.runtime_options,
        )
        return self.plugin.build_commands_line(exp, context)

    # -----------------------------------------------------------------
    # Helper: read remote status-file → enrich exp dict for dashboard
    # -----------------------------------------------------------------
    def _update_running_experiment_info(self, exp: dict) -> Optional[dict]:
        context = StatusIngestContext(
            connection_manager=self.connection_manager,
            hpc_configs=HPC_CONFIGS,
            load_yaml=self._load_yaml,
            save_yaml=self._save_yaml,
            projection_options=self.projection_options,
        )
        return update_running_experiment_info(exp, context)

    def _apply_target_status_to_experiment(self, exp: dict, status) -> None:
        apply_target_status_to_experiment(exp, status, self.projection_options)

    def _update_experiment_config_with_metrics(self, exp: dict, data: dict) -> None:
        """Update the experiment config file with the latest metrics from status file."""
        context = StatusIngestContext(
            connection_manager=self.connection_manager,
            hpc_configs=HPC_CONFIGS,
            load_yaml=self._load_yaml,
            save_yaml=self._save_yaml,
            projection_options=self.projection_options,
        )
        update_experiment_config_with_metrics(exp, data, context)

    def force_read_full_history(self, exp: dict[str, Any]) -> None:
        """Force a one-shot full history read for dashboard drill-in screens."""
        if not exp.get("save_path"):
            hpc_type = exp.get("hpc_assignment")
            cluster_config = HPC_CONFIGS.get(hpc_type)
            if cluster_config is None and hpc_type is not None:
                try:
                    cluster_config = HPC_CONFIGS.get(HPCType[str(hpc_type).upper()])
                except KeyError:
                    cluster_config = None
            save_path = getattr(cluster_config, "save_path", None) if cluster_config else None
            if save_path:
                exp["save_path"] = str(save_path)

        if not exp.get("job_id") or not exp.get("hpc_assignment") or not exp.get("save_path"):
            return

        context = StatusIngestContext(
            connection_manager=self.connection_manager,
            hpc_configs=HPC_CONFIGS,
            load_yaml=self._load_yaml,
            save_yaml=self._save_yaml,
            projection_options=self.projection_options,
        )
        force_read_full_history_from_status(exp, context)

    def read_log_tail_for(
        self, exp: dict[str, Any], *, lines: int = 500, offsets: Mapping[str, int] | None = None
    ) -> LogTailReadResult:
        """Read recent or newly-appended Slurm log text for a dashboard screen."""
        job_id = exp.get("job_id")
        hpc_type = self._coerce_hpc_type(exp.get("hpc_assignment"))
        if not job_id or hpc_type is None:
            return LogTailReadResult(text="", offsets=dict(offsets or {}))
        context = LogGatheringContext(
            connection_manager=self.connection_manager,
            hpc_configs=HPC_CONFIGS,
            plugin=self.plugin,
            is_local_hpc=self.is_local_hpc,
            global_time_hours_override=self.time_hours_override,
            retry_timeout_with_estimated_time=self.retry_timeout_with_estimated_time,
            timeout_retry_buffer=self.timeout_retry_buffer,
            timeout_retry_max_attempts=self.timeout_retry_max_attempts,
        )
        return read_log_tail_incremental(exp, str(job_id), hpc_type, context, lines=lines, offsets=offsets)

    @staticmethod
    def _coerce_hpc_type(value: object) -> HPCType | None:
        if isinstance(value, HPCType):
            return value
        if value is None:
            return None
        text = str(value).strip()
        try:
            return HPCType(text)
        except ValueError:
            try:
                return HPCType[text.upper()]
            except KeyError:
                return None

    def _extract_display_metrics(self, exp: dict) -> dict:
        """Extract display metrics for UI from target-schema display metadata.

        Status files now carry their own display metadata. This helper projects
        any metric shortforms into the experiment row before rendering.
        """
        return extract_display_metrics(exp)

    # -------------------------------------------------------------------------
    # Reassign logic
    # -------------------------------------------------------------------------
    def _maybe_reassign_experiments(self, exps: List[dict], concurrency_used: Dict[HPCType, int], data: dict):
        """
        If HPC is overloaded but another HPC is free,
        reassign queued experiments older than HPC_CONFIGS[hpc].unqueue_threshold_secs
        (skipping pinned datasets).
        """
        context = ReassignmentContext(
            concurrency_limits=self.concurrency_limits,
            hpc_configs=HPC_CONFIGS,
            connection_manager=self.connection_manager,
            max_unqueue_seconds=self.max_unqueue_seconds,
            pending_status=ExperimentStatus.PENDING,
            queued_status=ExperimentStatus.QUEUED,
            partial_status=ExperimentStatus.PARTIAL,
            replace_exp_in_list=self._replace_exp_in_list,
            save_yaml=self._save_yaml,
        )
        maybe_reassign_experiments(exps, concurrency_used, data, context)

    def _process_command_queue(self, exps: list[dict[str, Any]]) -> int:
        """Process pending operator commands for all known command queue roots."""
        processed = 0
        for save_path in self._command_queue_save_paths(exps):
            context = CommandQueueContext(
                save_path=save_path,
                handlers=default_command_handlers(),
                exps=exps,
                orchestrator=self,
                connection_manager=self.connection_manager,
            )
            processed += process_command_queue(context)
        return processed

    def _command_queue_save_paths(self, exps: list[dict[str, Any]]) -> list[Path]:
        """Return local command-queue roots, preferring configured SAVE_PATH values."""
        paths: list[Path] = []
        paths.append(self.experiment_dir)

        env_save_path = os.getenv("SAVE_PATH")
        if env_save_path:
            paths.append(Path(env_save_path))

        for exp in exps:
            if exp.get("save_path"):
                paths.append(Path(str(exp["save_path"])))

        for hpc_type, limit in self.concurrency_limits.items():
            if int(limit or 0) <= 0:
                continue
            cluster_config = HPC_CONFIGS.get(hpc_type)
            save_path = getattr(cluster_config, "save_path", None) if cluster_config else None
            if save_path:
                paths.append(Path(str(save_path)))

        deduped: list[Path] = []
        seen: set[str] = set()
        for path in paths:
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(path)
        return deduped

    # -------------------------------------------------------------------------
    # Done check
    # -------------------------------------------------------------------------
    def _all_done(self, exps: List[dict]) -> bool:
        terminal = {
            ExperimentStatus.COMPLETED,
            ExperimentStatus.FAILED,
            ExperimentStatus.CANCELLED,
            ExperimentStatus.TIMEOUT,
            ExperimentStatus.OOM,
            ExperimentStatus.PARTIAL,
            ExperimentStatus.KILLED,
        }
        return all(e["status"] in terminal for e in exps)

    # -------------------------------------------------------------------------
    # read/save YAML
    # -------------------------------------------------------------------------
    def _load_yaml(self) -> dict:
        """Load experiment YAML and ensure a dict with an 'experiments' list."""
        self.state_store.concurrency_limits = self.concurrency_limits
        return self.state_store.load()

    def _save_yaml(self, data: dict):
        self.state_store.save(data)

    def _replace_exp_in_list(self, exps: List[dict], new_e: dict) -> List[dict]:
        return replace_exp_in_list(exps, new_e)

    # -------------------------------------------------------------------------
    # local HPC detection
    # -------------------------------------------------------------------------
    def is_local_hpc(self, hpc_type: HPCType) -> bool:
        """Return True if running on the given HPC type."""
        return self.is_local_hpc_fn(hpc_type)

    def _ensure_remote_dirs(self, hpc_type: HPCType):
        """
        Create HPC directories for logs/out on HPC side (mkdir -p) or locally if is_local_hpc.
        """
        # Only ensure remote directories for HPCs with concurrency limit > 0
        if self.concurrency_limits.get(hpc_type, 0) <= 0:
            return

        c = HPC_CONFIGS.get(hpc_type)  # type: ignore[arg-type]
        if not c:
            logger.error(f"No HPC config for {hpc_type}; cannot prepare dirs")
            return
        if not c.save_path:
            return

        basep = Path(c.save_path)
        subdirs = [
            basep / "experiment_lists" / "outputs" / self.experiment_file.stem,
            basep / "checkpoints",
            basep / "outputs",
            basep / "logs",
        ]
        try:
            data = self._load_yaml()
            for exp in data.get("experiments", []):
                subdirs.extend(resolve_extra_remote_dirs(exp, base_path=basep))
        except Exception as exc:
            logger.debug("Could not resolve extra remote dirs from experiment rows: %s", exc)

        if not self.is_local_hpc(hpc_type):
            for d in subdirs:
                cmd = f"mkdir -p {d}"
                self.connection_manager.run_command(hpc_type, cmd)
        else:
            for d in subdirs:
                d.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # Queued ETA polling
    # ---------------------------------------------------------------------
    def _update_queue_estimates(self, exps: List[dict]):
        """For queued jobs, ask Slurm for estimated start times via `squeue --start -j`.

        Stores the epoch timestamp in `estimated_start_ts` on each experiment dict.
        """
        hpc_map = {}
        for e in exps:
            if e.get("status") == ExperimentStatus.QUEUED and e.get("job_id") and e.get("hpc_assignment"):
                h = e["hpc_assignment"]
                if self.concurrency_limits.get(h, 0) > 0:
                    hpc_map.setdefault(h, []).append((e, e["job_id"]))

        for hpc, pairs in hpc_map.items():
            jobids = [jid for (_, jid) in pairs]
            joined = ",".join(jobids)
            cmd = f"squeue --start -h -o '%i %S' -j {joined}"
            out, _ = self.connection_manager.run_command(hpc, cmd, prefer_remote=True)
            eta_map = {}
            for line in out.splitlines():
                line = line.strip()
                if not line:
                    continue
                jid, timestr = line.split(maxsplit=1)
                # timestr may be "N/A" if unknown
                if timestr.upper() == "N/A":
                    continue
                # Slurm usually returns ISO 8601 or "now"; handle both
                if timestr.lower() == "now":
                    eta_ts = time.time()
                else:
                    try:
                        from datetime import datetime

                        eta_ts = datetime.strptime(timestr, "%Y-%m-%dT%H:%M:%S").timestamp()
                    except Exception:
                        continue
                eta_map[jid] = eta_ts

            for e, jid in pairs:
                if jid in eta_map:
                    e["estimated_start_ts"] = eta_map[jid]

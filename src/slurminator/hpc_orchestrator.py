import copy
import importlib
import logging
import math
import os
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

# NOTE: The Rich-based dashboard has heavy runtime deps. Import it lazily so
# unit tests (and --debug mode) can run without Rich installed.

from slurminator.cli.override_parser import parse_override_list
from slurminator.command_queue import CommandQueueContext, default_command_handlers, process_command_queue
from slurminator.config import HPCType, HPC_CONFIGS, is_current_hpc
from slurminator.experiments import ExperimentStatus
from slurminator.experiments.experiment_info import ExperimentInfo
from slurminator.experiment_policy import resolve_extra_remote_dirs, resolve_resource_overrides
from slurminator.hpc_state import is_terminal_status
from slurminator.plugins import CommandBuildContext, DefaultOrchestratorPlugin, OrchestratorPlugin
from slurminator.connection_manager import HPCConnectionManager, HPCConnectionConfig
from slurminator.log_gathering import (
    LogGatheringContext,
    LogSource,
    LogTailReadResult,
    gather_logs,
    read_log_tail_incremental,
)
from slurminator.reassignment import ReassignmentContext, maybe_reassign_experiments
from slurminator.scheduler_polling import (
    SchedulerStatusUpdateResult,
    expand_short,
    map_state,
    poll_hpc,
    update_scheduler_statuses,
)
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

DEFAULT_HIGH_VOLUME_THRESHOLD = 100
DEFAULT_STATE_CHECKPOINT_INTERVAL_SECONDS = 60.0
DEFAULT_TELEMETRY_INTERVAL_SECONDS = 10.0
DEFAULT_TELEMETRY_BATCH_SIZE = 64
LOW_COMPUTE_POLL_INTERVAL_SECONDS = 60
LOW_COMPUTE_CHECKPOINT_INTERVAL_SECONDS = 300.0
LOW_COMPUTE_TELEMETRY_BATCH_SIZE = 32
LOW_COMPUTE_SUBMISSION_BATCH_SIZE = 8
LOW_COMPUTE_SUBMISSION_BATCH_SECONDS = 2.0
LOW_COMPUTE_SUBMISSION_FILL_SIZE = 32
LOW_COMPUTE_SUBMISSION_FILL_SECONDS = 30.0
LOW_COMPUTE_SUBMISSION_PAUSE_SECONDS_PER_JOB = 1.0
HIGH_VOLUME_TRANSIENT_HISTORY_FIELDS = frozenset(
    {"history", "history_last_read_offset", "history_attempt_max", "history_truncated"}
)

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
        dashboard_ui: str = "v4",
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
        submission_batch_size: int = 8,
        submission_batch_seconds: float = 2.0,
        submission_checkpoint_size: int = 32,
        high_volume_threshold: int = DEFAULT_HIGH_VOLUME_THRESHOLD,
        state_checkpoint_interval_seconds: float = DEFAULT_STATE_CHECKPOINT_INTERVAL_SECONDS,
        telemetry_interval_seconds: float = DEFAULT_TELEMETRY_INTERVAL_SECONDS,
        telemetry_batch_size: int = DEFAULT_TELEMETRY_BATCH_SIZE,
        low_compute: bool = False,
    ) -> None:
        if low_compute and (not math.isfinite(poll_interval) or poll_interval <= 0):
            raise ValueError("poll_interval must be a positive finite number in low-compute mode")
        self.low_compute = low_compute
        self.experiment_file = Path(experiment_file).resolve()
        self.experiment_dir = self.experiment_file.parent
        self.experiment_info = ExperimentInfo()

        self.output_dir = self.experiment_dir / "outputs" / self.experiment_file.stem
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.concurrency_limits = concurrency_limits or {}
        self.submissions_paused = False
        self._dashboard_exit_requested = False
        self._dashboard_snapshot: list[dict[str, Any]] = []
        self._dashboard_snapshot_version = 0
        self.state_store = ExperimentStateStore(self.experiment_file, self.concurrency_limits)
        if submission_batch_size <= 0:
            raise ValueError("submission_batch_size must be a positive integer")
        self.submission_batch_size = int(submission_batch_size)
        if not math.isfinite(submission_batch_seconds) or submission_batch_seconds <= 0:
            raise ValueError("submission_batch_seconds must be a positive finite number")
        self.submission_batch_seconds = float(submission_batch_seconds)
        if submission_checkpoint_size <= 0:
            raise ValueError("submission_checkpoint_size must be a positive integer")
        self.submission_checkpoint_size = int(submission_checkpoint_size)
        if high_volume_threshold <= 0:
            raise ValueError("high_volume_threshold must be a positive integer")
        self.high_volume_threshold = int(high_volume_threshold)
        if not math.isfinite(state_checkpoint_interval_seconds) or state_checkpoint_interval_seconds <= 0:
            raise ValueError("state_checkpoint_interval_seconds must be a positive finite number")
        self.state_checkpoint_interval_seconds = float(state_checkpoint_interval_seconds)
        if not math.isfinite(telemetry_interval_seconds) or telemetry_interval_seconds <= 0:
            raise ValueError("telemetry_interval_seconds must be a positive finite number")
        self.telemetry_interval_seconds = float(telemetry_interval_seconds)
        if telemetry_batch_size <= 0:
            raise ValueError("telemetry_batch_size must be a positive integer")
        self.telemetry_batch_size = int(telemetry_batch_size)
        self.high_volume_mode = False
        self._last_state_checkpoint_at: float | None = None
        self._last_telemetry_poll_at: float | None = None
        self._telemetry_cursor = 0
        self._prepared_submission_repositories: set[HPCType] = set()
        self.poll_interval = poll_interval
        if self.low_compute:
            self.poll_interval = max(self.poll_interval, LOW_COMPUTE_POLL_INTERVAL_SECONDS)
            self.telemetry_interval_seconds = max(self.telemetry_interval_seconds, self.poll_interval)
            self.state_checkpoint_interval_seconds = max(
                self.state_checkpoint_interval_seconds, LOW_COMPUTE_CHECKPOINT_INTERVAL_SECONDS, self.poll_interval
            )
            self.telemetry_batch_size = min(self.telemetry_batch_size, LOW_COMPUTE_TELEMETRY_BATCH_SIZE)
            self.submission_batch_size = min(self.submission_batch_size, LOW_COMPUTE_SUBMISSION_BATCH_SIZE)
            self.submission_batch_seconds = min(self.submission_batch_seconds, LOW_COMPUTE_SUBMISSION_BATCH_SECONDS)
            logger.info(
                "Low-compute mode: scheduler/telemetry intervals >= %ss, YAML checkpoints >= %ss, "
                "at most %d status reads and paced submission batches (up to %d jobs / %.0fs) per cycle.",
                self.poll_interval,
                self.state_checkpoint_interval_seconds,
                self.telemetry_batch_size,
                min(self.submission_checkpoint_size, LOW_COMPUTE_SUBMISSION_FILL_SIZE),
                LOW_COMPUTE_SUBMISSION_FILL_SECONDS,
            )
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

        # Resolved from the already-required startup load in ``run``. Avoiding
        # an eager parse here matters for very large experiment ledgers.
        self.multi_experiment = False

    def run(self) -> None:
        """Manage experiments and print the YAML reference after shutdown."""
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
            self._configure_volume_mode(data)
            self._preflight_validate_experiments(data)
        except Exception:
            logger.error("Pre-flight sanity-check failed. Aborting orchestrator startup.")
            raise

        all_done = False
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
                    self._ensure_remote_dirs(hpc_type, data=data)
                    self.plugin.prepare_remote_runtime(hpc_type=hpc_type, connection_manager=self.connection_manager)

            # Fix potential orphans before we start the dashboard
            data = self._recover_orphans(data)

            # Verify scheduler-owned state once before entering the normal
            # dashboard/submission cycle. Telemetry remains on its own cadence.
            resume_data = self._catch_up_existing_state(data)
            data = resume_data if resume_data is not None else data
            scheduler_verified = resume_data is not None
            first_cycle = True

            # ------------------------------------------------------------------
            # 2) Enter UI loop
            # ------------------------------------------------------------------
            if self.debug:
                if self.overview_printer is None:
                    from slurminator.orchestrator_ui import print_overview
                else:
                    print_overview = self.overview_printer

                while True:
                    if not first_cycle and not self.high_volume_mode:
                        data = self._load_yaml()
                    exps = data["experiments"]

                    self._maintain_cycle(data, scheduler_already_verified=scheduler_verified and first_cycle)
                    all_done = self._all_done(exps)
                    if self.high_volume_mode:
                        if all_done:
                            self._save_yaml(data)
                        elif self._state_checkpoint_due():
                            self._checkpoint_and_repoll(data)
                    self._publish_dashboard_snapshot(exps)

                    print_overview(exps)

                    if all_done:
                        logger.info("All experiments terminal => exiting orchestrator.")
                        break

                    first_cycle = False
                    time.sleep(self.poll_interval)
            else:
                DashboardCls = self._resolve_dashboard_cls()
                effective_dashboard_ui = self._effective_dashboard_ui(DashboardCls)
                dashboard_kwargs = {"n_recent": 0, "ui_version": effective_dashboard_ui}
                if effective_dashboard_ui == "v4":
                    dashboard_kwargs["sparkline_thresholds"] = getattr(self.dashboard_settings, "sparkline", None)
                    dashboard_kwargs["table_sort"] = getattr(self.dashboard_settings, "table_sort", None)
                dash = DashboardCls(**dashboard_kwargs)
                initial_exps = self._publish_current_dashboard_snapshot(data)

                with dash.mount(self) as live:
                    live.update(dash.render(initial_exps))
                    while True:
                        if not first_cycle and not self.high_volume_mode:
                            data = self._load_yaml()
                        exps = data["experiments"]

                        self._maintain_cycle(
                            data,
                            scheduler_already_verified=scheduler_verified and first_cycle,
                            on_refresh=lambda current_exps: self._render_dashboard_snapshot(
                                dash, current_exps, live=live
                            ),
                            should_stop=lambda: self._dashboard_requested_exit(dash),
                        )

                        exit_requested = self._dashboard_requested_exit(dash)
                        all_done = self._all_done(exps)
                        if self.high_volume_mode:
                            if exit_requested or all_done:
                                self._checkpoint_and_repoll(data, repoll=not all_done)
                            elif self._state_checkpoint_due():
                                self._checkpoint_and_repoll(data)

                        self._render_dashboard_snapshot(dash, exps, live=live)

                        if exit_requested:
                            logger.info("Dashboard requested orchestrator exit.")
                            break

                        if all_done:
                            logger.info("All experiments terminal => exiting orchestrator.")
                            # Ensure display metrics are populated for final render
                            self._render_dashboard_snapshot(dash, exps, live=live)
                            break

                        first_cycle = False
                        if self._sleep_until_next_poll(dash):
                            if self.high_volume_mode:
                                self._checkpoint_and_repoll(data)
                            logger.info("Dashboard requested orchestrator exit.")
                            break

        except KeyboardInterrupt:
            logger.info("User interrupted HPC Orchestrator.")
            if self.high_volume_mode and data is not None:
                self._checkpoint_and_repoll(data)
        except Exception as e:
            logger.error(f"Orchestrator error: {e}", exc_info=True)
            raise
        finally:
            logger.info("Closing HPC connections.")
            self.connection_manager.close_all()

        # Keep this plain and last so humans and agents can copy the YAML path.
        outcome = "COMPLETE" if all_done else "STOPPED"
        print(f"EXPERIMENT {outcome}: {self.experiment_file}", flush=True)

    def _configure_volume_mode(self, data: Mapping[str, Any]) -> None:
        """Use bounded in-memory maintenance for large ledgers or low-compute mode."""
        experiments = data.get("experiments", [])
        experiment_count = len(experiments) if isinstance(experiments, list) else 0
        self.high_volume_threshold = int(getattr(self, "high_volume_threshold", DEFAULT_HIGH_VOLUME_THRESHOLD))
        self.state_checkpoint_interval_seconds = float(
            getattr(self, "state_checkpoint_interval_seconds", DEFAULT_STATE_CHECKPOINT_INTERVAL_SECONDS)
        )
        self.telemetry_interval_seconds = float(
            getattr(self, "telemetry_interval_seconds", DEFAULT_TELEMETRY_INTERVAL_SECONDS)
        )
        self.telemetry_batch_size = int(getattr(self, "telemetry_batch_size", DEFAULT_TELEMETRY_BATCH_SIZE))
        self.multi_experiment = experiment_count > 1
        self.high_volume_mode = getattr(self, "low_compute", False) or experiment_count >= self.high_volume_threshold
        now = time.monotonic()
        self._last_state_checkpoint_at = now
        self._last_telemetry_poll_at = None
        self._telemetry_cursor = 0
        if self.high_volume_mode:
            histories_deferred = self._discard_cached_histories(experiments)
            logger.info(
                "High-volume maintenance enabled for %d experiments: scheduler state is journaled every cycle, "
                "status-only telemetry is bounded to %d rows every %.0fs, and the YAML ledger checkpoints every %.0fs.",
                experiment_count,
                self.telemetry_batch_size,
                self.telemetry_interval_seconds,
                self.state_checkpoint_interval_seconds,
            )
            if histories_deferred:
                logger.info(
                    "Deferred cached metric history for %d experiment(s); detail and plot screens reload it on demand.",
                    histories_deferred,
                )

    @staticmethod
    def _discard_cached_histories(experiments: list[dict[str, Any]]) -> int:
        """Drop reconstructable history payloads from a high-volume in-memory ledger."""
        changed_rows = 0
        for exp in experiments:
            if not isinstance(exp, dict):
                continue
            had_history_state = any(field in exp for field in HIGH_VOLUME_TRANSIENT_HISTORY_FIELDS)
            for field in HIGH_VOLUME_TRANSIENT_HISTORY_FIELDS:
                exp.pop(field, None)
            changed_rows += int(had_history_state)
        return changed_rows

    def _maintain_cycle(
        self,
        data: dict[str, Any],
        *,
        scheduler_already_verified: bool = False,
        on_refresh: Callable[[list[dict[str, Any]]], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> None:
        """Run one scheduler/submission cycle using the selected volume policy."""
        exps = data["experiments"]
        processed_commands = self._process_command_queue(exps)
        if processed_commands:
            data["experiments"] = exps
            self._save_yaml(data)
            if on_refresh is not None and not self.high_volume_mode:
                on_refresh(exps)

        stop_requested = should_stop is not None and should_stop()
        if not stop_requested:
            if self.high_volume_mode:
                if not scheduler_already_verified:
                    result = self._refresh_scheduler_statuses(exps)
                    if self._scheduler_retry_became_ready(result, exps):
                        self._save_yaml(data)
                if self._telemetry_poll_due():
                    self._update_telemetry(exps, include_history=False, max_experiments=self.telemetry_batch_size)
                    self._update_queue_estimates(exps)
                    self._last_telemetry_poll_at = time.monotonic()
            elif scheduler_already_verified:
                self._update_telemetry(exps, include_history=True)
                self._update_queue_estimates(exps)
            else:
                self._update_statuses(exps)
                self._update_queue_estimates(exps)

            if on_refresh is not None and not self.high_volume_mode:
                on_refresh(exps)

        stop_requested = stop_requested or (should_stop is not None and should_stop())
        if not stop_requested:
            concurrency_used = self._fill_available_capacity(
                exps,
                data,
                # Large snapshots are intentionally rendered once per outer
                # cycle, not between every submission microburst.
                on_refresh=None if self.high_volume_mode else on_refresh,
                should_stop=should_stop,
            )
            if not self.submissions_paused:
                self._maybe_reassign_experiments(exps, concurrency_used, data)

        data["experiments"] = exps
        if not self.high_volume_mode:
            self._save_yaml(data)

    def _telemetry_poll_due(self, *, now: float | None = None) -> bool:
        """Return whether the high-volume telemetry lane should run."""
        current = time.monotonic() if now is None else now
        last_poll = self._last_telemetry_poll_at
        return last_poll is None or current - last_poll >= self.telemetry_interval_seconds

    def _state_checkpoint_due(self, *, now: float | None = None) -> bool:
        """Return whether the high-volume in-memory ledger needs compaction."""
        current = time.monotonic() if now is None else now
        last_checkpoint = self._last_state_checkpoint_at
        return last_checkpoint is None or current - last_checkpoint >= self.state_checkpoint_interval_seconds

    def _checkpoint_and_repoll(self, data: dict[str, Any], *, repoll: bool = True) -> None:
        """Checkpoint the ledger; low-compute mode waits for the next scheduled poll."""
        self._save_yaml(data)
        if not repoll or getattr(self, "low_compute", False):
            return
        exps = data["experiments"]
        if not any(
            exp.get("job_id") and exp.get("status") in {ExperimentStatus.QUEUED, ExperimentStatus.RUNNING}
            for exp in exps
        ):
            return
        result = self._refresh_scheduler_statuses(exps)
        if self._scheduler_retry_became_ready(result, exps):
            self._save_yaml(data)

    def _preflight_validate_experiments(self, data: Mapping[str, Any]) -> int:
        """Validate a fresh ledger, or skip repeated validation for a clear resume."""
        experiments = data.get("experiments", [])
        if any(isinstance(exp, Mapping) and bool(exp.get("job_id")) for exp in experiments):
            logger.info(
                "Existing job IDs detected; skipping repeated experiment config preflight for this resume. "
                "Pending rows are assumed unchanged since the first successful validation."
            )
            return 0

        orchestrator_logger = logging.getLogger("slurminator")
        original_level = orchestrator_logger.level
        orchestrator_logger.setLevel(logging.WARNING)
        validated_experiments = 0

        try:
            for exp in experiments:
                dataset = exp.get("dataset_name")
                if not dataset:
                    continue

                sweep_params = exp.get("sweep_params")
                if not sweep_params:
                    if self.plugin.validate_experiment(exp, {}):
                        validated_experiments += 1
                    continue

                try:
                    overrides = self.parse_overrides(sweep_params)
                except Exception as exc:
                    logger.error(
                        "Failed to parse sweep_params for experiment '%s': %s",
                        exp.get("experiment_id", "<unknown>"),
                        exc,
                    )
                    raise

                try:
                    if self.plugin.validate_experiment(exp, overrides):
                        validated_experiments += 1
                except Exception as exc:
                    logger.error(
                        "Config sanity-check failed for experiment '%s' (dataset=%s): %s",
                        exp.get("experiment_id", "<unknown>"),
                        str(dataset),
                        exc,
                    )
                    raise
        finally:
            orchestrator_logger.setLevel(original_level)

        logger.info(
            "Experiment YAML sanity-check passed – %d experiments with overrides validated.", validated_experiments
        )
        return validated_experiments

    def _resolve_dashboard_cls(self):
        """Resolve the concrete dashboard class for the requested UI version."""
        if str(self.dashboard_ui).strip().lower() == "v4":
            try:
                return importlib.import_module("slurminator.dashboard_v4.app").TextualDashboardApp
            except ModuleNotFoundError as exc:
                if exc.name not in {"textual", "plotext"}:
                    raise
                logger.warning(
                    "Dashboard v4 dependency %r is not installed; falling back to dashboard v3. "
                    "Reinstall Slurminator to use v4: python -m pip install -e .",
                    exc.name,
                )
        if self.dashboard_cls is not None:
            return self.dashboard_cls
        from slurminator.ui_dashboard import TerminalDashboard

        return TerminalDashboard

    def _effective_dashboard_ui(self, dashboard_cls: Type[Any]) -> str:
        """Return the UI version string accepted by the resolved dashboard class."""
        requested = str(self.dashboard_ui).strip().lower() or "v3"
        if requested == "v4" and getattr(dashboard_cls, "__module__", "") != "slurminator.dashboard_v4.app":
            return "v3"
        return requested

    def _publish_dashboard_snapshot(self, exps: list[dict[str, Any]]) -> None:
        """Publish a copy of the latest ledger for threaded dashboard readers."""
        self._dashboard_snapshot = copy.deepcopy(exps)
        self._dashboard_snapshot_version += 1

    def _publish_current_dashboard_snapshot(self, data: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        """Publish the current ledger before expensive poll-cycle work starts."""
        if data is None:
            data = self._load_yaml()
        exps = data.get("experiments", [])
        populate_display_metrics(exps)
        self._publish_dashboard_snapshot(exps)
        return exps

    def _catch_up_existing_state(self, data: dict[str, Any] | None = None) -> dict[str, Any] | None:
        """Verify active scheduler rows once without running slow telemetry work."""
        if data is None:
            data = self._load_yaml()
        exps = data.get("experiments", [])
        recovered_timeout_retry = self._recover_timeout_retry_decisions(exps)
        if recovered_timeout_retry:
            data["experiments"] = exps
            self._save_yaml(data)
        active_before = sum(
            1
            for exp in exps
            if exp.get("job_id") and exp.get("status") in {ExperimentStatus.QUEUED, ExperimentStatus.RUNNING}
        )
        if active_before == 0:
            if recovered_timeout_retry:
                self._publish_dashboard_snapshot(exps)
                return data
            return None

        logger.info("Verifying %d active experiment(s) with Slurm before entering the submission loop.", active_before)
        result = self._refresh_scheduler_statuses(exps)
        if self._scheduler_retry_became_ready(result, exps):
            # A TIMEOUT retry decision is not scheduler-owned and therefore is
            # checkpointed before a replacement job can be submitted.
            self._save_yaml(data)
        self._publish_dashboard_snapshot(exps)

        active_after = sum(
            1
            for exp in exps
            if exp.get("job_id") and exp.get("status") in {ExperimentStatus.QUEUED, ExperimentStatus.RUNNING}
        )
        if result.succeeded:
            logger.info(
                "Scheduler verification complete: %d transition(s) journaled; %d active experiment(s) remain.",
                len(result.transitions),
                active_after,
            )
        else:
            unresolved_jobs = sum(len(poll.unresolved_job_ids) for poll in result.hpc_polls)
            logger.warning(
                "Scheduler verification incomplete: %d poll failure(s), %d unresolved job(s), %d skipped row(s); "
                "%d active experiment(s) remain and will be retried next cycle.",
                len(result.failures),
                unresolved_jobs,
                len(result.skipped),
                active_after,
            )
        return data

    def _recover_timeout_retry_decisions(self, exps: list[dict[str, Any]]) -> bool:
        """Reapply an interrupted TIMEOUT retry decision after journal replay."""
        if not self.retry_timeout_with_estimated_time:
            return False
        changed = False
        for exp in exps:
            if exp.get("status") != ExperimentStatus.TIMEOUT:
                continue
            hpc_type = self._coerce_hpc_type(exp.get("hpc_assignment"))
            if hpc_type is None:
                logger.warning(
                    "Cannot recover timeout retry for %s without a valid HPC assignment.", exp.get("experiment_id")
                )
                continue
            before = (exp.get("status"), exp.get("timeout_retry_count"), exp.get("time_hours_override"))
            self._apply_timeout_policy(exp, hpc_type, "resume after durable scheduler TIMEOUT")
            after = (exp.get("status"), exp.get("timeout_retry_count"), exp.get("time_hours_override"))
            changed = changed or after != before
        return changed

    def _render_dashboard_snapshot(self, dashboard: Any, exps: list[dict[str, Any]], *, live: Any) -> None:
        """Publish and render a dashboard snapshot after visible state changes."""
        populate_display_metrics(exps)
        self._publish_dashboard_snapshot(exps)
        live.update(dashboard.render(exps))

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
    def _recover_orphans(self, data: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        If experiment is QUEUED/RUNNING but missing 'job_id', revert => PENDING
        """
        if data is None:
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
        return data

    # -------------------------------------------------------------------------
    # HPC Status Polling
    # -------------------------------------------------------------------------
    def _update_statuses(self, exps: list[dict[str, Any]]) -> SchedulerStatusUpdateResult:
        """
        Poll scheduler state, then ingest legacy full telemetry/history eagerly.
        """
        result = self._refresh_scheduler_statuses(exps)
        self._update_telemetry(exps, include_history=True)
        return result

    def _refresh_scheduler_statuses(self, exps: list[dict[str, Any]]) -> SchedulerStatusUpdateResult:
        """Refresh and durably journal scheduler-owned states only."""
        result = update_scheduler_statuses(
            exps, connection_manager=self.connection_manager, concurrency_limits=self.concurrency_limits
        )
        if result.transitions:
            self.state_store.record_scheduler_transitions(result.transitions)
            rows_by_id = {str(exp.get("experiment_id")): exp for exp in exps}
            for transition in result.transitions:
                exp = rows_by_id[transition.experiment_id]
                if is_terminal_status(transition.new_status):
                    exp["terminal_telemetry"] = {"job_id": transition.job_id, "attempts": 0, "finalized": False}
                if transition.new_status == ExperimentStatus.TIMEOUT:
                    self._apply_timeout_policy(exp, transition.hpc_type, "scheduler reported TIMEOUT")
        return result

    @staticmethod
    def _scheduler_retry_became_ready(result: SchedulerStatusUpdateResult, exps: list[dict[str, Any]]) -> bool:
        """Return whether a terminal scheduler transition enabled a retry."""
        transitioned_ids = {transition.experiment_id for transition in result.transitions}
        return any(
            str(exp.get("experiment_id")) in transitioned_ids
            and exp.get("status") in {ExperimentStatus.PENDING, ExperimentStatus.PARTIAL}
            for exp in exps
        )

    def _update_telemetry(
        self, exps: list[dict[str, Any]], *, include_history: bool, max_experiments: int | None = None
    ) -> int:
        """Ingest a fair bounded set of live and explicitly pending final status files."""
        candidates = [
            exp
            for exp in exps
            if exp.get("status") == ExperimentStatus.RUNNING or self._terminal_telemetry_pending(exp)
        ]
        if not candidates:
            self._telemetry_cursor = 0
            return 0

        candidate_count = len(candidates)
        limit = candidate_count if max_experiments is None else min(max(int(max_experiments), 0), candidate_count)
        if limit == 0:
            return 0
        start = self._telemetry_cursor % candidate_count
        selected = [candidates[(start + offset) % candidate_count] for offset in range(limit)]
        self._telemetry_cursor = (start + limit) % candidate_count

        for exp in selected:
            try:
                self._update_running_experiment_info(exp, include_history=include_history)
            except Exception as exc:
                logger.warning("Telemetry update failed for %s: %s", exp.get("experiment_id"), exc)
        return len(selected)

    @staticmethod
    def _terminal_telemetry_pending(exp: Mapping[str, Any]) -> bool:
        """Return whether a terminal row has an explicit job-scoped final read pending."""
        if not is_terminal_status(exp.get("status")):
            return False
        job_id = exp.get("job_id")
        state = exp.get("terminal_telemetry")
        return (
            bool(job_id)
            and isinstance(state, Mapping)
            and str(state.get("job_id")) == str(job_id)
            and state.get("finalized") is not True
        )

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
                    logger.debug("%s queued/running on zero-limit HPC %s", e["experiment_id"], h)
        return usage

    # -------------------------------------------------------------------------
    # Submission
    # -------------------------------------------------------------------------
    def _fill_available_capacity(
        self,
        exps: list[dict[str, Any]],
        data: dict[str, Any],
        *,
        on_refresh: Callable[[list[dict[str, Any]]], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> dict[HPCType, int]:
        """Fill available slots while keeping scheduler state reasonably fresh.

        Submission receipts make accepted job IDs durable immediately. This
        method therefore checkpoints the large YAML ledger only after a group
        of small submission bursts, while polling Slurm and processing operator
        commands between bursts. Full status-file ingestion remains in the
        outer maintenance loop. Low-compute fills reuse cached concurrency and
        yield between batches instead of issuing extra scheduler queries. Their
        count/time bounds keep submission work finite even for large ledgers.
        """
        total_submitted = 0
        concurrency_used = self._count_concurrency(exps)
        low_compute = getattr(self, "low_compute", False)
        submission_limit = self.submission_checkpoint_size
        fill_deadline = None
        if low_compute:
            submission_limit = min(submission_limit, LOW_COMPUTE_SUBMISSION_FILL_SIZE)
            fill_deadline = time.monotonic() + LOW_COMPUTE_SUBMISSION_FILL_SECONDS
        # Cached low-compute state cannot free capacity between batches. Walk
        # the ledger once, avoiding repeated scans and retries of failed rows.
        submission_candidates = iter(exps) if low_compute else exps

        def should_yield() -> bool:
            return (fill_deadline is not None and time.monotonic() >= fill_deadline) or (
                should_stop is not None and should_stop()
            )

        while total_submitted < submission_limit:
            if self.submissions_paused or should_yield():
                break

            remaining_before_checkpoint = submission_limit - total_submitted
            submitted_count = self._submit_pending_batch(
                submission_candidates,
                concurrency_used,
                data,
                max_submissions=remaining_before_checkpoint,
                should_stop=should_yield,
            )
            if submitted_count == 0:
                break
            total_submitted += submitted_count

            if total_submitted >= submission_limit:
                break
            if low_compute and not any(
                concurrency_used.get(hpc, 0) < limit for hpc, limit in self.concurrency_limits.items()
            ):
                break

            if fill_deadline is not None:
                # Keep the existing small bursts, with a cooldown proportional
                # to accepted jobs. This is burst pacing, not an sbatch RPC rate
                # limit: failed attempts do not count as accepted submissions.
                resume_at = min(
                    time.monotonic() + submitted_count * LOW_COMPUTE_SUBMISSION_PAUSE_SECONDS_PER_JOB, fill_deadline
                )
                while time.monotonic() < resume_at:
                    if self.submissions_paused or (should_stop is not None and should_stop()):
                        break
                    time.sleep(min(0.2, max(0.0, resume_at - time.monotonic())))

            processed_commands = self._process_command_queue(exps)
            if processed_commands:
                data["experiments"] = exps
                self._save_yaml(data)

            if self.submissions_paused or (should_stop is not None and should_stop()):
                if on_refresh is not None:
                    on_refresh(exps)
                break

            if low_compute:
                # Accepted receipts already account for new jobs. Jobs that
                # finish before the next scheduled poll only make this cached
                # count conservative. Recount after possible operator commands.
                if processed_commands:
                    concurrency_used = self._count_concurrency(exps)
                continue

            active_before_refresh = {
                str(exp.get("experiment_id"))
                for exp in exps
                if exp.get("status") in {ExperimentStatus.QUEUED, ExperimentStatus.RUNNING}
            }
            self._refresh_scheduler_statuses(exps)
            retry_became_ready = any(
                str(exp.get("experiment_id")) in active_before_refresh
                and exp.get("status") in {ExperimentStatus.PENDING, ExperimentStatus.PARTIAL}
                for exp in exps
            )
            if retry_became_ready:
                # Checkpoint the terminal attempt before a retry replaces its
                # job ID. A subsequent receipt can then prove which prior job
                # it is allowed to supersede during crash recovery.
                data["experiments"] = exps
                self._save_yaml(data)
            concurrency_used = self._count_concurrency(exps)
            if on_refresh is not None:
                on_refresh(exps)

        if total_submitted:
            logger.info("Submission fill phase accepted %d job(s); submission receipts are durable.", total_submitted)
        return concurrency_used

    def _submit_pending_batch(
        self,
        exps: Iterable[dict[str, Any]],
        concurrency_used: dict[HPCType, int],
        data: dict[str, Any],
        *,
        max_submissions: int | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> int:
        """Submit at most one bounded batch and return its successful submission count.

        The count bound applies to successful submissions. The elapsed-time
        bound is checked after each synchronous submission attempt, allowing
        that attempt to finish before returning control to the outer polling
        loop.
        """
        submitted_count = 0
        submission_limit = self.submission_batch_size
        if max_submissions is not None:
            submission_limit = min(submission_limit, max(int(max_submissions), 0))
        if submission_limit == 0:
            return 0

        batch_started_at = time.monotonic()
        for exp in exps:
            if self.submissions_paused or (should_stop is not None and should_stop()):
                break
            if self._maybe_submit(exp, concurrency_used, data):
                submitted_count += 1
            if (
                submitted_count >= submission_limit
                or time.monotonic() - batch_started_at >= self.submission_batch_seconds
            ):
                break

        return submitted_count

    def _maybe_submit(self, exp: dict, concurrency_used: Dict[HPCType, int], data: dict) -> bool:
        """
        If PENDING or PARTIAL => pick HPC if not assigned, check concurrency, do sbatch universal_job.sh ...
        """
        return maybe_submit(
            exp,
            concurrency_used,
            data,
            concurrency_limits=self.concurrency_limits,
            hpc_configs=HPC_CONFIGS,
            submit_experiment=self._submit_experiment_universal,
            replace_exp_in_list=self._replace_exp_in_list,
            save_yaml=self._save_yaml,
            record_submission=self.state_store.record_submission,
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
            prepared_repositories=self._prepared_submission_repositories,
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
    def _update_running_experiment_info(
        self, exp: dict[str, Any], *, include_history: bool = True
    ) -> Optional[dict[str, Any]]:
        context = StatusIngestContext(
            connection_manager=self.connection_manager,
            hpc_configs=HPC_CONFIGS,
            load_yaml=self._load_yaml,
            save_yaml=self._save_yaml,
            projection_options=self.projection_options,
            persist_immediately=False,
            include_history=include_history,
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
        self,
        exp: dict[str, Any],
        *,
        lines: int = 500,
        offsets: Mapping[str, int] | None = None,
        source: LogSource = "combined",
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
        return read_log_tail_incremental(
            exp, str(job_id), hpc_type, context, lines=lines, offsets=offsets, source=source
        )

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
        all_terminal = all(e["status"] in terminal for e in exps)
        if getattr(self, "low_compute", False):
            # Bounded telemetry may need more than one cycle to collect the
            # last metrics. Each row already has a finite final-read budget.
            return all_terminal and not any(self._terminal_telemetry_pending(exp) for exp in exps)
        return all_terminal

    # -------------------------------------------------------------------------
    # read/save YAML
    # -------------------------------------------------------------------------
    def _load_yaml(self) -> dict:
        """Load experiment YAML and ensure a dict with an 'experiments' list."""
        self.state_store.concurrency_limits = self.concurrency_limits
        data = self.state_store.load()
        metadata = data.get("metadata")
        self.experiment_info = ExperimentInfo.from_metadata(metadata if isinstance(metadata, Mapping) else {})
        return data

    def _save_yaml(self, data: dict[str, Any]) -> None:
        checkpoint_data = data
        if getattr(self, "high_volume_mode", False):
            checkpoint_data = dict(data)
            checkpoint_data["experiments"] = [
                {key: value for key, value in exp.items() if key not in HIGH_VOLUME_TRANSIENT_HISTORY_FIELDS}
                for exp in data.get("experiments", [])
            ]
        self.state_store.save(checkpoint_data)
        if getattr(self, "high_volume_mode", False):
            self._last_state_checkpoint_at = time.monotonic()

    def _replace_exp_in_list(self, exps: List[dict], new_e: dict) -> List[dict]:
        return replace_exp_in_list(exps, new_e)

    # -------------------------------------------------------------------------
    # local HPC detection
    # -------------------------------------------------------------------------
    def is_local_hpc(self, hpc_type: HPCType) -> bool:
        """Return True if running on the given HPC type."""
        return self.is_local_hpc_fn(hpc_type)

    def _ensure_remote_dirs(self, hpc_type: HPCType, *, data: Mapping[str, Any] | None = None) -> None:
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
            if data is None:
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

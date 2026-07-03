"""Terminal dashboard for Slurminator orchestrators based on the Rich library.

This small helper renders a live, in-place overview of experiment progress so that
running the orchestrator no longer floods the terminal with logger output.
The dashboard shows a progress bar and tables of the most recent experiments
in RUNNING / COMPLETED / FAILED states.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from datetime import date, datetime as dt
from typing import Any, List, Optional, Tuple

from rich.console import Console  # type: ignore
from rich.columns import Columns  # type: ignore
from rich.layout import Layout  # type: ignore
from rich.live import Live  # type: ignore
from rich.progress import Progress, BarColumn, TextColumn, TaskProgressColumn  # type: ignore
from rich.table import Table  # type: ignore
from rich.text import Text  # type: ignore

from rich import box as _box  # type: ignore

from slurminator.config import HPCType, HPC_CONFIGS
from slurminator.experiments import ExperimentStatus
from slurminator.experiments.yaml_utils import load_yaml
from slurminator.quota import QuotaProvider, QuotaSnapshot, get_quota_provider

# ------------------------------------------------------------------
# UI Layout Configuration - Adjust these values to customize display
# ------------------------------------------------------------------
#
# For small monitors or tight spacing needs:
# - Set COLUMN_SPACING = 0 for maximum space efficiency
# - Set PROGRESS_BAR_WIDTH = 20 to make progress bars narrower
# - Set SHOW_QUEUED_IDS = False to save space in queued table
# - Set METRIC_VALUE_PRECISION = 3 for shorter metric values
#
# Note: Rich library inherits terminal font size - to reduce font size,
# adjust your terminal emulator settings, not these values.

# Refresh rate
REFRESH_RATE = 1.0  # Dashboard refresh rate in Hz (1.0 = 1 second between updates)

# Spacing and layout
COLUMN_SPACING = 0  # Space between table columns (0=tight, 1=normal, 2=loose)
PROGRESS_BAR_WIDTH = 30  # Width of progress bars
V3_PROGRESS_BAR_WIDTH = 20  # Width of top-row progress bars in the denser v3 dashboard
SUMMARY_HEIGHT = 2
BARS_HEIGHT = 2
LOGS_HEIGHT = 5
FOOTER_HEIGHT = 3
RESERVED_LINES = SUMMARY_HEIGHT + BARS_HEIGHT + LOGS_HEIGHT + FOOTER_HEIGHT
ALT_ROW_STYLE = "grey50"

# Table visibility toggles
SHOW_QUEUED_IDS = False  # Show experiment IDs in queued table (takes space)
SHOW_DELTA_TIME = True  # Show time deltas in running/completed tables

# Row limits (0 = auto-calculate based on terminal height)
DEFAULT_N_RECENT = 0  # Number of recent experiments to show per table

# Metric display
METRIC_VALUE_PRECISION = 4  # Decimal places for metric values (e.g. 0.1234)
PROGRESS_PERCENTAGE_WIDTH = 5  # Width for progress percentages (e.g. " 25.0%")

# Timeout risk estimation guards/thresholds (to avoid noisy early extrapolation)
TIMEOUT_RISK_MIN_PROGRESS = 0.20
TIMEOUT_RISK_MIN_RUNTIME_SECONDS = 15 * 60
TIMEOUT_RISK_MEDIUM_RATIO = 0.85
TIMEOUT_RISK_HIGH_RATIO = 1.0

# Table layout ratios - adjust these to change relative widths
# Total ratio = sum of all values
# Increase values to make tables wider, decrease to make narrower
# NOTE: Rich requires integer ratios, not floats!
TABLE_LAYOUT_RATIOS = {
    "queued": 3,  # Narrower (was 4)
    "running": 8,  # Wider (was 6)
    "completed": 8,  # Wider (was 6)
    "failed": 3,  # Narrower (was 4)
}

GPU_QUOTA_POLL_INTERVAL_SECONDS = 300.0

logger = logging.getLogger("slurminator")


class TerminalDashboard:
    """Render a simple in-place dashboard with Rich.

    Parameters
    ----------
    n_recent : int, optional
        How many of the most recent experiments to list in each table, by default 5.
    refresh_per_second : int | float, optional
        How often the dashboard should refresh, by default 1 time per second.
    """

    # Exp state considered "done"
    _DONE_STATES = [
        ExperimentStatus.COMPLETED,
        ExperimentStatus.FAILED,
        ExperimentStatus.CANCELLED,
        ExperimentStatus.TIMEOUT,
        ExperimentStatus.OOM,
        ExperimentStatus.KILLED,
    ]
    _FAILED_STATES = [
        ExperimentStatus.FAILED,
        ExperimentStatus.CANCELLED,
        ExperimentStatus.TIMEOUT,
        ExperimentStatus.OOM,
        ExperimentStatus.KILLED,
    ]

    def __init__(
        self,
        n_recent: int = DEFAULT_N_RECENT,
        refresh_per_second: int | float = REFRESH_RATE,
        is_sweep: bool = False,
        ui_version: str = "v3",
        timeout_risk_settings: object | None = None,
    ):
        self.n_recent = n_recent
        self._initial_n_recent = n_recent
        self.refresh_per_second = refresh_per_second
        self.console = Console()
        self.orchestrator = None  # Set by mount()
        self.is_sweep = is_sweep  # Will be updated via _infer_sweep_dataset
        self.ui_version = self._normalize_ui_version(ui_version)
        timeout_cfg = timeout_risk_settings
        self.timeout_risk_min_progress = TIMEOUT_RISK_MIN_PROGRESS
        self.timeout_risk_min_runtime_seconds = TIMEOUT_RISK_MIN_RUNTIME_SECONDS
        self.timeout_risk_medium_ratio = TIMEOUT_RISK_MEDIUM_RATIO
        self.timeout_risk_high_ratio = TIMEOUT_RISK_HIGH_RATIO
        if timeout_cfg is not None:
            self.timeout_risk_min_progress = max(
                0.0, min(float(getattr(timeout_cfg, "min_progress", self.timeout_risk_min_progress)), 1.0)
            )
            self.timeout_risk_min_runtime_seconds = max(
                1, int(float(getattr(timeout_cfg, "min_runtime_seconds", self.timeout_risk_min_runtime_seconds)))
            )
            self.timeout_risk_medium_ratio = max(
                0.0, float(getattr(timeout_cfg, "medium_ratio", self.timeout_risk_medium_ratio))
            )
            self.timeout_risk_high_ratio = max(
                self.timeout_risk_medium_ratio, float(getattr(timeout_cfg, "high_ratio", self.timeout_risk_high_ratio))
            )
        # Cache expensive HPC quota probes to keep UI refreshes lightweight.
        # Key: HPCType, Value: (timestamp, provider-snapshot-or-None)
        self._quota_cache: dict[HPCType, tuple[float, QuotaSnapshot | None]] = {}

    # ------------------------------------------------------------------
    # Public API: mount and render
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_ui_version(ui_version: str) -> str:
        if not ui_version:
            return "v3"
        normalized = str(ui_version).strip().lower()
        if normalized not in {"v2", "v3"}:
            raise ValueError(f"Unsupported dashboard UI version '{ui_version}'. Expected 'v2' or 'v3'.")
        return normalized

    def mount(self, orchestrator: Any) -> Live:
        """Context manager for rendering updates.

        Usage::

            dash = TerminalDashboard()
            with dash.mount(orc) as live:
                ...  # inside orchestrator loop call ``live.update(dash.render(exps))``
        """
        self.orchestrator = orchestrator
        # Use the dashboard's own refresh rate, independent of orchestrator's poll interval
        return Live(self.render([]), console=self.console, screen=False, refresh_per_second=self.refresh_per_second)

    def render(self, exps: List[dict]):  # noqa: D401
        """Return a Rich renderable representing the current dashboard state."""
        # Dynamically adjust number of rows based on terminal height if requested
        if self._initial_n_recent <= 0:
            # Reserve fixed lines: summary(3) + bars row(3) + logs(6) + footer(1)
            available_rows = max(self.console.size.height - RESERVED_LINES, 1)
            # Use almost all remaining rows for table entries (minus header row per table)
            self.n_recent = max(available_rows - 2, 1)
        return self._render(exps)

    # ------------------------------------------------------------------
    # Internal: build Rich layout
    # ------------------------------------------------------------------
    def _render(self, exps: List[dict]):
        if self.ui_version == "v3":
            return self._render_v3(exps)
        return self._render_v2(exps)

    def _render_v2(self, exps: List[dict]):
        # --- Statistics --------------------------------------------------
        counts: Counter[ExperimentStatus] = Counter(e["status"] for e in exps)
        total = len(exps)
        done = sum(counts[s] for s in self._DONE_STATES)
        now = time.time()
        timeout_risk_counts = self._timeout_risk_counts(exps, now=now)

        # Compute number of currently *running* jobs and the global concurrency limit
        # We intentionally exclude QUEUED jobs here because the goal of this bar is to
        # visualise real utilisation of the available slots (i.e. jobs that are
        # actually executing right now).
        running_jobs = counts[ExperimentStatus.RUNNING]
        limit_total = 0
        if self.orchestrator is not None and isinstance(getattr(self.orchestrator, "concurrency_limits", None), dict):
            limit_total = sum(self.orchestrator.concurrency_limits.values())

        # --- Progress bars -------------------------------------------------
        # «Completed» progress (existing behaviour)
        progress_completed = Progress(
            TextColumn("[progress.description]Completed"),
            BarColumn(bar_width=PROGRESS_BAR_WIDTH),
            TaskProgressColumn(show_speed=False),
            expand=False,
        )
        progress_completed.add_task("", total=total or 1, completed=done)

        # «Running jobs» progress (renamed from "Active")
        running_label = f"Running {running_jobs}/{limit_total}" if limit_total else f"Running {running_jobs}"
        progress_running = Progress(
            TextColumn(f"[progress.description]{running_label}"),
            BarColumn(bar_width=PROGRESS_BAR_WIDTH, complete_style="yellow"),
            TaskProgressColumn(show_speed=False),
            expand=False,
        )
        progress_running.add_task(
            "", total=limit_total or 1, completed=min(running_jobs, limit_total) if limit_total else running_jobs
        )

        # --- Recent experiment tables ------------------------------------
        layout = Layout()
        next_eta = self._next_allocation_eta(exps)
        project_label = self._infer_project_label(exps)
        layout.split_column(
            Layout(
                self._summary_table(counts, next_eta, timeout_risk_counts=timeout_risk_counts),
                name="summary",
                size=SUMMARY_HEIGHT,
            ),
            Layout(name="bars", size=BARS_HEIGHT),
            Layout(name="main"),
            Layout(name="logs", size=LOGS_HEIGHT),
            Layout(name="footer", size=FOOTER_HEIGHT),
        )

        # Place Completed and Active bars side-by-side to save vertical space
        layout["bars"].update(
            Columns(
                [progress_completed, Text("|", style="dim"), progress_running],
                expand=False,
                equal=False,
                padding=(0, 1),
            )
        )

        # Use custom ratios for table layout
        layout["main"].split_row(
            Layout(self._queued_table(exps), name="queued", ratio=TABLE_LAYOUT_RATIOS["queued"]),
            Layout(
                self._recent_table(exps, ExperimentStatus.RUNNING), name="running", ratio=TABLE_LAYOUT_RATIOS["running"]
            ),
            Layout(
                self._recent_table(exps, ExperimentStatus.COMPLETED, include_delta=False),
                name="done",
                ratio=TABLE_LAYOUT_RATIOS["completed"],
            ),
            Layout(self._failed_table(exps), name="failed", ratio=TABLE_LAYOUT_RATIOS["failed"]),
        )

        # Logs box --------------------------------------------------------
        layout["logs"].update(self._logs_table(exps))

        # Footer ----------------------------------------------------------
        remaining = total - done
        sweep_name = self._infer_sweep_dataset(exps) or "-"
        sweep_url = self._infer_sweep_url(exps)
        if sweep_url:
            sweep_part = f"{sweep_name} ( {sweep_url} )"
        else:
            sweep_part = sweep_name

        footer_bits = [
            f"[bold green]{counts[ExperimentStatus.COMPLETED]} / {total} completed[/]",
            f"[bold yellow]{remaining} left[/]",
            f"Sweep: {sweep_part}",
        ]
        if project_label:
            footer_bits.append(project_label)
        footer_bits.append(f"Updated: {dt.now().strftime('%H:%M:%S')}")
        footer_line = " • ".join(footer_bits)
        second_line_parts = []
        hpc_label = self._infer_hpc_label(exps)
        if hpc_label:
            second_line_parts.append(hpc_label)
        experiment_label = self._infer_experiment_label()
        if experiment_label:
            second_line_parts.append(experiment_label)
        slurm_request_label = self._slurm_request_label(exps)
        if slurm_request_label:
            second_line_parts.append(slurm_request_label)
        quota_label = self._quota_label(exps)
        oom_legend = "* = recovered from OOM" if self._has_oom_recovery(exps) else None
        if oom_legend:
            second_line_parts.append(oom_legend)
        if second_line_parts and quota_label:
            footer_renderable = f"{footer_line}\n{' • '.join(second_line_parts)}\n{quota_label}"
        elif second_line_parts:
            footer_renderable = f"{footer_line}\n{' • '.join(second_line_parts)}"
        elif quota_label:
            footer_renderable = f"{footer_line}\n{quota_label}"
        else:
            footer_renderable = footer_line
        layout["footer"].update(footer_renderable)
        return layout

    def _render_v3(self, exps: List[dict]):
        counts: Counter[ExperimentStatus] = Counter(e["status"] for e in exps)
        total = len(exps)
        done = sum(counts[s] for s in self._DONE_STATES)
        now = time.time()
        timeout_risk_counts = self._timeout_risk_counts(exps, now=now)

        running_jobs = counts[ExperimentStatus.RUNNING]
        limit_total = 0
        if self.orchestrator is not None and isinstance(getattr(self.orchestrator, "concurrency_limits", None), dict):
            limit_total = sum(self.orchestrator.concurrency_limits.values())

        progress_completed = Progress(
            TextColumn("[progress.description]Completed"),
            BarColumn(bar_width=V3_PROGRESS_BAR_WIDTH),
            TaskProgressColumn(show_speed=False),
            expand=False,
        )
        progress_completed.add_task("", total=total or 1, completed=done)

        progress_done, progress_total, _progress_fraction = self._overall_run_progress(exps)
        progress_overall = Progress(
            TextColumn("[progress.description]Progress"),
            BarColumn(bar_width=V3_PROGRESS_BAR_WIDTH, complete_style="cyan"),
            TaskProgressColumn(show_speed=False),
            expand=False,
        )
        progress_overall.add_task("", total=progress_total or 1, completed=progress_done)

        running_label = f"Running {running_jobs}/{limit_total}" if limit_total else f"Running {running_jobs}"
        progress_running = Progress(
            TextColumn(f"[progress.description]{running_label}"),
            BarColumn(bar_width=V3_PROGRESS_BAR_WIDTH, complete_style="yellow"),
            TaskProgressColumn(show_speed=False),
            expand=False,
        )
        progress_running.add_task(
            "", total=limit_total or 1, completed=min(running_jobs, limit_total) if limit_total else running_jobs
        )

        layout = Layout()
        next_eta = self._next_allocation_eta(exps)
        project_label = self._infer_project_label(exps)
        layout.split_column(
            Layout(
                self._summary_table(counts, next_eta, timeout_risk_counts=timeout_risk_counts),
                name="summary",
                size=SUMMARY_HEIGHT,
            ),
            Layout(name="bars", size=BARS_HEIGHT),
            Layout(name="main"),
            Layout(name="logs", size=LOGS_HEIGHT),
            Layout(name="footer", size=FOOTER_HEIGHT),
        )

        layout["bars"].update(
            Columns(
                [
                    progress_completed,
                    Text("|", style="dim"),
                    progress_overall,
                    Text("|", style="dim"),
                    progress_running,
                ],
                expand=False,
                equal=False,
                padding=(0, 1),
            )
        )

        layout["main"].update(self._all_runs_table(exps))
        layout["logs"].update(self._logs_table(exps))

        remaining = total - done
        sweep_name = self._infer_sweep_dataset(exps) or "-"
        sweep_url = self._infer_sweep_url(exps)
        if sweep_url:
            sweep_part = f"{sweep_name} ( {sweep_url} )"
        else:
            sweep_part = sweep_name

        footer_bits = [
            f"[bold green]{counts[ExperimentStatus.COMPLETED]} / {total} completed[/]",
            f"[bold yellow]{remaining} left[/]",
            f"Sweep: {sweep_part}",
        ]
        if project_label:
            footer_bits.append(project_label)
        footer_bits.append(f"Updated: {dt.now().strftime('%H:%M:%S')}")
        footer_line = " • ".join(footer_bits)
        second_line_parts = []
        hpc_label = self._infer_hpc_label(exps)
        if hpc_label:
            second_line_parts.append(hpc_label)
        experiment_label = self._infer_experiment_label()
        if experiment_label:
            second_line_parts.append(experiment_label)
        slurm_request_label = self._slurm_request_label(exps)
        if slurm_request_label:
            second_line_parts.append(slurm_request_label)
        quota_label = self._quota_label(exps)
        oom_legend = "* = recovered from OOM" if self._has_oom_recovery(exps) else None
        if oom_legend:
            second_line_parts.append(oom_legend)
        if second_line_parts and quota_label:
            footer_renderable = f"{footer_line}\n{' • '.join(second_line_parts)}\n{quota_label}"
        elif second_line_parts:
            footer_renderable = f"{footer_line}\n{' • '.join(second_line_parts)}"
        elif quota_label:
            footer_renderable = f"{footer_line}\n{quota_label}"
        else:
            footer_renderable = footer_line
        layout["footer"].update(footer_renderable)
        return layout

    # ------------------------------------------------------------------
    # Helper builders
    # ------------------------------------------------------------------
    def _summary_table(
        self, counts: Counter, next_eta: str | None = None, *, timeout_risk_counts: Tuple[int, int] | None = None
    ) -> Table:
        """Return a compact summary line with integer counts per status."""
        tbl = Table(box=_box.MINIMAL, show_header=False, row_styles=["", ""], padding=0)
        tbl.add_column("", justify="left")

        # Compose summary string (Pending, Queued, Running, Failed, Completed)
        status_parts = [
            f"[cyan]Pending[/]: {counts[ExperimentStatus.PENDING] + counts[ExperimentStatus.PARTIAL]}",
            f"[yellow]Queued[/]: {counts[ExperimentStatus.QUEUED]}",
            f"[green]Running[/]: {counts[ExperimentStatus.RUNNING]}",
            f"[bold green]Completed[/]: {counts[ExperimentStatus.COMPLETED]}",
            f"[red]Failed[/]: {counts[ExperimentStatus.FAILED] + counts[ExperimentStatus.CANCELLED] + counts[ExperimentStatus.TIMEOUT] + counts[ExperimentStatus.OOM] + counts[ExperimentStatus.KILLED]}",
        ]
        if next_eta:
            status_parts.append(f"[magenta]Next ETA[/]: {next_eta}")
        if timeout_risk_counts is not None:
            high_risk, med_risk = timeout_risk_counts
            if high_risk or med_risk:
                status_parts.append(f"[yellow]Time risk[/]: [red]H{high_risk}[/]/[yellow]M{med_risk}[/]")
        summary_str = " • ".join(status_parts)
        tbl.add_row(summary_str)
        return tbl

    @staticmethod
    def _coerce_progress_fraction(current_val, max_val, *, allow_zero: bool = False) -> Optional[float]:
        """Return a clamped progress fraction, or None when values are unusable."""
        if current_val is None or max_val in (None, 0):
            return None
        try:
            current_f = float(current_val)
            max_f = float(max_val)
        except (TypeError, ValueError):
            return None
        if max_f <= 0.0:
            return None
        frac = current_f / max_f
        if frac <= 0.0:
            return 0.0 if allow_zero else None
        return min(frac, 1.0)

    @staticmethod
    def _resolve_progress_fraction(exp: dict, *, allow_zero: bool = False) -> Optional[float]:
        """Return best-known completion fraction, or None."""
        candidate_pairs = [("current_step", "max_steps"), ("current_epoch", "max_epochs")]
        fractions: list[float] = []
        for current_key, max_key in candidate_pairs:
            frac = TerminalDashboard._coerce_progress_fraction(
                exp.get(current_key), exp.get(max_key), allow_zero=allow_zero
            )
            if frac is not None:
                fractions.append(frac)
        if not fractions:
            return None
        return max(fractions)

    @staticmethod
    def _resolve_display_progress_fraction(exp: dict, *, allow_zero: bool = False) -> Optional[float]:
        """Return completion fraction using the dashboard row-display precedence."""
        candidate_pairs = [("current_step", "max_steps"), ("current_epoch", "max_epochs")]
        for current_key, max_key in candidate_pairs:
            frac = TerminalDashboard._coerce_progress_fraction(
                exp.get(current_key), exp.get(max_key), allow_zero=allow_zero
            )
            if frac is not None:
                return frac
        return None

    def _overall_run_progress(self, exps: List[dict]) -> tuple[float, int, float]:
        """Return aggregate progress as completed-equivalent runs, eligible runs, and fraction.

        Failed terminal states are omitted. Pending and queued runs remain in the
        denominator and contribute zero progress. Completed runs contribute one
        full run even when old status files lack progress fields.
        """
        eligible = [exp for exp in exps if exp.get("status") not in self._FAILED_STATES]
        if not eligible:
            return 0.0, 0, 0.0

        progress_sum = 0.0
        for exp in eligible:
            status = exp.get("status")
            if status == ExperimentStatus.COMPLETED:
                progress_sum += 1.0
                continue
            if status in {ExperimentStatus.PENDING, ExperimentStatus.QUEUED}:
                continue
            progress_sum += self._resolve_display_progress_fraction(exp, allow_zero=True) or 0.0

        progress_fraction = progress_sum / len(eligible)
        return progress_sum, len(eligible), progress_fraction

    def _resolve_requested_hours_for_risk(self, exp: dict) -> Optional[float]:
        """Resolve configured walltime for timeout risk estimation."""
        for key in ("requested_time_hours", "time_hours_override"):
            val = exp.get(key)
            try:
                hours = float(val)
            except (TypeError, ValueError):
                continue
            if hours > 0.0:
                return hours

        if self.orchestrator is not None:
            val = getattr(self.orchestrator, "time_hours_override", None)
            try:
                hours = float(val)
            except (TypeError, ValueError):
                hours = None
            if hours is not None and hours > 0.0:
                return hours

        return None

    def _timeout_risk_level(self, exp: dict, *, now: float) -> Tuple[str, float] | None:
        """Estimate timeout risk for a running job based on runtime and progress."""
        if exp.get("status") != ExperimentStatus.RUNNING:
            return None

        progress = self._resolve_progress_fraction(exp)
        if progress is None:
            return None
        # Avoid very noisy extrapolation at startup.
        if progress < self.timeout_risk_min_progress:
            return None

        start_ts = exp.get("start_ts") or exp.get("last_change_ts")
        if not isinstance(start_ts, (int, float)):
            return None
        runtime_seconds = now - float(start_ts)
        if runtime_seconds <= 0:
            return None
        if runtime_seconds < self.timeout_risk_min_runtime_seconds:
            return None

        requested_hours = self._resolve_requested_hours_for_risk(exp)
        if requested_hours is None or requested_hours <= 0:
            return None

        estimated_total_hours = (runtime_seconds / 3600.0) / progress
        ratio = estimated_total_hours / requested_hours
        if ratio >= self.timeout_risk_high_ratio:
            return ("high", ratio)
        if ratio >= self.timeout_risk_medium_ratio:
            return ("medium", ratio)
        return None

    def _timeout_risk_counts(self, exps: List[dict], *, now: float) -> Tuple[int, int]:
        """Return timeout risk counts for running jobs as (high, medium)."""
        high = 0
        medium = 0
        for exp in exps:
            risk = self._timeout_risk_level(exp, now=now)
            if not risk:
                continue
            level, _ratio = risk
            if level == "high":
                high += 1
            elif level == "medium":
                medium += 1
        return (high, medium)

    def _infer_project_label(self, exps: List[dict]) -> str | None:
        projects: list[str] = []
        for exp in exps:
            meta = exp.get("metadata")
            if isinstance(meta, dict):
                proj = meta.get("project") or meta.get("tracker_project")
                if proj:
                    projects.append(str(proj))
                    continue
            proj = exp.get("project") or exp.get("tracker_project")
            if proj:
                projects.append(str(proj))

        if self.orchestrator is not None:
            proj = getattr(self.orchestrator, "project", None) or getattr(self.orchestrator, "tracker_project", None)
            if proj:
                projects.append(str(proj))
            exp_file = getattr(self.orchestrator, "experiment_file", None)
            if exp_file:
                try:
                    data = load_yaml(str(exp_file))
                    for exp in data.get("experiments", []):
                        meta = exp.get("metadata")
                        if isinstance(meta, dict):
                            proj = meta.get("project") or meta.get("tracker_project")
                            if proj:
                                projects.append(str(proj))
                                continue
                        proj = exp.get("project") or exp.get("tracker_project")
                        if proj:
                            projects.append(str(proj))
                except Exception:
                    pass

        unique = sorted({p for p in projects if p})
        if not unique:
            return None

        if len(unique) == 1:
            return f"[cyan]Project[/]: {unique[0]}"
        return f"[cyan]Projects[/]: {', '.join(unique)}"

    def _infer_hpc_label(self, exps: List[dict]) -> str | None:
        hpcs = []
        for exp in exps:
            hpc = exp.get("hpc_assignment")
            if hpc is None:
                continue
            name = getattr(hpc, "name", None) or str(hpc)
            if name:
                hpcs.append(str(name))
        unique = sorted({h for h in hpcs if h})
        if not unique:
            return None
        if len(unique) == 1:
            return f"[yellow]Host[/]: {unique[0]}"
        return f"HPCs: {', '.join(unique)}"

    def _slurm_request_label(self, exps: List[dict]) -> str | None:
        """Compact footer segment with a single effective Slurm resource triple."""
        if self.orchestrator is None:
            return None

        def _coerce_positive_int(value) -> int | None:
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                return None
            return parsed if parsed > 0 else None

        requested_hours = _coerce_positive_int(getattr(self.orchestrator, "time_hours_override", None))
        requested_ram = _coerce_positive_int(getattr(self.orchestrator, "memory_gb_override", None))
        requested_gpus = _coerce_positive_int(getattr(self.orchestrator, "max_gpus_per_job", None))

        # If globals are missing, use first resolved snapshot from submitted jobs.
        if requested_hours is None:
            for exp in exps:
                requested_hours = _coerce_positive_int(exp.get("requested_time_hours"))
                if requested_hours is not None:
                    break
        if requested_ram is None:
            for exp in exps:
                requested_ram = _coerce_positive_int(exp.get("requested_ram_gb"))
                if requested_ram is not None:
                    break
        if requested_gpus is None:
            for exp in exps:
                requested_gpus = _coerce_positive_int(exp.get("requested_gpu_count"))
                if requested_gpus is not None:
                    break

        # Final fallback: first enabled cluster default profile.
        if requested_hours is None or requested_ram is None or requested_gpus is None:
            active_hpcs = [
                hpc
                for hpc, limit in getattr(self.orchestrator, "concurrency_limits", {}).items()
                if limit and limit > 0
            ]
            fallback_hpc = active_hpcs[0] if active_hpcs else None
            if fallback_hpc is None:
                for hpc in (e.get("hpc_assignment") for e in exps):
                    if isinstance(hpc, HPCType):
                        fallback_hpc = hpc
                        break
            if fallback_hpc is not None:
                cfg = HPC_CONFIGS.get(fallback_hpc)
                if cfg is not None:
                    if requested_gpus is None:
                        requested_gpus = int(cfg.gpu_count)
                    if requested_hours is None:
                        requested_hours = int(cfg.base_time_hours)
                    if requested_ram is None:
                        if cfg.mem_per_gpu_gb is not None:
                            requested_ram = int(cfg.mem_per_gpu_gb) * int(requested_gpus)
                        else:
                            requested_ram = int(cfg.base_memory_gb)

        if requested_hours is None and requested_ram is None and requested_gpus is None:
            return None

        hours_txt = f"{requested_hours}h" if requested_hours is not None else "auto"
        ram_txt = f"{requested_ram}G" if requested_ram is not None else "auto"
        gpu_txt = f"{requested_gpus}" if requested_gpus is not None else "auto"
        return f"[cyan]Slurm[/]: h={hours_txt} ram={ram_txt} gpu={gpu_txt}"

    @staticmethod
    def _fmt_quota_amount(value: float) -> str:
        if value >= 100:
            return f"{value:,.0f}"
        if value >= 10:
            return f"{value:.1f}"
        return f"{value:.2f}"

    @staticmethod
    def _quota_label_text(label: str) -> str:
        return f"[yellow]{label}[/]"

    @staticmethod
    def _quota_number_text(value: str) -> str:
        return f"[bold yellow]{value}[/]"

    def _active_hpcs(self, exps: List[dict]) -> set[HPCType]:
        active: set[HPCType] = set()
        for exp in exps:
            hpc = exp.get("hpc_assignment")
            if isinstance(hpc, HPCType):
                active.add(hpc)
        if self.orchestrator is not None:
            limits = getattr(self.orchestrator, "concurrency_limits", None)
            if isinstance(limits, dict):
                for hpc, limit in limits.items():
                    if isinstance(hpc, HPCType):
                        try:
                            limit_val = int(limit)
                        except (TypeError, ValueError):
                            continue
                        if limit_val > 0:
                            active.add(hpc)
        return active

    def _get_quota_snapshot(self, hpc_type: HPCType) -> QuotaSnapshot | None:
        """Fetch/cached quota snapshot for one cluster."""
        if self.orchestrator is None:
            return None
        provider = get_quota_provider(hpc_type)
        if provider is None:
            return None

        now = time.time()
        cached = self._quota_cache.get(hpc_type)
        if cached is not None:
            cache_ts, cache_data = cached
            if now - cache_ts < GPU_QUOTA_POLL_INTERVAL_SECONDS:
                return cache_data

        connection_manager = getattr(self.orchestrator, "connection_manager", None)
        if connection_manager is None:
            self._quota_cache[hpc_type] = (now, None)
            return None

        account = str(getattr(HPC_CONFIGS.get(hpc_type), "account", "") or "").strip()
        if not account:
            self._quota_cache[hpc_type] = (now, None)
            return None

        snapshot = provider.fetch_snapshot(account=account, connection_manager=connection_manager)
        self._quota_cache[hpc_type] = (now, snapshot)
        return snapshot

    @staticmethod
    def _coerce_positive_int(value) -> int | None:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    def _resolve_effective_request_hours_and_gpus(self, exp: dict) -> tuple[int | None, int | None]:
        """Resolve requested walltime/GPU count for one experiment."""
        requested_hours = self._coerce_positive_int(exp.get("requested_time_hours"))
        requested_gpus = self._coerce_positive_int(exp.get("requested_gpu_count"))

        if self.orchestrator is not None:
            if requested_hours is None:
                requested_hours = self._coerce_positive_int(getattr(self.orchestrator, "time_hours_override", None))
            if requested_gpus is None:
                requested_gpus = self._coerce_positive_int(getattr(self.orchestrator, "max_gpus_per_job", None))

        if requested_hours is not None and requested_gpus is not None:
            return requested_hours, requested_gpus

        hpc = exp.get("hpc_assignment")
        cfg = HPC_CONFIGS.get(hpc) if isinstance(hpc, HPCType) else None
        if cfg is not None:
            if requested_hours is None:
                requested_hours = self._coerce_positive_int(getattr(cfg, "base_time_hours", None))
            if requested_gpus is None:
                requested_gpus = self._coerce_positive_int(getattr(cfg, "gpu_count", None))

        return requested_hours, requested_gpus

    def _estimate_orchestration_worst_case_hours_per_cluster(self, exps: List[dict]) -> dict[HPCType, float]:
        """Estimate worst-case remaining orchestration cost in GPU-hours by cluster."""
        worst_case: dict[HPCType, float] = {}
        seen_unfinished_hpcs: set[HPCType] = set()
        for exp in exps:
            status = exp.get("status")
            if status in self._DONE_STATES:
                continue
            hpc = exp.get("hpc_assignment")
            if not isinstance(hpc, HPCType):
                continue
            seen_unfinished_hpcs.add(hpc)
            requested_hours, requested_gpus = self._resolve_effective_request_hours_and_gpus(exp)
            if requested_hours is None or requested_gpus is None:
                continue
            worst_case[hpc] = worst_case.get(hpc, 0.0) + float(requested_hours * requested_gpus)

        for hpc in seen_unfinished_hpcs:
            worst_case.setdefault(hpc, 0.0)
        return worst_case

    @staticmethod
    def _allocation_period_elapsed_pct(start: date, end: date, today: date) -> float:
        """Return elapsed percentage for an allocation period."""
        total_days = max((end - start).days, 1)
        elapsed_days = max(min((today - start).days, total_days), 0)
        return (elapsed_days / float(total_days)) * 100.0

    def _quota_period_footer_status(
        self, provider: QuotaProvider, snapshot: QuotaSnapshot | None
    ) -> tuple[int, str, float] | None:
        """Return quota-period footer values when the provider exposes a period."""
        today = dt.now().date()
        if snapshot is not None and snapshot.period_start is not None and snapshot.period_end is not None:
            period_start, period_end = snapshot.period_start, snapshot.period_end
        else:
            try:
                period_bounds = provider.period_bounds(today=today)
            except Exception as exc:  # pragma: no cover - defensive path
                logger.debug("Quota provider period probe failed for %s: %s", provider.hpc_type, exc)
                return None
            if period_bounds is None:
                return None
            period_start, period_end = period_bounds
        elapsed_pct = self._allocation_period_elapsed_pct(period_start, period_end, today)
        days_left = max((period_end - today).days, 0)
        return days_left, period_end.strftime('%d-%m-%y'), elapsed_pct

    def _quota_period_segment(self, period_status: tuple[int, str, float], pace_delta_pp: float | None = None) -> str:
        """Render allocation-period footer segment."""
        days_left, period_end_txt, period_elapsed_pct = period_status
        if pace_delta_pp is None:
            return (
                f"{self._quota_label_text('Period')}: {self._quota_number_text(f'{days_left}d')} left "
                f"({self._quota_number_text(f'{period_elapsed_pct:.1f}%')} elapsed; "
                f"ends {self._quota_number_text(period_end_txt)})"
            )
        return (
            f"{self._quota_label_text('Period')}: {self._quota_number_text(f'{days_left}d')} left "
            f"({self._quota_number_text(f'{period_elapsed_pct:.1f}%')} elapsed, "
            f"{self._quota_number_text(f'{pace_delta_pp:+.1f}pp')}; "
            f"ends {self._quota_number_text(period_end_txt)})"
        )

    def _render_quota_line(
        self,
        *,
        hpc_type: HPCType,
        provider: QuotaProvider,
        snapshot: QuotaSnapshot | None,
        worst_case_hours: float | None,
    ) -> str:
        """Render one provider-backed quota footer line."""
        period_status = self._quota_period_footer_status(provider, snapshot)
        period_segment = self._quota_period_segment(period_status) if period_status is not None else None
        provider_label = str(getattr(provider, "resource_label", "Quota"))
        provider_cluster = getattr(provider, "cluster_name", hpc_type.value)
        if snapshot is None:
            hint = str(getattr(provider, "unavailable_hint", "quota probe unavailable"))
            quota_line = f"{self._quota_label_text(provider_label)}: {provider_cluster} unavailable ({hint})"
            return f"{quota_line} • {period_segment}" if period_segment else quota_line

        if snapshot.limit <= 0.0:
            quota_line = (
                f"{self._quota_label_text(snapshot.resource_label)}: "
                f"{snapshot.cluster_name} unavailable (invalid quota limit)"
            )
            return f"{quota_line} • {period_segment}" if period_segment else quota_line

        pace_delta_pp = snapshot.used_pct - period_status[2] if period_status is not None else None
        period_with_delta = (
            self._quota_period_segment(period_status, pace_delta_pp=pace_delta_pp)
            if period_status is not None
            else None
        )
        used_total = (
            f"{self._fmt_quota_amount(snapshot.used)}/" f"{self._fmt_quota_amount(snapshot.limit)}{snapshot.unit}"
        )
        quota_line = (
            f"{self._quota_label_text(snapshot.resource_label)}: {snapshot.cluster_name} "
            f"{self._quota_number_text(f'{self._fmt_quota_amount(snapshot.remaining)}{snapshot.unit}')} left "
            f"({self._quota_number_text(f'{snapshot.used_pct:.1f}%')} used; {self._quota_number_text(used_total)})"
        )
        if snapshot.worst_case_unit == "gpu_hours" and worst_case_hours is not None:
            pct_left = (worst_case_hours / snapshot.remaining * 100.0) if snapshot.remaining > 0 else 0.0
            quota_line += (
                f" • {self._quota_label_text('Orch worst-case')}: "
                f"{self._quota_number_text(f'{self._fmt_quota_amount(worst_case_hours)}{snapshot.unit}')} "
                f"({self._quota_number_text(f'{pct_left:.1f}%')} of left)"
            )
        return f"{quota_line} • {period_with_delta}" if period_with_delta else quota_line

    def _quota_label(self, exps: List[dict]) -> str | None:
        """Footer label with remaining project quota for active clusters."""
        active_hpcs = sorted(self._active_hpcs(exps), key=lambda hpc: hpc.value)
        worst_case_by_hpc = self._estimate_orchestration_worst_case_hours_per_cluster(exps)
        lines: list[str] = []
        for hpc_type in active_hpcs:
            provider = get_quota_provider(hpc_type)
            if provider is None:
                continue
            snapshot = self._get_quota_snapshot(hpc_type)
            lines.append(
                self._render_quota_line(
                    hpc_type=hpc_type,
                    provider=provider,
                    snapshot=snapshot,
                    worst_case_hours=worst_case_by_hpc.get(hpc_type),
                )
            )
        return "\n".join(lines) if lines else None

    def _infer_experiment_label(self) -> str | None:
        if self.orchestrator is None:
            return None
        exp_file = getattr(self.orchestrator, "experiment_file", None)
        if not exp_file:
            return None
        try:
            return f"[cyan]Experiment[/]: [dim]{exp_file.stem}[/]"
        except Exception:
            return None

    @staticmethod
    def _has_oom_recovery(exps: List[dict]) -> bool:
        return any(bool(exp.get("oom_recovered")) for exp in exps)

    def _next_allocation_eta(self, exps: List[dict]) -> str | None:
        eta_values = [
            float(e.get("estimated_start_ts"))
            for e in exps
            if e.get("status") == ExperimentStatus.QUEUED and isinstance(e.get("estimated_start_ts"), (int, float))
        ]
        if not eta_values:
            return None
        next_eta = min(eta_values)
        now = time.time()
        if next_eta <= now:
            return "now"
        return self._fmt_duration(next_eta - now, show_days=(next_eta - now) >= 86400)

    @staticmethod
    def _metric_info_map(exp: dict) -> dict[str, dict]:
        info = exp.get("display_metric_info", {})
        return info if isinstance(info, dict) else {}

    @staticmethod
    def _metric_column_specs(exp: dict) -> list[dict]:
        columns = exp.get("display_metric_columns", [])
        if not isinstance(columns, list):
            return []
        return [column for column in columns if isinstance(column, dict) and column.get("key")]

    def _metric_info_for(self, exp: dict, metric_key: str | None) -> dict | None:
        if not metric_key:
            return None
        info = self._metric_info_map(exp).get(metric_key)
        if isinstance(info, dict):
            return info
        for column in self._metric_column_specs(exp):
            if column.get("key") == metric_key:
                return column
        return None

    def _resolve_v3_metric_columns(self, exps: List[dict]) -> list[tuple[str, dict]]:
        metric_columns: list[tuple[str, dict]] = []
        seen: set[str] = set()

        for exp in exps:
            explicit_columns = self._metric_column_specs(exp)
            if explicit_columns:
                ordered_columns = [(str(column["key"]), column) for column in explicit_columns]
            else:
                ordered_columns = [
                    (metric_key, self._metric_info_for(exp, metric_key) or {})
                    for metric_key in (
                        exp.get("target_metric_name"),
                        exp.get("secondary_metric_name"),
                        *self._metric_info_map(exp).keys(),
                    )
                ]
            for metric_key, metric_info in ordered_columns:
                if not metric_key or metric_key in seen:
                    continue
                metric_columns.append((metric_key, metric_info))
                seen.add(metric_key)

        # Keep the table compact by dropping metrics that are entirely empty across
        # all visible experiments (no current and no best value anywhere).
        non_empty_columns: list[tuple[str, dict]] = []
        for metric_key, metric_info in metric_columns:
            shortform = metric_info.get("shortform")
            has_any_value = False
            for exp in exps:
                current_val = self._lookup_metric_value(exp, metric_key, shortform)
                best_val = self._lookup_best_metric_value(exp, metric_info)
                if current_val is not None or best_val is not None:
                    has_any_value = True
                    break
            if has_any_value:
                non_empty_columns.append((metric_key, metric_info))

        if non_empty_columns:
            return non_empty_columns
        return metric_columns

    def _resolve_primary_metric_key(self, exp: dict, metric_columns: list[tuple[str, dict]]) -> str | None:
        explicit_columns = self._metric_column_specs(exp)
        if explicit_columns:
            return str(explicit_columns[0]["key"])

        metric_key = exp.get("target_metric_name")
        if metric_key:
            return metric_key

        if metric_columns:
            return metric_columns[0][0]
        return None

    def _lookup_metric_value(self, exp: dict, metric_key: str, shortform: str | None = None) -> float | None:
        all_metrics = exp.get("all_metrics", {})
        candidates: list[str] = []
        if shortform:
            candidates.append(shortform)
        candidates.append(metric_key)
        for candidate in candidates:
            if candidate in exp:
                return exp.get(candidate)
            if candidate in all_metrics:
                return all_metrics.get(candidate)
        return None

    def _lookup_best_metric_value(self, exp: dict, metric_info: dict | None) -> float | None:
        if not metric_info:
            return None
        best_key = metric_info.get("best_key")
        if not best_key:
            return None
        return self._lookup_metric_value(exp, best_key)

    @staticmethod
    def _metric_sort_value(value: float | None, metric_info: dict | None) -> float:
        if value is None:
            return float("inf")
        higher_better = True if not isinstance(metric_info, dict) else metric_info.get("higher_better", True)
        return -value if higher_better is not False else value

    @staticmethod
    def _metric_color(value: float | None, metric_info: dict | None) -> str | None:
        if value is None or not isinstance(metric_info, dict):
            return None
        threshold = metric_info.get("threshold")
        if not isinstance(threshold, (int, float)):
            return None
        higher_better = metric_info.get("higher_better", True)
        if higher_better is False:
            return "green" if value <= threshold else "red"
        return "green" if value >= threshold else "red"

    def _fmt_metric_number(self, v: float | None, value_format: str | None = None) -> str:
        if v is None:
            return "—"
        if value_format == "integer":
            return f"{int(round(v))}"
        return f"{v:.{METRIC_VALUE_PRECISION}f}"

    def _format_metric_pair(
        self, current: float | None, best: float | None, style: str | None = None, metric_info: dict | None = None
    ) -> str | Text:
        value_format = metric_info.get("value_format") if isinstance(metric_info, dict) else None
        current_str = self._fmt_metric_number(current, value_format=value_format)
        if best is not None:
            best_str = self._fmt_metric_number(best, value_format=value_format)
            combined = f"{current_str} ({best_str})"
        else:
            combined = current_str
        if style:
            return Text(combined, style=style)
        return combined

    def _format_state(self, status: ExperimentStatus | None) -> Text | str:
        if not status:
            return "?"
        label = status.name
        style = {
            ExperimentStatus.PENDING: "cyan",
            ExperimentStatus.PARTIAL: "yellow",
            ExperimentStatus.QUEUED: "yellow",
            ExperimentStatus.RUNNING: "green",
            ExperimentStatus.COMPLETED: "bold green",
            ExperimentStatus.FAILED: "bold red",
            ExperimentStatus.CANCELLED: "bold red",
            ExperimentStatus.TIMEOUT: "bold red",
            ExperimentStatus.OOM: "bold red",
            ExperimentStatus.KILLED: "bold red",
        }.get(status, "bold")
        return Text(label, style=style)

    def _format_progress(self, exp: dict) -> str:
        current_step = exp.get("current_step")
        max_steps = exp.get("max_steps")
        if current_step is not None and max_steps not in (None, 0):
            try:
                current_val = int(current_step)
            except Exception:
                current_val = current_step
            try:
                max_val = int(max_steps)
            except Exception:
                max_val = max_steps
            pct = self._progress_pct(current_step, max_steps)
            it_per_sec = exp.get("it_per_sec_backbone")
            if isinstance(it_per_sec, (int, float)) and it_per_sec > 0:
                return f"{current_val}/{max_val} {pct} @ {it_per_sec:.2f}it/s"
            return f"{current_val}/{max_val} {pct}"

        current = exp.get("current_epoch")
        maximum = exp.get("max_epochs")
        if current is None or maximum is None or maximum == 0:
            return "?"
        try:
            current_val = int(current)
        except Exception:
            current_val = current
        try:
            max_val = int(maximum)
        except Exception:
            max_val = maximum
        pct = self._progress_pct(current, maximum)
        return f"{current_val}/{max_val} {pct}"

    def _all_runs_table(self, exps: List[dict]) -> Table:
        rows = list(exps)
        metric_columns = self._resolve_v3_metric_columns(rows)

        def _sort_key(entry: dict) -> tuple[int, int, float, float]:
            metric_key = self._resolve_primary_metric_key(entry, metric_columns)
            metric_info = self._metric_info_for(entry, metric_key)
            current_val = None
            if metric_key:
                current_val = self._lookup_metric_value(
                    entry, metric_key, metric_info.get("shortform") if metric_info else None
                )
            best_val = self._lookup_best_metric_value(entry, metric_info)
            sort_val = best_val if best_val is not None else current_val
            has_metric = sort_val is not None
            metric_sort = self._metric_sort_value(sort_val, metric_info)
            last_ts = entry.get("last_change_ts") or 0
            status = entry.get("status")
            if status == ExperimentStatus.RUNNING:
                status_priority = 0
            elif status == ExperimentStatus.COMPLETED:
                status_priority = 1
            else:
                status_priority = 2
            return (status_priority, 0 if has_metric else 1, metric_sort, -last_ts)

        rows.sort(key=_sort_key)
        rows = rows[: self.n_recent]

        tbl = Table(box=_box.MINIMAL, show_header=True, header_style="bold", row_styles=["", ALT_ROW_STYLE])
        tbl.add_column("ID")
        if not self.is_sweep:
            tbl.add_column("Dataset")
        tbl.add_column("HPC")
        tbl.add_column("State")
        tbl.add_column("Progress", justify="right")

        for metric_key, metric_info in metric_columns:
            header = self._get_metric_header(metric_key, metric_info=metric_info)
            tbl.add_column(header, justify="right")

        tbl.add_column("Queue/Δt", justify="right", style="dim")
        tbl.column_spacing = COLUMN_SPACING

        now = time.time()
        show_days_wait = any((now - (e.get("queued_timestamp") or now)) >= 86400 for e in rows)
        show_days_run = any(
            (now - (e.get("start_ts") or e.get("last_change_ts") or now)) >= 86400
            for e in rows
            if e.get("status") == ExperimentStatus.RUNNING
        )

        for exp in rows:
            exp_id = str(exp.get("experiment_id", "?"))
            if exp.get("oom_recovered"):
                exp_id += "*"
            row = [exp_id]
            if not self.is_sweep:
                row.append(self._format_dataset(exp))
            hpc = getattr(exp.get("hpc_assignment"), "name", "?")
            row.append(hpc)
            row.append(self._format_state(exp.get("status")))
            progress_renderable: str | Text = self._format_progress(exp)
            risk = self._timeout_risk_level(exp, now=now)
            if risk:
                level, _ratio = risk
                if level == "high":
                    progress_renderable = Text(str(progress_renderable), style="bold red")
                elif level == "medium":
                    progress_renderable = Text(str(progress_renderable), style="yellow")
            row.append(progress_renderable)

            for metric_key, metric_info in metric_columns:
                shortform = metric_info.get("shortform")
                current_val = self._lookup_metric_value(exp, metric_key, shortform)
                best_val = self._lookup_best_metric_value(exp, metric_info)
                style = self._metric_color(current_val, metric_info)
                row.append(self._format_metric_pair(current_val, best_val, style, metric_info=metric_info))

            queue_str = ""
            status = exp.get("status")
            if status == ExperimentStatus.QUEUED:
                queued_ts = exp.get("queued_timestamp")
                if queued_ts:
                    wait_sec = now - queued_ts
                    wait_str = self._fmt_duration(wait_sec, show_days_wait)
                else:
                    wait_str = "?"
                eta_ts = exp.get("estimated_start_ts")
                if eta_ts:
                    eta_in = eta_ts - now
                    eta_str = self._fmt_duration(eta_in, show_days_wait) if eta_in > 0 else "now"
                else:
                    eta_str = "?"
                queue_str = f"{wait_str} ({eta_str})"
            elif status == ExperimentStatus.RUNNING:
                start_ts = exp.get("start_ts") or exp.get("last_change_ts")
                if start_ts:
                    run_sec = now - start_ts
                    queue_str = self._fmt_duration(run_sec, show_days_run)
                else:
                    queue_str = "?"
            row.append(queue_str)

            tbl.add_row(*row)

        if not rows:
            blanks = ["—"]
            if not self.is_sweep:
                blanks.append("")
            blanks.extend(["", "", "", ""])
            blanks.extend(["" for _ in metric_columns])
            blanks.append("")
            tbl.add_row(*blanks)

        return tbl

    def _recent_table(
        self, exps: List[dict], status: ExperimentStatus | tuple | list, include_delta: bool = True
    ) -> Table:
        # Allow passing a single status or a collection
        if isinstance(status, (list, tuple, set)):
            status_set = set(status)
        else:
            status_set = {status}

        rows = [e for e in exps if e.get("status") in status_set and e.get("last_change_ts") is not None]
        # --------------------------------------------------------------
        # Sort: 1) by primary metric (if present), 2) by recency
        # --------------------------------------------------------------

        def _sort_key(entry: dict) -> tuple[int, float, float]:
            metric_val = entry.get("target_metric_value")
            has_metric = metric_val is not None
            if metric_val is None:
                metric_sort = float("inf")
            else:
                metric_info = self._metric_info_for(entry, entry.get("target_metric_name"))
                metric_sort = self._metric_sort_value(metric_val, metric_info)
            return (0 if has_metric else 1, metric_sort, -entry.get("last_change_ts", 0))

        rows.sort(key=_sort_key)

        rows = rows[: self.n_recent]

        # Pick representative style (first element)
        sample_status = next(iter(status_set))
        header_style = {
            ExperimentStatus.RUNNING: "bold green",
            ExperimentStatus.COMPLETED: "bold green",
            ExperimentStatus.FAILED: "bold red",
            ExperimentStatus.CANCELLED: "bold red",
            ExperimentStatus.TIMEOUT: "bold red",
            ExperimentStatus.OOM: "bold red",
            ExperimentStatus.PARTIAL: "bold yellow",
            ExperimentStatus.KILLED: "bold green",
        }.get(sample_status, "bold")

        # Use generic title when multiple statuses
        if len(status_set) == 1:
            title = next(iter(status_set)).name.title()
        else:
            title = "/".join(s.name.title() for s in status_set)
        tbl = Table(box=_box.MINIMAL, show_header=True, header_style=header_style, row_styles=["", ALT_ROW_STYLE])
        tbl.add_column(f"Newest {title}")
        if not self.is_sweep:  # Hide dataset column in sweep mode to save space
            tbl.add_column("Dataset")
        tbl.add_column("HPC")

        # 1. Detect if any of the rows contain progress info (current_epoch, max_epochs)
        sample_has_progress = any(
            (e.get("current_step") is not None and e.get("max_steps") is not None)
            or (e.get("current_epoch") is not None and e.get("max_epochs") is not None)
            for e in rows
        )
        if sample_has_progress:
            tbl.add_column("Prog%", justify="right")

        # 2. Detect if any row has metrics
        sample_has_metrics = any(
            e.get("target_metric_value") is not None or e.get("metric_value") is not None for e in rows
        )

        if sample_has_metrics:
            unique_names = []
            for r in rows:
                name = r.get("target_metric_name")
                if name and name not in unique_names:
                    unique_names.append(name)
                if len(unique_names) >= 2:
                    break
            if len(unique_names) < 2 and rows:
                sec = rows[0].get("secondary_metric_name")
                if sec and sec not in unique_names:
                    unique_names.append(sec)

            m1_info = self._metric_info_for(rows[0], unique_names[0]) if unique_names and rows else None
            m2_info = self._metric_info_for(rows[0], unique_names[1]) if len(unique_names) > 1 and rows else None
            m1_header = self._get_metric_header(unique_names[0], m1_info) if unique_names else "M1"
            m2_header = self._get_metric_header(unique_names[1], m2_info) if len(unique_names) > 1 else "M2"
            tbl.add_column(m1_header, justify="right")
            tbl.add_column(m2_header, justify="right")

        if include_delta and SHOW_DELTA_TIME:
            tbl.add_column("Δt", justify="right", style="dim")
        # Configurable column spacing
        tbl.column_spacing = COLUMN_SPACING

        now = time.time()
        # Determine if we should display days
        show_days = any((now - e["last_change_ts"]) >= 86400 for e in rows)

        if rows:
            for e in rows:
                dt_sec = now - e["last_change_ts"]
                hpc = getattr(e.get("hpc_assignment"), "name", "?")
                exp_id = str(e.get("experiment_id", "?"))
                if e.get("oom_recovered"):
                    exp_id += "*"
                row = [exp_id]
                if not self.is_sweep:
                    row.append(self._format_dataset(e))
                row.append(hpc)

                if sample_has_progress:
                    if e.get("current_step") is not None and e.get("max_steps") is not None:
                        row.append(self._progress_pct(e.get("current_step"), e.get("max_steps")))
                    else:
                        ce = e.get("current_epoch")
                        me = e.get("max_epochs")
                        row.append(self._progress_pct(ce, me))

                if sample_has_metrics:
                    m1v = (
                        e.get("target_metric_value")
                        if e.get("target_metric_value") is not None
                        else e.get("metric_value")
                    )
                    m2v = e.get("secondary_metric_value")

                    row.append(self._fmt_val(m1v))
                    row.append(self._fmt_val(m2v))

                if include_delta and SHOW_DELTA_TIME:
                    row.append(self._fmt_duration(dt_sec, show_days))

                tbl.add_row(*row)
        else:
            blanks = ["—"]
            if not self.is_sweep:
                blanks.append("")  # Dataset column
            blanks.append("")  # HPC column
            if sample_has_progress:
                blanks.append("")
            if sample_has_metrics:
                blanks.extend(["", ""])  # 2 legacy metrics
            if include_delta and SHOW_DELTA_TIME:
                blanks.append("")
            tbl.add_row(*blanks)
        return tbl

    def _get_metric_header(self, metric_name: str, metric_info: dict | None = None) -> str:
        """Get metric header using target-schema display info, with fallback to abbreviation."""
        if not metric_name:
            return "?"

        shortform = None
        if isinstance(metric_info, dict):
            shortform = metric_info.get("shortform")
        if shortform:
            return shortform.upper().replace("_", "-")

        # Fallback to abbreviation
        return self._abbr_metric(metric_name)

    def _queued_table(self, exps: List[dict]) -> Table:
        """Table specialised for QUEUED jobs showing HPC and ETA."""
        rows = [e for e in exps if e.get("status") == ExperimentStatus.QUEUED]
        # Sort by queued timestamp ascending (longest waiting first)
        rows.sort(key=lambda e: e.get("queued_timestamp", 0))
        rows = rows[: self.n_recent]

        tbl = Table(box=_box.MINIMAL, show_header=True, header_style="bold", row_styles=["", ALT_ROW_STYLE])

        # Configurable queued table columns
        if SHOW_QUEUED_IDS:
            tbl.add_column("Queued ID")
        tbl.add_column("Dataset", style="dim")
        tbl.add_column("HPC", style="dim")
        tbl.add_column("Wait", justify="right", style="dim")
        tbl.add_column("ETA", justify="right", style="dim")
        # Configurable column spacing
        tbl.column_spacing = COLUMN_SPACING

        now = time.time()
        show_days_wait = any((now - (e.get("queued_timestamp") or now)) >= 86400 for e in rows)

        for e in rows:
            queued_ts = e.get("queued_timestamp")
            wait_sec = now - queued_ts if queued_ts else 0
            eta_ts = e.get("estimated_start_ts")
            if eta_ts:
                eta_in = eta_ts - now
                eta_str = self._fmt_duration(eta_in, show_days_wait) if eta_in > 0 else "now"
            else:
                eta_str = "?"

            hpc = getattr(e.get("hpc_assignment"), "name", "?")
            dataset = self._format_dataset(e)

            # Build row data based on column configuration
            row_data = []
            if SHOW_QUEUED_IDS:
                exp_id = str(e.get("experiment_id", "?"))
                if e.get("oom_recovered"):
                    exp_id += "*"
                row_data.append(exp_id)
            row_data.extend([dataset, hpc, self._fmt_duration(wait_sec, show_days_wait), eta_str])
            tbl.add_row(*row_data)

        if not rows:
            # Empty row with correct number of columns
            empty_cols = []
            if SHOW_QUEUED_IDS:
                empty_cols.append("—")
            empty_cols.extend(["—", "", "", ""])
            tbl.add_row(*empty_cols)
        return tbl

    def _logs_table(self, exps: List[dict]) -> Table:
        """Show recent completed/failed runs with their log directory."""
        done_states = [ExperimentStatus.COMPLETED, ExperimentStatus.FAILED]
        rows = [e for e in exps if e.get("status") in done_states and e.get("output_dir")]
        rows.sort(
            key=lambda e: (0 if e.get("status") == ExperimentStatus.FAILED else 1, -float(e.get("last_change_ts", 0)))
        )
        rows = rows[: self.n_recent]

        tbl = Table(box=_box.MINIMAL, show_header=True, header_style="bold")
        tbl.add_column("Done ID")
        tbl.add_column("Status", style="dim")
        tbl.add_column("Logs path", overflow="fold")
        # Configurable column spacing
        tbl.column_spacing = COLUMN_SPACING

        for e in rows:
            exp_id = str(e.get("experiment_id", "?"))
            if e.get("oom_recovered"):
                exp_id += "*"
            stat = e.get("status").name if e.get("status") else "?"
            path = str(e.get("output_dir", ""))
            tbl.add_row(exp_id, stat, path)

        if not rows:
            tbl.add_row("—", "", "")
        return tbl

    def _failed_table(self, exps: List[dict]) -> Table:
        """Show recently failed experiments and their HPC assignment."""
        fail_states = [
            ExperimentStatus.FAILED,
            ExperimentStatus.CANCELLED,
            ExperimentStatus.TIMEOUT,
            ExperimentStatus.OOM,
            ExperimentStatus.KILLED,
        ]
        rows = [e for e in exps if e.get("status") in fail_states and e.get("last_change_ts") is not None]
        rows.sort(key=lambda e: e.get("last_change_ts", 0), reverse=True)
        rows = rows[: self.n_recent]

        tbl = Table(box=_box.MINIMAL, show_header=True, header_style="bold red", row_styles=["", ALT_ROW_STYLE])
        tbl.add_column("Failed ID")
        tbl.add_column("St", justify="right", style="dim")
        tbl.add_column("Dataset", style="dim")
        tbl.add_column("HPC", style="dim")
        tbl.column_spacing = COLUMN_SPACING

        for e in rows:
            exp_id = str(e.get("experiment_id", "?"))
            if e.get("oom_recovered"):
                exp_id += "*"
            stat = self._abbr_status(e.get("status"))
            dataset = self._format_dataset(e)
            hpc = getattr(e.get("hpc_assignment"), "name", "?")
            tbl.add_row(exp_id, stat, dataset, hpc)

        if not rows:
            tbl.add_row("—", "", "", "")
        return tbl

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------
    def _abbr_status(self, status: ExperimentStatus | None) -> str:
        """Abbreviate experiment status for compact display."""
        if not status:
            return "?"
        name = status.name
        return {"FAILED": "F", "CANCELLED": "C", "TIMEOUT": "TO", "OOM": "OM", "KILLED": "K"}.get(name, name[:2])

    @staticmethod
    def _abbr_metric(name: str | None) -> str:
        """Shorten a slash-delimited metric path for compact table headers."""
        if not name:
            return "?"

        parts = name.strip("/").split("/")
        if len(parts) >= 2:
            return parts[-1]
        return name[:12]

    def _fmt_val(self, v: float | None, style: str | None = None) -> str | Text:
        """Format a metric value with optional styling.

        Args:
            v: The metric value to format
            style: Optional Rich style string (e.g., "green", "yellow", "red")

        Returns:
            Formatted string or Rich Text object with styling
        """
        formatted = self._fmt_metric_number(v)
        if style:
            return Text(formatted, style=style)
        return formatted

    def _fmt_duration(self, seconds: float, show_days: bool = False) -> str:
        """Format a duration in a compact way."""
        if seconds < 0:
            return "?"
        d = int(seconds // 86400)
        h = int((seconds % 86400) // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)

        # Only show days if requested and non-zero
        if show_days and d > 0:
            if d == 1:
                return f"1d{h}h"
            else:
                return f"{d}d{h}h"
        elif h > 0:
            return f"{h}h{m:02d}m"
        elif m > 0:
            return f"{m}m{s:02d}s"
        else:
            return f"{s}s"

    def _progress_pct(self, current: int | None, max_val: int | None) -> str:
        """Format epoch progress as a percentage."""
        if current is None or max_val is None or max_val == 0:
            return "?"
        pct = (current / max_val) * 100
        return f"{pct:{PROGRESS_PERCENTAGE_WIDTH}.1f}%"

    def _format_dataset(self, exp: dict) -> str:
        """Format dataset name, potentially abbreviated for sweep mode."""
        dataset = exp.get("dataset_name", "?")
        # In sweep mode (when all experiments use same dataset), abbreviate
        if self.is_sweep and dataset.startswith("UCR_"):
            return dataset[4:]  # Remove "UCR_" prefix
        return dataset

    def _infer_sweep_dataset(self, exps: List[dict]) -> str | None:
        """Check if all experiments use the same dataset (sweep mode)."""
        datasets = {e.get("dataset_name") for e in exps if e.get("dataset_name")}
        if len(datasets) == 1:
            self.is_sweep = True
            return datasets.pop()
        self.is_sweep = False
        return None

    def _infer_sweep_url(self, exps: List[dict]) -> str | None:
        """Return a sweep URL from generic experiment link metadata."""
        for exp in exps:
            for key in ("sweep_url", "tracker_sweep_url"):
                value = exp.get(key)
                if value:
                    return str(value)
            links = exp.get("links")
            if isinstance(links, dict):
                for key, value in links.items():
                    if value and "sweep" in str(key).lower() and "url" in str(key).lower():
                        return str(value)
        return None

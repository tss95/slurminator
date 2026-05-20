"""Home screen for the Textual dashboard."""

from __future__ import annotations

import logging
import time
from collections import Counter
from datetime import date, datetime as dt
from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Container
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Static

from slurminator.config import HPCType, HPC_CONFIGS
from slurminator.dashboard_v4.commands import submit_command
from slurminator.dashboard_v4.keystrokes import HOME_BINDINGS
from slurminator.dashboard_v4.per_run_menu import PerRunMenuScreen
from slurminator.dashboard_v4.widgets import ExperimentsTable
from slurminator.experiments import ExperimentStatus
from slurminator.experiments.yaml_utils import load_yaml
from slurminator.quota import QuotaProvider, QuotaSnapshot, get_quota_provider

logger = logging.getLogger("slurminator")

TOP_BAR_WIDTH = 20
GPU_QUOTA_POLL_INTERVAL_SECONDS = 300.0
DONE_STATES = {
    ExperimentStatus.COMPLETED,
    ExperimentStatus.FAILED,
    ExperimentStatus.CANCELLED,
    ExperimentStatus.TIMEOUT,
    ExperimentStatus.OOM,
    ExperimentStatus.KILLED,
}
FAILED_STATES = {
    ExperimentStatus.FAILED,
    ExperimentStatus.CANCELLED,
    ExperimentStatus.TIMEOUT,
    ExperimentStatus.OOM,
    ExperimentStatus.KILLED,
}


class HomeScreen(Screen[None]):
    """Primary v4 dashboard screen."""

    BINDINGS = HOME_BINDINGS

    def __init__(self) -> None:
        super().__init__()
        self._quota_cache: dict[HPCType, tuple[float, QuotaSnapshot | None]] = {}
        self._last_summary_text = Text("")
        self._last_progress_text = Text("")
        self._last_footer_text = Text("")

    def compose(self) -> ComposeResult:
        """Compose the dashboard home screen."""
        yield Header()
        yield Container(
            Static("", id="summary"),
            Static("", id="progress-bars"),
            ExperimentsTable(id="exps"),
            Static("", id="quota"),
            id="home-content",
        )
        yield Footer()

    def on_mount(self) -> None:
        """Start periodic table refreshes from the orchestrator snapshot."""
        self.set_interval(getattr(self.app, "refresh_interval", 1.0), self.refresh_from_orchestrator)
        self.refresh_from_orchestrator()
        self.query_one(ExperimentsTable).focus()

    def on_data_table_row_selected(self, _event: DataTable.RowSelected) -> None:
        """Open the placeholder modal when Enter selects the current table row."""
        self.action_drill_in()

    def refresh_from_orchestrator(self) -> None:
        """Refresh summary, table, and footer from the app snapshot."""
        experiments = self.app.get_dashboard_snapshot()
        self._last_summary_text = self._summary_text(experiments)
        self._last_progress_text = self._progress_text(experiments)
        self._last_footer_text = self._footer_text(experiments)
        self.query_one("#summary", Static).update(self._last_summary_text)
        self.query_one("#progress-bars", Static).update(self._last_progress_text)
        self.query_one(ExperimentsTable).update_experiments(
            experiments,
            show_sparkline=bool(getattr(self.app, "sparkline_enabled", False)),
            sparkline_thresholds=getattr(self.app, "sparkline_thresholds", None),
        )
        self.query_one("#quota", Static).update(self._last_footer_text)

    def action_cursor_up(self) -> None:
        """Move the table cursor up."""
        self.query_one(ExperimentsTable).action_cursor_up()

    def action_cursor_down(self) -> None:
        """Move the table cursor down."""
        self.query_one(ExperimentsTable).action_cursor_down()

    def action_drill_in(self) -> None:
        """Open the placeholder per-run modal for the selected row."""
        experiments = self.app.get_dashboard_snapshot()
        exp = self.query_one(ExperimentsTable).selected_experiment(experiments)
        if exp is not None:
            self.app.push_screen(PerRunMenuScreen(exp))

    def action_toggle_pause(self) -> None:
        """Write a pause/resume command for the next orchestrator poll."""
        orchestrator = getattr(self.app, "orchestrator", None)
        action = (
            "resume_submissions" if bool(getattr(orchestrator, "submissions_paused", False)) else "pause_submissions"
        )
        submit_command(self.app.command_save_path(), action, {})

    def action_toggle_sparkline(self) -> None:
        """Toggle the trajectory column."""
        self.app.sparkline_enabled = not bool(getattr(self.app, "sparkline_enabled", False))
        self.refresh_from_orchestrator()

    def action_quit(self) -> None:
        """Request a graceful dashboard and orchestrator shutdown."""
        self.app.request_dashboard_exit()

    def action_noop(self) -> None:
        """Ignore Escape on the home screen for now."""
        return None

    def _summary_text(self, experiments: list[dict[str, Any]]) -> Text:
        counts = Counter(_status_text(exp.get("status")) for exp in experiments)
        summary = Text()
        parts: list[tuple[str, int, str]] = [
            ("Pending", counts["PENDING"] + counts["PARTIAL"], "cyan"),
            ("Queued", counts["QUEUED"], "yellow"),
            ("Running", counts["RUNNING"], "green"),
            ("Completed", counts["COMPLETED"], "bold green"),
            (
                "Failed",
                counts["FAILED"] + counts["CANCELLED"] + counts["TIMEOUT"] + counts["OOM"] + counts["KILLED"],
                "bold red",
            ),
        ]
        for index, (label, count, style) in enumerate(parts):
            if index:
                summary.append(" | ", style="dim")
            summary.append(f"{label}: ", style=style)
            summary.append(str(count), style="bold")

        next_eta = self._next_allocation_eta(experiments)
        if next_eta:
            summary.append(" | ", style="dim")
            summary.append("Next ETA: ", style="magenta")
            summary.append(next_eta)

        high_risk, med_risk = self._timeout_risk_counts(experiments, now=time.time())
        if high_risk or med_risk:
            summary.append(" | ", style="dim")
            summary.append("Time risk: ", style="yellow")
            summary.append(f"H{high_risk}", style="red")
            summary.append("/")
            summary.append(f"M{med_risk}", style="yellow")

        orchestrator = getattr(self.app, "orchestrator", None)
        if bool(getattr(orchestrator, "submissions_paused", False)):
            summary.append(" | ", style="dim")
            summary.append("Submissions paused", style="bold yellow")
        return summary

    def _progress_text(self, experiments: list[dict[str, Any]]) -> Text:
        counts = Counter(exp.get("status") for exp in experiments)
        total = len(experiments)
        done = sum(counts[state] for state in DONE_STATES)
        progress_done, progress_total, progress_fraction = self._overall_run_progress(experiments)
        running_jobs = counts[ExperimentStatus.RUNNING]
        limit_total = self._limit_total()

        line = Text()
        line.append_text(_bar_segment("Completed", done, total, "green", f"{done}/{total}"))
        line.append(" | ", style="dim")
        progress_label = f"{progress_fraction * 100.0:.0f}%" if progress_total else "0%"
        line.append_text(_bar_segment("Progress", progress_done, progress_total, "cyan", progress_label))
        line.append(" | ", style="dim")
        running_total = limit_total or max(running_jobs, 1)
        running_label = f"{running_jobs}/{limit_total}" if limit_total else str(running_jobs)
        line.append_text(_bar_segment("Running", running_jobs, running_total, "yellow", running_label))
        return line

    def _footer_text(self, experiments: list[dict[str, Any]]) -> Text:
        orchestrator = getattr(self.app, "orchestrator", None)
        paused = "paused" if bool(getattr(orchestrator, "submissions_paused", False)) else "active"
        counts = Counter(exp.get("status") for exp in experiments)
        total = len(experiments)
        done = sum(counts[state] for state in DONE_STATES)
        remaining = total - done
        sweep_name = self._infer_sweep_dataset(experiments) or "-"
        sweep_url = self._infer_sweep_url(experiments)
        sweep_part = f"{sweep_name} ({sweep_url})" if sweep_url else sweep_name

        footer = Text()
        footer.append(f"{counts[ExperimentStatus.COMPLETED]} / {total} completed", style="bold green")
        _append_separator(footer)
        footer.append(f"{remaining} left", style="bold yellow")
        _append_separator(footer)
        _append_label_value(footer, "Sweep", sweep_part, label_style="cyan")
        project_label = self._infer_project_label(experiments)
        if project_label:
            _append_separator(footer)
            _append_labeled_text(footer, project_label, label_style="cyan")
        _append_separator(footer)
        _append_label_value(footer, "Updated", dt.now().strftime("%H:%M:%S"), label_style="dim")

        footer.append("\n")
        _append_label_value(
            footer,
            "Submissions",
            paused,
            label_style="yellow",
            value_style="bold yellow" if paused == "paused" else "green",
        )
        _append_separator(footer)
        _append_label_value(footer, "Limits", self._limits_label(), label_style="cyan")
        hpc_label = self._infer_hpc_label(experiments)
        if hpc_label:
            _append_separator(footer)
            _append_labeled_text(footer, hpc_label, label_style="yellow")
        experiment_label = self._infer_experiment_label()
        if experiment_label:
            _append_separator(footer)
            _append_labeled_text(footer, experiment_label, label_style="cyan", value_style="dim")
        slurm_request_label = self._slurm_request_label(experiments)
        if slurm_request_label:
            _append_separator(footer)
            _append_labeled_text(footer, slurm_request_label, label_style="cyan")
        if any(bool(exp.get("oom_recovered")) for exp in experiments):
            _append_separator(footer)
            footer.append("* = recovered from OOM", style="bold red")

        quota_label = self._quota_label(experiments)
        if quota_label:
            footer.append("\n")
            footer.append_text(quota_label)
        return footer

    def _limit_total(self) -> int:
        orchestrator = getattr(self.app, "orchestrator", None)
        limits = getattr(orchestrator, "concurrency_limits", {}) if orchestrator is not None else {}
        total = 0
        for limit in limits.values():
            try:
                total += max(int(limit), 0)
            except (TypeError, ValueError):
                continue
        return total

    def _limits_label(self) -> str:
        orchestrator = getattr(self.app, "orchestrator", None)
        limits = getattr(orchestrator, "concurrency_limits", {}) if orchestrator is not None else {}
        parts = []
        for hpc, limit in sorted(limits.items(), key=lambda item: str(getattr(item[0], "value", item[0]))):
            try:
                if int(limit) <= 0:
                    continue
            except (TypeError, ValueError):
                continue
            parts.append(f"{getattr(hpc, 'value', hpc)}={limit}")
        return ", ".join(parts) if parts else "no active limits"

    def _next_allocation_eta(self, experiments: list[dict[str, Any]]) -> str | None:
        eta_values = [
            float(exp["estimated_start_ts"])
            for exp in experiments
            if exp.get("status") == ExperimentStatus.QUEUED and isinstance(exp.get("estimated_start_ts"), (int, float))
        ]
        if not eta_values:
            return None
        next_eta = min(eta_values)
        now = time.time()
        if next_eta <= now:
            return "now"
        return _format_duration(next_eta - now, show_days=(next_eta - now) >= 86400)

    def _overall_run_progress(self, experiments: list[dict[str, Any]]) -> tuple[float, int, float]:
        eligible = [exp for exp in experiments if exp.get("status") not in FAILED_STATES]
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
            progress_sum += _resolve_display_progress_fraction(exp, allow_zero=True) or 0.0

        fraction = progress_sum / len(eligible)
        return progress_sum, len(eligible), fraction

    def _timeout_risk_counts(self, experiments: list[dict[str, Any]], *, now: float) -> tuple[int, int]:
        high = 0
        medium = 0
        for exp in experiments:
            risk = self._timeout_risk_level(exp, now=now)
            if risk == "high":
                high += 1
            elif risk == "medium":
                medium += 1
        return high, medium

    def _timeout_risk_level(self, exp: dict[str, Any], *, now: float) -> str | None:
        if exp.get("status") != ExperimentStatus.RUNNING:
            return None
        progress = _resolve_progress_fraction(exp)
        if progress is None or progress < 0.20:
            return None
        start_ts = exp.get("start_ts") or exp.get("running_timestamp") or exp.get("last_change_ts")
        if not isinstance(start_ts, (int, float)):
            return None
        runtime_seconds = now - float(start_ts)
        if runtime_seconds < 15 * 60:
            return None
        requested_hours = self._resolve_requested_hours(exp)
        if requested_hours is None or requested_hours <= 0:
            return None
        estimated_total_hours = (runtime_seconds / 3600.0) / progress
        ratio = estimated_total_hours / requested_hours
        if ratio >= 1.0:
            return "high"
        if ratio >= 0.85:
            return "medium"
        return None

    def _resolve_requested_hours(self, exp: dict[str, Any]) -> float | None:
        for key in ("requested_time_hours", "time_hours_override"):
            hours = _coerce_positive_float(exp.get(key))
            if hours is not None:
                return hours
        orchestrator = getattr(self.app, "orchestrator", None)
        return _coerce_positive_float(getattr(orchestrator, "time_hours_override", None))

    @staticmethod
    def _infer_sweep_dataset(experiments: list[dict[str, Any]]) -> str | None:
        datasets = {str(exp.get("dataset_name")) for exp in experiments if exp.get("dataset_name")}
        if len(datasets) == 1:
            return datasets.pop()
        return None

    @staticmethod
    def _infer_sweep_url(experiments: list[dict[str, Any]]) -> str | None:
        for exp in experiments:
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

    def _infer_project_label(self, experiments: list[dict[str, Any]]) -> str | None:
        projects: list[str] = []
        for exp in experiments:
            meta = exp.get("metadata")
            if isinstance(meta, dict):
                project = meta.get("project") or meta.get("tracker_project")
                if project:
                    projects.append(str(project))
                    continue
            project = exp.get("project") or exp.get("tracker_project")
            if project:
                projects.append(str(project))

        orchestrator = getattr(self.app, "orchestrator", None)
        if orchestrator is not None:
            project = getattr(orchestrator, "project", None) or getattr(orchestrator, "tracker_project", None)
            if project:
                projects.append(str(project))
            exp_file = getattr(orchestrator, "experiment_file", None)
            if exp_file:
                try:
                    data = load_yaml(str(exp_file))
                    for exp in data.get("experiments", []):
                        meta = exp.get("metadata")
                        if isinstance(meta, dict):
                            project = meta.get("project") or meta.get("tracker_project")
                            if project:
                                projects.append(str(project))
                                continue
                        project = exp.get("project") or exp.get("tracker_project")
                        if project:
                            projects.append(str(project))
                except Exception:
                    pass

        unique = sorted({project for project in projects if project})
        if not unique:
            return None
        if len(unique) == 1:
            return f"Project: {unique[0]}"
        return f"Projects: {', '.join(unique)}"

    @staticmethod
    def _infer_hpc_label(experiments: list[dict[str, Any]]) -> str | None:
        hpcs = []
        for exp in experiments:
            hpc = exp.get("hpc_assignment")
            if hpc is None:
                continue
            name = getattr(hpc, "name", None) or getattr(hpc, "value", None) or str(hpc)
            if name:
                hpcs.append(str(name))
        unique = sorted({hpc for hpc in hpcs if hpc})
        if not unique:
            return None
        if len(unique) == 1:
            return f"Host: {unique[0]}"
        return f"HPCs: {', '.join(unique)}"

    def _infer_experiment_label(self) -> str | None:
        orchestrator = getattr(self.app, "orchestrator", None)
        exp_file = getattr(orchestrator, "experiment_file", None) if orchestrator is not None else None
        if not exp_file:
            return None
        try:
            return f"Experiment: {exp_file.stem}"
        except Exception:
            return None

    def _slurm_request_label(self, experiments: list[dict[str, Any]]) -> str | None:
        orchestrator = getattr(self.app, "orchestrator", None)
        requested_hours = _coerce_positive_int(getattr(orchestrator, "time_hours_override", None))
        requested_ram = _coerce_positive_int(getattr(orchestrator, "memory_gb_override", None))
        requested_gpus = _coerce_positive_int(getattr(orchestrator, "max_gpus_per_job", None))

        if requested_hours is None:
            requested_hours = _first_positive_int(experiments, "requested_time_hours")
        if requested_ram is None:
            requested_ram = _first_positive_int(experiments, "requested_ram_gb")
        if requested_gpus is None:
            requested_gpus = _first_positive_int(experiments, "requested_gpu_count")

        if requested_hours is None or requested_ram is None or requested_gpus is None:
            fallback_hpc = self._first_active_hpc(experiments)
            cfg = HPC_CONFIGS.get(fallback_hpc) if fallback_hpc is not None else None
            if cfg is not None:
                if requested_gpus is None:
                    requested_gpus = _coerce_positive_int(cfg.gpu_count)
                if requested_hours is None:
                    requested_hours = _coerce_positive_int(cfg.base_time_hours)
                if requested_ram is None:
                    if cfg.mem_per_gpu_gb is not None and requested_gpus is not None:
                        requested_ram = _coerce_positive_int(cfg.mem_per_gpu_gb * requested_gpus)
                    else:
                        requested_ram = _coerce_positive_int(cfg.base_memory_gb)

        if requested_hours is None and requested_ram is None and requested_gpus is None:
            return None
        hours_txt = f"{requested_hours}h" if requested_hours is not None else "auto"
        ram_txt = f"{requested_ram}G" if requested_ram is not None else "auto"
        gpu_txt = f"{requested_gpus}" if requested_gpus is not None else "auto"
        return f"Slurm: h={hours_txt} ram={ram_txt} gpu={gpu_txt}"

    def _first_active_hpc(self, experiments: list[dict[str, Any]]) -> HPCType | None:
        active_hpcs = sorted(self._active_hpcs(experiments), key=lambda hpc: hpc.value)
        return active_hpcs[0] if active_hpcs else None

    def _active_hpcs(self, experiments: list[dict[str, Any]]) -> set[HPCType]:
        active: set[HPCType] = set()
        for exp in experiments:
            hpc = exp.get("hpc_assignment")
            if isinstance(hpc, HPCType):
                active.add(hpc)
        orchestrator = getattr(self.app, "orchestrator", None)
        limits = getattr(orchestrator, "concurrency_limits", None) if orchestrator is not None else None
        if isinstance(limits, dict):
            for hpc, limit in limits.items():
                if not isinstance(hpc, HPCType):
                    continue
                try:
                    limit_val = int(limit)
                except (TypeError, ValueError):
                    continue
                if limit_val > 0:
                    active.add(hpc)
        return active

    def _quota_label(self, experiments: list[dict[str, Any]]) -> Text | None:
        active_hpcs = sorted(self._active_hpcs(experiments), key=lambda hpc: hpc.value)
        worst_case_by_hpc = self._estimate_orchestration_worst_case_hours_per_cluster(experiments)
        lines: list[Text] = []
        hpcs_without_provider: list[str] = []
        for hpc_type in active_hpcs:
            provider = get_quota_provider(hpc_type)
            if provider is None:
                hpcs_without_provider.append(hpc_type.value)
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
        if lines:
            return _join_text(lines, separator="\n")
        if hpcs_without_provider:
            text = Text()
            _append_label_value(
                text, "Quota", f"no provider for {', '.join(hpcs_without_provider)}", label_style="yellow"
            )
            return text
        if active_hpcs:
            text = Text()
            _append_label_value(text, "Quota", ", ".join(hpc.value for hpc in active_hpcs), label_style="yellow")
            return text
        return None

    def _get_quota_snapshot(self, hpc_type: HPCType) -> QuotaSnapshot | None:
        provider = get_quota_provider(hpc_type)
        if provider is None:
            return None
        now = time.time()
        cached = self._quota_cache.get(hpc_type)
        if cached is not None:
            cache_ts, cache_data = cached
            if now - cache_ts < GPU_QUOTA_POLL_INTERVAL_SECONDS:
                return cache_data

        orchestrator = getattr(self.app, "orchestrator", None)
        connection_manager = getattr(orchestrator, "connection_manager", None) if orchestrator is not None else None
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

    def _quota_period_footer_status(
        self, provider: QuotaProvider, snapshot: QuotaSnapshot | None
    ) -> tuple[int, str, float] | None:
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
        elapsed_pct = _allocation_period_elapsed_pct(period_start, period_end, today)
        days_left = max((period_end - today).days, 0)
        return days_left, period_end.strftime("%d-%m-%y"), elapsed_pct

    @staticmethod
    def _quota_period_segment(period_status: tuple[int, str, float], pace_delta_pp: float | None = None) -> Text:
        days_left, period_end_txt, period_elapsed_pct = period_status
        segment = Text()
        _append_label_value(segment, "Period", f"{days_left}d", label_style="yellow", value_style="bold yellow")
        segment.append(" left (")
        segment.append(f"{period_elapsed_pct:.1f}%", style="bold yellow")
        segment.append(" elapsed")
        if pace_delta_pp is None:
            segment.append("; ends ")
            segment.append(period_end_txt, style="bold yellow")
            segment.append(")")
            return segment
        segment.append(", ")
        segment.append(f"{pace_delta_pp:+.1f}pp", style="bold yellow")
        segment.append("; ends ")
        segment.append(period_end_txt, style="bold yellow")
        segment.append(")")
        return segment

    def _render_quota_line(
        self,
        *,
        hpc_type: HPCType,
        provider: QuotaProvider,
        snapshot: QuotaSnapshot | None,
        worst_case_hours: float | None,
    ) -> Text:
        period_status = self._quota_period_footer_status(provider, snapshot)
        period_segment = self._quota_period_segment(period_status) if period_status is not None else None
        provider_label = str(getattr(provider, "resource_label", "Quota"))
        provider_cluster = getattr(provider, "cluster_name", hpc_type.value)
        quota_line = Text()
        if snapshot is None:
            hint = str(getattr(provider, "unavailable_hint", "quota probe unavailable"))
            _append_label_value(quota_line, provider_label, provider_cluster, label_style="yellow")
            quota_line.append(f" unavailable ({hint})")
            if period_segment is not None:
                _append_separator(quota_line)
                quota_line.append_text(period_segment)
            return quota_line

        if snapshot.limit <= 0.0:
            _append_label_value(quota_line, snapshot.resource_label, snapshot.cluster_name, label_style="yellow")
            quota_line.append(" unavailable (invalid quota limit)")
            if period_segment is not None:
                _append_separator(quota_line)
                quota_line.append_text(period_segment)
            return quota_line

        pace_delta_pp = snapshot.used_pct - period_status[2] if period_status is not None else None
        period_with_delta = (
            self._quota_period_segment(period_status, pace_delta_pp=pace_delta_pp)
            if period_status is not None
            else None
        )
        used_total = f"{_fmt_quota_amount(snapshot.used)}/{_fmt_quota_amount(snapshot.limit)}{snapshot.unit}"
        quota_line.append(f"{snapshot.resource_label}: ", style="yellow")
        quota_line.append(f"{snapshot.cluster_name} ")
        quota_line.append(f"{_fmt_quota_amount(snapshot.remaining)}{snapshot.unit}", style="bold yellow")
        quota_line.append(" left (")
        quota_line.append(f"{snapshot.used_pct:.1f}%", style="bold yellow")
        quota_line.append(" used; ")
        quota_line.append(used_total, style="bold yellow")
        quota_line.append(")")
        if snapshot.worst_case_unit == "gpu_hours" and worst_case_hours is not None:
            pct_left = (worst_case_hours / snapshot.remaining * 100.0) if snapshot.remaining > 0 else 0.0
            _append_separator(quota_line)
            quota_line.append("Orch worst-case: ", style="yellow")
            quota_line.append(f"{_fmt_quota_amount(worst_case_hours)}{snapshot.unit}", style="bold yellow")
            quota_line.append(" (")
            quota_line.append(f"{pct_left:.1f}%", style="bold yellow")
            quota_line.append(" of left)")
        if period_with_delta is not None:
            _append_separator(quota_line)
            quota_line.append_text(period_with_delta)
        return quota_line

    def _estimate_orchestration_worst_case_hours_per_cluster(
        self, experiments: list[dict[str, Any]]
    ) -> dict[HPCType, float]:
        worst_case: dict[HPCType, float] = {}
        seen_unfinished_hpcs: set[HPCType] = set()
        for exp in experiments:
            if exp.get("status") in DONE_STATES:
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

    def _resolve_effective_request_hours_and_gpus(self, exp: dict[str, Any]) -> tuple[int | None, int | None]:
        requested_hours = _coerce_positive_int(exp.get("requested_time_hours"))
        requested_gpus = _coerce_positive_int(exp.get("requested_gpu_count"))
        orchestrator = getattr(self.app, "orchestrator", None)
        if orchestrator is not None:
            if requested_hours is None:
                requested_hours = _coerce_positive_int(getattr(orchestrator, "time_hours_override", None))
            if requested_gpus is None:
                requested_gpus = _coerce_positive_int(getattr(orchestrator, "max_gpus_per_job", None))
        if requested_hours is not None and requested_gpus is not None:
            return requested_hours, requested_gpus

        hpc = exp.get("hpc_assignment")
        cfg = HPC_CONFIGS.get(hpc) if isinstance(hpc, HPCType) else None
        if cfg is not None:
            if requested_hours is None:
                requested_hours = _coerce_positive_int(getattr(cfg, "base_time_hours", None))
            if requested_gpus is None:
                requested_gpus = _coerce_positive_int(getattr(cfg, "gpu_count", None))
        return requested_hours, requested_gpus


def _status_text(status: object) -> str:
    if isinstance(status, ExperimentStatus):
        return status.name
    return str(status or "").upper()


def _bar_segment(label: str, completed: float, total: float, style: str, value_text: str) -> Text:
    fraction = 0.0 if total <= 0 else max(0.0, min(float(completed) / float(total), 1.0))
    filled = int(round(fraction * TOP_BAR_WIDTH))
    segment = Text(f"{label} ")
    segment.append("█" * filled, style=style)
    segment.append("░" * (TOP_BAR_WIDTH - filled), style="dim")
    segment.append(f" {value_text}")
    return segment


def _append_separator(text: Text) -> None:
    text.append(" | ", style="dim")


def _append_label_value(
    text: Text, label: str, value: str, *, label_style: str, value_style: str | None = None
) -> None:
    text.append(f"{label}: ", style=label_style)
    text.append(value, style=value_style)


def _append_labeled_text(text: Text, labeled_text: str, *, label_style: str, value_style: str | None = None) -> None:
    if ":" not in labeled_text:
        text.append(labeled_text, style=value_style)
        return
    label, value = labeled_text.split(":", 1)
    _append_label_value(text, label, value.lstrip(), label_style=label_style, value_style=value_style)


def _join_text(parts: list[Text], *, separator: str) -> Text:
    joined = Text()
    for index, part in enumerate(parts):
        if index:
            joined.append(separator)
        joined.append_text(part)
    return joined


def _coerce_progress_fraction(current_val: object, max_val: object, *, allow_zero: bool = False) -> float | None:
    if current_val is None or max_val in (None, 0):
        return None
    try:
        current_f = float(current_val)
        max_f = float(max_val)
    except (TypeError, ValueError):
        return None
    if max_f <= 0.0:
        return None
    fraction = current_f / max_f
    if fraction <= 0.0:
        return 0.0 if allow_zero else None
    return min(fraction, 1.0)


def _resolve_progress_fraction(exp: dict[str, Any], *, allow_zero: bool = False) -> float | None:
    fractions: list[float] = []
    for current_key, max_key in (("current_step", "max_steps"), ("current_epoch", "max_epochs")):
        fraction = _coerce_progress_fraction(exp.get(current_key), exp.get(max_key), allow_zero=allow_zero)
        if fraction is not None:
            fractions.append(fraction)
    return max(fractions) if fractions else None


def _resolve_display_progress_fraction(exp: dict[str, Any], *, allow_zero: bool = False) -> float | None:
    for current_key, max_key in (("current_step", "max_steps"), ("current_epoch", "max_epochs")):
        fraction = _coerce_progress_fraction(exp.get(current_key), exp.get(max_key), allow_zero=allow_zero)
        if fraction is not None:
            return fraction
    return None


def _format_duration(seconds: float, show_days: bool = False) -> str:
    if seconds < 0:
        return "?"
    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if show_days and days > 0:
        return f"{days}d{hours}h"
    if hours > 0:
        return f"{hours}h{minutes:02d}m"
    if minutes > 0:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _coerce_positive_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0.0 else None


def _coerce_positive_int(value: object) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _first_positive_int(experiments: list[dict[str, Any]], key: str) -> int | None:
    for exp in experiments:
        value = _coerce_positive_int(exp.get(key))
        if value is not None:
            return value
    return None


def _fmt_quota_amount(value: float) -> str:
    if value >= 100:
        return f"{value:,.0f}"
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _allocation_period_elapsed_pct(start: date, end: date, today: date) -> float:
    total_days = max((end - start).days, 1)
    elapsed_days = max(min((today - start).days, total_days), 0)
    return (elapsed_days / float(total_days)) * 100.0


__all__ = ["HomeScreen"]

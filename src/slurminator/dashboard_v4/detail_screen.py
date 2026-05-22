"""Per-run detail screen for dashboard v4."""

from __future__ import annotations

import time
from collections.abc import Mapping
from enum import Enum
from typing import Any

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.screen import Screen
from textual.widgets import Footer, Header, Static

from slurminator.dashboard_v4.keystrokes import DETAIL_BINDINGS
from slurminator.experiments import ExperimentStatus


class PerRunDetailScreen(Screen[None]):
    """Show full ledger and status details for one run."""

    BINDINGS = DETAIL_BINDINGS

    def __init__(self, exp: dict[str, Any]) -> None:
        super().__init__()
        self.exp = exp
        self._last_detail_text = ""

    def compose(self) -> ComposeResult:
        """Compose the detail screen."""
        yield Header()
        with VerticalScroll(id="detail-scroll"):
            yield Static("", id="detail-content")
        yield Footer()

    def on_mount(self) -> None:
        """Force-load history and render the current details."""
        orchestrator = getattr(self.app, "orchestrator", None)
        if orchestrator is not None and hasattr(orchestrator, "force_read_full_history"):
            orchestrator.force_read_full_history(self.exp)
        self.set_interval(getattr(self.app, "refresh_interval", 1.0), self.refresh_from_orchestrator)
        self.refresh_from_orchestrator(force=True)

    def refresh_from_orchestrator(self, *, force: bool = False) -> None:
        """Refresh details from the latest dashboard snapshot."""
        latest = self._latest_snapshot_exp()
        if latest is not None:
            self.exp = latest
        detail_text = render_detail_text(self.exp)
        if force or detail_text != self._last_detail_text:
            self._last_detail_text = detail_text
            self.query_one("#detail-content", Static).update(detail_text)

    def action_refresh(self) -> None:
        """Force a history read and redraw."""
        orchestrator = getattr(self.app, "orchestrator", None)
        if orchestrator is not None and hasattr(orchestrator, "force_read_full_history"):
            orchestrator.force_read_full_history(self.exp)
        self.refresh_from_orchestrator(force=True)

    def action_scroll_up(self) -> None:
        """Scroll details upward."""
        self.query_one("#detail-scroll", VerticalScroll).scroll_up(animate=False)

    def action_scroll_down(self) -> None:
        """Scroll details downward."""
        self.query_one("#detail-scroll", VerticalScroll).scroll_down(animate=False)

    def _latest_snapshot_exp(self) -> dict[str, Any] | None:
        experiment_id = self.exp.get("experiment_id")
        for exp in self.app.get_dashboard_snapshot():
            if exp.get("experiment_id") == experiment_id:
                return exp
        return None


def render_detail_text(exp: Mapping[str, Any]) -> str:
    """Render detail sections as plain text for a Textual Static panel."""
    sections = [
        ("Run", _run_lines(exp)),
        ("Metrics", _metric_lines(exp)),
        ("Sweep Parameters", _sweep_lines(exp)),
        ("sacct Snapshot", _sacct_lines(exp)),
        ("Links", _link_lines(exp)),
        ("Git Provenance", _git_lines(exp)),
        ("Notes", _note_lines(exp)),
    ]
    chunks = []
    for title, lines in sections:
        chunks.append(f"{title}\n{'-' * len(title)}")
        chunks.extend(lines or ["No data yet"])
    return "\n\n".join(chunks)


def _run_lines(exp: Mapping[str, Any]) -> list[str]:
    return [
        f"experiment_id: {_value(exp.get('experiment_id'))}",
        f"status: {_enum_value(exp.get('status'))}",
        f"job_id: {_value(exp.get('job_id'))}",
        f"cluster: {_enum_value(exp.get('hpc_assignment'))}",
        f"walltime: requested={_requested_walltime(exp)} used={_used_walltime(exp)}",
        f"gpu_count: {_value(exp.get('requested_gpu_count') or exp.get('gpu_count'))}",
        f"dataset: {_value(exp.get('dataset_name') or exp.get('dataset'))}",
        f"output_dir: {_value(exp.get('output_dir'))}",
    ]


def _metric_lines(exp: Mapping[str, Any]) -> list[str]:
    metrics = exp.get("all_metrics")
    if not isinstance(metrics, Mapping) or not metrics:
        history = exp.get("history")
        if isinstance(history, list) and history:
            latest_metrics = history[-1].get("metrics") if isinstance(history[-1], Mapping) else None
            metrics = latest_metrics if isinstance(latest_metrics, Mapping) else {}
    if not isinstance(metrics, Mapping) or not metrics:
        return []
    return [f"{key}: {_value(value)}" for key, value in sorted(metrics.items(), key=lambda item: str(item[0]))]


def _sweep_lines(exp: Mapping[str, Any]) -> list[str]:
    keys = ("sweep_params", "config", "config_profile", "case", "named_case", "seed", "task_type")
    lines = [f"{key}: {_value(exp.get(key))}" for key in keys if exp.get(key) is not None]
    overrides = exp.get("resource_overrides")
    if isinstance(overrides, Mapping) and overrides:
        lines.append("resource_overrides:")
        lines.extend(f"  {key}: {_value(value)}" for key, value in sorted(overrides.items()))
    return lines


def _sacct_lines(exp: Mapping[str, Any]) -> list[str]:
    snapshot = exp.get("sacct_snapshot")
    if isinstance(snapshot, Mapping) and snapshot:
        return [f"{key}: {_value(value)}" for key, value in sorted(snapshot.items(), key=lambda item: str(item[0]))]

    keys = (
        "scheduler_state",
        "slurm_state",
        "last_change_ts",
        "queued_timestamp",
        "running_timestamp",
        "completed_timestamp",
        "cancelled_timestamp",
    )
    return [f"{key}: {_value(exp.get(key))}" for key in keys if exp.get(key) is not None]


def _link_lines(exp: Mapping[str, Any]) -> list[str]:
    links: dict[str, Any] = {}
    for field in ("links", "status_links"):
        value = exp.get(field)
        if isinstance(value, Mapping):
            links.update({str(key): item for key, item in value.items()})
    for field in ("wandb_run_url", "wandb_url", "tracker_url", "log_url"):
        if exp.get(field) is not None:
            links[field] = exp[field]
    if not links:
        return []
    return [f"{key}: {_value(value)}" for key, value in sorted(links.items())]


def _git_lines(exp: Mapping[str, Any]) -> list[str]:
    provenance = exp.get("git_sha_at_submission")
    if isinstance(provenance, Mapping) and provenance:
        return [f"{key}: {_value(value)}" for key, value in sorted(provenance.items())]
    if provenance:
        return [str(provenance)]
    return []


def _note_lines(exp: Mapping[str, Any]) -> list[str]:
    lines = []
    for key in ("notes", "note", "annotations"):
        value = exp.get(key)
        if value:
            lines.append(f"{key}: {_value(value)}")
    return lines


def _requested_walltime(exp: Mapping[str, Any]) -> str:
    requested = exp.get("requested_time_hours") or exp.get("time_hours_override") or exp.get("job_time_hours")
    return f"{requested}h" if requested is not None else "No data yet"


def _used_walltime(exp: Mapping[str, Any]) -> str:
    for key in ("walltime_used", "elapsed", "elapsed_time", "runtime_seconds"):
        value = exp.get(key)
        if value is not None:
            return _value(value)
    snapshot = exp.get("sacct_snapshot")
    if isinstance(snapshot, Mapping):
        for key in ("Elapsed", "ElapsedRaw", "elapsed"):
            if snapshot.get(key) is not None:
                return _value(snapshot[key])
    started = exp.get("running_timestamp")
    if started is None:
        return "No data yet"
    ended = exp.get("completed_timestamp") or exp.get("cancelled_timestamp") or time.time()
    try:
        seconds = max(float(ended) - float(started), 0.0)
    except (TypeError, ValueError):
        return "No data yet"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.2f}h"


def _value(value: object) -> str:
    if value is None:
        return "No data yet"
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _enum_value(value: object) -> str:
    if value == ExperimentStatus.CANCELLED:
        return "cancelled"
    if isinstance(value, Enum):
        return str(value.value)
    return _value(value)


__all__ = ["PerRunDetailScreen", "render_detail_text"]

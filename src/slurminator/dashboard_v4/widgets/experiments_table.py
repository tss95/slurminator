"""Experiment table widget for the Textual dashboard."""

from __future__ import annotations

import time
from enum import Enum
from typing import Any

from textual.widgets import DataTable


class ExperimentsTable(DataTable):
    """Main dashboard table for experiment rows."""

    BASE_COLUMNS = ("ID", "Dataset", "HPC", "State", "Progress", "Primary", "Secondary", "Queue/DT")

    def on_mount(self) -> None:
        """Initialize stable table columns."""
        self.cursor_type = "row"
        self.zebra_stripes = True
        self.show_cursor = True
        self._build_columns()

    def update_experiments(self, experiments: list[dict[str, Any]], *, show_sparkline: bool = False) -> None:
        """Replace table rows with the latest experiment snapshot."""
        cursor_row = min(max(self.cursor_row, 0), max(len(experiments) - 1, 0)) if experiments else 0
        self.clear(columns=True)
        self._build_columns(show_sparkline=show_sparkline)
        for exp in experiments:
            cells = [
                _text(exp.get("experiment_id"), "-"),
                _text(exp.get("dataset_name") or exp.get("dataset") or exp.get("config"), "-"),
                _format_enum(exp.get("hpc_assignment")),
                _format_status(exp.get("status")),
                _format_progress(exp),
                _format_metric(
                    exp.get("target_metric_name"), exp.get("target_metric_value") or exp.get("metric_value")
                ),
                _format_metric(exp.get("secondary_metric_name"), exp.get("secondary_metric_value")),
            ]
            if show_sparkline:
                cells.append("")
            cells.append(_format_queue_delta(exp))
            self.add_row(*cells, key=str(exp.get("experiment_id", len(self.rows))))
        if experiments:
            self.move_cursor(row=cursor_row, column=0, animate=False)

    def selected_experiment(self, experiments: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Return the row under the table cursor."""
        if not experiments:
            return None
        row = min(max(self.cursor_row, 0), len(experiments) - 1)
        return experiments[row]

    def _build_columns(self, *, show_sparkline: bool = False) -> None:
        columns = list(self.BASE_COLUMNS)
        if show_sparkline:
            columns.insert(6, "Trajectory")
        self.add_columns(*columns)


def _text(value: object, default: str = "") -> str:
    if value is None:
        return default
    text = str(value)
    return text if text else default


def _format_enum(value: object) -> str:
    if isinstance(value, Enum):
        return str(value.value)
    return _text(value, "-")


def _format_status(value: object) -> str:
    text = _format_enum(value)
    return text.upper() if text != "-" else text


def _format_progress(exp: dict[str, Any]) -> str:
    current = exp.get("current_epoch", exp.get("current_step"))
    total = exp.get("max_epochs", exp.get("max_steps"))
    if current is not None and total:
        try:
            pct = float(current) / float(total) * 100.0
        except (TypeError, ValueError, ZeroDivisionError):
            return f"{current}/{total}"
        return f"{current}/{total} {pct:.0f}%"
    return "-"


def _format_metric(name: object, value: object) -> str:
    if value is None:
        return "-"
    label = str(name) if name else "metric"
    if isinstance(value, float):
        return f"{label}={value:.4g}"
    return f"{label}={value}"


def _format_queue_delta(exp: dict[str, Any]) -> str:
    queued_at = exp.get("queued_timestamp")
    running_at = exp.get("running_timestamp")
    completed_at = exp.get("completed_timestamp")
    start = running_at or queued_at
    end = completed_at or time.time()
    if start is None:
        return "-"
    try:
        seconds = max(float(end) - float(start), 0.0)
    except (TypeError, ValueError):
        return "-"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


__all__ = ["ExperimentsTable"]

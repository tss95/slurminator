"""Experiment table widget for the Textual dashboard."""

from __future__ import annotations

import time
from enum import Enum
from typing import Any

from rich.text import Text
from textual.widgets import DataTable

from slurminator.dashboard_v4.widgets.sparkline import SparklineThresholds, render_sparkline
from slurminator.experiments import ExperimentStatus

METRIC_VALUE_PRECISION = 4
PROGRESS_PERCENTAGE_WIDTH = 5

FAILED_STATES = {
    ExperimentStatus.FAILED,
    ExperimentStatus.CANCELLED,
    ExperimentStatus.TIMEOUT,
    ExperimentStatus.OOM,
    ExperimentStatus.KILLED,
}


class ExperimentsTable(DataTable):
    """Main dashboard table for experiment rows."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._row_experiments: list[dict[str, Any]] = []

    def on_mount(self) -> None:
        """Initialize stable table columns."""
        self.cursor_type = "row"
        self.zebra_stripes = True
        self.show_cursor = True
        self._build_columns()

    def update_experiments(
        self,
        experiments: list[dict[str, Any]],
        *,
        show_sparkline: bool = False,
        sparkline_thresholds: SparklineThresholds | object | None = None,
    ) -> None:
        """Replace table rows with the latest experiment snapshot."""
        previous_key = None
        if self._row_experiments and 0 <= self.cursor_row < len(self._row_experiments):
            previous_key = _row_key(self._row_experiments[self.cursor_row], self.cursor_row)
        rows = _sorted_experiments(experiments)
        cursor_row = _resolve_cursor_row(rows, previous_key, self.cursor_row)
        self.clear(columns=True)
        self._build_columns(
            show_sparkline=show_sparkline,
            primary_label=_metric_column_label(rows, "target_metric_name", "Primary"),
            secondary_label=_metric_column_label(rows, "secondary_metric_name", "Secondary"),
        )
        self._row_experiments = rows
        for exp in rows:
            primary_name = exp.get("target_metric_name")
            primary_value = (
                exp.get("target_metric_value")
                if exp.get("target_metric_value") is not None
                else exp.get("metric_value")
            )
            cells = [
                _text(exp.get("experiment_id"), "-"),
                _text(exp.get("dataset_name") or exp.get("dataset") or exp.get("config"), "-"),
                _format_enum(exp.get("hpc_assignment")),
                _format_status(exp.get("status")),
                _format_progress(exp),
                _format_metric(exp, primary_name, primary_value),
            ]
            if show_sparkline:
                cells.append(_format_sparkline(exp, primary_name, thresholds=sparkline_thresholds))
            cells.append(_format_metric(exp, exp.get("secondary_metric_name"), exp.get("secondary_metric_value")))
            cells.append(_format_queue_delta(exp))
            self.add_row(*cells, key=str(exp.get("experiment_id", len(self.rows))))
        if rows:
            self.move_cursor(row=cursor_row, column=0, animate=False)

    def selected_experiment(self, experiments: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Return the row under the table cursor."""
        rows = self._row_experiments or _sorted_experiments(experiments)
        if not rows:
            return None
        row = min(max(self.cursor_row, 0), len(rows) - 1)
        return rows[row]

    def _build_columns(
        self, *, show_sparkline: bool = False, primary_label: str = "Primary", secondary_label: str = "Secondary"
    ) -> None:
        columns = ["ID", "Dataset", "HPC", "State", "Progress", primary_label, secondary_label, "Queue/DT"]
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


def _format_status(value: object) -> str | Text:
    text = _format_enum(value)
    label = text.upper() if text != "-" else text
    status = value if isinstance(value, ExperimentStatus) else None
    if status == ExperimentStatus.CANCELLED:
        label = "CANCELLED"
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
    }.get(status)
    return Text(label, style=style) if style else label


def _format_progress(exp: dict[str, Any]) -> str:
    current_step = exp.get("current_step")
    max_steps = exp.get("max_steps")
    if current_step is not None and max_steps not in (None, 0):
        progress = _format_progress_values(current_step, max_steps)
        speed = exp.get("it_per_sec_backbone")
        if isinstance(speed, (int, float)) and speed > 0:
            return f"{progress} @ {speed:.2f}it/s"
        return progress

    current_epoch = exp.get("current_epoch")
    max_epochs = exp.get("max_epochs")
    if current_epoch is not None and max_epochs not in (None, 0):
        return _format_progress_values(current_epoch, max_epochs)
    return "?"


def _format_progress_values(current: object, total: object) -> str:
    try:
        current_value = int(current)
    except Exception:
        current_value = current
    try:
        total_value = int(total)
    except Exception:
        total_value = total
    try:
        pct = (float(current) / float(total)) * 100.0
    except (TypeError, ValueError, ZeroDivisionError):
        return f"{current_value}/{total_value}"
    return f"{current_value}/{total_value} {pct:{PROGRESS_PERCENTAGE_WIDTH}.1f}%"


def _format_metric(exp: dict[str, Any], name: object, value: object) -> str | Text:
    metric_key = str(name) if name else ""
    metric_info = _metric_info_for(exp, metric_key)
    current = _coerce_float(value)
    best = _lookup_best_metric_value(exp, metric_info)
    if current is None and value is not None:
        return str(value)
    if current is None and best is None:
        return "-"
    style = _metric_color(current, metric_info)
    combined = _format_metric_pair(current, best, metric_info=metric_info)
    return Text(combined, style=style) if style else combined


def _metric_info_for(exp: dict[str, Any], metric_key: str) -> dict[str, Any] | None:
    metric_info = exp.get("display_metric_info") or exp.get("metric_info") or {}
    info = metric_info.get(metric_key) if isinstance(metric_info, dict) else None
    return info if isinstance(info, dict) else None


def _lookup_best_metric_value(exp: dict[str, Any], metric_info: dict[str, Any] | None) -> float | None:
    if not metric_info:
        return None
    best_key = metric_info.get("best_key")
    if not best_key:
        return None
    all_metrics = exp.get("all_metrics", {})
    for source in (exp, all_metrics if isinstance(all_metrics, dict) else {}):
        if best_key in source:
            return _coerce_float(source[best_key])
    return None


def _metric_color(value: float | None, metric_info: dict[str, Any] | None) -> str | None:
    if value is None or not isinstance(metric_info, dict):
        return None
    threshold = metric_info.get("threshold")
    if not isinstance(threshold, (int, float)):
        return None
    higher_better = metric_info.get("higher_better", True)
    if higher_better is False:
        return "green" if value <= threshold else "red"
    return "green" if value >= threshold else "red"


def _format_metric_pair(current: float | None, best: float | None, *, metric_info: dict[str, Any] | None = None) -> str:
    value_format = metric_info.get("value_format") if isinstance(metric_info, dict) else None
    current_text = _format_metric_number(current, value_format=value_format)
    if best is None:
        return current_text
    return f"{current_text} ({_format_metric_number(best, value_format=value_format)})"


def _format_metric_number(value: float | None, *, value_format: str | None = None) -> str:
    if value is None:
        return "-"
    if value_format == "integer":
        return f"{int(round(value))}"
    return f"{value:.{METRIC_VALUE_PRECISION}f}"


def _coerce_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sorted_experiments(experiments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = list(experiments)

    def sort_key(exp: dict[str, Any]) -> tuple[int, int, float, float]:
        status = exp.get("status")
        if status == ExperimentStatus.RUNNING:
            status_priority = 0
        elif status == ExperimentStatus.COMPLETED:
            status_priority = 1
        else:
            status_priority = 2
        metric_info = _metric_info_for(exp, str(exp.get("target_metric_name") or ""))
        sort_value = _coerce_float(
            exp.get("target_metric_value") if exp.get("target_metric_value") is not None else exp.get("metric_value")
        )
        has_metric = sort_value is not None
        metric_sort = _metric_sort_value(sort_value, metric_info)
        last_ts = _coerce_float(exp.get("last_change_ts")) or 0.0
        return status_priority, 0 if has_metric else 1, metric_sort, -last_ts

    rows.sort(key=sort_key)
    return rows


def _metric_sort_value(value: float | None, metric_info: dict[str, Any] | None) -> float:
    if value is None:
        return float("inf")
    higher_better = True if not isinstance(metric_info, dict) else metric_info.get("higher_better", True)
    return -value if higher_better is not False else value


def _resolve_cursor_row(rows: list[dict[str, Any]], previous_key: str | None, previous_row: int) -> int:
    if not rows:
        return 0
    if previous_key is not None:
        for index, exp in enumerate(rows):
            if _row_key(exp, index) == previous_key:
                return index
    return min(max(previous_row, 0), len(rows) - 1)


def _row_key(exp: dict[str, Any], fallback: int) -> str:
    return str(exp.get("experiment_id", fallback))


def _metric_column_label(rows: list[dict[str, Any]], metric_name_key: str, fallback: str) -> str:
    for exp in rows:
        metric_key = exp.get(metric_name_key)
        if metric_key:
            return _metric_header(str(metric_key), _metric_info_for(exp, str(metric_key)))
    return fallback


def _metric_header(metric_key: str, metric_info: dict[str, Any] | None) -> str:
    shortform = metric_info.get("shortform") if isinstance(metric_info, dict) else None
    if shortform:
        return str(shortform)
    return _abbr_metric(metric_key)


def _abbr_metric(metric_key: str | None) -> str:
    if not metric_key:
        return "-"
    name = str(metric_key)
    if "/" in name:
        name = name.split("/")[-1]
    replacements = {"accuracy": "acc", "balanced_accuracy": "bacc", "validation": "val"}
    return replacements.get(name, name)


def _format_sparkline(exp: dict[str, Any], primary_metric: object, *, thresholds: SparklineThresholds | object | None):
    for metric_name in _sparkline_metric_candidates(exp, primary_metric):
        values = _history_metric_values(exp, metric_name)
        if len(values) >= 2:
            return render_sparkline(
                values, width=20, higher_better=_higher_better(exp, metric_name), thresholds=thresholds
            )
    return "-"


def _sparkline_metric_candidates(exp: dict[str, Any], primary_metric: object) -> list[str]:
    candidates: list[str] = []

    def add(value: object) -> None:
        if value is None:
            return
        metric_name = str(value)
        if metric_name and metric_name not in candidates:
            candidates.append(metric_name)

    primary_name = str(primary_metric) if primary_metric else ""
    add(primary_name)

    metric_info = exp.get("display_metric_info") or exp.get("metric_info") or {}
    if isinstance(metric_info, dict):
        for metric_key, info in metric_info.items():
            if metric_key == primary_name:
                add(metric_key)
                continue
            shortform = info.get("shortform") if isinstance(info, dict) else None
            if shortform and str(shortform) == primary_name:
                add(metric_key)

    add(exp.get("target_metric_name"))
    add(exp.get("secondary_metric_name"))
    for metric_key in _history_metric_keys(exp):
        add(metric_key)
    return candidates


def _history_metric_keys(exp: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    history = exp.get("history")
    if not isinstance(history, list):
        return keys
    for entry in history:
        metrics = entry.get("metrics") if isinstance(entry, dict) else None
        if not isinstance(metrics, dict):
            continue
        for metric_key in metrics:
            metric_name = str(metric_key)
            if metric_name not in keys:
                keys.append(metric_name)
    return keys


def _history_metric_values(exp: dict[str, Any], metric_name: str) -> list[float]:
    if not metric_name:
        return []
    values: list[float] = []
    history = exp.get("history")
    if not isinstance(history, list):
        return values
    for entry in history:
        metrics = entry.get("metrics") if isinstance(entry, dict) else None
        if not isinstance(metrics, dict) or metric_name not in metrics:
            continue
        try:
            values.append(float(metrics[metric_name]))
        except (TypeError, ValueError):
            continue
    return values


def _higher_better(exp: dict[str, Any], metric_name: str) -> bool | None:
    metric_info = exp.get("display_metric_info") or exp.get("metric_info") or {}
    info = metric_info.get(metric_name) if isinstance(metric_info, dict) else None
    if isinstance(info, dict) and "higher_better" in info:
        return bool(info["higher_better"])
    return None


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

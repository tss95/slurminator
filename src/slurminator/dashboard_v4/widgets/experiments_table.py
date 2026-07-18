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
        table_sort: object | None = None,
    ) -> None:
        """Replace table rows with the latest experiment snapshot."""
        previous_key = None
        if self._row_experiments and 0 <= self.cursor_row < len(self._row_experiments):
            previous_key = _row_key(self._row_experiments[self.cursor_row], self.cursor_row)
        rows = _sorted_experiments(experiments, table_sort=table_sort)
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
                _format_status(exp),
                _format_progress(exp),
                _format_metric(exp, primary_name, primary_value),
            ]
            if show_sparkline:
                cells.append(_format_sparkline(exp, primary_name, thresholds=sparkline_thresholds))
            secondary_name = exp.get("secondary_metric_name")
            cells.append(_format_metric(exp, secondary_name, exp.get("secondary_metric_value")))
            if show_sparkline:
                cells.append(_format_sparkline(exp, secondary_name, thresholds=sparkline_thresholds))
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
        columns = ["ID", "Dataset", "HPC", "State", "Progress", primary_label]
        if show_sparkline:
            columns.append(_trajectory_column_label(primary_label))
        columns.append(secondary_label)
        if show_sparkline:
            columns.append(_trajectory_column_label(secondary_label))
        columns.append("Queue/DT")
        self.add_columns(*columns)


def table_render_signature(
    experiments: list[dict[str, Any]],
    *,
    show_sparkline: bool = False,
    sparkline_thresholds: SparklineThresholds | object | None = None,
    table_sort: object | None = None,
) -> tuple[object, ...]:
    """Return a display signature for deciding whether the table body changed."""
    rows = _sorted_experiments(experiments, table_sort=table_sort)
    return (
        bool(show_sparkline),
        _object_signature(sparkline_thresholds),
        tuple(sorted(_normalize_table_sort(table_sort).items())),
        tuple(_row_render_signature(exp, show_sparkline=show_sparkline) for exp in rows),
    )


def _text(value: object, default: str = "") -> str:
    if value is None:
        return default
    text = str(value)
    return text if text else default


def _trajectory_column_label(metric_label: str) -> str:
    return f"{metric_label}_traj"


def _format_enum(value: object) -> str:
    if isinstance(value, Enum):
        return str(value.value)
    return _text(value, "-")


def _format_status(exp: dict[str, Any]) -> str | Text:
    value = exp.get("status")
    status = _coerce_status(value)
    if exp.get("cancel_requested_at") is not None and status in {ExperimentStatus.QUEUED, ExperimentStatus.RUNNING}:
        return Text("CANCEL REQ", style="bold yellow")
    text = _format_enum(value)
    label = text.upper() if text != "-" else text
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


def _coerce_status(value: object) -> ExperimentStatus | None:
    if isinstance(value, ExperimentStatus):
        return value
    text = str(value).strip()
    if text.startswith("ExperimentStatus."):
        text = text.split(".", 1)[1]
    normalized = text.upper().rstrip("+*")
    if normalized.startswith("CANCELED") or normalized.startswith("CANCELLED"):
        return ExperimentStatus.CANCELLED
    try:
        return ExperimentStatus(text)
    except ValueError:
        try:
            return ExperimentStatus[normalized]
        except KeyError:
            return None


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
    if current is None and metric_key:
        current = _lookup_metric_value(exp, metric_key, _metric_shortform(metric_info))
    best = _lookup_best_metric_value(exp, metric_key, metric_info)
    if current is None and value is not None:
        return str(value)
    if current is None and best is None:
        return "-"
    style = _metric_color(current, metric_info)
    combined = _format_metric_pair(current, best, metric_info=metric_info)
    return Text(combined, style=style) if style else combined


def _metric_info_for(exp: dict[str, Any], metric_key: str) -> dict[str, Any] | None:
    metric_info = exp.get("display_metric_info") or exp.get("metric_info") or {}
    if not isinstance(metric_info, dict):
        return None
    info = metric_info.get(metric_key)
    if isinstance(info, dict):
        return info
    for candidate in metric_info.values():
        if not isinstance(candidate, dict):
            continue
        if candidate.get("shortform") == metric_key:
            return candidate
    return None


def _metric_shortform(metric_info: dict[str, Any] | None) -> str | None:
    shortform = metric_info.get("shortform") if isinstance(metric_info, dict) else None
    return str(shortform) if shortform else None


def _lookup_best_metric_value(
    exp: dict[str, Any], metric_key: str | None, metric_info: dict[str, Any] | None
) -> float | None:
    best_key = _resolve_best_metric_key(metric_key, metric_info)
    if not best_key:
        return None
    return _lookup_metric_value(exp, best_key)


def _resolve_best_metric_key(metric_key: str | None, metric_info: dict[str, Any] | None) -> str | None:
    best_key = metric_info.get("best_key") if isinstance(metric_info, dict) else None
    if best_key:
        return str(best_key)
    if not metric_key:
        return None
    if "/step_best_" in metric_key:
        return metric_key.replace("/step_best_", "/global_best_", 1)
    return None


def _lookup_metric_value(exp: dict[str, Any], metric_key: str, shortform: str | None = None) -> float | None:
    all_metrics = exp.get("all_metrics", {})
    candidates = [candidate for candidate in (shortform, metric_key) if candidate]
    for candidate in candidates:
        for source in (exp, all_metrics if isinstance(all_metrics, dict) else {}):
            if candidate in source:
                return _coerce_float(source[candidate])
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
    value_format = None
    if isinstance(metric_info, dict):
        value_format = metric_info.get("value_format") or metric_info.get("format")
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


def _sorted_experiments(experiments: list[dict[str, Any]], *, table_sort: object | None = None) -> list[dict[str, Any]]:
    rows = list(experiments)
    sort_settings = _normalize_table_sort(table_sort)

    def sort_key(exp: dict[str, Any]) -> tuple[Any, ...]:
        metric_name = _sort_metric_name(exp, sort_settings["metric"])
        metric_info = _metric_info_for(exp, metric_name)
        sort_value = _sort_metric_value(exp, metric_name, metric_info, sort_settings)
        has_metric = sort_value is not None
        metric_sort = _metric_sort_value(sort_value, metric_info, direction=sort_settings["direction"])
        keys: list[Any] = []
        if sort_settings["preserve_dataset_groups"]:
            keys.append(_dataset_group_key(exp))
        if sort_settings["preserve_state_groups"]:
            keys.append(_status_priority(exp.get("status")))
        keys.extend((0 if has_metric else 1, metric_sort))
        return tuple(keys)

    rows.sort(key=sort_key)
    return rows


def _normalize_table_sort(table_sort: object | None) -> dict[str, Any]:
    return {
        "metric": _choice_attr(table_sort, "metric", "primary", {"primary", "secondary"}),
        "value": _choice_attr(table_sort, "value", "current", {"current", "best"}),
        "direction": _choice_attr(table_sort, "direction", "auto", {"auto", "asc", "desc"}),
        "preserve_state_groups": _bool_attr(table_sort, "preserve_state_groups", True),
        "preserve_dataset_groups": _bool_attr(table_sort, "preserve_dataset_groups", False),
    }


def _choice_attr(settings: object | None, name: str, default: str, choices: set[str]) -> str:
    value = getattr(settings, name, default)
    text = str(value).strip().lower() if value is not None else default
    return text if text in choices else default


def _bool_attr(settings: object | None, name: str, default: bool) -> bool:
    value = getattr(settings, name, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _dataset_group_key(exp: dict[str, Any]) -> str:
    value = exp.get("dataset_name") or exp.get("dataset") or exp.get("config") or ""
    return str(value).lower()


def _status_priority(status: object) -> int:
    status = _coerce_status(status)
    if status == ExperimentStatus.RUNNING:
        return 0
    if status == ExperimentStatus.COMPLETED:
        return 1
    return 2


def _sort_metric_name(exp: dict[str, Any], metric_selector: str) -> str:
    if metric_selector == "secondary":
        return str(exp.get("secondary_metric_name") or "")
    return str(exp.get("target_metric_name") or "")


def _sort_metric_value(
    exp: dict[str, Any],
    metric_name: str,
    metric_info: dict[str, Any] | None,
    sort_settings: dict[str, Any],
) -> float | None:
    if sort_settings["value"] == "best":
        best = _lookup_best_metric_value(exp, metric_name, metric_info)
        if best is not None:
            return best
    if sort_settings["metric"] == "secondary":
        current = _coerce_float(exp.get("secondary_metric_value"))
    else:
        current = _coerce_float(
            exp.get("target_metric_value") if exp.get("target_metric_value") is not None else exp.get("metric_value")
        )
    if current is None and metric_name:
        current = _lookup_metric_value(exp, metric_name, _metric_shortform(metric_info))
    return current


def _metric_sort_value(value: float | None, metric_info: dict[str, Any] | None, *, direction: str = "auto") -> float:
    if value is None:
        return float("inf")
    if direction == "asc":
        return value
    if direction == "desc":
        return -value
    higher_better = True if not isinstance(metric_info, dict) else metric_info.get("higher_better", True)
    return -value if higher_better is not False else value


def _row_render_signature(exp: dict[str, Any], *, show_sparkline: bool) -> tuple[object, ...]:
    primary_name = str(exp.get("target_metric_name") or "")
    secondary_name = str(exp.get("secondary_metric_name") or "")
    primary_info = _metric_info_for(exp, primary_name)
    secondary_info = _metric_info_for(exp, secondary_name)
    primary_current = _coerce_float(
        exp.get("target_metric_value") if exp.get("target_metric_value") is not None else exp.get("metric_value")
    )
    secondary_current = _coerce_float(exp.get("secondary_metric_value"))
    signature: list[object] = [
        exp.get("experiment_id"),
        exp.get("dataset_name") or exp.get("dataset") or exp.get("config"),
        _format_enum(exp.get("hpc_assignment")),
        _format_enum(exp.get("status")),
        exp.get("cancel_requested_at"),
        exp.get("cancel_requested_job_id"),
        exp.get("current_step"),
        exp.get("max_steps"),
        exp.get("it_per_sec_backbone"),
        exp.get("current_epoch"),
        exp.get("max_epochs"),
        primary_name,
        primary_current,
        _lookup_best_metric_value(exp, primary_name, primary_info),
        _metric_info_signature(primary_info),
        secondary_name,
        secondary_current,
        _lookup_best_metric_value(exp, secondary_name, secondary_info),
        _metric_info_signature(secondary_info),
    ]
    if show_sparkline:
        signature.append(_sparkline_signature(exp, primary_name))
        signature.append(_sparkline_signature(exp, secondary_name))
    return tuple(signature)


def _metric_info_signature(metric_info: dict[str, Any] | None) -> tuple[tuple[str, object], ...] | None:
    if not isinstance(metric_info, dict):
        return None
    keys = ("shortform", "higher_better", "threshold", "value_format", "format", "best_key")
    return tuple((key, _signature_value(metric_info.get(key))) for key in keys if key in metric_info)


def _sparkline_signature(exp: dict[str, Any], metric_name: str) -> tuple[object, ...] | None:
    resolved = _resolve_sparkline_metric(exp, metric_name)
    if not resolved:
        return None
    return (resolved, _higher_better(exp, resolved), _history_metric_change_signature(exp, resolved))


def _history_metric_change_signature(exp: dict[str, Any], metric_name: str) -> tuple[int, float | None]:
    history = exp.get("history")
    if not isinstance(history, list):
        return (0, None)
    n_changes = 0
    last_value: float | None = None
    has_last = False
    for entry in history:
        metrics = entry.get("metrics") if isinstance(entry, dict) else None
        if not isinstance(metrics, dict) or metric_name not in metrics:
            continue
        value = _coerce_float(metrics.get(metric_name))
        if value is None:
            continue
        if not has_last or value != last_value:
            n_changes += 1
            last_value = value
            has_last = True
    return n_changes, last_value


def _object_signature(value: object) -> tuple[tuple[str, object], ...] | object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    attrs = getattr(value, "__dict__", None)
    if isinstance(attrs, dict):
        return tuple(sorted((str(key), _signature_value(item)) for key, item in attrs.items()))
    return repr(value)


def _signature_value(value: object) -> object:
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return repr(value)


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


def _format_sparkline(exp: dict[str, Any], metric: object, *, thresholds: SparklineThresholds | object | None):
    metric_name = _resolve_sparkline_metric(exp, metric)
    if metric_name:
        values = _history_metric_values(exp, metric_name, coalesce_repeats=True)
        if values:
            return render_sparkline(
                values, width=20, higher_better=_higher_better(exp, metric_name), thresholds=thresholds
            )
    return "-"


def _resolve_sparkline_metric(exp: dict[str, Any], metric: object) -> str | None:
    metric_name = str(metric) if metric else ""
    if not metric_name:
        return None
    return _resolve_history_metric_key(exp, metric_name)


def _resolve_history_metric_key(exp: dict[str, Any], metric_name: str) -> str | None:
    if _history_has_metric(exp, metric_name):
        return metric_name
    metric_info = exp.get("display_metric_info") or exp.get("metric_info") or {}
    if isinstance(metric_info, dict):
        for metric_key, info in metric_info.items():
            if metric_key == metric_name and _history_has_metric(exp, metric_key):
                return str(metric_key)
            shortform = info.get("shortform") if isinstance(info, dict) else None
            if shortform and str(shortform) == metric_name and _history_has_metric(exp, str(metric_key)):
                return str(metric_key)
    return None


def _history_has_metric(exp: dict[str, Any], metric_name: str) -> bool:
    history = exp.get("history")
    if not isinstance(history, list):
        return False
    return any(
        isinstance(entry, dict) and isinstance(entry.get("metrics"), dict) and metric_name in entry.get("metrics", {})
        for entry in history
    )


def _history_metric_values(exp: dict[str, Any], metric_name: str, *, coalesce_repeats: bool = False) -> list[float]:
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
            value = float(metrics[metric_name])
        except (TypeError, ValueError):
            continue
        if coalesce_repeats and values and value == values[-1]:
            continue
        values.append(value)
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


__all__ = ["ExperimentsTable", "table_render_signature"]

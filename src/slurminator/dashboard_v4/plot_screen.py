"""Per-run metric plot screen for dashboard v4."""

from __future__ import annotations

import asyncio
import math
import re
from typing import Any, Literal

import plotext as plt
from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.screen import Screen
from textual.widgets import Footer, Header, Label, ListItem, ListView, Static

from slurminator.dashboard_v4.keystrokes import PLOT_BINDINGS
from slurminator.experiments import ExperimentStatus

ProgressAxisUnit = Literal["epoch", "step"]
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
MAX_PLOT_WIDTH = 96
MAX_PLOT_HEIGHT = 22
PLOT_RUN_STATUSES = {ExperimentStatus.RUNNING, ExperimentStatus.COMPLETED}


class PerRunPlotScreen(Screen[None]):
    """Plot one run's history metrics."""

    BINDINGS = PLOT_BINDINGS

    def __init__(self, exp: dict[str, Any]) -> None:
        super().__init__()
        self.exp = exp
        self.history: list[dict[str, Any]] = list(exp.get("history") or [])
        self.metric_keys: list[str] = []
        self.selected_metric: str | None = None
        self.log_scale = False
        self.show_best_overlay = False
        self.plot_exps: list[dict[str, Any]] = []
        self._selected_experiment_id = str(exp.get("experiment_id", ""))
        self._run_list_signature: tuple[tuple[str, str], ...] = ()
        self._last_plot_text = ""
        self._last_plot_dimensions: tuple[int, int] | None = None
        self._last_axis_unit: ProgressAxisUnit | None = None
        self._last_axis_label = ""
        self._last_x_values: list[float] = []
        self._last_xticks: list[float] = []
        self._last_yticks: list[float] = []
        self._exp_by_item_id: dict[str, dict[str, Any]] = {}
        self._metric_by_item_id: dict[str, str] = {}
        self._run_list_lock = asyncio.Lock()
        self._metric_list_lock = asyncio.Lock()
        self._run_list_generation = 0
        self._metric_list_generation = 0

    def compose(self) -> ComposeResult:
        """Compose the metric selector and plot panel."""
        yield Header()
        with Horizontal(id="plot-content"):
            yield ListView(id="runs")
            yield ListView(id="metrics")
            yield Static("", id="plot")
        yield Footer()

    async def on_mount(self) -> None:
        """Force-load history and draw the initial plot."""
        self._force_read_history(self.exp)
        await self._rebuild_run_list(force=True)
        self.history = list(self.exp.get("history") or [])
        await self._rebuild_metric_list()
        self.set_interval(getattr(self.app, "refresh_interval", 1.0), self.refresh_from_orchestrator)
        self.call_after_refresh(self._redraw_plot)
        self.query_one("#metrics", ListView).focus()

    def on_resize(self, _event: events.Resize) -> None:
        """Regenerate the plot when the terminal layout changes."""
        self._redraw_plot()

    async def refresh_from_orchestrator(self) -> None:
        """Refresh the plot from the latest app snapshot when history changes."""
        previous_run_ids = [str(exp.get("experiment_id", "")) for exp in self.plot_exps]
        await self._rebuild_run_list()
        current_run_ids = [str(exp.get("experiment_id", "")) for exp in self.plot_exps]
        if current_run_ids != previous_run_ids:
            self._set_selected_run(self._selected_experiment_id)

        latest = self._latest_snapshot_exp()
        if latest is None:
            return
        latest_history = list(latest.get("history") or [])
        if latest is self.exp and latest_history == self.history:
            return
        if latest_history != self.history:
            self.exp = latest
            self.history = latest_history
            previous_metric = self.selected_metric
            await self._rebuild_metric_list()
            if previous_metric in self.metric_keys:
                self._set_selected_metric(previous_metric)
            self._redraw_plot()

    async def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        """Redraw when the highlighted run or metric changes."""
        if event.list_view.id == "runs":
            exp = self._exp_for_item(event.item)
            if exp is not None and str(exp.get("experiment_id", "")) != self._selected_experiment_id:
                await self._activate_experiment(exp)
            return
        if event.list_view.id != "metrics":
            return
        metric = self._metric_for_item(event.item)
        if metric is not None and metric != self.selected_metric:
            self.selected_metric = metric
            self._redraw_plot()

    async def action_previous_run(self) -> None:
        """Switch to the previous plottable run."""
        await self._move_run(-1)

    async def action_next_run(self) -> None:
        """Switch to the next plottable run."""
        await self._move_run(1)

    def action_metric_up(self) -> None:
        """Move to the previous metric."""
        self.query_one("#metrics", ListView).action_cursor_up()
        self._select_current_metric()

    def action_metric_down(self) -> None:
        """Move to the next metric."""
        self.query_one("#metrics", ListView).action_cursor_down()
        self._select_current_metric()

    def action_toggle_log_scale(self) -> None:
        """Toggle log-scale rendering."""
        self.log_scale = not self.log_scale
        self._redraw_plot()

    def action_toggle_best_overlay(self) -> None:
        """Toggle running-best overlay rendering."""
        self.show_best_overlay = not self.show_best_overlay
        self._redraw_plot()

    def _latest_snapshot_exp(self) -> dict[str, Any] | None:
        experiment_id = self._selected_experiment_id or self.exp.get("experiment_id")
        for exp in self.app.get_dashboard_snapshot():
            if str(exp.get("experiment_id", "")) == str(experiment_id):
                return exp
        return None

    async def _rebuild_run_list(self, *, force: bool = False) -> None:
        async with self._run_list_lock:
            self.plot_exps = _plot_run_candidates(self.app.get_dashboard_snapshot(), self.exp)
            signature = _run_list_signature(self.plot_exps)
            if force or signature != self._run_list_signature:
                self._run_list_signature = signature
                self._run_list_generation += 1
                runs = self.query_one("#runs", ListView)
                await runs.clear()
                items: list[ListItem] = []
                for index, exp in enumerate(self.plot_exps):
                    item_id = _run_item_id(exp, index, self._run_list_generation)
                    items.append(ListItem(Label(_run_label(exp)), id=item_id))
                if items:
                    await runs.extend(items)
            self._exp_by_item_id = {}
            for index, exp in enumerate(self.plot_exps):
                self._exp_by_item_id[_run_item_id(exp, index, self._run_list_generation)] = exp
            self._set_selected_run(self._selected_experiment_id)

    async def _activate_experiment(self, exp: dict[str, Any], *, preserve_metric: bool = True) -> None:
        previous_metric = self.selected_metric if preserve_metric else None
        self.exp = exp
        self._selected_experiment_id = str(exp.get("experiment_id", ""))
        self._force_read_history(self.exp)
        self.history = list(self.exp.get("history") or [])
        await self._rebuild_metric_list()
        if previous_metric in self.metric_keys:
            self._set_selected_metric(previous_metric)
        self._set_selected_run(self._selected_experiment_id)
        self._redraw_plot()

    async def _move_run(self, delta: int) -> None:
        if not self.plot_exps:
            return
        current_index = _run_index(self.plot_exps, self._selected_experiment_id)
        next_index = (current_index + delta) % len(self.plot_exps)
        await self._activate_experiment(self.plot_exps[next_index])

    def _set_selected_run(self, experiment_id: str | None) -> None:
        if not self.plot_exps:
            return
        runs = self.query_one("#runs", ListView)
        index = _run_index(self.plot_exps, experiment_id)
        runs.index = index

    def _force_read_history(self, exp: dict[str, Any]) -> None:
        orchestrator = getattr(self.app, "orchestrator", None)
        if orchestrator is not None and hasattr(orchestrator, "force_read_full_history"):
            existing_history = list(exp.get("history") or [])
            orchestrator.force_read_full_history(exp)
            if existing_history and not exp.get("history"):
                exp["history"] = existing_history

    async def _rebuild_metric_list(self) -> None:
        async with self._metric_list_lock:
            next_metric_keys = _metric_keys(self.history)
            metrics = self.query_one("#metrics", ListView)
            needs_rebuild = next_metric_keys != self.metric_keys or len(metrics.children) != len(next_metric_keys)
            self.metric_keys = next_metric_keys
            if needs_rebuild:
                self._metric_list_generation += 1
                await metrics.clear()
                items: list[ListItem] = []
                for key in self.metric_keys:
                    item_id = _metric_item_id(key, self._metric_list_generation)
                    items.append(ListItem(Label(key), id=item_id))
                if items:
                    await metrics.extend(items)
            self._metric_by_item_id = {}
            for key in self.metric_keys:
                self._metric_by_item_id[_metric_item_id(key, self._metric_list_generation)] = key
            if self.metric_keys:
                if self.selected_metric not in self.metric_keys:
                    self.selected_metric = self.metric_keys[0]
                self._set_selected_metric(self.selected_metric)
            else:
                self.selected_metric = None

    def _set_selected_metric(self, metric: str | None) -> None:
        if metric is None or metric not in self.metric_keys:
            return
        metrics = self.query_one("#metrics", ListView)
        metrics.index = self.metric_keys.index(metric)
        self.selected_metric = metric

    def _select_current_metric(self) -> None:
        metrics = self.query_one("#metrics", ListView)
        index = metrics.index
        if index is None or index < 0 or index >= len(self.metric_keys):
            return
        self.selected_metric = self.metric_keys[index]
        self._redraw_plot()

    def _metric_for_item(self, item: ListItem | None) -> str | None:
        if item is None or not item.id:
            return None
        return self._metric_by_item_id.get(item.id)

    def _exp_for_item(self, item: ListItem | None) -> dict[str, Any] | None:
        if item is None or not item.id:
            return None
        return self._exp_by_item_id.get(item.id)

    def _redraw_plot(self) -> None:
        plot = self.query_one("#plot", Static)
        if not self.history:
            self._last_plot_text = "No history available"
            self._clear_last_axis_state()
            plot.update(self._last_plot_text)
            return
        if not self.selected_metric:
            self._last_plot_text = "No metric history available"
            self._clear_last_axis_state()
            plot.update(self._last_plot_text)
            return

        axis_unit = self._resolve_progress_unit()
        points = _series_for_metric(self.history, self.selected_metric, unit=axis_unit, log_scale=self.log_scale)
        if not points:
            self._last_plot_text = f"No plottable values for {self.selected_metric}"
            self._clear_last_axis_state()
            plot.update(self._last_plot_text)
            return

        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        width, height = _plot_dimensions(
            plot,
            self.query_one("#plot-content", Horizontal),
            [self.query_one("#runs", ListView), self.query_one("#metrics", ListView)],
        )
        self._last_plot_dimensions = (width, height)
        self._last_axis_unit = axis_unit
        self._last_axis_label = "Step" if axis_unit == "step" else "Epoch"
        self._last_x_values = xs
        self._last_xticks = _distributed_ticks(xs, max_ticks=8)
        self._last_yticks = [] if self.log_scale else _linear_ticks(ys, max_ticks=6)

        plt.clear_figure()
        plt.plotsize(width, height)
        plt.theme("clear")
        plt.title(_shorten_middle(str(self.selected_metric), max(width - 8, 12)))
        plt.xlabel(self._last_axis_label)
        plt.ylabel(self.selected_metric)
        plt.yscale("log" if self.log_scale else "linear")
        plt.grid(False, False)
        if self._last_xticks:
            plt.xticks(self._last_xticks)
        if self._last_yticks:
            plt.yticks(self._last_yticks)
        plt.plot(xs, ys)
        if self.show_best_overlay:
            higher_better = self._higher_better(self.selected_metric)
            plt.plot(xs, _running_best(ys, higher_better=higher_better), label="best")
        self._last_plot_text = _strip_ansi(plt.build())
        if not self._last_plot_text or not self._last_plot_text.strip():
            self._last_plot_text = (
                f"plotext build returned empty output for {self.selected_metric} "
                f"(plot area: {width}x{height}, points: {len(xs)})"
            )
        else:
            self._last_plot_text = _clip_plot_output(self._last_plot_text, width=width, height=height)
        plt.yscale("linear")
        plot.update(Text(self._last_plot_text))

    def _resolve_progress_unit(self) -> ProgressAxisUnit:
        return _resolve_progress_unit(self.history, self.exp)

    def _clear_last_axis_state(self) -> None:
        self._last_axis_unit = None
        self._last_axis_label = ""
        self._last_x_values = []
        self._last_xticks = []
        self._last_yticks = []

    def _higher_better(self, metric: str) -> bool:
        metric_info = self.exp.get("display_metric_info") or self.exp.get("metric_info") or {}
        info = metric_info.get(metric) if isinstance(metric_info, dict) else None
        if isinstance(info, dict) and "higher_better" in info:
            return bool(info["higher_better"])
        return True


def _metric_keys(history: list[dict[str, Any]]) -> list[str]:
    keys: set[str] = set()
    for entry in history:
        metrics = entry.get("metrics") if isinstance(entry, dict) else None
        if not isinstance(metrics, dict):
            continue
        for key, value in metrics.items():
            if _is_finite_number(value):
                keys.add(str(key))
    return sorted(keys)


def _metric_item_id(metric: str, generation: int) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in metric)
    return f"metric-{generation}-{safe}"


def _plot_dimensions(plot: Static, container: Horizontal, sidebars: list[ListView]) -> tuple[int, int]:
    plot_width = int(plot.size.width or 0) - 2
    fallback_width = int(container.size.width or 0) - sum(int(sidebar.size.width or 0) for sidebar in sidebars) - 4
    width = plot_width if plot_width > 0 else fallback_width
    plot_height = int(plot.size.height or 0) - 1
    fallback_height = int(container.size.height or 0) - 1
    height = plot_height if plot_height > 0 else fallback_height
    return max(min(width, MAX_PLOT_WIDTH), 30), max(min(height, MAX_PLOT_HEIGHT), 8)


def _plot_run_candidates(snapshot: list[dict[str, Any]], current_exp: dict[str, Any]) -> list[dict[str, Any]]:
    current_id = str(current_exp.get("experiment_id", ""))
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for exp in snapshot:
        experiment_id = str(exp.get("experiment_id", ""))
        if not experiment_id or experiment_id in seen:
            continue
        if _is_plot_run_candidate(exp) or experiment_id == current_id:
            candidates.append(exp)
            seen.add(experiment_id)
    if current_id and current_id not in seen:
        candidates.insert(0, current_exp)
    elif not candidates:
        candidates.append(current_exp)
    return candidates


def _is_plot_run_candidate(exp: dict[str, Any]) -> bool:
    status = _coerce_experiment_status(exp.get("status"))
    return status in PLOT_RUN_STATUSES


def _coerce_experiment_status(value: object) -> ExperimentStatus | None:
    if isinstance(value, ExperimentStatus):
        return value
    try:
        return ExperimentStatus(str(value))
    except ValueError:
        try:
            return ExperimentStatus[str(value).upper()]
        except KeyError:
            return None


def _run_index(exps: list[dict[str, Any]], experiment_id: str | None) -> int:
    if not exps:
        return 0
    for index, exp in enumerate(exps):
        if str(exp.get("experiment_id", "")) == str(experiment_id):
            return index
    return 0


def _run_list_signature(exps: list[dict[str, Any]]) -> tuple[tuple[str, str], ...]:
    return tuple((str(exp.get("experiment_id", "")), str(_coerce_experiment_status(exp.get("status")))) for exp in exps)


def _run_item_id(exp: dict[str, Any], index: int, generation: int) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in str(exp.get("experiment_id", "")))
    return f"run-{generation}-{index}-{safe or 'unknown'}"


def _run_label(exp: dict[str, Any]) -> str:
    status = _coerce_experiment_status(exp.get("status"))
    prefix = {ExperimentStatus.RUNNING: "RUN", ExperimentStatus.COMPLETED: "DONE"}.get(status, "RUN")
    return f"{prefix} {_shorten_middle(str(exp.get('experiment_id', 'unknown')), 19)}"


def _resolve_progress_unit(history: list[dict[str, Any]], exp: dict[str, Any]) -> ProgressAxisUnit:
    """Resolve the canonical plot x-axis without package-specific pseudo-epoch knowledge."""
    history_unit = _unit_from_latest_history_entry(history)
    if history_unit is not None:
        return history_unit
    exp_unit = _unit_from_experiment(exp)
    if exp_unit is not None:
        return exp_unit
    return _infer_unit_from_history_shape(history)


def _unit_from_latest_history_entry(history: list[dict[str, Any]]) -> ProgressAxisUnit | None:
    for entry in reversed(history):
        if not isinstance(entry, dict):
            continue
        unit = entry.get("unit")
        if unit in ("epoch", "step"):
            return unit
    return None


def _unit_from_experiment(exp: dict[str, Any]) -> ProgressAxisUnit | None:
    progress = exp.get("progress")
    if isinstance(progress, dict):
        unit = progress.get("unit")
        if unit in ("epoch", "step"):
            return unit
    for key in ("progress_unit", "unit"):
        unit = exp.get(key)
        if unit in ("epoch", "step"):
            return unit
    return None


def _infer_unit_from_history_shape(history: list[dict[str, Any]]) -> ProgressAxisUnit:
    step_values = _axis_values(history, "step")
    epoch_values = _axis_values(history, "epoch")
    if step_values and not epoch_values:
        return "step"
    if epoch_values and not step_values:
        return "epoch"
    if not step_values or not epoch_values:
        return "epoch"

    step_unique = len(set(step_values))
    epoch_unique = len(set(epoch_values))
    if step_unique > epoch_unique:
        return "step"
    if epoch_unique > step_unique:
        return "epoch"

    step_span = max(step_values) - min(step_values)
    epoch_span = max(epoch_values) - min(epoch_values)
    if step_span > max(epoch_span * 4.0, 20.0):
        return "step"
    return "epoch"


def _axis_values(history: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for entry in history:
        if not isinstance(entry, dict):
            continue
        value = entry.get(key)
        if _is_finite_number(value):
            values.append(float(value))
    return values


def _series_for_metric(
    history: list[dict[str, Any]], metric: str, *, unit: ProgressAxisUnit = "epoch", log_scale: bool = False
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    fallback_x = 1
    for entry in history:
        metrics = entry.get("metrics") if isinstance(entry, dict) else None
        if not isinstance(metrics, dict):
            continue
        y_value = metrics.get(metric)
        if not _is_finite_number(y_value):
            continue
        y = float(y_value)
        if log_scale and y <= 0.0:
            continue
        x_value = _axis_value(entry, unit)
        if not _is_finite_number(x_value):
            x_value = fallback_x
        fallback_x += 1
        if points and points[-1][1] == y:
            continue
        points.append((float(x_value), y))
    return points


def _axis_value(entry: dict[str, Any], unit: ProgressAxisUnit) -> object:
    primary = "step" if unit == "step" else "epoch"
    secondary = "epoch" if unit == "step" else "step"
    x_value = entry.get(primary)
    if x_value is None:
        x_value = entry.get(secondary)
    return x_value


def _distributed_ticks(values: list[float], *, max_ticks: int = 8, min_ticks: int = 2) -> list[float]:
    if not values:
        return []
    if len(values) == 1:
        return [values[0]]
    n_ticks = min(max_ticks, max(min_ticks, len(values)))
    indices = [round(index * (len(values) - 1) / (n_ticks - 1)) for index in range(n_ticks)]
    ticks: list[float] = []
    seen: set[float] = set()
    for index in indices:
        value = values[index]
        if value in seen:
            continue
        seen.add(value)
        ticks.append(value)
    return ticks


def _linear_ticks(values: list[float], *, max_ticks: int = 6) -> list[float]:
    if not values:
        return []
    y_min = min(values)
    y_max = max(values)
    if y_max <= y_min:
        return []
    n_ticks = max(2, max_ticks)
    return [y_min + (y_max - y_min) * index / (n_ticks - 1) for index in range(n_ticks)]


def _strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def _clip_plot_output(text: str, *, width: int, height: int) -> str:
    lines = [line[:width].rstrip() for line in text.splitlines()]
    return "\n".join(lines[:height])


def _shorten_middle(text: str, max_length: int) -> str:
    if len(text) <= max_length:
        return text
    marker = "..."
    if max_length <= len(marker):
        return text[:max_length]
    left = max((max_length - len(marker)) // 2, 1)
    right = max(max_length - left - len(marker), 1)
    return f"{text[:left]}{marker}{text[-right:]}"


def _running_best(values: list[float], *, higher_better: bool) -> list[float]:
    best_values: list[float] = []
    running_best: float | None = None
    for value in values:
        if (
            running_best is None
            or (higher_better and value > running_best)
            or (not higher_better and value < running_best)
        ):
            running_best = value
        best_values.append(running_best)
    return best_values


def _is_finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


__all__ = ["PerRunPlotScreen"]

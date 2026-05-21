"""Per-run metric plot screen for dashboard v4."""

from __future__ import annotations

import math
from typing import Any

import plotext as plt
from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.screen import Screen
from textual.widgets import Footer, Header, Label, ListItem, ListView, Static

from slurminator.dashboard_v4.keystrokes import PLOT_BINDINGS


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
        self._last_plot_text = ""
        self._last_plot_dimensions: tuple[int, int] | None = None
        self._metric_by_item_id: dict[str, str] = {}

    def compose(self) -> ComposeResult:
        """Compose the metric selector and plot panel."""
        yield Header()
        with Horizontal(id="plot-content"):
            yield ListView(id="metrics")
            yield Static("", id="plot")
        yield Footer()

    async def on_mount(self) -> None:
        """Force-load history and draw the initial plot."""
        orchestrator = getattr(self.app, "orchestrator", None)
        if orchestrator is not None and hasattr(orchestrator, "force_read_full_history"):
            orchestrator.force_read_full_history(self.exp)
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

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        """Redraw when the highlighted metric changes."""
        metric = self._metric_for_item(event.item)
        if metric is not None and metric != self.selected_metric:
            self.selected_metric = metric
            self._redraw_plot()

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
        experiment_id = self.exp.get("experiment_id")
        for exp in self.app.get_dashboard_snapshot():
            if exp.get("experiment_id") == experiment_id:
                return exp
        return None

    async def _rebuild_metric_list(self) -> None:
        self.metric_keys = _metric_keys(self.history)
        self._metric_by_item_id = {}
        metrics = self.query_one("#metrics", ListView)
        await metrics.clear()
        items: list[ListItem] = []
        for key in self.metric_keys:
            item_id = _metric_item_id(key)
            self._metric_by_item_id[item_id] = key
            items.append(ListItem(Label(key), id=item_id))
        if items:
            await metrics.extend(items)
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

    def _redraw_plot(self) -> None:
        plot = self.query_one("#plot", Static)
        if not self.history:
            self._last_plot_text = "No history available"
            plot.update(self._last_plot_text)
            return
        if not self.selected_metric:
            self._last_plot_text = "No metric history available"
            plot.update(self._last_plot_text)
            return

        points = _series_for_metric(self.history, self.selected_metric, log_scale=self.log_scale)
        if not points:
            self._last_plot_text = f"No plottable values for {self.selected_metric}"
            plot.update(self._last_plot_text)
            return

        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        container = self.query_one("#plot-content", Horizontal)
        metrics_list = self.query_one("#metrics", ListView)
        width = max(int(container.size.width) - int(metrics_list.size.width) - 2, 30)
        height = max(int(container.size.height) - 1, 8)
        self._last_plot_dimensions = (width, height)

        plt.clear_figure()
        plt.plotsize(width, height)
        plt.theme("clear")
        plt.title(f"{self.exp.get('experiment_id', 'run')} - {self.selected_metric}")
        plt.xlabel("epoch/step")
        plt.ylabel(self.selected_metric)
        plt.yscale("log" if self.log_scale else "linear")
        plt.plot(xs, ys, label=self.selected_metric)
        if self.show_best_overlay:
            higher_better = self._higher_better(self.selected_metric)
            plt.plot(xs, _running_best(ys, higher_better=higher_better), label=f"best({self.selected_metric})")
        self._last_plot_text = plt.build()
        if not self._last_plot_text or not self._last_plot_text.strip():
            self._last_plot_text = (
                f"plotext build returned empty output for {self.selected_metric} "
                f"(plot area: {width}x{height}, points: {len(xs)})"
            )
        plt.yscale("linear")
        plot.update(Text.from_ansi(self._last_plot_text))

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


def _metric_item_id(metric: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in metric)
    return f"metric-{safe}"


def _series_for_metric(
    history: list[dict[str, Any]], metric: str, *, log_scale: bool = False
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
        x_value = entry.get("epoch")
        if x_value is None:
            x_value = entry.get("step")
        if not _is_finite_number(x_value):
            x_value = fallback_x
        points.append((float(x_value), y))
        fallback_x += 1
    return points


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

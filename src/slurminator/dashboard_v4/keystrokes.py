"""Centralized Textual binding tables for dashboard v4."""

from textual.binding import Binding

HOME_BINDINGS = [
    Binding("up", "cursor_up", "Up"),
    Binding("down", "cursor_down", "Down"),
    Binding("enter", "drill_in", "Drill in"),
    Binding("p,P", "toggle_pause", "Pause/resume"),
    Binding("q", "quit", "Quit"),
    Binding("escape", "noop", "", show=False),
]

PER_RUN_MENU_BINDINGS = [Binding("escape", "app.pop_screen", "Back")]

PLOT_BINDINGS = [
    Binding("escape", "app.pop_screen", "Back"),
    Binding("l", "toggle_log_scale", "Log scale"),
    Binding("b", "toggle_best_overlay", "Best overlay"),
    Binding("up", "metric_up", "Previous metric"),
    Binding("down", "metric_down", "Next metric"),
]

DETAIL_BINDINGS = [
    Binding("escape", "app.pop_screen", "Back"),
    Binding("r", "refresh", "Refresh"),
    Binding("up", "scroll_up", "Scroll up"),
    Binding("down", "scroll_down", "Scroll down"),
]

LOG_BINDINGS = [
    Binding("escape", "app.pop_screen", "Back"),
    Binding("r", "refresh", "Refresh"),
    Binding("up", "scroll_up", "Scroll up"),
    Binding("down", "scroll_down", "Scroll down"),
]

__all__ = ["DETAIL_BINDINGS", "HOME_BINDINGS", "LOG_BINDINGS", "PER_RUN_MENU_BINDINGS", "PLOT_BINDINGS"]

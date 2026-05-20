"""Centralized Textual binding tables for dashboard v4."""

from textual.binding import Binding

HOME_BINDINGS = [
    Binding("up", "cursor_up", "Up"),
    Binding("down", "cursor_down", "Down"),
    Binding("enter", "drill_in", "Drill in"),
    Binding("p,P", "toggle_pause", "Pause/resume"),
    Binding("g", "app.global_menu", "Global"),
    Binding("s", "toggle_sparkline", "Sparkline"),
    Binding("q", "quit", "Quit"),
    Binding("escape", "noop", "", show=False),
]

GLOBAL_MENU_BINDINGS = [Binding("escape", "app.pop_screen", "Back"), Binding("q", "app.quit", "Quit")]

PER_RUN_MENU_BINDINGS = [
    Binding("escape", "app.pop_screen", "Back"),
    Binding("g", "app.global_menu", "Global"),
    Binding("q", "app.quit", "Quit"),
]

PLOT_BINDINGS = [
    Binding("escape", "app.pop_screen", "Back"),
    Binding("g", "app.global_menu", "Global"),
    Binding("q", "app.quit", "Quit"),
    Binding("l", "toggle_log_scale", "Log scale"),
    Binding("b", "toggle_best_overlay", "Best overlay"),
    Binding("up", "metric_up", "Previous metric"),
    Binding("down", "metric_down", "Next metric"),
]

DETAIL_BINDINGS = [
    Binding("escape", "app.pop_screen", "Back"),
    Binding("g", "app.global_menu", "Global"),
    Binding("q", "app.quit", "Quit"),
    Binding("r", "refresh", "Refresh"),
    Binding("up", "scroll_up", "Scroll up"),
    Binding("down", "scroll_down", "Scroll down"),
]

LOG_BINDINGS = [
    Binding("escape", "app.pop_screen", "Back"),
    Binding("g", "app.global_menu", "Global"),
    Binding("q", "app.quit", "Quit"),
    Binding("r", "refresh", "Refresh"),
    Binding("up", "scroll_up", "Scroll up"),
    Binding("down", "scroll_down", "Scroll down"),
]

__all__ = [
    "DETAIL_BINDINGS",
    "GLOBAL_MENU_BINDINGS",
    "HOME_BINDINGS",
    "LOG_BINDINGS",
    "PER_RUN_MENU_BINDINGS",
    "PLOT_BINDINGS",
]

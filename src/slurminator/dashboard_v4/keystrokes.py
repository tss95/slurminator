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

__all__ = ["HOME_BINDINGS", "PER_RUN_MENU_BINDINGS"]

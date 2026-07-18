"""Help overlay for dashboard v4."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container
from textual.screen import ModalScreen
from textual.widgets import Label, ListItem, ListView, Static


class HelpScreen(ModalScreen[None]):
    """Small keyboard and action reference for dashboard v4."""

    BINDINGS = [("escape", "app.pop_screen", "Back")]

    def on_mount(self) -> None:
        """Focus the return action."""
        self.query_one("#help-actions", ListView).focus()

    def compose(self) -> ComposeResult:
        """Render help content."""
        yield Container(
            Label("Help", id="help-title"),
            Static(_HELP_TEXT, id="help-content"),
            ListView(ListItem(Label("Return"), id="return"), id="help-actions"),
            id="help-screen",
        )

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Close the help overlay when Return is selected."""
        if event.item.id == "return":
            self.app.pop_screen()


_HELP_TEXT = """Home
Up/Down: move selection
Enter: per-run menu
g: global menu
s: toggle sparkline
y: copy experiment-list ID
q: quit from home

Global menu
Pause/resume submissions
Set concurrency limits
Set Slurm overrides for pending runs
Set table sort
Cancel active jobs

Per-run menu
View plots, details, and logs
Cancel selected run
Relaunch terminal runs
Edit next-submission settings

Screens
Esc: return
r: refresh details/logs
s: switch log source
c: copy selected/loaded log text
m: toggle log terminal-selection mode
l: plot log scale
b: plot best overlay
?: help

Log copy mode
Space: mark start at top visible log line
Scroll with mouse wheel or arrows
Space again: mark end at top visible log line
y or c: copy selected line range
Esc: cancel active selection
c without selection: copy loaded tail"""


__all__ = ["HelpScreen"]

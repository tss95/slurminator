"""Placeholder per-run action modal for dashboard v4."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Container
from textual.screen import ModalScreen
from textual.widgets import Label, ListItem, ListView

from slurminator.dashboard_v4.keystrokes import PER_RUN_MENU_BINDINGS
from slurminator.dashboard_v4.plot_screen import PerRunPlotScreen


class PerRunMenuScreen(ModalScreen[None]):
    """Modal action menu for one selected experiment."""

    BINDINGS = PER_RUN_MENU_BINDINGS

    def __init__(self, exp: dict[str, Any]) -> None:
        super().__init__()
        self.exp = exp

    def on_mount(self) -> None:
        """Focus the action list for Enter/Escape-driven navigation."""
        self.query_one("#per-run-actions", ListView).focus()

    def compose(self) -> ComposeResult:
        """Render placeholder per-run actions."""
        experiment_id = self.exp.get("experiment_id", "selected run")
        yield Container(
            Label(str(experiment_id), id="per-run-title"),
            ListView(
                ListItem(Label("View plots"), id="view-plots"),
                ListItem(Label("Details")),
                ListItem(Label("Logs")),
                ListItem(Label("Cancel")),
                ListItem(Label("Relaunch")),
                ListItem(Label("Settings")),
                id="per-run-actions",
            ),
            id="per-run-menu",
        )

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Dispatch placeholder menu selections that are in-scope for this slice."""
        if event.item.id == "view-plots":
            self.app.push_screen(PerRunPlotScreen(self.exp))


__all__ = ["PerRunMenuScreen"]

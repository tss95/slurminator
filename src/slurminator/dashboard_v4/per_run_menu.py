"""Placeholder per-run action modal for dashboard v4."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Container
from textual.screen import ModalScreen
from textual.widgets import Label, ListItem, ListView

from slurminator.dashboard_v4.commands import submit_command
from slurminator.dashboard_v4.detail_screen import PerRunDetailScreen
from slurminator.dashboard_v4.forms.relaunch_form import RelaunchFormScreen
from slurminator.dashboard_v4.keystrokes import PER_RUN_MENU_BINDINGS
from slurminator.dashboard_v4.log_screen import PerRunLogScreen
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
                ListItem(Label("View details"), id="view-details"),
                ListItem(Label("View log tail"), id="view-log-tail"),
                ListItem(Label("Cancel selected run"), id="cancel-run"),
                ListItem(Label("Relaunch"), id="relaunch-run"),
                ListItem(Label("Settings"), id="settings"),
                ListItem(Label("Return"), id="return"),
                id="per-run-actions",
            ),
            id="per-run-menu",
        )

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Dispatch placeholder menu selections that are in-scope for this slice."""
        if event.item.id == "view-plots":
            self.app.push_screen(PerRunPlotScreen(self.exp))
        elif event.item.id == "view-details":
            self.app.push_screen(PerRunDetailScreen(self.exp))
        elif event.item.id == "view-log-tail":
            self.app.push_screen(PerRunLogScreen(self.exp))
        elif event.item.id == "cancel-run":
            target = {"experiment_id": self.exp.get("experiment_id")}
            if self.exp.get("job_id") is not None:
                target["job_id"] = self.exp.get("job_id")
            submit_command(self.app.command_save_path(), "cancel_run", target)
            self.app.pop_screen()
        elif event.item.id == "relaunch-run":
            self.app.push_screen(RelaunchFormScreen(self.exp))
        elif event.item.id == "return":
            self.app.pop_screen()


__all__ = ["PerRunMenuScreen"]

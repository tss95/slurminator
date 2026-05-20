"""Placeholder per-run action modal for dashboard v4."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Container
from textual.screen import ModalScreen
from textual.widgets import Label, ListItem, ListView

from slurminator.dashboard_v4.keystrokes import PER_RUN_MENU_BINDINGS


class PerRunMenuScreen(ModalScreen[None]):
    """Modal action menu for one selected experiment."""

    BINDINGS = PER_RUN_MENU_BINDINGS

    def __init__(self, exp: dict[str, Any]) -> None:
        super().__init__()
        self.exp = exp

    def compose(self) -> ComposeResult:
        """Render placeholder per-run actions."""
        experiment_id = self.exp.get("experiment_id", "selected run")
        yield Container(
            Label(str(experiment_id), id="per-run-title"),
            ListView(
                ListItem(Label("Plot")),
                ListItem(Label("Details")),
                ListItem(Label("Logs")),
                ListItem(Label("Cancel")),
                ListItem(Label("Relaunch")),
                ListItem(Label("Settings")),
                id="per-run-actions",
            ),
            id="per-run-menu",
        )


__all__ = ["PerRunMenuScreen"]

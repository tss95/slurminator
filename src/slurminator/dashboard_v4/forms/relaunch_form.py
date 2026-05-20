"""Relaunch confirmation modal for dashboard v4."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Container
from textual.screen import ModalScreen
from textual.widgets import Label, ListItem, ListView

from slurminator.dashboard_v4.commands import submit_command
from slurminator.experiments import ExperimentStatus


class RelaunchFormScreen(ModalScreen[None]):
    """Confirm relaunching a terminal experiment."""

    BINDINGS = [("escape", "app.pop_screen", "Back")]

    def __init__(self, exp: dict[str, Any]) -> None:
        super().__init__()
        self.exp = exp

    def on_mount(self) -> None:
        """Focus the relaunch action list."""
        self.query_one("#relaunch-actions", ListView).focus()

    def compose(self) -> ComposeResult:
        """Render relaunch confirmation content."""
        experiment_id = self.exp.get("experiment_id", "selected run")
        relaunchable = _is_relaunchable(self.exp.get("status"))
        message = (
            "Queue a new submission for this terminal run."
            if relaunchable
            else "Only terminal runs can be relaunched. Cancel active runs first."
        )
        actions = [ListItem(Label("Confirm relaunch"), id="confirm-relaunch")] if relaunchable else []
        actions.append(ListItem(Label("Back"), id="back"))
        yield Container(
            Label(f"Relaunch {experiment_id}", id="relaunch-title"),
            Label(message, id="relaunch-message"),
            ListView(*actions, id="relaunch-actions"),
            id="relaunch-form",
        )

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Write a relaunch command when confirmed."""
        if event.item.id == "confirm-relaunch":
            target = {"experiment_id": self.exp.get("experiment_id")}
            if self.exp.get("job_id") is not None:
                target["job_id"] = self.exp.get("job_id")
            submit_command(self.app.command_save_path(), "relaunch_run", target)
            self.app.pop_screen()
        elif event.item.id == "back":
            self.app.pop_screen()


def _is_relaunchable(status: object) -> bool:
    if isinstance(status, ExperimentStatus):
        coerced = status
    else:
        text = str(status).strip()
        if text.startswith("ExperimentStatus."):
            text = text.split(".", 1)[1]
        try:
            coerced = ExperimentStatus(text)
        except ValueError:
            try:
                coerced = ExperimentStatus[text.upper()]
            except KeyError:
                return False
    return coerced in {
        ExperimentStatus.COMPLETED,
        ExperimentStatus.FAILED,
        ExperimentStatus.CANCELLED,
        ExperimentStatus.TIMEOUT,
        ExperimentStatus.OOM,
        ExperimentStatus.KILLED,
    }


__all__ = ["RelaunchFormScreen"]

"""Relaunch confirmation modal for dashboard v4."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.screen import ModalScreen
from textual.widgets import Button, Label

from slurminator.dashboard_v4.commands import submit_command
from slurminator.experiments import ExperimentStatus


class RelaunchFormScreen(ModalScreen[None]):
    """Confirm relaunching a terminal experiment."""

    BINDINGS = [("escape", "app.pop_screen", "Back")]

    def __init__(self, exp: dict[str, Any]) -> None:
        super().__init__()
        self.exp = exp

    def on_mount(self) -> None:
        """Focus the primary relaunch action."""
        confirm = self.query_one("#confirm-relaunch", Button)
        if confirm.disabled:
            self.query_one("#return-relaunch", Button).focus()
        else:
            confirm.focus()

    def compose(self) -> ComposeResult:
        """Render relaunch confirmation content."""
        experiment_id = self.exp.get("experiment_id", "selected run")
        relaunchable = _is_relaunchable(self.exp.get("status"))
        message = (
            "Queue a new submission for this terminal run."
            if relaunchable
            else "Only terminal runs can be relaunched. Cancel active runs first."
        )
        yield Container(
            Label(f"Relaunch {experiment_id}", id="relaunch-title"),
            Label(message, id="relaunch-message"),
            Horizontal(
                Button("Confirm relaunch", id="confirm-relaunch", variant="primary", disabled=not relaunchable),
                Button("Return", id="return-relaunch"),
                id="relaunch-buttons",
            ),
            id="relaunch-form",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Write a relaunch command when confirmed."""
        if event.button.id == "confirm-relaunch":
            target = {"experiment_id": self.exp.get("experiment_id")}
            if self.exp.get("job_id") is not None:
                target["job_id"] = self.exp.get("job_id")
            submit_command(self.app.command_save_path(), "relaunch_run", target)
            self.app.pop_screen()
        elif event.button.id == "return-relaunch":
            self.app.pop_screen()


def _is_relaunchable(status: object) -> bool:
    coerced = _coerce_status(status)
    if coerced is None:
        return False
    return coerced in _RELAUNCHABLE_STATUSES


def _coerce_status(status: object) -> ExperimentStatus | None:
    if isinstance(status, ExperimentStatus):
        return status
    text = str(status).strip()
    if text.startswith("ExperimentStatus."):
        text = text.split(".", 1)[1]
    normalized = text.upper().rstrip("+*")
    if normalized.startswith("CANCELED") or normalized.startswith("CANCELLED"):
        return ExperimentStatus.CANCELLED
    try:
        return ExperimentStatus(text)
    except ValueError:
        try:
            return ExperimentStatus[normalized]
        except KeyError:
            return None


_RELAUNCHABLE_STATUSES = {
    ExperimentStatus.COMPLETED,
    ExperimentStatus.FAILED,
    ExperimentStatus.CANCELLED,
    ExperimentStatus.TIMEOUT,
    ExperimentStatus.OOM,
    ExperimentStatus.KILLED,
}


__all__ = ["RelaunchFormScreen"]

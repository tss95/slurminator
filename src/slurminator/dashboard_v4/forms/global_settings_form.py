"""Global Slurm settings form for dashboard v4."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label

from slurminator.dashboard_v4.commands import submit_command


class GlobalSettingsFormScreen(ModalScreen[None]):
    """Edit Slurm resource settings for pending next submissions."""

    BINDINGS = [("escape", "app.pop_screen", "Back")]

    def on_mount(self) -> None:
        """Focus the first editable field."""
        self.query_one("#global-settings-time-hours", Input).focus()

    def compose(self) -> ComposeResult:
        """Render the global Slurm settings form."""
        yield Container(
            Label("Slurm overrides", id="global-settings-title"),
            Label(
                "Applies to pending/partial next submissions. Blank fields are left unchanged.",
                id="global-settings-message",
            ),
            Label("Walltime override (hours)", classes="global-settings-field-label"),
            Input(placeholder="blank = unchanged", id="global-settings-time-hours"),
            Label("Memory override (GB)", classes="global-settings-field-label"),
            Input(placeholder="blank = unchanged", id="global-settings-memory-gb"),
            Label("GPU count override", classes="global-settings-field-label"),
            Input(placeholder="blank = unchanged", id="global-settings-gpu-count"),
            Container(
                Button("Apply overrides", id="apply-global-settings", variant="primary"),
                Button("Clear overrides", id="clear-global-settings"),
                Button("Return", id="return-global-settings"),
                id="global-settings-actions",
            ),
            id="global-settings-form",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Dispatch global settings form actions."""
        if event.button.id == "apply-global-settings":
            self._submit_settings(self._settings_payload())
        elif event.button.id == "clear-global-settings":
            self._submit_settings({"time_hours": None, "memory_gb": None, "gpu_count": None})
        elif event.button.id == "return-global-settings":
            self.app.pop_screen()

    def on_input_submitted(self, _event: Input.Submitted) -> None:
        """Apply settings when Enter is pressed in an editable field."""
        self._submit_settings(self._settings_payload())

    def _submit_settings(self, settings: dict[str, str | None]) -> None:
        submit_command(
            self.app.command_save_path(), "update_global_run_settings", {"scope": "pending", "settings": settings}
        )
        self.app.pop_screen()

    def _settings_payload(self) -> dict[str, str]:
        payload: dict[str, str] = {}
        values = {
            "time_hours": self.query_one("#global-settings-time-hours", Input).value,
            "memory_gb": self.query_one("#global-settings-memory-gb", Input).value,
            "gpu_count": self.query_one("#global-settings-gpu-count", Input).value,
        }
        for key, value in values.items():
            text = value.strip()
            if text:
                payload[key] = text
        return payload


__all__ = ["GlobalSettingsFormScreen"]

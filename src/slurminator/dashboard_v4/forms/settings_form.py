"""Per-run settings form for dashboard v4."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from textual.app import ComposeResult
from textual.containers import Container
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label

from slurminator.dashboard_v4.commands import submit_command


class SettingsFormScreen(ModalScreen[None]):
    """Edit settings that affect the selected run's next submission."""

    BINDINGS = [("escape", "app.pop_screen", "Back")]

    def __init__(self, exp: dict[str, Any]) -> None:
        super().__init__()
        self.exp = exp

    def on_mount(self) -> None:
        """Focus the first editable field."""
        self.query_one("#settings-time-hours", Input).focus()

    def compose(self) -> ComposeResult:
        """Render editable run settings."""
        experiment_id = self.exp.get("experiment_id", "selected run")
        yield Container(
            Label(f"Settings {experiment_id}", id="settings-title"),
            Label("Applies to the next submission", id="settings-message"),
            Label("Walltime override (hours)", classes="settings-field-label"),
            Input(
                value=_string_value(_time_hours_value(self.exp)),
                placeholder="blank = default",
                id="settings-time-hours",
            ),
            Label("Memory override (GB)", classes="settings-field-label"),
            Input(
                value=_string_value(_resource_override_value(self.exp, "memory_gb", "mem_gb")),
                placeholder="blank = default",
                id="settings-memory-gb",
            ),
            Label("GPU count override", classes="settings-field-label"),
            Input(
                value=_string_value(_resource_override_value(self.exp, "gpu_count")),
                placeholder="blank = default",
                id="settings-gpu-count",
            ),
            Label("Pinned HPC", classes="settings-field-label"),
            Input(
                value=str(self.exp.get("pinned_hpc") or ""),
                placeholder="blank = scheduler choice",
                id="settings-pinned-hpc",
            ),
            Container(
                Button("Save settings", id="save-settings", variant="primary"),
                Button("Clear overrides", id="clear-settings"),
                Button("Return", id="return-settings"),
                id="settings-actions",
            ),
            id="settings-form",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Dispatch selected settings action."""
        if event.button.id == "save-settings":
            self._submit_settings()
        elif event.button.id == "clear-settings":
            self._clear_settings()
        elif event.button.id == "return-settings":
            self.app.pop_screen()

    def on_input_submitted(self, _event: Input.Submitted) -> None:
        """Save settings when Enter is pressed in an editable field."""
        self._submit_settings()

    def _submit_settings(self) -> None:
        submit_command(
            self.app.command_save_path(),
            "update_run_settings",
            {"experiment_id": self.exp.get("experiment_id"), "settings": self._settings_payload()},
        )
        self.app.pop_screen()

    def _clear_settings(self) -> None:
        submit_command(
            self.app.command_save_path(),
            "update_run_settings",
            {
                "experiment_id": self.exp.get("experiment_id"),
                "settings": {"time_hours": None, "memory_gb": None, "gpu_count": None, "pinned_hpc": None},
            },
        )
        self.app.pop_screen()

    def _settings_payload(self) -> dict[str, str | None]:
        return {
            "time_hours": _blank_to_none(self.query_one("#settings-time-hours", Input).value),
            "memory_gb": _blank_to_none(self.query_one("#settings-memory-gb", Input).value),
            "gpu_count": _blank_to_none(self.query_one("#settings-gpu-count", Input).value),
            "pinned_hpc": _blank_to_none(self.query_one("#settings-pinned-hpc", Input).value),
        }


def _time_hours_value(exp: Mapping[str, Any]) -> object:
    if exp.get("time_hours_override") is not None:
        return exp.get("time_hours_override")
    return _resource_override_value(exp, "time_hours")


def _resource_override_value(exp: Mapping[str, Any], key: str, *aliases: str) -> object:
    overrides = exp.get("resource_overrides")
    if not isinstance(overrides, Mapping):
        return ""
    for item in (key, *aliases):
        if overrides.get(item) is not None:
            return overrides.get(item)
    return ""


def _string_value(value: object) -> str:
    return "" if value is None else str(value)


def _blank_to_none(value: str) -> str | None:
    text = value.strip()
    return text or None


__all__ = ["SettingsFormScreen"]

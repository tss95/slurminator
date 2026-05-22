"""Concurrency limit modal for dashboard v4."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Container
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label

from slurminator.dashboard_v4.commands import submit_command


class ConcurrencyFormScreen(ModalScreen[None]):
    """Edit per-cluster concurrency limits for the running orchestrator."""

    BINDINGS = [("escape", "app.pop_screen", "Back")]

    def __init__(self) -> None:
        super().__init__()
        self._input_ids_by_hpc: dict[str, str] = {}

    def on_mount(self) -> None:
        """Focus the first limit field when available."""
        first_input = next(iter(self._input_ids_by_hpc.values()), None)
        if first_input is not None:
            self.query_one(f"#{first_input}", Input).focus()
        else:
            self.query_one("#apply-concurrency-limits", Button).focus()

    def compose(self) -> ComposeResult:
        """Render one numeric field per configured HPC limit."""
        yield Container(
            Label("Concurrency limits", id="concurrency-title"),
            Label(
                "Only connected HPCs are editable. Enter or Apply saves for the next poll.", id="concurrency-message"
            ),
            *self._limit_widgets(),
            Label("", id="concurrency-error"),
            Container(
                Button("Apply limits", id="apply-concurrency-limits", variant="primary"),
                Button("Return", id="return-concurrency"),
                id="concurrency-buttons",
            ),
            id="concurrency-form",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Dispatch concurrency form actions."""
        if event.button.id == "apply-concurrency-limits":
            self._submit_limits()
        elif event.button.id == "return-concurrency":
            self.app.pop_screen()

    def on_input_submitted(self, _event: Input.Submitted) -> None:
        """Apply limits when Enter is pressed in a field."""
        self._submit_limits()

    def _submit_limits(self) -> None:
        limits = self._read_limits()
        if limits is None:
            return
        for hpc, limit in limits.items():
            submit_command(self.app.command_save_path(), "set_concurrency_limit", {"hpc": hpc, "limit": limit})
        self.app.pop_screen()

    def _limit_widgets(self) -> list[Label | Input]:
        limits = _orchestrator_limits(getattr(self.app, "orchestrator", None))
        self._input_ids_by_hpc = {}
        widgets: list[Label | Input] = []
        if not limits:
            widgets.append(Label("No connected HPC limits", classes="concurrency-field-label"))
            return widgets
        for hpc, limit in limits:
            input_id = f"concurrency-limit-{hpc.lower()}"
            self._input_ids_by_hpc[hpc] = input_id
            widgets.append(Label(hpc, classes="concurrency-field-label"))
            widgets.append(Input(value=str(limit), placeholder="0 = disabled", id=input_id))
        return widgets

    def _read_limits(self) -> dict[str, int] | None:
        limits: dict[str, int] = {}
        for hpc, input_id in self._input_ids_by_hpc.items():
            raw_value = self.query_one(f"#{input_id}", Input).value.strip()
            try:
                limit = int(raw_value)
            except ValueError:
                self._set_error(f"{hpc} limit must be a non-negative integer")
                return None
            if limit < 0:
                self._set_error(f"{hpc} limit must be a non-negative integer")
                return None
            limits[hpc] = limit
        self._set_error("")
        return limits

    def _set_error(self, message: str) -> None:
        self.query_one("#concurrency-error", Label).update(message)


def _orchestrator_limits(orchestrator: Any) -> list[tuple[str, int]]:
    raw_limits = getattr(orchestrator, "concurrency_limits", {}) if orchestrator is not None else {}
    connected_hpcs = _connected_hpcs(orchestrator)
    limits: list[tuple[str, int]] = []
    for key, value in raw_limits.items():
        if connected_hpcs is not None and key not in connected_hpcs:
            continue
        hpc = getattr(key, "name", str(key))
        try:
            limit = int(value)
        except (TypeError, ValueError):
            limit = 0
        limits.append((hpc, limit))
    return sorted(limits, key=lambda item: item[0])


def _connected_hpcs(orchestrator: Any) -> set[Any] | None:
    connection_manager = getattr(orchestrator, "connection_manager", None)
    connected = getattr(connection_manager, "_connected", None)
    if not isinstance(connected, dict):
        return None
    return {hpc_type for hpc_type, is_connected in connected.items() if bool(is_connected)}


__all__ = ["ConcurrencyFormScreen"]

"""Concurrency limit modal for dashboard v4."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Container
from textual.screen import ModalScreen
from textual.widgets import Input, Label, ListItem, ListView

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
            self.query_one("#concurrency-actions", ListView).focus()

    def compose(self) -> ComposeResult:
        """Render one numeric field per configured HPC limit."""
        yield Container(
            Label("Concurrency limits", id="concurrency-title"),
            Label("Applies on the next orchestrator poll", id="concurrency-message"),
            *self._limit_widgets(),
            Label("", id="concurrency-error"),
            ListView(
                ListItem(Label("Save limits"), id="save-limits"),
                ListItem(Label("Return"), id="return"),
                id="concurrency-actions",
            ),
            id="concurrency-form",
        )

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Dispatch concurrency form actions."""
        if event.item.id == "save-limits":
            limits = self._read_limits()
            if limits is None:
                return
            for hpc, limit in limits.items():
                submit_command(self.app.command_save_path(), "set_concurrency_limit", {"hpc": hpc, "limit": limit})
            self.app.pop_screen()
        elif event.item.id == "return":
            self.app.pop_screen()

    def _limit_widgets(self) -> list[Label | Input]:
        limits = _orchestrator_limits(getattr(self.app, "orchestrator", None))
        self._input_ids_by_hpc = {}
        widgets: list[Label | Input] = []
        if not limits:
            widgets.append(Label("No configured HPC limits", classes="concurrency-field-label"))
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
    limits: list[tuple[str, int]] = []
    for key, value in raw_limits.items():
        hpc = getattr(key, "name", str(key))
        try:
            limit = int(value)
        except (TypeError, ValueError):
            limit = 0
        limits.append((hpc, limit))
    return sorted(limits, key=lambda item: item[0])


__all__ = ["ConcurrencyFormScreen"]

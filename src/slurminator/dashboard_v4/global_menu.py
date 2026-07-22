"""Global action modal for dashboard v4."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container
from textual.screen import ModalScreen
from textual.widgets import Label, ListItem, ListView

from slurminator.dashboard_v4.commands import submit_command
from slurminator.dashboard_v4.forms.concurrency_form import ConcurrencyFormScreen
from slurminator.dashboard_v4.forms.global_settings_form import GlobalSettingsFormScreen
from slurminator.dashboard_v4.forms.table_sort_form import TableSortFormScreen
from slurminator.dashboard_v4.help_screen import HelpScreen
from slurminator.dashboard_v4.keystrokes import GLOBAL_MENU_BINDINGS


class GlobalMenuScreen(ModalScreen[None]):
    """Modal menu for dashboard-wide actions."""

    BINDINGS = GLOBAL_MENU_BINDINGS

    def on_mount(self) -> None:
        """Focus the action list for keyboard-driven operation."""
        self.query_one("#global-actions", ListView).focus()

    def compose(self) -> ComposeResult:
        """Render global actions."""
        yield Container(
            Label("Global actions", id="global-title"),
            ListView(
                ListItem(Label(self._pause_resume_label()), id="toggle-submissions"),
                ListItem(Label("Set concurrency limits"), id="set-concurrency"),
                ListItem(Label("Set Slurm overrides"), id="set-global-settings"),
                ListItem(Label("Set table sort"), id="set-table-sort"),
                ListItem(Label("Cancel all queued/running jobs"), id="cancel-all"),
                ListItem(Label("Help"), id="help"),
                ListItem(Label("Close"), id="close-global-menu"),
                id="global-actions",
            ),
            id="global-menu",
        )

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Dispatch selected global action."""
        if event.item.id == "toggle-submissions":
            self._toggle_submissions()
            self.app.pop_screen()
        elif event.item.id == "set-concurrency":
            self.app.push_screen(ConcurrencyFormScreen())
        elif event.item.id == "set-global-settings":
            self.app.push_screen(GlobalSettingsFormScreen())
        elif event.item.id == "set-table-sort":
            self.app.push_screen(TableSortFormScreen())
        elif event.item.id == "cancel-all":
            submit_command(self.app.command_save_path(), "cancel_all", {"scope": "session"})
            self.app.pop_screen()
        elif event.item.id == "help":
            self.app.push_screen(HelpScreen())
        elif event.item.id == "close-global-menu":
            self.app.pop_screen()

    def _toggle_submissions(self) -> None:
        orchestrator = getattr(self.app, "orchestrator", None)
        action = (
            "resume_submissions" if bool(getattr(orchestrator, "submissions_paused", False)) else "pause_submissions"
        )
        submit_command(self.app.command_save_path(), action, {})

    def _pause_resume_label(self) -> str:
        orchestrator = getattr(self.app, "orchestrator", None)
        return "Resume submissions" if bool(getattr(orchestrator, "submissions_paused", False)) else "Pause submissions"


__all__ = ["GlobalMenuScreen"]

"""Home screen for the Textual dashboard."""

from __future__ import annotations

from collections import Counter
from typing import Any

from textual.app import ComposeResult
from textual.containers import Container
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Static

from slurminator.dashboard_v4.commands import submit_command
from slurminator.dashboard_v4.keystrokes import HOME_BINDINGS
from slurminator.dashboard_v4.per_run_menu import PerRunMenuScreen
from slurminator.dashboard_v4.widgets import ExperimentsTable
from slurminator.experiments import ExperimentStatus


class HomeScreen(Screen[None]):
    """Primary v4 dashboard screen."""

    BINDINGS = HOME_BINDINGS

    def compose(self) -> ComposeResult:
        """Compose the dashboard home screen."""
        yield Header()
        yield Container(
            Static("", id="summary"), ExperimentsTable(id="exps"), Static("", id="quota"), id="home-content"
        )
        yield Footer()

    def on_mount(self) -> None:
        """Start periodic table refreshes from the orchestrator snapshot."""
        self.set_interval(getattr(self.app, "refresh_interval", 1.0), self.refresh_from_orchestrator)
        self.refresh_from_orchestrator()
        self.query_one(ExperimentsTable).focus()

    def on_data_table_row_selected(self, _event: DataTable.RowSelected) -> None:
        """Open the placeholder modal when Enter selects the current table row."""
        self.action_drill_in()

    def refresh_from_orchestrator(self) -> None:
        """Refresh summary, table, and footer from the app snapshot."""
        experiments = self.app.get_dashboard_snapshot()
        self.query_one("#summary", Static).update(self._summary_text(experiments))
        self.query_one(ExperimentsTable).update_experiments(
            experiments, show_sparkline=bool(getattr(self.app, "sparkline_enabled", False))
        )
        self.query_one("#quota", Static).update(self._quota_footer_text(experiments))

    def action_cursor_up(self) -> None:
        """Move the table cursor up."""
        self.query_one(ExperimentsTable).action_cursor_up()

    def action_cursor_down(self) -> None:
        """Move the table cursor down."""
        self.query_one(ExperimentsTable).action_cursor_down()

    def action_drill_in(self) -> None:
        """Open the placeholder per-run modal for the selected row."""
        experiments = self.app.get_dashboard_snapshot()
        exp = self.query_one(ExperimentsTable).selected_experiment(experiments)
        if exp is not None:
            self.app.push_screen(PerRunMenuScreen(exp))

    def action_toggle_pause(self) -> None:
        """Write a pause/resume command for the next orchestrator poll."""
        orchestrator = getattr(self.app, "orchestrator", None)
        action = (
            "resume_submissions" if bool(getattr(orchestrator, "submissions_paused", False)) else "pause_submissions"
        )
        submit_command(self.app.command_save_path(), action, {})

    def action_quit(self) -> None:
        """Close the dashboard UI without stopping the orchestrator."""
        self.app.request_dashboard_exit()

    def action_noop(self) -> None:
        """Ignore Escape on the home screen for now."""
        return None

    def _summary_text(self, experiments: list[dict[str, Any]]) -> str:
        counts = Counter(_status_text(exp.get("status")) for exp in experiments)
        total = len(experiments)
        done = sum(
            counts[state] for state in {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OOM", "OUT_OF_MEMORY", "KILLED"}
        )
        running = counts["RUNNING"]
        queued = counts["QUEUED"]
        pending = counts["PENDING"]
        paused = (
            "paused"
            if bool(getattr(getattr(self.app, "orchestrator", None), "submissions_paused", False))
            else "active"
        )
        return f"Experiments {done}/{total} done | running {running} | queued {queued} | pending {pending} | {paused}"

    def _quota_footer_text(self, experiments: list[dict[str, Any]]) -> str:
        orchestrator = getattr(self.app, "orchestrator", None)
        paused = "paused" if bool(getattr(orchestrator, "submissions_paused", False)) else "active"
        limit_parts = []
        limits = getattr(orchestrator, "concurrency_limits", {}) if orchestrator is not None else {}
        for hpc, limit in sorted(limits.items(), key=lambda item: str(getattr(item[0], "value", item[0]))):
            try:
                if int(limit) <= 0:
                    continue
            except (TypeError, ValueError):
                continue
            limit_parts.append(f"{getattr(hpc, 'value', hpc)}={limit}")
        limits_text = ", ".join(limit_parts) if limit_parts else "no active limits"
        active_hpcs = sorted(
            {
                str(getattr(exp.get("hpc_assignment"), "value", exp.get("hpc_assignment")))
                for exp in experiments
                if exp.get("hpc_assignment")
            }
        )
        hpc_text = ", ".join(active_hpcs) if active_hpcs else "no assigned HPC"
        return f"Submissions: {paused} | Limits: {limits_text} | Quota: {hpc_text}"


def _status_text(status: object) -> str:
    if isinstance(status, ExperimentStatus):
        return status.name
    return str(status or "").upper()


__all__ = ["HomeScreen"]

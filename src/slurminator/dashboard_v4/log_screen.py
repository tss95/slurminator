"""Per-run live log screen for dashboard v4."""

from __future__ import annotations

from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Footer, Header, RichLog

from slurminator.dashboard_v4.keystrokes import LOG_BINDINGS


class PerRunLogScreen(Screen[None]):
    """Live-tail one run's Slurm stdout/stderr logs."""

    BINDINGS = LOG_BINDINGS

    def __init__(self, exp: dict[str, Any], *, lines: int = 500) -> None:
        super().__init__()
        self.exp = exp
        self.lines = lines
        self._offsets: dict[str, int] = {}
        self._last_log_text = ""
        self._auto_scroll = True

    def compose(self) -> ComposeResult:
        """Compose the log screen."""
        yield Header()
        yield RichLog(id="log", wrap=True, auto_scroll=True, highlight=False, markup=False)
        yield Footer()

    def on_mount(self) -> None:
        """Fetch the initial log tail and start live refreshes."""
        self.refresh_log(force=True)
        self.set_interval(getattr(self.app, "refresh_interval", 1.0), self.refresh_log)
        self.query_one("#log", RichLog).focus()

    def refresh_log(self, *, force: bool = False) -> None:
        """Read and append new log text."""
        latest = self._latest_snapshot_exp()
        if latest is not None:
            self.exp = latest

        orchestrator = getattr(self.app, "orchestrator", None)
        if orchestrator is None or not hasattr(orchestrator, "read_log_tail_for"):
            self._write_log_text("No data yet", force=True)
            return

        result = orchestrator.read_log_tail_for(self.exp, lines=self.lines, offsets=None if force else self._offsets)
        self._offsets = dict(result.offsets)
        if force:
            self._last_log_text = ""
            self._write_log_text(result.text or "No data yet", force=True)
            return
        if result.text:
            self._write_log_text(result.text)

    def action_refresh(self) -> None:
        """Force a full latest-tail re-read."""
        self.refresh_log(force=True)

    def action_scroll_up(self) -> None:
        """Scroll up and disable automatic bottom-following."""
        log = self.query_one("#log", RichLog)
        log.action_scroll_up()
        self._auto_scroll = False

    def action_scroll_down(self) -> None:
        """Scroll down and re-enable automatic bottom-following at the bottom."""
        log = self.query_one("#log", RichLog)
        log.action_scroll_down()
        if log.is_vertical_scroll_end:
            self._auto_scroll = True

    def _write_log_text(self, text: str, *, force: bool = False) -> None:
        log = self.query_one("#log", RichLog)
        if force:
            log.clear()
        if force:
            self._last_log_text = text
        else:
            self._last_log_text = f"{self._last_log_text}\n{text}".strip()
        log.write(Text.from_ansi(text), scroll_end=self._auto_scroll)
        if self._auto_scroll:
            log.scroll_end(animate=False, immediate=True)

    def _latest_snapshot_exp(self) -> dict[str, Any] | None:
        experiment_id = self.exp.get("experiment_id")
        for exp in self.app.get_dashboard_snapshot():
            if exp.get("experiment_id") == experiment_id:
                return exp
        return None


__all__ = ["PerRunLogScreen"]

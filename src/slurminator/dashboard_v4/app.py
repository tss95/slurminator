"""Textual app entry point for dashboard v4."""

from __future__ import annotations

import copy
import logging
import threading
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

from textual.app import App
from textual.widget import Widget

from slurminator.dashboard_v4.home_screen import HomeScreen

logger = logging.getLogger("slurminator")


class TextualDashboardApp(App[None]):
    """Textual dashboard implementing Slurminator's dashboard contract."""

    CSS = """
    HomeScreen {
        layout: vertical;
    }

    #home-content {
        height: 1fr;
        layout: vertical;
    }

    #summary {
        height: 1;
        padding: 0 1;
    }

    #exps {
        height: 1fr;
    }

    #quota {
        height: 1;
        padding: 0 1;
    }

    #per-run-menu {
        width: 48;
        height: auto;
        max-height: 18;
        border: solid $accent;
        background: $surface;
        padding: 1 2;
    }

    #per-run-title {
        text-style: bold;
        margin-bottom: 1;
    }

    PerRunPlotScreen {
        layout: vertical;
    }

    #plot-content {
        height: 1fr;
        layout: horizontal;
    }

    #metrics {
        width: 28;
        height: 1fr;
        border-right: solid $accent;
    }

    #plot {
        width: 1fr;
        height: 1fr;
        padding: 0 1;
    }

    PerRunDetailScreen,
    PerRunLogScreen {
        layout: vertical;
    }

    #detail-scroll {
        height: 1fr;
        padding: 0 1;
    }

    #log {
        height: 1fr;
        padding: 0 1;
    }
    """

    def __init__(
        self,
        n_recent: int = 0,
        refresh_interval: float = 1.0,
        ui_version: str = "v4",
        *,
        headless: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.n_recent = n_recent
        self.refresh_interval = refresh_interval
        self.ui_version = ui_version
        self.headless = headless
        self.orchestrator: Any | None = None
        self.sparkline_enabled = False
        self.dashboard_exit_requested = False
        self._dashboard_snapshot: list[dict[str, Any]] = []
        self._snapshot_lock = threading.Lock()
        self._run_thread: threading.Thread | None = None
        self._run_error: BaseException | None = None

    def on_mount(self) -> None:
        """Install the home screen when Textual starts."""
        self.push_screen(HomeScreen())

    def mount(self, *widgets: Any, before: int | str | Widget | None = None, after: int | str | Widget | None = None):
        """Support both Textual widget mounting and Slurminator dashboard mounting."""
        if len(widgets) == 1 and before is None and after is None and not isinstance(widgets[0], Widget):
            return _TextualDashboardSession(self, widgets[0])
        return super().mount(*widgets, before=before, after=after)

    def render(self, exps: list[dict[str, Any]]) -> None:
        """Receive an orchestrator poll-complete signal."""
        self.notify_poll_complete(exps)
        return None

    def update(self, _renderable: object = None) -> None:
        """Compatibility no-op for the Rich Live ``update`` call site."""
        return None

    def notify_poll_complete(self, exps: list[dict[str, Any]]) -> None:
        """Swap the latest experiment snapshot for Textual screens to read."""
        snapshot = copy.deepcopy(exps)
        if self.orchestrator is not None:
            self.orchestrator._dashboard_snapshot = snapshot
        with self._snapshot_lock:
            self._dashboard_snapshot = snapshot

    def get_dashboard_snapshot(self) -> list[dict[str, Any]]:
        """Return the latest dashboard snapshot."""
        orchestrator = self.orchestrator
        if orchestrator is not None:
            snapshot = getattr(orchestrator, "_dashboard_snapshot", None)
            if snapshot is not None:
                return list(snapshot)
        with self._snapshot_lock:
            return list(self._dashboard_snapshot)

    def command_save_path(self) -> Path:
        """Return the command queue root used by dashboard-side commands."""
        orchestrator = self.orchestrator
        if orchestrator is not None and getattr(orchestrator, "experiment_dir", None) is not None:
            return Path(orchestrator.experiment_dir)
        snapshot = self.get_dashboard_snapshot()
        for exp in snapshot:
            if exp.get("save_path"):
                return Path(str(exp["save_path"]))
        return Path.cwd()

    def request_dashboard_exit(self) -> None:
        """Exit the Textual UI while leaving the orchestrator loop alive."""
        self.dashboard_exit_requested = True
        self.exit()

    def _attach_orchestrator(self, orchestrator: Any) -> None:
        self.orchestrator = orchestrator
        if not hasattr(orchestrator, "_dashboard_snapshot"):
            orchestrator._dashboard_snapshot = []

    def _run_in_thread(self) -> None:
        try:
            self.run(headless=self.headless)
        except BaseException as exc:  # pragma: no cover - defensive dashboard isolation
            self._run_error = exc
            logger.warning(
                "Textual dashboard exited with an error; orchestrator will continue headless.", exc_info=True
            )

    def _start_background(self) -> None:
        if self._run_thread is not None and self._run_thread.is_alive():
            return
        self._run_thread = threading.Thread(target=self._run_in_thread, name="slurminator-dashboard-v4", daemon=True)
        self._run_thread.start()

    def _stop_background(self) -> None:
        if self.is_running:
            try:
                self.call_from_thread(self.exit)
            except RuntimeError:
                self.exit()
        if self._run_thread is not None:
            self._run_thread.join(timeout=2.0)


class _TextualDashboardSession(AbstractContextManager[TextualDashboardApp]):
    """Context manager returned by ``TextualDashboardApp.mount(orchestrator)``."""

    def __init__(self, app: TextualDashboardApp, orchestrator: Any) -> None:
        self.app = app
        self.orchestrator = orchestrator

    def __enter__(self) -> TextualDashboardApp:
        self.app._attach_orchestrator(self.orchestrator)
        self.app._start_background()
        return self.app

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.app._stop_background()
        return False


__all__ = ["TextualDashboardApp"]

"""Textual app entry point for dashboard v4."""

from __future__ import annotations

import copy
import logging
import os
import signal
import sys
import threading
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import Any, Iterator

from textual import events
from textual.app import App
from textual.geometry import Size
from textual.widget import Widget

from slurminator.dashboard_v4.home_screen import HomeScreen
from slurminator.dashboard_v4.widgets import SparklineThresholds

logger = logging.getLogger("slurminator")
CONSOLE_LOG_LEVEL_WHILE_ACTIVE = logging.WARNING

try:
    from textual.drivers.linux_driver import LinuxDriver
except Exception:  # pragma: no cover - platform/import defensive
    LinuxDriver = None  # type: ignore[assignment]


if LinuxDriver is not None:

    class ThreadFriendlyLinuxDriver(LinuxDriver):
        """Linux driver variant that tolerates Textual running outside the main thread."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            with suppress_thread_signal_registration():
                super().__init__(*args, **kwargs)

else:  # pragma: no cover - non-Linux defensive
    ThreadFriendlyLinuxDriver = None  # type: ignore[assignment]


class TextualDashboardApp(App[None]):
    """Textual dashboard implementing Slurminator's dashboard contract."""

    TITLE = "Slurminator"
    BINDINGS = [("?", "help", "Help")]

    CSS = """
    HomeScreen {
        layout: vertical;
        height: 100%;
        width: 100%;
    }

    #home-content {
        height: 1fr;
        min-height: 1;
        width: 100%;
        layout: vertical;
    }

    #summary {
        height: 1;
        padding: 0 1;
    }

    #progress-bars {
        height: 1;
        padding: 0 1;
    }

    #exps {
        height: 1fr;
        min-height: 1;
        width: 100%;
    }

    #quota {
        height: 3;
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

    #global-menu {
        width: 52;
        height: auto;
        max-height: 14;
        border: solid $accent;
        background: $surface;
        padding: 1 2;
    }

    #global-title {
        text-style: bold;
        margin-bottom: 1;
    }

    #relaunch-form {
        width: 58;
        height: auto;
        max-height: 12;
        border: solid $accent;
        background: $surface;
        padding: 1 2;
    }

    #relaunch-title {
        text-style: bold;
        margin-bottom: 1;
    }

    #relaunch-message {
        margin-bottom: 1;
    }

    #settings-form {
        width: 64;
        height: auto;
        max-height: 24;
        border: solid $accent;
        background: $surface;
        padding: 1 2;
    }

    #settings-title {
        text-style: bold;
        margin-bottom: 1;
    }

    #settings-message {
        margin-bottom: 1;
    }

    .settings-field-label {
        margin-top: 1;
    }

    #concurrency-form {
        width: 56;
        height: auto;
        max-height: 22;
        border: solid $accent;
        background: $surface;
        padding: 1 2;
    }

    #concurrency-title {
        text-style: bold;
        margin-bottom: 1;
    }

    #concurrency-message {
        margin-bottom: 1;
    }

    .concurrency-field-label {
        margin-top: 1;
    }

    #concurrency-error {
        color: $error;
        height: 1;
        margin-top: 1;
    }

    #concurrency-buttons {
        layout: horizontal;
        height: 3;
        margin-top: 1;
    }

    #concurrency-buttons Button {
        width: 1fr;
        margin-right: 1;
    }

    #help-screen {
        width: 64;
        height: auto;
        max-height: 30;
        border: solid $accent;
        background: $surface;
        padding: 1 2;
    }

    #help-title {
        text-style: bold;
        margin-bottom: 1;
    }

    #help-content {
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
        sparkline_thresholds: SparklineThresholds | object | None = None,
        **kwargs: Any,
    ) -> None:
        app_title = str(kwargs.pop("title", "Slurminator"))
        super().__init__(**kwargs)
        self.title = app_title
        if LinuxDriver is not None and self.driver_class is LinuxDriver:
            self.driver_class = ThreadFriendlyLinuxDriver
        warn_if_incompatible_term()
        self.n_recent = n_recent
        self.refresh_interval = refresh_interval
        self.ui_version = ui_version
        self.headless = headless
        self.orchestrator: Any | None = None
        self.sparkline_enabled = True
        self.sparkline_thresholds = sparkline_thresholds or SparklineThresholds()
        self.dashboard_exit_requested = False
        self._dashboard_snapshot: list[dict[str, Any]] = []
        self._snapshot_lock = threading.Lock()
        self._run_thread: threading.Thread | None = None
        self._run_error: BaseException | None = None
        self._last_terminal_size: os.terminal_size | None = None

    def on_mount(self) -> None:
        """Install the home screen when Textual starts."""
        self.push_screen(HomeScreen())
        self.set_interval(2.0, self._poll_terminal_size)

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
        """Request graceful shutdown of both Textual and the orchestrator loop."""
        self.dashboard_exit_requested = True
        if self.orchestrator is not None:
            setattr(self.orchestrator, "_dashboard_exit_requested", True)
        self.exit()

    def action_global_menu(self) -> None:
        """Open the global action menu."""
        from slurminator.dashboard_v4.global_menu import GlobalMenuScreen

        if isinstance(self.screen, GlobalMenuScreen):
            return
        self.push_screen(GlobalMenuScreen())

    def action_help(self) -> None:
        """Open the dashboard help overlay."""
        from slurminator.dashboard_v4.help_screen import HelpScreen

        if isinstance(self.screen, HelpScreen):
            return
        self.push_screen(HelpScreen())

    def _attach_orchestrator(self, orchestrator: Any) -> None:
        self.orchestrator = orchestrator
        if not hasattr(orchestrator, "_dashboard_snapshot"):
            orchestrator._dashboard_snapshot = []

    def _run_in_thread(self) -> None:
        try:
            with suppress_thread_signal_registration():
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

    def _poll_terminal_size(self) -> None:
        """Refresh layout when resize signals are not delivered."""
        try:
            size = os.get_terminal_size()
        except OSError:
            return
        if size == self._last_terminal_size:
            return
        self._last_terminal_size = size
        textual_size = Size(size.columns, size.lines)
        self.post_message(events.Resize(textual_size, textual_size))
        self.refresh(layout=True)


class _TextualDashboardSession(AbstractContextManager[TextualDashboardApp]):
    """Context manager returned by ``TextualDashboardApp.mount(orchestrator)``."""

    def __init__(self, app: TextualDashboardApp, orchestrator: Any) -> None:
        self.app = app
        self.orchestrator = orchestrator

    def __enter__(self) -> TextualDashboardApp:
        self.app._attach_orchestrator(self.orchestrator)
        self._log_context = quiet_dashboard_console_logs()
        self._log_context.__enter__()
        self.app._start_background()
        return self.app

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.app._stop_background()
        self._log_context.__exit__(exc_type, exc, traceback)
        return False


@contextmanager
def suppress_thread_signal_registration() -> Iterator[None]:
    """Ignore ``signal.signal`` calls that cannot work outside the main thread."""
    if threading.current_thread() is threading.main_thread():
        yield
        return

    original_signal = signal.signal

    def thread_safe_signal(signum: int, handler: Any) -> Any:
        try:
            return original_signal(signum, handler)
        except ValueError as exc:
            if "main thread" not in str(exc):
                raise
            try:
                return signal.getsignal(signum)
            except Exception:
                return None

    signal.signal = thread_safe_signal  # type: ignore[assignment]
    try:
        yield
    finally:
        signal.signal = original_signal  # type: ignore[assignment]


@contextmanager
def quiet_dashboard_console_logs(
    *, level: int = CONSOLE_LOG_LEVEL_WHILE_ACTIVE, logger_names: tuple[str, ...] = ("slurminator", "PMT")
) -> Iterator[None]:
    """Raise console log handler thresholds while the Textual dashboard owns the terminal."""
    states: list[tuple[logging.Handler, int]] = []
    for logger_name in logger_names:
        active_logger = logging.getLogger(logger_name)
        for handler in active_logger.handlers:
            if not _is_console_stream_handler(handler):
                continue
            states.append((handler, handler.level))
            handler.setLevel(level if handler.level <= logging.NOTSET else max(handler.level, level))
    try:
        yield
    finally:
        for handler, previous_level in states:
            handler.setLevel(previous_level)


def _is_console_stream_handler(handler: logging.Handler) -> bool:
    if not isinstance(handler, logging.StreamHandler):
        return False
    stream = getattr(handler, "stream", None)
    return stream in {sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__}


def warn_if_incompatible_term() -> None:
    """Log a hint for tmux TERM values known to break Textual resize handling."""
    current_term = os.environ.get("TERM", "")
    if current_term in ("screen-256color", "screen", "") or current_term.startswith("screen-"):
        logger.warning(
            "Detected TERM=%r. The v4 dashboard requires tmux-256color or "
            "xterm-256color for correct resize handling. See "
            "docs/slurminator_ui_v4_phase4_decisions.md for setup instructions.",
            current_term or "<unset>",
        )


__all__ = ["TextualDashboardApp"]

"""Per-run live log screen for dashboard v4."""

from __future__ import annotations

from typing import Any

from rich.segment import Segment
from rich.style import Style
from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.dom import NoScreen
from textual.screen import Screen
from textual.strip import Strip
from textual.widgets import Footer, Header, RichLog, Static

from slurminator.dashboard_v4.clipboard import copy_text_to_clipboard
from slurminator.dashboard_v4.keystrokes import LOG_BINDINGS
from slurminator.dashboard_v4.terminal_mouse import set_terminal_mouse_reporting
from slurminator.log_gathering import LogSource

LOG_SELECTION_STYLE = Style(bgcolor="dark_green")
LOG_SELECTION_BOUNDARY_STYLE = Style(color="black", bgcolor="yellow", bold=True)
LOG_GUTTER_STYLE = Style(color="bright_black")
LOG_SOURCE_ORDER: tuple[LogSource, ...] = ("stdout", "stderr", "combined")
LOG_SOURCE_LABELS: dict[LogSource, str] = {"stdout": "stdout", "stderr": "stderr", "combined": "stdout+stderr"}


class SelectableLog(RichLog):
    """RichLog with screen-owned log range selection highlighting."""

    def render_line(self, y: int):
        """Render a line with a fixed line-number gutter and optional selection styling."""
        scroll_x, scroll_y = self.scroll_offset
        line_index = int(scroll_y) + y
        if line_index >= len(self.lines):
            return super().render_line(y)

        gutter_width = self._line_number_gutter_width()
        content_width = max(self.scrollable_content_region.width - gutter_width, 1)
        strip = self._render_line(line_index, scroll_x, content_width).apply_style(self.rich_style)
        style_for_line = getattr(self.screen, "_selection_style_for_line", None)
        style = style_for_line(line_index) if callable(style_for_line) else None
        if style is not None:
            strip = strip.apply_style(style)
        prefix_style = style if style is not None else LOG_GUTTER_STYLE
        return Strip([Segment(self._line_number_prefix(line_index), prefix_style), *strip._segments])

    def _line_number_gutter_width(self) -> int:
        digits = max(3, len(str(max(len(self.lines), 1))))
        return digits + 3

    def _line_number_prefix(self, line_index: int) -> str:
        digits = self._line_number_gutter_width() - 3
        return f"{line_index + 1:>{digits}} │ "

    def watch_scroll_y(self, old_value: float, new_value: float) -> None:
        """Refresh selection help as the log scrolls."""
        super().watch_scroll_y(old_value, new_value)
        try:
            screen = self.screen
        except NoScreen:
            return
        scroll_changed = getattr(screen, "_on_log_scroll_changed", None)
        if callable(scroll_changed):
            scroll_changed()

    def _on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        """Mark the screen as manually scrolled before Textual animates the change."""
        scroll_intent = getattr(self.screen, "_on_log_scroll_intent", None)
        if callable(scroll_intent):
            scroll_intent("up")
        super()._on_mouse_scroll_up(event)

    def _on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        """Let Textual scroll, then resume live-follow if the user reached bottom."""
        super()._on_mouse_scroll_down(event)
        scroll_intent = getattr(self.screen, "_on_log_scroll_intent", None)
        if callable(scroll_intent):
            scroll_intent("down")


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
        self._pending_log_text = ""
        self._terminal_selection_mode = False
        self._selection_anchor_line: int | None = None
        self._selection_end_line: int | None = None
        self._last_selection_range: tuple[int, int] | None = None
        self.log_source: LogSource = "stdout"

    def compose(self) -> ComposeResult:
        """Compose the log screen."""
        yield Header()
        yield Static("", id="log-copy-help")
        yield SelectableLog(id="log", wrap=True, auto_scroll=False, highlight=False, markup=False)
        yield Footer()

    def on_mount(self) -> None:
        """Fetch the initial log tail and start live refreshes."""
        self.refresh_log(force=True)
        self.set_interval(getattr(self.app, "refresh_interval", 1.0), self.refresh_log)
        self.query_one("#log", RichLog).focus()
        self._update_selection_help()

    def on_unmount(self) -> None:
        """Restore app mouse reporting when leaving terminal selection mode."""
        self._set_terminal_selection_mode(False, notify=False, refresh=False)

    def refresh_log(self, *, force: bool = False) -> None:
        """Read and append new log text."""
        if self._terminal_selection_mode and not force:
            return

        latest = self._latest_snapshot_exp()
        if latest is not None:
            self.exp = latest

        orchestrator = getattr(self.app, "orchestrator", None)
        if orchestrator is None or not hasattr(orchestrator, "read_log_tail_for"):
            self._write_log_text("No data yet", force=True)
            return

        result = orchestrator.read_log_tail_for(
            self.exp, lines=self.lines, offsets=None if force else self._offsets, source=self.log_source
        )
        self._offsets = dict(result.offsets)
        if force:
            self._pending_log_text = ""
            self._last_log_text = ""
            self._write_log_text(result.text or self._empty_log_message(), force=True)
            return
        if result.text:
            if self._should_buffer_log_update():
                self._buffer_log_text(result.text)
                return
            self._flush_pending_log_text(result.text)

    def action_refresh(self) -> None:
        """Force a full latest-tail re-read."""
        self.refresh_log(force=True)

    def action_toggle_log_source(self) -> None:
        """Cycle through stdout, stderr, and combined log sources."""
        next_source = LOG_SOURCE_ORDER[(LOG_SOURCE_ORDER.index(self.log_source) + 1) % len(LOG_SOURCE_ORDER)]
        self.log_source = next_source
        self._offsets = {}
        self._pending_log_text = ""
        self._clear_log_selection(notify=False)
        self.app.notify(f"Showing {LOG_SOURCE_LABELS[next_source]}", timeout=2.0)
        self.refresh_log(force=True)

    def action_scroll_up(self) -> None:
        """Scroll up and disable automatic bottom-following."""
        log = self.query_one("#log", RichLog)
        log.action_scroll_up()
        self._auto_scroll = False
        self._update_selection_help()

    def action_scroll_down(self) -> None:
        """Scroll down and re-enable automatic bottom-following at the bottom."""
        log = self.query_one("#log", RichLog)
        log.action_scroll_down()
        if log.is_vertical_scroll_end:
            self._resume_live_follow()
            return
        self._update_selection_help()
        self.app.call_after_refresh(self._resume_live_follow_if_at_bottom)

    def action_copy_log(self) -> None:
        """Copy a log selection, selected Textual text, or the loaded log tail."""
        if self._selection_anchor_line is not None:
            self._copy_log_selection()
            return
        selected_text = self.get_selected_text()
        text = selected_text if selected_text else self._last_log_text
        text = text.rstrip("\n")
        if not text or self._is_empty_log_message(text):
            self.app.notify("No log text to copy", severity="warning", timeout=2.0)
            return
        copy_text_to_clipboard(self.app, text)
        copied = "selected log text" if selected_text else "loaded log tail"
        self.app.notify(f"Copied {copied}", timeout=2.0)

    def action_mark_log_selection(self) -> None:
        """Mark the current top visible line as selection start/end."""
        line_index = self._current_log_line_index()
        if line_index is None:
            self.app.notify("No log lines to select", severity="warning", timeout=2.0)
            return
        if self._selection_anchor_line is None or self._selection_end_line is not None:
            self._selection_anchor_line = line_index
            self._selection_end_line = None
            self._auto_scroll = False
            self.app.notify(f"Log selection started at line {line_index + 1}", timeout=2.0)
        else:
            self._selection_end_line = line_index
            start, end = self._selection_range()
            self.app.notify(f"Log selection marked: lines {start + 1}-{end + 1}", timeout=2.0)
        self._update_selection_help()

    def action_copy_log_selection(self) -> None:
        """Copy the active log line range."""
        self._copy_log_selection()

    def action_cancel_selection_or_back(self) -> None:
        """Cancel log selection first; otherwise return to the previous screen."""
        if self._selection_anchor_line is not None:
            self._clear_log_selection(notify=True)
            return
        self.app.pop_screen()

    def action_toggle_terminal_selection(self) -> None:
        """Toggle native terminal selection by temporarily releasing mouse reporting."""
        self._set_terminal_selection_mode(not self._terminal_selection_mode)

    def _set_terminal_selection_mode(self, enabled: bool, *, notify: bool = True, refresh: bool = True) -> None:
        if enabled == self._terminal_selection_mode:
            return
        if not set_terminal_mouse_reporting(self.app, enabled=not enabled):
            if notify:
                self.app.notify("Terminal mouse mode unavailable", severity="warning", timeout=2.0)
            return

        self._terminal_selection_mode = enabled
        if enabled:
            self._clear_log_selection(notify=False)
            self._auto_scroll = False
            if notify:
                self.app.notify("Mouse selection enabled; press m to restore dashboard scrolling", timeout=4.0)
            return

        if notify:
            self.app.notify("Dashboard mouse scrolling restored", timeout=2.0)
        self._resume_live_follow_if_at_bottom()
        if refresh:
            self.refresh_log()

    def _should_buffer_log_update(self) -> bool:
        if self._selection_anchor_line is not None:
            return True
        if not self._auto_scroll:
            return True
        try:
            log = self.query_one("#log", RichLog)
        except Exception:
            return False
        if self._log_scroll_target_is_above_bottom(log):
            return True
        return not log.is_vertical_scroll_end

    def _log_scroll_target_is_above_bottom(self, log: RichLog) -> bool:
        scroll_y = float(getattr(log, "scroll_y", 0) or 0)
        scroll_target_y = float(getattr(log, "scroll_target_y", scroll_y) or 0)
        max_scroll_y = float(getattr(log, "max_scroll_y", 0) or 0)
        return scroll_target_y < max_scroll_y - 0.5

    def _buffer_log_text(self, text: str) -> None:
        text = text.strip("\n")
        if not text:
            return
        if self._pending_log_text:
            self._pending_log_text = f"{self._pending_log_text}\n{text}"
        else:
            self._pending_log_text = text
        self._update_selection_help()

    def _flush_pending_log_text(self, text: str = "") -> None:
        chunks = [chunk for chunk in (self._pending_log_text, text.strip("\n")) if chunk]
        if not chunks:
            return
        self._pending_log_text = ""
        self._write_log_text("\n".join(chunks))

    def _resume_live_follow_if_at_bottom(self) -> None:
        try:
            at_bottom = self.query_one("#log", RichLog).is_vertical_scroll_end
        except Exception:
            return
        if at_bottom:
            self._resume_live_follow()

    def _resume_live_follow(self) -> None:
        if self._selection_anchor_line is not None:
            self._auto_scroll = False
            self._update_selection_help()
            return
        self._auto_scroll = True
        self._flush_pending_log_text()
        self._update_selection_help()

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
        self._update_selection_help()

    def _copy_log_selection(self) -> None:
        if self._selection_anchor_line is None:
            self.app.notify("No log range selected; press Space to start", severity="warning", timeout=2.0)
            return
        start, end = self._selection_range()
        rendered_lines = self._rendered_log_lines()
        selected_lines = [line.rstrip() for line in rendered_lines[start : end + 1]]
        text = "\n".join(selected_lines).strip("\n")
        if not text or self._is_empty_log_message(text):
            self.app.notify("No log text to copy", severity="warning", timeout=2.0)
            return
        copy_text_to_clipboard(self.app, text)
        self._last_selection_range = (start, end)
        self.app.notify(f"Copied log lines {start + 1}-{end + 1}", timeout=2.0)

    def _selection_range(self) -> tuple[int, int]:
        assert self._selection_anchor_line is not None
        end_line = self._selection_end_line
        if end_line is None:
            end_line = self._current_log_line_index()
        if end_line is None:
            end_line = self._selection_anchor_line
        max_line = max(len(self._rendered_log_lines()) - 1, 0)
        anchor = min(max(self._selection_anchor_line, 0), max_line)
        end = min(max(end_line, 0), max_line)
        return (anchor, end) if anchor <= end else (end, anchor)

    def _current_log_line_index(self) -> int | None:
        rendered_lines = self._rendered_log_lines()
        if not rendered_lines:
            return None
        log = self.query_one("#log", RichLog)
        scroll_y = float(getattr(log, "scroll_y", 0) or 0)
        scroll_target_y = float(getattr(log, "scroll_target_y", scroll_y) or 0)
        index = int(round(scroll_target_y if scroll_target_y != scroll_y else scroll_y))
        return min(max(index, 0), len(rendered_lines) - 1)

    def _rendered_log_lines(self) -> list[str]:
        log = self.query_one("#log", RichLog)
        return [line.text.rstrip() for line in getattr(log, "lines", [])]

    def _clear_log_selection(self, *, notify: bool) -> None:
        self._selection_anchor_line = None
        self._selection_end_line = None
        self._last_selection_range = None
        self._update_selection_help()
        if notify:
            self.app.notify("Log selection cancelled", timeout=2.0)
        self._resume_live_follow_if_at_bottom()

    def _update_selection_help(self) -> None:
        try:
            help_bar = self.query_one("#log-copy-help", Static)
        except Exception:
            return
        if self._selection_anchor_line is None:
            pending_line_count = self._pending_log_line_count()
            pending_status = (
                f" {pending_line_count} new log line{'s' if pending_line_count != 1 else ''} buffered; "
                "scroll to bottom to resume. "
                if pending_line_count
                else ""
            )
            help_bar.update(
                f"Source: {LOG_SOURCE_LABELS[self.log_source]} | "
                f"{pending_status}"
                "Space marks start at top visible line; scroll; Space marks end; y/c copies. "
                "c alone copies loaded tail."
            )
            self._refresh_log_selection()
            return
        start, end = self._selection_range()
        state = "marked" if self._selection_end_line is not None else "active"
        pending_line_count = self._pending_log_line_count()
        pending_status = (
            f" {pending_line_count} new log line{'s' if pending_line_count != 1 else ''} buffered."
            if pending_line_count
            else ""
        )
        help_bar.update(
            f"Source: {LOG_SOURCE_LABELS[self.log_source]} | Selection {state}: lines {start + 1}-{end + 1}. "
            f"{pending_status} Scroll to extend; Space marks end/restarts; y/c copies; Esc cancels."
        )
        self._refresh_log_selection()

    def _pending_log_line_count(self) -> int:
        return len(self._pending_log_text.splitlines()) if self._pending_log_text else 0

    def _empty_log_message(self) -> str:
        return f"No {LOG_SOURCE_LABELS[self.log_source]} log data yet"

    def _is_empty_log_message(self, text: str) -> bool:
        return text in {"No data yet", self._empty_log_message()}

    def _selection_style_for_line(self, line_index: int) -> Style | None:
        if self._selection_anchor_line is None:
            return None
        start, end = self._selection_range()
        if line_index < start or line_index > end:
            return None
        current_line = self._current_log_line_index()
        boundary_lines = {self._selection_anchor_line}
        if self._selection_end_line is not None:
            boundary_lines.add(self._selection_end_line)
        elif current_line is not None:
            boundary_lines.add(current_line)
        return LOG_SELECTION_BOUNDARY_STYLE if line_index in boundary_lines else LOG_SELECTION_STYLE

    def _on_log_scroll_changed(self) -> None:
        try:
            at_bottom = self.query_one("#log", RichLog).is_vertical_scroll_end
        except Exception:
            at_bottom = False
        if self._selection_anchor_line is not None:
            self._auto_scroll = False
        elif at_bottom:
            self._resume_live_follow()
            return
        else:
            self._auto_scroll = False
        self._update_selection_help()

    def _on_log_scroll_intent(self, direction: str) -> None:
        if direction == "up":
            self._auto_scroll = False
            self._update_selection_help()
            return
        self.app.call_after_refresh(self._resume_live_follow_if_at_bottom)

    def _refresh_log_selection(self) -> None:
        try:
            self.query_one("#log", RichLog).refresh()
        except Exception:
            return

    def _latest_snapshot_exp(self) -> dict[str, Any] | None:
        experiment_id = self.exp.get("experiment_id")
        for exp in self.app.get_dashboard_snapshot():
            if exp.get("experiment_id") == experiment_id:
                return exp
        return None


__all__ = ["PerRunLogScreen"]

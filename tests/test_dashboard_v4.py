import asyncio
import logging
import os
import re
import signal
import shlex
import sys
import threading
import time
from pathlib import Path

import pytest
from rich.text import Text
from textual import events
from textual.widgets import Input, Label

from slurminator.command_queue import Command
from slurminator.config import HPCType
from slurminator.dashboard_v4.app import TextualDashboardApp, suppress_thread_signal_registration
from slurminator.dashboard_v4.commands import submit_command
from slurminator.dashboard_v4.detail_screen import PerRunDetailScreen
from slurminator.dashboard_v4.forms.concurrency_form import ConcurrencyFormScreen
from slurminator.dashboard_v4.forms.relaunch_form import RelaunchFormScreen
from slurminator.dashboard_v4.forms.settings_form import SettingsFormScreen
from slurminator.dashboard_v4.global_menu import GlobalMenuScreen
from slurminator.dashboard_v4.help_screen import HelpScreen
from slurminator.dashboard_v4.log_screen import PerRunLogScreen
from slurminator.dashboard_v4 import plot_screen as plot_screen_module
from slurminator.dashboard_v4.per_run_menu import PerRunMenuScreen
from slurminator.dashboard_v4.plot_screen import PerRunPlotScreen
from slurminator.dashboard_v4.widgets.sparkline import render_sparkline, slope_color
from slurminator.dashboard_v4.widgets import ExperimentsTable
from slurminator.experiments import ExperimentStatus
from slurminator.hpc_orchestrator import HPCOrchestrator
from slurminator.schemas.status_schema import HistoryEntry

pytestmark = pytest.mark.unit


class FakeConnection:
    def __init__(self, history_payload: str = "", files: dict[str, str] | None = None) -> None:
        self.history_payload = history_payload
        self.files = files if files is not None else {}

    def run_command(self, _hpc_type, command, prefer_remote=False):  # noqa: ARG002
        payload = self._payload_for_command(command)
        if command.startswith("stat "):
            return str(len(payload.encode("utf-8"))), ""
        if command.startswith("tail "):
            byte_match = re.search(r"tail -c \+(\d+)", command)
            if byte_match is not None:
                start_index = int(byte_match.group(1)) - 1
                return payload.encode("utf-8")[start_index:].decode("utf-8"), ""
            line_match = re.search(r"tail -n (\d+)", command)
            if line_match is not None:
                n_lines = int(line_match.group(1))
                return "\n".join(payload.splitlines()[-n_lines:]) + ("\n" if payload.endswith("\n") else ""), ""
        return "", ""

    def close_all(self):
        return None

    def _payload_for_command(self, command: str) -> str:
        tokens = shlex.split(command)
        for token in reversed(tokens):
            if token.endswith(".out") or token.endswith(".err"):
                path = token
                break
            if token.endswith(".jsonl"):
                path = token
                break
        else:
            path = ""
        if path.endswith(".jsonl"):
            return self.history_payload
        for key, payload in self.files.items():
            if path == key or path.endswith(key):
                return payload
        return ""


def _orchestrator(tmp_path: Path, connection: FakeConnection | None = None) -> HPCOrchestrator:
    tmp_path.mkdir(parents=True, exist_ok=True)
    exp_file = tmp_path / "experiments.yaml"
    exp_file.write_text("experiments: []\n", encoding="utf-8")
    return HPCOrchestrator(
        str(exp_file),
        concurrency_limits={HPCType.OLIVIA: 1},
        connection_manager=connection or FakeConnection(),
        is_local_hpc_fn=lambda _hpc_type: False,
    )


def _history_line(*, epoch: int, loss: float, acc: float) -> dict:
    return {
        "timestamp": 100.0 + epoch,
        "attempt": 1,
        "epoch": epoch,
        "step": None,
        "metrics": {"loss": loss, "acc": acc},
    }


def _history_jsonl() -> str:
    lines = [
        HistoryEntry(**_history_line(epoch=1, loss=1.2, acc=0.4)).model_dump_json(),
        HistoryEntry(**_history_line(epoch=2, loss=0.8, acc=0.6)).model_dump_json(),
    ]
    return "\n".join(lines) + "\n"


def _has_styled_span(text: Text, substring: str, style: str) -> bool:
    start = text.plain.index(substring)
    end = start + len(substring)
    return any(span.start <= start and span.end >= end and str(span.style) == style for span in text.spans)


def _pending_commands(root: Path) -> list[Command]:
    pending = root / ".orchestrator_status" / "_commands" / "pending"
    return [Command.model_validate_json(path.read_text(encoding="utf-8")) for path in sorted(pending.glob("*.json"))]


def _experiments() -> list[dict]:
    return [
        {
            "experiment_id": "exp-1",
            "dataset_name": "dataset-a",
            "status": ExperimentStatus.RUNNING,
            "hpc_assignment": HPCType.OLIVIA,
            "job_id": "12345",
            "save_path": "/remote/save",
            "output_dir": "/remote/logs",
            "requested_time_hours": 2,
            "requested_gpu_count": 1,
            "running_timestamp": 100.0,
            "all_metrics": {"loss": 0.8, "acc": 0.6},
            "sweep_params": "lr=0.1",
            "sacct_snapshot": {"State": "RUNNING", "Elapsed": "00:01:00"},
            "links": {"wandb_url": "https://wandb.test/run"},
            "wandb_run_url": "https://wandb.test/top-level-run",
            "git_sha_at_submission": {"project": "abc123", "slurminator": "def456"},
            "notes": "watch validation loss",
            "current_epoch": 1,
            "max_epochs": 4,
            "target_metric_name": "loss",
            "target_metric_value": 0.5,
            "display_metric_info": {"loss": {"higher_better": False}, "acc": {"higher_better": True}},
            "history": [_history_line(epoch=1, loss=1.2, acc=0.4), _history_line(epoch=2, loss=0.8, acc=0.6)],
        },
        {
            "experiment_id": "exp-2",
            "dataset_name": "dataset-b",
            "status": ExperimentStatus.PENDING,
            "hpc_assignment": HPCType.OLIVIA,
            "history": [_history_line(epoch=1, loss=2.0, acc=0.2), _history_line(epoch=2, loss=1.5, acc=0.3)],
        },
    ]


def test_submit_command_writes_pending_command(tmp_path: Path) -> None:
    cmd = submit_command(tmp_path, "pause_submissions", {})

    pending = tmp_path / ".orchestrator_status" / "_commands" / "pending"
    files = list(pending.glob("*.json"))
    assert len(files) == 1
    assert not list(pending.glob("*.tmp"))
    assert cmd.command_id in files[0].read_text(encoding="utf-8")


def test_orchestrator_dashboard_resolver_keeps_v3_and_resolves_v4(tmp_path: Path) -> None:
    class PluginDashboard:
        pass

    orch_v3 = _orchestrator(tmp_path / "v3")
    orch_v3.dashboard_cls = PluginDashboard
    orch_v4 = _orchestrator(tmp_path / "v4")
    orch_v4.dashboard_cls = PluginDashboard
    orch_v4.dashboard_ui = "v4"

    assert orch_v3._resolve_dashboard_cls() is PluginDashboard
    assert orch_v4._resolve_dashboard_cls() is TextualDashboardApp


def test_textual_home_table_renders_and_cursor_moves(tmp_path: Path) -> None:
    async def run() -> None:
        orch = _orchestrator(tmp_path)
        exps = _experiments()
        orch._publish_dashboard_snapshot(exps)
        app = TextualDashboardApp(refresh_interval=0.05)
        app.orchestrator = orch

        async with app.run_test(size=(120, 32)) as pilot:
            await pilot.pause(0.2)
            table = app.screen.query_one(ExperimentsTable)
            assert table.row_count == 2
            assert table.get_row_at(0)[0] == "exp-1"
            assert len(table.get_row_at(0)) == 9
            assert table.cursor_row == 0

            await pilot.press("down")
            await pilot.pause(0.05)
            assert table.cursor_row == 1

            await pilot.press("up")
            await pilot.pause(0.05)
            assert table.cursor_row == 0

    asyncio.run(run())


def test_textual_app_title_is_slurminator() -> None:
    app = TextualDashboardApp(refresh_interval=0.05)

    assert app.title == "Slurminator"


def test_textual_home_renders_summary_progress_and_footer(tmp_path: Path) -> None:
    async def run() -> None:
        orch = _orchestrator(tmp_path)
        orch._publish_dashboard_snapshot(_experiments())
        app = TextualDashboardApp(refresh_interval=0.05)
        app.orchestrator = orch

        async with app.run_test(size=(120, 32)) as pilot:
            await pilot.pause(0.2)
            screen = app.screen
            summary = screen._last_summary_text.plain
            assert "Pending: 1" in summary
            assert "Running: 1" in summary
            assert "Completed: 0" in summary

            progress = screen._last_progress_text.plain
            assert "Completed" in progress
            assert "Progress" in progress
            assert "Running" in progress
            assert "1/1" in progress
            assert "12%" in progress
            assert "█" in progress

            footer = screen._last_footer_text.plain
            assert "0 / 2 completed" in footer
            assert "2 left" in footer
            assert "Submissions: active" in footer
            assert "Limits: OLIVIA=1" in footer
            assert "Host: OLIVIA" in footer
            assert "Experiment: experiments" in footer
            assert "Slurm: h=2h ram=auto gpu=1" in footer
            assert "GPU quota: OLIVIA unavailable" in footer
            footer_text = screen._last_footer_text
            assert _has_styled_span(footer_text, "Sweep:", "cyan")
            assert _has_styled_span(footer_text, "Submissions:", "yellow")
            assert _has_styled_span(footer_text, "active", "green")
            assert _has_styled_span(footer_text, "Limits:", "cyan")
            assert _has_styled_span(footer_text, "Host:", "yellow")
            assert _has_styled_span(footer_text, "Experiment:", "cyan")
            assert _has_styled_span(footer_text, "experiments", "dim")
            assert _has_styled_span(footer_text, "Slurm:", "cyan")
            assert _has_styled_span(footer_text, "GPU quota:", "yellow")

    asyncio.run(run())


def test_textual_home_table_uses_status_colors_metric_colors_and_v3_order(tmp_path: Path) -> None:
    async def run() -> None:
        orch = _orchestrator(tmp_path)
        exps = _experiments()
        exps[0]["display_metric_info"]["loss"].update({"shortform": "vloss", "threshold": 1.0})
        completed = dict(exps[1])
        completed.update(
            {
                "experiment_id": "exp-3",
                "status": ExperimentStatus.COMPLETED,
                "target_metric_name": "acc",
                "target_metric_value": 0.9,
                "display_metric_info": {"acc": {"shortform": "vacc", "higher_better": True, "threshold": 0.5}},
            }
        )
        orch._publish_dashboard_snapshot([exps[1], completed, exps[0]])
        app = TextualDashboardApp(refresh_interval=0.05)
        app.orchestrator = orch

        async with app.run_test(size=(120, 32)) as pilot:
            await pilot.pause(0.2)
            table = app.screen.query_one(ExperimentsTable)
            assert [table.get_row_at(row)[0] for row in range(table.row_count)] == ["exp-1", "exp-3", "exp-2"]
            status_cell = table.get_row_at(0)[3]
            primary_cell = table.get_row_at(0)[5]
            assert isinstance(status_cell, Text)
            assert status_cell.plain == "RUNNING"
            assert status_cell.style == "green"
            assert isinstance(primary_cell, Text)
            assert primary_cell.plain == "0.5000"
            assert primary_cell.style == "green"
            assert table.get_row_at(0)[4] == "1/4  25.0%"
            assert str(list(table.columns.values())[5].label) == "vloss"

    asyncio.run(run())


def test_textual_home_table_renders_sparkline_and_toggles_column(tmp_path: Path) -> None:
    async def run() -> None:
        orch = _orchestrator(tmp_path)
        orch._publish_dashboard_snapshot(_experiments())
        app = TextualDashboardApp(refresh_interval=0.05)
        app.orchestrator = orch

        async with app.run_test(size=(120, 32)) as pilot:
            await pilot.pause(0.2)
            table = app.screen.query_one(ExperimentsTable)
            trajectory = table.get_row_at(0)[6]
            assert isinstance(trajectory, Text)
            assert trajectory.style == "dim"
            assert trajectory.plain.endswith("." * 18)

            await pilot.press("s")
            await pilot.pause(0.1)
            assert app.sparkline_enabled is False
            assert len(table.get_row_at(0)) == 8

            await pilot.press("s")
            await pilot.pause(0.1)
            assert app.sparkline_enabled is True
            assert len(table.get_row_at(0)) == 9

    asyncio.run(run())


def test_textual_home_table_trajectory_resolves_shortform_history_metric(tmp_path: Path) -> None:
    async def run() -> None:
        orch = _orchestrator(tmp_path)
        exp = dict(_experiments()[0])
        exp["target_metric_name"] = "vloss"
        exp["target_metric_value"] = 0.8
        exp["display_metric_info"] = {"val/loss": {"shortform": "vloss", "higher_better": False}}
        exp["history"] = [
            {"timestamp": 1.0, "attempt": 1, "epoch": 1, "step": None, "metrics": {"val/loss": 1.2}},
            {"timestamp": 2.0, "attempt": 1, "epoch": 2, "step": None, "metrics": {"val/loss": 0.8}},
        ]
        orch._publish_dashboard_snapshot([exp])
        app = TextualDashboardApp(refresh_interval=0.05)
        app.orchestrator = orch

        async with app.run_test(size=(120, 32)) as pilot:
            await pilot.pause(0.2)
            table = app.screen.query_one(ExperimentsTable)
            trajectory = table.get_row_at(0)[6]
            assert isinstance(trajectory, Text)
            assert trajectory.plain != "-"

    asyncio.run(run())


def test_textual_home_table_trajectory_falls_back_to_history_metric(tmp_path: Path) -> None:
    async def run() -> None:
        orch = _orchestrator(tmp_path)
        exp = dict(_experiments()[0])
        exp.pop("target_metric_name", None)
        exp.pop("target_metric_value", None)
        exp["history"] = [
            {"timestamp": 1.0, "attempt": 1, "epoch": 1, "step": None, "metrics": {"loss": 1.2}},
            {"timestamp": 2.0, "attempt": 1, "epoch": 2, "step": None, "metrics": {"loss": 0.8}},
        ]
        orch._publish_dashboard_snapshot([exp])
        app = TextualDashboardApp(refresh_interval=0.05)
        app.orchestrator = orch

        async with app.run_test(size=(120, 32)) as pilot:
            await pilot.pause(0.2)
            table = app.screen.query_one(ExperimentsTable)
            trajectory = table.get_row_at(0)[6]
            assert isinstance(trajectory, Text)
            assert trajectory.plain != "-"

    asyncio.run(run())


def test_textual_global_menu_opens_and_escape_closes(tmp_path: Path) -> None:
    async def run() -> None:
        orch = _orchestrator(tmp_path)
        orch._publish_dashboard_snapshot(_experiments())
        app = TextualDashboardApp(refresh_interval=0.05)
        app.orchestrator = orch

        async with app.run_test(size=(120, 32)) as pilot:
            await pilot.pause(0.2)
            await pilot.press("g")
            await pilot.pause(0.1)
            assert isinstance(app.screen, GlobalMenuScreen)
            assert app.screen.query_one("#global-actions").children[0].id == "toggle-submissions"

            await pilot.press("escape")
            await pilot.pause(0.1)
            assert not isinstance(app.screen, GlobalMenuScreen)

    asyncio.run(run())


def test_textual_global_menu_pause_and_resume_commands_are_observed(tmp_path: Path) -> None:
    async def run() -> None:
        orch = _orchestrator(tmp_path)
        exps = _experiments()
        orch._publish_dashboard_snapshot(exps)
        app = TextualDashboardApp(refresh_interval=0.05)
        app.orchestrator = orch

        async with app.run_test(size=(120, 32)) as pilot:
            await pilot.pause(0.2)
            await pilot.press("g")
            await pilot.press("enter")
            await pilot.pause(0.1)
            assert orch._process_command_queue(exps) == 1
            assert orch.submissions_paused is True

            await pilot.press("g")
            await pilot.pause(0.1)
            actions = app.screen.query_one("#global-actions")
            assert actions.children[0].query_one(Label).content == "Resume submissions"
            await pilot.press("enter")
            await pilot.pause(0.1)
            assert orch._process_command_queue(exps) == 1
            assert orch.submissions_paused is False

    asyncio.run(run())


def test_textual_global_menu_cancel_all_writes_command(tmp_path: Path) -> None:
    async def run() -> None:
        orch = _orchestrator(tmp_path)
        orch._publish_dashboard_snapshot(_experiments())
        app = TextualDashboardApp(refresh_interval=0.05)
        app.orchestrator = orch

        async with app.run_test(size=(120, 32)) as pilot:
            await pilot.pause(0.2)
            await pilot.press("g")
            actions = app.screen.query_one("#global-actions")
            actions.index = 2
            await pilot.press("enter")
            await pilot.pause(0.1)
            commands = _pending_commands(tmp_path)
            assert len(commands) == 1
            assert commands[0].action == "cancel_all"
            assert commands[0].target == {"scope": "session"}

    asyncio.run(run())


def test_textual_global_menu_concurrency_form_writes_limit_commands(tmp_path: Path) -> None:
    async def run() -> None:
        orch = _orchestrator(tmp_path)
        orch._publish_dashboard_snapshot(_experiments())
        app = TextualDashboardApp(refresh_interval=0.05)
        app.orchestrator = orch

        async with app.run_test(size=(120, 32)) as pilot:
            await pilot.pause(0.2)
            await pilot.press("g")
            await pilot.pause(0.1)
            global_actions = app.screen.query_one("#global-actions")
            assert [child.id for child in global_actions.children] == [
                "toggle-submissions",
                "set-concurrency",
                "cancel-all",
                "help",
                "close-global-menu",
            ]
            global_actions.index = 1
            await pilot.press("enter")
            await pilot.pause(0.1)
            assert isinstance(app.screen, ConcurrencyFormScreen)
            app.screen.query_one("#concurrency-limit-olivia", Input).value = "3"
            form_actions = app.screen.query_one("#concurrency-actions")
            form_actions.index = 0
            form_actions.focus()
            await pilot.press("enter")
            await pilot.pause(0.1)

            commands = _pending_commands(tmp_path)
            assert len(commands) == 1
            assert commands[0].action == "set_concurrency_limit"
            assert commands[0].target == {"hpc": "OLIVIA", "limit": 3}
            assert orch._process_command_queue(_experiments()) == 1
            assert orch.concurrency_limits[HPCType.OLIVIA] == 3

    asyncio.run(run())


def test_textual_concurrency_form_rejects_invalid_limit(tmp_path: Path) -> None:
    async def run() -> None:
        orch = _orchestrator(tmp_path)
        app = TextualDashboardApp(refresh_interval=0.05)
        app.orchestrator = orch

        async with app.run_test(size=(120, 32)) as pilot:
            await app.push_screen(ConcurrencyFormScreen())
            await pilot.pause(0.1)
            app.screen.query_one("#concurrency-limit-olivia", Input).value = "-1"
            actions = app.screen.query_one("#concurrency-actions")
            actions.index = 0
            actions.focus()
            await pilot.press("enter")
            await pilot.pause(0.1)

            assert isinstance(app.screen, ConcurrencyFormScreen)
            assert "non-negative integer" in str(app.screen.query_one("#concurrency-error", Label).content)
            assert _pending_commands(tmp_path) == []

    asyncio.run(run())


def test_textual_help_opens_from_question_mark_and_global_menu(tmp_path: Path) -> None:
    async def run() -> None:
        orch = _orchestrator(tmp_path)
        orch._publish_dashboard_snapshot(_experiments())
        app = TextualDashboardApp(refresh_interval=0.05)
        app.orchestrator = orch

        async with app.run_test(size=(120, 32)) as pilot:
            await pilot.pause(0.2)
            await pilot.press("?")
            await pilot.pause(0.1)
            assert isinstance(app.screen, HelpScreen)
            await pilot.press("escape")
            await pilot.pause(0.1)
            assert not isinstance(app.screen, HelpScreen)

            await pilot.press("g")
            await pilot.pause(0.1)
            global_actions = app.screen.query_one("#global-actions")
            global_actions.index = 3
            await pilot.press("enter")
            await pilot.pause(0.1)
            assert isinstance(app.screen, HelpScreen)
            assert "Set concurrency limits" in str(app.screen.query_one("#help-content").render())

    asyncio.run(run())


def test_textual_global_menu_opens_from_per_run_menu(tmp_path: Path) -> None:
    async def run() -> None:
        orch = _orchestrator(tmp_path)
        orch._publish_dashboard_snapshot(_experiments())
        app = TextualDashboardApp(refresh_interval=0.05)
        app.orchestrator = orch

        async with app.run_test(size=(120, 32)) as pilot:
            await pilot.pause(0.2)
            await pilot.press("enter")
            await pilot.pause(0.1)
            assert isinstance(app.screen, PerRunMenuScreen)

            await pilot.press("g")
            await pilot.pause(0.1)
            assert isinstance(app.screen, GlobalMenuScreen)

    asyncio.run(run())


def test_sparkline_color_matches_metric_direction() -> None:
    improving_accuracy = render_sparkline([0.1, 0.2, 0.3, 0.4], higher_better=True)
    worsening_accuracy = render_sparkline([0.4, 0.3, 0.2, 0.1], higher_better=True)
    improving_loss = render_sparkline([1.0, 0.8, 0.6, 0.4], higher_better=False)

    assert improving_accuracy.style == "green"
    assert worsening_accuracy.style == "red"
    assert improving_loss.style == "green"
    assert slope_color([0.1, 0.4, 0.4, 0.1], higher_better=True) == "yellow"


def test_textual_dashboard_mount_render_contract(tmp_path: Path) -> None:
    orch = _orchestrator(tmp_path)
    app = TextualDashboardApp(refresh_interval=0.05, headless=True)
    exps = _experiments()

    with app.mount(orch) as live:
        live.update(app.render(exps))
        time.sleep(0.1)
        assert app.orchestrator is orch
        assert app.get_dashboard_snapshot()[0]["experiment_id"] == "exp-1"


def test_textual_dashboard_mount_quiets_console_logs(tmp_path: Path) -> None:
    logger = logging.getLogger("slurminator")
    previous_level = logger.level
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    try:
        orch = _orchestrator(tmp_path)
        app = TextualDashboardApp(refresh_interval=0.05, headless=True)
        with app.mount(orch):
            assert handler.level == logging.WARNING
        assert handler.level == logging.INFO
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)


def test_textual_dashboard_warns_for_screen_term(monkeypatch, caplog) -> None:
    monkeypatch.setenv("TERM", "screen-256color")
    caplog.set_level(logging.WARNING, logger="slurminator")

    TextualDashboardApp(refresh_interval=0.05)

    assert "Detected TERM='screen-256color'" in caplog.text
    assert "tmux-256color or xterm-256color" in caplog.text
    assert "docs/slurminator_ui_v4_phase4_decisions.md" in caplog.text


def test_textual_dashboard_does_not_warn_for_tmux_term(monkeypatch, caplog) -> None:
    monkeypatch.setenv("TERM", "tmux-256color")
    caplog.set_level(logging.WARNING, logger="slurminator")

    TextualDashboardApp(refresh_interval=0.05)

    assert "Detected TERM=" not in caplog.text


def test_textual_terminal_size_poll_refreshes_only_on_change(monkeypatch) -> None:
    monkeypatch.setenv("TERM", "tmux-256color")
    app = TextualDashboardApp(refresh_interval=0.05)
    refresh_calls: list[dict[str, object]] = []
    resize_messages: list[events.Resize] = []

    def fake_refresh(*args, **kwargs) -> None:  # noqa: ANN002, ANN003
        refresh_calls.append(dict(kwargs))

    def fake_post_message(message) -> bool:  # noqa: ANN001
        if isinstance(message, events.Resize):
            resize_messages.append(message)
        return True

    monkeypatch.setattr(app, "refresh", fake_refresh)
    monkeypatch.setattr(app, "post_message", fake_post_message)
    monkeypatch.setattr(os, "get_terminal_size", lambda: os.terminal_size((100, 40)))
    app._poll_terminal_size()
    app._poll_terminal_size()
    assert refresh_calls == [{"layout": True}]
    assert [(message.size.width, message.size.height) for message in resize_messages] == [(100, 40)]

    monkeypatch.setattr(os, "get_terminal_size", lambda: os.terminal_size((120, 50)))
    app._poll_terminal_size()
    assert refresh_calls == [{"layout": True}, {"layout": True}]
    assert [(message.size.width, message.size.height) for message in resize_messages] == [(100, 40), (120, 50)]


def test_textual_thread_signal_registration_is_ignored() -> None:
    result: dict[str, object] = {}
    signal_number = getattr(signal, "SIGTSTP", signal.SIGTERM)

    def register_signal_in_thread() -> None:
        try:
            with suppress_thread_signal_registration():
                result["previous"] = signal.signal(signal_number, lambda *_args: None)
        except BaseException as exc:  # pragma: no cover - assertion payload
            result["error"] = exc

    thread = threading.Thread(target=register_signal_in_thread)
    thread.start()
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert "error" not in result
    assert "previous" in result


def test_textual_run_in_thread_wraps_signal_registration(monkeypatch) -> None:
    app = TextualDashboardApp(refresh_interval=0.05)
    result: dict[str, object] = {}
    signal_number = getattr(signal, "SIGTTOU", signal.SIGTERM)

    def fake_run(*, headless: bool = False) -> None:
        result["headless"] = headless
        result["previous"] = signal.signal(signal_number, lambda *_args: None)

    monkeypatch.setattr(app, "run", fake_run)
    thread = threading.Thread(target=app._run_in_thread)
    thread.start()
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert app._run_error is None
    assert result["headless"] is False
    assert "previous" in result


def test_textual_home_p_does_not_write_pause_command(tmp_path: Path) -> None:
    async def run() -> None:
        orch = _orchestrator(tmp_path)
        exps = _experiments()
        exps[0]["save_path"] = str(tmp_path / "remote_save")
        orch._publish_dashboard_snapshot(exps)
        app = TextualDashboardApp(refresh_interval=0.05)
        app.orchestrator = orch

        async with app.run_test(size=(120, 32)) as pilot:
            await pilot.pause(0.2)
            await pilot.press("p")
            await pilot.pause(0.05)
            assert orch.submissions_paused is False
            assert _pending_commands(tmp_path) == []
            assert orch._process_command_queue(exps) == 0

    asyncio.run(run())


def test_textual_enter_opens_placeholder_modal_and_escape_closes(tmp_path: Path) -> None:
    async def run() -> None:
        orch = _orchestrator(tmp_path)
        orch._publish_dashboard_snapshot(_experiments())
        app = TextualDashboardApp(refresh_interval=0.05)
        app.orchestrator = orch

        async with app.run_test(size=(120, 32)) as pilot:
            await pilot.pause(0.2)
            await pilot.press("enter")
            await pilot.pause(0.1)
            assert isinstance(app.screen, PerRunMenuScreen)

            await pilot.press("escape")
            await pilot.pause(0.1)
            assert not isinstance(app.screen, PerRunMenuScreen)

    asyncio.run(run())


def test_textual_per_run_menu_labels_cancel_and_return_actions(tmp_path: Path) -> None:
    async def run() -> None:
        orch = _orchestrator(tmp_path)
        app = TextualDashboardApp(refresh_interval=0.05)
        app.orchestrator = orch

        async with app.run_test(size=(120, 36)) as pilot:
            await app.push_screen(PerRunMenuScreen(_experiments()[0]))
            await pilot.pause(0.1)
            actions = app.screen.query_one("#per-run-actions")
            assert [child.id for child in actions.children] == [
                "view-plots",
                "view-details",
                "view-log-tail",
                "cancel-run",
                "relaunch-run",
                "settings",
                "return",
            ]
            assert actions.children[3].query_one(Label).content == "Cancel selected run"
            assert actions.children[-1].query_one(Label).content == "Return"

            actions.index = 6
            await pilot.press("enter")
            await pilot.pause(0.1)
            assert not isinstance(app.screen, PerRunMenuScreen)
            assert _pending_commands(tmp_path) == []

    asyncio.run(run())


def test_textual_per_run_menu_opens_plot_screen_for_selected_run(tmp_path: Path) -> None:
    async def run() -> None:
        orch = _orchestrator(tmp_path)
        orch._publish_dashboard_snapshot(_experiments())
        app = TextualDashboardApp(refresh_interval=0.05)
        app.orchestrator = orch

        async with app.run_test(size=(120, 36)) as pilot:
            await pilot.pause(0.2)
            await pilot.press("down")
            await pilot.press("enter")
            await pilot.pause(0.1)
            assert isinstance(app.screen, PerRunMenuScreen)
            assert app.screen.query_one("#per-run-actions").children[0].id == "view-plots"

            await pilot.press("enter")
            await pilot.pause(0.2)
            assert isinstance(app.screen, PerRunPlotScreen)
            assert app.screen.exp["experiment_id"] == "exp-2"

            await pilot.press("escape")
            await pilot.pause(0.1)
            assert isinstance(app.screen, PerRunMenuScreen)

    asyncio.run(run())


def test_textual_per_run_menu_opens_detail_and_log_screens(tmp_path: Path) -> None:
    async def run() -> None:
        connection = FakeConnection(files={"slurm-12345.out": "started\n", "slurm-12345.err": ""})
        orch = _orchestrator(tmp_path, connection=connection)
        orch._publish_dashboard_snapshot(_experiments())
        app = TextualDashboardApp(refresh_interval=0.05)
        app.orchestrator = orch

        async with app.run_test(size=(120, 36)) as pilot:
            await pilot.pause(0.2)
            await pilot.press("enter")
            await pilot.pause(0.1)
            assert isinstance(app.screen, PerRunMenuScreen)
            actions = app.screen.query_one("#per-run-actions")
            assert [child.id for child in actions.children[:3]] == ["view-plots", "view-details", "view-log-tail"]

            actions.index = 1
            await pilot.press("enter")
            await pilot.pause(0.2)
            assert isinstance(app.screen, PerRunDetailScreen)
            assert app.screen.exp["experiment_id"] == "exp-1"

            await pilot.press("escape")
            await pilot.pause(0.1)
            assert isinstance(app.screen, PerRunMenuScreen)

            app.screen.query_one("#per-run-actions").index = 2
            await pilot.press("enter")
            await pilot.pause(0.2)
            assert isinstance(app.screen, PerRunLogScreen)
            assert app.screen.exp["experiment_id"] == "exp-1"

            await pilot.press("escape")
            await pilot.pause(0.1)
            assert isinstance(app.screen, PerRunMenuScreen)

    asyncio.run(run())


def test_textual_per_run_menu_cancel_writes_cancel_run_command(tmp_path: Path) -> None:
    async def run() -> None:
        orch = _orchestrator(tmp_path)
        app = TextualDashboardApp(refresh_interval=0.05)
        app.orchestrator = orch

        async with app.run_test(size=(120, 36)) as pilot:
            await app.push_screen(PerRunMenuScreen(_experiments()[0]))
            await pilot.pause(0.1)
            actions = app.screen.query_one("#per-run-actions")
            actions.index = 3
            await pilot.press("enter")
            await pilot.pause(0.1)

            commands = _pending_commands(tmp_path)
            assert len(commands) == 1
            assert commands[0].action == "cancel_run"
            assert commands[0].target == {"experiment_id": "exp-1", "job_id": "12345"}

    asyncio.run(run())


def test_textual_per_run_menu_relaunch_writes_relaunch_command(tmp_path: Path) -> None:
    async def run() -> None:
        orch = _orchestrator(tmp_path)
        app = TextualDashboardApp(refresh_interval=0.05)
        app.orchestrator = orch
        exp = dict(_experiments()[0])
        exp["status"] = ExperimentStatus.FAILED

        async with app.run_test(size=(120, 36)) as pilot:
            await app.push_screen(PerRunMenuScreen(exp))
            await pilot.pause(0.1)
            actions = app.screen.query_one("#per-run-actions")
            actions.index = 4
            await pilot.press("enter")
            await pilot.pause(0.1)
            assert isinstance(app.screen, RelaunchFormScreen)
            assert app.screen.query_one("#relaunch-actions").children[0].id == "confirm-relaunch"

            await pilot.press("enter")
            await pilot.pause(0.1)

            commands = _pending_commands(tmp_path)
            assert len(commands) == 1
            assert commands[0].action == "relaunch_run"
            assert commands[0].target == {"experiment_id": "exp-1", "job_id": "12345"}

    asyncio.run(run())


def test_textual_per_run_menu_settings_writes_update_command(tmp_path: Path) -> None:
    async def run() -> None:
        orch = _orchestrator(tmp_path)
        app = TextualDashboardApp(refresh_interval=0.05)
        app.orchestrator = orch
        exp = dict(_experiments()[0])
        exp["time_hours_override"] = 4
        exp["resource_overrides"] = {"memory_gb": 120, "gpu_count": 1}
        exp["pinned_hpc"] = "FOX"

        async with app.run_test(size=(120, 40)) as pilot:
            await app.push_screen(PerRunMenuScreen(exp))
            await pilot.pause(0.1)
            actions = app.screen.query_one("#per-run-actions")
            actions.index = 5
            await pilot.press("enter")
            await pilot.pause(0.1)
            assert isinstance(app.screen, SettingsFormScreen)

            assert app.screen.query_one("#settings-time-hours", Input).value == "4"
            assert app.screen.query_one("#settings-memory-gb", Input).value == "120"
            assert app.screen.query_one("#settings-gpu-count", Input).value == "1"
            assert app.screen.query_one("#settings-pinned-hpc", Input).value == "FOX"

            app.screen.query_one("#settings-time-hours", Input).value = "8"
            app.screen.query_one("#settings-memory-gb", Input).value = "240"
            app.screen.query_one("#settings-gpu-count", Input).value = "2"
            app.screen.query_one("#settings-pinned-hpc", Input).value = "OLIVIA"
            settings_actions = app.screen.query_one("#settings-actions")
            settings_actions.index = 0
            settings_actions.focus()
            await pilot.press("enter")
            await pilot.pause(0.1)

            commands = _pending_commands(tmp_path)
            assert len(commands) == 1
            assert commands[0].action == "update_run_settings"
            assert commands[0].target == {
                "experiment_id": "exp-1",
                "settings": {"time_hours": "8", "memory_gb": "240", "gpu_count": "2", "pinned_hpc": "OLIVIA"},
            }

    asyncio.run(run())


def test_textual_settings_form_clear_overrides_writes_clear_command(tmp_path: Path) -> None:
    async def run() -> None:
        orch = _orchestrator(tmp_path)
        app = TextualDashboardApp(refresh_interval=0.05)
        app.orchestrator = orch
        exp = dict(_experiments()[0])
        exp["time_hours_override"] = 4
        exp["resource_overrides"] = {"memory_gb": 120, "gpu_count": 1}
        exp["pinned_hpc"] = "FOX"

        async with app.run_test(size=(120, 40)) as pilot:
            await app.push_screen(SettingsFormScreen(exp))
            await pilot.pause(0.1)
            settings_actions = app.screen.query_one("#settings-actions")
            settings_actions.index = 1
            settings_actions.focus()
            await pilot.press("enter")
            await pilot.pause(0.1)

            commands = _pending_commands(tmp_path)
            assert len(commands) == 1
            assert commands[0].action == "update_run_settings"
            assert commands[0].target == {
                "experiment_id": "exp-1",
                "settings": {"time_hours": None, "memory_gb": None, "gpu_count": None, "pinned_hpc": None},
            }

    asyncio.run(run())


def test_textual_relaunch_form_blocks_active_run_confirmation(tmp_path: Path) -> None:
    async def run() -> None:
        orch = _orchestrator(tmp_path)
        app = TextualDashboardApp(refresh_interval=0.05)
        app.orchestrator = orch

        async with app.run_test(size=(120, 36)) as pilot:
            await app.push_screen(RelaunchFormScreen(_experiments()[0]))
            await pilot.pause(0.1)
            actions = app.screen.query_one("#relaunch-actions")
            assert [child.id for child in actions.children] == ["back"]

            await pilot.press("enter")
            await pilot.pause(0.1)
            assert _pending_commands(tmp_path) == []

    asyncio.run(run())


def test_textual_detail_screen_renders_required_sections(tmp_path: Path) -> None:
    async def run() -> None:
        orch = _orchestrator(tmp_path)
        app = TextualDashboardApp(refresh_interval=0.05)
        app.orchestrator = orch

        async with app.run_test(size=(120, 36)) as pilot:
            await app.push_screen(PerRunDetailScreen(_experiments()[0]))
            await pilot.pause(0.2)
            screen = app.screen
            assert isinstance(screen, PerRunDetailScreen)
            detail_text = screen._last_detail_text
            assert "experiment_id: exp-1" in detail_text
            assert "status: running" in detail_text
            assert "job_id: 12345" in detail_text
            assert "cluster: OLIVIA" in detail_text
            assert "walltime: requested=2h used=00:01:00" in detail_text
            assert "gpu_count: 1" in detail_text
            assert "acc: 0.6" in detail_text
            assert "loss: 0.8" in detail_text
            assert "sweep_params: lr=0.1" in detail_text
            assert "State: RUNNING" in detail_text
            assert "wandb_url: https://wandb.test/run" in detail_text
            assert "wandb_run_url: https://wandb.test/top-level-run" in detail_text
            assert "project: abc123" in detail_text
            assert "notes: watch validation loss" in detail_text

    asyncio.run(run())


def test_textual_detail_screen_empty_state_and_force_history_read(tmp_path: Path) -> None:
    async def run() -> None:
        connection = FakeConnection(history_payload=_history_jsonl())
        orch = _orchestrator(tmp_path, connection=connection)
        exp = {
            "experiment_id": "exp-force",
            "job_id": "12345",
            "hpc_assignment": HPCType.OLIVIA,
            "save_path": str(tmp_path),
        }
        app = TextualDashboardApp(refresh_interval=0.05)
        app.orchestrator = orch

        async with app.run_test(size=(120, 36)) as pilot:
            await app.push_screen(PerRunDetailScreen(exp))
            await pilot.pause(0.2)
            screen = app.screen
            assert isinstance(screen, PerRunDetailScreen)
            detail_text = screen._last_detail_text
            assert "job_id: 12345" in detail_text
            assert "cluster: OLIVIA" in detail_text
            assert "loss: 0.8" in detail_text
            assert "No data yet" in detail_text

    asyncio.run(run())


def test_textual_log_screen_tails_and_appends_new_lines(tmp_path: Path) -> None:
    async def run() -> None:
        files = {"slurm-12345.out": "out1\nout2\n", "slurm-12345.err": "err1\n"}
        connection = FakeConnection(files=files)
        orch = _orchestrator(tmp_path, connection=connection)
        app = TextualDashboardApp(refresh_interval=60.0)
        app.orchestrator = orch

        async with app.run_test(size=(120, 36)) as pilot:
            await app.push_screen(PerRunLogScreen(_experiments()[0]))
            await pilot.pause(0.2)
            screen = app.screen
            assert isinstance(screen, PerRunLogScreen)
            assert "out2" in screen._last_log_text
            assert "err1" in screen._last_log_text
            assert screen._offsets["stdout"] == len(files["slurm-12345.out"].encode("utf-8"))
            assert screen._offsets["stderr"] == len(files["slurm-12345.err"].encode("utf-8"))

            files["slurm-12345.out"] += "out3\n"
            screen.refresh_log()
            await pilot.pause(0.1)
            assert "out3" in screen._last_log_text

    asyncio.run(run())


def test_textual_log_screen_scrollback_preserves_position_when_scrolled_up(tmp_path: Path) -> None:
    async def run() -> None:
        files = {"slurm-12345.out": "line1\nline2\n", "slurm-12345.err": ""}
        connection = FakeConnection(files=files)
        orch = _orchestrator(tmp_path, connection=connection)
        app = TextualDashboardApp(refresh_interval=60.0)
        app.orchestrator = orch

        async with app.run_test(size=(120, 36)) as pilot:
            await app.push_screen(PerRunLogScreen(_experiments()[0]))
            await pilot.pause(0.2)
            screen = app.screen
            assert isinstance(screen, PerRunLogScreen)
            assert screen._auto_scroll is True

            screen.action_scroll_up()
            assert screen._auto_scroll is False
            files["slurm-12345.out"] += "line3\n"
            screen.refresh_log()
            await pilot.pause(0.1)
            assert "line3" in screen._last_log_text
            assert screen._auto_scroll is False

    asyncio.run(run())


def test_textual_log_screen_empty_state(tmp_path: Path) -> None:
    async def run() -> None:
        orch = _orchestrator(tmp_path)
        app = TextualDashboardApp(refresh_interval=0.05)
        app.orchestrator = orch

        async with app.run_test(size=(120, 36)) as pilot:
            await app.push_screen(PerRunLogScreen({"experiment_id": "empty"}))
            await pilot.pause(0.2)
            screen = app.screen
            assert isinstance(screen, PerRunLogScreen)
            assert screen._last_log_text == "No data yet"

    asyncio.run(run())


def test_textual_plot_screen_renders_metrics_and_toggles(tmp_path: Path) -> None:
    async def run() -> None:
        orch = _orchestrator(tmp_path)
        orch._publish_dashboard_snapshot(_experiments())
        app = TextualDashboardApp(refresh_interval=0.05)
        app.orchestrator = orch

        async with app.run_test(size=(120, 36)) as pilot:
            await app.push_screen(PerRunPlotScreen(_experiments()[0]))
            await pilot.pause(0.3)
            screen = app.screen
            assert isinstance(screen, PerRunPlotScreen)
            assert screen.metric_keys == ["acc", "loss"]
            assert screen.selected_metric == "acc"
            assert "exp-1 - acc" in screen._last_plot_text

            await pilot.press("down")
            await pilot.pause(0.1)
            assert screen.selected_metric == "loss"
            assert "exp-1 - loss" in screen._last_plot_text

            await pilot.press("l")
            await pilot.pause(0.1)
            assert screen.log_scale is True

            await pilot.press("b")
            await pilot.pause(0.1)
            assert screen.show_best_overlay is True
            assert screen._higher_better("loss") is False
            assert "best(loss)" in screen._last_plot_text

    asyncio.run(run())


def test_textual_plot_screen_defers_initial_draw_until_after_refresh(monkeypatch, tmp_path: Path) -> None:
    async def run() -> None:
        orch = _orchestrator(tmp_path)
        app = TextualDashboardApp(refresh_interval=0.05)
        app.orchestrator = orch
        monkeypatch.setattr(PerRunPlotScreen, "on_resize", lambda _self, _event: None)
        monkeypatch.setattr(PerRunPlotScreen, "on_list_view_highlighted", lambda _self, _event: None)
        exp = dict(_experiments()[0])
        exp.pop("job_id", None)
        screen = PerRunPlotScreen(exp)
        callbacks: list[object] = []
        redraws: list[str] = []
        original_redraw = screen._redraw_plot

        def fake_call_after_refresh(callback, *args, **kwargs) -> None:  # noqa: ANN001, ANN002, ANN003
            callbacks.append(callback)

        def spy_redraw() -> None:
            redraws.append("redraw")
            original_redraw()

        monkeypatch.setattr(screen, "call_after_refresh", fake_call_after_refresh)
        monkeypatch.setattr(screen, "_redraw_plot", spy_redraw)

        async with app.run_test(size=(120, 36)):
            await app.push_screen(screen)
            assert callbacks[-1] == spy_redraw
            assert redraws == []
            callbacks[-1]()
            assert redraws == ["redraw"]
            assert "exp-1 - acc" in screen._last_plot_text

    asyncio.run(run())


def test_textual_plot_screen_resize_redraws_at_new_dimensions(tmp_path: Path) -> None:
    async def run() -> None:
        orch = _orchestrator(tmp_path)
        orch._publish_dashboard_snapshot(_experiments())
        app = TextualDashboardApp(refresh_interval=0.05)
        app.orchestrator = orch

        async with app.run_test(size=(120, 36)) as pilot:
            await app.push_screen(PerRunPlotScreen(_experiments()[0]))
            await pilot.pause(0.3)
            screen = app.screen
            assert isinstance(screen, PerRunPlotScreen)
            initial_dimensions = screen._last_plot_dimensions
            assert initial_dimensions is not None

            await pilot.resize_terminal(140, 40)
            await pilot.pause(0.2)

            resized_dimensions = screen._last_plot_dimensions
            assert resized_dimensions is not None
            assert resized_dimensions[0] > initial_dimensions[0]
            assert resized_dimensions[1] >= initial_dimensions[1]
            assert "exp-1 - acc" in screen._last_plot_text

    asyncio.run(run())


def test_textual_plot_screen_empty_plotext_output_reports_failure(monkeypatch, tmp_path: Path) -> None:
    async def run() -> None:
        orch = _orchestrator(tmp_path)
        app = TextualDashboardApp(refresh_interval=0.05)
        app.orchestrator = orch
        exp = dict(_experiments()[0])
        exp.pop("job_id", None)
        monkeypatch.setattr(plot_screen_module.plt, "build", lambda: "")

        async with app.run_test(size=(120, 36)) as pilot:
            await app.push_screen(PerRunPlotScreen(exp))
            await pilot.pause(0.3)
            screen = app.screen
            assert isinstance(screen, PerRunPlotScreen)
            assert "plotext build returned empty output for acc" in screen._last_plot_text
            assert "points: 2" in screen._last_plot_text

    asyncio.run(run())


def test_textual_plot_screen_empty_state(tmp_path: Path) -> None:
    async def run() -> None:
        orch = _orchestrator(tmp_path)
        app = TextualDashboardApp(refresh_interval=0.05)
        app.orchestrator = orch

        async with app.run_test(size=(120, 36)) as pilot:
            await app.push_screen(PerRunPlotScreen({"experiment_id": "empty"}))
            await pilot.pause(0.2)
            screen = app.screen
            assert isinstance(screen, PerRunPlotScreen)
            assert screen._last_plot_text == "No history available"

    asyncio.run(run())


def test_textual_plot_screen_force_reads_history_on_mount(tmp_path: Path) -> None:
    async def run() -> None:
        connection = FakeConnection(history_payload=_history_jsonl())
        orch = _orchestrator(tmp_path, connection=connection)
        exp = {
            "experiment_id": "exp-force",
            "job_id": "12345",
            "hpc_assignment": HPCType.OLIVIA,
            "save_path": str(tmp_path),
        }
        app = TextualDashboardApp(refresh_interval=0.05)
        app.orchestrator = orch

        async with app.run_test(size=(120, 36)) as pilot:
            await app.push_screen(PerRunPlotScreen(exp))
            await pilot.pause(0.2)
            screen = app.screen
            assert isinstance(screen, PerRunPlotScreen)
            assert len(screen.history) == 2
            assert screen.metric_keys == ["acc", "loss"]

    asyncio.run(run())


def test_textual_q_requests_dashboard_exit(tmp_path: Path) -> None:
    async def run() -> None:
        orch = _orchestrator(tmp_path)
        orch._publish_dashboard_snapshot(_experiments())
        app = TextualDashboardApp(refresh_interval=0.05)
        app.orchestrator = orch

        async with app.run_test(size=(120, 32)) as pilot:
            await pilot.pause(0.2)
            await pilot.press("q")
            await pilot.pause(0.05)
            assert app.dashboard_exit_requested is True
            assert orch._dashboard_exit_requested is True

    asyncio.run(run())


def test_textual_q_does_not_request_dashboard_exit_from_modal(tmp_path: Path) -> None:
    async def run() -> None:
        orch = _orchestrator(tmp_path)
        orch._publish_dashboard_snapshot(_experiments())
        app = TextualDashboardApp(refresh_interval=0.05)
        app.orchestrator = orch

        async with app.run_test(size=(120, 32)) as pilot:
            await pilot.pause(0.2)
            await pilot.press("enter")
            await pilot.pause(0.1)
            assert isinstance(app.screen, PerRunMenuScreen)

            await pilot.press("q")
            await pilot.pause(0.05)
            assert app.dashboard_exit_requested is False
            assert getattr(orch, "_dashboard_exit_requested", False) is False

    asyncio.run(run())


def test_orchestrator_poll_sleep_wakes_when_dashboard_requests_exit(tmp_path: Path) -> None:
    class Dashboard:
        dashboard_exit_requested = False

    orch = _orchestrator(tmp_path)
    orch.poll_interval = 10
    dashboard = Dashboard()
    timer = threading.Timer(0.05, lambda: setattr(dashboard, "dashboard_exit_requested", True))

    start = time.monotonic()
    timer.start()
    try:
        assert orch._sleep_until_next_poll(dashboard) is True
    finally:
        timer.cancel()

    assert time.monotonic() - start < 1.0


def test_orchestrator_poll_sleep_wakes_when_orchestrator_exit_flag_is_set(tmp_path: Path) -> None:
    class Dashboard:
        dashboard_exit_requested = False

    orch = _orchestrator(tmp_path)
    orch.poll_interval = 10
    dashboard = Dashboard()
    timer = threading.Timer(0.05, lambda: setattr(orch, "_dashboard_exit_requested", True))

    start = time.monotonic()
    timer.start()
    try:
        assert orch._sleep_until_next_poll(dashboard) is True
    finally:
        timer.cancel()

    assert time.monotonic() - start < 1.0

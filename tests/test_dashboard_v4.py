import asyncio
import time
from pathlib import Path

import pytest

from slurminator.config import HPCType
from slurminator.dashboard_v4.app import TextualDashboardApp
from slurminator.dashboard_v4.commands import submit_command
from slurminator.dashboard_v4.per_run_menu import PerRunMenuScreen
from slurminator.dashboard_v4.widgets import ExperimentsTable
from slurminator.experiments import ExperimentStatus
from slurminator.hpc_orchestrator import HPCOrchestrator
from slurminator.ui_dashboard import TerminalDashboard

pytestmark = pytest.mark.unit


class FakeConnection:
    def run_command(self, _hpc_type, _command, prefer_remote=False):  # noqa: ARG002
        return "", ""

    def close_all(self):
        return None


def _orchestrator(tmp_path: Path) -> HPCOrchestrator:
    tmp_path.mkdir(parents=True, exist_ok=True)
    exp_file = tmp_path / "experiments.yaml"
    exp_file.write_text("experiments: []\n", encoding="utf-8")
    return HPCOrchestrator(str(exp_file), concurrency_limits={HPCType.OLIVIA: 1}, connection_manager=FakeConnection())


def _experiments() -> list[dict]:
    return [
        {
            "experiment_id": "exp-1",
            "dataset_name": "dataset-a",
            "status": ExperimentStatus.RUNNING,
            "hpc_assignment": HPCType.OLIVIA,
            "current_epoch": 1,
            "max_epochs": 4,
            "target_metric_name": "loss",
            "target_metric_value": 0.5,
        },
        {
            "experiment_id": "exp-2",
            "dataset_name": "dataset-b",
            "status": ExperimentStatus.PENDING,
            "hpc_assignment": HPCType.OLIVIA,
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
    orch_v3 = _orchestrator(tmp_path / "v3")
    orch_v4 = _orchestrator(tmp_path / "v4")
    orch_v4.dashboard_ui = "v4"

    assert orch_v3._resolve_dashboard_cls() is TerminalDashboard
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
            assert table.cursor_row == 0

            await pilot.press("down")
            await pilot.pause(0.05)
            assert table.cursor_row == 1

            await pilot.press("up")
            await pilot.pause(0.05)
            assert table.cursor_row == 0

    asyncio.run(run())


def test_textual_dashboard_mount_render_contract(tmp_path: Path) -> None:
    orch = _orchestrator(tmp_path)
    app = TextualDashboardApp(refresh_interval=0.05, headless=True)
    exps = _experiments()

    with app.mount(orch) as live:
        live.update(app.render(exps))
        time.sleep(0.1)
        assert app.orchestrator is orch
        assert app.get_dashboard_snapshot()[0]["experiment_id"] == "exp-1"


def test_textual_pause_resume_commands_are_observed_by_orchestrator(tmp_path: Path) -> None:
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
            assert orch._process_command_queue(exps) == 1
            assert orch.submissions_paused is True

            await pilot.press("p")
            await pilot.pause(0.05)
            assert orch._process_command_queue(exps) == 1
            assert orch.submissions_paused is False

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

    asyncio.run(run())

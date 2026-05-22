import time

import pytest

from slurminator.command_queue import Command
from slurminator.config import HPCType
from slurminator.experiments import ExperimentStatus
from slurminator.hpc_orchestrator import HPCOrchestrator

pytestmark = pytest.mark.unit


class FakeConnection:
    def __init__(self) -> None:
        self.commands: list[tuple[HPCType, str, bool]] = []

    def run_command(self, hpc_type, command, prefer_remote=False):
        self.commands.append((hpc_type, command, prefer_remote))
        return "", ""

    def close_all(self):
        return None


def _write_pending(save_path, command: Command) -> None:
    pending = save_path / ".orchestrator_status" / "_commands" / "pending"
    pending.mkdir(parents=True, exist_ok=True)
    (pending / f"{int(command.issued_at * 1000):013d}_{command.command_id[:8]}.json").write_text(
        command.model_dump_json(), encoding="utf-8"
    )


def test_orchestrator_process_command_queue_dispatches_cancel_run(tmp_path) -> None:
    exp_file = tmp_path / "experiments.yaml"
    exp_file.write_text("experiments: []", encoding="utf-8")
    connection = FakeConnection()
    orchestrator = HPCOrchestrator(str(exp_file), concurrency_limits={}, connection_manager=connection)
    exps = [
        {
            "experiment_id": "exp-1",
            "status": ExperimentStatus.RUNNING,
            "hpc_assignment": HPCType.OLIVIA,
            "job_id": "12345",
            "save_path": str(tmp_path),
        }
    ]
    _write_pending(
        tmp_path,
        Command(
            command_id="cancel-exp-1",
            issued_at=time.time(),
            issued_by="tester",
            action="cancel_run",
            target={"experiment_id": "exp-1", "job_id": "12345"},
        ),
    )

    assert orchestrator._process_command_queue(exps) == 1

    assert connection.commands == [(HPCType.OLIVIA, "scancel 12345", True)]


def test_orchestrator_process_command_queue_noops_without_commands(tmp_path) -> None:
    exp_file = tmp_path / "experiments.yaml"
    exp_file.write_text("experiments: []", encoding="utf-8")
    orchestrator = HPCOrchestrator(str(exp_file), concurrency_limits={}, connection_manager=FakeConnection())

    assert orchestrator._process_command_queue([]) == 0
    assert orchestrator.submissions_paused is False

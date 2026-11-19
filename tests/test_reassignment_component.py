from __future__ import annotations

import pytest

from slurminator.experiments import ExperimentStatus
from slurminator.config import HPCType
from slurminator.reassignment import ReassignmentContext, maybe_reassign_experiments
from slurminator.state_store import replace_exp_in_list

pytestmark = pytest.mark.unit


class _FakeConnection:
    def __init__(self) -> None:
        self.commands: list[tuple[HPCType, str, bool]] = []

    def run_command(self, hpc_type, cmd, prefer_remote=False):  # noqa: ANN001
        self.commands.append((hpc_type, cmd, prefer_remote))
        return "", ""


def _context(
    connection: _FakeConnection, saved: list[dict], *, pinned_datasets: list[str] | None = None
) -> ReassignmentContext:
    class _Cluster:
        pass

    fox = _Cluster()
    fox.pinned_datasets = pinned_datasets or []
    olivia = _Cluster()
    olivia.pinned_datasets = []

    return ReassignmentContext(
        concurrency_limits={HPCType.FOX: 1, HPCType.OLIVIA: 2},
        hpc_configs={HPCType.FOX: fox, HPCType.OLIVIA: olivia},
        connection_manager=connection,  # type: ignore[arg-type]
        max_unqueue_seconds=60,
        pending_status=ExperimentStatus.PENDING,
        queued_status=ExperimentStatus.QUEUED,
        partial_status=ExperimentStatus.PARTIAL,
        replace_exp_in_list=replace_exp_in_list,
        save_yaml=saved.append,
    )


def test_maybe_reassign_experiments_moves_stale_queued_job_without_orchestrator() -> None:
    experiments = [
        {
            "experiment_id": "move-me",
            "dataset_name": "HAR",
            "status": ExperimentStatus.QUEUED,
            "hpc_assignment": HPCType.FOX,
            "job_id": "123",
            "queued_timestamp": 100.0,
        }
    ]
    data = {"experiments": list(experiments)}
    concurrency_used = {HPCType.FOX: 1, HPCType.OLIVIA: 0}
    connection = _FakeConnection()
    saved: list[dict] = []

    maybe_reassign_experiments(experiments, concurrency_used, data, _context(connection, saved), now=200.0)

    exp = experiments[0]
    assert exp["status"] == ExperimentStatus.PENDING
    assert exp["hpc_assignment"] == HPCType.OLIVIA
    assert "job_id" not in exp
    assert "queued_timestamp" not in exp
    assert concurrency_used == {HPCType.FOX: 0, HPCType.OLIVIA: 1}
    assert connection.commands == [(HPCType.FOX, "scancel 123", True)]
    assert saved[0]["experiments"][0]["hpc_assignment"] == HPCType.OLIVIA


def test_maybe_reassign_experiments_skips_pinned_experiment_without_orchestrator() -> None:
    experiments = [
        {
            "experiment_id": "pinned",
            "dataset_name": "sleepEDF",
            "status": ExperimentStatus.QUEUED,
            "hpc_assignment": HPCType.FOX,
            "job_id": "123",
            "queued_timestamp": 100.0,
        }
    ]
    data = {"experiments": list(experiments)}
    concurrency_used = {HPCType.FOX: 1, HPCType.OLIVIA: 0}
    connection = _FakeConnection()
    saved: list[dict] = []

    maybe_reassign_experiments(
        experiments, concurrency_used, data, _context(connection, saved, pinned_datasets=["sleepEDF"]), now=200.0
    )

    assert experiments[0]["status"] == ExperimentStatus.QUEUED
    assert experiments[0]["hpc_assignment"] == HPCType.FOX
    assert connection.commands == []
    assert saved == []

"""Focused scheduler-polling behavior tests."""

from __future__ import annotations

from typing import Any

import pytest

from slurminator.config import HPCType
from slurminator.experiments import ExperimentStatus
from slurminator.scheduler_polling import update_scheduler_statuses

pytestmark = pytest.mark.unit


class _ConnectionManager:
    """Return one deterministic active Slurm state."""

    def __init__(self, *, configured: bool = True) -> None:
        self.configs = {HPCType.OLIVIA: object()} if configured else {}
        self.commands: list[str] = []

    def run_command(self, _hpc_type: HPCType, command: str, prefer_remote: bool = False) -> tuple[str, str]:
        self.commands.append(command)
        assert prefer_remote is True
        return "12345 R\n", ""


def test_active_jobs_remain_pollable_after_runtime_limit_is_lowered_to_zero() -> None:
    manager = _ConnectionManager()
    experiments: list[dict[str, Any]] = [
        {
            "experiment_id": "exp-1",
            "status": ExperimentStatus.QUEUED,
            "hpc_assignment": HPCType.OLIVIA,
            "job_id": "12345",
        }
    ]

    update_scheduler_statuses(
        experiments,
        connection_manager=manager,
        concurrency_limits={HPCType.OLIVIA: 0},
        gather_logs=lambda *_args: (_ for _ in ()).throw(AssertionError("running jobs need no log read")),
    )

    assert experiments[0]["status"] == ExperimentStatus.RUNNING
    assert len(manager.commands) == 1
    assert manager.commands[0].startswith("squeue ")


def test_active_jobs_on_an_unconfigured_hpc_are_not_polled() -> None:
    manager = _ConnectionManager(configured=False)
    experiments: list[dict[str, Any]] = [
        {
            "experiment_id": "exp-1",
            "status": ExperimentStatus.QUEUED,
            "hpc_assignment": HPCType.OLIVIA,
            "job_id": "12345",
        }
    ]

    update_scheduler_statuses(
        experiments, connection_manager=manager, concurrency_limits={HPCType.OLIVIA: 0}, gather_logs=lambda *_args: None
    )

    assert experiments[0]["status"] == ExperimentStatus.QUEUED
    assert manager.commands == []

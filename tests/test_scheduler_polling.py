"""Focused scheduler-polling behavior tests."""

from __future__ import annotations

from typing import Any

import pytest

from slurminator import scheduler_polling
from slurminator.config import HPCType
from slurminator.experiments import ExperimentStatus
from slurminator.scheduler_polling import poll_hpc, poll_hpc_result, update_scheduler_statuses

pytestmark = pytest.mark.unit


class _ConnectionManager:
    """Return configurable deterministic Slurm command results."""

    def __init__(
        self,
        *,
        configured: bool = True,
        squeue: tuple[str, str] | Exception = ("12345 R\n", ""),
        sacct: tuple[str, str] | Exception = ("", ""),
    ) -> None:
        self.configs = {HPCType.OLIVIA: object()} if configured else {}
        self.commands: list[str] = []
        self.squeue = squeue
        self.sacct = sacct

    def run_command(self, _hpc_type: HPCType, command: str, prefer_remote: bool = False) -> tuple[str, str]:
        self.commands.append(command)
        assert prefer_remote is True
        response = self.squeue if command.startswith("squeue ") else self.sacct
        if isinstance(response, Exception):
            raise response
        return response


def _experiment(
    *, experiment_id: str = "exp-1", job_id: str = "12345", status: ExperimentStatus = ExperimentStatus.QUEUED
) -> dict[str, Any]:
    return {"experiment_id": experiment_id, "status": status, "hpc_assignment": HPCType.OLIVIA, "job_id": job_id}


def test_active_jobs_remain_pollable_after_runtime_limit_is_lowered_to_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _ConnectionManager()
    experiments = [_experiment()]
    monkeypatch.setattr(scheduler_polling.time, "time", lambda: 1_700_000_123.0)

    result = update_scheduler_statuses(
        experiments,
        connection_manager=manager,
        concurrency_limits={HPCType.OLIVIA: 0},
        gather_logs=lambda *_args: (_ for _ in ()).throw(AssertionError("lifecycle polling must not read logs")),
    )

    assert result.succeeded is True
    assert len(result.transitions) == 1
    transition = result.transitions[0]
    assert transition.experiment_id == "exp-1"
    assert transition.job_id == "12345"
    assert transition.hpc_type == HPCType.OLIVIA
    assert transition.old_status == ExperimentStatus.QUEUED
    assert transition.new_status == ExperimentStatus.RUNNING
    assert transition.timestamp == 1_700_000_123.0
    assert transition.running_timestamp == 1_700_000_123.0
    assert transition.terminal_timestamp is None
    assert experiments[0]["status"] == ExperimentStatus.RUNNING
    assert experiments[0]["last_change_ts"] == 1_700_000_123.0
    assert experiments[0]["running_timestamp"] == 1_700_000_123.0
    assert len(manager.commands) == 1
    assert manager.commands[0].startswith("squeue ")


@pytest.mark.parametrize(
    ("scheduler_state", "expected_status", "timestamp_field"),
    [
        ("COMPLETED", ExperimentStatus.COMPLETED, "completed_timestamp"),
        ("FAILED", ExperimentStatus.FAILED, "failed_timestamp"),
        ("CANCELLED", ExperimentStatus.CANCELLED, "cancelled_timestamp"),
        ("TIMEOUT", ExperimentStatus.TIMEOUT, "timeout_timestamp"),
        ("OUT_OF_MEMORY", ExperimentStatus.OOM, "failed_timestamp"),
    ],
)
def test_terminal_transitions_set_local_timestamp_fields_without_gathering_logs(
    monkeypatch: pytest.MonkeyPatch, scheduler_state: str, expected_status: ExperimentStatus, timestamp_field: str
) -> None:
    manager = _ConnectionManager(squeue=("", ""), sacct=(f"12345 {scheduler_state}\n", ""))
    experiments = [_experiment(status=ExperimentStatus.RUNNING)]
    monkeypatch.setattr(scheduler_polling.time, "time", lambda: 2_000.0)

    result = update_scheduler_statuses(
        experiments,
        connection_manager=manager,
        gather_logs=lambda *_args: (_ for _ in ()).throw(AssertionError("terminal logs must be on demand")),
    )

    transition = result.transitions[0]
    assert result.succeeded is True
    assert transition.new_status == expected_status
    assert transition.timestamp_field == timestamp_field
    assert transition.terminal_timestamp == 2_000.0
    assert transition.row_patch() == {"status": expected_status, "last_change_ts": 2_000.0, timestamp_field: 2_000.0}
    assert experiments[0][timestamp_field] == 2_000.0


def test_squeue_failure_is_explicit_and_does_not_change_experiments() -> None:
    manager = _ConnectionManager(squeue=RuntimeError("scheduler unavailable"))
    experiments = [_experiment()]

    result = update_scheduler_statuses(experiments, connection_manager=manager)

    assert result.succeeded is False
    assert result.transitions == ()
    assert experiments[0]["status"] == ExperimentStatus.QUEUED
    assert len(result.failures) == 1
    failure = result.failures[0]
    assert failure.hpc_type == HPCType.OLIVIA
    assert failure.stage == "squeue"
    assert failure.job_ids == ("12345",)
    assert failure.error_type == "RuntimeError"
    assert failure.message == "scheduler unavailable"


def test_sacct_failure_keeps_known_squeue_transition_but_marks_poll_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _ConnectionManager(squeue=("111 R\n", ""), sacct=TimeoutError("accounting timed out"))
    experiments = [
        _experiment(experiment_id="exp-active", job_id="111"),
        _experiment(experiment_id="exp-missing", job_id="222", status=ExperimentStatus.RUNNING),
    ]
    monkeypatch.setattr(scheduler_polling.time, "time", lambda: 3_000.0)

    result = update_scheduler_statuses(experiments, connection_manager=manager)

    assert result.succeeded is False
    assert [transition.experiment_id for transition in result.transitions] == ["exp-active"]
    assert experiments[0]["status"] == ExperimentStatus.RUNNING
    assert experiments[1]["status"] == ExperimentStatus.RUNNING
    assert result.failures[0].stage == "sacct"
    assert result.failures[0].job_ids == ("222",)
    assert result.hpc_polls[0].states == {"111": "RUNNING"}
    assert result.hpc_polls[0].unresolved_job_ids == ("222",)


def test_successful_queries_with_unresolved_job_are_incomplete_verification() -> None:
    manager = _ConnectionManager(squeue=("", ""), sacct=("", ""))
    experiments = [_experiment()]

    result = update_scheduler_statuses(experiments, connection_manager=manager)

    assert result.succeeded is False
    assert result.failures == ()
    assert result.hpc_polls[0].unresolved_job_ids == ("12345",)
    assert experiments[0]["status"] == ExperimentStatus.QUEUED


def test_malformed_scheduler_output_is_an_explicit_poll_failure() -> None:
    manager = _ConnectionManager(squeue=("12345 R unexpected-column\n", ""))

    result = poll_hpc_result(manager, HPCType.OLIVIA, ["12345"])

    assert result.succeeded is False
    assert result.failure is not None
    assert result.failure.stage == "squeue"
    assert result.failure.error_type == "ValueError"
    assert "Malformed squeue row" in result.failure.message


def test_active_jobs_on_an_unconfigured_hpc_are_reported_as_skipped() -> None:
    manager = _ConnectionManager(configured=False)
    experiments = [_experiment()]

    result = update_scheduler_statuses(experiments, connection_manager=manager, concurrency_limits={HPCType.OLIVIA: 0})

    assert result.succeeded is False
    assert result.transitions == ()
    assert result.hpc_polls == ()
    assert len(result.skipped) == 1
    assert result.skipped[0].reason == "HPC is not configured"
    assert experiments[0]["status"] == ExperimentStatus.QUEUED
    assert manager.commands == []


def test_poll_hpc_compatibility_wrapper_returns_only_known_states() -> None:
    manager = _ConnectionManager(squeue=("111 R\n", ""), sacct=RuntimeError("accounting unavailable"))

    states = poll_hpc(manager, HPCType.OLIVIA, ["111", "222"])

    assert states == {"111": "RUNNING"}

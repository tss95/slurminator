"""Focused tests for queued experiment reassignment."""

from types import SimpleNamespace
from typing import Any

import pytest

from slurminator.reassignment import ReassignmentContext, maybe_reassign_experiments

pytestmark = pytest.mark.unit


class _ConnectionManager:
    def __init__(self) -> None:
        self.commands: list[tuple[str, str, bool]] = []

    def run_command(self, hpc_type: str, command: str, prefer_remote: bool = False) -> tuple[str, str]:
        self.commands.append((hpc_type, command, prefer_remote))
        return "", ""


def _replace_experiment(experiments: list[dict[str, Any]], updated: dict[str, Any]) -> list[dict[str, Any]]:
    return [updated if exp["experiment_id"] == updated["experiment_id"] else exp for exp in experiments]


def _context(
    connection_manager: _ConnectionManager,
    save_calls: list[dict[str, Any]],
    *,
    pinned_datasets: list[str] | None = None,
) -> ReassignmentContext:
    return ReassignmentContext(
        concurrency_limits={"busy": 1, "free": 1},
        hpc_configs={
            "busy": SimpleNamespace(pinned_datasets=pinned_datasets or []),
            "free": SimpleNamespace(pinned_datasets=[]),
        },
        connection_manager=connection_manager,
        max_unqueue_seconds=10,
        pending_status="pending",
        queued_status="queued",
        partial_status="partial",
        replace_exp_in_list=_replace_experiment,
        save_yaml=save_calls.append,
    )


def _stale_queued_experiment(*, dataset_name: str = "movable") -> dict[str, Any]:
    return {
        "experiment_id": "exp-1",
        "dataset_name": dataset_name,
        "hpc_assignment": "busy",
        "status": "queued",
        "job_id": "12345",
        "queued_timestamp": 1.0,
    }


def test_pinned_candidate_does_not_save_unchanged_ledger() -> None:
    experiment = _stale_queued_experiment(dataset_name="pinned")
    data = {"experiments": [experiment]}
    save_calls: list[dict[str, Any]] = []
    connection_manager = _ConnectionManager()

    maybe_reassign_experiments(
        [experiment],
        {"busy": 1, "free": 0},
        data,
        _context(connection_manager, save_calls, pinned_datasets=["pinned"]),
        now=100.0,
    )

    assert save_calls == []
    assert connection_manager.commands == []
    assert experiment["status"] == "queued"
    assert experiment["hpc_assignment"] == "busy"


def test_real_reassignment_saves_once_after_state_change() -> None:
    experiment = _stale_queued_experiment()
    data = {"experiments": [experiment]}
    save_calls: list[dict[str, Any]] = []
    connection_manager = _ConnectionManager()
    concurrency_used = {"busy": 1, "free": 0}

    maybe_reassign_experiments(
        [experiment], concurrency_used, data, _context(connection_manager, save_calls), now=100.0
    )

    assert save_calls == [data]
    assert connection_manager.commands == [("busy", "scancel 12345", True)]
    assert experiment["status"] == "pending"
    assert experiment["hpc_assignment"] == "free"
    assert "job_id" not in experiment
    assert "queued_timestamp" not in experiment
    assert concurrency_used == {"busy": 0, "free": 1}

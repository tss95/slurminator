from __future__ import annotations

from pathlib import Path

import pytest

import slurminator.hpc_orchestrator as hpc_orchestrator_module
from slurminator.config import HPCType
from slurminator.experiments import ExperimentStatus
from slurminator.experiments.yaml_utils import dump_yaml
from slurminator.hpc_orchestrator import HPCOrchestrator
from slurminator.scheduler_polling import SchedulerStatusUpdateResult, SchedulerTransition
from slurminator.status_ingest import StatusIngestContext

pytestmark = pytest.mark.unit


class FakeConnection:
    """Minimal connection manager used by direct orchestrator helper tests."""

    def close_all(self) -> None:
        """Match the connection-manager cleanup contract."""
        return None


def _orchestrator(tmp_path: Path, experiments: list[dict]) -> HPCOrchestrator:
    experiment_file = tmp_path / "experiments.yaml"
    dump_yaml({"experiments": experiments}, experiment_file)
    return HPCOrchestrator(
        str(experiment_file), concurrency_limits={HPCType.OLIVIA: 2}, connection_manager=FakeConnection()
    )


def test_catch_up_existing_state_verifies_scheduler_without_slow_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    orchestrator = _orchestrator(
        tmp_path,
        [
            {
                "experiment_id": "queued-1",
                "status": ExperimentStatus.QUEUED,
                "hpc_assignment": HPCType.OLIVIA,
                "job_id": "12345",
            },
            {"experiment_id": "pending-1", "status": ExperimentStatus.PENDING},
        ],
    )
    update_calls: list[list[dict]] = []
    transition = SchedulerTransition(
        experiment_id="queued-1",
        job_id="12345",
        hpc_type=HPCType.OLIVIA,
        old_status=ExperimentStatus.QUEUED,
        new_status=ExperimentStatus.COMPLETED,
        timestamp=100.0,
        timestamp_field="completed_timestamp",
    )

    def refresh_scheduler(experiments: list[dict]) -> SchedulerStatusUpdateResult:
        update_calls.append(experiments)
        experiments[0]["status"] = ExperimentStatus.COMPLETED
        experiments[0]["last_change_ts"] = 100.0
        return SchedulerStatusUpdateResult((transition,), ())

    monkeypatch.setattr(orchestrator, "_refresh_scheduler_statuses", refresh_scheduler)
    monkeypatch.setattr(
        orchestrator,
        "_update_telemetry",
        lambda *_args, **_kwargs: pytest.fail("startup scheduler verification must not ingest telemetry"),
    )
    monkeypatch.setattr(
        orchestrator,
        "_update_queue_estimates",
        lambda _exps: pytest.fail("startup scheduler verification must not query queue estimates"),
    )
    monkeypatch.setattr(
        orchestrator,
        "_save_yaml",
        lambda _data: pytest.fail("scheduler transitions are journaled without a full YAML checkpoint"),
    )

    result = orchestrator._catch_up_existing_state()

    assert result is not None
    assert result["experiments"][0]["status"] == ExperimentStatus.COMPLETED
    assert len(update_calls) == 1
    assert orchestrator._dashboard_snapshot[0]["status"] == ExperimentStatus.COMPLETED


def test_catch_up_existing_state_skips_io_without_active_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    orchestrator = _orchestrator(
        tmp_path,
        [
            {"experiment_id": "completed-1", "status": ExperimentStatus.COMPLETED, "job_id": "12345"},
            {"experiment_id": "pending-1", "status": ExperimentStatus.PENDING},
        ],
    )

    def unexpected_update(_exps: list[dict]) -> None:
        raise AssertionError("terminal and pending rows need no startup catch-up")

    def unexpected_save(_data: dict) -> None:
        raise AssertionError("no catch-up means no ledger save")

    monkeypatch.setattr(orchestrator, "_refresh_scheduler_statuses", unexpected_update)
    monkeypatch.setattr(orchestrator, "_save_yaml", unexpected_save)

    data = orchestrator._load_yaml()
    monkeypatch.setattr(
        orchestrator, "_load_yaml", lambda: pytest.fail("the already-loaded startup ledger should be reused")
    )

    assert orchestrator._catch_up_existing_state(data) is None


def test_orchestrator_telemetry_ingest_uses_batched_persistence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    orchestrator = _orchestrator(tmp_path, [])
    captured: dict[str, object] = {}

    def capture_context(exp: dict, context: StatusIngestContext) -> None:
        captured["exp"] = exp
        captured["persist_immediately"] = context.persist_immediately
        captured["include_history"] = context.include_history

    monkeypatch.setattr(hpc_orchestrator_module, "update_running_experiment_info", capture_context)
    exp = {"experiment_id": "running-1"}

    orchestrator._update_running_experiment_info(exp)

    assert captured == {"exp": exp, "persist_immediately": False, "include_history": True}

    orchestrator._update_running_experiment_info(exp, include_history=False)

    assert captured["include_history"] is False

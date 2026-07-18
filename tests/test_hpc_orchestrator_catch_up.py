from __future__ import annotations

import copy
from pathlib import Path

import pytest

import slurminator.hpc_orchestrator as hpc_orchestrator_module
from slurminator.config import HPCType
from slurminator.experiments import ExperimentStatus
from slurminator.experiments.yaml_utils import dump_yaml
from slurminator.hpc_orchestrator import HPCOrchestrator
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


def test_catch_up_existing_state_reconciles_and_saves_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
    estimate_calls: list[list[dict]] = []
    saved: list[dict] = []

    def update_statuses(experiments: list[dict]) -> None:
        update_calls.append(experiments)
        experiments[0]["status"] = ExperimentStatus.COMPLETED
        experiments[0]["last_metrics_update"] = 100.0

    monkeypatch.setattr(orchestrator, "_update_statuses", update_statuses)
    monkeypatch.setattr(orchestrator, "_update_queue_estimates", lambda exps: estimate_calls.append(exps))
    monkeypatch.setattr(orchestrator, "_save_yaml", lambda data: saved.append(copy.deepcopy(data)))

    result = orchestrator._catch_up_existing_state()

    assert result is not None
    assert result["experiments"][0]["status"] == ExperimentStatus.COMPLETED
    assert len(update_calls) == 1
    assert len(estimate_calls) == 1
    assert len(saved) == 1
    assert saved[0]["experiments"][0]["status"] == ExperimentStatus.COMPLETED
    assert saved[0]["experiments"][0]["last_metrics_update"] == 100.0
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

    monkeypatch.setattr(orchestrator, "_update_statuses", unexpected_update)
    monkeypatch.setattr(orchestrator, "_save_yaml", unexpected_save)

    assert orchestrator._catch_up_existing_state() is None


def test_orchestrator_telemetry_ingest_uses_batched_persistence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    orchestrator = _orchestrator(tmp_path, [])
    captured: dict[str, object] = {}

    def capture_context(exp: dict, context: StatusIngestContext) -> None:
        captured["exp"] = exp
        captured["persist_immediately"] = context.persist_immediately

    monkeypatch.setattr(hpc_orchestrator_module, "update_running_experiment_info", capture_context)
    exp = {"experiment_id": "running-1"}

    orchestrator._update_running_experiment_info(exp)

    assert captured == {"exp": exp, "persist_immediately": False}

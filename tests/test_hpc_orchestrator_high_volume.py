"""Focused tests for adaptive high-volume orchestrator maintenance."""

from __future__ import annotations

from typing import Any

import pytest

import slurminator.hpc_orchestrator as hpc_orchestrator_module
from slurminator.config import HPCType
from slurminator.experiments import ExperimentStatus
from slurminator.hpc_orchestrator import HPCOrchestrator
from slurminator.scheduler_polling import SchedulerStatusUpdateResult, SchedulerTransition

pytestmark = pytest.mark.unit


def _bare_orchestrator(*, high_volume: bool = True) -> HPCOrchestrator:
    orchestrator = object.__new__(HPCOrchestrator)
    orchestrator.high_volume_threshold = 100
    orchestrator.state_checkpoint_interval_seconds = 60.0
    orchestrator.telemetry_interval_seconds = 10.0
    orchestrator.telemetry_batch_size = 2
    orchestrator.high_volume_mode = high_volume
    orchestrator._last_state_checkpoint_at = 0.0
    orchestrator._last_telemetry_poll_at = None
    orchestrator._telemetry_cursor = 0
    orchestrator.submissions_paused = False
    orchestrator.retry_timeout_with_estimated_time = False
    return orchestrator


def _transition(
    *, old_status: ExperimentStatus = ExperimentStatus.QUEUED, new_status: ExperimentStatus = ExperimentStatus.RUNNING
) -> SchedulerTransition:
    timestamp_fields = {
        ExperimentStatus.RUNNING: "running_timestamp",
        ExperimentStatus.COMPLETED: "completed_timestamp",
        ExperimentStatus.TIMEOUT: "timeout_timestamp",
    }
    return SchedulerTransition(
        experiment_id="exp-1",
        job_id="12345",
        hpc_type=HPCType.OLIVIA,
        old_status=old_status,
        new_status=new_status,
        timestamp=20.0,
        timestamp_field=timestamp_fields.get(new_status),
    )


@pytest.mark.parametrize(("experiment_count", "expected"), [(99, False), (100, True)])
def test_high_volume_boundary_is_one_hundred(experiment_count: int, expected: bool) -> None:
    orchestrator = _bare_orchestrator()
    data = {"experiments": [{"experiment_id": str(index)} for index in range(experiment_count)]}

    orchestrator._configure_volume_mode(data)

    assert orchestrator.high_volume_mode is expected
    assert orchestrator.multi_experiment is True


def test_only_high_volume_mode_discards_cached_history() -> None:
    low_volume = _bare_orchestrator()
    low_data = {"experiments": [{"experiment_id": str(index), "history": [index]} for index in range(99)]}
    low_volume._configure_volume_mode(low_data)

    high_volume = _bare_orchestrator()
    high_data = {"experiments": [{"experiment_id": str(index), "history": [index]} for index in range(100)]}
    high_volume._configure_volume_mode(high_data)

    assert "history" in low_data["experiments"][0]
    assert "history" not in high_data["experiments"][0]


def test_high_volume_cycle_journals_before_slow_work_and_skips_inner_renders(monkeypatch: pytest.MonkeyPatch) -> None:
    orchestrator = _bare_orchestrator()
    events: list[str] = []
    exp = {
        "experiment_id": "exp-1",
        "job_id": "12345",
        "hpc_assignment": HPCType.OLIVIA,
        "status": ExperimentStatus.QUEUED,
    }
    data = {"experiments": [exp]}
    transition = _transition()

    class StateStore:
        def record_scheduler_transitions(self, transitions: object) -> None:
            assert tuple(transitions) == (transition,)
            events.append("journal")

    orchestrator.state_store = StateStore()
    orchestrator.connection_manager = object()
    orchestrator.concurrency_limits = {HPCType.OLIVIA: 1}

    def poll(experiments: list[dict[str, Any]], **_kwargs: object) -> SchedulerStatusUpdateResult:
        events.append("poll")
        experiments[0].update(transition.row_patch())
        return SchedulerStatusUpdateResult((transition,), ())

    monkeypatch.setattr(hpc_orchestrator_module, "update_scheduler_statuses", poll)
    monkeypatch.setattr(orchestrator, "_process_command_queue", lambda _exps: 0)
    monkeypatch.setattr(orchestrator, "_update_telemetry", lambda *_args, **_kwargs: events.append("telemetry"))
    monkeypatch.setattr(orchestrator, "_update_queue_estimates", lambda _exps: events.append("eta"))
    monkeypatch.setattr(
        orchestrator, "_fill_available_capacity", lambda *_args, **_kwargs: events.append("fill") or {HPCType.OLIVIA: 1}
    )
    monkeypatch.setattr(orchestrator, "_maybe_reassign_experiments", lambda *_args: events.append("reassign"))

    orchestrator._maintain_cycle(data, on_refresh=lambda _exps: events.append("render"))

    assert events == ["poll", "journal", "telemetry", "eta", "fill", "reassign"]


def test_failed_lifecycle_journal_stops_telemetry_and_submission(monkeypatch: pytest.MonkeyPatch) -> None:
    orchestrator = _bare_orchestrator()
    exp = {
        "experiment_id": "exp-1",
        "job_id": "12345",
        "hpc_assignment": HPCType.OLIVIA,
        "status": ExperimentStatus.QUEUED,
    }
    transition = _transition()

    class StateStore:
        def record_scheduler_transitions(self, _transitions: object) -> None:
            raise OSError("journal fsync failed")

    orchestrator.state_store = StateStore()
    orchestrator.connection_manager = object()
    orchestrator.concurrency_limits = {HPCType.OLIVIA: 1}

    def poll(experiments: list[dict[str, Any]], **_kwargs: object) -> SchedulerStatusUpdateResult:
        experiments[0].update(transition.row_patch())
        return SchedulerStatusUpdateResult((transition,), ())

    monkeypatch.setattr(hpc_orchestrator_module, "update_scheduler_statuses", poll)
    monkeypatch.setattr(orchestrator, "_process_command_queue", lambda _exps: 0)
    monkeypatch.setattr(
        orchestrator,
        "_update_telemetry",
        lambda *_args, **_kwargs: pytest.fail("telemetry must not follow a failed lifecycle fsync"),
    )
    monkeypatch.setattr(
        orchestrator,
        "_fill_available_capacity",
        lambda *_args, **_kwargs: pytest.fail("submission must not follow a failed lifecycle fsync"),
    )

    with pytest.raises(OSError, match="journal fsync failed"):
        orchestrator._maintain_cycle({"experiments": [exp]})


def test_bounded_telemetry_round_robin_skips_historical_terminal_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    orchestrator = _bare_orchestrator()
    historical = [
        {"experiment_id": f"old-{index}", "job_id": f"old-job-{index}", "status": ExperimentStatus.COMPLETED}
        for index in range(200)
    ]
    running = [
        {"experiment_id": f"run-{index}", "job_id": f"run-job-{index}", "status": ExperimentStatus.RUNNING}
        for index in range(3)
    ]
    newly_terminal = {
        "experiment_id": "new-terminal",
        "job_id": "new-job",
        "status": ExperimentStatus.COMPLETED,
        "terminal_telemetry": {"job_id": "new-job", "attempts": 0, "finalized": False},
    }
    calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        orchestrator,
        "_update_running_experiment_info",
        lambda exp, *, include_history: calls.append((exp["experiment_id"], include_history)),
    )
    exps = [*historical, *running, newly_terminal]

    assert orchestrator._update_telemetry(exps, include_history=False, max_experiments=2) == 2
    assert orchestrator._update_telemetry(exps, include_history=False, max_experiments=2) == 2

    assert calls == [("run-0", False), ("run-1", False), ("run-2", False), ("new-terminal", False)]


def test_high_volume_cadence_and_checkpoint_repoll(monkeypatch: pytest.MonkeyPatch) -> None:
    orchestrator = _bare_orchestrator()
    assert orchestrator._telemetry_poll_due(now=0.0)
    orchestrator._last_telemetry_poll_at = 5.0
    assert not orchestrator._telemetry_poll_due(now=14.9)
    assert orchestrator._telemetry_poll_due(now=15.0)
    assert not orchestrator._state_checkpoint_due(now=59.9)
    assert orchestrator._state_checkpoint_due(now=60.0)

    events: list[str] = []
    data = {
        "experiments": [
            {
                "experiment_id": "exp-1",
                "job_id": "12345",
                "hpc_assignment": HPCType.OLIVIA,
                "status": ExperimentStatus.RUNNING,
            }
        ]
    }
    monkeypatch.setattr(orchestrator, "_save_yaml", lambda _data: events.append("save"))
    monkeypatch.setattr(
        orchestrator,
        "_refresh_scheduler_statuses",
        lambda _exps: events.append("poll") or SchedulerStatusUpdateResult((), ()),
    )

    orchestrator._checkpoint_and_repoll(data)

    assert events == ["save", "poll"]


def test_high_volume_checkpoint_omits_reconstructable_history(monkeypatch: pytest.MonkeyPatch) -> None:
    orchestrator = _bare_orchestrator()
    saved: list[dict[str, Any]] = []

    class StateStore:
        def save(self, data: dict[str, Any]) -> None:
            saved.append(data)

    orchestrator.state_store = StateStore()
    exp = {
        "experiment_id": "exp-1",
        "status": ExperimentStatus.RUNNING,
        "history": [{"epoch": 1}],
        "history_last_read_offset": 100,
        "history_attempt_max": 1,
        "history_truncated": False,
    }
    monkeypatch.setattr(hpc_orchestrator_module.time, "monotonic", lambda: 70.0)

    orchestrator._save_yaml({"experiments": [exp], "metadata": "kept"})

    assert saved[0]["metadata"] == "kept"
    assert saved[0]["experiments"] == [{"experiment_id": "exp-1", "status": ExperimentStatus.RUNNING}]
    assert exp["history"] == [{"epoch": 1}]
    assert orchestrator._last_state_checkpoint_at == 70.0


def test_timeout_retry_is_checkpointed_before_replacement_submission(monkeypatch: pytest.MonkeyPatch) -> None:
    orchestrator = _bare_orchestrator()
    orchestrator.retry_timeout_with_estimated_time = True
    orchestrator.connection_manager = object()
    orchestrator.concurrency_limits = {HPCType.OLIVIA: 1}
    events: list[str] = []
    exp = {
        "experiment_id": "exp-1",
        "job_id": "12345",
        "hpc_assignment": HPCType.OLIVIA,
        "status": ExperimentStatus.RUNNING,
    }
    transition = _transition(old_status=ExperimentStatus.RUNNING, new_status=ExperimentStatus.TIMEOUT)

    class StateStore:
        def record_scheduler_transitions(self, _transitions: object) -> None:
            events.append("journal")

    orchestrator.state_store = StateStore()

    def poll(experiments: list[dict[str, Any]], **_kwargs: object) -> SchedulerStatusUpdateResult:
        events.append("poll")
        experiments[0].update(transition.row_patch())
        return SchedulerStatusUpdateResult((transition,), ())

    def timeout_policy(row: dict[str, Any], *_args: object) -> None:
        events.append("policy")
        row["status"] = ExperimentStatus.PARTIAL

    monkeypatch.setattr(hpc_orchestrator_module, "update_scheduler_statuses", poll)
    monkeypatch.setattr(orchestrator, "_apply_timeout_policy", timeout_policy)
    monkeypatch.setattr(orchestrator, "_process_command_queue", lambda _exps: 0)
    monkeypatch.setattr(orchestrator, "_save_yaml", lambda _data: events.append("save"))
    monkeypatch.setattr(orchestrator, "_update_telemetry", lambda *_args, **_kwargs: events.append("telemetry"))
    monkeypatch.setattr(orchestrator, "_update_queue_estimates", lambda _exps: events.append("eta"))
    monkeypatch.setattr(
        orchestrator,
        "_fill_available_capacity",
        lambda *_args, **_kwargs: events.append("submit") or {HPCType.OLIVIA: 1},
    )
    monkeypatch.setattr(orchestrator, "_maybe_reassign_experiments", lambda *_args: None)

    orchestrator._maintain_cycle({"experiments": [exp]})

    assert events == ["poll", "journal", "policy", "save", "telemetry", "eta", "submit"]


def test_failed_timeout_retry_checkpoint_prevents_replacement_submission(monkeypatch: pytest.MonkeyPatch) -> None:
    orchestrator = _bare_orchestrator()
    transition = _transition(old_status=ExperimentStatus.RUNNING, new_status=ExperimentStatus.TIMEOUT)
    exp = {
        "experiment_id": "exp-1",
        "job_id": "12345",
        "hpc_assignment": HPCType.OLIVIA,
        "status": ExperimentStatus.RUNNING,
    }

    def refresh(experiments: list[dict[str, Any]]) -> SchedulerStatusUpdateResult:
        experiments[0]["status"] = ExperimentStatus.PARTIAL
        return SchedulerStatusUpdateResult((transition,), ())

    monkeypatch.setattr(orchestrator, "_process_command_queue", lambda _exps: 0)
    monkeypatch.setattr(orchestrator, "_refresh_scheduler_statuses", refresh)
    monkeypatch.setattr(orchestrator, "_save_yaml", lambda _data: (_ for _ in ()).throw(OSError("checkpoint failed")))
    monkeypatch.setattr(
        orchestrator,
        "_update_telemetry",
        lambda *_args, **_kwargs: pytest.fail("telemetry must not follow the failed retry checkpoint"),
    )
    monkeypatch.setattr(
        orchestrator,
        "_fill_available_capacity",
        lambda *_args, **_kwargs: pytest.fail("replacement submission must not follow the failed checkpoint"),
    )

    with pytest.raises(OSError, match="checkpoint failed"):
        orchestrator._maintain_cycle({"experiments": [exp]})


def test_resume_recovers_interrupted_timeout_retry_before_submission(monkeypatch: pytest.MonkeyPatch) -> None:
    orchestrator = _bare_orchestrator()
    orchestrator.retry_timeout_with_estimated_time = True
    events: list[str] = []
    exp = {
        "experiment_id": "exp-1",
        "job_id": "12345",
        "hpc_assignment": HPCType.OLIVIA,
        "status": ExperimentStatus.TIMEOUT,
    }
    data = {"experiments": [exp]}

    def timeout_policy(row: dict[str, Any], *_args: object) -> None:
        events.append("policy")
        row["status"] = ExperimentStatus.PARTIAL
        row["timeout_retry_count"] = 1

    monkeypatch.setattr(orchestrator, "_apply_timeout_policy", timeout_policy)
    monkeypatch.setattr(orchestrator, "_save_yaml", lambda _data: events.append("save"))
    monkeypatch.setattr(orchestrator, "_publish_dashboard_snapshot", lambda _exps: events.append("publish"))

    result = orchestrator._catch_up_existing_state(data)

    assert result is data
    assert exp["status"] == ExperimentStatus.PARTIAL
    assert events == ["policy", "save", "publish"]

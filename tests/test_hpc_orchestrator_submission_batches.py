"""Focused tests for bounded orchestrator submission batches."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import pytest

import slurminator.hpc_orchestrator as hpc_orchestrator_module
from slurminator.config import HPCType
from slurminator.hpc_orchestrator import HPCOrchestrator

pytestmark = pytest.mark.unit


def _bare_orchestrator(
    *, batch_size: int = 2, batch_seconds: float = 2.0, checkpoint_size: int = 32
) -> HPCOrchestrator:
    orchestrator = object.__new__(HPCOrchestrator)
    orchestrator.submission_batch_size = batch_size
    orchestrator.submission_batch_seconds = batch_seconds
    orchestrator.submission_checkpoint_size = checkpoint_size
    orchestrator.submissions_paused = False
    return orchestrator


def test_submit_pending_batch_counts_successes_and_stops_at_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    orchestrator = _bare_orchestrator(batch_size=2)
    experiments = [{"experiment_id": f"exp-{index}"} for index in range(5)]
    outcomes = {"exp-0": False, "exp-1": True, "exp-2": False, "exp-3": True, "exp-4": True}
    calls: list[str] = []
    concurrency_used = {HPCType.OLIVIA: 0}
    data: dict[str, Any] = {"experiments": experiments}

    def maybe_submit(exp: dict[str, Any], usage: dict[HPCType, int], ledger: dict[str, Any]) -> bool:
        assert usage is concurrency_used
        assert ledger is data
        experiment_id = str(exp["experiment_id"])
        calls.append(experiment_id)
        return outcomes[experiment_id]

    monkeypatch.setattr(orchestrator, "_maybe_submit", maybe_submit)

    submitted_count = orchestrator._submit_pending_batch(experiments, concurrency_used, data)

    assert submitted_count == 2
    assert calls == ["exp-0", "exp-1", "exp-2", "exp-3"]


def test_submit_pending_batch_stops_at_elapsed_time_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    orchestrator = _bare_orchestrator(batch_size=5, batch_seconds=2.0)
    experiments = [{"experiment_id": f"exp-{index}"} for index in range(3)]
    calls: list[str] = []
    monotonic_values = iter([10.0, 10.5, 12.1])

    monkeypatch.setattr(hpc_orchestrator_module.time, "monotonic", lambda: next(monotonic_values))

    def maybe_submit(exp: dict[str, Any], _usage: dict[HPCType, int], _ledger: dict[str, Any]) -> bool:
        calls.append(str(exp["experiment_id"]))
        return True

    monkeypatch.setattr(orchestrator, "_maybe_submit", maybe_submit)

    submitted_count = orchestrator._submit_pending_batch(experiments, {}, {"experiments": experiments})

    assert submitted_count == 2
    assert calls == ["exp-0", "exp-1"]


def test_submit_pending_batch_preserves_concurrency_limit_result(monkeypatch: pytest.MonkeyPatch) -> None:
    orchestrator = _bare_orchestrator(batch_size=5)
    experiments = [{"experiment_id": f"exp-{index}"} for index in range(3)]
    concurrency_used = {HPCType.OLIVIA: 1}

    def maybe_submit(_exp: dict[str, Any], usage: dict[HPCType, int], _ledger: dict[str, Any]) -> bool:
        if usage[HPCType.OLIVIA] >= 2:
            return False
        usage[HPCType.OLIVIA] += 1
        return True

    monkeypatch.setattr(orchestrator, "_maybe_submit", maybe_submit)

    submitted_count = orchestrator._submit_pending_batch(experiments, concurrency_used, {"experiments": experiments})

    assert submitted_count == 1
    assert concurrency_used == {HPCType.OLIVIA: 2}


def test_submit_pending_batch_honors_pause_before_and_during_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    orchestrator = _bare_orchestrator(batch_size=3)
    experiments = [{"experiment_id": f"exp-{index}"} for index in range(3)]
    calls: list[str] = []

    def maybe_submit(exp: dict[str, Any], _usage: dict[HPCType, int], _ledger: dict[str, Any]) -> bool:
        calls.append(str(exp["experiment_id"]))
        orchestrator.submissions_paused = True
        return True

    monkeypatch.setattr(orchestrator, "_maybe_submit", maybe_submit)
    orchestrator.submissions_paused = True

    assert orchestrator._submit_pending_batch(experiments, {}, {"experiments": experiments}) == 0
    assert calls == []

    orchestrator.submissions_paused = False

    assert orchestrator._submit_pending_batch(experiments, {}, {"experiments": experiments}) == 1
    assert calls == ["exp-0"]


def test_fill_available_capacity_repolls_between_bursts_and_defers_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    orchestrator = _bare_orchestrator(batch_size=2, checkpoint_size=3)
    experiments = [{"experiment_id": f"exp-{index}"} for index in range(5)]
    data: dict[str, Any] = {"experiments": experiments}
    events: list[str] = []

    monkeypatch.setattr(orchestrator, "_count_concurrency", lambda _exps: {})
    monkeypatch.setattr(orchestrator, "_process_command_queue", lambda _exps: 0)
    monkeypatch.setattr(orchestrator, "_refresh_scheduler_statuses", lambda _exps: events.append("poll"))
    monkeypatch.setattr(orchestrator, "_save_yaml", lambda _data: events.append("save"))

    def maybe_submit(exp: dict[str, Any], _usage: dict[HPCType, int], _ledger: dict[str, Any]) -> bool:
        if exp.get("submitted"):
            return False
        exp["submitted"] = True
        events.append(f"submit:{exp['experiment_id']}")
        return True

    monkeypatch.setattr(orchestrator, "_maybe_submit", maybe_submit)

    orchestrator._fill_available_capacity(experiments, data, on_refresh=lambda _exps: events.append("render"))

    assert events == ["submit:exp-0", "submit:exp-1", "poll", "render", "submit:exp-2"]
    assert "save" not in events


def test_fill_available_capacity_checkpoints_attempt_before_retry_submission(monkeypatch: pytest.MonkeyPatch) -> None:
    orchestrator = _bare_orchestrator(batch_size=1, checkpoint_size=2)
    experiments = [
        {"experiment_id": "retry", "status": hpc_orchestrator_module.ExperimentStatus.QUEUED, "job_id": "old-job"},
        {"experiment_id": "new", "status": hpc_orchestrator_module.ExperimentStatus.PENDING},
    ]
    data: dict[str, Any] = {"experiments": experiments}
    events: list[str] = []

    monkeypatch.setattr(orchestrator, "_count_concurrency", lambda _exps: {})
    monkeypatch.setattr(orchestrator, "_process_command_queue", lambda _exps: 0)
    monkeypatch.setattr(orchestrator, "_save_yaml", lambda _data: events.append("save"))

    def refresh(_exps: list[dict[str, Any]]) -> None:
        events.append("poll")
        experiments[0]["status"] = hpc_orchestrator_module.ExperimentStatus.PARTIAL

    def maybe_submit(exp: dict[str, Any], _usage: dict[HPCType, int], _ledger: dict[str, Any]) -> bool:
        if exp.get("submitted") or exp.get("status") not in {
            hpc_orchestrator_module.ExperimentStatus.PENDING,
            hpc_orchestrator_module.ExperimentStatus.PARTIAL,
        }:
            return False
        exp["submitted"] = True
        events.append(f"submit:{exp['experiment_id']}")
        return True

    monkeypatch.setattr(orchestrator, "_refresh_scheduler_statuses", refresh)
    monkeypatch.setattr(orchestrator, "_maybe_submit", maybe_submit)

    orchestrator._fill_available_capacity(experiments, data)

    assert events == ["submit:new", "poll", "save", "submit:retry"]


class _FakeConnectionManager:
    """Connection manager sufficient for exercising the orchestrator run loop."""

    configs: dict[HPCType, object] = {}

    def connect_all(self) -> None:
        return None

    def close_all(self) -> None:
        return None


class _FakeLive:
    def __init__(self, events: list[str]):
        self.events = events

    def update(self, _renderable: object) -> None:
        return None


class _FakeDashboard:
    events: list[str] = []
    dashboard_exit_requested = False

    def __init__(self, **_kwargs: object):
        return None

    @contextmanager
    def mount(self, _orchestrator: HPCOrchestrator) -> Iterator[_FakeLive]:
        yield _FakeLive(self.events)

    def render(self, _experiments: list[dict[str, Any]]) -> object:
        self.events.append("render")
        return object()


@pytest.mark.parametrize("debug", [True, False])
def test_run_polls_between_submission_batches_and_does_not_render_per_job(
    debug: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    orchestrator = _bare_orchestrator(batch_size=2)
    experiments = [{"experiment_id": f"exp-{index}"} for index in range(3)]
    data: dict[str, Any] = {"experiments": experiments}
    events: list[str] = []
    _FakeDashboard.events = events

    orchestrator.debug = debug
    orchestrator.connection_manager = _FakeConnectionManager()
    orchestrator.concurrency_limits = {}
    orchestrator.poll_interval = 0
    orchestrator.plugin = object()
    orchestrator.overview_printer = lambda _exps: events.append("render")
    orchestrator.dashboard_settings = None
    orchestrator._dashboard_exit_requested = False

    monkeypatch.setattr(orchestrator, "_load_yaml", lambda: data)
    monkeypatch.setattr(orchestrator, "_preflight_validate_experiments", lambda _data: 0)
    monkeypatch.setattr(orchestrator, "_preflight_test_hpcs", lambda: None)
    monkeypatch.setattr(orchestrator, "_recover_orphans", lambda: None)
    monkeypatch.setattr(orchestrator, "_catch_up_existing_state", lambda: None)
    monkeypatch.setattr(orchestrator, "_process_command_queue", lambda _exps: 0)
    monkeypatch.setattr(orchestrator, "_update_statuses", lambda _exps: events.append("poll"))
    monkeypatch.setattr(orchestrator, "_refresh_scheduler_statuses", lambda _exps: events.append("poll"))
    monkeypatch.setattr(orchestrator, "_update_queue_estimates", lambda _exps: None)
    monkeypatch.setattr(orchestrator, "_save_yaml", lambda _data: events.append("save"))
    monkeypatch.setattr(orchestrator, "_publish_dashboard_snapshot", lambda _exps: None)
    monkeypatch.setattr(orchestrator, "_publish_current_dashboard_snapshot", lambda: experiments)
    monkeypatch.setattr(orchestrator, "_count_concurrency", lambda _exps: {})
    monkeypatch.setattr(orchestrator, "_maybe_reassign_experiments", lambda *_args: None)
    monkeypatch.setattr(orchestrator, "_resolve_dashboard_cls", lambda: _FakeDashboard)
    monkeypatch.setattr(orchestrator, "_effective_dashboard_ui", lambda _dashboard_cls: "v3")
    monkeypatch.setattr(orchestrator, "_sleep_until_next_poll", lambda _dashboard: False)
    monkeypatch.setattr(hpc_orchestrator_module.time, "sleep", lambda _seconds: None)

    def maybe_submit(exp: dict[str, Any], _usage: dict[HPCType, int], _ledger: dict[str, Any]) -> bool:
        if exp.get("submitted"):
            return False
        exp["submitted"] = True
        events.append(f"submit:{exp['experiment_id']}")
        return True

    monkeypatch.setattr(orchestrator, "_maybe_submit", maybe_submit)
    monkeypatch.setattr(orchestrator, "_all_done", lambda exps: all(exp.get("submitted") for exp in exps))

    orchestrator.run()

    first_submit = events.index("submit:exp-0")
    second_submit = events.index("submit:exp-1")
    third_submit = events.index("submit:exp-2")
    polls = [index for index, event in enumerate(events) if event == "poll"]

    assert second_submit == first_submit + 1
    assert len(polls) == 3
    assert any(second_submit < poll_index < third_submit for poll_index in polls)
    assert events.count("save") == 1

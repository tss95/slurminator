"""Hermetic regressions for the low-compute controller preset."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import slurminator.hpc_orchestrator as orchestrator_module
from slurminator.config import HPCType
from slurminator.experiments import ExperimentStatus
from slurminator.hpc_orchestrator import HPCOrchestrator
from slurminator.scheduler_polling import SchedulerStatusUpdateResult

pytestmark = pytest.mark.unit


def _orchestrator(tmp_path: Path, **kwargs) -> HPCOrchestrator:
    return HPCOrchestrator(
        str(tmp_path / "experiments.yaml"),
        concurrency_limits={HPCType.OLIVIA: 100},
        connection_manager=SimpleNamespace(configs={}),
        local_log_dir=str(tmp_path / "logs"),
        **kwargs,
    )


@pytest.mark.parametrize("low_compute", [False, True])
def test_controller_preset_preserves_normal_defaults(tmp_path: Path, low_compute: bool) -> None:
    orchestrator = _orchestrator(tmp_path, low_compute=low_compute)
    data = {"experiments": [{"experiment_id": "one", "history": [1, 2]}]}
    orchestrator._configure_volume_mode(data)

    assert orchestrator.poll_interval == (60 if low_compute else 2)
    assert orchestrator.telemetry_interval_seconds == (60 if low_compute else 10)
    assert orchestrator.telemetry_batch_size == (32 if low_compute else 64)
    assert orchestrator.state_checkpoint_interval_seconds == (300 if low_compute else 60)
    assert orchestrator.dashboard_ui == "v4"
    assert orchestrator.high_volume_mode is low_compute
    assert ("history" not in data["experiments"][0]) is low_compute


def test_controller_preserves_more_conservative_settings(tmp_path: Path) -> None:
    orchestrator = _orchestrator(
        tmp_path,
        low_compute=True,
        poll_interval=900,
        telemetry_interval_seconds=1200,
        state_checkpoint_interval_seconds=1800,
        telemetry_batch_size=2,
        submission_batch_size=1,
        submission_batch_seconds=0.5,
        dashboard_ui="v2",
    )
    assert orchestrator.poll_interval == 900
    assert orchestrator.telemetry_interval_seconds == 1200
    assert orchestrator.state_checkpoint_interval_seconds == 1800
    assert orchestrator.telemetry_batch_size == 2
    assert orchestrator.submission_batch_size == 1
    assert orchestrator.submission_batch_seconds == 0.5
    assert orchestrator.dashboard_ui == "v2"


@pytest.mark.parametrize("poll_interval", [0, -1, float("inf"), float("nan")])
def test_controller_rejects_invalid_low_compute_interval(tmp_path: Path, poll_interval: float) -> None:
    with pytest.raises(ValueError, match="poll_interval"):
        _orchestrator(tmp_path, low_compute=True, poll_interval=poll_interval)


def test_low_compute_cycle_bounds_status_reads_and_defers_writes(tmp_path: Path, monkeypatch) -> None:
    orchestrator = _orchestrator(tmp_path, low_compute=True)
    data = {
        "experiments": [
            {"experiment_id": str(index), "job_id": str(index), "status": ExperimentStatus.RUNNING}
            for index in range(40)
        ]
    }
    now = [0.0]
    monkeypatch.setattr(orchestrator_module.time, "monotonic", lambda: now[0])
    orchestrator._configure_volume_mode(data)
    monkeypatch.setattr(orchestrator, "_process_command_queue", lambda _exps: 0)
    monkeypatch.setattr(orchestrator, "_refresh_scheduler_statuses", lambda _exps: SchedulerStatusUpdateResult((), ()))
    read_status = Mock()
    monkeypatch.setattr(orchestrator, "_update_running_experiment_info", read_status)
    eta = Mock()
    monkeypatch.setattr(orchestrator, "_update_queue_estimates", eta)
    monkeypatch.setattr(orchestrator, "_fill_available_capacity", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(orchestrator, "_maybe_reassign_experiments", lambda *_args: None)
    monkeypatch.setattr(orchestrator, "_save_yaml", lambda _data: pytest.fail("no periodic write before 300s"))

    orchestrator._maintain_cycle(data)
    assert read_status.call_count == 32
    assert all(call.kwargs == {"include_history": False} for call in read_status.call_args_list)
    now[0] = 59.9
    orchestrator._maintain_cycle(data)
    assert read_status.call_count == 32
    now[0] = 60.0
    orchestrator._maintain_cycle(data)
    assert read_status.call_count == 64
    assert {call.args[0]["experiment_id"] for call in read_status.call_args_list} == {str(i) for i in range(40)}
    assert eta.call_count == 2
    assert not orchestrator._state_checkpoint_due(now=299.9)
    assert orchestrator._state_checkpoint_due(now=300.0)


def test_low_compute_checkpoints_without_an_extra_scheduler_query(tmp_path: Path, monkeypatch) -> None:
    orchestrator = _orchestrator(tmp_path, low_compute=True)
    data = {"experiments": [{"job_id": "1", "status": ExperimentStatus.RUNNING}]}
    save = Mock()
    monkeypatch.setattr(orchestrator, "_save_yaml", save)
    monkeypatch.setattr(orchestrator, "_refresh_scheduler_statuses", lambda _exps: pytest.fail("extra scheduler poll"))

    orchestrator._checkpoint_and_repoll(data)

    save.assert_called_once_with(data)


def test_low_compute_yields_after_one_submission_burst(tmp_path: Path, monkeypatch) -> None:
    orchestrator = _orchestrator(tmp_path, low_compute=True, submission_batch_size=100, submission_batch_seconds=30.0)
    assert orchestrator.submission_batch_seconds == 2.0
    data = {"experiments": [{"experiment_id": str(i), "status": ExperimentStatus.PENDING} for i in range(20)]}
    monkeypatch.setattr(orchestrator_module.time, "monotonic", lambda: 0.0)

    def submit(exp, usage, _data):
        exp.update(status=ExperimentStatus.QUEUED, hpc_assignment=HPCType.OLIVIA)
        usage[HPCType.OLIVIA] += 1
        return True

    monkeypatch.setattr(orchestrator, "_maybe_submit", submit)
    monkeypatch.setattr(orchestrator, "_refresh_scheduler_statuses", lambda _exps: pytest.fail("extra scheduler poll"))
    monkeypatch.setattr(
        orchestrator, "_save_yaml", lambda _data: pytest.fail("accepted receipts suffice between checkpoints")
    )

    usage = orchestrator._fill_available_capacity(data["experiments"], data)

    assert usage[HPCType.OLIVIA] == 8
    assert sum(exp["status"] == ExperimentStatus.QUEUED for exp in data["experiments"]) == 8


def test_low_compute_waits_for_bounded_final_telemetry(tmp_path: Path) -> None:
    orchestrator = _orchestrator(tmp_path, low_compute=True)
    exp = {
        "status": ExperimentStatus.COMPLETED,
        "job_id": "1",
        "terminal_telemetry": {"job_id": "1", "attempts": 0, "finalized": False},
    }
    assert not orchestrator._all_done([exp])
    exp["terminal_telemetry"]["finalized"] = True
    assert orchestrator._all_done([exp])
    assert orchestrator._all_done([])


def test_low_compute_preserves_dashboard_update_contract(tmp_path: Path) -> None:
    orchestrator = _orchestrator(tmp_path, low_compute=True)
    live = Mock()
    dashboard = Mock()

    orchestrator._render_dashboard_snapshot(dashboard, [], live=live)

    live.update.assert_called_once_with(dashboard.render.return_value)


@pytest.mark.parametrize("debug", [False, True])
def test_low_compute_run_polls_once_per_minute_and_exits_after_final_metrics(
    tmp_path: Path, monkeypatch, debug: bool
) -> None:
    live = Mock()

    class Dashboard:
        def __init__(self, **kwargs):
            assert kwargs == {"n_recent": 0, "ui_version": "v3"}

        @contextmanager
        def mount(self, _orchestrator):
            yield live

        def render(self, _exps):
            return "snapshot"

    orchestrator = _orchestrator(
        tmp_path,
        low_compute=True,
        debug=debug,
        dashboard_ui="v3",
        dashboard_cls=Dashboard,
        overview_printer=lambda _exps: None,
    )
    orchestrator.connection_manager.connect_all = Mock()
    orchestrator.connection_manager.close_all = Mock()
    exp = {"experiment_id": "one", "job_id": "1", "hpc_assignment": HPCType.OLIVIA, "status": ExperimentStatus.QUEUED}
    data = {"experiments": [exp]}
    load = Mock(return_value=data)
    save = Mock()
    monkeypatch.setattr(orchestrator, "_load_yaml", load)
    monkeypatch.setattr(orchestrator, "_save_yaml", save)
    monkeypatch.setattr(orchestrator, "_preflight_validate_experiments", lambda _data: 0)
    monkeypatch.setattr(orchestrator, "_preflight_test_hpcs", lambda: None)
    monkeypatch.setattr(orchestrator, "_recover_orphans", lambda _data: data)
    monkeypatch.setattr(orchestrator, "_process_command_queue", lambda _exps: 0)
    monkeypatch.setattr(orchestrator, "_maybe_submit", lambda *_args: False)
    monkeypatch.setattr(orchestrator, "_maybe_reassign_experiments", lambda *_args: None)
    monkeypatch.setattr(orchestrator, "_update_queue_estimates", lambda _exps: None)
    now = [0.0]
    polls = []
    monkeypatch.setattr(orchestrator_module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(orchestrator_module.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))

    def poll(exps):
        polls.append(now[0])
        if len(polls) == 1:
            exps[0]["status"] = ExperimentStatus.RUNNING
        else:
            exps[0].update(
                status=ExperimentStatus.COMPLETED, terminal_telemetry={"job_id": "1", "attempts": 0, "finalized": False}
            )
        return SchedulerStatusUpdateResult((), ())

    def read_status(exp, *, include_history):
        assert include_history is False
        if exp["status"] == ExperimentStatus.COMPLETED:
            exp["terminal_telemetry"]["finalized"] = True
            exp["all_metrics"] = {"loss": 0.1}

    monkeypatch.setattr(orchestrator, "_refresh_scheduler_statuses", poll)
    monkeypatch.setattr(orchestrator, "_update_running_experiment_info", read_status)

    orchestrator.run()

    assert polls == pytest.approx([0.0, 60.0])
    load.assert_called_once()
    save.assert_called_once_with(data)
    assert exp["all_metrics"] == {"loss": 0.1}
    orchestrator.connection_manager.close_all.assert_called_once()
    if not debug:
        assert all(call.kwargs == {} for call in live.update.call_args_list)

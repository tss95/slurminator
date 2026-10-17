"""Hermetic regressions for the low-compute controller preset."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import slurminator.hpc_orchestrator as orchestrator_module
import slurminator.submission as submission_module
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


def _submission_fill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, submission_seconds: float = 0.0, **kwargs
) -> SimpleNamespace:
    """Fake scheduler latency while retaining concurrency accounting and durable receipts."""
    orchestrator = _orchestrator(tmp_path, low_compute=True, **kwargs)
    data = {"experiments": [{"experiment_id": str(i), "status": ExperimentStatus.PENDING} for i in range(40)]}
    now = [0.0]
    starts = []
    monkeypatch.setattr(orchestrator_module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(orchestrator_module.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    monkeypatch.setattr(submission_module, "capture_provenance", lambda: {"project": "a" * 40})

    def submit(_exp, _hpc):
        starts.append(now[0])
        now[0] += submission_seconds
        return str(1000 + len(starts))

    submit_mock = Mock(side_effect=submit)
    commands = Mock(return_value=0)
    monkeypatch.setattr(orchestrator, "_submit_experiment_universal", submit_mock)
    monkeypatch.setattr(orchestrator, "_process_command_queue", commands)
    monkeypatch.setattr(orchestrator, "_refresh_scheduler_statuses", lambda _exps: pytest.fail("extra scheduler poll"))
    monkeypatch.setattr(
        orchestrator, "_update_running_experiment_info", lambda *_args, **_kwargs: pytest.fail("extra telemetry read")
    )
    monkeypatch.setattr(
        orchestrator, "_save_yaml", lambda _data: pytest.fail("accepted receipts suffice between checkpoints")
    )
    return SimpleNamespace(
        orchestrator=orchestrator, data=data, now=now, starts=starts, submit=submit_mock, commands=commands
    )


@pytest.mark.parametrize("checkpoint_size, expected", [(100, 32), (11, 11)])
def test_low_compute_fills_multiple_paced_batches_with_durable_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, checkpoint_size: int, expected: int
) -> None:
    fill = _submission_fill(
        tmp_path,
        monkeypatch,
        submission_batch_size=100,
        submission_batch_seconds=30.0,
        submission_checkpoint_size=checkpoint_size,
    )
    assert fill.orchestrator.submission_batch_size == 8
    assert fill.orchestrator.submission_batch_seconds == 2.0

    usage = fill.orchestrator._fill_available_capacity(fill.data["experiments"], fill.data)

    assert usage[HPCType.OLIVIA] == expected
    assert fill.starts == pytest.approx([float(index // 8 * 8) for index in range(expected)])
    receipts = fill.orchestrator.state_store.submission_journal.read()
    assert [receipt.experiment_id for receipt in receipts] == [str(index) for index in range(expected)]
    assert all(receipt.git_sha_at_submission == {"project": "a" * 40} for receipt in receipts)
    assert fill.commands.call_count == (expected - 1) // 8


@pytest.mark.parametrize("submission_seconds, expected", [(3.0, 8), (0.9, 17)])
def test_low_compute_slow_submissions_progress_until_fill_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, submission_seconds: float, expected: int
) -> None:
    fill = _submission_fill(tmp_path, monkeypatch, submission_seconds=submission_seconds)

    usage = fill.orchestrator._fill_available_capacity(fill.data["experiments"], fill.data)

    assert usage[HPCType.OLIVIA] == expected
    assert len(fill.starts) == expected
    assert all(start < 30.0 for start in fill.starts)
    assert 30.0 <= fill.now[0] <= 30.0 + submission_seconds
    assert len(fill.orchestrator.state_store.submission_journal.read()) == expected


def test_low_compute_respects_cached_concurrency_across_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fill = _submission_fill(tmp_path, monkeypatch)
    fill.orchestrator.concurrency_limits[HPCType.OLIVIA] = 11
    for exp in fill.data["experiments"][:2]:
        exp.update(
            status=ExperimentStatus.RUNNING, hpc_assignment=HPCType.OLIVIA, job_id=f"existing-{exp['experiment_id']}"
        )

    usage = fill.orchestrator._fill_available_capacity(fill.data["experiments"], fill.data)

    assert usage[HPCType.OLIVIA] == 11
    assert fill.submit.call_count == 9
    assert fill.now[0] == pytest.approx(8.0)
    assert len(fill.orchestrator.state_store.submission_journal.read()) == 9


def test_low_compute_stops_without_busy_retry_when_submission_makes_no_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fill = _submission_fill(tmp_path, monkeypatch)
    fill.submit.side_effect = None
    fill.submit.return_value = None

    usage = fill.orchestrator._fill_available_capacity(fill.data["experiments"], fill.data)

    assert usage[HPCType.OLIVIA] == 0
    assert fill.submit.call_count == len(fill.data["experiments"])
    assert fill.now[0] == 0.0
    assert not fill.orchestrator.state_store.submission_journal.path.exists()
    fill.commands.assert_not_called()


def test_low_compute_attempts_failed_rows_only_once_per_fill(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fill = _submission_fill(tmp_path, monkeypatch)
    accept = fill.submit.side_effect

    def mixed_result(exp, hpc):
        job_id = accept(exp, hpc)
        return job_id if int(exp["experiment_id"]) % 2 == 0 else None

    fill.submit.side_effect = mixed_result

    usage = fill.orchestrator._fill_available_capacity(fill.data["experiments"], fill.data)

    assert usage[HPCType.OLIVIA] == 20
    assert [call.args[0]["experiment_id"] for call in fill.submit.call_args_list] == [str(index) for index in range(40)]
    assert len(fill.orchestrator.state_store.submission_journal.read()) == 20


@pytest.mark.parametrize("signal", ["pause", "exit"])
def test_low_compute_cooldown_responds_to_pause_and_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, signal: str
) -> None:
    fill = _submission_fill(tmp_path, monkeypatch)

    def sleep(seconds):
        fill.now[0] += seconds
        if signal == "pause" and fill.now[0] >= 0.6:
            fill.orchestrator.submissions_paused = True

    monkeypatch.setattr(orchestrator_module.time, "sleep", sleep)
    refresh = Mock()

    usage = fill.orchestrator._fill_available_capacity(
        fill.data["experiments"],
        fill.data,
        should_stop=lambda: signal == "exit" and fill.now[0] >= 0.6,
        on_refresh=refresh,
    )

    assert usage[HPCType.OLIVIA] == 8
    assert fill.submit.call_count == 8
    assert fill.now[0] == pytest.approx(0.6)
    refresh.assert_called_once_with(fill.data["experiments"])


def test_low_compute_processes_operator_commands_between_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fill = _submission_fill(tmp_path, monkeypatch)
    save = Mock()
    monkeypatch.setattr(fill.orchestrator, "_save_yaml", save)

    def pause(_experiments):
        fill.orchestrator.submissions_paused = True
        return 1

    fill.commands.side_effect = pause

    usage = fill.orchestrator._fill_available_capacity(fill.data["experiments"], fill.data)

    assert usage[HPCType.OLIVIA] == 8
    assert fill.submit.call_count == 8
    fill.commands.assert_called_once_with(fill.data["experiments"])
    save.assert_called_once_with(fill.data)


def test_low_compute_receipt_failure_prevents_further_submissions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fill = _submission_fill(tmp_path, monkeypatch)
    record = fill.orchestrator.state_store.record_submission

    def fail_second_receipt(exp, previous_job_id):
        if exp["experiment_id"] == "1":
            raise OSError("receipt fsync failed")
        return record(exp, previous_job_id)

    monkeypatch.setattr(fill.orchestrator.state_store, "record_submission", fail_second_receipt)

    with pytest.raises(OSError, match="receipt fsync failed"):
        fill.orchestrator._fill_available_capacity(fill.data["experiments"], fill.data)

    assert fill.submit.call_count == 2
    assert fill.data["experiments"][2]["status"] == ExperimentStatus.PENDING
    assert len(fill.orchestrator.state_store.submission_journal.read()) == 1


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

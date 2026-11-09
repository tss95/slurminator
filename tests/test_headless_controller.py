"""Hermetic headless lifecycle checks without scheduler or dashboard access."""

from __future__ import annotations

import json
import logging
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import slurminator.hpc_orchestrator as orchestrator_module
from slurminator.controller_status import ControllerStatusPublisher, ledger_fingerprint, status_path
from slurminator.cli.orchestrator import run_orchestrator_cli
from slurminator.config import HPCType
from slurminator.experiments import ExperimentStatus
from slurminator.hpc_orchestrator import HPCOrchestrator
from slurminator.plugins import DefaultOrchestratorPlugin
from slurminator.scheduler_polling import SchedulerStatusUpdateResult

pytestmark = pytest.mark.unit


def _controller(tmp_path: Path, monkeypatch, rows: list[dict], **kwargs) -> HPCOrchestrator:
    ledger = tmp_path / "experiments.yaml"
    connection = SimpleNamespace(configs={}, connect_all=Mock(), close_all=Mock())
    controller = HPCOrchestrator(
        str(ledger),
        concurrency_limits={},
        headless=True,
        connection_manager=connection,
        local_log_dir=str(tmp_path / "logs"),
        **kwargs,
    )
    controller._save_yaml({"experiments": rows})
    monkeypatch.setattr(controller, "_preflight_test_hpcs", lambda: None)
    monkeypatch.setattr(controller, "_preflight_validate_experiments", lambda _data: None)
    monkeypatch.setattr(controller, "_catch_up_existing_state", lambda _data: None)
    monkeypatch.setattr(controller, "_process_command_queue", lambda _exps: 0)
    monkeypatch.setattr(controller, "_refresh_scheduler_statuses", lambda _exps: SchedulerStatusUpdateResult((), ()))
    monkeypatch.setattr(controller, "_update_queue_estimates", lambda _exps: None)
    monkeypatch.setattr(controller, "_fill_available_capacity", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(controller, "_maybe_reassign_experiments", lambda *_args: None)
    monkeypatch.setattr(controller, "_resolve_dashboard_cls", lambda: pytest.fail("dashboard requested"))
    monkeypatch.setattr(controller, "_publish_dashboard_snapshot", lambda _exps: pytest.fail("dashboard snapshot"))
    monkeypatch.setattr(orchestrator_module.time, "sleep", lambda _seconds: None)
    return controller


def test_headless_skips_ui_hooks_and_preserves_normal_controller_defaults(tmp_path: Path, monkeypatch, capsys) -> None:
    plugin = DefaultOrchestratorPlugin()
    monkeypatch.setattr(plugin, "dashboard_class", lambda: pytest.fail("dashboard hook"), raising=False)
    monkeypatch.setattr(plugin, "overview_printer", lambda: pytest.fail("overview hook"), raising=False)
    controller = _controller(tmp_path, monkeypatch, [], plugin=plugin, debug=True)
    assert not controller.low_compute
    assert controller.poll_interval == 2
    assert controller.telemetry_batch_size == 64
    controller.run()
    receipt = json.loads(status_path(controller.experiment_file).read_text())
    assert receipt["state"] == "finished"
    assert receipt["counts"]["total"] == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["state"] == "finished"
    assert summary["status_path"] == str(status_path(controller.experiment_file))
    assert "runs" not in summary


@pytest.mark.parametrize("status", [ExperimentStatus.COMPLETED, ExperimentStatus.FAILED, ExperimentStatus.TIMEOUT])
def test_terminal_telemetry_budget_settles_before_finished(tmp_path: Path, monkeypatch, status) -> None:
    controller = _controller(tmp_path, monkeypatch, [{"experiment_id": "one", "job_id": "42", "status": status}])
    receipts = []
    publish = ControllerStatusPublisher.publish

    def collect(self, state, rows, **kwargs):
        result = publish(self, state, rows, **kwargs)
        if result is not None:
            receipts.append(result)
        return result

    monkeypatch.setattr(ControllerStatusPublisher, "publish", collect)
    controller.run()
    receipt = receipts[-1]
    assert "finalizing" in [item["state"] for item in receipts]
    assert receipt["state"] == "finished"
    assert receipt["results_finalized"] is True
    assert receipt["runs"][0]["terminal_telemetry"]["attempts"] == 3
    assert receipt["runs"][0]["terminal_telemetry"]["finalized_reason"] == "retry_limit"
    assert receipt["missing_metrics"] == ["one"]
    assert receipt["ledger_fingerprint"] == ledger_fingerprint(controller.experiment_file)
    controller.connection_manager.close_all.assert_called_once()


@pytest.mark.parametrize("initial_status", [ExperimentStatus.PARTIAL, ExperimentStatus.PENDING])
def test_resubmittable_rows_never_publish_finished(tmp_path: Path, monkeypatch, initial_status) -> None:
    controller = _controller(tmp_path, monkeypatch, [{"experiment_id": "one", "status": initial_status}])
    sleeps = []

    def interrupt(_seconds):
        sleeps.append(True)
        raise KeyboardInterrupt

    monkeypatch.setattr(orchestrator_module.time, "sleep", interrupt)
    with pytest.raises(KeyboardInterrupt):
        controller.run()
    receipt = json.loads(status_path(controller.experiment_file).read_text())
    assert sleeps == [True]
    assert receipt["state"] == "stopped"
    assert receipt["results_finalized"] is False
    assert receipt["ledger_fingerprint"] is None


@pytest.mark.parametrize("failure_stage", ["connection", "preflight", "telemetry", "checkpoint", "cleanup"])
def test_failure_propagates_and_never_publishes_finished(tmp_path: Path, monkeypatch, failure_stage: str) -> None:
    controller = _controller(
        tmp_path, monkeypatch, [{"experiment_id": "one", "job_id": "42", "status": ExperimentStatus.COMPLETED}]
    )
    failure = Mock(side_effect=RuntimeError("synthetic failure"))
    if failure_stage == "connection":
        monkeypatch.setattr(controller.connection_manager, "connect_all", failure)
    elif failure_stage == "preflight":
        monkeypatch.setattr(controller, "_preflight_test_hpcs", failure)
    elif failure_stage == "telemetry":
        monkeypatch.setattr(controller, "_update_running_experiment_info", failure)
    elif failure_stage == "checkpoint":
        monkeypatch.setattr(controller, "_save_yaml", failure)
    else:
        monkeypatch.setattr(controller.connection_manager, "close_all", failure)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        controller.run()
    receipt = json.loads(status_path(controller.experiment_file).read_text())
    assert receipt["state"] == "error"
    assert receipt["results_finalized"] is False
    assert "synthetic failure" in receipt["error"]


def test_high_volume_waits_for_every_terminal_telemetry_batch(tmp_path: Path, monkeypatch) -> None:
    controller = _controller(
        tmp_path,
        monkeypatch,
        [{"experiment_id": str(i), "job_id": str(i + 1), "status": ExperimentStatus.COMPLETED} for i in range(5)],
        high_volume_threshold=2,
        telemetry_batch_size=2,
    )
    now = [0.0]
    monkeypatch.setattr(orchestrator_module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(orchestrator_module.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    controller.run()
    receipt = json.loads(status_path(controller.experiment_file).read_text())
    assert receipt["results_finalized"] is True
    assert all(run["terminal_telemetry"]["attempts"] == 3 for run in receipt["runs"])
    assert now[0] >= 70  # Fifteen reads, at most two every ten seconds.


def test_recovered_timeout_retry_cannot_publish_a_completion_receipt(tmp_path: Path, monkeypatch) -> None:
    controller = _controller(
        tmp_path,
        monkeypatch,
        [
            {
                "experiment_id": "one",
                "job_id": "42",
                "status": ExperimentStatus.TIMEOUT,
                "hpc_assignment": HPCType.OLIVIA,
            }
        ],
        retry_timeout_with_estimated_time=True,
    )
    monkeypatch.setattr(
        controller, "_catch_up_existing_state", HPCOrchestrator._catch_up_existing_state.__get__(controller)
    )
    monkeypatch.setattr(
        controller, "_apply_timeout_policy", lambda exp, *_args: exp.update(status=ExperimentStatus.PARTIAL)
    )
    monkeypatch.setattr(orchestrator_module.time, "sleep", Mock(side_effect=KeyboardInterrupt))
    with pytest.raises(KeyboardInterrupt):
        controller.run()
    receipt = json.loads(status_path(controller.experiment_file).read_text())
    assert receipt["state"] == "stopped"
    assert receipt["counts"]["partial"] == 1
    assert receipt["results_finalized"] is False


def test_headless_cli_forwards_option_and_routes_console_logs_only(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.delenv("SLURMINATOR_PLUGIN", raising=False)
    ledger = tmp_path / "ledger.yaml"
    ledger.write_text("experiments: []\n")
    console = logging.StreamHandler(sys.stdout)
    logfile = logging.FileHandler(tmp_path / "logs.txt")
    test_logger = logging.getLogger("slurminator_headless_test")
    monkeypatch.setattr(test_logger, "handlers", [console, logfile])
    test_logger.setLevel(logging.INFO)
    captured = {}

    class Controller:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self):
            test_logger.info("controller log")

    try:
        run_orchestrator_cli(
            argv=["--yaml", str(ledger), "--headless"],
            load_configs=False,
            launch_guard=lambda: None,
            orchestrator_cls=Controller,
        )
        output = capsys.readouterr()
        assert "controller log" in output.err
        assert output.out == ""
        assert (tmp_path / "logs.txt").read_text() == "controller log\n"
        assert captured["headless"] is True
        assert "low_compute" not in captured
    finally:
        logfile.close()


def test_headless_keeps_existing_gpu_launch_guard(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("SLURMINATOR_PLUGIN", raising=False)
    ledger = tmp_path / "ledger.yaml"
    ledger.write_text("experiments: []\n")
    with pytest.raises(SystemExit) as error:
        run_orchestrator_cli(
            argv=["--yaml", str(ledger), "--headless"],
            load_configs=False,
            launch_guard=lambda: "GPU controllers blocked",
            orchestrator_cls=lambda **_kwargs: pytest.fail("must not launch"),
        )
    assert error.value.code == 2


def test_ledger_edit_during_cleanup_prevents_finished_and_is_preserved(tmp_path: Path, monkeypatch, capsys) -> None:
    controller = _controller(tmp_path, monkeypatch, [{"experiment_id": "one", "status": ExperimentStatus.COMPLETED}])
    external_edit = "experiments: []\n# external edit during connection cleanup\n"
    monkeypatch.setattr(
        controller.connection_manager, "close_all", lambda: controller.experiment_file.write_text(external_edit)
    )
    with pytest.raises(RuntimeError, match="changed after the final controller save"):
        controller.run()
    receipt = json.loads(status_path(controller.experiment_file).read_text())
    assert receipt["state"] == "error"
    assert receipt["results_finalized"] is False
    assert receipt["ledger_fingerprint"] is None
    assert controller.experiment_file.read_text() == external_edit
    assert capsys.readouterr().out == ""


def test_heartbeat_continues_during_blocking_startup_and_maintenance(tmp_path: Path, monkeypatch) -> None:
    controller = _controller(tmp_path, monkeypatch, [])
    original_init = ControllerStatusPublisher.__init__
    original_write = ControllerStatusPublisher._write_payload
    heartbeat_seen = threading.Event()
    phase = [None]
    snapshots = []

    def quick_heartbeat(self, ledger, **kwargs):
        kwargs["heartbeat_interval_seconds"] = 0.01
        original_init(self, ledger, **kwargs)

    def observe_write(self, payload: str) -> None:
        original_write(self, payload)
        snapshot = json.loads(payload)
        if snapshot["state"] == phase[0] and snapshot["heartbeat_at"] != snapshot["progress_updated_at"]:
            snapshots.append(snapshot)
            heartbeat_seen.set()

    def blocking_operation(state: str) -> None:
        phase[0] = state
        heartbeat_seen.clear()
        assert heartbeat_seen.wait(1), f"No independent heartbeat during {state}"

    monkeypatch.setattr(ControllerStatusPublisher, "__init__", quick_heartbeat)
    monkeypatch.setattr(ControllerStatusPublisher, "_write_payload", observe_write)
    monkeypatch.setattr(controller.connection_manager, "connect_all", lambda: blocking_operation("starting"))
    maintain = controller._maintain_cycle

    def blocking_maintenance(*args, **kwargs):
        blocking_operation("running")
        return maintain(*args, **kwargs)

    monkeypatch.setattr(controller, "_maintain_cycle", blocking_maintenance)
    controller.run()
    assert {snapshot["state"] for snapshot in snapshots} == {"starting", "running"}
    assert all(snapshot["heartbeat_at"] > snapshot["progress_updated_at"] for snapshot in snapshots)
    assert json.loads(status_path(controller.experiment_file).read_text())["state"] == "finished"

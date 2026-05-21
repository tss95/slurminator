import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from slurminator.command_queue import (
    Command,
    CommandQueueContext,
    default_command_handlers,
    handle_cancel_all,
    handle_cancel_run,
    handle_relaunch_run,
    handle_update_run_settings,
    handle_pause_submissions,
    handle_resume_submissions,
    handle_set_concurrency_limit,
    process_command_queue,
)
from slurminator.config import HPCType
from slurminator.experiments import ExperimentStatus

pytestmark = pytest.mark.unit


class FakeConnection:
    def __init__(self) -> None:
        self.commands: list[tuple[HPCType, str, bool]] = []

    def run_command(self, hpc_type, command, prefer_remote=False):
        self.commands.append((hpc_type, command, prefer_remote))
        return "", ""


def _context(tmp_path: Path, exps: list[dict] | None = None, orchestrator=None, connection=None) -> CommandQueueContext:
    return CommandQueueContext(
        save_path=tmp_path,
        handlers=default_command_handlers(),
        exps=exps or [],
        orchestrator=orchestrator or SimpleNamespace(submissions_paused=False, concurrency_limits={}),
        connection_manager=connection or FakeConnection(),
    )


def _command(action: str, target: dict) -> Command:
    return Command(command_id=f"cmd-{action}", issued_at=time.time(), issued_by="tester", action=action, target=target)


def _write_pending(save_path: Path, command: Command, name: str = "0000000000001_cmd.json") -> Path:
    pending = save_path / ".orchestrator_status" / "_commands" / "pending"
    pending.mkdir(parents=True, exist_ok=True)
    path = pending / name
    path.write_text(command.model_dump_json(), encoding="utf-8")
    return path


def test_handle_cancel_run_issues_scancel_for_active_run(tmp_path) -> None:
    connection = FakeConnection()
    exps = [
        {
            "experiment_id": "exp-1",
            "status": ExperimentStatus.RUNNING,
            "hpc_assignment": HPCType.OLIVIA,
            "job_id": "12345",
        }
    ]

    handle_cancel_run(
        _command("cancel_run", {"experiment_id": "exp-1", "job_id": "12345"}),
        _context(tmp_path, exps, connection=connection),
    )

    assert connection.commands == [(HPCType.OLIVIA, "scancel 12345", True)]


def test_handle_cancel_run_ignores_inactive_or_missing_run(tmp_path) -> None:
    connection = FakeConnection()
    exps = [{"experiment_id": "exp-1", "status": ExperimentStatus.COMPLETED, "job_id": "12345"}]

    handle_cancel_run(
        _command("cancel_run", {"experiment_id": "exp-1", "job_id": "12345"}),
        _context(tmp_path, exps, connection=connection),
    )
    handle_cancel_run(
        _command("cancel_run", {"experiment_id": "missing", "job_id": "9"}),
        _context(tmp_path, exps, connection=connection),
    )

    assert connection.commands == []


def test_handle_cancel_all_issues_scancel_for_all_active_runs(tmp_path) -> None:
    connection = FakeConnection()
    exps = [
        {"experiment_id": "queued", "status": "queued", "hpc_assignment": "olivia", "job_id": "1"},
        {"experiment_id": "running", "status": ExperimentStatus.RUNNING, "hpc_assignment": HPCType.FOX, "job_id": "2"},
        {"experiment_id": "done", "status": ExperimentStatus.COMPLETED, "hpc_assignment": HPCType.FOX, "job_id": "3"},
    ]

    handle_cancel_all(_command("cancel_all", {"scope": "session"}), _context(tmp_path, exps, connection=connection))

    assert connection.commands == [(HPCType.OLIVIA, "scancel 1", True), (HPCType.FOX, "scancel 2", True)]


def test_handle_relaunch_run_resets_terminal_experiment_for_submission(tmp_path) -> None:
    exp = {
        "experiment_id": "exp-1",
        "status": "ExperimentStatus.FAILED",
        "hpc_assignment": HPCType.OLIVIA,
        "job_id": "12345",
        "queued_timestamp": 1.0,
        "running_timestamp": 2.0,
        "failed_timestamp": 3.0,
        "output_dir": "/old/logs",
        "save_path": "/remote/save",
        "history": [{"epoch": 1}],
        "history_last_read_offset": 100,
        "sweep_params": "lr=0.1",
    }

    handle_relaunch_run(
        _command("relaunch_run", {"experiment_id": "exp-1", "job_id": "12345"}), _context(tmp_path, [exp])
    )

    assert exp["status"] == ExperimentStatus.PENDING
    assert exp["hpc_assignment"] == HPCType.OLIVIA
    assert exp["save_path"] == "/remote/save"
    assert exp["sweep_params"] == "lr=0.1"
    assert exp["manual_relaunch_count"] == 1
    assert exp["relaunch_previous_status"] == "failed"
    assert exp["relaunch_source_job_id"] == "12345"
    assert isinstance(exp["relaunch_requested_at"], float)
    for key in [
        "job_id",
        "queued_timestamp",
        "running_timestamp",
        "failed_timestamp",
        "output_dir",
        "history",
        "history_last_read_offset",
    ]:
        assert key not in exp


def test_handle_relaunch_run_accepts_canceled_status_alias(tmp_path) -> None:
    exp = {"experiment_id": "exp-1", "status": "CANCELED by 12345", "hpc_assignment": HPCType.OLIVIA, "job_id": "12345"}

    handle_relaunch_run(
        _command("relaunch_run", {"experiment_id": "exp-1", "job_id": "12345"}), _context(tmp_path, [exp])
    )

    assert exp["status"] == ExperimentStatus.PENDING
    assert exp["relaunch_previous_status"] == "cancelled"
    assert exp["relaunch_source_job_id"] == "12345"


def test_handle_relaunch_run_rejects_active_experiment(tmp_path) -> None:
    exp = {
        "experiment_id": "exp-1",
        "status": ExperimentStatus.RUNNING,
        "hpc_assignment": HPCType.OLIVIA,
        "job_id": "12345",
    }

    with pytest.raises(ValueError, match="cannot relaunch"):
        handle_relaunch_run(
            _command("relaunch_run", {"experiment_id": "exp-1", "job_id": "12345"}), _context(tmp_path, [exp])
        )

    assert exp["status"] == ExperimentStatus.RUNNING
    assert exp["job_id"] == "12345"


def test_handle_relaunch_run_rejects_stale_job_id(tmp_path) -> None:
    exp = {
        "experiment_id": "exp-1",
        "status": ExperimentStatus.FAILED,
        "hpc_assignment": HPCType.OLIVIA,
        "job_id": "new-job",
    }

    with pytest.raises(ValueError, match="stale relaunch command"):
        handle_relaunch_run(
            _command("relaunch_run", {"experiment_id": "exp-1", "job_id": "old-job"}), _context(tmp_path, [exp])
        )

    assert exp["status"] == ExperimentStatus.FAILED
    assert exp["job_id"] == "new-job"


def test_handle_update_run_settings_updates_next_submission_fields(tmp_path) -> None:
    exp = {
        "experiment_id": "exp-1",
        "status": ExperimentStatus.PENDING,
        "resource_overrides": {"mem_gb": 80, "gpu_count": 1},
        "pinned_hpc": "FOX",
    }

    handle_update_run_settings(
        _command(
            "update_run_settings",
            {
                "experiment_id": "exp-1",
                "settings": {"time_hours": "6", "memory_gb": "160", "gpu_count": "2", "pinned_hpc": "OLIVIA"},
            },
        ),
        _context(tmp_path, [exp]),
    )

    assert exp["time_hours_override"] == 6
    assert exp["resource_overrides"] == {"memory_gb": 160, "gpu_count": 2}
    assert exp["pinned_hpc"] == "OLIVIA"
    assert isinstance(exp["settings_updated_at"], float)


def test_handle_update_run_settings_clears_blank_overrides(tmp_path) -> None:
    exp = {
        "experiment_id": "exp-1",
        "status": ExperimentStatus.PENDING,
        "time_hours_override": 6,
        "resource_overrides": {"memory_gb": 160, "gpu_count": 2},
        "pinned_hpc": "OLIVIA",
    }

    handle_update_run_settings(
        _command(
            "update_run_settings",
            {
                "experiment_id": "exp-1",
                "settings": {"time_hours": None, "memory_gb": "", "gpu_count": None, "pinned_hpc": ""},
            },
        ),
        _context(tmp_path, [exp]),
    )

    assert "time_hours_override" not in exp
    assert "resource_overrides" not in exp
    assert "pinned_hpc" not in exp
    assert isinstance(exp["settings_updated_at"], float)


def test_handle_update_run_settings_rejects_invalid_values(tmp_path) -> None:
    exp = {"experiment_id": "exp-1", "status": ExperimentStatus.PENDING}

    with pytest.raises(ValueError, match="memory_gb must be a positive integer"):
        handle_update_run_settings(
            _command("update_run_settings", {"experiment_id": "exp-1", "settings": {"memory_gb": "0"}}),
            _context(tmp_path, [exp]),
        )

    with pytest.raises(ValueError, match="unknown hpc"):
        handle_update_run_settings(
            _command("update_run_settings", {"experiment_id": "exp-1", "settings": {"pinned_hpc": "UNKNOWN"}}),
            _context(tmp_path, [exp]),
        )


def test_pause_and_resume_submission_handlers_are_idempotent(tmp_path) -> None:
    orchestrator = SimpleNamespace(submissions_paused=False, concurrency_limits={})
    ctx = _context(tmp_path, orchestrator=orchestrator)

    handle_pause_submissions(_command("pause_submissions", {}), ctx)
    handle_pause_submissions(_command("pause_submissions", {}), ctx)
    assert orchestrator.submissions_paused is True

    handle_resume_submissions(_command("resume_submissions", {}), ctx)
    handle_resume_submissions(_command("resume_submissions", {}), ctx)
    assert orchestrator.submissions_paused is False


def test_handle_set_concurrency_limit_updates_orchestrator_limits(tmp_path) -> None:
    orchestrator = SimpleNamespace(submissions_paused=False, concurrency_limits={HPCType.OLIVIA: 1})

    handle_set_concurrency_limit(
        _command("set_concurrency_limit", {"hpc": "OLIVIA", "limit": 3}), _context(tmp_path, orchestrator=orchestrator)
    )

    assert orchestrator.concurrency_limits[HPCType.OLIVIA] == 3


def test_handle_set_concurrency_limit_rejects_unconnected_hpc(tmp_path) -> None:
    orchestrator = SimpleNamespace(
        submissions_paused=False,
        concurrency_limits={HPCType.OLIVIA: 1, HPCType.FOX: 0},
        connection_manager=SimpleNamespace(_connected={HPCType.OLIVIA: True, HPCType.FOX: False}),
    )

    with pytest.raises(ValueError, match="unconnected hpc: FOX"):
        handle_set_concurrency_limit(
            _command("set_concurrency_limit", {"hpc": "FOX", "limit": 3}), _context(tmp_path, orchestrator=orchestrator)
        )

    assert orchestrator.concurrency_limits[HPCType.FOX] == 0


def test_process_command_queue_moves_successful_commands_to_processed(tmp_path) -> None:
    _write_pending(tmp_path, _command("pause_submissions", {}))
    orchestrator = SimpleNamespace(submissions_paused=False, concurrency_limits={})

    count = process_command_queue(_context(tmp_path, orchestrator=orchestrator))

    assert count == 1
    assert orchestrator.submissions_paused is True
    assert not (tmp_path / ".orchestrator_status" / "_commands" / "pending" / "0000000000001_cmd.json").exists()
    assert (tmp_path / ".orchestrator_status" / "_commands" / "processed" / "0000000000001_cmd.json").exists()


def test_process_command_queue_moves_handler_failures_to_failed_with_sidecar(tmp_path) -> None:
    command = _command("explode", {})
    _write_pending(tmp_path, command)

    def explode(_cmd, _ctx):
        raise RuntimeError("boom")

    ctx = _context(tmp_path)
    ctx.handlers = {"explode": explode}

    count = process_command_queue(ctx)

    failed_dir = tmp_path / ".orchestrator_status" / "_commands" / "failed"
    assert count == 0
    assert (failed_dir / "0000000000001_cmd.json").exists()
    error_text = (failed_dir / f"{command.command_id}.error.txt").read_text(encoding="utf-8")
    assert "handler_error: boom" in error_text
    assert "Traceback" in error_text


def test_process_command_queue_moves_unknown_actions_to_failed(tmp_path) -> None:
    command = _command("unknown", {})
    _write_pending(tmp_path, command)

    count = process_command_queue(_context(tmp_path))

    failed_dir = tmp_path / ".orchestrator_status" / "_commands" / "failed"
    assert count == 0
    assert (failed_dir / "0000000000001_cmd.json").exists()
    assert "unknown_action: unknown" in (failed_dir / f"{command.command_id}.error.txt").read_text(encoding="utf-8")


def test_process_command_queue_returns_zero_when_no_pending_commands(tmp_path) -> None:
    assert process_command_queue(_context(tmp_path)) == 0

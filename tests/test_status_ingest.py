import pytest

from slurminator.config import HPCType
from slurminator.experiments import ExperimentStatus
from slurminator.schemas.status_schema import HistoryEntry, OrchestratorStatus
from slurminator.status_ingest import (
    HISTORY_TERMINAL_BOUND,
    StatusIngestContext,
    force_read_full_history,
    history_file_path,
    status_file_path,
    update_running_experiment_info,
)

pytestmark = pytest.mark.unit


class FakeStatusConnection:
    def __init__(self, *, status_payload: str = "", history_payload: str = "") -> None:
        self.status_payload = status_payload
        self.history_payload = history_payload
        self.commands: list[tuple[HPCType, str]] = []

    def run_command(self, hpc_type, command):
        self.commands.append((hpc_type, command))
        if command.startswith("cat "):
            return self.status_payload, ""
        if command.startswith("stat "):
            return str(len(self.history_payload.encode("utf-8"))), ""
        if command.startswith("tail "):
            marker = "tail -c +"
            start = int(command.split(marker, 1)[1].split()[0]) - 1
            return self.history_payload.encode("utf-8")[start:].decode("utf-8"), ""
        raise AssertionError(f"unexpected command: {command}")


def _status_json(*, job_id: str = "12345") -> str:
    return OrchestratorStatus(
        experiment_id="exp-1",
        job_id=job_id,
        status="running",
        last_update=100.0,
        progress={"unit": "epoch", "current": 1, "total": 2, "current_epoch": 1, "total_epochs": 2},
        metrics={"train/loss": 0.5},
        display={"run_name": "run-1"},
    ).model_dump_json()


def _history_line(*, attempt: int = 1, epoch: int = 1, loss: float = 0.5) -> str:
    return (
        HistoryEntry(
            timestamp=100.0 + epoch, attempt=attempt, epoch=epoch, step=None, metrics={"train/loss": loss}
        ).model_dump_json()
        + "\n"
    )


def _context(connection: FakeStatusConnection) -> StatusIngestContext:
    saved: list[dict] = []
    return StatusIngestContext(
        connection_manager=connection,
        hpc_configs={},
        load_yaml=lambda: {"experiments": [{"experiment_id": "exp-1"}]},
        save_yaml=saved.append,
    )


def test_status_and_history_file_paths_use_sweep_directory_when_present() -> None:
    assert status_file_path("/save", "4242", "abc") == "/save/.orchestrator_status/sweep_abc/status_4242.json"
    assert history_file_path("/save", "4242", "abc") == "/save/.orchestrator_status/sweep_abc/history_4242.jsonl"


def test_update_running_experiment_info_merges_incremental_history() -> None:
    history_payload = _history_line(epoch=1, loss=1.0) + _history_line(epoch=2, loss=0.8)
    connection = FakeStatusConnection(status_payload=_status_json(), history_payload=history_payload)
    exp = {
        "experiment_id": "exp-1",
        "status": "running",
        "job_id": "12345",
        "hpc_assignment": HPCType.OLIVIA,
        "save_path": "/save",
    }

    data = update_running_experiment_info(exp, _context(connection))

    assert data is not None
    assert len(exp["history"]) == 2
    assert exp["history"][1]["metrics"] == {"train/loss": 0.8}
    assert exp["history_attempt_max"] == 1
    assert exp["history_last_read_offset"] == len(history_payload.encode("utf-8"))


def test_force_read_full_history_resets_offset_and_replaces_history() -> None:
    history_payload = _history_line(attempt=2, epoch=3, loss=0.3)
    connection = FakeStatusConnection(history_payload=history_payload)
    exp = {
        "experiment_id": "exp-1",
        "status": ExperimentStatus.COMPLETED,
        "job_id": "12345",
        "hpc_assignment": HPCType.OLIVIA,
        "save_path": "/save",
        "history": [{"old": True}],
        "history_last_read_offset": 999,
    }

    force_read_full_history(exp, _context(connection))

    assert exp["history"] == [HistoryEntry.model_validate_json(history_payload).model_dump(mode="json")]
    assert exp["history_last_read_offset"] == len(history_payload.encode("utf-8"))
    assert exp["history_attempt_max"] == 2


def test_terminal_history_is_bounded_once_status_is_ingested() -> None:
    connection = FakeStatusConnection(status_payload=_status_json(), history_payload="")
    exp = {
        "experiment_id": "exp-1",
        "status": ExperimentStatus.COMPLETED,
        "job_id": "12345",
        "hpc_assignment": HPCType.OLIVIA,
        "save_path": "/save",
        "history": [{"epoch": idx} for idx in range(HISTORY_TERMINAL_BOUND + 5)],
    }

    update_running_experiment_info(exp, _context(connection))

    assert len(exp["history"]) == HISTORY_TERMINAL_BOUND
    assert exp["history"][0] == {"epoch": 5}
    assert exp["history_truncated"] is True

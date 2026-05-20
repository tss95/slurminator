import re

import pytest

from slurminator.config import HPCType
from slurminator.history_ingest import read_history_incremental
from slurminator.schemas.status_schema import HistoryEntry

pytestmark = pytest.mark.unit


class FakeHistoryConnection:
    def __init__(self, content: str | None) -> None:
        self.content = content
        self.commands: list[tuple[HPCType, str]] = []

    def run_command(self, hpc_type, command):
        self.commands.append((hpc_type, command))
        if command.startswith("stat "):
            return (str(len(self.content.encode("utf-8"))) if self.content is not None else "0", "")
        if command.startswith("tail "):
            match = re.search(r"tail -c \+(\d+)", command)
            assert match is not None
            start_index = int(match.group(1)) - 1
            payload = self.content or ""
            return payload.encode("utf-8")[start_index:].decode("utf-8"), ""
        raise AssertionError(f"unexpected command: {command}")


def _entry(*, attempt: int = 1, epoch: int = 1, step: int = 10, loss: float = 0.5) -> str:
    return HistoryEntry(
        timestamp=100.0 + epoch, attempt=attempt, epoch=epoch, step=step, metrics={"train/loss": loss}
    ).model_dump_json()


def test_read_history_incremental_reads_new_lines_and_skips_invalid() -> None:
    first = _entry(epoch=1, loss=1.0)
    second = _entry(epoch=2, loss=0.8)
    content = f"{first}\nnot-json\n{second}\n"
    connection = FakeHistoryConnection(content)

    result = read_history_incremental(
        connection_manager=connection,
        hpc_type=HPCType.OLIVIA,
        history_path="/save/.orchestrator_status/history_123.jsonl",
        last_offset=len(f"{first}\n".encode("utf-8")),
    )

    assert result.truncated is False
    assert result.new_offset == len(content.encode("utf-8"))
    assert result.new_entries == [HistoryEntry.model_validate_json(second).model_dump(mode="json")]
    assert len(connection.commands) == 2


def test_read_history_incremental_returns_empty_for_missing_file() -> None:
    connection = FakeHistoryConnection(None)

    result = read_history_incremental(
        connection_manager=connection,
        hpc_type=HPCType.OLIVIA,
        history_path="/missing/history_123.jsonl",
        last_offset=25,
    )

    assert result.new_entries == []
    assert result.new_offset == 0
    assert result.truncated is False
    assert len(connection.commands) == 1


def test_read_history_incremental_refetches_from_start_after_truncation() -> None:
    content = f"{_entry(attempt=2, epoch=3, loss=0.25)}\n"
    connection = FakeHistoryConnection(content)

    result = read_history_incremental(
        connection_manager=connection,
        hpc_type=HPCType.OLIVIA,
        history_path="/save/.orchestrator_status/history_123.jsonl",
        last_offset=len(content.encode("utf-8")) + 100,
    )

    assert result.truncated is True
    assert result.new_offset == len(content.encode("utf-8"))
    assert result.new_entries[0]["attempt"] == 2
    assert result.new_entries[0]["epoch"] == 3
    assert len(connection.commands) == 2

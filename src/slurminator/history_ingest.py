"""Incremental JSONL history ingestion helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from shlex import quote

from slurminator.schemas.status_schema import HistoryEntry

logger = logging.getLogger("slurminator")


@dataclass
class HistoryReadResult:
    """Result of one incremental history-file read."""

    new_entries: list[dict]
    new_offset: int
    truncated: bool = False


def read_history_incremental(*, connection_manager, hpc_type, history_path: str, last_offset: int) -> HistoryReadResult:
    """Read and parse history JSONL entries appended after ``last_offset``."""
    safe_offset = max(int(last_offset or 0), 0)
    path = quote(str(history_path))
    size_out, _ = connection_manager.run_command(hpc_type, f'stat -c "%s" {path} 2>/dev/null || echo 0')
    size = _parse_size(size_out)
    if size <= 0:
        return HistoryReadResult(new_entries=[], new_offset=0)

    truncated = size < safe_offset
    read_offset = 0 if truncated else safe_offset
    if size == read_offset:
        return HistoryReadResult(new_entries=[], new_offset=size, truncated=truncated)

    tail_out, _ = connection_manager.run_command(hpc_type, f"tail -c +{read_offset + 1} {path} 2>/dev/null")
    return HistoryReadResult(new_entries=_parse_history_lines(tail_out), new_offset=size, truncated=truncated)


def _parse_size(stdout: str) -> int:
    try:
        return max(int(str(stdout).strip().splitlines()[-1]), 0)
    except Exception:
        return 0


def _parse_history_lines(payload: str) -> list[dict]:
    entries: list[dict] = []
    for line in str(payload).splitlines():
        if not line.strip():
            continue
        try:
            entries.append(HistoryEntry.model_validate_json(line).model_dump(mode="json"))
        except Exception as exc:
            logger.debug("Skipping invalid history line: %s", exc)
    return entries


__all__ = ["HistoryReadResult", "read_history_incremental"]

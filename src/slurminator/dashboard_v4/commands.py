"""Dashboard-side command queue writes."""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

from slurminator.command_queue import Command


def submit_command(save_path: Path, action: str, target: dict, confirm_token: str | None = None) -> Command:
    """Atomically write one pending command and return its schema object."""
    cmd_dir = Path(save_path) / ".orchestrator_status" / "_commands" / "pending"
    cmd_dir.mkdir(parents=True, exist_ok=True)

    cmd = Command(
        command_id=str(uuid.uuid4()),
        issued_at=time.time(),
        issued_by=os.getenv("USER", "unknown"),
        action=action,
        target=target,
        confirm_token=confirm_token,
    )

    path = cmd_dir / f"{int(cmd.issued_at * 1000):013d}_{cmd.command_id[:8]}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(cmd.model_dump_json(indent=2), encoding="utf-8")
    tmp.rename(path)
    return cmd


__all__ = ["submit_command"]

"""Operator command queue processing for running orchestrators."""

from __future__ import annotations

import logging
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from slurminator.config import HPCType
from slurminator.experiments import ExperimentStatus

logger = logging.getLogger("slurminator")


class Command(BaseModel):
    """A user-issued operator command."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    command_id: str
    issued_at: float
    issued_by: str
    action: str
    target: dict[str, Any]
    confirm_token: str | None = None


CommandHandler = Callable[[Command, "CommandQueueContext"], None]


@dataclass
class CommandQueueContext:
    """Dependencies required to process command queue files."""

    save_path: Path
    handlers: dict[str, CommandHandler]
    exps: list[dict[str, Any]]
    orchestrator: Any
    connection_manager: Any


def default_command_handlers() -> dict[str, CommandHandler]:
    """Return the Slice 3 command handler set."""
    return {
        "cancel_run": handle_cancel_run,
        "cancel_all": handle_cancel_all,
        "pause_submissions": handle_pause_submissions,
        "resume_submissions": handle_resume_submissions,
        "set_concurrency_limit": handle_set_concurrency_limit,
    }


def process_command_queue(context: CommandQueueContext) -> int:
    """Read pending commands, dispatch handlers, and move files by result."""
    pending_dir = _queue_dir(context, "pending")
    if not pending_dir.exists():
        return 0

    processed_count = 0
    for path in sorted(pending_dir.glob("*.json")):
        try:
            cmd = Command.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as exc:
            _move_to_failed(path, error=f"parse_error: {exc}", context=context)
            logger.warning("Command file %s failed to parse: %s", path.name, exc)
            continue

        handler = context.handlers.get(cmd.action)
        if handler is None:
            error = f"unknown_action: {cmd.action}"
            _move_to_failed(path, error=error, context=context, command=cmd)
            logger.warning("Command %s (action=%s) failed: %s", cmd.command_id, cmd.action, error)
            continue

        processing_path = _move_to_processing(path, context=context)
        try:
            handler(cmd, context)
        except Exception as exc:
            error = f"handler_error: {exc}"
            _move_to_failed(
                processing_path, error=error, context=context, command=cmd, traceback_text=traceback.format_exc()
            )
            logger.warning("Command %s (action=%s) failed: %s", cmd.command_id, cmd.action, exc)
            continue

        _move_to_processed(processing_path, context=context)
        processed_count += 1

    return processed_count


def handle_cancel_run(cmd: Command, ctx: CommandQueueContext) -> None:
    """Cancel one queued or running experiment if it is still active."""
    exp = _find_experiment(ctx, cmd.target.get("experiment_id"))
    if exp is None or not _status_in(exp.get("status"), {ExperimentStatus.QUEUED, ExperimentStatus.RUNNING}):
        return
    hpc_type = _coerce_hpc(exp.get("hpc_assignment"))
    job_id = exp.get("job_id")
    if hpc_type is None or not job_id:
        return
    scancel_via_connection(ctx.connection_manager, hpc_type, str(job_id))


def handle_cancel_all(cmd: Command, ctx: CommandQueueContext) -> None:
    """Cancel all queued or running experiments in the session."""
    for exp in ctx.exps:
        if not _status_in(exp.get("status"), {ExperimentStatus.QUEUED, ExperimentStatus.RUNNING}):
            continue
        hpc_type = _coerce_hpc(exp.get("hpc_assignment"))
        job_id = exp.get("job_id")
        if hpc_type is None or not job_id:
            continue
        scancel_via_connection(ctx.connection_manager, hpc_type, str(job_id))


def handle_pause_submissions(cmd: Command, ctx: CommandQueueContext) -> None:
    """Pause new job submission for the current orchestrator session."""
    ctx.orchestrator.submissions_paused = True


def handle_resume_submissions(cmd: Command, ctx: CommandQueueContext) -> None:
    """Resume new job submission for the current orchestrator session."""
    ctx.orchestrator.submissions_paused = False


def handle_set_concurrency_limit(cmd: Command, ctx: CommandQueueContext) -> None:
    """Update one per-cluster concurrency limit for the current session."""
    hpc_type = _coerce_hpc(cmd.target.get("hpc"))
    if hpc_type is None:
        raise ValueError(f"unknown hpc: {cmd.target.get('hpc')!r}")
    limit = int(cmd.target.get("limit"))
    if limit < 0:
        raise ValueError(f"concurrency limit must be >= 0, got {limit}")
    ctx.orchestrator.concurrency_limits[hpc_type] = limit


def scancel_via_connection(connection_manager: Any, hpc_type: HPCType, job_id: str) -> None:
    """Issue ``scancel`` through the orchestrator connection manager."""
    connection_manager.run_command(hpc_type, f"scancel {job_id}", prefer_remote=True)


def _find_experiment(ctx: CommandQueueContext, experiment_id: object) -> dict[str, Any] | None:
    if experiment_id is None:
        return None
    for exp in ctx.exps:
        if exp.get("experiment_id") == experiment_id:
            return exp
    return None


def _status_in(status: object, allowed: set[ExperimentStatus]) -> bool:
    if not isinstance(status, ExperimentStatus):
        try:
            status = ExperimentStatus(str(status))
        except ValueError:
            try:
                status = ExperimentStatus[str(status).upper()]
            except KeyError:
                return False
    return status in allowed


def _coerce_hpc(value: object) -> HPCType | None:
    if isinstance(value, HPCType):
        return value
    if value is None:
        return None
    text = str(value).strip()
    try:
        return HPCType(text)
    except ValueError:
        try:
            return HPCType[text.upper()]
        except KeyError:
            return None


def _queue_dir(context: CommandQueueContext, name: str) -> Path:
    return context.save_path / ".orchestrator_status" / "_commands" / name


def _move_to_processing(path: Path, *, context: CommandQueueContext) -> Path:
    return _move_to_state(path, "processing", context=context)


def _move_to_processed(path: Path, *, context: CommandQueueContext) -> Path:
    return _move_to_state(path, "processed", context=context)


def _move_to_failed(
    path: Path,
    *,
    error: str,
    context: CommandQueueContext,
    command: Command | None = None,
    traceback_text: str | None = None,
) -> Path:
    failed_path = _move_to_state(path, "failed", context=context)
    error_id = command.command_id if command is not None else failed_path.stem
    sidecar = failed_path.with_name(f"{error_id}.error.txt")
    payload = error if traceback_text is None else f"{error}\n\n{traceback_text}"
    sidecar.write_text(payload + "\n", encoding="utf-8")
    return failed_path


def _move_to_state(path: Path, state: str, *, context: CommandQueueContext) -> Path:
    target_dir = _queue_dir(context, state)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / path.name
    path.replace(target)
    return target


__all__ = [
    "Command",
    "CommandQueueContext",
    "default_command_handlers",
    "handle_cancel_all",
    "handle_cancel_run",
    "handle_pause_submissions",
    "handle_resume_submissions",
    "handle_set_concurrency_limit",
    "process_command_queue",
    "scancel_via_connection",
]

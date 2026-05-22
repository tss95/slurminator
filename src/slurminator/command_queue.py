"""Operator command queue processing for running orchestrators."""

from __future__ import annotations

import logging
import time
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
        "relaunch_run": handle_relaunch_run,
        "update_run_settings": handle_update_run_settings,
        "update_global_run_settings": handle_update_global_run_settings,
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
    """Cancel active experiments and pause further submissions in the session."""
    ctx.orchestrator.submissions_paused = True
    for exp in ctx.exps:
        if not _status_in(exp.get("status"), {ExperimentStatus.QUEUED, ExperimentStatus.RUNNING}):
            continue
        hpc_type = _coerce_hpc(exp.get("hpc_assignment"))
        job_id = exp.get("job_id")
        if hpc_type is None or not job_id:
            continue
        scancel_via_connection(ctx.connection_manager, hpc_type, str(job_id))


def handle_relaunch_run(cmd: Command, ctx: CommandQueueContext) -> None:
    """Reset one terminal experiment so the next poll can submit it again."""
    exp = _find_experiment(ctx, cmd.target.get("experiment_id"))
    if exp is None:
        raise ValueError(f"unknown experiment_id: {cmd.target.get('experiment_id')!r}")

    expected_job_id = cmd.target.get("job_id")
    current_job_id = exp.get("job_id")
    if expected_job_id is not None and current_job_id is not None and str(expected_job_id) != str(current_job_id):
        raise ValueError(
            f"stale relaunch command for {exp.get('experiment_id')!r}: "
            f"expected job_id {expected_job_id!r}, found {current_job_id!r}"
        )

    status = _coerce_status(exp.get("status"))
    if status is None or status not in _RELAUNCHABLE_STATUSES:
        raise ValueError(f"cannot relaunch experiment {exp.get('experiment_id')!r} from status {exp.get('status')!r}")

    previous_status = status.value
    previous_job_id = exp.get("job_id")
    exp["status"] = ExperimentStatus.PENDING
    exp["manual_relaunch_count"] = int(exp.get("manual_relaunch_count", 0) or 0) + 1
    exp["relaunch_requested_at"] = time.time()
    exp["relaunch_previous_status"] = previous_status
    if previous_job_id is not None:
        exp["relaunch_source_job_id"] = str(previous_job_id)

    for key in _RELAUNCH_RESET_FIELDS:
        exp.pop(key, None)


def handle_update_run_settings(cmd: Command, ctx: CommandQueueContext) -> None:
    """Update per-run settings that affect the next submission."""
    exp = _find_experiment(ctx, cmd.target.get("experiment_id"))
    if exp is None:
        raise ValueError(f"unknown experiment_id: {cmd.target.get('experiment_id')!r}")

    settings = cmd.target.get("settings")
    if not isinstance(settings, dict):
        raise ValueError("update_run_settings requires target.settings")

    _apply_run_settings(exp, settings, updated_at=time.time())


def handle_update_global_run_settings(cmd: Command, ctx: CommandQueueContext) -> None:
    """Update settings for all runs whose next submission has not happened yet."""
    settings = cmd.target.get("settings")
    if not isinstance(settings, dict):
        raise ValueError("update_global_run_settings requires target.settings")

    scope = str(cmd.target.get("scope", "pending")).strip().lower()
    if scope != "pending":
        raise ValueError(f"unsupported global settings scope: {scope!r}")

    updated_at = time.time()
    for exp in ctx.exps:
        if not _status_in(exp.get("status"), _NEXT_SUBMISSION_SETTINGS_STATUSES):
            continue
        _apply_run_settings(exp, settings, updated_at=updated_at)


def _apply_run_settings(exp: dict[str, Any], settings: dict[str, Any], *, updated_at: float) -> None:
    """Apply validated next-submission settings to one experiment row."""
    if "time_hours" in settings:
        time_hours = _coerce_optional_positive_int(settings["time_hours"], "time_hours")
        if time_hours is None:
            exp.pop("time_hours_override", None)
            _remove_resource_override(exp, "time_hours")
        else:
            exp["time_hours_override"] = time_hours
            _remove_resource_override(exp, "time_hours")

    if "memory_gb" in settings:
        memory_gb = _coerce_optional_positive_int(settings["memory_gb"], "memory_gb")
        _set_resource_override(exp, "memory_gb", memory_gb, aliases=("mem_gb",))

    if "gpu_count" in settings:
        gpu_count = _coerce_optional_positive_int(settings["gpu_count"], "gpu_count")
        _set_resource_override(exp, "gpu_count", gpu_count)

    if "pinned_hpc" in settings:
        pinned_hpc = _coerce_optional_hpc(settings["pinned_hpc"])
        if pinned_hpc is None:
            exp.pop("pinned_hpc", None)
        else:
            exp["pinned_hpc"] = pinned_hpc.value

    exp["settings_updated_at"] = updated_at


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
    if not _is_connected_hpc(ctx.orchestrator, hpc_type):
        raise ValueError(f"cannot set concurrency limit for unconnected hpc: {hpc_type.name}")
    limit = int(cmd.target.get("limit"))
    if limit < 0:
        raise ValueError(f"concurrency limit must be >= 0, got {limit}")
    ctx.orchestrator.concurrency_limits[hpc_type] = limit


def _is_connected_hpc(orchestrator: Any, hpc_type: HPCType) -> bool:
    """Return False only when the orchestrator exposes a disconnected HPC map."""
    connection_manager = getattr(orchestrator, "connection_manager", None)
    connected = getattr(connection_manager, "_connected", None)
    if not isinstance(connected, dict):
        return True
    return bool(connected.get(hpc_type, False))


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
    coerced = _coerce_status(status)
    return coerced in allowed if coerced is not None else False


def _coerce_status(status: object) -> ExperimentStatus | None:
    if isinstance(status, ExperimentStatus):
        return status
    text = str(status).strip()
    if text.startswith("ExperimentStatus."):
        text = text.split(".", 1)[1]
    normalized = text.upper().rstrip("+*")
    if normalized.startswith("CANCELED") or normalized.startswith("CANCELLED"):
        return ExperimentStatus.CANCELLED
    try:
        return ExperimentStatus(text)
    except ValueError:
        try:
            return ExperimentStatus[normalized]
        except KeyError:
            return None


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


def _coerce_optional_positive_int(value: object, field_name: str) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = int(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive integer or blank") from exc
    if parsed <= 0:
        raise ValueError(f"{field_name} must be a positive integer or blank")
    return parsed


def _coerce_optional_hpc(value: object) -> HPCType | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    hpc_type = _coerce_hpc(text)
    if hpc_type is None:
        raise ValueError(f"unknown hpc: {value!r}")
    return hpc_type


def _set_resource_override(exp: dict[str, Any], key: str, value: int | None, *, aliases: tuple[str, ...] = ()) -> None:
    if value is None:
        _remove_resource_override(exp, key, *aliases)
        return
    overrides = exp.get("resource_overrides")
    if not isinstance(overrides, dict):
        overrides = {}
        exp["resource_overrides"] = overrides
    for alias in aliases:
        overrides.pop(alias, None)
    overrides[key] = value


def _remove_resource_override(exp: dict[str, Any], key: str, *aliases: str) -> None:
    overrides = exp.get("resource_overrides")
    if not isinstance(overrides, dict):
        return
    for item in (key, *aliases):
        overrides.pop(item, None)
    if not overrides:
        exp.pop("resource_overrides", None)


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
    "handle_relaunch_run",
    "handle_update_run_settings",
    "handle_update_global_run_settings",
    "handle_pause_submissions",
    "handle_resume_submissions",
    "handle_set_concurrency_limit",
    "process_command_queue",
    "scancel_via_connection",
]


_RELAUNCHABLE_STATUSES = {
    ExperimentStatus.COMPLETED,
    ExperimentStatus.FAILED,
    ExperimentStatus.CANCELLED,
    ExperimentStatus.TIMEOUT,
    ExperimentStatus.OOM,
    ExperimentStatus.KILLED,
}

_NEXT_SUBMISSION_SETTINGS_STATUSES = {ExperimentStatus.PENDING, ExperimentStatus.PARTIAL}

_RELAUNCH_RESET_FIELDS = {
    "job_id",
    "queued_timestamp",
    "running_timestamp",
    "completed_timestamp",
    "failed_timestamp",
    "cancelled_timestamp",
    "timeout_timestamp",
    "killed_timestamp",
    "output_dir",
    "sacct_snapshot",
    "scheduler_state",
    "slurm_state",
    "history",
    "history_last_read_offset",
    "history_truncated",
    "history_attempt_max",
}

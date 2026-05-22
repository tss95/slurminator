"""Slurm log-tail interpretation helpers for Slurminator orchestration."""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from shlex import quote
from typing import Any, Literal

from slurminator.experiments import ExperimentStatus
from slurminator.config import HPCType
from slurminator.experiment_policy import resolve_resource_overrides
from slurminator.plugins import OrchestratorPlugin
from slurminator.timeout_policy import apply_timeout_policy

logger = logging.getLogger("slurminator")

IsLocalHPC = Callable[[HPCType], bool]
LogSource = Literal["stdout", "stderr", "combined"]


@dataclass
class LogGatheringContext:
    """Dependencies needed to inspect a finished job's Slurm logs."""

    connection_manager: Any
    hpc_configs: Mapping[HPCType, Any]
    plugin: OrchestratorPlugin
    is_local_hpc: IsLocalHPC
    global_time_hours_override: int | None = None
    retry_timeout_with_estimated_time: bool = False
    timeout_retry_buffer: float = 1.3
    timeout_retry_max_attempts: int = 1


@dataclass
class LogTailReadResult:
    """Result of one log-tail read."""

    text: str
    offsets: dict[str, int]
    truncated: bool = False


def gather_logs(exp: dict[str, Any], job_id: str, hpc_type: HPCType, context: LogGatheringContext) -> None:
    """Inspect Slurm logs and adjust terminal status when log evidence is stronger."""
    log_tail = read_log_tail(exp, job_id, hpc_type, context)
    if log_tail is None:
        return

    context.plugin.annotate_log_tail(exp=exp, log_tail=log_tail)

    plugin_status = context.plugin.interpret_log_tail(
        exp=exp, log_tail=log_tail, current_status=exp["status"], stage="pre_heuristics"
    )
    if _apply_plugin_status(exp, plugin_status, source="log pre-heuristic"):
        return

    if exp["status"] in [ExperimentStatus.COMPLETED, ExperimentStatus.FAILED]:
        return

    if exp["status"] == ExperimentStatus.TIMEOUT:
        _apply_timeout_retry(exp, hpc_type, context, reason="slurm-timeout")
        return

    plugin_status = context.plugin.interpret_log_tail(
        exp=exp, log_tail=log_tail, current_status=exp["status"], stage="heuristics"
    )
    if plugin_status == ExperimentStatus.TIMEOUT:
        _apply_timeout_retry(exp, hpc_type, context, reason="log-time-limit")
        return
    if _apply_plugin_status(exp, plugin_status, source="log heuristic"):
        return

    plugin_status = context.plugin.interpret_log_tail(
        exp=exp, log_tail=log_tail, current_status=exp["status"], stage="post_heuristics"
    )
    _apply_plugin_status(exp, plugin_status, source="log post-heuristic")


def _apply_timeout_retry(exp: dict[str, Any], hpc_type: HPCType, context: LogGatheringContext, *, reason: str) -> None:
    """Apply timeout retry policy for a scheduler- or plugin-derived timeout."""
    apply_timeout_policy(
        exp,
        hpc_type,
        reason=reason,
        cluster_config=context.hpc_configs.get(hpc_type),
        resource_overrides=resolve_resource_overrides(exp, hpc_type=hpc_type, cluster_configs=context.hpc_configs),
        global_time_hours_override=context.global_time_hours_override,
        retry_timeout_with_estimated_time=context.retry_timeout_with_estimated_time,
        timeout_retry_buffer=context.timeout_retry_buffer,
        timeout_retry_max_attempts=context.timeout_retry_max_attempts,
    )


def _apply_plugin_status(exp: dict[str, Any], plugin_status: Any | None, *, source: str) -> bool:
    """Apply a plugin-derived status and return True when processing should stop."""
    if not plugin_status or plugin_status == exp["status"]:
        return False

    old_status = exp["status"]
    logger.warning(
        "%s: %s indicates %s - overriding %s.",
        exp["experiment_id"],
        source,
        plugin_status.name if hasattr(plugin_status, "name") else plugin_status,
        old_status.name if hasattr(old_status, "name") else old_status,
    )
    exp["status"] = plugin_status
    return True


def read_log_tail(exp: dict[str, Any], job_id: str, hpc_type: HPCType, context: LogGatheringContext) -> str | None:
    """Return the combined stdout/stderr log tail for a Slurm job."""
    out_dir = exp.get("output_dir")
    if not out_dir:
        logger.debug("No output_dir recorded for %s - skipping log parse.", exp["experiment_id"])
        return None

    log_path_out = Path(out_dir) / f"slurm-{job_id}.out"
    log_path_err = Path(out_dir) / f"slurm-{job_id}.err"
    tail_cmd = (
        f"(tail -n 200 {quote(str(log_path_out))} 2>/dev/null; "
        f"tail -n 200 {quote(str(log_path_err))} 2>/dev/null) || true"
    )

    try:
        if context.is_local_hpc(hpc_type):
            result = subprocess.run(tail_cmd, shell=True, capture_output=True, text=True)
            return result.stdout
        log_tail, _ = context.connection_manager.run_command(hpc_type, tail_cmd)
        return log_tail
    except Exception as exc:
        logger.debug("Could not read log tail for %s: %s", exp["experiment_id"], exc)
        return None


def read_log_tail_incremental(
    exp: dict[str, Any],
    job_id: str,
    hpc_type: HPCType,
    context: LogGatheringContext,
    *,
    lines: int = 500,
    offsets: Mapping[str, int] | None = None,
    source: LogSource = "combined",
) -> LogTailReadResult:
    """Read recent or newly-appended Slurm stdout/stderr log text."""
    out_dir = exp.get("output_dir")
    if not out_dir:
        return LogTailReadResult(text="", offsets={})

    previous_offsets = dict(offsets or {})
    paths = _log_paths(out_dir, job_id, source=source)
    new_offsets: dict[str, int] = {}
    chunks: list[str] = []
    truncated = False
    include_headers = source == "combined"

    for label, path in paths.items():
        previous_offset = max(int(previous_offsets.get(label, 0) or 0), 0)
        size = _log_file_size(path, hpc_type, context)
        new_offsets[label] = size
        if size <= 0:
            continue

        if previous_offset <= 0:
            text = _tail_log_lines(path, lines, hpc_type, context)
        elif size < previous_offset:
            truncated = True
            text = _tail_log_lines(path, lines, hpc_type, context)
        elif size > previous_offset:
            text = _tail_log_bytes(path, previous_offset, hpc_type, context)
        else:
            text = ""

        if text:
            body = text.rstrip()
            if include_headers:
                chunks.append(f"===== {label}: {path} =====\n{body}\n")
            else:
                chunks.append(body)

    return LogTailReadResult(text="\n".join(chunks).strip(), offsets=new_offsets, truncated=truncated)


def _log_paths(out_dir: object, job_id: str, *, source: LogSource) -> dict[str, Path]:
    all_paths = {
        "stdout": Path(str(out_dir)) / f"slurm-{job_id}.out",
        "stderr": Path(str(out_dir)) / f"slurm-{job_id}.err",
    }
    if source == "stdout":
        return {"stdout": all_paths["stdout"]}
    if source == "stderr":
        return {"stderr": all_paths["stderr"]}
    return all_paths


def _log_file_size(path: Path, hpc_type: HPCType, context: LogGatheringContext) -> int:
    out = _run_log_command(f'stat -c "%s" {quote(str(path))} 2>/dev/null || echo 0', hpc_type, context)
    try:
        return max(int(str(out).strip().splitlines()[-1]), 0)
    except Exception:
        return 0


def _tail_log_lines(path: Path, lines: int, hpc_type: HPCType, context: LogGatheringContext) -> str:
    safe_lines = max(int(lines), 1)
    return _run_log_command(f"tail -n {safe_lines} {quote(str(path))} 2>/dev/null || true", hpc_type, context)


def _tail_log_bytes(path: Path, previous_offset: int, hpc_type: HPCType, context: LogGatheringContext) -> str:
    return _run_log_command(f"tail -c +{previous_offset + 1} {quote(str(path))} 2>/dev/null || true", hpc_type, context)


def _run_log_command(command: str, hpc_type: HPCType, context: LogGatheringContext) -> str:
    if context.is_local_hpc(hpc_type):
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.stdout
    out, _ = context.connection_manager.run_command(hpc_type, command)
    return out


__all__ = [
    "LogGatheringContext",
    "LogSource",
    "LogTailReadResult",
    "gather_logs",
    "read_log_tail",
    "read_log_tail_incremental",
]

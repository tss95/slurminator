"""Timeout retry and submission-resource helpers for Slurminator orchestration."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from typing import Any

from slurminator.experiments import ExperimentStatus
from slurminator.config import HPCType

logger = logging.getLogger("slurminator")


@dataclass(frozen=True)
class SubmissionResources:
    """Concrete resources resolved for one Slurm submission."""

    time_hours: int
    memory_gb: int
    cpus: int
    gpu_count: int
    mem_per_gpu_gb: int | None = None

    @property
    def requested_ram_gb(self) -> int:
        """Return total requested RAM in GB for experiment-state bookkeeping."""
        if self.mem_per_gpu_gb is not None:
            return int(self.mem_per_gpu_gb * self.gpu_count)
        return int(self.memory_gb)


def apply_timeout_retry_to_resources(
    exp: dict[str, Any],
    base_resources: SubmissionResources,
    *,
    global_time_hours_override: int | None = None,
    global_memory_gb_override: int | None = None,
) -> SubmissionResources:
    """Return resources adjusted by global and per-experiment retry overrides.

    This is pure: it does not mutate ``exp``. Timeout retry writes
    ``exp["time_hours_override"]`` when a timed-out run should be retried;
    submission consumes that field here when building the next sbatch request.
    """
    resources = base_resources

    if global_time_hours_override is not None:
        resources = replace(resources, time_hours=int(global_time_hours_override))

    exp_time_override = exp.get("time_hours_override")
    if exp_time_override is not None:
        try:
            exp_time_hours = int(exp_time_override)
        except (TypeError, ValueError):
            logger.warning(
                "%s: invalid per-experiment time_hours_override=%r; ignoring.",
                exp.get("experiment_id", "<unknown>"),
                exp_time_override,
            )
        else:
            if exp_time_hours > 0:
                resources = replace(resources, time_hours=exp_time_hours)
            else:
                logger.warning(
                    "%s: non-positive per-experiment time_hours_override=%r; ignoring.",
                    exp.get("experiment_id", "<unknown>"),
                    exp_time_override,
                )

    if global_memory_gb_override is not None:
        resources = replace(resources, memory_gb=int(global_memory_gb_override), mem_per_gpu_gb=None)

    return resources


def resolve_progress_fraction(exp: dict[str, Any]) -> float | None:
    """Return best-known completion fraction in ``(0, 1]``, or ``None`` when unavailable."""
    candidate_pairs = [
        ("current_step", "max_steps"),
        ("current_pseudo_epoch", "max_pseudo_epochs"),
        ("current_epoch", "max_epochs"),
    ]
    fractions: list[float] = []
    for current_key, max_key in candidate_pairs:
        current_val = exp.get(current_key)
        max_val = exp.get(max_key)
        if current_val is None or max_val is None:
            continue
        try:
            current_f = float(current_val)
            max_f = float(max_val)
        except (TypeError, ValueError):
            continue
        if max_f <= 0.0:
            continue
        frac = current_f / max_f
        if frac > 0.0:
            fractions.append(min(frac, 1.0))
    if not fractions:
        return None
    return max(fractions)


def resolve_requested_time_hours(
    exp: dict[str, Any],
    hpc_type: HPCType,
    *,
    cluster_config: Any | None,
    resource_overrides: dict[str, Any],
    global_time_hours_override: int | None = None,
) -> int:
    """Best-effort resolution of walltime hours used by the latest submission."""
    tracked = exp.get("requested_time_hours")
    try:
        tracked_int = int(tracked)
        if tracked_int > 0:
            return tracked_int
    except (TypeError, ValueError):
        pass

    exp_override = exp.get("time_hours_override")
    try:
        exp_override_int = int(exp_override)
        if exp_override_int > 0:
            return exp_override_int
    except (TypeError, ValueError):
        pass

    if global_time_hours_override is not None and global_time_hours_override > 0:
        return int(global_time_hours_override)

    fallback_hours = int(getattr(cluster_config, "base_time_hours", 1) or 1)
    dataset_time = resource_overrides.get("time_hours")
    try:
        dataset_time_int = int(dataset_time)
        if dataset_time_int > 0:
            return dataset_time_int
    except (TypeError, ValueError):
        pass
    return fallback_hours


def estimate_timeout_retry_hours(
    exp: dict[str, Any],
    hpc_type: HPCType,
    *,
    cluster_config: Any | None,
    resource_overrides: dict[str, Any],
    global_time_hours_override: int | None,
    timeout_retry_buffer: float,
) -> int | None:
    """Estimate required walltime after timeout using observed progress and configured buffer."""
    progress = resolve_progress_fraction(exp)
    if progress is None or progress <= 0.0:
        return None

    current_hours = resolve_requested_time_hours(
        exp,
        hpc_type,
        cluster_config=cluster_config,
        resource_overrides=resource_overrides,
        global_time_hours_override=global_time_hours_override,
    )
    estimated_hours = math.ceil((current_hours / progress) * timeout_retry_buffer)

    max_time_hours = int(getattr(cluster_config, "max_time_hours", 0) or 0)
    if max_time_hours > 0 and estimated_hours > max_time_hours:
        logger.warning(
            "%s: timeout retry estimate %sh exceeds cluster max (%sh) on %s; clamping.",
            exp.get("experiment_id", "<unknown>"),
            estimated_hours,
            max_time_hours,
            hpc_type.name,
        )
        estimated_hours = max_time_hours

    if estimated_hours <= current_hours:
        return None
    return estimated_hours


def apply_timeout_policy(
    exp: dict[str, Any],
    hpc_type: HPCType,
    *,
    reason: str,
    cluster_config: Any | None,
    resource_overrides: dict[str, Any],
    global_time_hours_override: int | None,
    retry_timeout_with_estimated_time: bool,
    timeout_retry_buffer: float,
    timeout_retry_max_attempts: int,
) -> None:
    """Apply timeout terminal-state policy for a finished experiment."""
    exp_id = exp.get("experiment_id", "<unknown>")
    if not retry_timeout_with_estimated_time:
        exp["status"] = ExperimentStatus.TIMEOUT
        return

    current_retry_count = int(exp.get("timeout_retry_count", 0) or 0)
    if current_retry_count >= timeout_retry_max_attempts:
        logger.warning(
            "%s: timeout detected (%s), retry cap reached (%s). Keeping TIMEOUT.",
            exp_id,
            reason,
            timeout_retry_max_attempts,
        )
        exp["status"] = ExperimentStatus.TIMEOUT
        return

    estimated_hours = estimate_timeout_retry_hours(
        exp,
        hpc_type,
        cluster_config=cluster_config,
        resource_overrides=resource_overrides,
        global_time_hours_override=global_time_hours_override,
        timeout_retry_buffer=timeout_retry_buffer,
    )
    if estimated_hours is None:
        logger.warning(
            "%s: timeout detected (%s), but progress-based walltime estimate was unavailable or non-increasing. Keeping TIMEOUT.",
            exp_id,
            reason,
        )
        exp["status"] = ExperimentStatus.TIMEOUT
        return

    previous_hours = resolve_requested_time_hours(
        exp,
        hpc_type,
        cluster_config=cluster_config,
        resource_overrides=resource_overrides,
        global_time_hours_override=global_time_hours_override,
    )
    exp["time_hours_override"] = estimated_hours
    exp["timeout_retry_count"] = current_retry_count + 1
    exp["status"] = ExperimentStatus.PARTIAL
    logger.warning(
        "%s: timeout detected (%s). Scheduling retry %s/%s with estimated walltime %sh (previous %sh, buffer=%.2f).",
        exp_id,
        reason,
        exp["timeout_retry_count"],
        timeout_retry_max_attempts,
        estimated_hours,
        previous_hours,
        timeout_retry_buffer,
    )


__all__ = [
    "SubmissionResources",
    "apply_timeout_policy",
    "apply_timeout_retry_to_resources",
    "estimate_timeout_retry_hours",
    "resolve_progress_fraction",
    "resolve_requested_time_hours",
]

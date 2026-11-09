"""Scheduler polling and scheduler-state transitions for Slurminator orchestration."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from slurminator.config import HPCType
from slurminator.experiments import ExperimentStatus
from slurminator.hpc_state import expand_slurm_state, is_terminal_status, map_scheduler_state_to_experiment_status

logger = logging.getLogger("slurminator")

GatherLogs = Callable[[dict[str, Any], str, HPCType], None]
PollStage = Literal["squeue", "sacct"]


@dataclass(frozen=True, slots=True)
class SchedulerPollFailure:
    """One failed scheduler query within an HPC poll."""

    hpc_type: HPCType
    stage: PollStage
    job_ids: tuple[str, ...]
    error_type: str
    message: str


@dataclass(frozen=True, slots=True)
class HPCPollResult:
    """Scheduler states and verification metadata for one HPC."""

    hpc_type: HPCType
    requested_job_ids: tuple[str, ...]
    states: Mapping[str, str]
    failure: SchedulerPollFailure | None = None

    @property
    def succeeded(self) -> bool:
        """Return whether every scheduler command completed and parsed successfully."""
        return self.failure is None

    @property
    def unresolved_job_ids(self) -> tuple[str, ...]:
        """Return requested jobs absent from both scheduler command results."""
        return tuple(job_id for job_id in self.requested_job_ids if job_id not in self.states)


@dataclass(frozen=True, slots=True)
class SchedulerTransition:
    """One observed scheduler-owned experiment status transition."""

    experiment_id: str
    job_id: str
    hpc_type: HPCType
    old_status: ExperimentStatus
    new_status: ExperimentStatus
    timestamp: float
    timestamp_field: str | None

    @property
    def running_timestamp(self) -> float | None:
        """Return the observed start time for a queued-to-running transition."""
        if self.timestamp_field == "running_timestamp":
            return self.timestamp
        return None

    @property
    def terminal_timestamp(self) -> float | None:
        """Return the observed terminal time for a transition into a terminal state."""
        if is_terminal_status(self.new_status):
            return self.timestamp
        return None

    def row_patch(self) -> dict[str, Any]:
        """Return the scheduler-owned fields applied to the experiment row."""
        patch: dict[str, Any] = {"status": self.new_status, "last_change_ts": self.timestamp}
        if self.timestamp_field is not None:
            patch[self.timestamp_field] = self.timestamp
        return patch


@dataclass(frozen=True, slots=True)
class SchedulerPollSkip:
    """One active experiment that could not be included in scheduler verification."""

    experiment_id: str
    job_id: str | None
    hpc_type: HPCType | None
    reason: str


@dataclass(frozen=True, slots=True)
class SchedulerStatusUpdateResult:
    """Outcome of one scheduler lifecycle refresh across all configured HPCs."""

    transitions: tuple[SchedulerTransition, ...]
    hpc_polls: tuple[HPCPollResult, ...]
    skipped: tuple[SchedulerPollSkip, ...] = ()

    @property
    def failures(self) -> tuple[SchedulerPollFailure, ...]:
        """Return all failed scheduler queries."""
        return tuple(poll.failure for poll in self.hpc_polls if poll.failure is not None)

    @property
    def succeeded(self) -> bool:
        """Return whether every active experiment was submitted to a successful poll."""
        return not self.failures and not self.skipped and all(not poll.unresolved_job_ids for poll in self.hpc_polls)


_STATUS_TIMESTAMP_FIELDS: Mapping[ExperimentStatus, str] = {
    ExperimentStatus.RUNNING: "running_timestamp",
    ExperimentStatus.COMPLETED: "completed_timestamp",
    ExperimentStatus.FAILED: "failed_timestamp",
    ExperimentStatus.CANCELLED: "cancelled_timestamp",
    ExperimentStatus.TIMEOUT: "timeout_timestamp",
    # OOM is a failure subtype in the existing experiment-row timestamp convention.
    ExperimentStatus.OOM: "failed_timestamp",
    ExperimentStatus.KILLED: "killed_timestamp",
}


def poll_hpc_result(connection_manager: Any, hpc_type: HPCType, jobids: list[str]) -> HPCPollResult:
    """Poll one HPC and return states plus an explicit partial/failure result."""
    requested_job_ids = tuple(jobids)
    if not jobids:
        return HPCPollResult(hpc_type=hpc_type, requested_job_ids=(), states={})
    joined = ",".join(jobids)

    sq_cmd = f"squeue -h -o '%i %t' -j {joined}"
    try:
        out, _err = connection_manager.run_command(hpc_type, sq_cmd, prefer_remote=True)
        state_map = _parse_squeue_states(out)
    except Exception as exc:
        failure = _poll_failure(hpc_type, "squeue", requested_job_ids, exc)
        _log_poll_failure(failure)
        return HPCPollResult(hpc_type, requested_job_ids, {}, failure)

    missing = [job_id for job_id in jobids if job_id not in state_map]
    if missing:
        sac_cmd = f"sacct -n -o 'JobID,State' -j {','.join(missing)}"
        try:
            out2, _err2 = connection_manager.run_command(hpc_type, sac_cmd, prefer_remote=True)
            state_map.update(_parse_sacct_states(out2, expected_job_ids=set(missing)))
        except Exception as exc:
            failure = _poll_failure(hpc_type, "sacct", tuple(missing), exc)
            _log_poll_failure(failure)
            return HPCPollResult(hpc_type, requested_job_ids, state_map, failure)

    return HPCPollResult(hpc_type=hpc_type, requested_job_ids=requested_job_ids, states=state_map)


def poll_hpc(connection_manager: Any, hpc_type: HPCType, jobids: list[str]) -> dict[str, str]:
    """Return scheduler states for compatibility; use :func:`poll_hpc_result` for verification."""
    return dict(poll_hpc_result(connection_manager, hpc_type, jobids).states)


def update_scheduler_statuses(
    experiments: list[dict[str, Any]],
    *,
    connection_manager: Any,
    concurrency_limits: Mapping[HPCType, int] | None = None,
    gather_logs: GatherLogs | None = None,
) -> SchedulerStatusUpdateResult:
    """Poll active jobs, apply scheduler lifecycle fields, and return explicit transitions.

    ``concurrency_limits`` and ``gather_logs`` remain accepted while callers migrate from
    the previous interface. Active jobs remain pollable when a submission limit is zero,
    and terminal log I/O is deliberately not performed in this lifecycle-only function.
    """
    del concurrency_limits, gather_logs

    configured_hpcs = getattr(connection_manager, "configs", None)
    pollmap: dict[HPCType, list[tuple[dict[str, Any], str]]] = {}
    skipped: list[SchedulerPollSkip] = []
    for exp in experiments:
        status = exp.get("status")
        if status not in [ExperimentStatus.QUEUED, ExperimentStatus.RUNNING]:
            continue
        experiment_id = str(exp.get("experiment_id") or "")
        raw_job_id = exp.get("job_id")
        hpc = exp.get("hpc_assignment")
        job_id = str(raw_job_id) if raw_job_id else None
        if job_id is None:
            skipped.append(SchedulerPollSkip(experiment_id, None, hpc, "missing job_id"))
            continue
        if hpc is None:
            skipped.append(SchedulerPollSkip(experiment_id, job_id, None, "missing hpc_assignment"))
            continue
        if isinstance(configured_hpcs, Mapping) and hpc not in configured_hpcs:
            skipped.append(SchedulerPollSkip(experiment_id, job_id, hpc, "HPC is not configured"))
            continue
        pollmap.setdefault(hpc, []).append((exp, job_id))

    transitions: list[SchedulerTransition] = []
    hpc_polls: list[HPCPollResult] = []
    for hpc_type, exp_job_pairs in pollmap.items():
        jobids = [job_id for (_exp, job_id) in exp_job_pairs]
        poll_result = poll_hpc_result(connection_manager, hpc_type, jobids)
        hpc_polls.append(poll_result)

        for exp, job_id in exp_job_pairs:
            old_status = exp["status"]
            new_state = poll_result.states.get(job_id)
            if not new_state:
                logger.debug("%s not found => might be completed or unknown.", job_id)
                continue

            new_status = map_state(new_state)
            if new_status == old_status:
                continue

            timestamp = time.time()
            timestamp_field = _STATUS_TIMESTAMP_FIELDS.get(new_status)
            transition = SchedulerTransition(
                experiment_id=str(exp["experiment_id"]),
                job_id=job_id,
                hpc_type=hpc_type,
                old_status=old_status,
                new_status=new_status,
                timestamp=timestamp,
                timestamp_field=timestamp_field,
            )
            logger.info("%s changes %s => %s", transition.experiment_id, old_status, new_status)
            exp.update(transition.row_patch())
            transitions.append(transition)

    return SchedulerStatusUpdateResult(tuple(transitions), tuple(hpc_polls), tuple(skipped))


def _parse_squeue_states(output: str) -> dict[str, str]:
    """Parse ``squeue`` job/state output, failing on ambiguous rows."""
    states: dict[str, str] = {}
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        columns = stripped.split()
        if len(columns) != 2:
            raise ValueError(f"Malformed squeue row: {stripped!r}")
        job_id, short_state = columns
        states[job_id] = expand_short(short_state)
    return states


def _parse_sacct_states(output: str, *, expected_job_ids: set[str]) -> dict[str, str]:
    """Parse top-level ``sacct`` job states for the requested missing jobs."""
    states: dict[str, str] = {}
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        columns = stripped.split()
        if len(columns) < 2:
            raise ValueError(f"Malformed sacct row: {stripped!r}")
        job_col, state_col = columns[:2]
        if "." in job_col:
            continue
        base_job = job_col.split(".")[0]
        if base_job in expected_job_ids:
            states[base_job] = expand_short(state_col)
    return states


def _poll_failure(
    hpc_type: HPCType, stage: PollStage, job_ids: tuple[str, ...], error: Exception
) -> SchedulerPollFailure:
    """Build a serializable scheduler query failure."""
    return SchedulerPollFailure(
        hpc_type=hpc_type, stage=stage, job_ids=job_ids, error_type=type(error).__name__, message=str(error)
    )


def _log_poll_failure(failure: SchedulerPollFailure) -> None:
    """Log a scheduler query failure with its explicit stage."""
    logger.warning(
        "Polling via %s failed on %s: %s. Will retry next cycle.", failure.stage, failure.hpc_type.name, failure.message
    )


def expand_short(code: str) -> str:
    """Convert short Slurm code or sacct state to a standard scheduler state."""
    return expand_slurm_state(code)


def map_state(state: str) -> ExperimentStatus:
    """Map scheduler textual state to an ``ExperimentStatus`` enum."""
    return map_scheduler_state_to_experiment_status(state)


__all__ = [
    "HPCPollResult",
    "SchedulerPollFailure",
    "SchedulerPollSkip",
    "SchedulerStatusUpdateResult",
    "SchedulerTransition",
    "expand_short",
    "map_state",
    "poll_hpc",
    "poll_hpc_result",
    "update_scheduler_statuses",
]

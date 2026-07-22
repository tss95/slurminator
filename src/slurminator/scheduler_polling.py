"""Scheduler polling and scheduler-state transitions for Slurminator orchestration."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from typing import Any

from slurminator.experiments import ExperimentStatus
from slurminator.config import HPCType
from slurminator.hpc_state import expand_slurm_state, is_terminal_status, map_scheduler_state_to_experiment_status

logger = logging.getLogger("slurminator")

GatherLogs = Callable[[dict[str, Any], str, HPCType], None]


def poll_hpc(connection_manager: Any, hpc_type: HPCType, jobids: list[str]) -> dict[str, str]:
    """Return ``{job_id: scheduler_state}`` from ``squeue`` plus ``sacct`` fallback."""
    if not jobids:
        return {}
    joined = ",".join(jobids)

    sq_cmd = f"squeue -h -o '%i %t' -j {joined}"
    try:
        out, _err = connection_manager.run_command(hpc_type, sq_cmd, prefer_remote=True)
    except Exception as exc:
        logger.warning("Polling via squeue failed on %s: %s. Will retry next cycle.", hpc_type.name, exc)
        return {}

    state_map: dict[str, str] = {}
    for line in out.splitlines():
        line = line.strip()
        if line:
            job_id, short_state = line.split()
            state_map[job_id] = expand_short(short_state)

    missing = [job_id for job_id in jobids if job_id not in state_map]
    if missing:
        missing_set = set(missing)
        sac_cmd = f"sacct -n -o 'JobID,State' -j {','.join(missing)}"
        try:
            out2, _err2 = connection_manager.run_command(hpc_type, sac_cmd, prefer_remote=True)
        except Exception as exc:
            logger.warning("Polling via sacct failed on %s: %s. Will retry next cycle.", hpc_type.name, exc)
            return state_map
        for line in out2.splitlines():
            line = line.strip()
            if not line:
                continue
            job_col, state_col = line.split()[:2]
            if "." in job_col:
                continue
            base_job = job_col.split(".")[0]
            if base_job not in missing_set:
                continue
            state_map[base_job] = expand_short(state_col)

    return state_map


def update_scheduler_statuses(
    experiments: list[dict[str, Any]],
    *,
    connection_manager: Any,
    concurrency_limits: Mapping[HPCType, int],
    gather_logs: GatherLogs,
) -> None:
    """Poll queued/running experiments and update scheduler-owned status fields."""
    configured_hpcs = getattr(connection_manager, "configs", None)
    pollmap: dict[HPCType, list[tuple[dict[str, Any], str]]] = {}
    for exp in experiments:
        status = exp.get("status")
        if status not in [ExperimentStatus.QUEUED, ExperimentStatus.RUNNING]:
            continue
        job_id = exp.get("job_id")
        hpc = exp.get("hpc_assignment")
        hpc_is_pollable = not isinstance(configured_hpcs, Mapping) or hpc in configured_hpcs
        if job_id and hpc and hpc_is_pollable:
            pollmap.setdefault(hpc, []).append((exp, job_id))

    for hpc_type, exp_job_pairs in pollmap.items():
        jobids = [job_id for (_exp, job_id) in exp_job_pairs]
        state_map = poll_hpc(connection_manager, hpc_type, jobids)

        for exp, job_id in exp_job_pairs:
            old_status = exp["status"]
            new_state = state_map.get(job_id)
            if not new_state:
                logger.debug("%s not found => might be completed or unknown.", job_id)
                continue

            new_status = map_state(new_state)
            if new_status == old_status:
                continue

            logger.info("%s changes %s => %s", exp["experiment_id"], old_status, new_status)
            exp["status"] = new_status
            exp["last_change_ts"] = time.time()
            if is_terminal_status(new_status):
                gather_logs(exp, job_id, hpc_type)


def expand_short(code: str) -> str:
    """Convert short Slurm code or sacct state to a standard scheduler state."""
    return expand_slurm_state(code)


def map_state(state: str) -> ExperimentStatus:
    """Map scheduler textual state to an ``ExperimentStatus`` enum."""
    return map_scheduler_state_to_experiment_status(state)


__all__ = ["expand_short", "map_state", "poll_hpc", "update_scheduler_statuses"]

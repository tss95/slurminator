"""HPC scheduler state normalization helpers."""

from __future__ import annotations

from slurminator.experiments import ExperimentStatus


def expand_slurm_state(code: str) -> str:
    """Return a normalized Slurm state string from short or long scheduler output."""
    normalized = str(code).upper().rstrip("+*")
    shortmap = {
        "PD": "PENDING",
        "CF": "CONFIGURING",
        "R": "RUNNING",
        "CG": "COMPLETING",
        "CD": "COMPLETED",
        "CA": "CANCELLED",
        "F": "FAILED",
        "TO": "TIMEOUT",
        "OOM": "OUT_OF_MEMORY",
        "NF": "NODE_FAIL",
        "PR": "PREEMPTED",
    }
    return shortmap.get(normalized, normalized)


def map_scheduler_state_to_experiment_status(state: str) -> ExperimentStatus:
    """Map a normalized scheduler state to an orchestrator experiment status."""
    normalized = expand_slurm_state(state)
    mapping = {
        "PENDING": ExperimentStatus.QUEUED,
        "CONFIGURING": ExperimentStatus.QUEUED,
        "RUNNING": ExperimentStatus.RUNNING,
        "COMPLETING": ExperimentStatus.RUNNING,
        "COMPLETED": ExperimentStatus.COMPLETED,
        "CANCELLED": ExperimentStatus.CANCELLED,
        "TIMEOUT": ExperimentStatus.TIMEOUT,
        "OUT_OF_MEMORY": ExperimentStatus.OOM,
        "FAILED": ExperimentStatus.FAILED,
        "NODE_FAIL": ExperimentStatus.FAILED,
        "PREEMPTED": ExperimentStatus.FAILED,
    }
    return mapping.get(normalized, ExperimentStatus.FAILED)


def is_terminal_status(status: ExperimentStatus | str) -> bool:
    """Return whether an experiment status is terminal."""
    if not isinstance(status, ExperimentStatus):
        try:
            status = ExperimentStatus(str(status))
        except ValueError:
            try:
                status = ExperimentStatus[str(status).upper()]
            except KeyError:
                return False
    return status in {
        ExperimentStatus.COMPLETED,
        ExperimentStatus.FAILED,
        ExperimentStatus.CANCELLED,
        ExperimentStatus.TIMEOUT,
        ExperimentStatus.OOM,
        ExperimentStatus.KILLED,
    }


__all__ = ["expand_slurm_state", "is_terminal_status", "map_scheduler_state_to_experiment_status"]

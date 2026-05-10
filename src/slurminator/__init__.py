"""Reusable SLURM/HPC experiment orchestration package."""

from typing import Any

from slurminator.hpc_state import expand_slurm_state, is_terminal_status, map_scheduler_state_to_experiment_status
from slurminator.experiment_policy import (
    resolve_extra_remote_dirs,
    resolve_pinned_hpc,
    resolve_resource_overrides,
    resolve_sbatch_export_vars,
)

__all__ = [
    "expand_slurm_state",
    "is_terminal_status",
    "map_scheduler_state_to_experiment_status",
    "project_status_to_experiment",
    "resolve_extra_remote_dirs",
    "resolve_pinned_hpc",
    "resolve_resource_overrides",
    "resolve_sbatch_export_vars",
    "status_projection_fields",
]


def __getattr__(name: str) -> Any:
    """Load pydantic-backed status helpers only when they are requested."""
    if name == "project_status_to_experiment":
        from slurminator.status_projection import project_status_to_experiment

        return project_status_to_experiment
    if name == "status_projection_fields":
        from slurminator.status_projection import status_projection_fields

        return status_projection_fields
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

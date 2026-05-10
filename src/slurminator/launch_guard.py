"""Launch guards for orchestrator entry points."""

from __future__ import annotations

import os
import shutil
import socket

from slurminator.config import HPCType, determine_current_hpc


def _is_cuda_available() -> bool:
    """Best-effort GPU-node check without importing training frameworks."""
    return shutil.which("nvidia-smi") is not None or shutil.which("rocm-smi") is not None


def get_orchestrator_gpu_hpc_launch_block_message(
    *, current_hpc: HPCType | None = None, cuda_available: bool | None = None
) -> str | None:
    """Return a fail-fast message if an orchestrator is launched from a GPU-capable HPC runtime."""
    hpc = determine_current_hpc() if current_hpc is None else current_hpc
    if hpc is None:
        return None

    has_cuda = _is_cuda_available() if cuda_available is None else bool(cuda_available)
    if not has_cuda:
        return None

    hostname = socket.gethostname()
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    slurm_hint = f", SLURM_JOB_ID={slurm_job_id}" if slurm_job_id else ""
    return (
        "Refusing to launch orchestrator from a GPU-capable HPC runtime: "
        f"hpc={hpc.name}, host={hostname}{slurm_hint}. "
        "run_orchestrator should be launched from a login/control node, not from an allocated GPU node."
    )


__all__ = ["get_orchestrator_gpu_hpc_launch_block_message"]

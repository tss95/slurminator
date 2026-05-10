"""Best-effort detection of the current HPC cluster."""

from __future__ import annotations

import os
import socket

from slurminator.config.cluster_registry import HPCType, coerce_hpc_type


def determine_current_hpc() -> HPCType | None:
    """Determine which known HPC system this process is running on."""
    cluster_env = os.environ.get("CLUSTER", "").strip().lower()
    if cluster_env in {"fox", "lumi", "saga", "olivia"}:
        return coerce_hpc_type(cluster_env)

    hostname = socket.gethostname().lower()
    try:
        fqdn = socket.getfqdn().lower()
    except Exception:
        fqdn = hostname

    parts = set(hostname.split(".")) | set(fqdn.split("."))

    if any(part.startswith("fox") for part in parts) or "fox.educloud.no" in fqdn:
        return HPCType.FOX

    if (
        any(part.startswith("lumi") for part in parts)
        or "lumi.csc.fi" in fqdn
        or hostname.startswith("nid")
        or os.environ.get("LUMI_STACK_NAME")
        or os.environ.get("SLURM_CLUSTER_NAME", "").lower().startswith("lumi")
    ):
        return HPCType.LUMI

    slurm_cluster = os.environ.get("SLURM_CLUSTER_NAME", "").lower()
    if slurm_cluster.startswith("olivia"):
        return HPCType.OLIVIA
    if slurm_cluster.startswith("saga"):
        return HPCType.SAGA
    if "saga" in fqdn or "saga" in hostname:
        return HPCType.SAGA

    if "olivia" in fqdn or "olivia" in hostname or fqdn.endswith(".cm.americas.sgi.com"):
        return HPCType.OLIVIA

    return None


def is_current_hpc(hpc_type: HPCType) -> bool:
    """Return True if this process appears to be running on ``hpc_type``."""
    return determine_current_hpc() == hpc_type


__all__ = ["determine_current_hpc", "is_current_hpc"]

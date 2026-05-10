"""Experiment lifecycle states used by slurminator."""

from __future__ import annotations

from enum import Enum


class ExperimentStatus(Enum):
    """Experiment execution status owned by the orchestrator."""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"
    TIMEOUT = "timeout"
    OOM = "out_of_memory"
    CANCELLED = "cancelled"
    KILLED = "killed"


__all__ = ["ExperimentStatus"]

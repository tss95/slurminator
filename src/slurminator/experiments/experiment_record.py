"""Experiment record dataclasses for slurminator."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from slurminator.experiments.status_enum import ExperimentStatus


@dataclass
class ResourceRequirements:
    """Concrete resources requested for one experiment."""

    memory_gb: int
    time_hours: int
    gpu_count: int
    gpu_memory_gb: int | None = None
    cpus: int = 4
    priority: int = 1


@dataclass
class ExperimentMetadata:
    """Metadata for tracking experiment status and history."""

    creation_time: datetime = field(default_factory=datetime.now)
    last_update: datetime = field(default_factory=datetime.now)
    status: ExperimentStatus = ExperimentStatus.PENDING
    priority: int = 1
    retry_count: int = 0
    max_retries: int = 3
    last_checkpoint: str | None = None
    resume_from: str | None = None
    error_history: list[dict[str, str]] = field(default_factory=list)
    comments: str | None = None

    def __post_init__(self) -> None:
        self.status = coerce_experiment_status(self.status)

    def add_error(self, error_msg: str, error_type: str) -> None:
        """Add an error entry with a timestamp."""
        self.error_history.append(
            {"timestamp": datetime.now().isoformat(), "error_type": error_type, "message": error_msg}
        )

    def update_status(self, new_status: ExperimentStatus | str) -> None:
        """Update status and last-update timestamp."""
        self.status = coerce_experiment_status(new_status)
        self.last_update = datetime.now()


@dataclass
class CheckpointInfo:
    """Information about an experiment checkpoint."""

    path: Path
    epoch: int
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    """Complete orchestrator experiment record.

    ``task_type`` is intentionally an opaque string. Project-specific task
    taxonomy belongs in adapters, not in the package.
    """

    task_type: str
    dataset_name: str
    hpc_params: Any | None = None
    experiment_id: str | None = None
    status: ExperimentStatus = ExperimentStatus.PENDING
    sweep_params: str | None = None
    extra_command: str | None = None
    metadata: ExperimentMetadata | Mapping[str, Any] = field(default_factory=ExperimentMetadata)
    resource_requirements: ResourceRequirements | None = None
    checkpoints: list[CheckpointInfo] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.task_type = coerce_task_type_string(self.task_type)
        self.status = coerce_experiment_status(self.status)

        if self.resource_requirements is None and self.hpc_params is not None:
            self.resource_requirements = _resource_requirements_from_hpc_params(self.hpc_params)

    def get_latest_checkpoint(self) -> CheckpointInfo | None:
        """Return the most recent checkpoint by timestamp."""
        if not self.checkpoints:
            return None
        return max(self.checkpoints, key=lambda checkpoint: checkpoint.timestamp)

    def add_checkpoint(self, path: str | Path, epoch: int, metadata: dict[str, Any] | None = None) -> None:
        """Add a new checkpoint entry."""
        self.checkpoints.append(
            CheckpointInfo(path=Path(path), epoch=epoch, timestamp=datetime.now(), metadata=metadata or {})
        )


@dataclass
class ExperimentGroup:
    """Group of related experiments."""

    group_id: str
    description: str
    experiments: list[ExperimentConfig]
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_experiment_count(self, status: ExperimentStatus | str | None = None) -> int:
        """Return experiment count, optionally filtered by status."""
        if status is None:
            return len(self.experiments)
        target = coerce_experiment_status(status)
        return len([exp for exp in self.experiments if exp.status == target])


def coerce_task_type_string(value: object) -> str:
    """Return an opaque task-type string from strings or enum-like values."""
    if isinstance(value, Enum):
        return str(value.value)
    text = str(value or "").strip()
    enum_name, sep, member_name = text.partition(".")
    if sep and enum_name.endswith("Type") and member_name:
        return member_name.lower()
    return text or "self_supervised"


def coerce_experiment_status(value: ExperimentStatus | str) -> ExperimentStatus:
    """Coerce status strings or enums into ``ExperimentStatus``."""
    if isinstance(value, ExperimentStatus):
        return value
    text = str(value).strip()
    if text.startswith("ExperimentStatus."):
        text = text.split(".", 1)[1]
    try:
        return ExperimentStatus(text)
    except ValueError:
        return ExperimentStatus[text.upper()]


def _resource_requirements_from_hpc_params(hpc_params: object) -> ResourceRequirements:
    get_cluster_config = getattr(hpc_params, "get_cluster_config", None)
    if not callable(get_cluster_config):
        raise TypeError("hpc_params must expose get_cluster_config() when resource requirements are omitted.")
    cfg = get_cluster_config()
    return ResourceRequirements(
        memory_gb=int(getattr(cfg, "base_memory_gb")),
        time_hours=int(getattr(cfg, "base_time_hours")),
        gpu_count=int(getattr(cfg, "gpu_count")),
        cpus=int(getattr(cfg, "cpus_per_task")),
    )


__all__ = [
    "CheckpointInfo",
    "ExperimentConfig",
    "ExperimentGroup",
    "ExperimentMetadata",
    "ResourceRequirements",
    "coerce_experiment_status",
    "coerce_task_type_string",
]

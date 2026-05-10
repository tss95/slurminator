"""Cluster registry models for Slurminator."""

from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class HPCType(Enum):
    """Known cluster identifiers."""

    FOX = "FOX"
    LUMI = "LUMI"
    SAGA = "SAGA"
    OLIVIA = "OLIVIA"


class HPCPartition(Enum):
    """Known Slurm partitions used by bundled cluster templates."""

    ACCEL = "accel"
    ACCEL_LONG = "accel_long"
    IFI_ACCEL = "ifi_accel"
    NORMAL = "normal"
    STANDARD_G = "standard-g"
    DEV_G = "dev-g"
    A100 = "a100"
    SMALL_G = "small-g"


class ResourceStatus(Enum):
    """Cluster resource availability state."""

    AVAILABLE = "available"
    IN_USE = "in_use"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"
    ERROR = "error"


def coerce_hpc_type(value: HPCType | str) -> HPCType:
    """Return ``value`` as an :class:`HPCType`."""
    if isinstance(value, HPCType):
        return value
    text = str(value).strip()
    if text in HPCType.__members__:
        return HPCType[text]
    upper = text.upper()
    if upper in HPCType.__members__:
        return HPCType[upper]
    return HPCType(text)


def coerce_hpc_partition(value: HPCPartition | str) -> HPCPartition:
    """Return ``value`` as an :class:`HPCPartition`."""
    if isinstance(value, HPCPartition):
        return value
    text = str(value).strip()
    if text in HPCPartition.__members__:
        return HPCPartition[text]
    upper = text.upper().replace("-", "_")
    if upper in HPCPartition.__members__:
        return HPCPartition[upper]
    return HPCPartition(text)


@dataclass
class HPCClusterConfig:
    """Cluster resource and SSH connection configuration."""

    cluster_type: HPCType
    partition: HPCPartition
    account: str

    hostname: str
    username: str
    port: int = 22
    use_key: bool = False
    key_path: str | None = None
    repo_path: str | None = None
    save_path: str | None = None
    data_path: str | None = None
    two_factor: bool = False
    proxy_jump: str | None = None
    proxy_jump_username: str | None = None
    proxy_jump_port: int = 22

    base_memory_gb: int = 80
    min_memory_gb: int = 80
    max_memory_gb: int = 80
    base_time_hours: int = 14
    max_time_hours: int = 14
    cpus_per_task: int = 4

    gpu_count: int = 1
    gpu_type: str = "a100"
    gpu_gres_name: str | None = None
    enable_ddp: bool = True
    backend: str = "nccl"
    num_workers: int | None = None

    cpus_per_gpu: int | None = None
    mem_per_gpu_gb: int | None = None
    gpu_bind_pattern: str | None = None
    exclude_nodes: list[str] | None = None

    request_gpu_pair: bool = False

    no_wandb: bool = False
    environment_setup: str = "step_0.sh"
    unqueue_threshold_secs: int = 600

    sync_pipe_dir: str | None = None
    wandb_runs_dir: str | None = None
    scratch_template: str = "/cluster/work/users/{user}/wandb_tmp/$SLURM_JOB_ID"

    submission_host: str | None = None
    submission_username: str | None = None
    submission_port: int | None = None
    submission_use_key: bool | None = None
    submission_key_path: str | None = None
    submission_two_factor: bool | None = None

    def __post_init__(self) -> None:
        self.cluster_type = coerce_hpc_type(self.cluster_type)
        self.partition = coerce_hpc_partition(self.partition)
        if self.use_key and not self.key_path:
            self.key_path = os.path.expanduser("~/.ssh/id_rsa")
        if isinstance(self.exclude_nodes, str):
            raw = self.exclude_nodes.replace(" ", ",")
            self.exclude_nodes = [node for node in (part.strip() for part in raw.split(",")) if node]
        if self.save_path:
            if not self.sync_pipe_dir:
                self.sync_pipe_dir = str(Path(self.save_path) / "wandb_sync_pipe")
            if not self.wandb_runs_dir:
                self.wandb_runs_dir = str(Path(self.save_path) / "wandb")


HPC_CONFIGS: dict[HPCType, HPCClusterConfig] = {}


@dataclass
class HPCParameters:
    """Reference a cluster in the active ``HPC_CONFIGS`` registry."""

    cluster_type: HPCType

    def __post_init__(self) -> None:
        self.cluster_type = coerce_hpc_type(self.cluster_type)

    def get_cluster_config(self, registry: Mapping[HPCType, HPCClusterConfig] | None = None) -> HPCClusterConfig:
        """Return this parameter object's cluster configuration."""
        return (registry or HPC_CONFIGS)[self.cluster_type]

    def to_dict(self) -> dict[str, str]:
        """Return a YAML-friendly representation."""
        return {"cluster_type": self.cluster_type.value}


def set_cluster_configs(configs: Mapping[HPCType, HPCClusterConfig]) -> None:
    """Replace the process-wide cluster registry in place."""
    HPC_CONFIGS.clear()
    HPC_CONFIGS.update(configs)


def parse_cluster_configs(
    raw: Mapping[str, Any], *, source: str | Path | None = None, logger: Any | None = None
) -> dict[HPCType, HPCClusterConfig]:
    """Parse a raw YAML mapping into cluster configs."""
    cluster_data = raw.get("clusters", raw)
    if not isinstance(cluster_data, Mapping):
        raise ValueError("hpc_config.yaml must contain a mapping or a top-level 'clusters' mapping.")

    configs: dict[HPCType, HPCClusterConfig] = {}
    for name, cfg in cluster_data.items():
        if not isinstance(cfg, MutableMapping):
            continue
        try:
            hpc_type = coerce_hpc_type(str(name))
        except ValueError:
            if logger is not None:
                logger.warning("Unknown HPC type %r in %s; skipping", name, source or "config")
            continue

        cfg_dict = dict(cfg)
        partition = cfg_dict.get("partition") or HPCPartition.ACCEL
        try:
            cfg_dict["partition"] = coerce_hpc_partition(partition)
        except ValueError:
            if logger is not None:
                logger.warning("Unknown partition %r for %s in %s; using ACCEL", partition, name, source or "config")
            cfg_dict["partition"] = HPCPartition.ACCEL

        cfg_dict["cluster_type"] = hpc_type
        configs[hpc_type] = HPCClusterConfig(**cfg_dict)

    return configs


__all__ = [
    "HPCClusterConfig",
    "HPCParameters",
    "HPCPartition",
    "HPCType",
    "HPC_CONFIGS",
    "ResourceStatus",
    "coerce_hpc_partition",
    "coerce_hpc_type",
    "parse_cluster_configs",
    "set_cluster_configs",
]

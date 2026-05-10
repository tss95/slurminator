from pathlib import Path
from types import SimpleNamespace

import pytest

from slurminator.config import HPCClusterConfig, HPCPartition, HPCType
from slurminator.experiment_policy import (
    resolve_extra_remote_dirs,
    resolve_pinned_hpc,
    resolve_resource_overrides,
    resolve_sbatch_export_vars,
)

pytestmark = pytest.mark.unit


def _cluster(**kwargs):
    defaults = {
        "cluster_type": HPCType.OLIVIA,
        "partition": HPCPartition.ACCEL,
        "account": "nn",
        "hostname": "example",
        "username": "user",
        "repo_path": "/repo",
        "save_path": "/save",
    }
    defaults.update(kwargs)
    return HPCClusterConfig(**defaults)


def test_resolve_pinned_hpc_uses_row_then_cluster_dataset_config() -> None:
    clusters = {
        HPCType.FOX: _cluster(cluster_type=HPCType.FOX, pinned_datasets=["sleepEDF"]),
        HPCType.OLIVIA: _cluster(cluster_type=HPCType.OLIVIA),
    }

    assert resolve_pinned_hpc({"pinned_hpc": "OLIVIA"}, clusters) == HPCType.OLIVIA
    assert resolve_pinned_hpc({"dataset_name": "sleepEDF"}, clusters) == HPCType.FOX
    assert resolve_pinned_hpc({"dataset_name": "HAR"}, clusters) is None


def test_resolve_resource_overrides_merges_dataset_and_row_aliases() -> None:
    clusters = {
        HPCType.OLIVIA: _cluster(dataset_resource_overrides={"electricity": {"memory_gb": 250, "time_hours": 1}})
    }

    overrides = resolve_resource_overrides(
        {"dataset_name": "electricity", "resource_overrides": {"gpu_count": 2}},
        hpc_type=HPCType.OLIVIA,
        cluster_configs=clusters,
    )

    assert overrides == {"mem_gb": 250, "time_hours": 1, "gpu_count": 2}


def test_resolve_sbatch_export_vars_formats_cluster_fields() -> None:
    cluster = SimpleNamespace(save_path="/save", repo_path="/repo", sbatch_env={"TRACKER_DIR": "{save_path}/runs"})

    assert resolve_sbatch_export_vars(cluster) == {"TRACKER_DIR": "/save/runs"}


def test_resolve_extra_remote_dirs_supports_relative_paths_and_placeholders(tmp_path) -> None:
    dirs = resolve_extra_remote_dirs(
        {"experiment_id": "exp-1", "ensure_dirs": ["logs/{experiment_id}", "/absolute/path"]}, base_path=Path(tmp_path)
    )

    assert dirs == (tmp_path / "logs" / "exp-1", Path("/absolute/path"))

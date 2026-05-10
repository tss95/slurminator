from enum import Enum

import pytest

from slurminator.experiments import ExperimentConfig, ExperimentGroup, ExperimentMetadata, ExperimentStatus

pytestmark = pytest.mark.unit


class AdapterTaskType(Enum):
    SELF_SUPERVISED = "self_supervised"
    SUPERVISED = "supervised"


class DummyHPCParams:
    def get_cluster_config(self):
        return type("Cluster", (), {"base_memory_gb": 32, "base_time_hours": 2, "gpu_count": 1, "cpus_per_task": 8})()


def test_experiment_config_uses_opaque_task_type_strings(tmp_path):
    exp = ExperimentConfig(task_type=AdapterTaskType.SUPERVISED, dataset_name="HAR", hpc_params=DummyHPCParams())

    assert exp.task_type == "supervised"
    assert exp.resource_requirements is not None
    assert exp.resource_requirements.memory_gb == 32
    assert exp.resource_requirements.cpus == 8

    exp.add_checkpoint(tmp_path / "ckpt.pt", epoch=3)
    latest = exp.get_latest_checkpoint()
    assert latest is not None
    assert latest.epoch == 3


def test_experiment_metadata_and_group_status_coercion():
    meta = ExperimentMetadata(status="pending")
    meta.update_status("running")

    assert meta.status == ExperimentStatus.RUNNING

    group = ExperimentGroup(
        "g",
        "desc",
        [ExperimentConfig("self_supervised", "a"), ExperimentConfig("self_supervised", "b", status="completed")],
    )
    assert group.get_experiment_count() == 2
    assert group.get_experiment_count("pending") == 1
    assert group.get_experiment_count(ExperimentStatus.COMPLETED) == 1

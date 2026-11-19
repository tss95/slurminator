from types import SimpleNamespace

import pytest

from slurminator.experiments import ExperimentStatus
from slurminator.config import HPCType
from slurminator.timeout_policy import (
    SubmissionResources,
    apply_timeout_policy,
    apply_timeout_retry_to_resources,
    estimate_timeout_retry_hours,
    resolve_requested_time_hours,
)

pytestmark = pytest.mark.unit


def test_apply_timeout_retry_to_resources_is_pure_and_prefers_experiment_override() -> None:
    exp = {"experiment_id": "exp-1", "time_hours_override": "7"}
    before = dict(exp)
    base = SubmissionResources(time_hours=2, memory_gb=80, cpus=4, gpu_count=2, mem_per_gpu_gb=40)

    resources = apply_timeout_retry_to_resources(exp, base, global_time_hours_override=4, global_memory_gb_override=192)

    assert resources == SubmissionResources(time_hours=7, memory_gb=192, cpus=4, gpu_count=2, mem_per_gpu_gb=None)
    assert resources.requested_ram_gb == 192
    assert base == SubmissionResources(time_hours=2, memory_gb=80, cpus=4, gpu_count=2, mem_per_gpu_gb=40)
    assert exp == before


def test_resolve_requested_time_hours_uses_tracked_then_retry_then_configured_defaults() -> None:
    cluster = SimpleNamespace(base_time_hours=3)

    assert (
        resolve_requested_time_hours(
            {"requested_time_hours": "11", "time_hours_override": 9},
            HPCType.OLIVIA,
            cluster_config=cluster,
            resource_overrides={"time_hours": 5},
            global_time_hours_override=4,
        )
        == 11
    )
    assert (
        resolve_requested_time_hours(
            {"time_hours_override": "9"},
            HPCType.OLIVIA,
            cluster_config=cluster,
            resource_overrides={"time_hours": 5},
            global_time_hours_override=4,
        )
        == 9
    )
    assert (
        resolve_requested_time_hours(
            {},
            HPCType.OLIVIA,
            cluster_config=cluster,
            resource_overrides={"time_hours": 5},
            global_time_hours_override=4,
        )
        == 4
    )


def test_estimate_timeout_retry_hours_uses_best_progress_and_clamps_to_cluster_max() -> None:
    exp = {
        "experiment_id": "exp-1",
        "requested_time_hours": 5,
        "current_epoch": 20,
        "max_epochs": 100,
        "current_step": 70,
        "max_steps": 100,
    }
    cluster = SimpleNamespace(base_time_hours=2, max_time_hours=8)

    estimated = estimate_timeout_retry_hours(
        exp,
        HPCType.OLIVIA,
        cluster_config=cluster,
        resource_overrides={},
        global_time_hours_override=None,
        timeout_retry_buffer=1.5,
    )

    assert estimated == 8


def test_apply_timeout_policy_marks_partial_retry_with_estimated_walltime() -> None:
    exp = {
        "experiment_id": "exp-1",
        "status": ExperimentStatus.RUNNING,
        "requested_time_hours": 4,
        "current_step": 500,
        "max_steps": 1000,
    }
    cluster = SimpleNamespace(base_time_hours=2, max_time_hours=24)

    apply_timeout_policy(
        exp,
        HPCType.OLIVIA,
        reason="slurm-timeout",
        cluster_config=cluster,
        resource_overrides={},
        global_time_hours_override=None,
        retry_timeout_with_estimated_time=True,
        timeout_retry_buffer=1.25,
        timeout_retry_max_attempts=2,
    )

    assert exp["status"] == ExperimentStatus.PARTIAL
    assert exp["time_hours_override"] == 10
    assert exp["timeout_retry_count"] == 1

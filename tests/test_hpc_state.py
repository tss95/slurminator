import pytest

from slurminator.experiments import ExperimentStatus
from slurminator.hpc_state import expand_slurm_state, is_terminal_status, map_scheduler_state_to_experiment_status

pytestmark = pytest.mark.unit


def test_expand_slurm_state_handles_short_and_decorated_long_states() -> None:
    assert expand_slurm_state("PD") == "PENDING"
    assert expand_slurm_state("R") == "RUNNING"
    assert expand_slurm_state("CANCELLED+") == "CANCELLED"
    assert expand_slurm_state("COMPLETED*") == "COMPLETED"
    assert expand_slurm_state("OUT_OF_MEMORY") == "OUT_OF_MEMORY"


def test_map_scheduler_state_to_experiment_status() -> None:
    assert map_scheduler_state_to_experiment_status("PENDING") == ExperimentStatus.QUEUED
    assert map_scheduler_state_to_experiment_status("CD") == ExperimentStatus.COMPLETED
    assert map_scheduler_state_to_experiment_status("TO") == ExperimentStatus.TIMEOUT
    assert map_scheduler_state_to_experiment_status("OOM") == ExperimentStatus.OOM
    assert map_scheduler_state_to_experiment_status("NODE_FAIL") == ExperimentStatus.FAILED
    assert map_scheduler_state_to_experiment_status("unknown") == ExperimentStatus.FAILED


def test_is_terminal_status_accepts_enum_and_value_strings() -> None:
    assert is_terminal_status(ExperimentStatus.COMPLETED) is True
    assert is_terminal_status("timeout") is True
    assert is_terminal_status("OOM") is True
    assert is_terminal_status(ExperimentStatus.RUNNING) is False
    assert is_terminal_status("unknown") is False

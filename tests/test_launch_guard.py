import socket

import pytest

import slurminator.launch_guard as launch_guard
from slurminator.config import HPCType

pytestmark = pytest.mark.unit


def test_guard_no_block_when_not_on_hpc(monkeypatch) -> None:
    monkeypatch.setattr(launch_guard, "determine_current_hpc", lambda: None)
    monkeypatch.setattr(launch_guard, "_is_cuda_available", lambda: True)

    assert launch_guard.get_orchestrator_gpu_hpc_launch_block_message() is None


def test_guard_no_block_when_no_cuda(monkeypatch) -> None:
    monkeypatch.setattr(launch_guard, "determine_current_hpc", lambda: HPCType.OLIVIA)
    monkeypatch.setattr(launch_guard, "_is_cuda_available", lambda: False)

    assert launch_guard.get_orchestrator_gpu_hpc_launch_block_message() is None


def test_guard_blocks_when_hpc_and_cuda(monkeypatch) -> None:
    monkeypatch.setattr(launch_guard, "determine_current_hpc", lambda: HPCType.OLIVIA)
    monkeypatch.setattr(launch_guard, "_is_cuda_available", lambda: True)
    monkeypatch.setattr(socket, "gethostname", lambda: "gpu-1-13")
    monkeypatch.setenv("SLURM_JOB_ID", "12345")

    msg = launch_guard.get_orchestrator_gpu_hpc_launch_block_message()

    assert msg is not None
    assert "hpc=OLIVIA" in msg
    assert "host=gpu-1-13" in msg
    assert "SLURM_JOB_ID=12345" in msg
    assert "login/control node" in msg

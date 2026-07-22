"""Tests for safe Slurm submission transport and response parsing."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from slurminator.config import HPCClusterConfig, HPCPartition, HPCType
from slurminator.submission import SubmissionContext, _parse_sbatch_job_id, submit_experiment_universal

pytestmark = pytest.mark.unit


class RecordingConnectionManager:
    """Record a single hermetic sbatch invocation and return configured streams."""

    def __init__(self, stdout: str, stderr: str) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.calls: list[tuple[HPCType, str, bool, bool]] = []

    def run_command(
        self, hpc_type: HPCType, command: str, prefer_remote: bool = False, retry_on_failure: bool = True
    ) -> tuple[str, str]:
        """Return configured sbatch streams while recording transport flags."""
        self.calls.append((hpc_type, command, prefer_remote, retry_on_failure))
        return self.stdout, self.stderr

    def run_submission_command(self, hpc_type: HPCType, command: str) -> tuple[str, str]:
        """Record the built-in manager's non-retrying submission contract."""
        return self.run_command(hpc_type, command, prefer_remote=True, retry_on_failure=False)


def _cluster(save_path: Path) -> HPCClusterConfig:
    return HPCClusterConfig(
        cluster_type=HPCType.OLIVIA,
        partition=HPCPartition.ACCEL,
        account="nn-test",
        hostname="olivia.example",
        username="test-user",
        repo_path="/remote/repo",
        save_path=str(save_path),
    )


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("12345\n", "12345"),
        ("12345;olivia\n", "12345"),
        ("Submitted batch job 12345\n", "12345"),
        ("Submitted batch job invalid\n", None),
        ("warning: job 12345 was not submitted\n", None),
    ],
)
def test_parse_sbatch_job_id_accepts_only_exact_supported_formats(output: str, expected: str | None) -> None:
    assert _parse_sbatch_job_id(output) == expected


def test_submit_uses_parsable_no_retry_and_keeps_job_id_despite_stderr_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    cluster = _cluster(tmp_path)
    connection_manager = RecordingConnectionManager(
        stdout="12345;olivia\n", stderr="sbatch: warning: requested QoS was normalized\n"
    )
    context = SubmissionContext(
        experiment_file=tmp_path / "experiments.yaml",
        concurrency_limits={HPCType.OLIVIA: 1},
        hpc_configs={HPCType.OLIVIA: cluster},
        connection_manager=connection_manager,
        build_commands_line=lambda _exp, _gpu_count, _hpc_type: "python train.py",
        is_local_hpc=lambda _hpc_type: True,
        prepared_repositories={HPCType.OLIVIA},
    )
    exp = {"experiment_id": "exp-1", "dataset_name": "synthetic"}

    with caplog.at_level(logging.WARNING, logger="slurminator"):
        job_id = submit_experiment_universal(exp, HPCType.OLIVIA, context)

    assert job_id == "12345"
    assert len(connection_manager.calls) == 1
    called_hpc, command, prefer_remote, retry_on_failure = connection_manager.calls[0]
    assert called_hpc == HPCType.OLIVIA
    assert command.startswith("sbatch ")
    assert "--parsable" in command.split()
    assert prefer_remote is True
    assert retry_on_failure is False
    assert exp["requested_gpu_count"] == 1
    assert "sbatch warning" in caplog.text

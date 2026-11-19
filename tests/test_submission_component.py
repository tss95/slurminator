from __future__ import annotations

from pathlib import Path
from shlex import split
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from slurminator.experiments import ExperimentStatus
from slurminator.config import HPCType
from slurminator.state_store import replace_exp_in_list
from slurminator.submission import (
    SubmissionContext,
    _push_critical_scripts,
    build_sbatch_command,
    maybe_submit,
    resolve_submission_resources,
    submit_experiment_universal,
)

pytestmark = pytest.mark.unit


class _FakeConnection:
    def __init__(self) -> None:
        self.commands: list[tuple[HPCType, str, bool]] = []

    def run_command(self, hpc_type, cmd, prefer_remote=False):  # noqa: ANN001
        self.commands.append((hpc_type, cmd, prefer_remote))
        if "echo git" in cmd:
            return ("git\n", "")
        if cmd.startswith("sbatch"):
            return ("Submitted batch job 4242\n", "")
        return ("", "")


def _cluster_config(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        partition=SimpleNamespace(value="gpu"),
        account="nn-test",
        base_time_hours=2,
        max_time_hours=24,
        base_memory_gb=80,
        cpus_per_task=8,
        cpus_per_gpu=None,
        mem_per_gpu_gb=None,
        gpu_count=1,
        request_gpu_pair=False,
        gpu_type="a100",
        gpu_gres_name=None,
        gpu_bind_pattern=None,
        exclude_nodes=[],
        dataset_resource_overrides={},
        sbatch_env={},
        save_path=str(tmp_path / "save"),
        repo_path=str(tmp_path / "repo"),
        environment_setup="env.sh",
    )


def _context(
    tmp_path: Path,
    *,
    cluster: SimpleNamespace | None = None,
    connection: _FakeConnection | None = None,
    time_hours_override: int | None = None,
    memory_gb_override: int | None = None,
) -> SubmissionContext:
    return SubmissionContext(
        experiment_file=tmp_path / "experiment_list.yaml",
        concurrency_limits={HPCType.OLIVIA: 1},
        hpc_configs={HPCType.OLIVIA: cluster or _cluster_config(tmp_path)},
        connection_manager=connection or _FakeConnection(),
        build_commands_line=lambda exp, gpus, hpc: f"python run_experiment.py --job {exp['experiment_id']} --gpus {gpus}",
        is_local_hpc=lambda hpc: True,
        time_hours_override=time_hours_override,
        memory_gb_override=memory_gb_override,
    )


def test_resolve_submission_resources_applies_config_and_retry_overrides_without_orchestrator(tmp_path) -> None:
    cluster = _cluster_config(tmp_path)
    cluster.dataset_resource_overrides = {"synthetic": {"time_hours": 3, "mem_gb": 120, "gpu_count": 2}}
    context = _context(tmp_path, cluster=cluster, time_hours_override=5, memory_gb_override=256)
    exp = {"experiment_id": "exp-1", "dataset_name": "synthetic", "time_hours_override": 7}

    resources = resolve_submission_resources(exp, HPCType.OLIVIA, context)

    assert resources.time_hours == 7
    assert resources.memory_gb == 256
    assert resources.gpu_count == 2
    assert resources.cpus == 8
    assert "requested_time_hours" not in exp


@pytest.mark.parametrize("use_shared_wrapper", [False, True])
def test_submit_experiment_universal_builds_sbatch_and_records_requested_resources(
    tmp_path, use_shared_wrapper
) -> None:
    connection = _FakeConnection()
    cluster = _cluster_config(tmp_path)
    cluster.sbatch_env = {"SLURMINATOR_TEST_SWITCH": "1"}
    if use_shared_wrapper:
        cluster.job_script = str(tmp_path / "shared slurminator" / "universal_job.sh")
    context = _context(tmp_path, cluster=cluster, connection=connection)
    exp = {
        "experiment_id": "exp-1",
        "dataset_name": "synthetic",
        "resource_overrides": {"time_hours": 3, "mem_gb": 120},
    }

    job_id = submit_experiment_universal(exp, HPCType.OLIVIA, context)

    assert job_id == "4242"
    assert exp["requested_time_hours"] == 3
    assert exp["requested_ram_gb"] == 120
    assert exp["requested_gpu_count"] == 1
    sbatch_cmd = next(cmd for _, cmd, _ in connection.commands if cmd.startswith("sbatch"))
    assert "--time=3:00:00" in sbatch_cmd
    assert "--mem=120G" in sbatch_cmd
    assert "--export=ALL,SLURMINATOR_TEST_SWITCH=1,SLURMINATOR_NPROC_PER_NODE=1" in sbatch_cmd
    assert sbatch_cmd.count("--export=") == 1
    argv = split(sbatch_cmd)
    expected_script = cluster.job_script if use_shared_wrapper else f"{cluster.repo_path}/universal_job.sh"
    assert argv[argv.index("--env_script") - 1] == expected_script
    assert argv[argv.index("--env_script") + 1] == f"{cluster.repo_path}/env.sh"


def test_build_sbatch_command_uses_cluster_config_for_excludes_gres_and_gpu_bind(tmp_path) -> None:
    cluster = _cluster_config(tmp_path)
    cluster.exclude_nodes = ["gpu-1", "gpu-2"]
    cluster.gpu_gres_name = "a100_80"
    cluster.gpu_bind_pattern = "closest"

    sbatch_cmd = build_sbatch_command(
        exp={"experiment_id": "exp-1"},
        hpc_type=HPCType.FOX,
        cluster_config=cluster,
        resources=resolve_submission_resources({"experiment_id": "exp-1"}, HPCType.OLIVIA, _context(tmp_path)),
        partition="gpu",
        job_name="exp-1",
        exp_output_dir=tmp_path / "out",
        commands_line="python train.py",
    )

    assert "--exclude=gpu-1,gpu-2" in sbatch_cmd
    assert "--gres=gpu:a100_80:1" in sbatch_cmd
    assert "--gpu-bind=closest" in sbatch_cmd


def test_maybe_submit_assigns_hpc_and_persists_without_orchestrator() -> None:
    exp = {"experiment_id": "exp-1", "status": ExperimentStatus.PENDING}
    data = {"experiments": [exp]}
    saved: list[dict[str, Any]] = []

    maybe_submit(
        exp,
        {HPCType.OLIVIA: 0},
        data,
        concurrency_limits={HPCType.OLIVIA: 1},
        hpc_configs={HPCType.OLIVIA: _cluster_config(Path("/tmp"))},
        submit_experiment=lambda row, hpc: "4242",
        replace_exp_in_list=replace_exp_in_list,
        save_yaml=saved.append,
    )

    assert exp["status"] == ExperimentStatus.QUEUED
    assert exp["job_id"] == "4242"
    assert exp["hpc_assignment"] == HPCType.OLIVIA
    assert len(saved) == 1
    assert saved[0]["experiments"][0]["experiment_id"] == "exp-1"


def test_maybe_submit_respects_config_pinned_hpc_without_orchestrator() -> None:
    exp = {
        "experiment_id": "exp-1",
        "status": ExperimentStatus.PENDING,
        "hpc_assignment": HPCType.FOX,
        "pinned_hpc": "OLIVIA",
    }
    data = {"experiments": [exp]}

    maybe_submit(
        exp,
        {HPCType.OLIVIA: 0},
        data,
        concurrency_limits={HPCType.OLIVIA: 1, HPCType.FOX: 0},
        hpc_configs={HPCType.OLIVIA: _cluster_config(Path("/tmp")), HPCType.FOX: _cluster_config(Path("/tmp"))},
        submit_experiment=lambda row, hpc: "4242",
        replace_exp_in_list=replace_exp_in_list,
        save_yaml=lambda data: None,
    )

    assert exp["hpc_assignment"] == HPCType.OLIVIA
    assert exp["status"] == ExperimentStatus.QUEUED


@pytest.mark.parametrize("configured", [False, True])
def test_remote_preparation_uses_default_wrapper_only_when_unconfigured(tmp_path, configured) -> None:
    cluster = _cluster_config(tmp_path)
    if configured:
        cluster.job_script = "/shared/tools/universal_job.sh"
    connection = Mock()

    _push_critical_scripts(HPCType.FOX, cluster, connection)

    if configured:
        connection.upload_file.assert_not_called()
        assert "universal_job.sh" not in connection.run_command.call_args.args[1]
    else:
        connection.upload_file.assert_called_once()
        hpc_type, local, remote = connection.upload_file.call_args.args
        assert hpc_type == HPCType.FOX
        assert Path(local) == Path(__file__).resolve().parents[1] / "universal_job.sh"
        assert remote == f"{cluster.repo_path}/universal_job.sh"

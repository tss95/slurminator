import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def _write_fake_nvidia_smi(fake_bin: Path, gpu_count: int) -> None:
    fake_bin.mkdir(parents=True, exist_ok=True)
    lines = "".join(f"echo 'GPU {idx}: Fake GPU (UUID: GPU-fake-{idx})'\n" for idx in range(gpu_count))
    nvidia_smi = fake_bin / "nvidia-smi"
    nvidia_smi.write_text(f"#!/usr/bin/env bash\n{lines}", encoding="utf-8")
    nvidia_smi.chmod(0o755)


def _write_fake_python_gpu_count(fake_bin: Path, gpu_count: int) -> None:
    """Install a minimal Python shim for the runtime GPU-count preflight."""
    python = fake_bin / "python"
    python.write_text(f"#!/usr/bin/env bash\nprintf '%s\\n' '{gpu_count}'\n", encoding="utf-8")
    python.chmod(0o755)


def _run_universal_job(
    tmp_path: Path,
    *,
    command: str,
    env_overrides: dict[str, str],
    env_script: Path | None = None,
    experiment_id: str | None = None,
) -> subprocess.CompletedProcess[str]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(exist_ok=True)

    save_path = tmp_path / "save"
    save_path.mkdir()
    env = {key: value for key, value in os.environ.items() if not key.startswith("SLURM_")}
    env.update({"PATH": f"{fake_bin}:{env.get('PATH', '')}", **env_overrides})

    args = ["bash", "universal_job.sh", "--save_path", str(save_path)]
    if experiment_id is not None:
        args.extend(["--exp_id", experiment_id])
    if env_script is not None:
        args.extend(["--env_script", str(env_script)])
    args.extend(["--commands", command])

    return subprocess.run(args, cwd=Path(__file__).resolve().parents[1], env=env, text=True, capture_output=True)


def test_universal_job_exports_exact_ledger_identity_before_environment_setup(tmp_path: Path) -> None:
    """Seed-qualified IDs override inherited controller identity and reach containers."""
    env_script = tmp_path / 'env.sh'
    checks = (
        'test "$ORCHESTRATOR_EXPERIMENT_ID" = "example_s43" '
        '&& test "$APPTAINERENV_ORCHESTRATOR_EXPERIMENT_ID" = "example_s43" '
        '&& test "$SINGULARITYENV_ORCHESTRATOR_EXPERIMENT_ID" = "example_s43"'
    )
    env_script.write_text(checks + '\n', encoding='utf-8')
    result = _run_universal_job(
        tmp_path,
        command=checks,
        env_script=env_script,
        experiment_id='example_s43',
        env_overrides={'ORCHESTRATOR_EXPERIMENT_ID': 'stale-controller-id'},
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_universal_job_normalizes_single_slurm_gpu_id_to_accessible_cuda_ordinal(tmp_path: Path) -> None:
    """Olivia exposes allocated physical GPU ids, while containers need visible ordinals."""
    _write_fake_nvidia_smi(tmp_path / "bin", gpu_count=1)
    command = (
        'test "$CUDA_VISIBLE_DEVICES" = "0" '
        '&& test "$APPTAINERENV_CUDA_VISIBLE_DEVICES" = "0" '
        '&& test "$SINGULARITYENV_CUDA_VISIBLE_DEVICES" = "0"'
    )
    result = _run_universal_job(
        tmp_path, command=command, env_overrides={"SLURM_JOB_GPUS": "2", "SLURMINATOR_NPROC_PER_NODE": "1"}
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Detected GPUs on node: 1" in result.stdout
    assert "Normalized CUDA_VISIBLE_DEVICES to accessible set: 0" in result.stdout
    assert "Launch process count: 1" in result.stdout


def test_universal_job_counts_multi_gpu_id_list_and_normalizes_ordinals(tmp_path: Path) -> None:
    """A Slurm GPU id list like ``2,3`` should count as two GPUs, not three."""
    _write_fake_nvidia_smi(tmp_path / "bin", gpu_count=2)
    command = (
        'test "$CUDA_VISIBLE_DEVICES" = "0,1" '
        '&& test "$APPTAINERENV_CUDA_VISIBLE_DEVICES" = "0,1" '
        '&& test "$SINGULARITYENV_CUDA_VISIBLE_DEVICES" = "0,1"'
    )
    result = _run_universal_job(
        tmp_path, command=command, env_overrides={"SLURM_JOB_GPUS": "2,3", "SLURMINATOR_NPROC_PER_NODE": "2"}
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Detected GPUs on node: 2" in result.stdout
    assert "Normalized CUDA_VISIBLE_DEVICES to accessible set: 0,1" in result.stdout
    assert "Launch process count: 2" in result.stdout


def test_universal_job_rejects_torchrun_when_runtime_sees_fewer_gpus(tmp_path: Path) -> None:
    """A container-visible GPU shortfall must fail before the training command starts."""
    fake_bin = tmp_path / "bin"
    _write_fake_nvidia_smi(fake_bin, gpu_count=2)
    _write_fake_python_gpu_count(fake_bin, gpu_count=1)
    marker = tmp_path / "save" / "training-started"

    result = _run_universal_job(
        tmp_path,
        command=f"touch {marker} # torchrun",
        env_overrides={"SLURM_JOB_GPUS": "0,1", "SLURMINATOR_NPROC_PER_NODE": "2"},
    )

    assert result.returncode != 0
    assert not marker.exists()
    assert "Torch-visible GPUs in training runtime: 1" in result.stdout
    assert "launch requests 2 rank(s), but the training runtime exposes only 1 GPU(s)" in result.stderr


def test_universal_job_sources_environment_before_running_command(tmp_path: Path) -> None:
    """Job steps use the initialized runtime and export its environment."""
    env_script = tmp_path / "env.sh"
    marker = tmp_path / "save" / "initialized.txt"
    env_script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                'test "${SLURM_EXPORT_ENV:-}" = "ALL" || exit 14',
                'test "$1" = "--save_path" && test "$2" = "$SAVE_PATH" || exit 12',
                'printf ready > "${SAVE_PATH}/initialized.txt"',
                "export RUNTIME_RELOADED=1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    env_script.chmod(0o755)

    result = _run_universal_job(
        tmp_path,
        command=f'test "$RUNTIME_RELOADED" = "1" && test "$(cat {marker})" = "ready"',
        env_script=env_script,
        env_overrides={"SLURM_EXPORT_ENV": "NONE"},
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Sourcing environment script:" in result.stdout

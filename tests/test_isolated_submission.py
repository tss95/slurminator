"""Isolated submission uses fake sbatch processes; no scheduler calls are made."""

from __future__ import annotations

import json
import os
from pathlib import Path
import pwd
import shlex
import sys
from types import SimpleNamespace

import pytest

from slurminator.cli.orchestrator import _bootstrap_connection_manager, run_orchestrator_cli
from slurminator.config import HPCClusterConfig, HPCPartition, HPCType
from slurminator.connection_manager import HPCConnectionConfig, HPCConnectionManager
from slurminator.submission import SubmissionContext, build_sbatch_command, submit_experiment_universal
from slurminator.timeout_policy import SubmissionResources

pytestmark = pytest.mark.unit


@pytest.fixture
def manager(monkeypatch: pytest.MonkeyPatch) -> HPCConnectionManager:
    connection = HPCConnectionManager({HPCType.OLIVIA: HPCConnectionConfig(hostname="local", username="test")})
    monkeypatch.setattr(connection, "is_local_hpc", lambda _hpc: True)
    monkeypatch.delenv("SLURMINATOR_SUBMISSION_HOST_OLIVIA", raising=False)
    monkeypatch.delenv("PMT_SUBMISSION_HOST_OLIVIA", raising=False)
    return connection


def _fake_sbatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, exit_code: int = 0) -> Path:
    executable = tmp_path / "sbatch"
    capture = tmp_path / "calls.jsonl"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        f"with open({str(capture)!r}, 'a') as stream:\n"
        "    stream.write(json.dumps({'argv': sys.argv, 'env': dict(os.environ)}) + '\\n')\n"
        f"print({'test rejection' if exit_code else '12345'!r}, file=sys.stderr if {exit_code} else sys.stdout)\n"
        f"sys.exit({exit_code})\n"
    )
    executable.chmod(0o700)
    monkeypatch.setenv("PATH", f"{tmp_path}:/usr/bin:/bin")
    return capture


def test_native_fake_sbatch_gets_only_minimal_identity_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, manager: HPCConnectionManager
) -> None:
    capture = _fake_sbatch(tmp_path, monkeypatch)
    for key in (
        "SLURM_JOB_ID",
        "SLURM_CPUS_PER_TASK",
        "SLURM_EXPORT_ENV",
        "SBATCH_PARTITION",
        "SRUN_EXPORT_ENV",
        "PMT_AGENT_SANDBOX",
        "APPTAINER_BIND",
        "SINGULARITYENV_SECRET",
        "WANDB_API_KEY",
        "TMPDIR",
        "BASH_ENV",
        "PYTHONPATH",
        "HF_TOKEN",
        "CUDA_VISIBLE_DEVICES",
        "HOME",
        "USER",
        "LOGNAME",
        "LANG",
    ):
        monkeypatch.setenv(key, "inherited-untrusted-value")
    command = shlex.join(["sbatch", "--export=ALL,EXPLICIT=approved value", "job with spaces.sh", "$(touch never)"])
    out, err = manager.run_submission_command(HPCType.OLIVIA, command, isolated=True)
    assert (out, err) == ("12345\n", "")
    [call] = [json.loads(line) for line in capture.read_text().splitlines()]
    assert call["argv"] == [str(tmp_path / "sbatch"), *shlex.split(command)[1:]]
    identity = pwd.getpwuid(os.getuid())
    environment = call["env"]
    # Python's startup may coerce the POSIX locale in this fake executable.
    environment.pop("LC_CTYPE", None)
    assert environment == {
        "PATH": "/usr/bin:/bin",
        "HOME": identity.pw_dir,
        "USER": identity.pw_name,
        "LOGNAME": identity.pw_name,
        "LANG": "C",
    }


def test_isolated_stub_rejection_is_not_retried_or_bypassed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, manager: HPCConnectionManager
) -> None:
    capture = _fake_sbatch(tmp_path, monkeypatch, exit_code=73)
    monkeypatch.setattr(manager, "run_command", lambda *_a, **_kw: pytest.fail("No fallback permitted"))
    with pytest.raises(RuntimeError, match="exit status 73: test rejection"):
        manager.run_submission_command(HPCType.OLIVIA, "sbatch job.sh", isolated=True)
    assert len(capture.read_text().splitlines()) == 1


@pytest.mark.parametrize("submission_host", ["submit.remote.invalid", "login"])
def test_remote_submission_host_rejected_before_process(
    monkeypatch: pytest.MonkeyPatch, manager: HPCConnectionManager, submission_host: str
) -> None:
    manager.configs[HPCType.OLIVIA].submission_host = submission_host
    monkeypatch.setattr("subprocess.run", lambda *_a, **_kw: pytest.fail("Must not spawn"))
    with pytest.raises(RuntimeError, match="remote submission_host"):
        manager.run_submission_command(HPCType.OLIVIA, "sbatch job.sh", isolated=True)


def test_environment_submission_host_override_is_rejected(
    monkeypatch: pytest.MonkeyPatch, manager: HPCConnectionManager
) -> None:
    monkeypatch.setenv("SLURMINATOR_SUBMISSION_HOST_OLIVIA", "submit.remote.invalid")
    with pytest.raises(RuntimeError, match="remote submission_host"):
        manager.run_submission_command(HPCType.OLIVIA, "sbatch job.sh", isolated=True)


def test_nonlocal_cluster_is_rejected(monkeypatch: pytest.MonkeyPatch, manager: HPCConnectionManager) -> None:
    monkeypatch.setattr(manager, "is_local_hpc", lambda _hpc: False)
    with pytest.raises(RuntimeError, match="selected cluster to be local"):
        manager.run_submission_command(HPCType.OLIVIA, "sbatch job.sh", isolated=True)


@pytest.mark.parametrize("host", [None, "localhost", "127.0.0.1", "::1"])
def test_local_submission_host_is_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, manager: HPCConnectionManager, host: str | None
) -> None:
    capture = _fake_sbatch(tmp_path, monkeypatch)
    manager.configs[HPCType.OLIVIA].submission_host = host
    manager.run_submission_command(HPCType.OLIVIA, "sbatch job.sh", isolated=True)
    assert json.loads(capture.read_text())["argv"][1] == "--export=ALL"


@pytest.mark.parametrize(
    "command",
    [
        "env sbatch job.sh",
        "bash -lc 'sbatch job.sh'",
        "/usr/bin/sbatch job.sh",
        "",
        "sbatch --export=NONE job.sh",
        "sbatch --export=PATH job.sh",
        "sbatch --export ALL job.sh",
        "sbatch --get-user-env job.sh",
        "sbatch --export-file=env.txt job.sh",
    ],
)
def test_unsafe_command_shapes_fail(manager: HPCConnectionManager, command: str) -> None:
    with pytest.raises(ValueError):
        manager.run_submission_command(HPCType.OLIVIA, command, isolated=True)


def test_missing_visible_sbatch_fails_without_fallback(
    monkeypatch: pytest.MonkeyPatch, manager: HPCConnectionManager
) -> None:
    monkeypatch.setattr("slurminator.connection_manager.shutil.which", lambda _name: None)
    with pytest.raises(RuntimeError, match="unavailable"):
        manager.run_submission_command(HPCType.OLIVIA, "sbatch job.sh", isolated=True)


def test_default_transport_preserved(monkeypatch: pytest.MonkeyPatch, manager: HPCConnectionManager) -> None:
    calls = []
    monkeypatch.setattr(manager, "run_command", lambda *a, **kw: calls.append((a, kw)) or ("12345", ""))
    assert manager.run_submission_command(HPCType.OLIVIA, "sbatch job.sh") == ("12345", "")
    assert calls == [((HPCType.OLIVIA, "sbatch job.sh"), {"prefer_remote": True, "retry_on_failure": False})]


def _cluster(tmp_path: Path) -> HPCClusterConfig:
    return HPCClusterConfig(
        cluster_type=HPCType.OLIVIA,
        partition=HPCPartition.ACCEL,
        account="test-account",
        hostname="local",
        username="test",
        repo_path=str(tmp_path / "repo with spaces"),
        save_path=str(tmp_path / "save path"),
        environment_setup="step_0.sh",
        sbatch_env={"EXPLICIT": "hello world; $(touch never)"},
    )


def test_command_preserves_quoted_paths_commands_and_explicit_exports(tmp_path: Path) -> None:
    cfg = _cluster(tmp_path)
    command = build_sbatch_command(
        exp={},
        hpc_type=HPCType.OLIVIA,
        cluster_config=cfg,
        resources=SubmissionResources(time_hours=2, memory_gb=8, cpus=1, gpu_count=1),
        partition="accel",
        job_name="experiment with spaces",
        exp_output_dir=tmp_path / "out dir",
        commands_line="python train.py --name 'a b' && echo done",
    )
    args = shlex.split(command)
    assert "--export=ALL,EXPLICIT=hello world; $(touch never),SLURMINATOR_NPROC_PER_NODE=1" in args
    assert f"{cfg.repo_path}/universal_job.sh" in args
    assert args[args.index("--env_script") + 1] == f"{cfg.repo_path}/step_0.sh"
    assert args[args.index("--commands") + 1] == "python train.py --name 'a b' && echo done"
    assert args[args.index("--exp_id") + 1] == "experiment with spaces"


@pytest.mark.parametrize("exports", [{"INVALID=NAME": "value"}, {"GOOD": "bad,ambiguous"}])
def test_ambiguous_explicit_exports_fail(tmp_path: Path, exports: dict[str, str]) -> None:
    cfg = _cluster(tmp_path)
    cfg.sbatch_env = exports
    with pytest.raises(ValueError, match="sbatch export"):
        build_sbatch_command(
            exp={},
            hpc_type=HPCType.OLIVIA,
            cluster_config=cfg,
            resources=SubmissionResources(time_hours=2, memory_gb=8, cpus=1, gpu_count=1),
            partition="accel",
            job_name="experiment",
            exp_output_dir=tmp_path,
            commands_line="true",
        )


def _context(tmp_path: Path, connection_manager: object) -> SubmissionContext:
    return SubmissionContext(
        experiment_file=tmp_path / "ledger.yaml",
        concurrency_limits={HPCType.OLIVIA: 1},
        hpc_configs={HPCType.OLIVIA: _cluster(tmp_path)},
        connection_manager=connection_manager,
        build_commands_line=lambda *_a: "true",
        is_local_hpc=lambda _hpc: True,
        isolated_submission_env=True,
    )


def test_isolated_submission_skips_remote_repo_preparation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, manager: HPCConnectionManager
) -> None:
    capture = _fake_sbatch(tmp_path, monkeypatch)
    monkeypatch.setattr(manager, "run_command", lambda *_a, **_kw: pytest.fail("No repo prep/remote command"))
    exp = {"experiment_id": "synthetic"}
    assert submit_experiment_universal(exp, HPCType.OLIVIA, _context(tmp_path, manager)) == "12345"
    assert len(capture.read_text().splitlines()) == 1
    assert exp["requested_gpu_count"] == 1


@pytest.mark.parametrize("stdout", ["", "unrecognized submission response", "12345\\n67890"])
def test_ambiguous_success_stops_controller_instead_of_leaving_pending_to_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, manager: HPCConnectionManager, stdout: str
) -> None:
    calls = []

    def submit(_hpc, _command, *, isolated=False):
        calls.append(isolated)
        return stdout, ""

    monkeypatch.setattr(manager, "run_submission_command", submit)
    with pytest.raises(RuntimeError, match="submission outcome is unknown"):
        submit_experiment_universal({"experiment_id": "synthetic"}, HPCType.OLIVIA, _context(tmp_path, manager))
    assert calls == [True]


@pytest.mark.parametrize(
    "legacy",
    [
        SimpleNamespace(run_command=lambda *_a, **_kw: pytest.fail("No legacy fallback")),
        SimpleNamespace(
            validate_isolated_submission_target=lambda _hpc: None,
            run_submission_command=lambda _hpc, _cmd: pytest.fail("No unisolated call"),
        ),
    ],
)
def test_injected_manager_must_explicitly_support_isolation(tmp_path: Path, legacy: object) -> None:
    with pytest.raises(RuntimeError, match="does not support isolated submission"):
        submit_experiment_universal({"experiment_id": "synthetic"}, HPCType.OLIVIA, _context(tmp_path, legacy))
    assert not list(tmp_path.iterdir())


def test_cli_bootstrap_validates_before_any_connections(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(HPCConnectionManager, "is_local_hpc", lambda *_a: False)
    monkeypatch.setattr(HPCConnectionManager, "connect", lambda *_a, **_kw: pytest.fail("No SSH allowed"))
    with pytest.raises(RuntimeError, match="selected cluster to be local"):
        _bootstrap_connection_manager(
            {HPCType.OLIVIA: 1}, cluster_configs={HPCType.OLIVIA: _cluster(tmp_path)}, isolated_submission_env=True
        )


def test_cli_passes_explicit_isolation_to_controller(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SLURMINATOR_PLUGIN", raising=False)
    ledger = tmp_path / "ledger.yaml"
    ledger.write_text("experiments: []\n")
    captured = {}

    class Controller:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self):
            captured["ran"] = True

    run_orchestrator_cli(
        argv=["--yaml", str(ledger), "--isolated-submission-env"],
        load_configs=False,
        launch_guard=lambda: None,
        orchestrator_cls=Controller,
    )
    assert captured["isolated_submission_env"] is True
    assert captured["ran"] is True


def test_isolation_rejects_plugin_sweep_before_preparation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SLURMINATOR_PLUGIN", raising=False)

    def extend(parser):
        parser.add_argument("--wandb-sweep")
        return parser

    def forbidden(*_args, **_kwargs):
        pytest.fail("Plugin preparation, connections and sweep runners must not start")

    with pytest.raises(SystemExit) as error:
        run_orchestrator_cli(
            argv=["--wandb-sweep", "pending.yaml", "--isolated-submission-env"],
            parser_extender=extend,
            args_preparer=forbidden,
            load_configs=False,
            launch_guard=forbidden,
            connection_manager_cls=forbidden,
            sweep_mode_runner=forbidden,
        )
    assert error.value.code == 2

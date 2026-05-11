from pathlib import Path

import pytest

from slurminator.cli import parse_partition_overrides, run_orchestrator_cli
from slurminator.config import HPCClusterConfig, HPCPartition, HPCType
from slurminator.plugins import SimpleCommandPlugin

pytestmark = pytest.mark.unit


def _olivia_config() -> HPCClusterConfig:
    return HPCClusterConfig(
        cluster_type=HPCType.OLIVIA,
        partition=HPCPartition.ACCEL,
        account="acct",
        hostname="olivia.example",
        username="user",
        submission_host="submit.example",
        submission_username="submit-user",
        submission_port=2200,
        submission_use_key=False,
        submission_key_path="~/.ssh/submit",
        submission_two_factor=False,
    )


def test_cli_bootstraps_connection_manager_for_active_limit(tmp_path: Path) -> None:
    exp_file = tmp_path / "exps.yaml"
    exp_file.write_text("experiments: []\n")
    captured = {}

    class FakeConnectionManager:
        def __init__(self, cfgs):
            captured["cfgs"] = cfgs

        def is_local_hpc(self, _hpc):
            return True

        def connect(self, _hpc, force_remote=False):  # noqa: ARG002
            return None

        def run_command(self, _hpc, _cmd, prefer_remote=False):  # noqa: ARG002
            return ("", "")

    run_orchestrator_cli(
        argv=["--yaml", str(exp_file), "--dry-run", "--olivia-limit", "1"],
        launch_guard=lambda: None,
        load_configs=False,
        cluster_configs={HPCType.OLIVIA: _olivia_config()},
        connection_manager_cls=FakeConnectionManager,
    )

    cfg = captured["cfgs"][HPCType.OLIVIA]
    assert cfg.submission_host == "submit.example"
    assert cfg.submission_username == "submit-user"
    assert cfg.submission_port == 2200
    assert cfg.submission_use_key is False
    assert cfg.submission_key_path == "~/.ssh/submit"
    assert cfg.submission_two_factor is False


def test_cli_constructs_orchestrator_with_simple_command_plugin(tmp_path: Path) -> None:
    exp_file = tmp_path / "exps.yaml"
    exp_file.write_text("experiments: []\n")
    captured = {}

    class FakeOrchestrator:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self):
            captured["ran"] = True

    run_orchestrator_cli(
        argv=[
            "--yaml",
            str(exp_file),
            "--simple-command-entrypoint",
            "python train.py",
            "--simple-command-config-arg=--config",
        ],
        launch_guard=lambda: None,
        load_configs=False,
        orchestrator_cls=FakeOrchestrator,
    )

    assert captured["experiment_file"] == str(exp_file)
    assert isinstance(captured["plugin"], SimpleCommandPlugin)
    assert captured["plugin"].entrypoint == "python train.py"
    assert captured["plugin"].config_arg == "--config"
    assert captured["ran"] is True


def test_parse_partition_overrides_accepts_lumi_alias() -> None:
    class Args:
        partition_override = ["FOX=accel_long"]
        lumi_partition = "dev-g"

    parsed = parse_partition_overrides(Args())

    assert parsed == {HPCType.FOX: "accel_long", HPCType.LUMI: "dev-g"}

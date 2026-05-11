from pathlib import Path
import sys
import types

import pytest

from slurminator.cli import discover_plugin, parse_partition_overrides, run_orchestrator_cli
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


def test_cli_constructs_orchestrator_with_simple_command_plugin(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("SLURMINATOR_PLUGIN", raising=False)
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
            "--simple-command-sweep-params-arg=--overrides",
        ],
        launch_guard=lambda: None,
        load_configs=False,
        orchestrator_cls=FakeOrchestrator,
    )

    assert captured["experiment_file"] == str(exp_file)
    assert isinstance(captured["plugin"], SimpleCommandPlugin)
    assert captured["plugin"].entrypoint == "python train.py"
    assert captured["plugin"].config_arg == "--config"
    assert captured["plugin"].sweep_params_arg == "--overrides"
    assert captured["ran"] is True


def test_cli_uses_orchestrator_config_simple_command_defaults(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("SLURMINATOR_PLUGIN", raising=False)
    hpc_file = tmp_path / "hpc_config.yaml"
    hpc_file.write_text(
        "\n".join(
            [
                "clusters:",
                "  OLIVIA:",
                "    partition: accel",
                "    account: acct",
                "    hostname: olivia.example",
                "    username: user",
                "",
            ]
        )
    )
    orchestrator_file = tmp_path / "orchestrator_config.yaml"
    orchestrator_file.write_text(
        "\n".join(
            [
                "command:",
                "  entrypoint: python train.py",
                "  config_arg: --cfg",
                "  sweep_params_arg: --overrides",
                "  extra_args: [--quiet]",
                "",
            ]
        )
    )
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
            "--hpc-config-file",
            str(hpc_file),
            "--orchestrator-config-file",
            str(orchestrator_file),
        ],
        launch_guard=lambda: None,
        orchestrator_cls=FakeOrchestrator,
    )

    assert isinstance(captured["plugin"], SimpleCommandPlugin)
    assert captured["plugin"].entrypoint == "python train.py"
    assert captured["plugin"].config_arg == "--cfg"
    assert captured["plugin"].sweep_params_arg == "--overrides"
    assert captured["plugin"].extra_args == ("--quiet",)
    assert captured["ran"] is True


def test_discover_plugin_loads_env_class(monkeypatch) -> None:
    module = types.ModuleType("fake_slurminator_plugin")

    class FakePlugin:
        def build_commands_line(self, exp, context):  # noqa: ANN001, ARG002
            return "python train.py"

    module.FakePlugin = FakePlugin
    monkeypatch.setitem(sys.modules, module.__name__, module)

    plugin = discover_plugin({"SLURMINATOR_PLUGIN": "fake_slurminator_plugin:FakePlugin"})

    assert isinstance(plugin, FakePlugin)


def test_cli_uses_env_plugin_hooks(monkeypatch, tmp_path: Path) -> None:
    module = types.ModuleType("fake_slurminator_cli_plugin")
    exp_file = tmp_path / "exps.yaml"
    exp_file.write_text("experiments: []\n")
    captured = {}

    class FakeOrchestrator:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self):
            captured["ran"] = True

    class FakePlugin:
        @property
        def orchestrator_cls(self):
            return FakeOrchestrator

        def pre_parse_argv(self, argv):
            captured["pre_parse_argv"] = list(argv)
            return list(argv) + ["--fake-flag", "ok"]

        def extend_parser(self, parser):
            parser.add_argument("--fake-flag")
            return parser

        def prepare_args(self, args):
            captured["fake_flag"] = args.fake_flag

        def configure_from_args(self, args):  # noqa: ARG002
            captured["configured"] = True
            return self

        def build_commands_line(self, exp, context):  # noqa: ANN001, ARG002
            return "python train.py"

        def validate_experiment(self, exp, overrides):  # noqa: ANN001, ARG002
            return False

        def prepare_remote_runtime(self, *, hpc_type, connection_manager):  # noqa: ANN001, ARG002
            return None

        def interpret_log_tail(self, *, exp, log_tail, current_status, stage="pre_heuristics"):  # noqa: ANN001, ARG002
            return None

        def annotate_log_tail(self, *, exp, log_tail):  # noqa: ANN001, ARG002
            return None

    module.FakePlugin = FakePlugin
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setenv("SLURMINATOR_PLUGIN", "fake_slurminator_cli_plugin:FakePlugin")

    run_orchestrator_cli(argv=["--yaml", str(exp_file)], launch_guard=lambda: None, load_configs=False)

    assert captured["pre_parse_argv"] == ["--yaml", str(exp_file)]
    assert captured["fake_flag"] == "ok"
    assert captured["configured"] is True
    assert captured["plugin"].__class__.__name__ == "FakePlugin"
    assert captured["ran"] is True


def test_cli_uses_plugin_default_repo_root_for_config_discovery(monkeypatch, tmp_path: Path) -> None:
    module = types.ModuleType("fake_slurminator_repo_plugin")
    repo = tmp_path / "repo"
    (repo / "user_configs").mkdir(parents=True)
    (repo / "user_configs" / "hpc_config.yaml").write_text(
        "\n".join(
            [
                "clusters:",
                "  OLIVIA:",
                "    partition: accel",
                "    account: acct",
                "    hostname: olivia.example",
                "    username: user",
                "",
            ]
        )
    )
    exp_file = tmp_path / "exps.yaml"
    exp_file.write_text("experiments: []\n")

    class FakeOrchestrator:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run(self):
            return None

    class FakePlugin:
        @property
        def orchestrator_cls(self):
            return FakeOrchestrator

        def default_repo_root(self):
            return repo

        def build_commands_line(self, exp, context):  # noqa: ANN001, ARG002
            return "python train.py"

    module.FakePlugin = FakePlugin
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setenv("SLURMINATOR_PLUGIN", "fake_slurminator_repo_plugin:FakePlugin")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    run_orchestrator_cli(argv=["--yaml", str(exp_file), "--dry-run"], launch_guard=lambda: None)


def test_parse_partition_overrides_accepts_lumi_alias() -> None:
    class Args:
        partition_override = ["FOX=accel_long"]
        lumi_partition = "dev-g"

    parsed = parse_partition_overrides(Args())

    assert parsed == {HPCType.FOX: "accel_long", HPCType.LUMI: "dev-g"}

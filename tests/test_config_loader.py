from pathlib import Path

import pytest

from slurminator.config import (
    HPC_CONFIGS,
    HPCPartition,
    HPCType,
    OrchestratorSettings,
    find_user_config,
    load_user_config,
    parse_orchestrator_settings,
)

pytestmark = pytest.mark.unit


def _write_hpc_config(path: Path, *, account: str = "acct") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "clusters:",
                "  FOX:",
                "    partition: accel",
                f'    account: "{account}"',
                '    hostname: "fox.example"',
                '    username: "user"',
                '    repo_path: "/repo"',
                '    save_path: "/save"',
                '    exclude_nodes: "gpu-1 gpu-2"',
                "",
            ]
        )
    )


def test_load_user_config_finds_home_hpc_and_defaults_orchestrator(tmp_path: Path) -> None:
    home = tmp_path / "home"
    repo = tmp_path / "repo"
    _write_hpc_config(home / ".slurminator_config" / "hpc_config.yaml")

    loaded = load_user_config(repo_root=repo, home=home)

    assert loaded.paths.hpc_config == home / ".slurminator_config" / "hpc_config.yaml"
    assert loaded.paths.orchestrator_config is None
    assert loaded.cluster_configs[HPCType.FOX].partition == HPCPartition.ACCEL
    assert loaded.cluster_configs[HPCType.FOX].exclude_nodes == ["gpu-1", "gpu-2"]
    assert isinstance(loaded.orchestrator, OrchestratorSettings)
    assert loaded.orchestrator.dashboard.timeout_risk.min_runtime_seconds == 15 * 60
    assert HPC_CONFIGS[HPCType.FOX].account == "acct"


def test_find_user_config_prefers_override_when_provided(tmp_path: Path) -> None:
    home = tmp_path / "home"
    repo = tmp_path / "repo"
    home_path = home / ".slurminator_config" / "hpc_config.yaml"
    override_path = tmp_path / "override.yaml"
    _write_hpc_config(home_path)
    _write_hpc_config(override_path, account="override")

    found = find_user_config("hpc_config.yaml", override_path=override_path, repo_root=repo, home=home)

    assert found == override_path


def test_load_user_config_accepts_env_overrides(tmp_path: Path) -> None:
    home = tmp_path / "home"
    repo = tmp_path / "repo"
    hpc_path = tmp_path / "env_hpc.yaml"
    orchestrator_path = tmp_path / "env_orchestrator.yaml"
    _write_hpc_config(hpc_path, account="env-acct")
    (orchestrator_path).write_text("dashboard:\n  ui_version: v2\n")

    loaded = load_user_config(
        home=home,
        repo_root=repo,
        env={
            "SLURMINATOR_HPC_CONFIG_FILE": str(hpc_path),
            "SLURMINATOR_ORCHESTRATOR_CONFIG_FILE": str(orchestrator_path),
        },
    )

    assert loaded.paths.hpc_config == hpc_path
    assert loaded.paths.orchestrator_config == orchestrator_path
    assert loaded.cluster_configs[HPCType.FOX].account == "env-acct"
    assert loaded.orchestrator.dashboard.ui_version == "v2"


def test_load_user_config_uses_env_repo_root_for_repo_configs(tmp_path: Path) -> None:
    home = tmp_path / "home"
    repo = tmp_path / "repo"
    _write_hpc_config(repo / "user_configs" / "hpc_config.yaml", account="repo-env")

    loaded = load_user_config(home=home, env={"SLURMINATOR_REPO_ROOT": str(repo)})

    assert loaded.paths.hpc_config == repo / "user_configs" / "hpc_config.yaml"
    assert loaded.cluster_configs[HPCType.FOX].account == "repo-env"


def test_find_user_config_keeps_legacy_home_fallback(tmp_path: Path) -> None:
    home = tmp_path / "home"
    repo = tmp_path / "repo"
    legacy_path = home / ".slurminator" / "hpc_config.yaml"
    _write_hpc_config(legacy_path)

    found = find_user_config("hpc_config.yaml", repo_root=repo, home=home)

    assert found == legacy_path


def test_load_user_config_uses_repo_orchestrator_override(tmp_path: Path) -> None:
    home = tmp_path / "home"
    repo = tmp_path / "repo"
    _write_hpc_config(repo / "user_configs" / "hpc_config.yaml")
    (repo / "user_configs" / "orchestrator_config.yaml").write_text(
        "\n".join(
            [
                "dashboard:",
                "  ui_version: v2",
                "  poll_interval_seconds: 7",
                "  timeout_risk:",
                "    min_progress: 0.25",
                "    min_runtime_minutes: 20",
                "  sparkline:",
                "    flat_slope_norm: 0.02",
                "    directional_slope_norm: 0.05",
                "    oscillation_residual_norm: 0.2",
                "retry:",
                "  retry_timeout_with_estimated_time: true",
                "  timeout_retry_buffer: 1.5",
                "  timeout_retry_max_attempts: 2",
                "polling:",
                "  status_file_stale_threshold_seconds: 45",
                "",
            ]
        )
    )

    loaded = load_user_config(repo_root=repo, home=home)

    assert loaded.paths.hpc_config == repo / "user_configs" / "hpc_config.yaml"
    assert loaded.paths.orchestrator_config == repo / "user_configs" / "orchestrator_config.yaml"
    assert loaded.orchestrator.dashboard.ui_version == "v2"
    assert loaded.orchestrator.dashboard.poll_interval_seconds == 7
    assert loaded.orchestrator.dashboard.timeout_risk.min_progress == pytest.approx(0.25)
    assert loaded.orchestrator.dashboard.timeout_risk.min_runtime_seconds == 20 * 60
    assert loaded.orchestrator.dashboard.sparkline.flat_slope_norm == pytest.approx(0.02)
    assert loaded.orchestrator.dashboard.sparkline.directional_slope_norm == pytest.approx(0.05)
    assert loaded.orchestrator.dashboard.sparkline.oscillation_residual_norm == pytest.approx(0.2)
    assert loaded.orchestrator.retry.retry_timeout_with_estimated_time is True
    assert loaded.orchestrator.retry.timeout_retry_buffer == pytest.approx(1.5)
    assert loaded.orchestrator.retry.timeout_retry_max_attempts == 2
    assert loaded.orchestrator.polling.status_file_stale_threshold_seconds == 45


def test_missing_required_hpc_config_fails_with_template_hint(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="hpc_config.example.yaml"):
        load_user_config(repo_root=tmp_path / "repo", home=tmp_path / "home")


def test_parse_embedded_orchestrator_config_clamps_timeout_values() -> None:
    settings = parse_orchestrator_settings(
        {
            "orchestrator": {
                "dashboard": {
                    "timeout_risk": {"min_progress": 9, "min_runtime_seconds": 0, "medium_ratio": -1, "high_ratio": -2},
                    "sparkline": {"flat_slope_norm": -1, "directional_slope_norm": -2, "oscillation_residual_norm": -3},
                }
            }
        }
    )

    assert settings.dashboard.timeout_risk.min_progress == 1.0
    assert settings.dashboard.timeout_risk.min_runtime_seconds == 1
    assert settings.dashboard.timeout_risk.medium_ratio == 0.0
    assert settings.dashboard.timeout_risk.high_ratio == 0.0
    assert settings.dashboard.sparkline.flat_slope_norm == 0.0
    assert settings.dashboard.sparkline.directional_slope_norm == 0.0
    assert settings.dashboard.sparkline.oscillation_residual_norm == 0.0


def test_parse_command_settings_from_orchestrator_config() -> None:
    settings = parse_orchestrator_settings(
        {
            "command": {
                "entrypoint": "python train.py",
                "config_arg": "--cfg",
                "sweep_params_arg": "--overrides",
                "extra_args": ["--quiet"],
                "orchestrator_flag": "--from-slurminator",
                "multi_experiment_flag": "--multi",
            }
        }
    )

    assert settings.command.entrypoint == "python train.py"
    assert settings.command.config_arg == "--cfg"
    assert settings.command.sweep_params_arg == "--overrides"
    assert settings.command.extra_args == ("--quiet",)
    assert settings.command.orchestrator_flag == "--from-slurminator"
    assert settings.command.multi_experiment_flag == "--multi"

from pathlib import Path

import pytest

from slurminator.experiments import CustomSweepCase, CustomSweepConfig, MasterExperimentConfig

pytestmark = pytest.mark.unit


def test_custom_sweep_dataset_name_alias_normalizes_to_datasets() -> None:
    config = CustomSweepConfig(dataset_name="HAR")

    assert config.datasets == ["HAR"]


def test_custom_sweep_cases_accept_yaml_dicts() -> None:
    config = CustomSweepConfig(cases=[{"name": "case_a", "overrides": {"seed": 42}}])

    assert config.cases == [CustomSweepCase(name="case_a", overrides={"seed": 42})]


def test_master_config_rejects_custom_sweep_flag_without_sweeps() -> None:
    with pytest.raises(ValueError, match="run_custom_sweeps=True"):
        MasterExperimentConfig(run_custom_sweeps=True)


def test_master_config_normalizes_custom_sweep_dicts() -> None:
    config = MasterExperimentConfig(
        run_custom_sweeps=True, custom_sweeps=[{"dataset_name": "HAR", "cases": [{"name": "baseline"}]}]
    )

    assert isinstance(config.custom_sweeps[0], CustomSweepConfig)
    assert config.custom_sweeps[0].datasets == ["HAR"]
    assert config.custom_sweeps[0].cases == [CustomSweepCase(name="baseline")]


def test_master_config_from_yaml(tmp_path: Path) -> None:
    path = tmp_path / "master.yaml"
    path.write_text(
        "\n".join(
            [
                "run_custom_sweeps: true",
                "seed: 123",
                "custom_sweeps:",
                "  - dataset_name: HAR",
                "    experiment_prefix: demo",
                "    sweep_keys:",
                "      seed: [1, 2]",
                "",
            ]
        )
    )

    config = MasterExperimentConfig.from_yaml(path)

    assert config.seed == 123
    assert config.custom_sweeps[0].experiment_prefix == "demo"
    assert config.custom_sweeps[0].datasets == ["HAR"]
    assert config.custom_sweeps[0].sweep_keys == {"seed": [1, 2]}

from pathlib import Path

import pytest
import yaml

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
                "experiment_name: HAR encoder comparison",
                "experiment_description: Compare encoder variants on HAR.",
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
    assert config.experiment_name == "HAR encoder comparison"
    assert config.experiment_description == "Compare encoder variants on HAR."
    assert config.custom_sweeps[0].experiment_prefix == "demo"
    assert config.custom_sweeps[0].datasets == ["HAR"]
    assert config.custom_sweeps[0].sweep_keys == {"seed": [1, 2]}


@pytest.mark.parametrize("text", [None, "", " \n\t "])
def test_blank_experiment_info_is_optional(text: str | None) -> None:
    config = MasterExperimentConfig(experiment_name=text, experiment_description=text)

    assert config.experiment_name is None
    assert config.experiment_description is None


@pytest.mark.parametrize("field_name", ["experiment_name", "experiment_description"])
@pytest.mark.parametrize("value", [False, 0, 1.5, ["title"], {"text": "title"}])
def test_master_config_rejects_non_text_experiment_info(field_name: str, value: object) -> None:
    with pytest.raises(ValueError, match=f"{field_name} must be a string or null"):
        MasterExperimentConfig(**{field_name: value})


@pytest.mark.parametrize(
    "info",
    [
        {},
        {"experiment_name": "  HAR\n encoder   comparison  "},
        {"experiment_description": "  Compare encoders on HAR.\n\nKeep [dataset] labels literal.  "},
        {"experiment_name": "HAR encoders", "experiment_description": "Compare encoder variants on HAR."},
    ],
)
def test_sweep_file_loads_optional_experiment_info(tmp_path: Path, info: dict[str, str]) -> None:
    path = tmp_path / "sweep.yaml"
    path.write_text(
        yaml.safe_dump({**info, "custom_sweeps": [{"dataset_name": "HAR", "cases": [{"name": "baseline"}]}]}),
        encoding="utf-8",
    )

    config = MasterExperimentConfig.from_sweep_yaml(path, num_epochs=5, config_profile="baseline")

    assert config.run_custom_sweeps is True
    assert config.custom_sweeps[0].datasets == ["HAR"]
    assert config.custom_sweeps[0].cases == [CustomSweepCase(name="baseline")]
    assert config.num_epochs == 5
    assert config.config_profile == "baseline"
    expected_name = " ".join(info["experiment_name"].split()) if "experiment_name" in info else None
    assert config.experiment_name == expected_name
    expected_description = info["experiment_description"].strip() if "experiment_description" in info else None
    assert config.experiment_description == expected_description


@pytest.mark.parametrize("content", ["[]", "custom_sweeps: invalid", "custom_sweeps: []"])
def test_sweep_file_rejects_invalid_sweep_structure(tmp_path: Path, content: str) -> None:
    path = tmp_path / "sweep.yaml"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError):
        MasterExperimentConfig.from_sweep_yaml(path)

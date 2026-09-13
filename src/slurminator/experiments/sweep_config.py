"""Sweep and master experiment-generation config dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from slurminator.experiments.experiment_info import ExperimentInfo
from slurminator.experiments.yaml_utils import load_yaml

DEFAULT_SEED = 42


@dataclass
class CustomSweepCase:
    """A named set of overrides to run as one experiment variant."""

    name: str = "case"
    overrides: dict[str, Any] = field(default_factory=dict)
    base_overrides: dict[str, Any] = field(default_factory=dict)
    resume_from: str | dict[str, str] | None = None
    checkpoint_probe: bool | None = None
    checkpoint_probe_epoch_offset: int | None = None


@dataclass
class CustomSweepConfig:
    """Parameters for custom sweeps that start from a base config."""

    sweep_keys: dict[str, list[int | float | str]] = field(default_factory=dict)
    datasets: list[str] | None = None
    dataset_name: str | None = None
    base_overrides: dict[str, Any] = field(default_factory=dict)
    cases: list[CustomSweepCase] = field(default_factory=list)
    num_epochs: int | None = None
    task_type: str = "self_supervised"
    experiment_prefix: str = "abl"
    wandb_project: str | None = None
    config_profile: str | None = None
    run_name_prefix: str | None = None
    run_name_suffix: str | None = None
    parameters_prefix: dict[str, str] = field(default_factory=dict)
    seeds: list[int] | None = None
    num_seeds: int | None = None
    resume_from: str | dict[str, str] | None = None
    checkpoint_probe: bool = False
    checkpoint_probe_epoch_offset: int = 1

    def __post_init__(self) -> None:
        if self.dataset_name and not self.datasets:
            self.datasets = [self.dataset_name]

        normalized_cases: list[CustomSweepCase] = []
        for case in self.cases:
            if isinstance(case, CustomSweepCase):
                normalized_cases.append(case)
            elif isinstance(case, dict):
                normalized_cases.append(CustomSweepCase(**case))
        if normalized_cases:
            self.cases = normalized_cases


@dataclass
class MasterExperimentConfig:
    """Master configuration for experiment generation."""

    experiment_name: str | None = field(default=None, kw_only=True)
    experiment_description: str | None = field(default=None, kw_only=True)

    run_custom_sweeps: bool = False
    run_wandb_sweep: bool = False
    custom_sweeps: list[CustomSweepConfig] = field(default_factory=list)

    seed: int = DEFAULT_SEED
    debug: bool = False
    seeds: list[int] = field(default_factory=lambda: [42, 45, 47])
    dataset_seed_overrides: dict[str, list[int]] = field(default_factory=dict)

    num_epochs: int | None = None
    config_profile: str | None = None

    def __post_init__(self) -> None:
        info = ExperimentInfo(name=self.experiment_name, description=self.experiment_description)
        self.experiment_name = info.name
        self.experiment_description = info.description
        self.custom_sweeps = [
            sweep if isinstance(sweep, CustomSweepConfig) else CustomSweepConfig(**sweep)
            for sweep in self.custom_sweeps
        ]
        if self.run_custom_sweeps and not self.custom_sweeps:
            raise ValueError("run_custom_sweeps=True but no custom_sweeps config provided!")

    @staticmethod
    def from_yaml(yaml_path: str | Path) -> "MasterExperimentConfig":
        """Load a master experiment config from YAML."""
        with Path(yaml_path).open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"{yaml_path} must contain a YAML mapping.")
        return MasterExperimentConfig(**raw)

    @staticmethod
    def from_sweep_yaml(
        yaml_path: str | Path, *, num_epochs: int | None = None, config_profile: str | None = None
    ) -> MasterExperimentConfig:
        """Load a CLI sweep file, including context for the whole experiment list."""
        data = load_yaml(yaml_path)
        if not isinstance(data, dict):
            raise ValueError(f"{yaml_path} must contain a YAML mapping.")
        sweeps_raw = data.get("custom_sweeps", data)
        if isinstance(sweeps_raw, dict):
            sweeps_raw = sweeps_raw.get("custom_sweeps", [])
        if not isinstance(sweeps_raw, list):
            raise ValueError("custom_sweeps must be a list in the provided YAML file.")
        return MasterExperimentConfig(
            run_custom_sweeps=True,
            custom_sweeps=sweeps_raw,
            num_epochs=num_epochs,
            config_profile=config_profile,
            experiment_name=data.get("experiment_name"),
            experiment_description=data.get("experiment_description"),
        )


__all__ = ["DEFAULT_SEED", "CustomSweepCase", "CustomSweepConfig", "MasterExperimentConfig"]

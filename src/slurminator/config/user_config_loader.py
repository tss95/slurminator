"""User-facing config file discovery and loading."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from slurminator.config.cluster_registry import HPCClusterConfig, HPCType, parse_cluster_configs, set_cluster_configs
from slurminator.config.orchestrator_config import OrchestratorSettings, parse_orchestrator_settings

HPC_CONFIG_FILE = "hpc_config.yaml"
ORCHESTRATOR_CONFIG_FILE = "orchestrator_config.yaml"
USER_CONFIG_DIR = ".slurminator_config"
LEGACY_USER_CONFIG_DIR = ".slurminator"
HPC_CONFIG_FILE_ENV = "SLURMINATOR_HPC_CONFIG_FILE"
ORCHESTRATOR_CONFIG_FILE_ENV = "SLURMINATOR_ORCHESTRATOR_CONFIG_FILE"
REPO_ROOT_ENV = "SLURMINATOR_REPO_ROOT"


@dataclass(frozen=True)
class UserConfigPaths:
    """Resolved user config file paths."""

    hpc_config: Path
    orchestrator_config: Path | None = None


@dataclass(frozen=True)
class LoadedUserConfig:
    """Loaded user-facing Slurminator configuration."""

    cluster_configs: dict[HPCType, HPCClusterConfig]
    orchestrator: OrchestratorSettings
    paths: UserConfigPaths


def load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping from ``path``."""
    resolved = Path(path).expanduser()
    with resolved.open("r") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{resolved} must contain a YAML mapping.")
    return data


def find_user_config(
    file_name: str,
    *,
    override_path: str | Path | None = None,
    repo_root: str | Path | None = None,
    home: str | Path | None = None,
    required: bool = False,
) -> Path | None:
    """Find a user config using Slurminator's documented search order."""
    if override_path is not None:
        path = Path(override_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Config override path does not exist: {path}")
        return path

    home_root = Path(home).expanduser() if home is not None else Path.home()
    candidates = [home_root / USER_CONFIG_DIR / file_name, home_root / LEGACY_USER_CONFIG_DIR / file_name]
    if repo_root is not None:
        candidates.append(Path(repo_root).expanduser() / "user_configs" / file_name)
    else:
        candidates.append(Path.cwd() / "user_configs" / file_name)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    if required:
        searched = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(
            f"Required {file_name} was not found. Searched: {searched}. "
            "Copy scripts/templates/hpc_config.example.yaml to ~/.slurminator_config/hpc_config.yaml and edit it."
        )
    return None


def load_hpc_config_file(path: str | Path) -> dict[HPCType, HPCClusterConfig]:
    """Load cluster configs from a required ``hpc_config.yaml`` file."""
    raw = load_yaml_mapping(path)
    return parse_cluster_configs(raw, source=path)


def load_orchestrator_config_file(path: str | Path | None) -> OrchestratorSettings:
    """Load optional orchestrator settings, returning defaults when absent."""
    if path is None:
        return OrchestratorSettings()
    raw = load_yaml_mapping(path)
    return parse_orchestrator_settings(raw, source=path)


def load_user_config(
    *,
    hpc_config_file: str | Path | None = None,
    orchestrator_config_file: str | Path | None = None,
    repo_root: str | Path | None = None,
    home: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    update_global_registry: bool = True,
) -> LoadedUserConfig:
    """Load the two user-facing Slurminator YAML files."""
    env_map = os.environ if env is None else env
    hpc_config_file = hpc_config_file or env_map.get(HPC_CONFIG_FILE_ENV)
    orchestrator_config_file = orchestrator_config_file or env_map.get(ORCHESTRATOR_CONFIG_FILE_ENV)
    repo_root = repo_root or env_map.get(REPO_ROOT_ENV)

    hpc_path = find_user_config(
        HPC_CONFIG_FILE, override_path=hpc_config_file, repo_root=repo_root, home=home, required=True
    )
    assert hpc_path is not None
    orchestrator_path = find_user_config(
        ORCHESTRATOR_CONFIG_FILE, override_path=orchestrator_config_file, repo_root=repo_root, home=home, required=False
    )

    cluster_configs = load_hpc_config_file(hpc_path)
    orchestrator = load_orchestrator_config_file(orchestrator_path)
    if update_global_registry:
        set_cluster_configs(cluster_configs)
    return LoadedUserConfig(
        cluster_configs=cluster_configs,
        orchestrator=orchestrator,
        paths=UserConfigPaths(hpc_config=hpc_path, orchestrator_config=orchestrator_path),
    )


__all__ = [
    "HPC_CONFIG_FILE",
    "HPC_CONFIG_FILE_ENV",
    "LEGACY_USER_CONFIG_DIR",
    "LoadedUserConfig",
    "ORCHESTRATOR_CONFIG_FILE",
    "ORCHESTRATOR_CONFIG_FILE_ENV",
    "REPO_ROOT_ENV",
    "USER_CONFIG_DIR",
    "UserConfigPaths",
    "find_user_config",
    "load_hpc_config_file",
    "load_orchestrator_config_file",
    "load_user_config",
    "load_yaml_mapping",
]

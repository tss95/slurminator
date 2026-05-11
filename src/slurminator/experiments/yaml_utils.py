"""YAML helpers for orchestrator experiment files."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from slurminator.config.cluster_registry import HPCPartition, HPCType
from slurminator.experiments.status_enum import ExperimentStatus

try:
    from yaml.scalarstring import LiteralScalarString  # type: ignore
    from yaml.scalarstring import SingleQuotedScalarString  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - older PyYAML shim

    class LiteralScalarString(str):
        """Shim for PyYAML versions without scalarstring helpers."""

    def _literal_scalar_representer(dumper: yaml.Dumper, data: str) -> yaml.Node:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")

    yaml.add_representer(LiteralScalarString, _literal_scalar_representer)
    yaml.representer.SafeRepresenter.add_representer(LiteralScalarString, _literal_scalar_representer)

    class SingleQuotedScalarString(str):
        """Shim that forces single-quoted YAML scalar style."""

    def _single_quoted_representer(dumper: yaml.Dumper, data: str) -> yaml.Node:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="'")

    yaml.add_representer(SingleQuotedScalarString, _single_quoted_representer)
    yaml.representer.SafeRepresenter.add_representer(SingleQuotedScalarString, _single_quoted_representer)


class NoAliasDumper(yaml.Dumper):
    """YAML dumper that avoids aliases for repeated nodes."""

    def ignore_aliases(self, data: Any) -> bool:
        return True


class ExperimentYAMLLoader(yaml.SafeLoader):
    """YAML loader with orchestrator enum support."""


class ExperimentYAMLDumper(NoAliasDumper):
    """YAML dumper with orchestrator enum support."""


def register_yaml_enum(enum_type: type[Enum], tag: str) -> None:
    """Register one enum type for experiment YAML load/dump."""

    def constructor(loader: yaml.Loader, node: yaml.Node) -> Enum:
        value = loader.construct_scalar(node)
        return enum_type(value)

    def representer(dumper: yaml.Dumper, data: Enum) -> yaml.Node:
        return dumper.represent_scalar(tag, str(data.value))

    ExperimentYAMLLoader.add_constructor(tag, constructor)
    ExperimentYAMLDumper.add_representer(enum_type, representer)


register_yaml_enum(ExperimentStatus, "!ExperimentStatus")
register_yaml_enum(HPCType, "!HPCType")
register_yaml_enum(HPCPartition, "!HPCPartition")


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file with registered custom type handling."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.load(handle, Loader=ExperimentYAMLLoader)


def dump_yaml(data: dict[str, Any], path: str | Path) -> None:
    """Atomically dump YAML with registered custom type handling."""
    target = Path(path)
    tmp_fd, tmp_name = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as handle:
            yaml.dump(
                _wrap_multiline(data),
                handle,
                Dumper=ExperimentYAMLDumper,
                sort_keys=False,
                default_flow_style=False,
                allow_unicode=True,
                width=4096,
            )

        if target.exists():
            backup_path = target.with_suffix(target.suffix + ".bak")
            try:
                shutil.copy2(target, backup_path)
            except Exception:
                pass

        os.replace(tmp_name, target)
    finally:
        if os.path.exists(tmp_name):
            try:
                os.unlink(tmp_name)
            except Exception:
                pass


def resolve_sweep_yaml_path(
    dataset_name: str,
    *,
    root: str | Path = ".",
    env_var: str = "SLURMINATOR_SWEEP_YAML",
    legacy_env_var: str | None = None,
    exists: Callable[[Path], bool] | None = None,
) -> Path:
    """Return the sweep-configuration YAML to use for one dataset."""
    path_exists = exists or Path.exists
    for name in (env_var, legacy_env_var):
        if not name:
            continue
        override = os.environ.get(name)
        if override:
            candidate = Path(override)
            if path_exists(candidate):
                return candidate

    default_path = Path(root) / "cfg" / "sweep" / dataset_name / "sweep_config.yaml"
    if path_exists(default_path):
        return default_path

    raise FileNotFoundError(f"No sweep configuration found for dataset '{dataset_name}'. Checked: {default_path}")


def _wrap_multiline(obj: object) -> object:
    if isinstance(obj, str) and "\n" in obj:
        return LiteralScalarString(obj)
    if isinstance(obj, list):
        return [_wrap_multiline(item) for item in obj]
    if isinstance(obj, dict):
        return {key: _wrap_multiline(value) for key, value in obj.items()}
    return obj


__all__ = [
    "ExperimentYAMLDumper",
    "ExperimentYAMLLoader",
    "LiteralScalarString",
    "NoAliasDumper",
    "SingleQuotedScalarString",
    "dump_yaml",
    "load_yaml",
    "register_yaml_enum",
    "resolve_sweep_yaml_path",
]

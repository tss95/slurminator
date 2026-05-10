"""Config-driven experiment policy helpers.

These helpers intentionally operate on plain experiment-row dictionaries and
cluster config objects. They replace project-plugin hooks for behavior that is
better expressed as data: dataset pinning, per-dataset resources, extra remote
directories, and per-cluster sbatch environment variables.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def resolve_pinned_hpc(exp: Mapping[str, Any], cluster_configs: Mapping[Any, Any]) -> Any | None:
    """Return the HPC an experiment is pinned to, if configured."""
    row_pin = _first_present(exp, ("pinned_hpc", "required_hpc"))
    if row_pin is not None:
        return _coerce_hpc_key(row_pin, cluster_configs)

    dataset = _dataset_name(exp)
    if not dataset:
        return None

    for hpc_type, cluster_config in cluster_configs.items():
        pinned_datasets = getattr(cluster_config, "pinned_datasets", None) or ()
        if dataset in {str(item) for item in pinned_datasets}:
            return hpc_type
    return None


def resolve_resource_overrides(
    exp: Mapping[str, Any], *, hpc_type: Any | None = None, cluster_configs: Mapping[Any, Any] | None = None
) -> dict[str, Any]:
    """Return merged resource overrides from cluster dataset config and the row."""
    merged: dict[str, Any] = {}
    dataset = _dataset_name(exp)
    if dataset and hpc_type is not None and cluster_configs is not None:
        cluster_config = cluster_configs.get(hpc_type)
        dataset_overrides = getattr(cluster_config, "dataset_resource_overrides", None) or {}
        raw = dataset_overrides.get(dataset)
        if isinstance(raw, Mapping):
            merged.update(_normalise_resource_override_keys(raw))

    raw_row = _first_present(exp, ("resource_overrides", "resources"))
    if isinstance(raw_row, Mapping):
        merged.update(_normalise_resource_override_keys(raw_row))
    return merged


def resolve_sbatch_export_vars(cluster_config: Any) -> dict[str, str]:
    """Return per-cluster environment variables for ``sbatch --export``."""
    raw = getattr(cluster_config, "sbatch_env", None) or {}
    if not isinstance(raw, Mapping):
        raise TypeError("cluster_config.sbatch_env must be a mapping when set.")

    context = _format_context(cluster_config)
    exports: dict[str, str] = {}
    for key, value in raw.items():
        key_text = str(key).strip()
        if not key_text:
            raise ValueError("cluster_config.sbatch_env contains an empty variable name.")
        try:
            exports[key_text] = str(value).format(**context)
        except KeyError as exc:
            raise ValueError(f"Unknown sbatch_env placeholder {exc.args[0]!r} for {key_text}.") from exc
    return exports


def resolve_extra_remote_dirs(exp: Mapping[str, Any], *, base_path: Path) -> tuple[Path, ...]:
    """Return extra directories that should exist before submitting ``exp``."""
    raw = _first_present(exp, ("ensure_dirs", "extra_remote_dirs"))
    if raw is None:
        return ()
    if isinstance(raw, str):
        entries = [raw]
    elif isinstance(raw, list | tuple):
        entries = list(raw)
    else:
        raise TypeError("ensure_dirs must be a string or sequence of strings.")

    context = {"base_path": str(base_path), **{key: str(value) for key, value in exp.items() if value is not None}}
    dirs: list[Path] = []
    for entry in entries:
        text = str(entry).strip()
        if not text:
            continue
        try:
            formatted = text.format(**context)
        except KeyError as exc:
            raise ValueError(f"Unknown ensure_dirs placeholder {exc.args[0]!r} in {text!r}.") from exc
        path = Path(formatted)
        dirs.append(path if path.is_absolute() else base_path / path)
    return tuple(dirs)


def _dataset_name(exp: Mapping[str, Any]) -> str | None:
    value = exp.get("dataset_name")
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _first_present(exp: Mapping[str, Any], keys: tuple[str, ...]) -> Any | None:
    for key in keys:
        value = exp.get(key)
        if value is not None:
            return value
    metadata = exp.get("metadata")
    if isinstance(metadata, Mapping):
        for key in keys:
            value = metadata.get(key)
            if value is not None:
                return value
    return None


def _coerce_hpc_key(value: Any, cluster_configs: Mapping[Any, Any]) -> Any:
    text = str(value).strip()
    if not text:
        raise ValueError("pinned_hpc cannot be blank.")
    for key in cluster_configs:
        if (
            value == key
            or text == str(key)
            or text == getattr(key, "name", None)
            or text == getattr(key, "value", None)
        ):
            return key
    return value


def _normalise_resource_override_keys(raw: Mapping[str, Any]) -> dict[str, Any]:
    aliases = {"memory_gb": "mem_gb", "gpu_memory_gb": "mem_per_gpu_gb", "gpu_mem_gb": "mem_per_gpu_gb"}
    return {aliases.get(str(key), str(key)): value for key, value in raw.items()}


def _format_context(cluster_config: Any) -> dict[str, Any]:
    if is_dataclass(cluster_config):
        raw = asdict(cluster_config)
    else:
        raw = dict(getattr(cluster_config, "__dict__", {}))
    for key in dir(cluster_config):
        if key.startswith("_") or key in raw:
            continue
        try:
            value = getattr(cluster_config, key)
        except Exception:
            continue
        if not callable(value):
            raw[key] = value
    return raw


__all__ = [
    "resolve_extra_remote_dirs",
    "resolve_pinned_hpc",
    "resolve_resource_overrides",
    "resolve_sbatch_export_vars",
]

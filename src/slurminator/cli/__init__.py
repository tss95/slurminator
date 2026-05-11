"""Command-line helpers and entry points for slurminator."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from slurminator.cli.override_parser import parse_override_list


def main(argv: Sequence[str] | None = None) -> None:
    """Run the generic Slurminator CLI."""
    from slurminator.cli.orchestrator import main as _main

    _main(argv)


def build_base_parser() -> Any:
    """Build Slurminator's generic orchestrator parser."""
    from slurminator.cli.orchestrator import build_base_parser as _build_base_parser

    return _build_base_parser()


def build_concurrency_limits(args: Any) -> Any:
    """Return per-cluster concurrency limits from parsed CLI arguments."""
    from slurminator.cli.orchestrator import build_concurrency_limits as _build_concurrency_limits

    return _build_concurrency_limits(args)


def discover_plugin(*args: Any, **kwargs: Any) -> Any:
    """Load the plugin declared by ``SLURMINATOR_PLUGIN``, if configured."""
    from slurminator.cli.orchestrator import discover_plugin as _discover_plugin

    return _discover_plugin(*args, **kwargs)


def generate_experiment_yaml_from_flags(args: Any, **kwargs: Any) -> str:
    """Generate an experiment YAML from generic custom-sweep CLI flags."""
    from slurminator.cli.orchestrator import generate_experiment_yaml_from_flags as _generate

    return _generate(args, **kwargs)


def parse_partition_overrides(args: Any) -> Any:
    """Return per-cluster partition overrides from parsed CLI arguments."""
    from slurminator.cli.orchestrator import parse_partition_overrides as _parse_partition_overrides

    return _parse_partition_overrides(args)


def run_orchestrator_cli(*args: Any, **kwargs: Any) -> None:
    """Run orchestration from CLI arguments with optional project hooks."""
    from slurminator.cli.orchestrator import run_orchestrator_cli as _run_orchestrator_cli

    _run_orchestrator_cli(*args, **kwargs)


__all__ = [
    "build_base_parser",
    "build_concurrency_limits",
    "discover_plugin",
    "generate_experiment_yaml_from_flags",
    "main",
    "parse_override_list",
    "parse_partition_overrides",
    "run_orchestrator_cli",
]

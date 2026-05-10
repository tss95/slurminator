"""Plugin hooks for project-specific orchestrator behavior."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from slurminator.schemas.status_schema import OrchestratorStatus
from slurminator.status_projection import project_status_to_experiment, status_projection_fields


@dataclass(frozen=True)
class CommandBuildContext:
    """Context passed to project plugins when building a job command."""

    gpus: int
    hpc_type: Any
    multi_experiment: bool = False
    runtime_options: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class OrchestratorPlugin(Protocol):
    """Extension points needed by the reusable HPC orchestrator core.

    The package owns scheduling, polling, YAML persistence, and status-file
    reading. Project adapters own command-line semantics, tracker integration,
    project-specific validation, and compatibility row fields.
    """

    def validate_experiment(self, exp: Mapping[str, Any], overrides: Mapping[str, Any]) -> bool:
        """Validate one experiment before submission.

        Return ``True`` when a meaningful override validation was performed so
        callers can report a count. Return ``False`` for no-op validation.
        """

    def build_commands_line(self, exp: Mapping[str, Any], context: CommandBuildContext) -> str:
        """Return the shell command executed by the cluster job wrapper."""

    def prepare_remote_runtime(self, *, hpc_type: Any, connection_manager: Any) -> None:
        """Prepare optional project runtime services after connections exist."""

    def pinned_hpc_for_experiment(self, exp: Mapping[str, Any]) -> Any | None:
        """Return a required HPC assignment for ``exp``, or ``None``."""

    def resource_overrides_for_experiment(self, exp: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return project-specific resource overrides for ``exp``."""

    def sbatch_export_vars(self, *, hpc_type: Any, cluster_config: Any) -> Mapping[str, str]:
        """Return extra environment variables to include in ``sbatch --export``."""

    def extra_remote_dirs(self, *, base_path: Path, experiment_file: Path) -> Sequence[Path]:
        """Return project-specific directories to ensure before submission."""

    def interpret_log_tail(
        self, *, exp: Mapping[str, Any], log_tail: str, current_status: Any, stage: str = "pre_heuristics"
    ) -> Any | None:
        """Return a project-specific terminal status override from job logs."""

    def status_projection_options(self) -> Mapping[str, Any]:
        """Return options for projecting target status files into experiment rows."""

    def status_projection_fields(self) -> tuple[str, ...]:
        """Return experiment-row fields that may be written during projection."""

    def project_status_to_experiment(self, exp: dict[str, Any], status: OrchestratorStatus) -> set[str]:
        """Project one target status payload into ``exp``."""

    def extract_display_metrics(self, exp: Mapping[str, Any]) -> dict[str, Any]:
        """Return display-friendly metric shortforms for a dashboard row."""


class DefaultOrchestratorPlugin:
    """No-op plugin suitable for package tests and simple adopters."""

    def validate_experiment(self, exp: Mapping[str, Any], overrides: Mapping[str, Any]) -> bool:
        """Perform no project-specific validation."""
        return False

    def build_commands_line(self, exp: Mapping[str, Any], context: CommandBuildContext) -> str:
        """Reject submission unless an adopter supplies command construction."""
        raise NotImplementedError("A project plugin must implement build_commands_line().")

    def prepare_remote_runtime(self, *, hpc_type: Any, connection_manager: Any) -> None:
        """No-op runtime preparation."""

    def pinned_hpc_for_experiment(self, exp: Mapping[str, Any]) -> Any | None:
        """Return no pinned HPC by default."""
        return None

    def resource_overrides_for_experiment(self, exp: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return no resource overrides by default."""
        return {}

    def sbatch_export_vars(self, *, hpc_type: Any, cluster_config: Any) -> Mapping[str, str]:
        """Return no extra sbatch exports by default."""
        return {}

    def extra_remote_dirs(self, *, base_path: Path, experiment_file: Path) -> Sequence[Path]:
        """Return no extra remote directories by default."""
        return ()

    def interpret_log_tail(
        self, *, exp: Mapping[str, Any], log_tail: str, current_status: Any, stage: str = "pre_heuristics"
    ) -> Any | None:
        """Return no log-derived status override by default."""
        return None

    def status_projection_options(self) -> Mapping[str, Any]:
        """Return package-default status projection options."""
        return {}

    def status_projection_fields(self) -> tuple[str, ...]:
        """Return package-default projected row fields."""
        return status_projection_fields(**self.status_projection_options())

    def project_status_to_experiment(self, exp: dict[str, Any], status: OrchestratorStatus) -> set[str]:
        """Project target status using package-default row fields."""
        return project_status_to_experiment(exp, status, **self.status_projection_options())

    def extract_display_metrics(self, exp: Mapping[str, Any]) -> dict[str, Any]:
        """Extract display shortforms declared in target-schema display metadata."""
        display_metrics: dict[str, Any] = {}
        all_metrics = exp.get("all_metrics", {})
        metric_info = exp.get("display_metric_info", {})
        if not isinstance(all_metrics, Mapping) or not isinstance(metric_info, Mapping):
            return display_metrics

        for metric_key, info in metric_info.items():
            if not isinstance(info, Mapping):
                continue
            shortform = info.get("shortform")
            if not shortform:
                continue
            if shortform in exp:
                display_metrics[str(shortform)] = exp[shortform]  # type: ignore[index]
            elif metric_key in all_metrics:
                display_metrics[str(shortform)] = all_metrics[metric_key]
            elif shortform in all_metrics:
                display_metrics[str(shortform)] = all_metrics[shortform]

        return display_metrics


__all__ = ["CommandBuildContext", "DefaultOrchestratorPlugin", "OrchestratorPlugin"]

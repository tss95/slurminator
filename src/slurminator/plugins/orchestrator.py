"""Plugin hooks for project-specific orchestrator behavior."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from shlex import quote, split
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
    """Default plugin with explicit-command support and no project-specific hooks."""

    def validate_experiment(self, exp: Mapping[str, Any], overrides: Mapping[str, Any]) -> bool:
        """Perform no project-specific validation."""
        return False

    def build_commands_line(self, exp: Mapping[str, Any], context: CommandBuildContext) -> str:
        """Return an explicit experiment command, or raise with setup guidance."""
        command = _explicit_command_from_experiment(exp)
        if command:
            return command
        raise _missing_command_error(exp)

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


@dataclass
class SimpleCommandPlugin(DefaultOrchestratorPlugin):
    """Build shell commands from an entrypoint and experiment-row config fields.

    This is the onboarding path for adopters that do not need custom Python
    hooks. Per-experiment ``extra_command`` or ``command`` still wins when set.
    """

    entrypoint: str | Sequence[str] | None = None
    config_field: str = "config"
    config_arg: str | None = "--config"
    extra_args: Sequence[str] = ()
    experiment_args_field: str = "command_args"
    sweep_params_arg: str | None = None
    orchestrator_flag: str | None = "--orchestrator"
    multi_experiment_flag: str | None = None

    def build_commands_line(self, exp: Mapping[str, Any], context: CommandBuildContext) -> str:
        """Return an explicit command or build one from the configured entrypoint."""
        command = _explicit_command_from_experiment(exp)
        if command:
            return command

        if not self.entrypoint:
            raise _missing_command_error(exp)

        parts = _entrypoint_parts(self.entrypoint)
        if not parts:
            raise _missing_command_error(exp)

        if self.config_arg:
            config_value = exp.get(self.config_field)
            if config_value is None and self.config_field == "config":
                config_value = exp.get("config_path")
            if config_value is None:
                exp_id = exp.get("experiment_id", "<unknown>")
                raise ValueError(
                    f"Experiment {exp_id!r} cannot be submitted with SimpleCommandPlugin because it has no "
                    f"{self.config_field!r} field. Add {self.config_field}: path/to/config.yaml to the experiment row, "
                    "set config_arg=None if your entrypoint does not take a config file, or provide an explicit "
                    "extra_command/command."
                )
            parts.extend([self.config_arg, quote(str(config_value))])

        parts.extend(quote(str(arg)) for arg in self.extra_args)

        experiment_args = exp.get(self.experiment_args_field)
        if experiment_args is not None:
            parts.extend(_normalise_command_args(experiment_args, field_name=self.experiment_args_field))

        if self.sweep_params_arg and exp.get("sweep_params"):
            parts.extend([self.sweep_params_arg, quote(str(exp["sweep_params"]))])

        if self.multi_experiment_flag and context.multi_experiment:
            parts.append(self.multi_experiment_flag)

        if self.orchestrator_flag:
            parts.append(self.orchestrator_flag)

        return " ".join(parts)


def _explicit_command_from_experiment(exp: Mapping[str, Any]) -> str | None:
    """Return a non-empty explicit command from a generic experiment row."""
    for key in ("extra_command", "command"):
        value = exp.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _entrypoint_parts(entrypoint: str | Sequence[str]) -> list[str]:
    """Return shell-safe entrypoint tokens."""
    if isinstance(entrypoint, str):
        return [quote(part) for part in split(entrypoint)]
    return [quote(str(part)) for part in entrypoint if str(part).strip()]


def _normalise_command_args(value: object, *, field_name: str) -> list[str]:
    """Return command argument tokens from a string or sequence."""
    if isinstance(value, str):
        return [quote(part) for part in split(value)]
    if isinstance(value, Sequence):
        return [quote(str(item)) for item in value]
    raise TypeError(f"{field_name} must be a string or sequence of strings, got {type(value).__name__}.")


def _missing_command_error(exp: Mapping[str, Any]) -> NotImplementedError:
    """Return an actionable command-construction error."""
    exp_id = exp.get("experiment_id", "<unknown>")
    return NotImplementedError(
        "Slurminator cannot build a job command for experiment "
        f"{exp_id!r} because no explicit command or command-building plugin is configured.\n\n"
        "Resolve this in one of three ways:\n"
        "1. Add an explicit command to the experiment YAML, for example:\n"
        "   extra_command: \"python train.py --config cfg/train.yaml --orchestrator\"\n"
        "   or:\n"
        "   command: \"python train.py --config cfg/train.yaml --orchestrator\"\n"
        "2. Use slurminator.plugins.SimpleCommandPlugin(entrypoint=\"python train.py\", config_arg=\"--config\") "
        "and add a config/config_path field to each experiment row.\n"
        "3. Implement a project plugin by subclassing DefaultOrchestratorPlugin and overriding build_commands_line()."
    )


__all__ = ["CommandBuildContext", "DefaultOrchestratorPlugin", "OrchestratorPlugin", "SimpleCommandPlugin"]

"""Plugin hooks for project-specific orchestrator behavior."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from shlex import quote, split
from typing import Any, Protocol, runtime_checkable

from slurminator.experiments.status_enum import ExperimentStatus


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

    def interpret_log_tail(
        self, *, exp: Mapping[str, Any], log_tail: str, current_status: Any, stage: str = "pre_heuristics"
    ) -> Any | None:
        """Return a project-specific terminal status override from job logs."""

    def annotate_log_tail(self, *, exp: dict[str, Any], log_tail: str) -> None:
        """Optionally annotate an experiment row from job logs."""


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

    def interpret_log_tail(
        self, *, exp: Mapping[str, Any], log_tail: str, current_status: Any, stage: str = "pre_heuristics"
    ) -> Any | None:
        """Interpret common opt-in status markers and broadly useful failure keywords."""
        log_lower = log_tail.lower()
        if stage == "pre_heuristics":
            return _interpret_explicit_status_marker(log_lower)
        if stage == "heuristics":
            return _interpret_common_failure_keywords(log_lower)
        return None

    def annotate_log_tail(self, *, exp: dict[str, Any], log_tail: str) -> None:
        """Default plugin does not annotate experiment rows from logs."""


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


def _interpret_explicit_status_marker(log_lower: str) -> ExperimentStatus | None:
    """Interpret Slurminator's optional plain-text terminal status convention."""
    for line in reversed(log_lower.splitlines()):
        if "experiment_status" not in line:
            continue
        if "failed" in line or "failure" in line:
            return ExperimentStatus.FAILED
        if "success" in line or "completed" in line:
            return ExperimentStatus.COMPLETED
        break
    return None


def _interpret_common_failure_keywords(log_lower: str) -> ExperimentStatus | None:
    """Interpret framework-neutral-ish timeout/OOM markers as a convenience default."""
    timeout_keywords = ("job timed out", "due to time limit")
    if any(keyword in log_lower for keyword in timeout_keywords):
        return ExperimentStatus.TIMEOUT

    oom_keywords = (
        "out of memory",
        "outofmemoryerror",
        "cuda out of memory",
        "hip out of memory",
        "cudaerror: out of memory",
    )
    if any(keyword in log_lower for keyword in oom_keywords):
        return ExperimentStatus.OOM
    return None


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

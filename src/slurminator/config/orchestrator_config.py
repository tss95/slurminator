"""Generic orchestrator behavior settings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TimeoutRiskSettings:
    """Dashboard timeout-risk visualization thresholds."""

    min_progress: float = 0.20
    min_runtime_seconds: int = 15 * 60
    medium_ratio: float = 0.85
    high_ratio: float = 1.0


@dataclass
class SparklineSettings:
    """Dashboard v4 sparkline trend-color thresholds."""

    flat_slope_norm: float = 0.01
    directional_slope_norm: float = 0.02
    oscillation_residual_norm: float = 0.10


@dataclass
class DashboardSettings:
    """Dashboard behavior settings."""

    ui_version: str = "v4"
    poll_interval_seconds: int = 30
    timeout_risk: TimeoutRiskSettings = field(default_factory=TimeoutRiskSettings)
    sparkline: SparklineSettings = field(default_factory=SparklineSettings)


@dataclass
class RetrySettings:
    """Timeout retry behavior settings."""

    retry_timeout_with_estimated_time: bool = False
    timeout_retry_buffer: float = 1.30
    timeout_retry_max_attempts: int = 1


@dataclass
class PollSettings:
    """Polling and stale-status behavior settings."""

    ssh_keepalive_interval_seconds: int = 60
    status_file_stale_threshold_seconds: int = 300


@dataclass
class CommandSettings:
    """Generic SimpleCommandPlugin defaults."""

    entrypoint: str | None = None
    config_field: str = "config"
    config_arg: str | None = "--config"
    extra_args: tuple[str, ...] = ()
    experiment_args_field: str = "command_args"
    sweep_params_arg: str | None = None
    orchestrator_flag: str | None = "--orchestrator"
    multi_experiment_flag: str | None = None


@dataclass
class OrchestratorSettings:
    """Top-level generic orchestrator settings."""

    dashboard: DashboardSettings = field(default_factory=DashboardSettings)
    retry: RetrySettings = field(default_factory=RetrySettings)
    polling: PollSettings = field(default_factory=PollSettings)
    command: CommandSettings = field(default_factory=CommandSettings)


def _coerce_float(value: Any, default: float, name: str, *, logger: Any | None, source: Path | str | None) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        if logger is not None:
            logger.warning(
                "Invalid orchestrator.%s=%r in %s; using default %s", name, value, source or "config", default
            )
        return default


def _coerce_int(value: Any, default: int, name: str, *, logger: Any | None, source: Path | str | None) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        if logger is not None:
            logger.warning(
                "Invalid orchestrator.%s=%r in %s; using default %s", name, value, source or "config", default
            )
        return default


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _coerce_optional_str(value: Any, default: str | None) -> str | None:
    if value is None:
        return default
    text = str(value).strip()
    return text or None


def _coerce_str(value: Any, default: str) -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def _coerce_str_tuple(value: Any, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        return tuple(part for part in value.split() if part)
    if isinstance(value, (list, tuple)):
        return tuple(str(part) for part in value if str(part).strip())
    return default


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def parse_orchestrator_settings(
    raw: Mapping[str, Any] | None, *, logger: Any | None = None, source: Path | str | None = None
) -> OrchestratorSettings:
    """Parse generic orchestrator settings from a raw YAML mapping."""
    raw = raw or {}
    orchestrator_raw = _mapping_or_empty(raw.get("orchestrator")) if "orchestrator" in raw else _mapping_or_empty(raw)
    dashboard_raw = _mapping_or_empty(orchestrator_raw.get("dashboard"))
    timeout_raw = _mapping_or_empty(dashboard_raw.get("timeout_risk"))
    sparkline_raw = _mapping_or_empty(dashboard_raw.get("sparkline"))
    retry_raw = _mapping_or_empty(orchestrator_raw.get("retry"))
    polling_raw = _mapping_or_empty(orchestrator_raw.get("polling"))
    command_raw = _mapping_or_empty(orchestrator_raw.get("command"))

    timeout_defaults = TimeoutRiskSettings()
    min_runtime_seconds_raw = timeout_raw.get("min_runtime_seconds")
    if min_runtime_seconds_raw is None and timeout_raw.get("min_runtime_minutes") is not None:
        minutes = _coerce_float(
            timeout_raw.get("min_runtime_minutes"),
            timeout_defaults.min_runtime_seconds / 60.0,
            "dashboard.timeout_risk.min_runtime_minutes",
            logger=logger,
            source=source,
        )
        min_runtime_seconds_raw = int(minutes * 60)

    timeout_settings = TimeoutRiskSettings(
        min_progress=_coerce_float(
            timeout_raw.get("min_progress"),
            timeout_defaults.min_progress,
            "dashboard.timeout_risk.min_progress",
            logger=logger,
            source=source,
        ),
        min_runtime_seconds=_coerce_int(
            min_runtime_seconds_raw,
            timeout_defaults.min_runtime_seconds,
            "dashboard.timeout_risk.min_runtime_seconds",
            logger=logger,
            source=source,
        ),
        medium_ratio=_coerce_float(
            timeout_raw.get("medium_ratio"),
            timeout_defaults.medium_ratio,
            "dashboard.timeout_risk.medium_ratio",
            logger=logger,
            source=source,
        ),
        high_ratio=_coerce_float(
            timeout_raw.get("high_ratio"),
            timeout_defaults.high_ratio,
            "dashboard.timeout_risk.high_ratio",
            logger=logger,
            source=source,
        ),
    )
    timeout_settings.min_progress = max(0.0, min(timeout_settings.min_progress, 1.0))
    timeout_settings.min_runtime_seconds = max(1, int(timeout_settings.min_runtime_seconds))
    timeout_settings.medium_ratio = max(0.0, timeout_settings.medium_ratio)
    timeout_settings.high_ratio = max(timeout_settings.medium_ratio, timeout_settings.high_ratio)

    sparkline_defaults = SparklineSettings()
    sparkline_settings = SparklineSettings(
        flat_slope_norm=max(
            0.0,
            _coerce_float(
                sparkline_raw.get("flat_slope_norm"),
                sparkline_defaults.flat_slope_norm,
                "dashboard.sparkline.flat_slope_norm",
                logger=logger,
                source=source,
            ),
        ),
        directional_slope_norm=max(
            0.0,
            _coerce_float(
                sparkline_raw.get("directional_slope_norm"),
                sparkline_defaults.directional_slope_norm,
                "dashboard.sparkline.directional_slope_norm",
                logger=logger,
                source=source,
            ),
        ),
        oscillation_residual_norm=max(
            0.0,
            _coerce_float(
                sparkline_raw.get("oscillation_residual_norm"),
                sparkline_defaults.oscillation_residual_norm,
                "dashboard.sparkline.oscillation_residual_norm",
                logger=logger,
                source=source,
            ),
        ),
    )

    dashboard_defaults = DashboardSettings()
    dashboard_settings = DashboardSettings(
        ui_version=str(dashboard_raw.get("ui_version", dashboard_defaults.ui_version)),
        poll_interval_seconds=max(
            1,
            _coerce_int(
                dashboard_raw.get("poll_interval_seconds"),
                dashboard_defaults.poll_interval_seconds,
                "dashboard.poll_interval_seconds",
                logger=logger,
                source=source,
            ),
        ),
        timeout_risk=timeout_settings,
        sparkline=sparkline_settings,
    )

    retry_defaults = RetrySettings()
    retry_settings = RetrySettings(
        retry_timeout_with_estimated_time=_coerce_bool(
            retry_raw.get("retry_timeout_with_estimated_time"), retry_defaults.retry_timeout_with_estimated_time
        ),
        timeout_retry_buffer=max(
            1.0,
            _coerce_float(
                retry_raw.get("timeout_retry_buffer"),
                retry_defaults.timeout_retry_buffer,
                "retry.timeout_retry_buffer",
                logger=logger,
                source=source,
            ),
        ),
        timeout_retry_max_attempts=max(
            0,
            _coerce_int(
                retry_raw.get("timeout_retry_max_attempts"),
                retry_defaults.timeout_retry_max_attempts,
                "retry.timeout_retry_max_attempts",
                logger=logger,
                source=source,
            ),
        ),
    )

    polling_defaults = PollSettings()
    polling_settings = PollSettings(
        ssh_keepalive_interval_seconds=max(
            1,
            _coerce_int(
                polling_raw.get("ssh_keepalive_interval_seconds"),
                polling_defaults.ssh_keepalive_interval_seconds,
                "polling.ssh_keepalive_interval_seconds",
                logger=logger,
                source=source,
            ),
        ),
        status_file_stale_threshold_seconds=max(
            1,
            _coerce_int(
                polling_raw.get("status_file_stale_threshold_seconds"),
                polling_defaults.status_file_stale_threshold_seconds,
                "polling.status_file_stale_threshold_seconds",
                logger=logger,
                source=source,
            ),
        ),
    )

    command_defaults = CommandSettings()
    command_settings = CommandSettings(
        entrypoint=_coerce_optional_str(command_raw.get("entrypoint"), command_defaults.entrypoint),
        config_field=_coerce_str(command_raw.get("config_field"), command_defaults.config_field),
        config_arg=_coerce_optional_str(command_raw.get("config_arg"), command_defaults.config_arg),
        extra_args=_coerce_str_tuple(command_raw.get("extra_args"), command_defaults.extra_args),
        experiment_args_field=_coerce_str(
            command_raw.get("experiment_args_field"), command_defaults.experiment_args_field
        ),
        sweep_params_arg=_coerce_optional_str(command_raw.get("sweep_params_arg"), command_defaults.sweep_params_arg),
        orchestrator_flag=_coerce_optional_str(
            command_raw.get("orchestrator_flag"), command_defaults.orchestrator_flag
        ),
        multi_experiment_flag=_coerce_optional_str(
            command_raw.get("multi_experiment_flag"), command_defaults.multi_experiment_flag
        ),
    )

    return OrchestratorSettings(
        dashboard=dashboard_settings, retry=retry_settings, polling=polling_settings, command=command_settings
    )


__all__ = [
    "CommandSettings",
    "DashboardSettings",
    "OrchestratorSettings",
    "PollSettings",
    "RetrySettings",
    "SparklineSettings",
    "TimeoutRiskSettings",
    "parse_orchestrator_settings",
]

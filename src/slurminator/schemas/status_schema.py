"""Pydantic model for the orchestrator target status-file schema."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "1.1"
StatusSchemaVersion = Literal["1.0", "1.1"]
StatusState = Literal["initializing", "running", "completed"]
ProgressUnit = Literal["epoch", "step"]

ALLOWED_STATUS_TRANSITIONS: dict[str, tuple[str, ...]] = {
    "initializing": ("running", "completed"),
    "running": ("completed",),
    "completed": (),
}


def can_transition(previous: StatusState, next_status: StatusState) -> bool:
    """Return whether a callback may move from one live status to another."""
    if previous == next_status:
        return True
    return next_status in ALLOWED_STATUS_TRANSITIONS[previous]


class Speed(BaseModel):
    """Primary live throughput measurement."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    value: float = Field(gt=0)
    unit: str = Field(min_length=1)


class Progress(BaseModel):
    """Training progress block with one declared primary axis."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    unit: ProgressUnit
    current: int = Field(ge=0)
    total: int | None = Field(default=None, ge=0)
    current_epoch: int | None = Field(default=None, ge=0)
    total_epochs: int | None = Field(default=None, ge=0)
    current_step: int | None = Field(default=None, ge=0)
    total_steps: int | None = Field(default=None, ge=0)
    speed: Speed | None = None
    samples_per_sec: float | None = Field(default=None, ge=0)
    step_time_ms_ema: float | None = Field(default=None, ge=0)
    eta_seconds: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _primary_axis_must_mirror_specific_fields(self) -> "Progress":
        if self.unit == "epoch":
            if self.current_epoch is None:
                raise ValueError("progress.current_epoch is required when progress.unit == 'epoch'.")
            if self.current != self.current_epoch:
                raise ValueError("progress.current must equal progress.current_epoch when unit == 'epoch'.")
            if self.total != self.total_epochs:
                raise ValueError("progress.total must equal progress.total_epochs when unit == 'epoch'.")
        elif self.unit == "step":
            if self.current_step is None:
                raise ValueError("progress.current_step is required when progress.unit == 'step'.")
            if self.current != self.current_step:
                raise ValueError("progress.current must equal progress.current_step when unit == 'step'.")
            if self.total != self.total_steps:
                raise ValueError("progress.total must equal progress.total_steps when unit == 'step'.")

        if self.total is not None and self.current > self.total:
            raise ValueError("progress.current must be <= progress.total when progress.total is known.")
        if self.current_epoch is not None and self.total_epochs is not None and self.current_epoch > self.total_epochs:
            raise ValueError("progress.current_epoch must be <= progress.total_epochs when total_epochs is known.")
        if self.current_step is not None and self.total_steps is not None and self.current_step > self.total_steps:
            raise ValueError("progress.current_step must be <= progress.total_steps when total_steps is known.")
        return self


class MetricInfo(BaseModel):
    """Optional dashboard rendering hints for one metric key."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    shortform: str | None = None
    higher_better: bool | None = None
    format: str | None = None
    threshold: float | None = None

    @field_validator("shortform", "format")
    @classmethod
    def _blank_strings_become_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None


class Display(BaseModel):
    """Dashboard-facing labels and rendering metadata."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    run_name: str = Field(min_length=1)
    primary_metric: str | None = None
    secondary_metric: str | None = None
    metric_info: dict[str, MetricInfo] = Field(default_factory=dict)

    @field_validator("run_name")
    @classmethod
    def _run_name_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("display.run_name must not be blank.")
        return stripped

    @field_validator("primary_metric", "secondary_metric")
    @classmethod
    def _metric_reference_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("display metric references must not be blank.")
        return stripped

    @field_validator("metric_info")
    @classmethod
    def _metric_info_keys_must_not_be_blank(cls, value: dict[str, MetricInfo]) -> dict[str, MetricInfo]:
        for key in value:
            if not isinstance(key, str) or not key.strip():
                raise ValueError("display.metric_info keys must be non-empty strings.")
        return value


class OrchestratorStatus(BaseModel):
    """Target live-status schema written by the Phase 2B status callback."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, validate_assignment=True)

    schema_version: StatusSchemaVersion = SCHEMA_VERSION
    experiment_id: str = Field(min_length=1)
    job_id: str = Field(min_length=1)
    status: StatusState
    last_update: float = Field(gt=0)
    progress: Progress
    metrics: dict[str, float] = Field(default_factory=dict)
    display: Display
    links: dict[str, str] = Field(default_factory=dict)
    attempt: int = Field(default=1, ge=1)

    @field_validator("experiment_id", "job_id")
    @classmethod
    def _identity_fields_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("identity fields must not be blank.")
        return stripped

    @field_validator("last_update")
    @classmethod
    def _last_update_must_be_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("last_update must be finite.")
        return value

    @field_validator("metrics", mode="before")
    @classmethod
    def _metrics_must_be_flat_finite_numbers(cls, value: object) -> object:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("metrics must be a mapping from string keys to finite numeric values.")
        for key, metric_value in value.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("metrics keys must be non-empty strings.")
            if isinstance(metric_value, bool) or not isinstance(metric_value, (int, float)):
                raise ValueError(f"metrics[{key!r}] must be a finite JSON number.")
            if not math.isfinite(float(metric_value)):
                raise ValueError(f"metrics[{key!r}] must be finite.")
        return value

    @field_validator("links", mode="before")
    @classmethod
    def _links_must_be_flat_strings(cls, value: object) -> object:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("links must be a mapping from string keys to string values.")
        for key, link_value in value.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("links keys must be non-empty strings.")
            if not isinstance(link_value, str):
                raise ValueError(f"links[{key!r}] must be a string.")
        return value

    @model_validator(mode="after")
    def _display_references_must_point_to_metrics(self) -> "OrchestratorStatus":
        metric_keys = set(self.metrics)
        for label, metric_key in (
            ("display.primary_metric", self.display.primary_metric),
            ("display.secondary_metric", self.display.secondary_metric),
        ):
            if metric_key is not None and metric_key not in metric_keys:
                raise ValueError(f"{label}={metric_key!r} is not present in metrics.")

        for metric_key in self.display.metric_info:
            if metric_key not in metric_keys:
                raise ValueError(f"display.metric_info[{metric_key!r}] has no corresponding metrics entry.")
        return self


class HistoryEntry(BaseModel):
    """One JSONL history entry for a status write with metrics."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: Literal["1.1"] = "1.1"
    timestamp: float
    attempt: int
    epoch: int | None
    step: int | None
    unit: ProgressUnit | None = None
    metrics: dict[str, float]


__all__ = [
    "ALLOWED_STATUS_TRANSITIONS",
    "SCHEMA_VERSION",
    "Display",
    "HistoryEntry",
    "MetricInfo",
    "OrchestratorStatus",
    "Progress",
    "ProgressUnit",
    "Speed",
    "StatusSchemaVersion",
    "StatusState",
    "can_transition",
]

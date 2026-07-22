"""Pure helpers for building target orchestrator status payloads."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from slurminator.schemas.status_schema import (
    Display,
    MetricColumn,
    MetricInfo,
    OrchestratorStatus,
    Progress,
    ProgressUnit,
    Speed,
    StatusState,
)


@dataclass(frozen=True)
class MetricDisplayCandidate:
    """Display metadata candidate for a metric that may appear later."""

    shortform: str | None = None
    higher_better: bool | None = None
    format: str | None = None
    threshold: float | None = None
    best_key: str | None = None


@dataclass(frozen=True)
class GenericProgressSnapshot:
    """Generic progress fields after framework-specific choices are resolved."""

    unit: ProgressUnit
    current_epoch: int | None = None
    total_epochs: int | None = None
    current_step: int | None = None
    total_steps: int | None = None
    speed_value: float | None = None
    speed_unit: str = "it/sec"
    samples_per_sec: float | None = None
    step_time_ms_ema: float | None = None
    eta_seconds: int | None = None


def normalize_status_payload(
    *,
    experiment_id: str,
    job_id: str,
    status: StatusState,
    last_update: float,
    progress: GenericProgressSnapshot,
    run_name: str,
    metrics: Mapping[str, object] | None = None,
    primary_metric: str | None = None,
    secondary_metric: str | None = None,
    metric_columns: Sequence[object] | None = None,
    metric_info: Mapping[str, MetricDisplayCandidate | MetricInfo | Mapping[str, object]] | None = None,
    links: Mapping[str, object] | None = None,
) -> OrchestratorStatus:
    """Return a validated target-schema status object from generic inputs.

    This function is intentionally project-agnostic: callers must decide what the
    epoch and step fields mean before invoking it.
    """

    clean_metrics = _filter_finite_numeric_metrics(metrics or {})
    present_metric_keys = set(clean_metrics)
    clean_columns = _materialize_metric_columns(metric_columns or (), present_metric_keys=present_metric_keys)
    clean_metric_info = _materialize_metric_info(metric_info or {}, present_metric_keys=present_metric_keys)
    for column in clean_columns:
        clean_metric_info.setdefault(
            column.key,
            MetricInfo(
                shortform=column.shortform,
                higher_better=column.higher_better,
                format=column.format,
                threshold=column.threshold,
                best_key=column.best_key,
            ),
        )

    clean_primary = _materialize_metric_reference(primary_metric, present_metric_keys=present_metric_keys)
    clean_secondary = _materialize_metric_reference(secondary_metric, present_metric_keys=present_metric_keys)
    if clean_primary is None and clean_columns:
        clean_primary = clean_columns[0].key
    if clean_secondary is None and len(clean_columns) > 1:
        clean_secondary = clean_columns[1].key
    progress_block = _build_progress(progress)

    return OrchestratorStatus(
        experiment_id=experiment_id,
        job_id=job_id,
        status=status,
        last_update=float(last_update),
        progress=progress_block,
        metrics=clean_metrics,
        display=Display(
            run_name=run_name,
            primary_metric=clean_primary,
            secondary_metric=clean_secondary,
            metric_columns=clean_columns,
            metric_info=clean_metric_info,
        ),
        links=_filter_links(links or {}),
    )


def _build_progress(snapshot: GenericProgressSnapshot) -> Progress:
    if snapshot.unit == "epoch":
        if snapshot.current_epoch is None:
            raise ValueError("current_epoch is required for epoch progress.")
        current = int(snapshot.current_epoch)
        total = _optional_int(snapshot.total_epochs)
    elif snapshot.unit == "step":
        if snapshot.current_step is None:
            raise ValueError("current_step is required for step progress.")
        current = int(snapshot.current_step)
        total = _optional_int(snapshot.total_steps)
    else:  # pragma: no cover - typing/runtime guard
        raise ValueError(f"Unsupported progress unit: {snapshot.unit!r}")

    speed = None
    if (
        snapshot.speed_value is not None
        and math.isfinite(float(snapshot.speed_value))
        and float(snapshot.speed_value) > 0
    ):
        speed = Speed(value=float(snapshot.speed_value), unit=snapshot.speed_unit)

    return Progress(
        unit=snapshot.unit,
        current=current,
        total=total,
        current_epoch=_optional_int(snapshot.current_epoch),
        total_epochs=_optional_int(snapshot.total_epochs),
        current_step=_optional_int(snapshot.current_step),
        total_steps=_optional_int(snapshot.total_steps),
        speed=speed,
        samples_per_sec=_optional_non_negative_float(snapshot.samples_per_sec),
        step_time_ms_ema=_optional_non_negative_float(snapshot.step_time_ms_ema),
        eta_seconds=_optional_int(snapshot.eta_seconds),
    )


def _filter_finite_numeric_metrics(metrics: Mapping[str, object]) -> dict[str, float]:
    clean: dict[str, float] = {}
    for key, value in metrics.items():
        if not isinstance(key, str) or not key.strip():
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        value_float = float(value)
        if not math.isfinite(value_float):
            continue
        clean[key.strip()] = value_float
    return clean


def _materialize_metric_reference(metric_key: str | None, *, present_metric_keys: set[str]) -> str | None:
    if metric_key is None:
        return None
    stripped = str(metric_key).strip()
    if not stripped:
        return None
    if stripped not in present_metric_keys:
        return None
    return stripped


def _materialize_metric_info(
    metric_info: Mapping[str, MetricDisplayCandidate | MetricInfo | Mapping[str, object]],
    *,
    present_metric_keys: set[str],
) -> dict[str, MetricInfo]:
    materialized: dict[str, MetricInfo] = {}
    for key, candidate in metric_info.items():
        if not isinstance(key, str):
            continue
        stripped_key = key.strip()
        if not stripped_key or stripped_key not in present_metric_keys:
            continue
        materialized[stripped_key] = _coerce_metric_info(candidate)
    return materialized


def _materialize_metric_columns(
    metric_columns: Sequence[object], *, present_metric_keys: set[str]
) -> list[MetricColumn]:
    materialized: list[MetricColumn] = []
    seen: set[str] = set()
    for column in metric_columns:
        coerced = _coerce_metric_column(column)
        if coerced is None:
            continue
        if coerced.key not in present_metric_keys or coerced.key in seen:
            continue
        materialized.append(coerced)
        seen.add(coerced.key)
    return materialized


def _coerce_metric_column(candidate: object) -> MetricColumn | None:
    if isinstance(candidate, MetricColumn):
        return candidate
    if isinstance(candidate, Mapping):
        key = _optional_string(candidate.get("key"))
        if not key:
            return None
        return MetricColumn(
            key=key,
            shortform=_optional_string(candidate.get("shortform")) or _optional_string(candidate.get("label")),
            higher_better=_optional_bool(candidate.get("higher_better")),
            format=_optional_string(candidate.get("format")) or _optional_string(candidate.get("value_format")),
            threshold=_optional_finite_float(candidate.get("threshold")),
            best_key=_optional_string(candidate.get("best_key")),
        )
    key = _optional_string(getattr(candidate, "key", None))
    if not key:
        return None
    return MetricColumn(
        key=key,
        shortform=_optional_string(getattr(candidate, "shortform", None))
        or _optional_string(getattr(candidate, "label", None)),
        higher_better=_optional_bool(getattr(candidate, "higher_better", None)),
        format=_optional_string(getattr(candidate, "format", None))
        or _optional_string(getattr(candidate, "value_format", None)),
        threshold=_optional_finite_float(getattr(candidate, "threshold", None)),
        best_key=_optional_string(getattr(candidate, "best_key", None)),
    )


def _coerce_metric_info(candidate: MetricDisplayCandidate | MetricInfo | Mapping[str, object]) -> MetricInfo:
    if isinstance(candidate, MetricInfo):
        return candidate
    if isinstance(candidate, MetricDisplayCandidate):
        return MetricInfo(
            shortform=candidate.shortform,
            higher_better=candidate.higher_better,
            format=candidate.format,
            threshold=candidate.threshold,
            best_key=candidate.best_key,
        )
    return MetricInfo(
        shortform=_optional_string(candidate.get("shortform")),
        higher_better=_optional_bool(candidate.get("higher_better")),
        format=_optional_string(candidate.get("format")),
        threshold=_optional_finite_float(candidate.get("threshold")),
        best_key=_optional_string(candidate.get("best_key")),
    )


def _filter_links(links: Mapping[str, object]) -> dict[str, str]:
    clean: dict[str, str] = {}
    for key, value in links.items():
        if not isinstance(key, str) or not key.strip():
            continue
        if not isinstance(value, str):
            continue
        clean[key.strip()] = value
    return clean


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("boolean values are not valid integer progress fields.")
    int_value = int(value)
    if int_value < 0:
        raise ValueError("progress integer fields must be non-negative.")
    return int_value


def _optional_non_negative_float(value: object) -> float | None:
    finite = _optional_finite_float(value)
    if finite is None:
        return None
    if finite < 0:
        raise ValueError("progress float fields must be non-negative.")
    return finite


def _optional_finite_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    float_value = float(value)
    if not math.isfinite(float_value):
        return None
    return float_value


def _optional_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


__all__ = ["GenericProgressSnapshot", "MetricDisplayCandidate", "normalize_status_payload"]

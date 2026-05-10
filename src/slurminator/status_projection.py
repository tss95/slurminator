"""Project target-schema status files into experiment-row dictionaries."""

from __future__ import annotations

from collections.abc import Mapping

from slurminator.schemas.status_schema import OrchestratorStatus

_BASE_PROJECTION_FIELDS = (
    "status_schema_version",
    "status_experiment_id",
    "current_epoch",
    "max_epochs",
    "current_step",
    "max_steps",
    "target_metric_name",
    "target_metric_value",
    "secondary_metric_name",
    "secondary_metric_value",
    "metric_value",
    "all_metrics",
    "display_metric_info",
)


def project_status_to_experiment(
    exp: dict,
    status: OrchestratorStatus,
    *,
    run_name_field: str | None = "status_run_name",
    links_field: str | None = "status_links",
    link_field_map: Mapping[str, str] | None = None,
    step_current_epoch_field: str | None = None,
    step_total_epochs_field: str | None = None,
    speed_value_field: str | None = "speed_value",
    samples_per_sec_field: str | None = "samples_per_sec",
    step_time_ms_ema_field: str | None = "step_time_ms_ema",
    eta_seconds_field: str | None = "eta_seconds",
) -> set[str]:
    """Mutate ``exp`` with values projected from ``status``.

    The status schema is tracker-agnostic. Callers that need compatibility row
    fields for a project-specific tracker can map status ``links`` entries via
    ``link_field_map`` and choose a custom ``run_name_field``.
    """

    updated: set[str] = set()

    def put(key: str | None, value: object, *, skip_none: bool = True) -> None:
        if not key:
            return
        if skip_none and value is None:
            return
        exp[key] = value
        updated.add(key)

    progress = status.progress
    metrics = dict(status.metrics)
    metric_info = {key: info.model_dump(mode="json") for key, info in status.display.metric_info.items()}
    links = dict(status.links)

    put("status_schema_version", status.schema_version)
    put("status_experiment_id", status.experiment_id)
    put(run_name_field, status.display.run_name)
    put(links_field, links, skip_none=False)

    for link_key, exp_key in (link_field_map or {}).items():
        put(exp_key, links.get(link_key))

    put("current_step", progress.current_step)
    put("max_steps", progress.total_steps)
    put("current_epoch", progress.current_epoch)
    put("max_epochs", progress.total_epochs)
    if progress.unit == "step":
        put(step_current_epoch_field, progress.current_epoch)
        put(step_total_epochs_field, progress.total_epochs)

    if progress.speed is not None:
        put(speed_value_field, progress.speed.value)
    put(samples_per_sec_field, progress.samples_per_sec)
    put(step_time_ms_ema_field, progress.step_time_ms_ema)
    put(eta_seconds_field, progress.eta_seconds)

    put("all_metrics", dict(metrics), skip_none=False)
    put("display_metric_info", metric_info, skip_none=False)

    primary_metric = status.display.primary_metric
    if primary_metric:
        put("target_metric_name", primary_metric)
        if primary_metric in metrics:
            put("target_metric_value", metrics[primary_metric])
            put("metric_value", metrics[primary_metric])

    secondary_metric = status.display.secondary_metric
    if secondary_metric:
        put("secondary_metric_name", secondary_metric)
        if secondary_metric in metrics:
            put("secondary_metric_value", metrics[secondary_metric])

    for metric_key, info in metric_info.items():
        metric_value = metrics.get(metric_key)
        shortform = info.get("shortform")
        if metric_value is not None and shortform:
            put(str(shortform), metric_value)
            exp["all_metrics"][shortform] = metric_value

    return updated


def status_projection_fields(
    *,
    run_name_field: str | None = "status_run_name",
    links_field: str | None = "status_links",
    link_field_map: Mapping[str, str] | None = None,
    step_current_epoch_field: str | None = None,
    step_total_epochs_field: str | None = None,
    speed_value_field: str | None = "speed_value",
    samples_per_sec_field: str | None = "samples_per_sec",
    step_time_ms_ema_field: str | None = "step_time_ms_ema",
    eta_seconds_field: str | None = "eta_seconds",
) -> tuple[str, ...]:
    """Return stable experiment-row fields produced by the projection helper."""

    fields: list[str] = list(_BASE_PROJECTION_FIELDS)
    fields.extend(
        field
        for field in (
            run_name_field,
            links_field,
            step_current_epoch_field,
            step_total_epochs_field,
            speed_value_field,
            samples_per_sec_field,
            step_time_ms_ema_field,
            eta_seconds_field,
        )
        if field
    )
    fields.extend(field for field in (link_field_map or {}).values() if field)
    return tuple(dict.fromkeys(fields))


__all__ = ["project_status_to_experiment", "status_projection_fields"]

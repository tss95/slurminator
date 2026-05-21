import pytest

from slurminator.callbacks.status_normalization import GenericProgressSnapshot, MetricDisplayCandidate
from slurminator.callbacks.status_normalization import normalize_status_payload
from slurminator.status_projection import project_status_to_experiment, status_projection_fields

pytestmark = pytest.mark.unit


def _status(**kwargs):
    return normalize_status_payload(
        experiment_id=kwargs.get("experiment_id", "exp-1"),
        job_id=kwargs.get("job_id", "123"),
        status="running",
        last_update=100.0,
        progress=kwargs.get("progress") or GenericProgressSnapshot(unit="epoch", current_epoch=5, total_epochs=10),
        run_name=kwargs.get("run_name", "run-1"),
        metrics=kwargs.get("metrics") or {},
        primary_metric=kwargs.get("primary"),
        secondary_metric=kwargs.get("secondary"),
        metric_info=kwargs.get("metric_info") or {},
        links=kwargs.get("links") or {},
    )


def test_project_status_to_experiment_uses_generic_defaults() -> None:
    status = _status(
        metrics={"val/acc": 0.91, "val/loss": 0.2},
        primary="val/acc",
        secondary="val/loss",
        metric_info={
            "val/acc": MetricDisplayCandidate(shortform="acc", higher_better=True),
            "val/loss": MetricDisplayCandidate(shortform="loss", higher_better=False),
        },
        links={"tracker_run_id": "abc"},
    )
    exp: dict = {}

    updated = project_status_to_experiment(exp, status)

    assert exp["status_schema_version"] == "1.1"
    assert exp["status_experiment_id"] == "exp-1"
    assert exp["progress_unit"] == "epoch"
    assert exp["progress"]["unit"] == "epoch"
    assert exp["status_run_name"] == "run-1"
    assert exp["status_links"] == {"tracker_run_id": "abc"}
    assert exp["target_metric_name"] == "val/acc"
    assert exp["target_metric_value"] == 0.91
    assert exp["metric_value"] == 0.91
    assert exp["secondary_metric_value"] == 0.2
    assert exp["acc"] == 0.91
    assert exp["all_metrics"]["acc"] == 0.91
    assert "status_run_name" in updated


def test_project_status_to_experiment_supports_project_specific_aliases() -> None:
    progress = GenericProgressSnapshot(
        unit="step",
        current_step=12,
        total_steps=50,
        current_epoch=3,
        total_epochs=10,
        speed_value=2.5,
        samples_per_sec=20.0,
        step_time_ms_ema=400.0,
        eta_seconds=15,
    )
    status = _status(progress=progress, links={"tracker_run_id": "abc", "tracker_url": "https://example/run"})
    exp: dict = {}

    project_status_to_experiment(
        exp,
        status,
        run_name_field="tracker_run_name",
        links_field=None,
        link_field_map={"tracker_run_id": "tracker_id", "tracker_url": "tracker_url"},
        step_current_epoch_field="pseudo_epoch",
        step_total_epochs_field="max_pseudo_epochs",
        speed_value_field="it_per_sec",
        samples_per_sec_field="samples_per_sec_project",
        step_time_ms_ema_field="step_time_ms_project",
        eta_seconds_field="eta_project_seconds",
    )

    assert exp["tracker_run_name"] == "run-1"
    assert "status_links" not in exp
    assert exp["tracker_id"] == "abc"
    assert exp["tracker_url"] == "https://example/run"
    assert exp["progress_unit"] == "step"
    assert exp["progress"]["unit"] == "step"
    assert exp["current_step"] == 12
    assert exp["max_steps"] == 50
    assert exp["current_epoch"] == 3
    assert exp["max_epochs"] == 10
    assert exp["pseudo_epoch"] == 3
    assert exp["max_pseudo_epochs"] == 10
    assert exp["it_per_sec"] == 2.5
    assert exp["samples_per_sec_project"] == 20.0
    assert exp["step_time_ms_project"] == 400.0
    assert exp["eta_project_seconds"] == 15


def test_status_projection_fields_reflects_configured_aliases() -> None:
    fields = status_projection_fields(
        run_name_field="tracker_run_name",
        links_field=None,
        link_field_map={"tracker_run_id": "tracker_id"},
        step_current_epoch_field="pseudo_epoch",
        speed_value_field="it_per_sec",
    )

    assert "tracker_run_name" in fields
    assert "tracker_id" in fields
    assert "progress" in fields
    assert "progress_unit" in fields
    assert "pseudo_epoch" in fields
    assert "it_per_sec" in fields
    assert "status_links" not in fields

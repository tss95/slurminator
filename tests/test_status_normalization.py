import pytest

from slurminator.callbacks.status_normalization import (
    GenericProgressSnapshot,
    MetricDisplayCandidate,
    normalize_status_payload,
)

pytestmark = pytest.mark.unit


def test_normalize_status_payload_materializes_display_only_for_present_metrics():
    status = normalize_status_payload(
        experiment_id="exp-1",
        job_id="123",
        status="running",
        last_update=100.0,
        progress=GenericProgressSnapshot(unit="epoch", current_epoch=1, total_epochs=2),
        run_name="run-1",
        metrics={"val/acc": 0.91},
        primary_metric="val/acc",
        secondary_metric="val/loss",
        metric_info={
            "val/acc": MetricDisplayCandidate(shortform="acc", higher_better=True),
            "val/loss": MetricDisplayCandidate(shortform="loss", higher_better=False),
        },
    )

    assert status.display.primary_metric == "val/acc"
    assert status.display.secondary_metric is None
    assert set(status.display.metric_info) == {"val/acc"}


def test_normalize_status_payload_filters_non_numeric_metrics_and_speed_zero():
    status = normalize_status_payload(
        experiment_id="exp-1",
        job_id="123",
        status="running",
        last_update=100.0,
        progress=GenericProgressSnapshot(unit="step", current_step=3, total_steps=10, speed_value=0.0),
        run_name="run-1",
        metrics={"ok": 1.5, "flag": True, "text": "bad"},
        metric_info={"ok": MetricDisplayCandidate(shortform="ok")},
    )

    assert status.metrics == {"ok": 1.5}
    assert status.progress.speed is None
    assert status.display.metric_info["ok"].shortform == "ok"

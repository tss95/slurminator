import pytest
from pydantic import ValidationError

from slurminator.schemas.status_schema import OrchestratorStatus, can_transition

pytestmark = pytest.mark.unit


def test_status_schema_roundtrip_validates_display_references():
    status = OrchestratorStatus(
        experiment_id="exp-1",
        job_id="123",
        status="running",
        last_update=100.0,
        progress={
            "unit": "step",
            "current": 5,
            "total": 10,
            "current_step": 5,
            "total_steps": 10,
            "current_epoch": 1,
            "total_epochs": 2,
        },
        metrics={"probe/lp0.010/step_best_top1_acc": 0.72},
        display={
            "run_name": "run-1",
            "primary_metric": "probe/lp0.010/step_best_top1_acc",
            "metric_info": {"probe/lp0.010/step_best_top1_acc": {"shortform": "lp0.010_top1"}},
        },
    )

    assert OrchestratorStatus.model_validate_json(status.model_dump_json()) == status


def test_status_schema_rejects_orphan_display_metric_info():
    with pytest.raises(ValidationError, match="has no corresponding metrics entry"):
        OrchestratorStatus(
            experiment_id="exp-1",
            job_id="123",
            status="running",
            last_update=100.0,
            progress={"unit": "epoch", "current": 1, "total": 2, "current_epoch": 1, "total_epochs": 2},
            metrics={},
            display={"run_name": "run-1", "metric_info": {"val/acc": {"shortform": "acc"}}},
        )


def test_status_state_machine_allows_forward_only():
    assert can_transition("initializing", "running")
    assert can_transition("running", "completed")
    assert can_transition("completed", "completed")
    assert not can_transition("completed", "running")

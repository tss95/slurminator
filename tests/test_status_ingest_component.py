from __future__ import annotations

import pytest

from slurminator.config import HPCType
from slurminator.status_ingest import StatusIngestContext, status_file_path, update_running_experiment_info

pytestmark = pytest.mark.unit


class _FakeConnection:
    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.commands: list[tuple[HPCType, str]] = []

    def run_command(self, hpc_type, cmd):  # noqa: ANN001
        self.commands.append((hpc_type, cmd))
        return self.payload, ""


def _minimal_status_json(job_id: str = "123") -> str:
    return f"""
    {{
      "schema_version": "1.0",
      "experiment_id": "exp-1",
      "job_id": "{job_id}",
      "status": "running",
      "last_update": 1710000000.0,
      "progress": {{
        "unit": "epoch",
        "current": 2,
        "total": 5,
        "current_epoch": 2,
        "total_epochs": 5
      }},
      "metrics": {{
        "accuracy": 0.75
      }},
      "display": {{
        "run_name": "run-exp-1",
        "primary_metric": "accuracy",
        "metric_info": {{
          "accuracy": {{
            "shortform": "acc",
            "higher_better": true,
            "format": ".2%"
          }}
        }}
      }},
      "links": {{
        "tracker": "https://example.invalid/run/123"
      }}
    }}
    """


def test_update_running_experiment_info_projects_target_status_without_orchestrator() -> None:
    exp = {"experiment_id": "exp-1", "job_id": "123", "hpc_assignment": HPCType.OLIVIA, "save_path": "/tmp/save"}
    saved: list[dict] = []
    context = StatusIngestContext(
        connection_manager=_FakeConnection(_minimal_status_json()),
        hpc_configs={},
        load_yaml=lambda: {"experiments": [{"experiment_id": "exp-1"}]},
        save_yaml=saved.append,
    )

    data = update_running_experiment_info(exp, context)

    assert data is not None
    assert exp["current_epoch"] == 2
    assert exp["max_epochs"] == 5
    assert exp["target_metric_name"] == "accuracy"
    assert exp["target_metric_value"] == 0.75
    assert exp["acc"] == 0.75
    assert saved[0]["experiments"][0]["target_metric_value"] == 0.75
    assert saved[0]["experiments"][0]["last_metrics_update"] == 1710000000.0


def test_update_running_experiment_info_rejects_job_mismatch_without_orchestrator() -> None:
    exp = {"experiment_id": "exp-1", "job_id": "expected", "hpc_assignment": HPCType.OLIVIA, "save_path": "/tmp/save"}
    context = StatusIngestContext(
        connection_manager=_FakeConnection(_minimal_status_json(job_id="other")),
        hpc_configs={},
        load_yaml=lambda: {"experiments": [{"experiment_id": "exp-1"}]},
        save_yaml=lambda data: None,
    )

    assert update_running_experiment_info(exp, context) is None
    assert "target_metric_value" not in exp


def test_status_file_path_uses_sweep_directory_when_present() -> None:
    assert status_file_path("/save", "4242", "abc") == "/save/.orchestrator_status/sweep_abc/status_4242.json"

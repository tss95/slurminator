from __future__ import annotations

from typing import Any

import pytest

from slurminator.experiments import ExperimentStatus
from slurminator.config import HPCType
from slurminator.plugins import DefaultOrchestratorPlugin
from slurminator.log_gathering import LogGatheringContext, gather_logs

pytestmark = pytest.mark.unit


class _FakeConnection:
    def __init__(self, log_tail: str) -> None:
        self.log_tail = log_tail
        self.commands: list[tuple[HPCType, str, bool]] = []

    def run_command(self, hpc_type, cmd, prefer_remote=False):  # noqa: ANN001
        self.commands.append((hpc_type, cmd, prefer_remote))
        return self.log_tail, ""


class _MarkerPlugin(DefaultOrchestratorPlugin):
    def interpret_log_tail(
        self, *, exp: dict[str, Any], log_tail: str, current_status: Any, stage: str = "pre_heuristics"
    ) -> Any | None:
        if stage == "pre_heuristics":
            if "EXPERIMENT_STATUS: FAILED" in log_tail or "EXPERIMENT_STATUS=FAILED" in log_tail:
                return ExperimentStatus.FAILED
            if "EXPERIMENT_STATUS: SUCCESS" in log_tail or "EXPERIMENT_STATUS=SUCCESS" in log_tail:
                return ExperimentStatus.COMPLETED
        return super().interpret_log_tail(exp=exp, log_tail=log_tail, current_status=current_status, stage=stage)

    def annotate_log_tail(self, *, exp: dict[str, Any], log_tail: str) -> None:
        if "OOM_Retry" in log_tail:
            exp["oom_recovered"] = True


def _context(log_tail: str) -> LogGatheringContext:
    return LogGatheringContext(
        connection_manager=_FakeConnection(log_tail),
        hpc_configs={},
        plugin=_MarkerPlugin(),
        is_local_hpc=lambda hpc: False,
    )


def test_gather_logs_uses_status_marker_to_override_completed_job_without_orchestrator() -> None:
    exp = {"experiment_id": "exp-1", "status": ExperimentStatus.COMPLETED, "output_dir": "/logs/exp-1"}

    gather_logs(exp, "123", HPCType.OLIVIA, _context("training\nEXPERIMENT_STATUS: FAILED\n"))

    assert exp["status"] == ExperimentStatus.FAILED


def test_gather_logs_detects_oom_recovery_marker_without_orchestrator() -> None:
    exp = {"experiment_id": "exp-1", "status": ExperimentStatus.COMPLETED, "output_dir": "/logs/exp-1"}

    gather_logs(exp, "123", HPCType.OLIVIA, _context("OOM detected\nwandb: tags: [OOM_Retry]\n"))

    assert exp["oom_recovered"] is True
    assert exp["status"] == ExperimentStatus.COMPLETED


def test_gather_logs_keeps_manual_cancel_without_timeout_or_oom_markers() -> None:
    exp = {
        "experiment_id": "exp-1",
        "status": ExperimentStatus.CANCELLED,
        "output_dir": "/logs/exp-1",
        "requested_time_hours": 3,
        "current_step": 300,
        "max_steps": 1000,
    }

    gather_logs(exp, "123", HPCType.OLIVIA, _context("cancelled by user\n"))

    assert exp["status"] == ExperimentStatus.CANCELLED
    assert "time_hours_override" not in exp


def test_gather_logs_uses_default_plugin_heuristics_for_common_oom_keywords() -> None:
    exp = {"experiment_id": "exp-1", "status": ExperimentStatus.CANCELLED, "output_dir": "/logs/exp-1"}

    gather_logs(exp, "123", HPCType.OLIVIA, _context("RuntimeError: CUDA out of memory."))

    assert exp["status"] == ExperimentStatus.OOM

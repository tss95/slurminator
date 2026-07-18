from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from slurminator.hpc_orchestrator import HPCOrchestrator
from slurminator.plugins import DefaultOrchestratorPlugin

pytestmark = pytest.mark.unit


class FakeConnection:
    """Minimal connection manager for direct preflight-helper tests."""

    def close_all(self) -> None:
        """Match the connection-manager cleanup contract."""
        return None


class RecordingPlugin(DefaultOrchestratorPlugin):
    """Record config validation calls made by the orchestrator."""

    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, Any], dict[str, Any]]] = []

    def validate_experiment(self, exp: dict[str, Any], overrides: dict[str, Any]) -> bool:
        """Record one experiment validation request."""
        self.calls.append((exp, overrides))
        return True


def _orchestrator(tmp_path: Path, plugin: RecordingPlugin, *, parse_overrides: Any) -> HPCOrchestrator:
    experiment_file = tmp_path / "experiments.yaml"
    experiment_file.write_text("experiments: []\n", encoding="utf-8")
    return HPCOrchestrator(
        str(experiment_file),
        concurrency_limits={},
        connection_manager=FakeConnection(),
        plugin=plugin,
        parse_overrides=parse_overrides,
    )


def test_fresh_ledger_runs_experiment_config_preflight(tmp_path: Path) -> None:
    plugin = RecordingPlugin()
    orchestrator = _orchestrator(tmp_path, plugin, parse_overrides=lambda raw: {"parsed": raw})
    exp = {"experiment_id": "fresh-1", "dataset_name": "HAR", "sweep_params": "x=1"}

    validated = orchestrator._preflight_validate_experiments({"experiments": [exp]})

    assert validated == 1
    assert plugin.calls == [(exp, {"parsed": "x=1"})]


def test_resume_ledger_skips_repeated_config_preflight(tmp_path: Path) -> None:
    plugin = RecordingPlugin()

    def unexpected_parse(_raw: object) -> dict[str, Any]:
        raise AssertionError("resume must not reparse experiment overrides")

    orchestrator = _orchestrator(tmp_path, plugin, parse_overrides=unexpected_parse)
    experiments = [
        {"experiment_id": "completed-1", "job_id": "12345", "sweep_params": "malformed"},
        {"experiment_id": "pending-1", "dataset_name": "HAR", "sweep_params": "x=2"},
    ]

    validated = orchestrator._preflight_validate_experiments({"experiments": experiments})

    assert validated == 0
    assert plugin.calls == []


def test_empty_job_id_does_not_skip_fresh_validation(tmp_path: Path) -> None:
    plugin = RecordingPlugin()
    orchestrator = _orchestrator(tmp_path, plugin, parse_overrides=lambda raw: {"parsed": raw})
    exp = {"experiment_id": "fresh-1", "job_id": "", "dataset_name": "HAR", "sweep_params": "x=1"}

    assert orchestrator._preflight_validate_experiments({"experiments": [exp]}) == 1
    assert plugin.calls == [(exp, {"parsed": "x=1"})]


def test_malformed_fresh_overrides_still_fail_preflight(tmp_path: Path) -> None:
    plugin = RecordingPlugin()

    def fail_parse(_raw: object) -> dict[str, Any]:
        raise ValueError("malformed overrides")

    orchestrator = _orchestrator(tmp_path, plugin, parse_overrides=fail_parse)
    exp = {"experiment_id": "fresh-1", "dataset_name": "HAR", "sweep_params": "malformed"}

    with pytest.raises(ValueError, match="malformed overrides"):
        orchestrator._preflight_validate_experiments({"experiments": [exp]})

    assert plugin.calls == []

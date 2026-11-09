"""Regression checks for the final, copyable experiment YAML reference."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from slurminator.experiments import ExperimentStatus
from slurminator.hpc_orchestrator import HPCOrchestrator

pytestmark = pytest.mark.unit


@pytest.fixture(params=[True, False], ids=["debug", "dashboard"])
def orchestrator(request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> HPCOrchestrator:
    """Exercise the real run loop with local, deterministic HPC and UI stand-ins."""

    class Dashboard:
        def __init__(self, **_kwargs: object) -> None:
            pass

        @contextmanager
        def mount(self, _orchestrator: HPCOrchestrator) -> Iterator[SimpleNamespace]:
            try:
                yield SimpleNamespace(update=print)
            finally:
                print("dashboard closed")

        def render(self, _experiments: list[dict]) -> str:
            return "dashboard rendered"

    exp_file = tmp_path / "experiment lists" / "experiments_20260911_163454.yaml"
    connection_manager = SimpleNamespace(
        configs={}, connect_all=Mock(), close_all=Mock(side_effect=lambda: print("connections closed"))
    )
    instance = HPCOrchestrator(
        experiment_file=str(exp_file),
        concurrency_limits={},
        local_log_dir=str(tmp_path / "logs"),
        debug=request.param,
        dashboard_ui="v3",
        dashboard_cls=Dashboard,
        overview_printer=lambda _exps: print("overview rendered"),
        connection_manager=connection_manager,
    )
    data = {"experiments": [{"experiment_id": "example", "status": ExperimentStatus.COMPLETED}]}
    monkeypatch.setattr(instance, "_load_yaml", lambda: data)
    monkeypatch.setattr(instance, "_preflight_validate_experiments", lambda _data: 0)
    monkeypatch.setattr(instance, "_preflight_test_hpcs", lambda: None)
    monkeypatch.setattr(instance, "_recover_orphans", lambda _data: data)
    monkeypatch.setattr(instance, "_catch_up_existing_state", lambda _data: data)
    monkeypatch.setattr(instance, "_maintain_cycle", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(instance, "_sleep_until_next_poll", lambda _dash: True)
    return instance


@pytest.mark.parametrize("status", [ExperimentStatus.COMPLETED, ExperimentStatus.FAILED, None])
def test_completion_reference_is_last_after_cleanup(
    orchestrator: HPCOrchestrator,
    status: ExperimentStatus | None,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Terminal and empty ledgers finish with a plain reference even without INFO logs."""
    caplog.set_level(logging.WARNING, logger="slurminator")
    experiments = orchestrator._load_yaml()["experiments"]
    if status is None:
        experiments.clear()
    else:
        experiments[0]["status"] = status

    orchestrator.run()

    lines = capsys.readouterr().out.splitlines()
    assert lines[-2:] == ["connections closed", f"EXPERIMENT COMPLETE: {orchestrator.experiment_file}"]
    if not orchestrator.debug:
        assert lines[-3] == "dashboard closed"
    orchestrator.connection_manager.close_all.assert_called_once()


def test_interrupt_ends_with_stopped_reference(
    orchestrator: HPCOrchestrator, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An interrupt before the completion check must retain the YAML reference."""
    monkeypatch.setattr(orchestrator, "_maintain_cycle", Mock(side_effect=KeyboardInterrupt))

    orchestrator.run()

    lines = capsys.readouterr().out.splitlines()
    assert lines[-2:] == ["connections closed", f"EXPERIMENT STOPPED: {orchestrator.experiment_file}"]


@pytest.mark.parametrize("orchestrator", [False], indirect=True, ids=["dashboard"])
@pytest.mark.parametrize("exit_during_cycle", [True, False])
def test_dashboard_exit_ends_with_stopped_reference(
    orchestrator: HPCOrchestrator, exit_during_cycle: bool, capsys: pytest.CaptureFixture[str]
) -> None:
    """Both dashboard exit paths retain the reference without claiming completion."""
    orchestrator._load_yaml()["experiments"][0]["status"] = ExperimentStatus.RUNNING
    orchestrator._dashboard_exit_requested = exit_during_cycle

    orchestrator.run()

    lines = capsys.readouterr().out.splitlines()
    assert lines[-3:] == [
        "dashboard closed",
        "connections closed",
        f"EXPERIMENT STOPPED: {orchestrator.experiment_file}",
    ]


@pytest.mark.parametrize("fail_during_cleanup", [False, True])
def test_errors_propagate_without_completion_reference(
    orchestrator: HPCOrchestrator,
    fail_during_cleanup: bool,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Failures, including cleanup failures, must not emit a completion marker."""
    failure = Mock(side_effect=RuntimeError("test failure"))
    if fail_during_cleanup:
        monkeypatch.setattr(orchestrator.connection_manager, "close_all", failure)
    else:
        monkeypatch.setattr(orchestrator, "_maintain_cycle", failure)

    with pytest.raises(RuntimeError, match="test failure"):
        orchestrator.run()

    assert "EXPERIMENT " not in capsys.readouterr().out
    orchestrator.connection_manager.close_all.assert_called_once()

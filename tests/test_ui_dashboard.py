from datetime import date
from types import SimpleNamespace

import pytest

from slurminator.config import HPCType
from slurminator.experiments import ExperimentStatus
from slurminator.ui_dashboard import TerminalDashboard

pytestmark = pytest.mark.unit


def test_dashboard_uses_explicit_timeout_risk_settings() -> None:
    settings = SimpleNamespace(min_progress=0.3, min_runtime_seconds=120, medium_ratio=0.75, high_ratio=0.95)

    dash = TerminalDashboard(timeout_risk_settings=settings)

    assert dash.timeout_risk_min_progress == pytest.approx(0.3)
    assert dash.timeout_risk_min_runtime_seconds == 120
    assert dash.timeout_risk_medium_ratio == pytest.approx(0.75)
    assert dash.timeout_risk_high_ratio == pytest.approx(0.95)


def test_dashboard_project_label_uses_generic_project_keys() -> None:
    dash = TerminalDashboard()

    label = dash._infer_project_label([{"metadata": {"project": "demo"}}])

    assert label == "[cyan]Project[/]: demo"


def test_dashboard_sweep_url_uses_generic_links() -> None:
    dash = TerminalDashboard()

    url = dash._infer_sweep_url([{"links": {"tracker_sweep_url": "https://example.test/sweep/1"}}])

    assert url == "https://example.test/sweep/1"


def test_dashboard_metric_color_uses_display_threshold() -> None:
    dash = TerminalDashboard()

    assert dash._metric_color(0.9, {"higher_better": True, "threshold": 0.8}) == "green"
    assert dash._metric_color(0.7, {"higher_better": True, "threshold": 0.8}) == "red"
    assert dash._metric_color(0.1, {"higher_better": False, "threshold": 0.2}) == "green"


def test_dashboard_active_hpcs_includes_orchestrator_limits() -> None:
    dash = TerminalDashboard()
    dash.orchestrator = SimpleNamespace(concurrency_limits={HPCType.OLIVIA: 1})

    assert dash._active_hpcs([]) == {HPCType.OLIVIA}


def test_dashboard_allocation_period_end_date() -> None:
    assert TerminalDashboard._sigma2_allocation_period_end_date(date(2026, 4, 28)) == date(2026, 9, 30)


def test_dashboard_format_state_for_status_enum() -> None:
    dash = TerminalDashboard()

    formatted = dash._format_state(ExperimentStatus.RUNNING)

    assert "RUNNING" in str(formatted)


def test_dashboard_progress_ignores_project_specific_pseudo_epoch_fields() -> None:
    exp = {"current_pseudo_epoch": 8, "max_pseudo_epochs": 10, "current_epoch": 2, "max_epochs": 10}

    assert TerminalDashboard._resolve_progress_fraction(exp) == pytest.approx(0.2)

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


def test_dashboard_format_state_for_status_enum() -> None:
    dash = TerminalDashboard()

    formatted = dash._format_state(ExperimentStatus.RUNNING)

    assert "RUNNING" in str(formatted)


def test_dashboard_progress_ignores_project_specific_pseudo_epoch_fields() -> None:
    exp = {"current_pseudo_epoch": 8, "max_pseudo_epochs": 10, "current_epoch": 2, "max_epochs": 10}

    assert TerminalDashboard._resolve_progress_fraction(exp) == pytest.approx(0.2)


def test_dashboard_v3_uses_metric_columns_instead_of_full_metric_info() -> None:
    dash = TerminalDashboard(n_recent=1, ui_version="v3")
    exp = {
        "experiment_id": "exp1",
        "status": ExperimentStatus.RUNNING,
        "dataset_name": "demo",
        "hpc_assignment": HPCType.OLIVIA,
        "last_change_ts": 100.0,
        "current_epoch": 1,
        "max_epochs": 2,
        "all_metrics": {"val/acc": 0.91, "val/loss": 0.2, "train/loss": 0.5},
        "display_metric_columns": [
            {"key": "val/acc", "shortform": "acc", "higher_better": True},
            {"key": "val/loss", "shortform": "vloss", "higher_better": False},
        ],
        "display_metric_info": {
            "val/acc": {"shortform": "acc", "higher_better": True},
            "val/loss": {"shortform": "vloss", "higher_better": False},
            "train/loss": {"shortform": "loss", "higher_better": False},
        },
    }

    table = dash._render([exp])["main"].renderable
    headers = [column.header for column in table.columns]

    assert "ACC" in headers
    assert "VLOSS" in headers
    assert "LOSS" not in headers

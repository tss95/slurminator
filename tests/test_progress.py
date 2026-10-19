"""Shared progress behavior for UI and headless consumers."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from slurminator.experiments import ExperimentStatus
from slurminator.progress import display_progress_fraction, overall_run_progress

pytestmark = pytest.mark.unit


def test_step_progress_takes_precedence_over_epoch_progress() -> None:
    """A step-budget run retains its step fraction even when epochs are ahead."""
    row = {"current_step": 25, "max_steps": 100, "current_epoch": 9, "max_epochs": 10}

    assert display_progress_fraction(row) == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("current", "maximum"),
    [
        (None, 100),
        (5, None),
        (5, 0),
        (5, -2),
        ("bad", 100),
        (5, "bad"),
        ([], 100),
        (float("nan"), 100),
        (float("inf"), 100),
        (5, float("nan")),
        (5, float("inf")),
        (True, 100),
        (5, True),
    ],
)
def test_unusable_steps_fall_back_to_epochs(current: object, maximum: object) -> None:
    """Missing or malformed step telemetry does not hide valid epoch progress."""
    row = {"current_step": current, "max_steps": maximum, "current_epoch": 3, "max_epochs": 10}

    assert display_progress_fraction(row, allow_zero=True) == pytest.approx(0.3)


@pytest.mark.parametrize("current", [0, -5])
def test_zero_progress_precedence_depends_on_allow_zero(current: int) -> None:
    """Display callers can retain a zero fraction instead of falling back."""
    row = {"current_step": current, "max_steps": 100, "current_epoch": 3, "max_epochs": 10}

    assert display_progress_fraction(row, allow_zero=True) == 0.0
    assert display_progress_fraction(row) == pytest.approx(0.3)


def test_numeric_strings_and_overshoot_are_supported() -> None:
    """Serialized numeric counters work and completed counters cannot exceed one."""
    assert display_progress_fraction({"current_step": "25", "max_steps": "100"}) == pytest.approx(0.25)
    assert display_progress_fraction({"current_step": 105, "max_steps": 100}) == 1.0


@pytest.mark.parametrize(
    "row",
    [
        {},
        {"current_step": "bad", "max_steps": 10},
        {"current_epoch": 2, "max_epochs": 0},
        {"current_step": float("nan"), "max_steps": 10},
        {"current_step": True, "max_steps": 10},
    ],
)
def test_unknown_progress_is_none(row: dict) -> None:
    """Unknown telemetry remains distinguishable from observed zero progress."""
    assert display_progress_fraction(row, allow_zero=True) is None


def test_mixed_statuses_use_dashboard_denominator_and_partial_progress() -> None:
    """Failure exclusions and submission states preserve the dashboard meaning."""
    rows = [
        {"status": ExperimentStatus.COMPLETED},
        {"status": ExperimentStatus.RUNNING, "current_step": 25, "max_steps": 100},
        {"status": ExperimentStatus.PARTIAL, "current_epoch": 5, "max_epochs": 10},
        {"status": ExperimentStatus.PENDING, "current_step": 100, "max_steps": 100},
        {"status": ExperimentStatus.QUEUED, "current_step": 100, "max_steps": 100},
        {"status": ExperimentStatus.RUNNING},
        *[
            {"status": status, "current_step": 100, "max_steps": 100}
            for status in (
                ExperimentStatus.FAILED,
                ExperimentStatus.CANCELLED,
                ExperimentStatus.TIMEOUT,
                ExperimentStatus.OOM,
                ExperimentStatus.KILLED,
            )
        ],
    ]

    done, eligible, fraction = overall_run_progress(rows)

    assert done == pytest.approx(1.75)
    assert eligible == 6
    assert fraction == pytest.approx(1.75 / 6)


def test_runs_with_different_budgets_have_equal_weight() -> None:
    """Progress measures mean run completion rather than pooled step completion."""
    rows = [
        {"status": ExperimentStatus.RUNNING, "current_step": 50, "max_steps": 100},
        {"status": ExperimentStatus.RUNNING, "current_step": 100, "max_steps": 1000},
    ]

    assert overall_run_progress(rows) == pytest.approx((0.6, 2, 0.3))


def test_canonical_string_statuses_match_enum_statuses() -> None:
    """Persisted reader rows and live dashboard rows produce identical progress."""
    rows = [{"status": status, "current_step": 25, "max_steps": 100} for status in ExperimentStatus]
    serialized_rows = [{**row, "status": row["status"].value} for row in rows]

    assert overall_run_progress(serialized_rows) == overall_run_progress(rows)
    assert overall_run_progress(serialized_rows) == pytest.approx((1.5, 5, 0.3))


@pytest.mark.parametrize("rows", [[], [{"status": ExperimentStatus.FAILED}], [{"status": ExperimentStatus.TIMEOUT}]])
def test_no_eligible_runs_returns_zero(rows: list[dict]) -> None:
    """Empty and entirely failed sweeps have no training progress denominator."""
    assert overall_run_progress(rows) == (0.0, 0, 0.0)


def test_progress_import_does_not_load_ui_or_controller_modules() -> None:
    """The shared contract can be imported in headless environments."""
    package_src = Path(__file__).resolve().parents[1] / "src"
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{package_src}{os.pathsep}{env.get('PYTHONPATH', '')}"
    code = """
import sys
from slurminator.progress import display_progress_fraction, overall_run_progress
assert display_progress_fraction({"current_step": 1, "max_steps": 2}) == 0.5
assert not any(name.split('.')[0] in {"rich", "textual"} for name in sys.modules)
assert "slurminator.hpc_orchestrator" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)

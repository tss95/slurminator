"""UI-independent progress calculations shared by dashboards and observers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

from slurminator.experiments import ExperimentStatus

_FAILED_STATES = frozenset(
    {
        ExperimentStatus.FAILED.value,
        ExperimentStatus.CANCELLED.value,
        ExperimentStatus.TIMEOUT.value,
        ExperimentStatus.OOM.value,
        ExperimentStatus.KILLED.value,
    }
)


def display_progress_fraction(exp: Mapping[str, Any], *, allow_zero: bool = False) -> float | None:
    """Return the step fraction, falling back to epochs when steps are unusable.

    Fractions above one are clamped to one. Nonpositive fractions are usable
    only with ``allow_zero=True``, in which case they contribute zero. Missing,
    nonnumeric, boolean, or nonfinite counters and nonpositive totals are
    unusable.
    """
    for current_key, max_key in (("current_step", "max_steps"), ("current_epoch", "max_epochs")):
        current = exp.get(current_key)
        maximum = exp.get(max_key)
        if current is None or maximum in (None, 0):
            continue
        if isinstance(current, bool) or isinstance(maximum, bool):
            continue
        try:
            current_value = float(current)
            maximum_value = float(maximum)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(current_value) or not math.isfinite(maximum_value) or maximum_value <= 0.0:
            continue
        fraction = current_value / maximum_value
        if fraction <= 0.0:
            if allow_zero:
                return 0.0
            continue
        return min(fraction, 1.0)
    return None


def overall_run_progress(experiments: Sequence[Mapping[str, Any]]) -> tuple[float, int, float]:
    """Return completed-equivalent runs, eligible run count, and mean fraction.

    Rows accept ``ExperimentStatus`` enums or their canonical string values.
    Failed terminal states are excluded. Completed runs contribute one, pending
    and queued runs contribute zero, and other runs use their displayed fraction
    or zero when unknown.
    Every eligible run has equal weight regardless of its training budget.
    An empty eligible set returns ``(0.0, 0, 0.0)``.
    """
    eligible = [exp for exp in experiments if _status_value(exp.get("status")) not in _FAILED_STATES]
    if not eligible:
        return 0.0, 0, 0.0

    progress_sum = 0.0
    for exp in eligible:
        status = _status_value(exp.get("status"))
        if status == ExperimentStatus.COMPLETED.value:
            progress_sum += 1.0
        elif status not in {ExperimentStatus.PENDING.value, ExperimentStatus.QUEUED.value}:
            progress_sum += display_progress_fraction(exp, allow_zero=True) or 0.0

    return progress_sum, len(eligible), progress_sum / len(eligible)


def _status_value(status: object) -> object:
    return status.value if isinstance(status, ExperimentStatus) else status


__all__ = ["display_progress_fraction", "overall_run_progress"]

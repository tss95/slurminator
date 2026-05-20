"""Sparkline renderables for dashboard v4 trajectory cells."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from rich.text import Text

BLOCK_CHARS = "▁▂▃▄▅▆▇█"


@dataclass(frozen=True)
class SparklineThresholds:
    """Thresholds for regression-based sparkline coloring."""

    flat_slope_norm: float = 0.01
    directional_slope_norm: float = 0.02
    oscillation_residual_norm: float = 0.10


def render_sparkline(
    values: Iterable[float],
    *,
    width: int = 20,
    higher_better: bool | None = None,
    thresholds: SparklineThresholds | object | None = None,
) -> Text:
    """Render a single-line block-character sparkline with regression-colored output."""
    safe_width = max(int(width), 1)
    vals = [_coerce_float(value) for value in values]
    vals = [value for value in vals if value is not None][-safe_width:]
    if not vals:
        return Text("-" * safe_width, style="dim")

    lo = min(vals)
    hi = max(vals)
    value_range = hi - lo if hi > lo else 1.0
    n_chars = len(BLOCK_CHARS)
    chars = [BLOCK_CHARS[min(n_chars - 1, int(((value - lo) / value_range) * (n_chars - 1)))] for value in vals]
    chars += ["."] * (safe_width - len(chars))

    color = slope_color(vals, higher_better=higher_better, thresholds=thresholds)
    return Text("".join(chars), style=color)


def slope_color(
    values: Iterable[float], *, higher_better: bool | None, thresholds: SparklineThresholds | object | None = None
) -> str:
    """Return the color style implied by the metric trend."""
    vals = [_coerce_float(value) for value in values]
    vals = [value for value in vals if value is not None]
    if len(vals) < 3 or higher_better is None:
        return "dim"

    cfg = _coerce_thresholds(thresholds)
    slope, intercept = _linear_regression(vals)
    value_range = max(vals) - min(vals) if max(vals) > min(vals) else 1.0
    slope_norm = slope / value_range
    direction = 1.0 if higher_better else -1.0
    signal = direction * slope_norm

    if abs(slope_norm) < cfg.flat_slope_norm:
        return (
            "yellow"
            if _residual_norm(vals, slope=slope, intercept=intercept) >= cfg.oscillation_residual_norm
            else "dim"
        )
    if signal > cfg.directional_slope_norm:
        return "green"
    if signal < -cfg.directional_slope_norm:
        return "red"
    return "yellow"


def _linear_regression(values: list[float]) -> tuple[float, float]:
    n_values = len(values)
    xs = list(range(n_values))
    mean_x = sum(xs) / n_values
    mean_y = sum(values) / n_values
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, values))
    denominator = sum((x - mean_x) ** 2 for x in xs)
    slope = numerator / denominator if denominator else 0.0
    intercept = mean_y - slope * mean_x
    return slope, intercept


def _residual_norm(values: list[float], *, slope: float, intercept: float) -> float:
    value_range = max(values) - min(values)
    if value_range <= 0.0:
        return 0.0
    residuals = [value - (slope * index + intercept) for index, value in enumerate(values)]
    rms = math.sqrt(sum(residual * residual for residual in residuals) / len(residuals))
    return rms / value_range


def _coerce_thresholds(thresholds: SparklineThresholds | object | None) -> SparklineThresholds:
    if thresholds is None:
        return SparklineThresholds()
    if isinstance(thresholds, SparklineThresholds):
        return thresholds
    return SparklineThresholds(
        flat_slope_norm=max(float(getattr(thresholds, "flat_slope_norm", SparklineThresholds.flat_slope_norm)), 0.0),
        directional_slope_norm=max(
            float(getattr(thresholds, "directional_slope_norm", SparklineThresholds.directional_slope_norm)), 0.0
        ),
        oscillation_residual_norm=max(
            float(getattr(thresholds, "oscillation_residual_norm", SparklineThresholds.oscillation_residual_norm)), 0.0
        ),
    )


def _coerce_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


__all__ = ["BLOCK_CHARS", "SparklineThresholds", "render_sparkline", "slope_color"]

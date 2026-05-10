"""Helpers for deriving dashboard display metrics from experiment rows."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def extract_display_metrics(exp: Mapping[str, Any]) -> dict[str, Any]:
    """Extract display shortforms declared in target-schema display metadata."""
    display_metrics: dict[str, Any] = {}
    all_metrics = exp.get("all_metrics", {})
    metric_info = exp.get("display_metric_info", {})
    if not isinstance(all_metrics, Mapping) or not isinstance(metric_info, Mapping):
        return display_metrics

    for metric_key, info in metric_info.items():
        if not isinstance(info, Mapping):
            continue
        shortform = info.get("shortform")
        if not shortform:
            continue
        if shortform in exp:
            display_metrics[str(shortform)] = exp[shortform]  # type: ignore[index]
        elif metric_key in all_metrics:
            display_metrics[str(shortform)] = all_metrics[metric_key]
        elif shortform in all_metrics:
            display_metrics[str(shortform)] = all_metrics[shortform]

    return display_metrics


__all__ = ["extract_display_metrics"]

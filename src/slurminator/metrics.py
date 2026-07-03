"""Metric layout contracts for status history and dashboard display.

The layout separates three metric surfaces:

* ``metrics`` in the status payload: the latest scalar dump.
* ``history_metric_keys``/``history_metric_prefixes``: the subset appended to
  history JSONL.
* ``table_columns``: the ordered dashboard columns.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, TYPE_CHECKING, runtime_checkable

if TYPE_CHECKING:
    from slurminator.callbacks.status_normalization import MetricDisplayCandidate


@dataclass(frozen=True)
class MetricColumnSpec:
    """One ordered dashboard metric column."""

    key: str
    shortform: str | None = None
    higher_better: bool | None = None
    format: str | None = None
    threshold: float | None = None
    best_key: str | None = None

    def display_candidate(self) -> "MetricDisplayCandidate":
        """Return display metadata compatible with the status schema."""
        from slurminator.callbacks.status_normalization import MetricDisplayCandidate

        return MetricDisplayCandidate(
            shortform=self.shortform,
            higher_better=self.higher_better,
            format=self.format,
            threshold=self.threshold,
            best_key=self.best_key,
        )


HistorySelector = Callable[[str], bool]


@dataclass(frozen=True)
class MetricLayout:
    """Metric metadata, dashboard columns, and history filtering policy."""

    metric_info: Mapping[str, "MetricDisplayCandidate"] = field(default_factory=dict)
    table_columns: Sequence[MetricColumnSpec] = field(default_factory=tuple)
    history_metric_keys: frozenset[str] | None = None
    history_metric_prefixes: tuple[str, ...] = ()
    history_selector: HistorySelector | None = None

    @classmethod
    def from_legacy(
        cls,
        *,
        primary_metric: str | None = None,
        secondary_metric: str | None = None,
        metric_info: Mapping[str, "MetricDisplayCandidate"] | None = None,
    ) -> "MetricLayout":
        """Build a layout from the old primary/secondary display contract."""
        info = dict(metric_info or {})
        columns: list[MetricColumnSpec] = []
        for metric_key in (primary_metric, secondary_metric):
            if not metric_key:
                continue
            candidate = info.get(metric_key)
            columns.append(_column_from_candidate(metric_key, candidate))
        return cls(metric_info=info, table_columns=tuple(columns))

    def merged(self, other: "MetricLayout") -> "MetricLayout":
        """Return ``self`` overlaid with another layout.

        Later table columns win ordering for duplicate keys. History filters are
        combined conservatively; ``None`` keeps the default all-history behavior.
        """
        metric_info = {**self.metric_info, **other.metric_info}
        columns_by_key: dict[str, MetricColumnSpec] = {}
        for column in (*self.table_columns, *other.table_columns):
            if column.key:
                columns_by_key[column.key] = column

        self_has_history_policy = self._has_history_policy()
        other_has_history_policy = other._has_history_policy()
        if self_has_history_policy and other_has_history_policy:
            if self.history_metric_keys is None or other.history_metric_keys is None:
                history_keys = None
            else:
                history_keys = frozenset((*self.history_metric_keys, *other.history_metric_keys))
            history_prefixes = tuple(dict.fromkeys((*self.history_metric_prefixes, *other.history_metric_prefixes)))
            history_selector = _combine_selectors(self.history_selector, other.history_selector)
        elif other_has_history_policy:
            history_keys = other.history_metric_keys
            history_prefixes = other.history_metric_prefixes
            history_selector = other.history_selector
        else:
            history_keys = self.history_metric_keys
            history_prefixes = self.history_metric_prefixes
            history_selector = self.history_selector

        return MetricLayout(
            metric_info=metric_info,
            table_columns=tuple(columns_by_key.values()),
            history_metric_keys=history_keys,
            history_metric_prefixes=history_prefixes,
            history_selector=history_selector,
        )

    def history_metrics(self, metrics: Mapping[str, float]) -> dict[str, float]:
        """Return the metrics that should be appended to history."""
        if self.history_metric_keys is None and not self.history_metric_prefixes and self.history_selector is None:
            return dict(metrics)

        selected: dict[str, float] = {}
        keys = self.history_metric_keys or frozenset()
        for key, value in metrics.items():
            if key in keys:
                selected[key] = value
                continue
            if any(key.startswith(prefix) for prefix in self.history_metric_prefixes):
                selected[key] = value
                continue
            if self.history_selector is not None and self.history_selector(key):
                selected[key] = value
        return selected

    def _has_history_policy(self) -> bool:
        return (
            self.history_metric_keys is not None
            or bool(self.history_metric_prefixes)
            or self.history_selector is not None
        )


@runtime_checkable
class MetricLayoutFactory(Protocol):
    """Factory for task/config-aware metric layouts."""

    def build(
        self, *, cfg: object | None = None, trainer: object | None = None, metrics: Mapping[str, float] | None = None
    ) -> MetricLayout:
        """Return the metric layout for the current run context."""


@dataclass(frozen=True)
class StaticMetricLayoutFactory:
    """Factory that always returns the same layout."""

    layout: MetricLayout

    def build(
        self, *, cfg: object | None = None, trainer: object | None = None, metrics: Mapping[str, float] | None = None
    ) -> MetricLayout:
        """Return the static layout."""
        return self.layout


def coerce_metric_layout_factory(value: MetricLayout | MetricLayoutFactory | None) -> MetricLayoutFactory | None:
    """Return a factory for a layout or factory-like object."""
    if value is None:
        return None
    if isinstance(value, MetricLayout):
        return StaticMetricLayoutFactory(value)
    if hasattr(value, "build"):
        return value
    raise TypeError("metric_layout_factory must be a MetricLayout, MetricLayoutFactory, or None.")


def metric_column_from_mapping(value: Mapping[str, object]) -> MetricColumnSpec | None:
    """Coerce a user/plugin mapping into a metric column specification."""
    key = value.get("key")
    if not isinstance(key, str) or not key.strip():
        return None
    direction = str(value.get("direction") or "").strip().lower()
    higher_better = value.get("higher_better")
    if not isinstance(higher_better, bool):
        if direction == "maximize":
            higher_better = True
        elif direction == "minimize":
            higher_better = False
        else:
            higher_better = None
    return MetricColumnSpec(
        key=key.strip(),
        shortform=_optional_string(value.get("shortform")) or _optional_string(value.get("label")),
        higher_better=higher_better,
        format=_optional_string(value.get("format")) or _optional_string(value.get("value_format")),
        threshold=_optional_float(value.get("threshold")),
        best_key=_optional_string(value.get("best_key")),
    )


def _column_from_candidate(metric_key: str, candidate: object | None) -> MetricColumnSpec:
    if candidate is None:
        return MetricColumnSpec(key=metric_key)
    return MetricColumnSpec(
        key=metric_key,
        shortform=getattr(candidate, "shortform", None),
        higher_better=getattr(candidate, "higher_better", None),
        format=getattr(candidate, "format", None),
        threshold=getattr(candidate, "threshold", None),
        best_key=getattr(candidate, "best_key", None),
    )


def _combine_selectors(left: HistorySelector | None, right: HistorySelector | None) -> HistorySelector | None:
    if left is None:
        return right
    if right is None:
        return left

    def combined(metric_key: str) -> bool:
        return left(metric_key) or right(metric_key)

    return combined


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    stripped = value.strip()
    return stripped or None


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


__all__ = [
    "MetricColumnSpec",
    "MetricLayout",
    "MetricLayoutFactory",
    "StaticMetricLayoutFactory",
    "coerce_metric_layout_factory",
    "metric_column_from_mapping",
]

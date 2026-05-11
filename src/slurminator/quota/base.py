"""Quota provider interfaces for dashboard budget checks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Protocol

from slurminator.config import HPCType
from slurminator.config.cluster_registry import coerce_hpc_type


@dataclass(frozen=True)
class QuotaSnapshot:
    """A cluster budget snapshot suitable for dashboard rendering.

    ``worst_case_unit`` should only be set when the dashboard's literal
    walltime-by-resource estimate matches the cluster's charging model closely
    enough to be useful. Set it to ``None`` for partition, node, or site-specific
    billing rules that need a custom estimator.
    """

    hpc_type: HPCType
    cluster_name: str
    resource_label: str
    used: float
    limit: float
    unit: str = "h"
    period_start: date | None = None
    period_end: date | None = None
    worst_case_unit: str | None = "gpu_hours"

    @property
    def remaining(self) -> float:
        """Return remaining quota in ``unit``."""
        return max(self.limit - self.used, 0.0)

    @property
    def used_pct(self) -> float:
        """Return consumed percentage clamped to the display range."""
        if self.limit <= 0.0:
            return 0.0
        return max(0.0, min((self.used / self.limit) * 100.0, 100.0))


class QuotaProvider(Protocol):
    """Fetch quota snapshots for one cluster family."""

    hpc_type: HPCType
    unavailable_hint: str

    def fetch_snapshot(self, *, account: str, connection_manager: Any) -> QuotaSnapshot | None:
        """Return a quota snapshot, or ``None`` when the probe is unavailable."""

    def period_bounds(self, *, today: date | None = None) -> tuple[date, date] | None:
        """Return the current allocation period if the cluster exposes one."""


_PROVIDERS: dict[HPCType, QuotaProvider] = {}


def register_quota_provider(provider: QuotaProvider) -> None:
    """Register or replace the quota provider for one cluster."""
    _PROVIDERS[coerce_hpc_type(provider.hpc_type)] = provider


def get_quota_provider(hpc_type: HPCType | str) -> QuotaProvider | None:
    """Return the registered quota provider for ``hpc_type`` if one exists."""
    return _PROVIDERS.get(coerce_hpc_type(hpc_type))


def get_quota_providers() -> dict[HPCType, QuotaProvider]:
    """Return a copy of the current quota-provider registry."""
    return dict(_PROVIDERS)


def clear_quota_providers() -> None:
    """Clear registered providers for tests or custom bootstrap code."""
    _PROVIDERS.clear()


__all__ = [
    "QuotaProvider",
    "QuotaSnapshot",
    "clear_quota_providers",
    "get_quota_provider",
    "get_quota_providers",
    "register_quota_provider",
]

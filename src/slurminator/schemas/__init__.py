"""Stable JSON schemas used by slurminator."""

from slurminator.schemas.status_schema import (
    ALLOWED_STATUS_TRANSITIONS,
    SCHEMA_VERSION,
    Display,
    HistoryEntry,
    MetricInfo,
    OrchestratorStatus,
    Progress,
    ProgressUnit,
    Speed,
    StatusSchemaVersion,
    StatusState,
    can_transition,
)

__all__ = [
    "ALLOWED_STATUS_TRANSITIONS",
    "SCHEMA_VERSION",
    "Display",
    "HistoryEntry",
    "MetricInfo",
    "OrchestratorStatus",
    "Progress",
    "ProgressUnit",
    "Speed",
    "StatusSchemaVersion",
    "StatusState",
    "can_transition",
]

"""Human-readable context shared by an experiment list and its dashboard."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExperimentInfo:
    """Optional list-wide title and description, rendered as plain text."""

    name: str | None = None
    description: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("name", "description"):
            value = getattr(self, field_name)
            if value is None:
                continue
            if not isinstance(value, str):
                raise ValueError(f"experiment_{field_name} must be a string or null.")
            value = " ".join(value.split()) if field_name == "name" else value.strip()
            object.__setattr__(self, field_name, value or None)

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> ExperimentInfo:
        """Read optional experiment context from config or ledger metadata."""
        return cls(name=metadata.get("experiment_name"), description=metadata.get("experiment_description"))

    def to_metadata(self) -> dict[str, str]:
        """Return only populated fields for persistence in the run ledger."""
        return {
            f"experiment_{field_name}": value
            for field_name in ("name", "description")
            if (value := getattr(self, field_name)) is not None
        }

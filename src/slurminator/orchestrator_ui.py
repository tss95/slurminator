"""Text overview helpers for Slurminator orchestrators."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Sequence

from slurminator.experiments import ExperimentStatus

logger = logging.getLogger("slurminator")


def print_overview(
    experiments: Sequence[dict], *, overview_logger: logging.Logger | None = None, fill_char: str = "█"
) -> None:
    """Log a compact textual overview of experiment statuses."""
    active_logger = overview_logger or logger
    status_counts = Counter(exp["status"] for exp in experiments)
    total = len(experiments)
    done_states = {
        ExperimentStatus.COMPLETED,
        ExperimentStatus.FAILED,
        ExperimentStatus.CANCELLED,
        ExperimentStatus.TIMEOUT,
        ExperimentStatus.OOM,
        ExperimentStatus.PARTIAL,
    }
    done_count = sum(status_counts[s] for s in done_states)
    bar_len = 40
    fill = int(bar_len * done_count / total) if total > 0 else bar_len
    bar = fill_char * fill + "-" * (bar_len - fill)
    breakdown = ", ".join(f"{s.name}={status_counts[s]}" for s in ExperimentStatus if status_counts[s] > 0)
    active_logger.info("Status breakdown: %s", breakdown)
    active_logger.info("Progress: |%s| %d/%d done\n", bar, done_count, total)


__all__ = ["print_overview"]

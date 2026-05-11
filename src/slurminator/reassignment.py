"""Queued experiment reassignment helpers for Slurminator orchestration."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from slurminator.experiment_policy import resolve_pinned_hpc
from slurminator.connection_manager import HPCConnectionManager

logger = logging.getLogger("slurminator")

ReplaceExperiment = Callable[[list[dict[str, Any]], dict[str, Any]], list[dict[str, Any]]]
SaveYaml = Callable[[dict[str, Any]], None]


@dataclass
class ReassignmentContext:
    """Dependencies needed to reassign queued jobs without a full orchestrator."""

    concurrency_limits: Mapping[Any, int]
    hpc_configs: Mapping[Any, Any]
    connection_manager: HPCConnectionManager
    max_unqueue_seconds: int
    pending_status: Any
    queued_status: Any
    partial_status: Any
    replace_exp_in_list: ReplaceExperiment
    save_yaml: SaveYaml


def maybe_reassign_experiments(
    experiments: list[dict[str, Any]],
    concurrency_used: dict[Any, int],
    data: dict[str, Any],
    context: ReassignmentContext,
    *,
    now: float | None = None,
) -> None:
    """Move stale queued experiments from overloaded HPCs to HPCs with free slots."""
    now = time.time() if now is None else now
    free_hpcs: list[tuple[Any, int]] = []
    overloaded_hpcs: list[Any] = []

    for hpc_type, usage in concurrency_used.items():
        limit = context.concurrency_limits.get(hpc_type, 0)
        if limit <= 0:
            continue

        remain = limit - usage
        if remain > 0:
            free_hpcs.append((hpc_type, remain))
        else:
            overloaded_hpcs.append(hpc_type)

    if not free_hpcs or not overloaded_hpcs:
        return

    free_hpcs.sort(key=lambda item: item[1], reverse=True)

    for busy_hpc in overloaded_hpcs:
        queued_exps = [
            exp
            for exp in experiments
            if exp.get("hpc_assignment") == busy_hpc
            and exp.get("status") == context.queued_status
            and ("queued_timestamp" in exp)
            and ((now - exp["queued_timestamp"]) > context.max_unqueue_seconds)
            and exp["status"] != context.partial_status
        ]
        if not queued_exps:
            continue

        for exp in queued_exps:
            if resolve_pinned_hpc(exp, context.hpc_configs):
                logger.info("Skipping reassignment for pinned dataset: %s", exp.get("dataset_name"))
                continue

            new_hpc = next((hpc for hpc, slots in free_hpcs if slots > 0), None)
            if not new_hpc:
                break

            old_job_id = exp.get("job_id")
            if old_job_id:
                logger.info("scancel job %s on HPC %s", old_job_id, busy_hpc)
                try:
                    context.connection_manager.run_command(busy_hpc, f"scancel {old_job_id}", prefer_remote=True)
                    logger.info("Cancelled job %s on HPC %s", old_job_id, busy_hpc)
                except Exception as exc:
                    logger.error("Failed scancel job %s: %s", old_job_id, exc)

            logger.info("Reassigning %s from %s => %s", exp["experiment_id"], busy_hpc, new_hpc)
            exp["hpc_assignment"] = new_hpc
            exp["status"] = context.pending_status
            exp.pop("job_id", None)
            exp.pop("queued_timestamp", None)

            if concurrency_used.get(busy_hpc, 0) > 0:
                concurrency_used[busy_hpc] -= 1
            concurrency_used[new_hpc] += 1

            for idx, (hpc, slots) in enumerate(free_hpcs):
                if hpc == new_hpc:
                    free_hpcs[idx] = (hpc, slots - 1)
                    break

            data["experiments"] = context.replace_exp_in_list(data["experiments"], exp)

    context.save_yaml(data)


__all__ = ["ReassignmentContext", "maybe_reassign_experiments"]

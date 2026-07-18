"""Target status-file ingestion helpers for Slurminator orchestration."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from slurminator.display_helpers import extract_display_metrics as _extract_display_metrics
from slurminator.experiments import ExperimentStatus
from slurminator.history_ingest import read_history_incremental
from slurminator.hpc_state import is_terminal_status
from slurminator.schemas.status_schema import OrchestratorStatus
from slurminator.status_projection import project_status_to_experiment, status_projection_fields

logger = logging.getLogger("slurminator")
HISTORY_TERMINAL_BOUND = 100

LoadYaml = Callable[[], dict[str, Any]]
SaveYaml = Callable[[dict[str, Any]], None]


@dataclass
class StatusIngestContext:
    """Dependencies needed to ingest one target-schema status file."""

    connection_manager: Any
    hpc_configs: Mapping[Any, Any]
    load_yaml: LoadYaml
    save_yaml: SaveYaml
    projection_options: Mapping[str, Any] | None = None
    persist_immediately: bool = True


def status_file_path(save_path: str, job_id: str, sweep_id: object | None = None) -> str:
    """Return the target status-file path for a job."""
    if sweep_id:
        return f"{save_path}/.orchestrator_status/sweep_{sweep_id}/status_{job_id}.json"
    return f"{save_path}/.orchestrator_status/status_{job_id}.json"


def history_file_path(save_path: str, job_id: str, sweep_id: object | None = None) -> str:
    """Return the target history-file path for a job."""
    if sweep_id:
        return f"{save_path}/.orchestrator_status/sweep_{sweep_id}/history_{job_id}.jsonl"
    return f"{save_path}/.orchestrator_status/history_{job_id}.jsonl"


def update_running_experiment_info(exp: dict[str, Any], context: StatusIngestContext) -> dict[str, Any] | None:
    """Read a target status file and project it into an experiment row."""
    job_id = exp.get("job_id")
    hpc_type = exp.get("hpc_assignment")
    if not job_id or not hpc_type:
        return None

    save_path = exp.get("save_path")
    if not save_path:
        cluster_config = context.hpc_configs.get(hpc_type)
        save_path = getattr(cluster_config, "save_path", None) if cluster_config else None
    if not save_path:
        return None
    exp.setdefault("save_path", str(save_path))

    path = status_file_path(str(save_path), str(job_id), exp.get("sweep_id"))
    out, _ = context.connection_manager.run_command(hpc_type, f"cat {path} 2>/dev/null")
    if not out.strip():
        return None

    try:
        status = OrchestratorStatus.model_validate_json(out)
    except Exception as exc:
        logger.debug("Ignoring invalid target status file for job %s: %s", job_id, exc)
        return None

    if str(status.job_id) != str(job_id):
        logger.debug("Ignoring status file job mismatch for %s: payload has %s", job_id, status.job_id)
        return None

    data = status.model_dump(mode="json")
    status_changed = not _status_payload_already_ingested(exp, status)
    if status_changed:
        apply_target_status_to_experiment(exp, status, context.projection_options)
        if not context.persist_immediately:
            exp["last_metrics_update"] = status.last_update
    if _is_active_history_status(exp.get("status")):
        _read_and_merge_history(exp, context)
    elif is_terminal_status(exp.get("status")):
        _bound_terminal_history(exp)
    if status_changed and context.persist_immediately:
        update_experiment_config_with_metrics(exp, data, context)
    return data


def force_read_full_history(exp: dict[str, Any], context: StatusIngestContext) -> None:
    """Force a one-shot full history read for an experiment row."""
    exp["history"] = []
    exp["history_last_read_offset"] = 0
    _read_and_merge_history(exp, context)


def _read_and_merge_history(exp: dict[str, Any], context: StatusIngestContext) -> None:
    """Merge new history entries into ``exp['history']``."""
    job_id = exp.get("job_id")
    hpc_type = exp.get("hpc_assignment")
    save_path = exp.get("save_path")
    if not job_id or not hpc_type or not save_path:
        return
    exp.setdefault("history_truncated", False)

    path = history_file_path(str(save_path), str(job_id), exp.get("sweep_id"))
    last_offset = int(exp.get("history_last_read_offset", 0) or 0)
    try:
        result = read_history_incremental(
            connection_manager=context.connection_manager, hpc_type=hpc_type, history_path=path, last_offset=last_offset
        )
    except Exception as exc:
        logger.debug("History read failed for %s: %s", exp.get("experiment_id"), exc)
        return

    if result.truncated:
        exp["history"] = []
        exp["history_last_read_offset"] = 0

    history = exp.setdefault("history", [])
    history.extend(result.new_entries)
    exp["history_last_read_offset"] = result.new_offset

    if result.new_entries:
        exp["history_attempt_max"] = max(
            int(exp.get("history_attempt_max", 0) or 0), max(int(entry["attempt"]) for entry in result.new_entries)
        )


def _bound_terminal_history(exp: dict[str, Any]) -> None:
    """Trim terminal rows to the retained in-memory history bound."""
    if exp.get("history_truncated"):
        return
    exp.setdefault("history_truncated", False)
    history = exp.get("history", [])
    if len(history) > HISTORY_TERMINAL_BOUND:
        exp["history"] = history[-HISTORY_TERMINAL_BOUND:]
        exp["history_truncated"] = True


def _is_active_history_status(status: Any) -> bool:
    if not isinstance(status, ExperimentStatus):
        try:
            status = ExperimentStatus(str(status))
        except ValueError:
            try:
                status = ExperimentStatus[str(status).upper()]
            except KeyError:
                return False
    return status in {ExperimentStatus.RUNNING, ExperimentStatus.QUEUED}


def _status_payload_already_ingested(exp: Mapping[str, Any], status: OrchestratorStatus) -> bool:
    """Return whether ``exp`` already contains this version of the status payload."""
    last_metrics_update = exp.get("last_metrics_update")
    if isinstance(last_metrics_update, bool):
        return False
    try:
        return float(last_metrics_update) == status.last_update
    except (TypeError, ValueError):
        return False


def apply_target_status_to_experiment(
    exp: dict[str, Any], status: OrchestratorStatus, projection_options: Mapping[str, Any] | None = None
) -> None:
    """Project target-schema status into one experiment row."""
    project_status_to_experiment(exp, status, **dict(projection_options or {}))


def update_experiment_config_with_metrics(
    exp: dict[str, Any], data: dict[str, Any], context: StatusIngestContext
) -> None:
    """Persist projected status fields back to the experiment YAML."""
    try:
        current_data = context.load_yaml()

        for current_exp in current_data["experiments"]:
            if current_exp.get("experiment_id") == exp.get("experiment_id"):
                for key in status_projection_fields(**dict(context.projection_options or {})):
                    if key in exp:
                        current_exp[key] = exp[key]
                current_exp["last_metrics_update"] = data.get("last_update")
                break

        context.save_yaml(current_data)
        exp["last_metrics_update"] = data.get("last_update")
    except Exception as exc:
        if isinstance(exc, ValueError):
            raise
        logger.debug("Failed to update experiment config with metrics: %s", exc)


def extract_display_metrics(exp: Mapping[str, Any]) -> dict[str, Any]:
    """Extract display-friendly metric shortforms from target-schema display metadata."""
    return _extract_display_metrics(exp)


def populate_display_metrics(experiments: list[dict[str, Any]]) -> None:
    """Mutate experiment rows with display metric shortforms before rendering."""
    for exp in experiments:
        exp.update(extract_display_metrics(exp))


__all__ = [
    "HISTORY_TERMINAL_BOUND",
    "StatusIngestContext",
    "apply_target_status_to_experiment",
    "extract_display_metrics",
    "force_read_full_history",
    "history_file_path",
    "populate_display_metrics",
    "status_file_path",
    "update_experiment_config_with_metrics",
    "update_running_experiment_info",
]

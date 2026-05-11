"""Target status-file ingestion helpers for Slurminator orchestration."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from slurminator.display_helpers import extract_display_metrics as _extract_display_metrics
from slurminator.schemas.status_schema import OrchestratorStatus
from slurminator.status_projection import project_status_to_experiment, status_projection_fields

logger = logging.getLogger("slurminator")

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


def status_file_path(save_path: str, job_id: str, sweep_id: object | None = None) -> str:
    """Return the target status-file path for a job."""
    if sweep_id:
        return f"{save_path}/.orchestrator_status/sweep_{sweep_id}/status_{job_id}.json"
    return f"{save_path}/.orchestrator_status/status_{job_id}.json"


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
    apply_target_status_to_experiment(exp, status, context.projection_options)
    update_experiment_config_with_metrics(exp, data, context)
    return data


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
    "StatusIngestContext",
    "apply_target_status_to_experiment",
    "extract_display_metrics",
    "populate_display_metrics",
    "status_file_path",
    "update_experiment_config_with_metrics",
    "update_running_experiment_info",
]

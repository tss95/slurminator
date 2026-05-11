"""Experiment-state YAML loading and persistence for the Slurminator orchestrator."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from slurminator.experiments import ExperimentStatus
from slurminator.config import HPCType
from slurminator.experiments.yaml_utils import dump_yaml, load_yaml

logger = logging.getLogger("slurminator")


class ExperimentStateStore:
    """Load, coerce, and save the orchestrator experiment-state YAML."""

    def __init__(self, experiment_file: str | Path, concurrency_limits: Mapping[HPCType, int] | None = None) -> None:
        self.experiment_file = Path(experiment_file)
        self.concurrency_limits = concurrency_limits or {}

    def load(self) -> dict[str, Any]:
        """Load experiment YAML and ensure a dict with an ``experiments`` list."""
        if not self.experiment_file.exists():
            logger.error("File not found: %s", self.experiment_file)
            return {"experiments": []}

        try:
            data = load_yaml(str(self.experiment_file))
        except Exception as exc:
            data = self._load_backup_after_parse_error(exc)

        if not isinstance(data, dict) or not data:
            logger.warning("Experiment file is empty or invalid. Defaulting to empty experiment list.")
            data = {"experiments": []}
        elif "experiments" not in data:
            data["experiments"] = []

        self._coerce_experiment_rows(data["experiments"])
        return data

    def save(self, data: Mapping[str, Any]) -> None:
        """Persist experiment YAML using the package YAML dumper."""
        dump_yaml(dict(data), str(self.experiment_file))
        logger.debug("Saved updated YAML => %s", self.experiment_file)

    def _load_backup_after_parse_error(self, exc: Exception) -> dict[str, Any]:
        """Try to recover from a truncated/corrupted YAML file via its ``.bak``."""
        import yaml as _yaml

        logger.error("Failed to parse experiment YAML (%s): %s – attempting to load backup.", self.experiment_file, exc)
        bak_path = self.experiment_file.with_suffix(self.experiment_file.suffix + ".bak")
        if bak_path.exists():
            try:
                data = load_yaml(str(bak_path))
                logger.warning("Loaded backup YAML: %s", bak_path)
                return data
            except _yaml.YAMLError:
                logger.critical("Backup YAML (%s) also corrupted. Resetting experiments list.", bak_path)
                return {"experiments": []}

        logger.critical("No backup YAML found. Resetting experiments list.")
        return {"experiments": []}

    def _coerce_experiment_rows(self, experiments: object) -> None:
        """Coerce status/HPC enum-like strings and clear disabled assignments in-place."""
        if not isinstance(experiments, list):
            return

        for exp in experiments:
            if not isinstance(exp, dict):
                continue

            status = exp.get("status")
            if isinstance(status, str):
                try:
                    exp["status"] = ExperimentStatus(status)
                except ValueError:
                    logger.warning(
                        "Invalid status '%s' found in YAML for exp %s. Keeping raw string.",
                        status,
                        exp.get("experiment_id"),
                    )

            hpc_assignment = exp.get("hpc_assignment")
            if isinstance(hpc_assignment, str):
                try:
                    exp["hpc_assignment"] = HPCType(hpc_assignment)
                    hpc_assignment = exp["hpc_assignment"]
                except ValueError:
                    logger.warning(
                        "Invalid hpc_assignment '%s' found in YAML for exp %s. Keeping raw string.",
                        hpc_assignment,
                        exp.get("experiment_id"),
                    )

            if hpc_assignment and self.concurrency_limits.get(hpc_assignment, 0) == 0:
                logger.warning(
                    "Experiment %s assigned to disabled HPC %s - resetting to PENDING",
                    exp.get("experiment_id"),
                    hpc_assignment,
                )
                exp["hpc_assignment"] = None
                exp["status"] = ExperimentStatus.PENDING


def replace_exp_in_list(experiments: list[dict[str, Any]], new_exp: dict[str, Any]) -> list[dict[str, Any]]:
    """Return ``experiments`` with the matching experiment_id replaced."""
    uid = new_exp["experiment_id"]
    return [new_exp if exp["experiment_id"] == uid else exp for exp in experiments]


__all__ = ["ExperimentStateStore", "replace_exp_in_list"]

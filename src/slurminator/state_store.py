"""Experiment-state YAML loading and persistence for the Slurminator orchestrator."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from slurminator.config import HPCType
from slurminator.experiments import ExperimentStatus
from slurminator.experiments.yaml_utils import dump_yaml, load_yaml
from slurminator.hpc_state import is_terminal_status
from slurminator.lifecycle_journal import SchedulerLifecycleJournal, SchedulerLifecycleTransition
from slurminator.scheduler_polling import SchedulerTransition
from slurminator.submission_journal import SubmissionReceipt, SubmissionReceiptJournal

logger = logging.getLogger("slurminator")


class ExperimentStateStore:
    """Load, coerce, and save the orchestrator experiment-state YAML."""

    def __init__(self, experiment_file: str | Path, concurrency_limits: Mapping[HPCType, int] | None = None) -> None:
        self.experiment_file = Path(experiment_file)
        self.concurrency_limits = concurrency_limits or {}
        self.submission_journal = SubmissionReceiptJournal(self.experiment_file)
        self.lifecycle_journal = SchedulerLifecycleJournal(self.experiment_file)
        self._warned_unpolled_hpcs: set[object] = set()
        self._reported_replayed_journals: set[Path] = set()

    def load(self) -> dict[str, Any]:
        """Load experiment YAML and ensure a dict with an ``experiments`` list."""
        if not self.experiment_file.exists():
            remaining_journals = [
                journal.path for journal in (self.submission_journal, self.lifecycle_journal) if journal.path.exists()
            ]
            if remaining_journals:
                raise FileNotFoundError(
                    f"Experiment ledger {self.experiment_file} is missing while durable journal state remains at "
                    f"{', '.join(str(path) for path in remaining_journals)}"
                )
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
        superseded_jobs = self._superseded_submission_jobs()
        replayed_receipts = self.submission_journal.apply(data)
        replayed_transitions = self.lifecycle_journal.apply(data, superseded_jobs=superseded_jobs)
        self._warn_about_zero_limit_active_rows(data["experiments"])
        if replayed_receipts and self.submission_journal.path not in self._reported_replayed_journals:
            logger.warning(
                "Replayed %d accepted submission(s) from %s in memory; they will be compacted at the next "
                "explicit checkpoint.",
                replayed_receipts,
                self.submission_journal.path,
            )
            self._reported_replayed_journals.add(self.submission_journal.path)
        if replayed_transitions and self.lifecycle_journal.path not in self._reported_replayed_journals:
            logger.warning(
                "Replayed %d scheduler lifecycle transition(s) from %s in memory; they will be compacted at the "
                "next explicit checkpoint.",
                replayed_transitions,
                self.lifecycle_journal.path,
            )
            self._reported_replayed_journals.add(self.lifecycle_journal.path)
        return data

    def save(self, data: Mapping[str, Any]) -> None:
        """Merge durable journals, then persist and checkpoint the ledger."""
        checkpoint = dict(data)
        submission_journal_exists = self.submission_journal.path.exists()
        lifecycle_journal_exists = self.lifecycle_journal.path.exists()
        superseded_jobs = self._superseded_submission_jobs()
        self.submission_journal.apply(checkpoint)
        self.lifecycle_journal.apply(checkpoint, superseded_jobs=superseded_jobs)
        dump_yaml(checkpoint, str(self.experiment_file))
        if submission_journal_exists or lifecycle_journal_exists:
            # ``dump_yaml`` atomically replaces the file. Make both its data
            # and directory entry durable before deleting the only compact
            # record of accepted submissions and scheduler transitions.
            _fsync_file(self.experiment_file)
            _fsync_directory(self.experiment_file.parent)
            if lifecycle_journal_exists:
                self.lifecycle_journal.clear()
                self._reported_replayed_journals.discard(self.lifecycle_journal.path)
            if submission_journal_exists:
                # Clear retry provenance last so interrupted cleanup retains
                # the old-to-new job identity needed for strict replay.
                self.submission_journal.clear()
                self._reported_replayed_journals.discard(self.submission_journal.path)
        logger.debug("Saved updated YAML => %s", self.experiment_file)

    def record_submission(self, experiment: Mapping[str, Any], previous_job_id: str | None = None) -> SubmissionReceipt:
        """Durably record one accepted submission without rewriting the full ledger."""
        return self.submission_journal.append_experiment(experiment, previous_job_id)

    def record_scheduler_transitions(
        self, transitions: Iterable[SchedulerTransition]
    ) -> list[SchedulerLifecycleTransition]:
        """Durably record one scheduler poll's lifecycle changes with one fsync."""
        return self.lifecycle_journal.append_transitions(transitions)

    def _superseded_submission_jobs(self) -> set[tuple[str, str]]:
        """Return old job identities explicitly replaced by durable receipts."""
        return {
            (receipt.experiment_id, receipt.previous_job_id)
            for receipt in self.submission_journal.read()
            if receipt.previous_job_id is not None
        }

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

            if (
                hpc_assignment
                and self.concurrency_limits.get(hpc_assignment, 0) == 0
                and not exp.get("job_id")
                and not is_terminal_status(exp.get("status"))
            ):
                logger.warning(
                    "Experiment %s assigned to disabled HPC %s - resetting to PENDING",
                    exp.get("experiment_id"),
                    hpc_assignment,
                )
                exp["hpc_assignment"] = None
                exp["status"] = ExperimentStatus.PENDING

    def _warn_about_zero_limit_active_rows(self, experiments: object) -> None:
        """Warn about accepted active jobs whose HPC has a zero submission limit."""
        if not isinstance(experiments, list):
            return

        unpolled_by_hpc: dict[object, list[dict[str, Any]]] = {}
        for exp in experiments:
            if not isinstance(exp, dict):
                continue
            hpc_assignment = exp.get("hpc_assignment")
            if (
                exp.get("status") in {ExperimentStatus.QUEUED, ExperimentStatus.RUNNING}
                and exp.get("job_id")
                and hpc_assignment
                and self.concurrency_limits.get(hpc_assignment, 0) == 0
            ):
                unpolled_by_hpc.setdefault(hpc_assignment, []).append(exp)

        active_hpcs = set(unpolled_by_hpc)
        self._warned_unpolled_hpcs.intersection_update(active_hpcs)
        for hpc_assignment, rows in unpolled_by_hpc.items():
            if hpc_assignment in self._warned_unpolled_hpcs:
                continue
            example_ids = ", ".join(str(exp.get("experiment_id")) for exp in rows[:3])
            logger.warning(
                "%d accepted active experiment(s) on disabled HPC %s are preserving scheduler state. "
                "Existing jobs remain pollable only if that HPC was connected at startup; otherwise restart "
                "with a positive limit (examples: %s).",
                len(rows),
                hpc_assignment,
                example_ids,
            )
            self._warned_unpolled_hpcs.add(hpc_assignment)


def _fsync_file(path: Path) -> None:
    """Make the contents of one existing file durable."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_directory(path: Path) -> None:
    """Make a directory-entry update durable."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def replace_exp_in_list(experiments: list[dict[str, Any]], new_exp: dict[str, Any]) -> list[dict[str, Any]]:
    """Return ``experiments`` with the matching experiment_id replaced."""
    uid = new_exp["experiment_id"]
    return [new_exp if exp["experiment_id"] == uid else exp for exp in experiments]


__all__ = ["ExperimentStateStore", "replace_exp_in_list"]

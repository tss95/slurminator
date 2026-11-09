"""Durable scheduler lifecycle transitions between full YAML checkpoints."""

from __future__ import annotations

import math
import os
import uuid
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from slurminator.experiments import ExperimentStatus
from slurminator.hpc_state import is_terminal_status
from slurminator.scheduler_polling import SchedulerTransition


_STATUS_TIMESTAMP_FIELDS: Mapping[ExperimentStatus, str] = {
    ExperimentStatus.RUNNING: "running_timestamp",
    ExperimentStatus.COMPLETED: "completed_timestamp",
    ExperimentStatus.FAILED: "failed_timestamp",
    ExperimentStatus.CANCELLED: "cancelled_timestamp",
    ExperimentStatus.TIMEOUT: "timeout_timestamp",
    ExperimentStatus.OOM: "failed_timestamp",
    ExperimentStatus.KILLED: "killed_timestamp",
}
_SCHEDULER_TIMESTAMP_FIELDS = frozenset(_STATUS_TIMESTAMP_FIELDS.values())


class SchedulerLifecycleTransition(BaseModel):
    """Immutable scheduler-owned status change for one submitted job."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: Literal["1.0"] = "1.0"
    transition_id: str = Field(min_length=1)
    experiment_id: str = Field(min_length=1)
    job_id: str = Field(min_length=1)
    old_status: ExperimentStatus
    new_status: ExperimentStatus
    timestamp: float = Field(gt=0)
    timestamp_field: str | None = None

    @model_validator(mode="after")
    def validate_scheduler_transition(self) -> "SchedulerLifecycleTransition":
        """Reject transitions or timestamp fields not owned by scheduler polling."""
        if self.old_status not in {ExperimentStatus.QUEUED, ExperimentStatus.RUNNING}:
            raise ValueError(f"Scheduler transition has invalid old_status {self.old_status.value!r}")
        if self.new_status in {ExperimentStatus.PENDING, ExperimentStatus.PARTIAL}:
            raise ValueError(
                f"Scheduler lifecycle journal cannot record non-scheduler status {self.new_status.value!r}"
            )
        expected_timestamp_field = _STATUS_TIMESTAMP_FIELDS.get(self.new_status)
        if self.timestamp_field != expected_timestamp_field:
            raise ValueError(
                f"Scheduler transition to {self.new_status.value!r} requires timestamp_field "
                f"{expected_timestamp_field!r}, got {self.timestamp_field!r}"
            )
        return self

    @classmethod
    def from_scheduler_transition(cls, transition: SchedulerTransition) -> "SchedulerLifecycleTransition":
        """Build a durable record from the scheduler poller's transition contract."""
        return cls(
            transition_id=uuid.uuid4().hex,
            experiment_id=_required_string(transition.experiment_id, "experiment_id"),
            job_id=_required_string(transition.job_id, "job_id"),
            old_status=transition.old_status,
            new_status=transition.new_status,
            timestamp=_required_timestamp(transition.timestamp, "timestamp"),
            timestamp_field=transition.timestamp_field,
        )

    def row_patch(self) -> dict[str, Any]:
        """Return the scheduler-owned row fields represented by this transition."""
        patch: dict[str, Any] = {"status": self.new_status, "last_change_ts": self.timestamp}
        if self.timestamp_field is not None:
            patch[self.timestamp_field] = self.timestamp
        return patch


class SchedulerLifecycleJournal:
    """Append and replay fsynced scheduler transitions for one controller writer."""

    def __init__(self, experiment_file: str | Path) -> None:
        experiment_path = Path(experiment_file)
        self.path = (
            experiment_path.parent / ".orchestrator_status" / "_scheduler_lifecycle" / f"{experiment_path.name}.jsonl"
        )

    def append_transitions(
        self, scheduler_transitions: Iterable[SchedulerTransition]
    ) -> list[SchedulerLifecycleTransition]:
        """Append one poll's transitions with a single write sequence and fsync."""
        transitions = [
            SchedulerLifecycleTransition.from_scheduler_transition(transition) for transition in scheduler_transitions
        ]
        if not transitions:
            return []

        payload = b"".join((transition.model_dump_json() + "\n").encode("utf-8") for transition in transitions)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        journal_exists = self.path.exists()
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            view = memoryview(payload)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise OSError(f"Could not append scheduler lifecycle transitions to {self.path}")
                view = view[written:]
            os.fsync(fd)
        finally:
            os.close(fd)
        if not journal_exists:
            _fsync_directory(self.path.parent)
        return transitions

    def read(self) -> list[SchedulerLifecycleTransition]:
        """Read and strictly validate all durable scheduler transitions."""
        if not self.path.exists():
            return []
        payload = self.path.read_bytes()
        if not payload:
            raise ValueError(f"Scheduler lifecycle journal is unexpectedly empty: {self.path}")
        if not payload.endswith(b"\n"):
            raise ValueError(f"Scheduler lifecycle journal has a torn final record: {self.path}")

        transitions: list[SchedulerLifecycleTransition] = []
        transition_ids: set[str] = set()
        latest_timestamp_by_job: dict[tuple[str, str], float] = {}
        for line_number, line in enumerate(payload.splitlines(), start=1):
            if not line:
                raise ValueError(f"Scheduler lifecycle journal has a blank record at line {line_number}: {self.path}")
            try:
                transition = SchedulerLifecycleTransition.model_validate_json(line)
            except Exception as exc:
                raise ValueError(f"Invalid scheduler lifecycle transition at {self.path}:{line_number}: {exc}") from exc
            if transition.transition_id in transition_ids:
                raise ValueError(
                    f"Duplicate scheduler lifecycle transition_id {transition.transition_id!r} in {self.path}"
                )
            transition_ids.add(transition.transition_id)
            job_key = (transition.experiment_id, transition.job_id)
            previous_timestamp = latest_timestamp_by_job.get(job_key)
            if previous_timestamp is not None and transition.timestamp < previous_timestamp:
                raise ValueError(
                    f"Out-of-order scheduler lifecycle transition for experiment {transition.experiment_id!r}, "
                    f"job {transition.job_id!r} in {self.path}: {transition.timestamp} < {previous_timestamp}"
                )
            latest_timestamp_by_job[job_key] = transition.timestamp
            transitions.append(transition)
        return transitions

    def apply(self, data: dict[str, Any], *, superseded_jobs: set[tuple[str, str]] | None = None) -> int:
        """Idempotently replay transitions without overwriting newer job state."""
        transitions = self.read()
        if not transitions:
            return 0

        rows = _index_experiment_rows(data, self.path)
        projected_rows = {experiment_id: dict(exp) for experiment_id, exp in rows.items()}
        touched_experiments: set[str] = set()
        known_superseded_jobs = superseded_jobs or set()

        # Validate and project the complete journal before mutating the caller's
        # ledger. A malformed/conflicting late record therefore cannot leave a
        # partially replayed in-memory state behind.
        for transition in transitions:
            projected = projected_rows.get(transition.experiment_id)
            if projected is None:
                raise ValueError(
                    f"Scheduler lifecycle transition references missing experiment "
                    f"{transition.experiment_id!r} in {self.path}"
                )
            if _apply_transition(projected, transition, superseded_jobs=known_superseded_jobs):
                touched_experiments.add(transition.experiment_id)

        for experiment_id in touched_experiments:
            projected = projected_rows[experiment_id]
            target = rows[experiment_id]
            for field_name in ("status", "last_change_ts", *_SCHEDULER_TIMESTAMP_FIELDS):
                if field_name in projected:
                    target[field_name] = projected[field_name]
        return len(transitions)

    def clear(self) -> None:
        """Remove transitions only after their state is durable in the main ledger."""
        if not self.path.exists():
            return
        self.path.unlink()
        _fsync_directory(self.path.parent)


def _index_experiment_rows(data: Mapping[str, Any], journal_path: Path) -> dict[str, dict[str, Any]]:
    """Index valid experiment rows and reject ambiguous duplicate identities."""
    rows: dict[str, dict[str, Any]] = {}
    experiments = data.get("experiments", [])
    if not isinstance(experiments, list):
        raise ValueError(f"Experiment ledger has a non-list 'experiments' value while replaying {journal_path}")
    for exp in experiments:
        if not isinstance(exp, dict) or exp.get("experiment_id") is None:
            continue
        experiment_id = str(exp["experiment_id"])
        if experiment_id in rows:
            raise ValueError(f"Duplicate experiment_id {experiment_id!r} in ledger while replaying {journal_path}")
        rows[experiment_id] = exp
    return rows


def _apply_transition(
    row: dict[str, Any], transition: SchedulerLifecycleTransition, *, superseded_jobs: set[tuple[str, str]]
) -> bool:
    """Project one transition, returning whether scheduler-owned fields changed."""
    current_job_id = _optional_string(row.get("job_id"))
    if current_job_id != transition.job_id:
        transition_key = (transition.experiment_id, transition.job_id)
        if transition_key in superseded_jobs:
            return False
        if current_job_id is not None and _is_transition_from_older_attempt(row, transition):
            return False
        raise ValueError(
            f"Scheduler lifecycle transition conflicts with ledger job_id for {transition.experiment_id!r}: "
            f"{transition.job_id!r} != {current_job_id!r}"
        )

    current_status = _coerce_status(row.get("status"))
    current_timestamp = _optional_timestamp(row.get("last_change_ts"))

    # PARTIAL denotes a post-scheduler retry decision, while terminal states
    # are final for a job attempt. Neither may be moved backwards by a journal
    # record left behind after an already-durable checkpoint.
    if current_status == ExperimentStatus.PARTIAL:
        if transition.new_status == ExperimentStatus.TIMEOUT:
            return _merge_historical_timestamp_field(row, transition)
        return False
    if current_status == ExperimentStatus.PENDING:
        raise ValueError(
            f"Scheduler lifecycle transition targets pending experiment {transition.experiment_id!r} "
            f"with job_id {transition.job_id!r}"
        )

    if is_terminal_status(current_status):
        if current_status != transition.new_status:
            if transition.new_status == ExperimentStatus.RUNNING and (
                current_timestamp is None or current_timestamp >= transition.timestamp
            ):
                return _merge_historical_timestamp_field(row, transition)
            return False
        if current_timestamp is not None and current_timestamp != transition.timestamp:
            return False
        return _merge_matching_transition_fields(row, transition)

    if current_timestamp is not None:
        if current_timestamp > transition.timestamp:
            return False
        if current_timestamp == transition.timestamp:
            if current_status == transition.new_status:
                return _merge_matching_transition_fields(row, transition)
            raise ValueError(
                f"Conflicting scheduler lifecycle states for experiment {transition.experiment_id!r}, "
                f"job {transition.job_id!r} at {transition.timestamp}: "
                f"{current_status.value!r} != {transition.new_status.value!r}"
            )
    elif _lifecycle_rank(current_status) > _lifecycle_rank(transition.new_status):
        return False

    if current_status not in {transition.old_status, transition.new_status}:
        raise ValueError(
            f"Scheduler lifecycle transition for experiment {transition.experiment_id!r}, job "
            f"{transition.job_id!r} expected old status {transition.old_status.value!r}, found "
            f"{current_status.value!r}"
        )
    row.update(transition.row_patch())
    return True


def _merge_matching_transition_fields(row: dict[str, Any], transition: SchedulerLifecycleTransition) -> bool:
    """Fill fields from an idempotent matching transition without regression."""
    changed = False
    if "last_change_ts" not in row:
        row["last_change_ts"] = transition.timestamp
        changed = True
    if transition.timestamp_field is not None and transition.timestamp_field not in row:
        row[transition.timestamp_field] = transition.timestamp
        changed = True
    return changed


def _merge_historical_timestamp_field(row: dict[str, Any], transition: SchedulerLifecycleTransition) -> bool:
    """Fill one missing historical timestamp without changing current lifecycle state."""
    if transition.timestamp_field is None or transition.timestamp_field in row:
        return False
    row[transition.timestamp_field] = transition.timestamp
    return True


def _is_transition_from_older_attempt(row: Mapping[str, Any], transition: SchedulerLifecycleTransition) -> bool:
    """Return whether the ledger clearly identifies a newer retry attempt."""
    queued_timestamp = _optional_timestamp(row.get("queued_timestamp"))
    return queued_timestamp is not None and queued_timestamp >= transition.timestamp


def _lifecycle_rank(status: ExperimentStatus) -> int:
    """Return the monotonic rank used only when a durable timestamp is absent."""
    if status == ExperimentStatus.QUEUED:
        return 1
    if status == ExperimentStatus.RUNNING:
        return 2
    if status == ExperimentStatus.PARTIAL or is_terminal_status(status):
        return 3
    return 0


def _coerce_status(value: object) -> ExperimentStatus:
    """Strictly coerce enum/value/name representations to ``ExperimentStatus``."""
    if isinstance(value, ExperimentStatus):
        return value
    text = str(value or "").strip()
    if text.startswith("ExperimentStatus."):
        text = text.split(".", 1)[1]
    try:
        return ExperimentStatus(text)
    except ValueError:
        try:
            return ExperimentStatus[text.upper()]
        except KeyError as exc:
            raise ValueError(f"Invalid experiment lifecycle status {value!r}") from exc


def _required_string(value: object, field_name: str) -> str:
    text = _optional_string(value)
    if text is None:
        raise ValueError(f"Scheduler lifecycle transition requires a non-empty {field_name}")
    return text


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _required_timestamp(value: object, field_name: str) -> float:
    timestamp = _optional_timestamp(value)
    if timestamp is None or timestamp <= 0:
        raise ValueError(f"Scheduler lifecycle transition requires a positive finite {field_name}")
    return timestamp


def _optional_timestamp(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        timestamp = float(value)
    except (TypeError, ValueError):
        return None
    return timestamp if math.isfinite(timestamp) else None


def _fsync_directory(path: Path) -> None:
    """Fsync directory metadata after creating or removing a journal."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


__all__ = ["SchedulerLifecycleJournal", "SchedulerLifecycleTransition"]

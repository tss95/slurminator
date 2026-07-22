"""Durable, compact submission receipts for batched ledger persistence."""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field

from slurminator.config import HPCType
from slurminator.experiments import ExperimentStatus


class SubmissionReceipt(BaseModel):
    """Immutable state needed to recover one accepted Slurm submission."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    receipt_id: str = Field(min_length=1)
    experiment_id: str = Field(min_length=1)
    hpc_assignment: str = Field(min_length=1)
    previous_job_id: str | None = None
    job_id: str = Field(min_length=1)
    queued_timestamp: float = Field(gt=0)
    git_sha_at_submission: dict[str, str | None] = Field(default_factory=dict)
    output_dir: str | None = None
    save_path: str | None = None
    requested_time_hours: int | None = None
    requested_ram_gb: int | None = None
    requested_gpu_count: int | None = None

    @classmethod
    def from_experiment(cls, exp: Mapping[str, Any], previous_job_id: str | None = None) -> "SubmissionReceipt":
        """Build a receipt from a row after ``sbatch`` has been accepted."""
        hpc_assignment = exp.get("hpc_assignment")
        if isinstance(hpc_assignment, HPCType):
            hpc_assignment = hpc_assignment.value

        provenance = exp.get("git_sha_at_submission")
        if not isinstance(provenance, Mapping):
            provenance = {}

        return cls(
            receipt_id=uuid.uuid4().hex,
            experiment_id=str(exp.get("experiment_id", "")),
            hpc_assignment=str(hpc_assignment or ""),
            previous_job_id=_optional_string(previous_job_id),
            job_id=str(exp.get("job_id", "")),
            queued_timestamp=float(exp.get("queued_timestamp", 0.0) or 0.0),
            git_sha_at_submission={
                str(key): None if value is None else str(value) for key, value in provenance.items()
            },
            output_dir=_optional_string(exp.get("output_dir")),
            save_path=_optional_string(exp.get("save_path")),
            requested_time_hours=_optional_int(exp.get("requested_time_hours")),
            requested_ram_gb=_optional_int(exp.get("requested_ram_gb")),
            requested_gpu_count=_optional_int(exp.get("requested_gpu_count")),
        )

    def row_patch(self) -> dict[str, Any]:
        """Return the experiment-row fields made durable by this receipt."""
        patch: dict[str, Any] = {
            "hpc_assignment": HPCType(self.hpc_assignment),
            "status": ExperimentStatus.QUEUED,
            "job_id": self.job_id,
            "queued_timestamp": self.queued_timestamp,
            "git_sha_at_submission": dict(self.git_sha_at_submission),
        }
        for field_name in (
            "output_dir",
            "save_path",
            "requested_time_hours",
            "requested_ram_gb",
            "requested_gpu_count",
        ):
            value = getattr(self, field_name)
            if value is not None:
                patch[field_name] = value
        return patch


class SubmissionReceiptJournal:
    """Append and replay fsynced receipts for one single-writer experiment ledger."""

    def __init__(self, experiment_file: str | Path) -> None:
        experiment_path = Path(experiment_file)
        self.path = (
            experiment_path.parent / ".orchestrator_status" / "_submission_receipts" / f"{experiment_path.name}.jsonl"
        )

    def append_experiment(self, exp: Mapping[str, Any], previous_job_id: str | None = None) -> SubmissionReceipt:
        """Append and fsync one accepted submission before another job is submitted."""
        receipt = SubmissionReceipt.from_experiment(exp, previous_job_id)
        payload = (receipt.model_dump_json() + "\n").encode("utf-8")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        journal_exists = self.path.exists()
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            view = memoryview(payload)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise OSError(f"Could not append submission receipt to {self.path}")
                view = view[written:]
            os.fsync(fd)
        finally:
            os.close(fd)
        if not journal_exists:
            _fsync_directory(self.path.parent)
        return receipt

    def read(self) -> list[SubmissionReceipt]:
        """Read and strictly validate all durable receipts."""
        if not self.path.exists():
            return []
        payload = self.path.read_bytes()
        if not payload:
            raise ValueError(f"Submission receipt journal is unexpectedly empty: {self.path}")
        if not payload.endswith(b"\n"):
            raise ValueError(f"Submission receipt journal has a torn final record: {self.path}")

        receipts: list[SubmissionReceipt] = []
        for line_number, line in enumerate(payload.splitlines(), start=1):
            if not line:
                raise ValueError(f"Submission receipt journal has a blank record at line {line_number}: {self.path}")
            try:
                receipts.append(SubmissionReceipt.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"Invalid submission receipt at {self.path}:{line_number}: {exc}") from exc
        return receipts

    def apply(self, data: dict[str, Any]) -> int:
        """Idempotently replay receipts into an in-memory experiment ledger."""
        receipts = self.read()
        if not receipts:
            return 0

        rows: dict[str, dict[str, Any]] = {}
        for exp in data.get("experiments", []):
            if not isinstance(exp, dict) or exp.get("experiment_id") is None:
                continue
            experiment_id = str(exp["experiment_id"])
            if experiment_id in rows:
                raise ValueError(f"Duplicate experiment_id {experiment_id!r} in ledger while replaying {self.path}")
            rows[experiment_id] = exp

        seen_receipt_ids: set[str] = set()
        seen_job_ids: dict[str, str] = {}
        patches: list[dict[str, Any]] = []
        for receipt in receipts:
            if receipt.receipt_id in seen_receipt_ids:
                raise ValueError(f"Duplicate submission receipt_id {receipt.receipt_id!r} in {self.path}")
            seen_receipt_ids.add(receipt.receipt_id)

            previous_job_id = seen_job_ids.setdefault(receipt.experiment_id, receipt.job_id)
            if previous_job_id != receipt.job_id:
                raise ValueError(
                    f"Conflicting submission receipts for experiment {receipt.experiment_id!r}: "
                    f"{previous_job_id!r} != {receipt.job_id!r}"
                )

            exp = rows.get(receipt.experiment_id)
            if exp is None:
                raise ValueError(
                    f"Submission receipt references missing experiment {receipt.experiment_id!r} in {self.path}"
                )

            current_job_id = exp.get("job_id")
            if current_job_id is not None and str(current_job_id) != receipt.job_id:
                can_apply_retry = (
                    receipt.previous_job_id is not None
                    and str(current_job_id) == receipt.previous_job_id
                    and exp.get("status") in {ExperimentStatus.PENDING, ExperimentStatus.PARTIAL}
                )
                if not can_apply_retry:
                    raise ValueError(
                        f"Submission receipt conflicts with ledger job_id for {receipt.experiment_id!r}: "
                        f"{receipt.job_id!r} != {current_job_id!r}"
                    )
            patches.append(receipt.row_patch())

        for receipt, patch in zip(receipts, patches, strict=True):
            exp = rows[receipt.experiment_id]
            current_job_id = exp.get("job_id")
            if current_job_id is not None and str(current_job_id) == receipt.job_id:
                # The main ledger may contain scheduler state newer than the
                # original submission. Once it identifies the same job, the
                # receipt must never move that lifecycle state backwards.
                patch.pop("status")
            exp.update(patch)
        return len(receipts)

    def clear(self) -> None:
        """Remove receipts only after their state is durable in the main ledger."""
        if not self.path.exists():
            return
        self.path.unlink()
        _fsync_directory(self.path.parent)


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _fsync_directory(path: Path) -> None:
    """Fsync directory metadata after creating, appending, or removing a journal."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


__all__ = ["SubmissionReceipt", "SubmissionReceiptJournal"]

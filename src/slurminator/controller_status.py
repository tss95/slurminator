"""Atomic, compact status receipts for observers of a headless controller."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import socket
import tempfile
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, IO

from slurminator.experiments import ExperimentStatus
from slurminator.progress import display_progress_fraction, overall_run_progress

SCHEMA_VERSION = 1
CONTROLLER_STATES = frozenset({"starting", "running", "finalizing", "finished", "stopped", "error"})
TERMINAL_CONTROLLER_STATES = frozenset({"finished", "stopped", "error"})


def status_path(ledger: Path) -> Path:
    """Return the shared observer receipt path for an absolute ledger identity."""
    ledger = Path(ledger).resolve()
    return ledger.parent / ".orchestrator_status" / "agent" / f"{ledger.name}.json"


def ledger_fingerprint(ledger: Path, *, durable: bool = False) -> dict[str, Any]:
    """Hash one ledger version, optionally syncing it before issuing a receipt.

    Replacing or editing the ledger during the read is an error; a completion
    receipt must identify the exact saved version rather than a mixed read.
    """
    ledger = Path(ledger)
    digest = hashlib.sha256()
    with ledger.open("rb") as source:
        before = os.fstat(source.fileno())
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
        if durable:
            os.fsync(source.fileno())
        after = os.fstat(source.fileno())
    current = ledger.stat()

    def identity(stat: os.stat_result) -> tuple[int, int, int, int]:
        return stat.st_ino, stat.st_dev, stat.st_size, stat.st_mtime_ns

    if identity(before) != identity(after) or identity(after) != identity(current):
        raise RuntimeError(f"Ledger changed while reading completion fingerprint: {ledger}")
    if durable:
        _fsync_directory(ledger.parent)
    return {"sha256": digest.hexdigest(), "size": after.st_size, "mtime_ns": after.st_mtime_ns}


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _normalised_status(value: Any) -> ExperimentStatus:
    if isinstance(value, ExperimentStatus):
        return value
    try:
        return ExperimentStatus(str(value).lower())
    except ValueError:
        return ExperimentStatus[str(value).upper()]


def _compact_run(exp: Mapping[str, Any]) -> dict[str, Any]:
    status = _normalised_status(exp.get("status"))
    fraction = display_progress_fraction(exp, allow_zero=True)
    if status == ExperimentStatus.COMPLETED:
        fraction = 1.0
    elif status in {ExperimentStatus.PENDING, ExperimentStatus.QUEUED}:
        fraction = 0.0
    run = {
        "experiment_id": str(exp.get("experiment_id", "")),
        "job_id": str(exp["job_id"]) if exp.get("job_id") is not None else None,
        "status": status.value,
        "fraction": fraction,
        "last_metrics_update": _optional_number(exp.get("last_metrics_update")),
        "current_step": _optional_number(exp.get("current_step")),
        "max_steps": _optional_number(exp.get("max_steps")),
        "current_epoch": _optional_number(exp.get("current_epoch")),
        "max_epochs": _optional_number(exp.get("max_epochs")),
        "metrics_available": bool(exp.get("all_metrics")) or exp.get("metric_value") is not None,
    }
    terminal = exp.get("terminal_telemetry")
    run["terminal_telemetry"] = (
        {key: terminal.get(key) for key in ("job_id", "attempts", "finalized", "finalized_reason", "last_result")}
        if isinstance(terminal, Mapping)
        else None
    )
    return run


def _optional_number(value: Any) -> int | float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    if not math.isfinite(number):
        return None
    return value if isinstance(value, (int, float)) else number


class ControllerStatusPublisher:
    """Hold a same-ledger headless lock and publish bounded, atomic snapshots.

    The lock prevents competing headless writers; it does not protect against a
    separate legacy UI controller. A caller must still run only one controller
    per ledger. Routine snapshots need no ledger read or scheduler query.
    """

    def __init__(
        self, ledger: Path, *, heartbeat_interval_seconds: float = 10.0, stale_after_seconds: float | None = None
    ) -> None:
        self.ledger = Path(ledger).resolve()
        self.path = status_path(self.ledger)
        if not math.isfinite(heartbeat_interval_seconds) or heartbeat_interval_seconds <= 0:
            raise ValueError("heartbeat_interval_seconds must be positive and finite")
        self.heartbeat_interval_seconds = float(heartbeat_interval_seconds)
        self.stale_after_seconds = (
            max(120.0, 3 * self.heartbeat_interval_seconds)
            if stale_after_seconds is None
            else float(stale_after_seconds)
        )
        if not math.isfinite(self.stale_after_seconds) or self.stale_after_seconds <= self.heartbeat_interval_seconds:
            raise ValueError("stale_after_seconds must be finite and greater than the heartbeat interval")
        self.run_id = str(uuid.uuid4())
        self._lock: IO[str] | None = None
        self._last_published_at: float | None = None
        self._last_state: str | None = None
        self._publication_lock = threading.RLock()
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self._heartbeat_error: Exception | None = None
        self._last_snapshot_payload: str | None = None
        self._last_write_at: float | None = None
        self._closing = False

    def __enter__(self) -> ControllerStatusPublisher:
        """Acquire the controller lock without overwriting another live receipt."""
        if self._lock is not None or self._closing:
            raise RuntimeError("Controller status publisher is already active")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock = self.path.with_suffix(self.path.suffix + ".lock").open("a+")
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            lock.close()
            raise RuntimeError(f"A headless controller already holds the ledger lock: {self.ledger}") from exc
        self._lock = lock
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, name=f"slurminator-heartbeat-{self.run_id}", daemon=True
        )
        try:
            self._heartbeat_thread.start()
        except BaseException:
            self._lock.close()
            self._lock = None
            raise
        return self

    def __exit__(self, exc_type: object, *_exc: object) -> None:
        """Stop heartbeat writes before releasing the persistent ledger lock."""
        with self._publication_lock:
            self._closing = True
            self._heartbeat_stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join()
        with self._publication_lock:
            if self._lock is not None:
                self._lock.close()
                self._lock = None
            if exc_type is None and self._heartbeat_error is not None:
                raise RuntimeError("Controller heartbeat failed") from self._heartbeat_error

    def publish(
        self,
        state: str,
        experiments: Sequence[Mapping[str, Any]],
        *,
        force: bool = False,
        error: str | None = None,
        finalized: bool = False,
        expected_fingerprint: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Publish a snapshot, or return ``None`` when the cadence has not elapsed.

        ``finished`` is permitted only with ``finalized=True`` after the caller
        has settled retries, collected terminal telemetry, and saved the ledger.
        ``expected_fingerprint`` must identify the ledger synced immediately
        after that final save. A subsequent edit prevents a finished receipt.
        Heartbeats refresh ``updated_at``/``heartbeat_at`` independently while
        leaving ``progress_updated_at`` and the published counters unchanged.
        """
        with self._publication_lock:
            return self._publish_locked(
                state,
                experiments,
                force=force,
                error=error,
                finalized=finalized,
                expected_fingerprint=expected_fingerprint,
            )

    def _publish_locked(
        self,
        state: str,
        experiments: Sequence[Mapping[str, Any]],
        *,
        force: bool,
        error: str | None,
        finalized: bool,
        expected_fingerprint: Mapping[str, Any] | None,
    ) -> dict[str, Any] | None:
        if self._lock is None or self._closing:
            raise RuntimeError("Enter an active controller publisher context before publishing")
        if self._last_state in TERMINAL_CONTROLLER_STATES:
            raise RuntimeError("A terminal controller receipt cannot be replaced by this publisher")
        if self._heartbeat_error is not None and state not in {"stopped", "error"}:
            raise RuntimeError("Controller heartbeat failed") from self._heartbeat_error
        if state not in CONTROLLER_STATES:
            raise ValueError(f"Unknown controller state: {state}")
        if finalized != (state == "finished"):
            raise ValueError("Only a finished controller may publish finalized results")
        if finalized != (expected_fingerprint is not None):
            raise ValueError("Finalized results require the fingerprint captured immediately after the final save")
        if finalized and ledger_fingerprint(self.ledger) != expected_fingerprint:
            raise RuntimeError(f"Ledger changed after the final controller save: {self.ledger}")
        now = time.monotonic()
        if (
            not force
            and state == self._last_state
            and self._last_published_at is not None
            and now - self._last_published_at < self.heartbeat_interval_seconds
        ):
            return None
        rows = [dict(exp, status=_normalised_status(exp.get("status"))) for exp in experiments]
        counts = {status.value: 0 for status in ExperimentStatus}
        for exp in rows:
            counts[exp["status"].value] += 1
        counts["total"] = len(rows)
        completed_equivalent, eligible, fraction = overall_run_progress(rows)
        runs = [_compact_run(exp) for exp in rows]
        failed_runs = len(rows) - eligible
        timestamp = datetime.now(timezone.utc).isoformat()
        snapshot = {
            "schema_version": SCHEMA_VERSION,
            "ledger": str(self.ledger),
            "run_id": self.run_id,
            "state": state,
            "controller": {"host": socket.gethostname(), "pid": os.getpid()},
            "updated_at": timestamp,
            "heartbeat_at": timestamp,
            "progress_updated_at": timestamp,
            "heartbeat_interval_seconds": self.heartbeat_interval_seconds,
            "stale_after_seconds": self.stale_after_seconds,
            "counts": counts,
            "progress": {
                "fraction": fraction,
                "percent": fraction * 100,
                "completed_equivalent_runs": completed_equivalent,
                "eligible_runs": eligible,
                "excluded_failed_runs": failed_runs,
                "unknown_runs": sum(run["fraction"] is None for run in runs),
            },
            "runs": runs,
            "results_finalized": finalized,
            "outcome": ("completed_with_failures" if failed_runs else "success") if finalized else None,
            "missing_metrics": [run["experiment_id"] for run in runs if not run["metrics_available"]],
            "ledger_fingerprint": dict(expected_fingerprint) if finalized else None,
            "error": error,
        }
        payload = json.dumps(snapshot, allow_nan=False, separators=(",", ":")) + "\n"
        self._write_payload(payload)
        self._last_snapshot_payload = payload
        self._last_published_at = now
        self._last_state = state
        if state in TERMINAL_CONTROLLER_STATES:
            self._heartbeat_stop.set()
        return snapshot

    def _heartbeat_loop(self) -> None:
        """Refresh a serialized snapshot without reading mutable controller data."""
        delay = self.heartbeat_interval_seconds
        while not self._heartbeat_stop.wait(delay):
            with self._publication_lock:
                if self._closing or self._heartbeat_stop.is_set():
                    return
                if self._last_snapshot_payload is None or self._last_write_at is None:
                    continue
                remaining = self._last_write_at + self.heartbeat_interval_seconds - time.monotonic()
                if remaining > 0:
                    delay = remaining
                    continue
                try:
                    snapshot = json.loads(self._last_snapshot_payload)
                    timestamp = datetime.now(timezone.utc).isoformat()
                    snapshot["updated_at"] = timestamp
                    snapshot["heartbeat_at"] = timestamp
                    payload = json.dumps(snapshot, allow_nan=False, separators=(",", ":")) + "\n"
                    self._write_payload(payload)
                    self._last_snapshot_payload = payload
                except Exception as exc:
                    self._heartbeat_error = exc
                    self._heartbeat_stop.set()
                    return
                delay = self.heartbeat_interval_seconds

    def _write_payload(self, payload: str) -> None:
        """Atomically replace the receipt while holding the publication lock."""
        descriptor, temporary = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent)
        try:
            with os.fdopen(descriptor, "w") as destination:
                destination.write(payload)
                destination.flush()
                os.fsync(destination.fileno())
            os.replace(temporary, self.path)
            _fsync_directory(self.path.parent)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        self._last_write_at = time.monotonic()


__all__ = ["ControllerStatusPublisher", "ledger_fingerprint", "status_path"]

"""Framework-neutral callback for target-schema orchestrator status files."""

from __future__ import annotations

import math
import os
import threading
import time
import json
import logging
from collections.abc import Callable, Mapping
from pathlib import Path

from slurminator.callbacks.status_normalization import (
    GenericProgressSnapshot,
    MetricDisplayCandidate,
    normalize_status_payload,
)
from slurminator.schemas.status_schema import HistoryEntry, OrchestratorStatus, StatusState, can_transition

logger = logging.getLogger("slurminator")


class OrchestratorStatusCallback:
    """Write target-schema orchestrator status snapshots for a training run.

    The class is intentionally framework-neutral: trainer objects are duck-typed,
    and project-specific progress or display-metric policy belongs in subclasses.
    """

    def __init__(
        self,
        cfg: object | None = None,
        *,
        min_write_interval_seconds: float = 10.0,
        save_path: str | Path | None = None,
        job_id: str | None = None,
        sweep_id: str | None = None,
        experiment_id: str | None = None,
        primary_metric: str | None = None,
        secondary_metric: str | None = None,
        metric_info: Mapping[str, MetricDisplayCandidate | Mapping[str, object]] | None = None,
        time_fn: Callable[[], float] | None = None,
        pre_replace_hook: Callable[[Path, Path], None] | None = None,
        status_root_name: str = ".orchestrator_status",
    ) -> None:
        self.cfg = cfg
        self.min_write_interval_seconds = max(0.0, float(min_write_interval_seconds))
        self._save_path_override = Path(save_path) if save_path is not None else None
        self._job_id_override = _clean_string(job_id)
        self._sweep_id_override = _clean_string(sweep_id)
        self._experiment_id_override = _clean_string(experiment_id)
        self._explicit_primary_metric = _clean_string(primary_metric)
        self._explicit_secondary_metric = _clean_string(secondary_metric)
        self._explicit_metric_info = dict(metric_info or {})
        self._time_fn = time_fn or time.time
        self._pre_replace_hook = pre_replace_hook
        self._status_root_name = _clean_string(status_root_name) or ".orchestrator_status"

        self.status_file: Path | None = None
        self.history_file: Path | None = None
        self._attempt: int = 1
        self._job_id: str | None = None
        self._sweep_id: str | None = None
        self._experiment_id: str | None = None
        self._run_name: str | None = None
        self._state: StatusState | None = None
        self._last_write_at: float | None = None
        self._metrics: dict[str, float] = {}
        self._links: dict[str, str] = {}
        self._primary_metric: str | None = None
        self._secondary_metric: str | None = None
        self._metric_info: dict[str, MetricDisplayCandidate] = {}

        self._last_step: int | None = None
        self._last_step_at: float | None = None
        self._it_per_sec_ema: float | None = None
        self._samples_per_sec_ema: float | None = None
        self._step_time_ms_ema: float | None = None

    def on_train_start(self, trainer: object) -> None:
        """Initialize the status file."""
        self.cfg = getattr(trainer, "cfg", None) or getattr(trainer, "config", None) or self.cfg
        if not self._should_write(trainer):
            return

        self._initialize_identity_and_path(trainer)
        self._resolve_display_candidates(trainer)
        self._links = self._resolve_links(trainer)
        self._last_step = _safe_int(getattr(trainer, "global_step", 0), default=0)
        self._last_step_at = self._time_fn()
        self._write_status("initializing", trainer=trainer, force=True)

    def on_train_batch_end(self, trainer: object, batch: object, batch_idx: int, logs: Mapping[str, object]) -> None:
        """Update live step progress, respecting the batch-update throttle."""
        if self.status_file is None or not self._should_write(trainer):
            return
        self._update_speed_from_step(trainer)
        self._write_status("running", trainer=trainer, force=False)

    def on_epoch_end(
        self,
        trainer: object,
        epoch: int,
        train_logs: Mapping[str, object] | None,
        val_logs: Mapping[str, object] | None,
    ) -> None:
        """Update epoch progress and scalar train/validation metrics."""
        if self.status_file is None or not self._should_write(trainer):
            return
        self._merge_metrics(_prefix_epoch_logs(train_logs or {}, prefix="train"))
        self._merge_metrics(_prefix_epoch_logs(val_logs or {}, prefix="val"))
        self._write_status("running", trainer=trainer, epoch=epoch, epoch_completed=True, force=True)

    def on_train_end(self, trainer: object) -> None:
        """Write a clean process-completion status."""
        if self.status_file is None or not self._should_write(trainer):
            return
        self._write_status("completed", trainer=trainer, force=True)

    def update_metrics(
        self, metrics: Mapping[str, object], *, trainer: object | None = None, force: bool = True
    ) -> None:
        """Merge scalar metrics and write a status update."""
        if self.status_file is None:
            return
        self._merge_metrics(metrics)
        current_state = self._state or self._read_existing_state()
        next_status: StatusState = "completed" if current_state == "completed" else "running"
        self._write_status(next_status, trainer=trainer, force=force)

    def _write_status(
        self,
        next_status: StatusState,
        *,
        trainer: object | None,
        epoch: int | None = None,
        epoch_completed: bool = False,
        force: bool = False,
    ) -> bool:
        if self.status_file is None:
            return False

        previous_status = self._state or self._read_existing_state()
        if previous_status is not None and not can_transition(previous_status, next_status):
            raise ValueError(f"Invalid orchestrator status transition: {previous_status} -> {next_status}")

        now = float(self._time_fn())
        if not force and previous_status == next_status and self._is_throttled(now):
            return False

        if trainer is not None:
            self._links.update(self._resolve_links(trainer))
        progress = self._build_progress_snapshot(trainer, epoch=epoch, epoch_completed=epoch_completed)
        status = normalize_status_payload(
            experiment_id=self._require_experiment_id(),
            job_id=self._require_job_id(),
            status=next_status,
            last_update=now,
            progress=progress,
            run_name=self._resolve_run_name(trainer),
            metrics=self._metrics,
            primary_metric=self._primary_metric,
            secondary_metric=self._secondary_metric,
            metric_info=self._metric_info,
            links=self._links,
        )
        status.attempt = self._attempt
        self._atomic_write(status)
        self._append_history(status)
        self._state = status.status
        self._last_write_at = now
        return True

    def _initialize_identity_and_path(self, trainer: object) -> None:
        save_path = self._resolve_save_path(trainer)
        if save_path is None:
            raise RuntimeError("SAVE_PATH is required for OrchestratorStatusCallback.")

        job_id = self._resolve_job_id(trainer)
        if job_id is None:
            raise RuntimeError("SLURM_JOB_ID, PBS_JOBID, or JOB_ID is required for OrchestratorStatusCallback.")

        sweep_id = self._resolve_sweep_id(trainer)
        status_dir = save_path / self._status_root_name
        if sweep_id:
            status_dir = status_dir / f"sweep_{sweep_id}"
        status_dir.mkdir(parents=True, exist_ok=True)

        self._job_id = job_id
        self._sweep_id = sweep_id
        self._experiment_id = self._resolve_experiment_id(trainer, job_id=job_id)
        self.status_file = status_dir / f"status_{job_id}.json"
        self.history_file = self.status_file.with_name(f"history_{job_id}.jsonl")
        self._attempt = self._resolve_attempt_from_history()

    def _resolve_attempt_from_history(self) -> int:
        """Read the last history line and return its attempt plus one."""
        if self.history_file is None or not self.history_file.exists():
            return 1
        try:
            with self.history_file.open("rb") as handle:
                handle.seek(0, 2)
                size = handle.tell()
                if size == 0:
                    return 1
                handle.seek(max(0, size - 1024))
                tail = handle.read().decode("utf-8", errors="replace")
            lines = [line for line in tail.splitlines() if line.strip()]
            if not lines:
                return 1
            last = json.loads(lines[-1])
            return int(last.get("attempt", 1)) + 1
        except Exception as exc:
            logger.debug("Failed to read history file for attempt resolution: %s", exc)
            return 1

    def _resolve_save_path(self, trainer: object | None) -> Path | None:
        return self._save_path_override or _path_from_env("SAVE_PATH")

    def _resolve_job_id(self, trainer: object | None) -> str | None:
        return self._job_id_override or _first_env("SLURM_JOB_ID", "PBS_JOBID", "JOB_ID")

    def _resolve_sweep_id(self, trainer: object | None) -> str | None:
        return self._sweep_id_override or _first_env("ORCHESTRATOR_SWEEP_ID", "SWEEP_ID")

    def _resolve_experiment_id(self, trainer: object, *, job_id: str) -> str:
        candidates = (
            self._experiment_id_override,
            _clean_string(getattr(self.cfg, "experiment_id", None)),
            _clean_string(getattr(trainer, "experiment_id", None)),
            _clean_string(os.environ.get("ORCHESTRATOR_EXPERIMENT_ID")),
            _clean_string(os.environ.get("EXPERIMENT_ID")),
            _clean_string(getattr(self.cfg, "run_name", None)),
            job_id,
        )
        for candidate in candidates:
            if candidate:
                return candidate
        raise RuntimeError("Could not resolve orchestrator experiment_id.")

    def _resolve_run_name(self, trainer: object | None) -> str:
        if self._run_name:
            return self._run_name
        self._run_name = _clean_string(getattr(self.cfg, "run_name", None)) or self._experiment_id or self._job_id
        if not self._run_name:
            raise RuntimeError("Could not resolve display.run_name for orchestrator status.")
        return self._run_name

    def _resolve_links(self, trainer: object) -> dict[str, str]:
        return {}

    def _resolve_display_candidates(self, trainer: object) -> None:
        candidates: dict[str, MetricDisplayCandidate] = {}
        for key, info in self._explicit_metric_info.items():
            candidate = _metric_candidate_from_mapping(info) if isinstance(info, Mapping) else info
            candidates[key] = candidate

        self._primary_metric = self._explicit_primary_metric
        self._secondary_metric = self._explicit_secondary_metric
        self._metric_info = candidates

    def _build_progress_snapshot(
        self, trainer: object | None, *, epoch: int | None = None, epoch_completed: bool = False
    ) -> GenericProgressSnapshot:
        progress = _runtime_progress(trainer)
        current_epoch = _first_int(progress.get("current_epoch"), getattr(trainer, "current_epoch", None), default=0)
        if epoch is not None and epoch_completed:
            current_epoch = max(current_epoch or 0, int(epoch) + 1)
        total_epochs = _first_positive_int(
            progress.get("max_epochs"),
            getattr(getattr(self.cfg, "training_configs", None), "num_epochs", None),
            getattr(trainer, "epochs", None),
        )
        current_step = _first_int(progress.get("current_step"), getattr(trainer, "global_step", None), default=None)
        total_steps = _first_positive_int(progress.get("max_steps"), getattr(trainer, "max_train_steps", None))
        return GenericProgressSnapshot(
            unit="epoch",
            current_epoch=current_epoch or 0,
            total_epochs=total_epochs,
            current_step=current_step,
            total_steps=total_steps,
            speed_value=self._it_per_sec_ema,
            speed_unit="it/sec",
            samples_per_sec=self._samples_per_sec_ema,
            step_time_ms_ema=self._step_time_ms_ema,
            eta_seconds=self._eta_seconds(current_step=current_step, total_steps=total_steps),
        )

    def _merge_metrics(self, metrics: Mapping[str, object]) -> None:
        self._metrics.update(self._filter_numeric_metrics(metrics))

    def _filter_numeric_metrics(self, metrics: Mapping[str, object]) -> dict[str, float]:
        latest: dict[str, float] = {}
        for key, value in (metrics or {}).items():
            if not isinstance(key, str) or not key.strip():
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            value_float = float(value)
            if not math.isfinite(value_float):
                continue
            latest[key.strip()] = value_float
        return latest

    def _update_speed_from_step(self, trainer: object) -> None:
        current_step = _safe_int(getattr(trainer, "global_step", 0), default=0)
        now = self._time_fn()
        if self._last_step is None or self._last_step_at is None:
            self._last_step = current_step
            self._last_step_at = now
            return

        step_delta = current_step - self._last_step
        elapsed = now - self._last_step_at
        self._last_step = current_step
        self._last_step_at = now
        if step_delta <= 0 or elapsed <= 0:
            return

        inst_it_per_sec = float(step_delta) / float(elapsed)
        alpha = 0.2
        self._it_per_sec_ema = _ema(self._it_per_sec_ema, inst_it_per_sec, alpha=alpha)
        batch_size = _first_positive_int(getattr(getattr(self.cfg, "optimizer_parameters", None), "batch_size", None))
        if batch_size is not None:
            self._samples_per_sec_ema = _ema(
                self._samples_per_sec_ema, float(batch_size) * self._it_per_sec_ema, alpha=alpha
            )
        if self._it_per_sec_ema > 0:
            self._step_time_ms_ema = _ema(self._step_time_ms_ema, 1000.0 / self._it_per_sec_ema, alpha=alpha)

    def _eta_seconds(self, *, current_step: int | None, total_steps: int | None) -> int | None:
        if current_step is None or total_steps is None:
            return None
        if self._it_per_sec_ema is None or self._it_per_sec_ema <= 0:
            return None
        return int(max(total_steps - current_step, 0) / self._it_per_sec_ema)

    def _atomic_write(self, status: OrchestratorStatus) -> None:
        path = self.status_file
        if path is None:
            raise RuntimeError("status_file is not initialized.")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}.tmp")
        try:
            with tmp_path.open("w", encoding="utf-8") as handle:
                handle.write(status.model_dump_json(indent=2))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            if self._pre_replace_hook is not None:
                self._pre_replace_hook(tmp_path, path)
            os.replace(tmp_path, path)
        finally:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except FileNotFoundError:  # pragma: no cover - race-safe cleanup
                pass

    def _append_history(self, status: OrchestratorStatus) -> None:
        """Append a history entry when a status write contains metrics."""
        if self.history_file is None or not status.metrics:
            return
        metrics = self._history_metrics(status)
        if not metrics:
            return
        entry = HistoryEntry(
            timestamp=status.last_update,
            attempt=status.attempt,
            epoch=status.progress.current_epoch,
            step=status.progress.current_step,
            unit=status.progress.unit,
            metrics=metrics,
        )
        try:
            with self.history_file.open("a", encoding="utf-8") as handle:
                handle.write(entry.model_dump_json())
                handle.write("\n")
        except Exception as exc:
            logger.debug("Failed to append history entry: %s", exc)

    def _history_metrics(self, status: OrchestratorStatus) -> dict[str, float]:
        """Return the metrics to append to the history file for this status."""
        return dict(status.metrics)

    def _read_existing_state(self) -> StatusState | None:
        if self.status_file is None or not self.status_file.exists():
            return None
        try:
            existing = OrchestratorStatus.model_validate_json(self.status_file.read_text(encoding="utf-8"))
        except Exception:
            return None
        self._metrics.update(existing.metrics)
        self._links.update(existing.links)
        return existing.status

    def _is_throttled(self, now: float) -> bool:
        if self.min_write_interval_seconds <= 0:
            return False
        if self._last_write_at is None:
            return False
        return (now - self._last_write_at) < self.min_write_interval_seconds

    def _should_write(self, trainer: object | None) -> bool:
        if trainer is not None and _safe_int(getattr(trainer, "rank", 0), default=0) != 0:
            return False
        if self.cfg is not None and hasattr(self.cfg, "orchestrated"):
            return bool(getattr(self.cfg, "orchestrated", False))
        return True

    def _require_job_id(self) -> str:
        if not self._job_id:
            raise RuntimeError("job_id is not initialized.")
        return self._job_id

    def _require_experiment_id(self) -> str:
        if not self._experiment_id:
            raise RuntimeError("experiment_id is not initialized.")
        return self._experiment_id


def _runtime_progress(trainer: object | None) -> dict[str, object]:
    if trainer is None:
        return {}
    get_progress = getattr(trainer, "get_runtime_progress", None)
    if not callable(get_progress):
        return {}
    progress = get_progress()
    return progress if isinstance(progress, dict) else {}


def _prefix_epoch_logs(logs: Mapping[str, object], *, prefix: str) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in (logs or {}).items():
        if key == "epoch":
            continue
        metric_key = key if "/" in str(key) else f"{prefix}/{key}"
        payload[metric_key] = value
    return payload


def _metric_candidate_from_mapping(info: Mapping[str, object]) -> MetricDisplayCandidate:
    direction = _clean_string(info.get("direction"))
    higher_better = None
    if direction == "maximize":
        higher_better = True
    elif direction == "minimize":
        higher_better = False
    return MetricDisplayCandidate(
        shortform=_clean_string(info.get("shortform")),
        higher_better=higher_better,
        format=_clean_string(info.get("format")) or _clean_string(info.get("value_format")),
        threshold=_finite_float_or_none(info.get("threshold")),
    )


def _first_env(*names: str) -> str | None:
    for name in names:
        value = _clean_string(os.environ.get(name))
        if value:
            return value
    return None


def _path_from_env(name: str) -> Path | None:
    value = _clean_string(os.environ.get(name))
    return Path(value) if value else None


def _clean_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    stripped = value.strip()
    return stripped or None


def _safe_int(value: object, *, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _first_int(*values: object, default: int | None) -> int | None:
    for value in values:
        if value is None or isinstance(value, bool):
            continue
        try:
            int_value = int(value)
        except Exception:
            continue
        if int_value >= 0:
            return int_value
    return default


def _first_positive_int(*values: object) -> int | None:
    for value in values:
        if value is None or isinstance(value, bool):
            continue
        try:
            int_value = int(value)
        except Exception:
            continue
        if int_value > 0:
            return int_value
    return None


def _finite_float_or_none(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value_float = float(value)
    return value_float if math.isfinite(value_float) else None


def _ema(previous: float | None, current: float, *, alpha: float) -> float:
    if previous is None:
        return float(current)
    return (1.0 - alpha) * previous + alpha * float(current)


__all__ = [
    "OrchestratorStatusCallback",
    "_clean_string",
    "_finite_float_or_none",
    "_first_int",
    "_first_positive_int",
    "_metric_candidate_from_mapping",
    "_runtime_progress",
    "_safe_int",
]

"""Focused durability and replay tests for scheduler lifecycle transitions."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import pytest

from slurminator import lifecycle_journal as lifecycle_journal_module
from slurminator import state_store as state_store_module
from slurminator.config import HPCType
from slurminator.experiments import ExperimentStatus
from slurminator.experiments.yaml_utils import dump_yaml, load_yaml
from slurminator.lifecycle_journal import SchedulerLifecycleJournal
from slurminator.scheduler_polling import SchedulerTransition
from slurminator.state_store import ExperimentStateStore

pytestmark = pytest.mark.unit


def _row(
    *,
    experiment_id: str = "exp-1",
    job_id: str | None = "12345",
    status: ExperimentStatus = ExperimentStatus.QUEUED,
    last_change_ts: float | None = None,
    queued_timestamp: float | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {"experiment_id": experiment_id, "status": status, "hpc_assignment": HPCType.OLIVIA}
    if job_id is not None:
        row["job_id"] = job_id
    if last_change_ts is not None:
        row["last_change_ts"] = last_change_ts
    if queued_timestamp is not None:
        row["queued_timestamp"] = queued_timestamp
    return row


def _transition(
    new_status: ExperimentStatus,
    timestamp: float,
    *,
    experiment_id: str = "exp-1",
    job_id: str = "12345",
    old_status: ExperimentStatus | None = None,
) -> SchedulerTransition:
    timestamp_fields = {
        ExperimentStatus.RUNNING: "running_timestamp",
        ExperimentStatus.COMPLETED: "completed_timestamp",
        ExperimentStatus.FAILED: "failed_timestamp",
        ExperimentStatus.CANCELLED: "cancelled_timestamp",
        ExperimentStatus.TIMEOUT: "timeout_timestamp",
        ExperimentStatus.OOM: "failed_timestamp",
        ExperimentStatus.KILLED: "killed_timestamp",
    }
    if old_status is None:
        old_status = ExperimentStatus.RUNNING if new_status != ExperimentStatus.RUNNING else ExperimentStatus.QUEUED
    return SchedulerTransition(
        experiment_id=experiment_id,
        job_id=job_id,
        hpc_type=HPCType.OLIVIA,
        old_status=old_status,
        new_status=new_status,
        timestamp=timestamp,
        timestamp_field=timestamp_fields.get(new_status),
    )


def _accepted_row() -> dict[str, Any]:
    return {
        **_row(queued_timestamp=10.0),
        "git_sha_at_submission": {"project": "a" * 40},
        "output_dir": "/tmp/output",
        "save_path": "/tmp/save",
    }


def test_append_batches_one_poll_into_one_write_and_fsync(tmp_path, monkeypatch) -> None:
    journal = SchedulerLifecycleJournal(tmp_path / "experiments.yaml")
    journal.append_transitions([_transition(ExperimentStatus.RUNNING, 20.0)])
    write_calls: list[int] = []
    fsync_calls: list[int] = []
    original_write = lifecycle_journal_module.os.write
    original_fsync = lifecycle_journal_module.os.fsync

    def tracking_write(fd: int, payload: object) -> int:
        write_calls.append(fd)
        return original_write(fd, payload)

    def tracking_fsync(fd: int) -> None:
        fsync_calls.append(fd)
        original_fsync(fd)

    monkeypatch.setattr(lifecycle_journal_module.os, "write", tracking_write)
    monkeypatch.setattr(lifecycle_journal_module.os, "fsync", tracking_fsync)

    recorded = journal.append_transitions(
        [
            _transition(ExperimentStatus.COMPLETED, 30.0),
            _transition(ExperimentStatus.RUNNING, 21.0, experiment_id="exp-2", job_id="67890"),
        ]
    )

    assert [transition.new_status for transition in recorded] == [ExperimentStatus.COMPLETED, ExperimentStatus.RUNNING]
    assert len(write_calls) == 1
    assert len(fsync_calls) == 1
    assert len(journal.read()) == 3


def test_apply_replays_ordered_transitions_idempotently(tmp_path) -> None:
    journal = SchedulerLifecycleJournal(tmp_path / "experiments.yaml")
    journal.append_transitions(
        [_transition(ExperimentStatus.RUNNING, 20.0), _transition(ExperimentStatus.COMPLETED, 30.0)]
    )
    data = {"experiments": [_row()]}

    assert journal.apply(data) == 2
    first_result = dict(data["experiments"][0])
    assert first_result["status"] == ExperimentStatus.COMPLETED
    assert first_result["last_change_ts"] == 30.0
    assert first_result["running_timestamp"] == 20.0
    assert first_result["completed_timestamp"] == 30.0

    assert journal.apply(data) == 2
    assert data["experiments"][0] == first_result


@pytest.mark.parametrize(
    ("status", "timestamp_field"),
    [
        (ExperimentStatus.COMPLETED, "completed_timestamp"),
        (ExperimentStatus.FAILED, "failed_timestamp"),
        (ExperimentStatus.CANCELLED, "cancelled_timestamp"),
        (ExperimentStatus.TIMEOUT, "timeout_timestamp"),
        (ExperimentStatus.OOM, "failed_timestamp"),
        (ExperimentStatus.KILLED, "killed_timestamp"),
    ],
)
def test_terminal_transition_replays_its_allowlisted_timestamp_field(tmp_path, status, timestamp_field) -> None:
    journal = SchedulerLifecycleJournal(tmp_path / "experiments.yaml")
    journal.append_transitions([_transition(status, 20.0)])
    data = {"experiments": [_row(status=ExperimentStatus.RUNNING, last_change_ts=10.0)]}

    journal.apply(data)

    row = data["experiments"][0]
    assert row["status"] == status
    assert row["last_change_ts"] == 20.0
    assert row[timestamp_field] == 20.0


@pytest.mark.parametrize("status", [ExperimentStatus.COMPLETED, ExperimentStatus.FAILED, ExperimentStatus.PARTIAL])
def test_replay_never_regresses_terminal_or_post_scheduler_state(tmp_path, status) -> None:
    journal = SchedulerLifecycleJournal(tmp_path / "experiments.yaml")
    journal.append_transitions([_transition(ExperimentStatus.RUNNING, 20.0)])
    data = {"experiments": [_row(status=status, last_change_ts=30.0)]}

    journal.apply(data)

    assert data["experiments"][0]["status"] == status
    assert data["experiments"][0]["last_change_ts"] == 30.0


def test_replay_restores_historical_running_timestamp_without_regressing_terminal_state(tmp_path) -> None:
    journal = SchedulerLifecycleJournal(tmp_path / "experiments.yaml")
    journal.append_transitions([_transition(ExperimentStatus.RUNNING, 20.0)])
    data = {"experiments": [_row(status=ExperimentStatus.COMPLETED, last_change_ts=30.0)]}

    journal.apply(data)

    row = data["experiments"][0]
    assert row["status"] == ExperimentStatus.COMPLETED
    assert row["last_change_ts"] == 30.0
    assert row["running_timestamp"] == 20.0


def test_replay_preserves_newer_timestamped_active_state(tmp_path) -> None:
    journal = SchedulerLifecycleJournal(tmp_path / "experiments.yaml")
    journal.append_transitions([_transition(ExperimentStatus.QUEUED, 20.0)])
    data = {"experiments": [_row(status=ExperimentStatus.RUNNING, last_change_ts=30.0)]}

    journal.apply(data)

    assert data["experiments"][0]["status"] == ExperimentStatus.RUNNING
    assert data["experiments"][0]["last_change_ts"] == 30.0


def test_old_attempt_transition_does_not_overwrite_newer_retry(tmp_path) -> None:
    journal = SchedulerLifecycleJournal(tmp_path / "experiments.yaml")
    journal.append_transitions([_transition(ExperimentStatus.COMPLETED, 40.0, job_id="old-job")])
    data = {"experiments": [_row(job_id="new-job", status=ExperimentStatus.QUEUED, queued_timestamp=50.0)]}

    journal.apply(data)

    row = data["experiments"][0]
    assert row["job_id"] == "new-job"
    assert row["status"] == ExperimentStatus.QUEUED


def test_submission_retry_receipt_proves_old_job_transition_is_superseded(tmp_path) -> None:
    ledger_path = tmp_path / "experiments.yaml"
    dump_yaml(
        {"experiments": [_row(job_id="old-job", status=ExperimentStatus.PARTIAL, last_change_ts=40.0)]}, ledger_path
    )
    store = ExperimentStateStore(ledger_path, {HPCType.OLIVIA: 1})
    new_attempt = {**_accepted_row(), "job_id": "new-job", "queued_timestamp": 50.0}
    store.record_submission(new_attempt, previous_job_id="old-job")
    store.record_scheduler_transitions([_transition(ExperimentStatus.COMPLETED, 60.0, job_id="old-job")])

    loaded = store.load()

    row = loaded["experiments"][0]
    assert row["job_id"] == "new-job"
    assert row["status"] == ExperimentStatus.QUEUED
    assert row["queued_timestamp"] == 50.0


def test_unproven_job_mismatch_fails_without_mutating_any_rows(tmp_path) -> None:
    journal = SchedulerLifecycleJournal(tmp_path / "experiments.yaml")
    journal.append_transitions(
        [
            _transition(ExperimentStatus.RUNNING, 20.0),
            _transition(ExperimentStatus.COMPLETED, 60.0, experiment_id="exp-2", job_id="old-job"),
        ]
    )
    data = {"experiments": [_row(), _row(experiment_id="exp-2", job_id="new-job", queued_timestamp=50.0)]}

    with pytest.raises(ValueError, match="conflicts with ledger job_id"):
        journal.apply(data)

    assert data["experiments"][0]["status"] == ExperimentStatus.QUEUED
    assert "last_change_ts" not in data["experiments"][0]


def test_submission_and_lifecycle_journals_replay_without_eager_checkpoint(tmp_path) -> None:
    ledger_path = tmp_path / "experiments.yaml"
    stale_data = {"experiments": [_row(job_id=None, status=ExperimentStatus.PENDING)]}
    dump_yaml(stale_data, ledger_path)
    store = ExperimentStateStore(ledger_path, {HPCType.OLIVIA: 1})
    store.record_submission(_accepted_row())
    recorded = store.record_scheduler_transitions([_transition(ExperimentStatus.RUNNING, 20.0)])

    loaded = store.load()

    assert len(recorded) == 1
    assert loaded["experiments"][0]["status"] == ExperimentStatus.RUNNING
    assert loaded["experiments"][0]["last_change_ts"] == 20.0
    assert load_yaml(ledger_path)["experiments"][0]["status"] == ExperimentStatus.PENDING
    assert store.submission_journal.path.exists()
    assert store.lifecycle_journal.path.exists()
    assert store.load()["experiments"][0]["status"] == ExperimentStatus.RUNNING

    store.save(stale_data)

    checkpoint = load_yaml(ledger_path)["experiments"][0]
    assert checkpoint["status"] == ExperimentStatus.RUNNING
    assert checkpoint["job_id"] == "12345"
    assert checkpoint["last_change_ts"] == 20.0
    assert checkpoint["running_timestamp"] == 20.0
    assert not store.submission_journal.path.exists()
    assert not store.lifecycle_journal.path.exists()


def test_failed_checkpoint_retains_lifecycle_journal(tmp_path, monkeypatch) -> None:
    ledger_path = tmp_path / "experiments.yaml"
    data = {"experiments": [_row()]}
    dump_yaml(data, ledger_path)
    store = ExperimentStateStore(ledger_path, {HPCType.OLIVIA: 1})
    store.record_scheduler_transitions([_transition(ExperimentStatus.RUNNING, 20.0)])

    def fail_dump(_data, _path):
        raise OSError("simulated checkpoint failure")

    monkeypatch.setattr(state_store_module, "dump_yaml", fail_dump)

    with pytest.raises(OSError, match="simulated checkpoint failure"):
        store.save(data)

    assert store.lifecycle_journal.path.exists()
    assert load_yaml(ledger_path)["experiments"][0]["status"] == ExperimentStatus.QUEUED


def test_torn_lifecycle_record_fails_strictly(tmp_path) -> None:
    journal = SchedulerLifecycleJournal(tmp_path / "experiments.yaml")
    journal.path.parent.mkdir(parents=True)
    journal.path.write_text(json.dumps({"schema_version": "1.0"}), encoding="utf-8")

    with pytest.raises(ValueError, match="torn final record"):
        journal.read()


def test_missing_ledger_with_lifecycle_state_fails_instead_of_discarding_it(tmp_path) -> None:
    ledger_path = tmp_path / "experiments.yaml"
    store = ExperimentStateStore(ledger_path, {HPCType.OLIVIA: 1})
    store.record_scheduler_transitions([_transition(ExperimentStatus.RUNNING, 20.0)])

    with pytest.raises(FileNotFoundError, match="durable journal state remains"):
        store.load()

    assert store.lifecycle_journal.path.exists()


@pytest.mark.parametrize(
    "transition, message",
    [
        (_transition(ExperimentStatus.PENDING, 20.0), "non-scheduler status"),
        (_transition(ExperimentStatus.RUNNING, 0.0), "positive finite timestamp"),
    ],
)
def test_record_rejects_invalid_scheduler_transition(tmp_path, transition, message) -> None:
    journal = SchedulerLifecycleJournal(tmp_path / "experiments.yaml")

    with pytest.raises(ValueError, match=message):
        journal.append_transitions([transition])

    assert not journal.path.exists()


def test_record_rejects_non_allowlisted_timestamp_field(tmp_path) -> None:
    journal = SchedulerLifecycleJournal(tmp_path / "experiments.yaml")
    transition = replace(_transition(ExperimentStatus.COMPLETED, 20.0), timestamp_field="running_timestamp")

    with pytest.raises(ValueError, match="requires timestamp_field 'completed_timestamp'"):
        journal.append_transitions([transition])

    assert not journal.path.exists()

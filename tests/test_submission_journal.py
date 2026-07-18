from __future__ import annotations

import json

import pytest

from slurminator import state_store as state_store_module
from slurminator.config import HPCType
from slurminator.experiments import ExperimentStatus
from slurminator.experiments.yaml_utils import dump_yaml, load_yaml
from slurminator.state_store import ExperimentStateStore
from slurminator.submission_journal import SubmissionReceiptJournal

pytestmark = pytest.mark.unit


def _row(*, status: ExperimentStatus = ExperimentStatus.PENDING, job_id: str | None = None) -> dict[str, object]:
    row: dict[str, object] = {"experiment_id": "exp-1", "status": status, "hpc_assignment": HPCType.OLIVIA}
    if job_id is not None:
        row["job_id"] = job_id
    return row


def _accepted_row(job_id: str = "12345") -> dict[str, object]:
    return {
        **_row(status=ExperimentStatus.QUEUED, job_id=job_id),
        "queued_timestamp": 1_700_000_000.0,
        "git_sha_at_submission": {"project": "a" * 40, "slurminator": "b" * 40},
        "output_dir": "/tmp/output",
        "save_path": "/tmp/save",
        "requested_time_hours": 3,
        "requested_ram_gb": 64,
        "requested_gpu_count": 1,
    }


def test_save_merges_receipts_before_checkpoint_and_clears_journal(tmp_path) -> None:
    ledger_path = tmp_path / "experiments.yaml"
    stale_data = {"experiments": [_row()]}
    dump_yaml(stale_data, ledger_path)
    store = ExperimentStateStore(ledger_path, {HPCType.OLIVIA: 1})
    store.record_submission(_accepted_row())

    store.save(stale_data)

    checkpoint = load_yaml(ledger_path)
    saved_row = checkpoint["experiments"][0]
    assert saved_row["status"] == ExperimentStatus.QUEUED
    assert saved_row["job_id"] == "12345"
    assert saved_row["git_sha_at_submission"]["project"] == "a" * 40
    assert saved_row["requested_ram_gb"] == 64
    assert not store.submission_journal.path.exists()


def test_failed_save_retains_submission_journal(tmp_path, monkeypatch) -> None:
    ledger_path = tmp_path / "experiments.yaml"
    stale_data = {"experiments": [_row()]}
    dump_yaml(stale_data, ledger_path)
    store = ExperimentStateStore(ledger_path, {HPCType.OLIVIA: 1})
    store.record_submission(_accepted_row())

    def fail_dump(_data, _path):
        raise OSError("simulated checkpoint failure")

    monkeypatch.setattr(state_store_module, "dump_yaml", fail_dump)

    with pytest.raises(OSError, match="simulated checkpoint failure"):
        store.save(stale_data)

    assert store.submission_journal.path.exists()
    assert [receipt.job_id for receipt in store.submission_journal.read()] == ["12345"]
    assert load_yaml(ledger_path)["experiments"][0]["status"] == ExperimentStatus.PENDING


def test_failed_checkpoint_fsync_retains_submission_journal(tmp_path, monkeypatch) -> None:
    ledger_path = tmp_path / "experiments.yaml"
    stale_data = {"experiments": [_row()]}
    dump_yaml(stale_data, ledger_path)
    store = ExperimentStateStore(ledger_path, {HPCType.OLIVIA: 1})
    store.record_submission(_accepted_row())

    def fail_fsync(_path):
        raise OSError("simulated fsync failure")

    monkeypatch.setattr(state_store_module, "_fsync_file", fail_fsync)

    with pytest.raises(OSError, match="simulated fsync failure"):
        store.save(stale_data)

    assert store.submission_journal.path.exists()
    checkpoint = load_yaml(ledger_path)
    assert checkpoint["experiments"][0]["job_id"] == "12345"


def test_load_replays_receipt_idempotently_without_regressing_terminal_status(tmp_path) -> None:
    ledger_path = tmp_path / "experiments.yaml"
    terminal_row = {**_accepted_row(), "status": ExperimentStatus.COMPLETED, "completed_timestamp": 1_700_000_100.0}
    dump_yaml({"experiments": [terminal_row]}, ledger_path)
    store = ExperimentStateStore(ledger_path, {HPCType.OLIVIA: 1})
    store.record_submission(_accepted_row())

    loaded = store.load()

    loaded_row = loaded["experiments"][0]
    assert loaded_row["status"] == ExperimentStatus.COMPLETED
    assert loaded_row["completed_timestamp"] == 1_700_000_100.0
    assert not store.submission_journal.path.exists()
    assert store.load()["experiments"][0]["status"] == ExperimentStatus.COMPLETED


def test_apply_is_idempotent_for_the_same_receipt(tmp_path) -> None:
    journal = SubmissionReceiptJournal(tmp_path / "experiments.yaml")
    journal.append_experiment(_accepted_row())
    data = {"experiments": [_row()]}

    journal.apply(data)
    first_result = dict(data["experiments"][0])
    journal.apply(data)

    assert data["experiments"][0] == first_result


def test_retry_receipt_replaces_expected_previous_job(tmp_path) -> None:
    journal = SubmissionReceiptJournal(tmp_path / "experiments.yaml")
    journal.append_experiment(_accepted_row("67890"), previous_job_id="12345")
    data = {"experiments": [_row(status=ExperimentStatus.PARTIAL, job_id="12345")]}

    journal.apply(data)

    row = data["experiments"][0]
    assert row["status"] == ExperimentStatus.QUEUED
    assert row["job_id"] == "67890"


def test_retry_receipt_rejects_unexpected_previous_job_without_mutation(tmp_path) -> None:
    journal = SubmissionReceiptJournal(tmp_path / "experiments.yaml")
    journal.append_experiment(_accepted_row("67890"), previous_job_id="12345")
    data = {"experiments": [_row(status=ExperimentStatus.PARTIAL, job_id="11111")]}

    with pytest.raises(ValueError, match="conflicts with ledger job_id"):
        journal.apply(data)

    assert data["experiments"][0]["job_id"] == "11111"
    assert data["experiments"][0]["status"] == ExperimentStatus.PARTIAL


def test_conflicting_job_receipts_fail_without_mutating_the_ledger(tmp_path) -> None:
    journal = SubmissionReceiptJournal(tmp_path / "experiments.yaml")
    journal.append_experiment(_accepted_row("12345"))
    journal.append_experiment(_accepted_row("67890"))
    data = {"experiments": [_row()]}

    with pytest.raises(ValueError, match="Conflicting submission receipts"):
        journal.apply(data)

    assert data["experiments"][0]["status"] == ExperimentStatus.PENDING
    assert "job_id" not in data["experiments"][0]
    assert journal.path.exists()


def test_torn_journal_record_fails_strictly(tmp_path) -> None:
    journal = SubmissionReceiptJournal(tmp_path / "experiments.yaml")
    journal.path.parent.mkdir(parents=True)
    journal.path.write_text(json.dumps({"schema_version": "1.0"}), encoding="utf-8")

    with pytest.raises(ValueError, match="torn final record"):
        journal.read()


def test_empty_existing_journal_fails_strictly(tmp_path) -> None:
    journal = SubmissionReceiptJournal(tmp_path / "experiments.yaml")
    journal.path.parent.mkdir(parents=True)
    journal.path.touch()

    with pytest.raises(ValueError, match="unexpectedly empty"):
        journal.read()


def test_missing_ledger_with_receipts_fails_instead_of_discarding_state(tmp_path) -> None:
    ledger_path = tmp_path / "experiments.yaml"
    store = ExperimentStateStore(ledger_path, {HPCType.OLIVIA: 1})
    store.record_submission(_accepted_row())

    with pytest.raises(FileNotFoundError, match="durable submission receipts remain"):
        store.load()

    assert store.submission_journal.path.exists()


def test_disabled_hpc_preserves_accepted_active_job_and_warns(tmp_path, caplog) -> None:
    ledger_path = tmp_path / "experiments.yaml"
    dump_yaml({"experiments": [_accepted_row()]}, ledger_path)
    store = ExperimentStateStore(ledger_path, {HPCType.OLIVIA: 0})

    with caplog.at_level("WARNING", logger="slurminator"):
        loaded = store.load()
        store.load()

    loaded_row = loaded["experiments"][0]
    assert loaded_row["status"] == ExperimentStatus.QUEUED
    assert loaded_row["hpc_assignment"] == HPCType.OLIVIA
    assert loaded_row["job_id"] == "12345"
    assert "preserving scheduler state" in caplog.text
    assert "connected at startup" in caplog.text
    assert caplog.text.count("accepted active experiment(s)") == 1


def test_disabled_hpc_still_resets_unsubmitted_row(tmp_path) -> None:
    ledger_path = tmp_path / "experiments.yaml"
    dump_yaml({"experiments": [_row(status=ExperimentStatus.QUEUED)]}, ledger_path)
    store = ExperimentStateStore(ledger_path, {HPCType.OLIVIA: 0})

    loaded = store.load()

    loaded_row = loaded["experiments"][0]
    assert loaded_row["status"] == ExperimentStatus.PENDING
    assert loaded_row["hpc_assignment"] is None


def test_disabled_hpc_does_not_reset_terminal_row_without_job_id(tmp_path) -> None:
    ledger_path = tmp_path / "experiments.yaml"
    dump_yaml({"experiments": [_row(status=ExperimentStatus.COMPLETED)]}, ledger_path)
    store = ExperimentStateStore(ledger_path, {HPCType.OLIVIA: 0})

    loaded = store.load()

    loaded_row = loaded["experiments"][0]
    assert loaded_row["status"] == ExperimentStatus.COMPLETED
    assert loaded_row["hpc_assignment"] == HPCType.OLIVIA

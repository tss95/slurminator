from __future__ import annotations

import pytest

from slurminator import submission
from slurminator.config import HPCType
from slurminator.experiments import ExperimentStatus
from slurminator.submission import maybe_submit

pytestmark = pytest.mark.unit


def test_maybe_submit_records_git_sha_at_submission(monkeypatch) -> None:
    provenance = {"project": "a" * 40, "slurminator": "b" * 40}
    monkeypatch.setattr(submission, "capture_provenance", lambda: provenance)

    exp = {"experiment_id": "exp-1", "status": ExperimentStatus.PENDING}
    data = {"experiments": [exp]}
    saved = {}

    def submit_experiment(submitted_exp, hpc_type):
        assert submitted_exp["git_sha_at_submission"] == provenance
        assert hpc_type == HPCType.OLIVIA
        return "12345"

    def replace_exp_in_list(_experiments, new_exp):
        return [dict(new_exp)]

    def save_yaml(new_data):
        saved.update(new_data)

    submitted = maybe_submit(
        exp,
        {HPCType.OLIVIA: 0},
        data,
        concurrency_limits={HPCType.OLIVIA: 1},
        hpc_configs={},
        submit_experiment=submit_experiment,
        replace_exp_in_list=replace_exp_in_list,
        save_yaml=save_yaml,
    )

    assert submitted is True
    row = saved["experiments"][0]
    assert row["status"] == ExperimentStatus.QUEUED
    assert row["job_id"] == "12345"
    assert row["git_sha_at_submission"] == provenance


def test_maybe_submit_returns_false_without_capacity() -> None:
    exp = {"experiment_id": "exp-1", "status": ExperimentStatus.PENDING}
    data = {"experiments": [exp]}

    def unexpected_submit(_exp, _hpc_type):
        raise AssertionError("submission must not run without free capacity")

    def unexpected_save(_data):
        raise AssertionError("the ledger must not be saved when no submission occurred")

    submitted = maybe_submit(
        exp,
        {HPCType.OLIVIA: 1},
        data,
        concurrency_limits={HPCType.OLIVIA: 1},
        hpc_configs={},
        submit_experiment=unexpected_submit,
        replace_exp_in_list=lambda experiments, _exp: experiments,
        save_yaml=unexpected_save,
    )

    assert submitted is False


def test_maybe_submit_records_retry_receipt_before_consuming_capacity(monkeypatch) -> None:
    monkeypatch.setattr(submission, "capture_provenance", lambda: {})
    exp = {
        "experiment_id": "exp-1",
        "status": ExperimentStatus.PARTIAL,
        "hpc_assignment": HPCType.OLIVIA,
        "job_id": "old-job",
    }
    data = {"experiments": [exp]}
    concurrency_used = {HPCType.OLIVIA: 0}
    recorded: list[tuple[str, str | None]] = []

    submitted = maybe_submit(
        exp,
        concurrency_used,
        data,
        concurrency_limits={HPCType.OLIVIA: 1},
        hpc_configs={},
        submit_experiment=lambda _exp, _hpc: "new-job",
        replace_exp_in_list=lambda _experiments, new_exp: [new_exp],
        save_yaml=lambda _data: (_ for _ in ()).throw(AssertionError("journal path must not rewrite YAML")),
        record_submission=lambda accepted, previous: recorded.append((str(accepted["job_id"]), previous)),
    )

    assert submitted is True
    assert recorded == [("new-job", "old-job")]
    assert concurrency_used[HPCType.OLIVIA] == 1


def test_maybe_submit_stops_when_receipt_persistence_fails(monkeypatch) -> None:
    monkeypatch.setattr(submission, "capture_provenance", lambda: {})
    exp = {"experiment_id": "exp-1", "status": ExperimentStatus.PENDING}
    data = {"experiments": [exp]}
    concurrency_used = {HPCType.OLIVIA: 0}

    def fail_receipt(_accepted, _previous_job_id):
        raise OSError("receipt fsync failed")

    with pytest.raises(OSError, match="receipt fsync failed"):
        maybe_submit(
            exp,
            concurrency_used,
            data,
            concurrency_limits={HPCType.OLIVIA: 1},
            hpc_configs={},
            submit_experiment=lambda _exp, _hpc: "12345",
            replace_exp_in_list=lambda _experiments, new_exp: [new_exp],
            save_yaml=lambda _data: None,
            record_submission=fail_receipt,
        )

    assert concurrency_used[HPCType.OLIVIA] == 0

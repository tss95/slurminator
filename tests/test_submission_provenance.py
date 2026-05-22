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

    maybe_submit(
        exp,
        {HPCType.OLIVIA: 0},
        data,
        concurrency_limits={HPCType.OLIVIA: 1},
        hpc_configs={},
        submit_experiment=submit_experiment,
        replace_exp_in_list=replace_exp_in_list,
        save_yaml=save_yaml,
    )

    row = saved["experiments"][0]
    assert row["status"] == ExperimentStatus.QUEUED
    assert row["job_id"] == "12345"
    assert row["git_sha_at_submission"] == provenance

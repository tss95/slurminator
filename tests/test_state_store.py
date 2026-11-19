import pytest

from slurminator.experiments import ExperimentStatus
from slurminator.config import HPCType
from slurminator.experiments.yaml_utils import dump_yaml, load_yaml
from slurminator.state_store import ExperimentStateStore, replace_exp_in_list

pytestmark = pytest.mark.unit


def test_state_store_coerces_rows_and_resets_disabled_hpc(tmp_path) -> None:
    yaml_file = tmp_path / "experiments.yaml"
    dump_yaml(
        {
            "experiments": [
                {"experiment_id": "enabled", "status": "running", "hpc_assignment": "OLIVIA"},
                {"experiment_id": "disabled", "status": "queued", "hpc_assignment": "FOX"},
            ]
        },
        str(yaml_file),
    )
    store = ExperimentStateStore(yaml_file, concurrency_limits={HPCType.OLIVIA: 1, HPCType.FOX: 0})

    data = store.load()

    enabled, disabled = data["experiments"]
    assert enabled["status"] == ExperimentStatus.RUNNING
    assert enabled["hpc_assignment"] == HPCType.OLIVIA
    assert disabled["status"] == ExperimentStatus.PENDING
    assert disabled["hpc_assignment"] is None


def test_state_store_saves_and_replaces_without_orchestrator(tmp_path) -> None:
    yaml_file = tmp_path / "experiments.yaml"
    store = ExperimentStateStore(yaml_file)

    data = {"experiments": [{"experiment_id": "a", "status": ExperimentStatus.PENDING}]}
    store.save(data)
    loaded = load_yaml(str(yaml_file))
    replaced = replace_exp_in_list(loaded["experiments"], {"experiment_id": "a", "status": ExperimentStatus.RUNNING})

    assert replaced == [{"experiment_id": "a", "status": ExperimentStatus.RUNNING}]


def test_state_store_missing_file_returns_empty_experiments(tmp_path) -> None:
    store = ExperimentStateStore(tmp_path / "missing.yaml")

    assert store.load() == {"experiments": []}

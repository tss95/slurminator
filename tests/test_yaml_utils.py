from enum import Enum

import pytest

from slurminator.experiments import ExperimentStatus
from slurminator.experiments.yaml_utils import dump_yaml, load_yaml, register_yaml_enum

pytestmark = pytest.mark.unit


class AdapterEnum(Enum):
    VALUE = "value"


def test_dump_and_load_roundtrip_registered_enums(tmp_path):
    register_yaml_enum(AdapterEnum, "!AdapterEnum")
    path = tmp_path / "exp.yaml"
    data = {"status": ExperimentStatus.PENDING, "adapter": AdapterEnum.VALUE, "command": "echo one\necho two"}

    dump_yaml(data, path)
    loaded = load_yaml(path)

    assert loaded["status"] == ExperimentStatus.PENDING
    assert loaded["adapter"] == AdapterEnum.VALUE
    assert loaded["command"] == "echo one\necho two"

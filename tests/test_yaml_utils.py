from enum import Enum

import pytest
import yaml

from slurminator.config import HPCPartition, HPCType
from slurminator.experiments import ExperimentStatus
from slurminator.experiments.yaml_utils import ExperimentYAMLLoader, dump_yaml, load_yaml, register_yaml_enum

pytestmark = pytest.mark.unit


class AdapterEnum(Enum):
    VALUE = "value"


def test_dump_and_load_roundtrip_registered_enums(tmp_path):
    register_yaml_enum(AdapterEnum, "!AdapterEnum")
    path = tmp_path / "exp.yaml"
    shared = {"nested": 1}
    data = {
        "status": ExperimentStatus.PENDING,
        "hpc": HPCType.OLIVIA,
        "partition": HPCPartition.ACCEL,
        "adapter": AdapterEnum.VALUE,
        "command": "echo one\necho two",
        "first": shared,
        "second": shared,
    }

    dump_yaml(data, path)
    loaded = load_yaml(path)
    rendered = path.read_text(encoding="utf-8")

    assert loaded["status"] == ExperimentStatus.PENDING
    assert loaded["hpc"] == HPCType.OLIVIA
    assert loaded["partition"] == HPCPartition.ACCEL
    assert loaded["adapter"] == AdapterEnum.VALUE
    assert loaded["command"] == "echo one\necho two"
    assert "status: !ExperimentStatus 'pending'" in rendered
    assert "hpc: !HPCType 'OLIVIA'" in rendered
    assert "partition: !HPCPartition 'accel'" in rendered
    assert "command: |-" in rendered
    assert "&id" not in rendered
    assert "*id" not in rendered


def test_experiment_loader_prefers_libyaml_when_available() -> None:
    c_safe_loader = getattr(yaml, "CSafeLoader", None)

    if c_safe_loader is None:
        assert issubclass(ExperimentYAMLLoader, yaml.SafeLoader)
    else:
        assert issubclass(ExperimentYAMLLoader, c_safe_loader)


def test_c_and_python_loaders_preserve_registered_enum_semantics(tmp_path) -> None:
    register_yaml_enum(AdapterEnum, "!AdapterEnum")
    path = tmp_path / "exp.yaml"
    path.write_text(
        "\n".join(
            (
                "status: !ExperimentStatus 'running'",
                "hpc: !HPCType 'OLIVIA'",
                "partition: !HPCPartition 'accel'",
                "adapter: !AdapterEnum 'value'",
            )
        ),
        encoding="utf-8",
    )

    class PythonExperimentYAMLLoader(yaml.SafeLoader):
        pass

    for tag in ("!ExperimentStatus", "!HPCType", "!HPCPartition", "!AdapterEnum"):
        PythonExperimentYAMLLoader.add_constructor(tag, ExperimentYAMLLoader.yaml_constructors[tag])

    with path.open("r", encoding="utf-8") as handle:
        python_loaded = yaml.load(handle, Loader=PythonExperimentYAMLLoader)

    assert load_yaml(path) == python_loaded

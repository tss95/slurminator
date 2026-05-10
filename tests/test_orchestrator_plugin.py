from types import SimpleNamespace

import pytest

from slurminator.callbacks.status_normalization import GenericProgressSnapshot, MetricDisplayCandidate
from slurminator.callbacks.status_normalization import normalize_status_payload
from slurminator.experiments.status_enum import ExperimentStatus
from slurminator.plugins import CommandBuildContext, DefaultOrchestratorPlugin, OrchestratorPlugin, SimpleCommandPlugin

pytestmark = pytest.mark.unit


def test_default_plugin_is_protocol_compatible() -> None:
    plugin = DefaultOrchestratorPlugin()

    assert isinstance(plugin, OrchestratorPlugin)
    assert plugin.validate_experiment({}, {}) is False
    assert plugin.pinned_hpc_for_experiment({}) is None
    assert plugin.resource_overrides_for_experiment({}) == {}
    assert plugin.sbatch_export_vars(hpc_type="cluster", cluster_config=SimpleNamespace()) == {}
    assert plugin.extra_remote_dirs(base_path=SimpleNamespace(), experiment_file=SimpleNamespace()) == ()
    assert plugin.interpret_log_tail(exp={}, log_tail="", current_status="running", stage="pre_heuristics") is None
    assert plugin.interpret_log_tail(exp={}, log_tail="", current_status="running", stage="post_heuristics") is None
    plugin.annotate_log_tail(exp={}, log_tail="")

    with pytest.raises(NotImplementedError) as excinfo:
        plugin.build_commands_line({}, CommandBuildContext(gpus=1, hpc_type="cluster"))
    message = str(excinfo.value)
    assert "extra_command" in message
    assert "command" in message
    assert "SimpleCommandPlugin" in message
    assert "build_commands_line" in message


def test_default_plugin_uses_explicit_experiment_command() -> None:
    plugin = DefaultOrchestratorPlugin()
    context = CommandBuildContext(gpus=1, hpc_type="cluster")

    assert plugin.build_commands_line({"extra_command": "python train.py --epochs 1"}, context) == (
        "python train.py --epochs 1"
    )
    assert plugin.build_commands_line({"command": "bash scripts/train.sh"}, context) == "bash scripts/train.sh"


def test_simple_command_plugin_builds_entrypoint_config_command() -> None:
    plugin = SimpleCommandPlugin(
        entrypoint="python train.py",
        config_arg="--config",
        extra_args=("--no-progress",),
        sweep_params_arg="--overrides",
        multi_experiment_flag="--auto-retry",
    )

    command = plugin.build_commands_line(
        {
            "experiment_id": "exp-1",
            "config": "configs/train har.yaml",
            "command_args": ["--seed", "42"],
            "sweep_params": "model.lr=0.001;trainer.max_epochs=2",
        },
        CommandBuildContext(gpus=1, hpc_type="cluster", multi_experiment=True),
    )

    assert command == (
        "python train.py --config 'configs/train har.yaml' --no-progress --seed 42 "
        "--overrides 'model.lr=0.001;trainer.max_epochs=2' --auto-retry --orchestrator"
    )


def test_simple_command_plugin_accepts_config_path_alias() -> None:
    plugin = SimpleCommandPlugin(entrypoint=("python", "-m", "mypkg.train"))

    command = plugin.build_commands_line(
        {"experiment_id": "exp-1", "config_path": "configs/train.yaml"}, CommandBuildContext(gpus=1, hpc_type="cluster")
    )

    assert command == "python -m mypkg.train --config configs/train.yaml --orchestrator"


def test_simple_command_plugin_missing_config_field_is_actionable() -> None:
    plugin = SimpleCommandPlugin(entrypoint="python train.py")

    with pytest.raises(ValueError) as excinfo:
        plugin.build_commands_line({"experiment_id": "exp-1"}, CommandBuildContext(gpus=1, hpc_type="cluster"))

    message = str(excinfo.value)
    assert "config" in message
    assert "config_arg=None" in message
    assert "extra_command/command" in message


def test_default_plugin_projects_target_status_and_display_metrics() -> None:
    status = normalize_status_payload(
        experiment_id="exp-1",
        job_id="123",
        status="running",
        last_update=10.0,
        progress=GenericProgressSnapshot(unit="epoch", current_epoch=2, total_epochs=4),
        run_name="run-1",
        metrics={"val/acc": 0.9},
        primary_metric="val/acc",
        metric_info={"val/acc": MetricDisplayCandidate(shortform="acc", higher_better=True)},
        links={"tracker": "abc"},
    )
    plugin = DefaultOrchestratorPlugin()
    exp: dict = {}

    updated = plugin.project_status_to_experiment(exp, status)

    assert "status_schema_version" in updated
    assert exp["status_run_name"] == "run-1"
    assert exp["status_links"] == {"tracker": "abc"}
    assert exp["acc"] == 0.9
    assert plugin.extract_display_metrics(exp) == {"acc": 0.9}


def test_default_plugin_interprets_optional_status_marker_and_common_failure_keywords() -> None:
    plugin = DefaultOrchestratorPlugin()

    assert (
        plugin.interpret_log_tail(
            exp={},
            log_tail="training\nEXPERIMENT_STATUS=FAILED\n",
            current_status=ExperimentStatus.COMPLETED,
            stage="pre_heuristics",
        )
        == ExperimentStatus.FAILED
    )
    assert (
        plugin.interpret_log_tail(
            exp={},
            log_tail="slurmstepd: error: *** JOB 123 CANCELLED DUE TO TIME LIMIT ***",
            current_status=ExperimentStatus.COMPLETED,
            stage="heuristics",
        )
        == ExperimentStatus.TIMEOUT
    )
    assert (
        plugin.interpret_log_tail(
            exp={},
            log_tail="RuntimeError: CUDA out of memory.",
            current_status=ExperimentStatus.COMPLETED,
            stage="heuristics",
        )
        == ExperimentStatus.OOM
    )

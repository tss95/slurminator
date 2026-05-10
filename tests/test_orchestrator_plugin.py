from types import SimpleNamespace

import pytest

from slurminator.callbacks.status_normalization import GenericProgressSnapshot, MetricDisplayCandidate
from slurminator.callbacks.status_normalization import normalize_status_payload
from slurminator.plugins import CommandBuildContext, DefaultOrchestratorPlugin, OrchestratorPlugin

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

    with pytest.raises(NotImplementedError):
        plugin.build_commands_line({}, CommandBuildContext(gpus=1, hpc_type="cluster"))


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

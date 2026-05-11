from __future__ import annotations

import pytest

from slurminator.config import HPCType
from slurminator.hpc_orchestrator import HPCOrchestrator
from slurminator.plugins import DefaultOrchestratorPlugin

pytestmark = pytest.mark.unit


class _Dashboard:
    pass


def _overview(_experiments):
    return None


class _RuntimeHookPlugin(DefaultOrchestratorPlugin):
    def status_projection_options(self):
        return {"run_name_field": "custom_run_name"}

    def parse_sweep_overrides(self, raw):
        return {"raw": raw}

    def is_local_hpc(self, hpc_type):
        return hpc_type == HPCType.OLIVIA

    def dashboard_class(self):
        return _Dashboard

    def overview_printer(self):
        return _overview


def test_hpc_orchestrator_reads_optional_runtime_hooks_from_plugin(tmp_path):
    exp_file = tmp_path / "experiments.yaml"
    exp_file.write_text("experiments: []")

    orch = HPCOrchestrator(str(exp_file), concurrency_limits={}, plugin=_RuntimeHookPlugin())

    assert orch.projection_options == {"run_name_field": "custom_run_name"}
    assert orch.parse_overrides("a=1") == {"raw": "a=1"}
    assert orch.is_local_hpc_fn(HPCType.OLIVIA) is True
    assert orch.is_local_hpc_fn(HPCType.FOX) is False
    assert orch.dashboard_cls is _Dashboard
    assert orch.overview_printer is _overview


def test_hpc_orchestrator_explicit_constructor_hooks_override_plugin(tmp_path):
    exp_file = tmp_path / "experiments.yaml"
    exp_file.write_text("experiments: []")

    def parse(raw):
        return {"explicit": raw}

    orch = HPCOrchestrator(
        str(exp_file),
        concurrency_limits={},
        plugin=_RuntimeHookPlugin(),
        projection_options={"run_name_field": "explicit"},
        parse_overrides=parse,
        is_local_hpc_fn=lambda _hpc: False,
        dashboard_cls=dict,
        overview_printer=_overview,
    )

    assert orch.projection_options == {"run_name_field": "explicit"}
    assert orch.parse_overrides("a=1") == {"explicit": "a=1"}
    assert orch.is_local_hpc_fn(HPCType.OLIVIA) is False
    assert orch.dashboard_cls is dict
    assert orch.overview_printer is _overview

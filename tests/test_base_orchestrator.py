from datetime import datetime
import types

import pytest

from slurminator import base_orchestrator as base_module
from slurminator.base_orchestrator import BaseOrchestrator
from slurminator.experiments import CustomSweepCase, CustomSweepConfig, ExperimentConfig, ExperimentStatus
from slurminator.experiments import MasterExperimentConfig
from slurminator.experiments.yaml_utils import load_yaml

pytestmark = pytest.mark.unit


@pytest.fixture
def make_orchestrator():
    return BaseOrchestrator(MasterExperimentConfig())


def test_build_override_str_and_merge(make_orchestrator):
    orchestrator = make_orchestrator
    part1 = orchestrator._build_override_str(training_configs__num_epochs=5)
    part2 = orchestrator._build_override_str(**{"model.d_model": 16})
    merged = orchestrator._merge_override_strings(part1 + ";", None, "  " + part2)
    assert merged == "training_configs.num_epochs=5;model.d_model=16"


def test_seeded_respects_overrides(make_orchestrator):
    orchestrator = make_orchestrator
    orchestrator.master_config.seeds = [1, 2]
    orchestrator.master_config.dataset_seed_overrides = {"ds1": [9]}
    template = ExperimentConfig(
        task_type="self_supervised",
        dataset_name="ds1",
        experiment_id="exp",
        status=ExperimentStatus.PENDING,
        metadata={},
    )
    runs = orchestrator._seeded(template, "ds1")
    assert [run.experiment_id for run in runs] == ["exp_s9"]
    assert runs[0].metadata["seed"] == 9
    assert runs[0].sweep_params is None


def test_get_project_name_stable(make_orchestrator, monkeypatch):
    orchestrator = make_orchestrator
    fixed_time = datetime(2020, 1, 2, 3, 4, 5)
    monkeypatch.setattr(base_module, "datetime", types.SimpleNamespace(now=lambda: fixed_time))
    name1 = orchestrator._get_project_name("test")
    name2 = orchestrator._get_project_name("test")
    assert name1 == name2 == "test_20200102_030405"
    assert orchestrator._get_project_name("other") == "other_20200102_030405"


def test_generate_experiment_file_includes_git_provenance_metadata(make_orchestrator, monkeypatch, tmp_path):
    orchestrator = make_orchestrator
    orchestrator.output_dir = tmp_path
    provenance = {"project": "a" * 40, "slurminator": "b" * 40}
    monkeypatch.setattr(base_module, "capture_provenance", lambda: provenance)
    orchestrator.experiments = [
        ExperimentConfig(
            task_type="self_supervised",
            dataset_name="demo",
            experiment_id="demo_exp",
            status=ExperimentStatus.PENDING,
            metadata={},
        )
    ]

    path = orchestrator.generate_experiment_file()

    data = load_yaml(path)
    assert data["metadata"]["project_git_sha"] == provenance["project"]
    assert data["metadata"]["slurminator_git_sha"] == provenance["slurminator"]
    assert data["metadata"]["experiment_count"] == 1
    assert data["experiments"][0]["experiment_id"] == "demo_exp"


def test_custom_sweep_generates_seeded_runs(make_orchestrator):
    orchestrator = make_orchestrator
    orchestrator.master_config.run_custom_sweeps = True
    orchestrator.master_config.custom_sweeps = [
        CustomSweepConfig(
            datasets=["UCR_FordA"],
            sweep_keys={"tokenizer_patch_size": [0.12, 0.05], "tokenizer_stride": [0.5]},
            num_epochs=150,
            parameters_prefix={"tokenizer_patch_size": "patch", "tokenizer_stride": "stride"},
            run_name_prefix="FordAB_patch",
            wandb_project="FordAB_Patch_Ablations",
        )
    ]

    orchestrator.generate_all_experiments()

    experiments = orchestrator.experiments
    assert len(experiments) == 6
    ids = {exp.experiment_id for exp in experiments}
    assert any("patch0_12_stride0_5" in exp_id for exp_id in ids)

    sample = next(exp for exp in experiments if "patch0_12_stride0_5" in exp.experiment_id)
    params = dict(part.split("=", 1) for part in sample.sweep_params.split(";"))
    assert params["model.tokenizer_patch_size"] == "0.12"
    assert params["model.tokenizer_stride"] == "0.5"
    assert params["training_configs.num_epochs"] == "150"
    assert sample.metadata["ablation_type"] == "custom_sweep"
    assert sample.metadata["wandb_project"] == "FordAB_Patch_Ablations"
    assert len({exp.metadata["seed"] for exp in experiments}) == 3


def test_custom_sweep_named_cases(make_orchestrator):
    orchestrator = make_orchestrator
    orchestrator.master_config.run_custom_sweeps = True
    orchestrator.master_config.custom_sweeps = [
        CustomSweepConfig(
            datasets=["HAR"],
            num_epochs=120,
            run_name_prefix="memory_variant",
            experiment_prefix="memory_variant",
            cases=[
                CustomSweepCase(
                    name="stateful", overrides={"model.core.experiment_mode": "stateful", "loss.lambda_aux": 0.25}
                ),
                CustomSweepCase(
                    name="stateless", overrides={"model.core.experiment_mode": "stateless", "loss.lambda_aux": 0.0}
                ),
            ],
        )
    ]

    orchestrator.generate_all_experiments()

    experiments = orchestrator.experiments
    assert len(experiments) == 6
    stateless = next(exp for exp in experiments if "stateless" in exp.experiment_id)
    params = dict(part.split("=", 1) for part in stateless.sweep_params.split(";"))
    assert params["model.core.experiment_mode"] == "stateless"
    assert float(params["loss.lambda_aux"]) == 0.0
    assert params["training_configs.num_epochs"] == "120"


def test_custom_sweep_infers_epochs_from_step_budget_when_unset(make_orchestrator):
    orchestrator = make_orchestrator
    orchestrator.master_config.run_custom_sweeps = True
    orchestrator.master_config.custom_sweeps = [
        CustomSweepConfig(
            datasets=["ETTh1"],
            cases=[
                CustomSweepCase(
                    name="step_budget",
                    overrides={"training_configs.max_train_steps": 10000, "training_configs.pseudo_epoch_steps": 100},
                )
            ],
            num_epochs=None,
        )
    ]

    orchestrator.generate_all_experiments()

    sample = orchestrator.experiments[0]
    params = dict(part.split("=", 1) for part in sample.sweep_params.split(";"))
    assert params["training_configs.num_epochs"] == "100"


def test_checkpoint_probe_requires_adapter_hook(make_orchestrator):
    orchestrator = make_orchestrator

    with pytest.raises(NotImplementedError, match="checkpoint_probe requires a project adapter"):
        orchestrator._apply_checkpoint_probe_overrides(
            {}, dataset_name="HAR", resume_from="/tmp/checkpoint.pt", epoch_offset=1
        )

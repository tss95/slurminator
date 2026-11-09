from datetime import datetime
import logging
import types

import pytest

from slurminator import base_orchestrator as base_module
from slurminator.base_orchestrator import BaseOrchestrator
from slurminator.experiments import CustomSweepCase, CustomSweepConfig, ExperimentConfig, ExperimentStatus
from slurminator.experiments import MasterExperimentConfig
from slurminator.experiments.yaml_utils import load_yaml
from slurminator.state_store import ExperimentStateStore

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
    assert "experiment_name" not in data["metadata"]
    assert "experiment_description" not in data["metadata"]
    assert data["experiments"][0]["experiment_id"] == "demo_exp"


def test_generated_experiment_info_survives_ledger_updates(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    config = MasterExperimentConfig(
        experiment_name="HAR encoder comparison",
        experiment_description="Compare [baseline] and smaller encoders on HAR.\n\nEvaluate accuracy and memory.",
        run_custom_sweeps=True,
        custom_sweeps=[CustomSweepConfig(dataset_name="HAR", cases=[CustomSweepCase(name="baseline")], seeds=[42])],
    )
    orchestrator = BaseOrchestrator(config)
    orchestrator.generate_all_experiments()
    path = orchestrator.generate_experiment_file()
    store = ExperimentStateStore(path)
    data = store.load()
    assert data["metadata"]["experiment_name"] == config.experiment_name
    assert data["metadata"]["experiment_description"] == config.experiment_description

    data["experiments"][0]["status"] = ExperimentStatus.COMPLETED
    store.save(data)
    resumed = ExperimentStateStore(path).load()

    assert resumed["metadata"] == data["metadata"]
    assert resumed["experiments"][0]["status"] == ExperimentStatus.COMPLETED
    assert "experiment_name" not in resumed["experiments"][0]["sweep_params"]
    assert "experiment_description" not in resumed["experiments"][0]["sweep_params"]


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


def test_custom_sweep_prefix_named_cases_are_all_generated(make_orchestrator):
    """Regression: case ids that are string-prefixes of earlier seeded ids were silently dropped.

    ``..._b7`` is a prefix of ``..._b70_s42`` (and ``b35`` of ``b3500``, ``b14`` of
    ``b140``); the old ``startswith`` dedupe treated them as duplicates.
    """
    orchestrator = make_orchestrator
    orchestrator.master_config.run_custom_sweeps = True
    budgets = ["b70", "b140", "b700", "b3500", "b7000", "b35", "b14", "b7", "b4"]
    arms = ("skynet", "pmt")
    seeds = [42, 7, 123]
    orchestrator.master_config.custom_sweeps = [
        CustomSweepConfig(
            datasets=["SB_CREW"],
            experiment_prefix="sb_crew_test_sup",
            seeds=seeds,
            num_seeds=len(seeds),
            cases=[CustomSweepCase(name=f"tsup_{arm}_{budget}", overrides={"budget": budget}) for budget in budgets],
        )
        for arm in arms
    ]

    orchestrator.generate_all_experiments()

    ids = {exp.experiment_id for exp in orchestrator.experiments}
    expected = {
        f"sb_crew_test_sup_SB_CREW_tsup_{arm}_{budget}_s{seed}" for arm in arms for budget in budgets for seed in seeds
    }
    assert ids == expected
    assert len(orchestrator.experiments) == 54
    assert orchestrator.skipped_duplicate_templates == []


def test_custom_sweep_exact_duplicate_case_is_skipped_with_warning(make_orchestrator, caplog):
    orchestrator = make_orchestrator
    orchestrator.master_config.run_custom_sweeps = True
    orchestrator.master_config.custom_sweeps = [
        CustomSweepConfig(
            datasets=["HAR"],
            experiment_prefix="abl",
            seeds=[1, 2],
            cases=[CustomSweepCase(name="b7", overrides={"x": 1})],
        ),
        CustomSweepConfig(
            datasets=["HAR"],
            experiment_prefix="abl",
            seeds=[1, 2],
            cases=[CustomSweepCase(name="b7", overrides={"x": 2}), CustomSweepCase(name="b70", overrides={"x": 3})],
        ),
    ]

    with caplog.at_level(logging.WARNING, logger="slurminator"):
        orchestrator.generate_all_experiments()

    ids = {exp.experiment_id for exp in orchestrator.experiments}
    assert ids == {"abl_HAR_b7_s1", "abl_HAR_b7_s2", "abl_HAR_b70_s1", "abl_HAR_b70_s2"}
    first = next(exp for exp in orchestrator.experiments if exp.experiment_id == "abl_HAR_b7_s1")
    assert "x=1" in first.sweep_params.split(";")
    assert orchestrator.skipped_duplicate_templates == ["abl_HAR_b7"]

    messages = [record.getMessage() for record in caplog.records if record.levelno >= logging.WARNING]
    assert any("Skipping abl_HAR_b7:" in message and "abl_HAR_b7_s1" in message for message in messages)
    assert any("Skipped 1 custom-sweep template(s) as duplicates: abl_HAR_b7" in message for message in messages)

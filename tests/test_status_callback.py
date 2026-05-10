from types import SimpleNamespace

import pytest

from slurminator.callbacks.status_callback import OrchestratorStatusCallback
from slurminator.callbacks.status_normalization import MetricDisplayCandidate
from slurminator.schemas.status_schema import OrchestratorStatus

pytestmark = pytest.mark.unit


class FakeClock:
    def __init__(self, initial: float = 100.0) -> None:
        self.now = float(initial)

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += float(seconds)


class DummyTrainer:
    def __init__(self, cfg: object) -> None:
        self.cfg = cfg
        self.rank = 0
        self.current_epoch = 0
        self.global_step = 0
        self.epochs = int(getattr(cfg.training_configs, "num_epochs", 3))
        self.max_train_steps = 10

    def get_runtime_progress(self) -> dict[str, int]:
        return {
            "current_epoch": self.current_epoch,
            "max_epochs": self.epochs,
            "current_step": self.global_step,
            "max_steps": self.max_train_steps,
        }


def make_cfg(**overrides):
    cfg = SimpleNamespace(
        orchestrated=True,
        experiment_id="exp-1",
        run_name="cfg-run",
        training_configs=SimpleNamespace(num_epochs=3),
        optimizer_parameters=SimpleNamespace(batch_size=4),
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def read_status(callback: OrchestratorStatusCallback) -> OrchestratorStatus:
    assert callback.status_file is not None
    return OrchestratorStatus.model_validate_json(callback.status_file.read_text())


def test_status_callback_schema_roundtrip_and_sweep_path(tmp_path):
    clock = FakeClock()
    cfg = make_cfg()
    trainer = DummyTrainer(cfg)
    callback = OrchestratorStatusCallback(
        cfg=cfg, save_path=tmp_path, job_id="12345", sweep_id="sweep-1", min_write_interval_seconds=0.0, time_fn=clock
    )

    callback.on_train_start(trainer)

    status = read_status(callback)
    assert status.schema_version == "1.0"
    assert status.status == "initializing"
    assert status.experiment_id == "exp-1"
    assert status.display.run_name == "cfg-run"
    assert callback.status_file == tmp_path / ".orchestrator_status" / "sweep_sweep-1" / "status_12345.json"
    assert OrchestratorStatus.model_validate_json(status.model_dump_json()) == status


def test_status_callback_state_machine_rejects_backward_transition(tmp_path):
    clock = FakeClock()
    cfg = make_cfg()
    trainer = DummyTrainer(cfg)
    callback = OrchestratorStatusCallback(
        cfg=cfg, save_path=tmp_path, job_id="12345", min_write_interval_seconds=0.0, time_fn=clock
    )

    callback.on_train_start(trainer)
    callback.on_train_batch_end(trainer, batch=None, batch_idx=0, logs={})
    assert read_status(callback).status == "running"
    callback.on_train_end(trainer)
    assert read_status(callback).status == "completed"

    with pytest.raises(ValueError, match="completed -> running"):
        callback.on_train_batch_end(trainer, batch=None, batch_idx=1, logs={})


def test_status_callback_throttles_and_materializes_display_strictly(tmp_path):
    clock = FakeClock()
    cfg = make_cfg()
    trainer = DummyTrainer(cfg)
    callback = OrchestratorStatusCallback(
        cfg=cfg,
        save_path=tmp_path,
        job_id="12345",
        primary_metric="val/acc",
        secondary_metric="val/loss",
        metric_info={
            "val/acc": MetricDisplayCandidate(shortform="acc", higher_better=True),
            "val/loss": MetricDisplayCandidate(shortform="loss", higher_better=False),
        },
        min_write_interval_seconds=10.0,
        time_fn=clock,
    )

    callback.on_train_start(trainer)
    initial = read_status(callback)
    assert initial.metrics == {}
    assert initial.display.primary_metric is None
    assert initial.display.metric_info == {}

    trainer.global_step = 1
    clock.advance(1.0)
    callback.on_train_batch_end(trainer, batch=None, batch_idx=0, logs={})
    assert read_status(callback).progress.current_step == 1

    trainer.global_step = 2
    clock.advance(1.0)
    callback.on_train_batch_end(trainer, batch=None, batch_idx=1, logs={})
    assert read_status(callback).progress.current_step == 1

    callback.update_metrics({"val/acc": 0.91}, trainer=trainer)
    updated = read_status(callback)
    assert updated.display.primary_metric == "val/acc"
    assert updated.display.secondary_metric is None
    assert set(updated.display.metric_info) == {"val/acc"}


def test_status_callback_atomic_replace_keeps_existing_file_readable(tmp_path):
    clock = FakeClock()
    cfg = make_cfg()
    trainer = DummyTrainer(cfg)
    callback = OrchestratorStatusCallback(
        cfg=cfg, save_path=tmp_path, job_id="12345", min_write_interval_seconds=0.0, time_fn=clock
    )

    callback.on_train_start(trainer)
    observations: list[str] = []

    def observe_before_replace(tmp_path_arg, final_path):
        assert tmp_path_arg.exists()
        observations.append(OrchestratorStatus.model_validate_json(final_path.read_text()).status)

    callback._pre_replace_hook = observe_before_replace
    callback.on_epoch_end(trainer, epoch=0, train_logs={"loss": 1.25}, val_logs={})

    assert observations == ["initializing"]
    assert read_status(callback).metrics["train/loss"] == 1.25
    assert not list(callback.status_file.parent.glob("*.tmp"))

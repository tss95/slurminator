from types import SimpleNamespace

import pytest

from slurminator.callbacks.status_callback import OrchestratorStatusCallback
from slurminator.callbacks.status_normalization import MetricDisplayCandidate
from slurminator.schemas.status_schema import HistoryEntry, OrchestratorStatus

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


def read_history(callback: OrchestratorStatusCallback) -> list[HistoryEntry]:
    assert callback.history_file is not None
    return [HistoryEntry.model_validate_json(line) for line in callback.history_file.read_text().splitlines()]


def test_status_callback_schema_roundtrip_and_sweep_path(tmp_path):
    clock = FakeClock()
    cfg = make_cfg()
    trainer = DummyTrainer(cfg)
    callback = OrchestratorStatusCallback(
        cfg=cfg, save_path=tmp_path, job_id="12345", sweep_id="sweep-1", min_write_interval_seconds=0.0, time_fn=clock
    )

    callback.on_train_start(trainer)

    status = read_status(callback)
    assert status.schema_version == "1.1"
    assert status.attempt == 1
    assert status.status == "initializing"
    assert status.experiment_id == "exp-1"
    assert status.display.run_name == "cfg-run"
    assert callback.status_file == tmp_path / ".orchestrator_status" / "sweep_sweep-1" / "status_12345.json"
    assert callback.history_file == tmp_path / ".orchestrator_status" / "sweep_sweep-1" / "history_12345.jsonl"
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
            "val/acc": MetricDisplayCandidate(shortform="acc", higher_better=True, best_key="val/best_acc"),
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
    assert updated.display.metric_info["val/acc"].best_key == "val/best_acc"


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


def test_status_callback_writes_one_history_line_per_epoch_for_fresh_run(tmp_path):
    clock = FakeClock()
    cfg = make_cfg()
    trainer = DummyTrainer(cfg)
    callback = OrchestratorStatusCallback(
        cfg=cfg, save_path=tmp_path, job_id="12345", min_write_interval_seconds=0.0, time_fn=clock
    )

    callback.on_train_start(trainer)
    assert callback.history_file is not None
    assert not callback.history_file.exists()

    callback.on_epoch_end(trainer, epoch=0, train_logs={"loss": 1.25}, val_logs={"acc": 0.5})
    trainer.current_epoch = 1
    clock.advance(1.0)
    callback.on_epoch_end(trainer, epoch=1, train_logs={"loss": 0.75}, val_logs={"acc": 0.75})

    history = read_history(callback)
    assert [entry.attempt for entry in history] == [1, 1]
    assert [entry.epoch for entry in history] == [1, 2]
    assert [entry.unit for entry in history] == ["epoch", "epoch"]
    assert history[0].metrics == {"train/loss": 1.25, "val/acc": 0.5}
    assert history[1].metrics == {"train/loss": 0.75, "val/acc": 0.75}
    assert read_status(callback).attempt == 1


def test_status_callback_resume_increments_attempt_from_existing_history(tmp_path):
    status_dir = tmp_path / ".orchestrator_status"
    status_dir.mkdir()
    history_file = status_dir / "history_12345.jsonl"
    history_file.write_text(
        "\n".join(
            [
                HistoryEntry(timestamp=90.0, attempt=1, epoch=1, step=5, metrics={"train/loss": 1.0}).model_dump_json(),
                HistoryEntry(
                    timestamp=95.0, attempt=1, epoch=2, step=10, metrics={"train/loss": 0.8}
                ).model_dump_json(),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    clock = FakeClock()
    cfg = make_cfg()
    trainer = DummyTrainer(cfg)
    callback = OrchestratorStatusCallback(
        cfg=cfg, save_path=tmp_path, job_id="12345", min_write_interval_seconds=0.0, time_fn=clock
    )

    callback.on_train_start(trainer)
    callback.on_epoch_end(trainer, epoch=0, train_logs={"loss": 0.6}, val_logs={})

    history = read_history(callback)
    assert [entry.attempt for entry in history] == [1, 1, 2]
    assert history[-1].unit == "epoch"
    assert history[-1].metrics == {"train/loss": 0.6}
    assert read_status(callback).attempt == 2


def test_status_callback_history_append_uses_status_write_throttle(tmp_path):
    clock = FakeClock()
    cfg = make_cfg()
    trainer = DummyTrainer(cfg)
    callback = OrchestratorStatusCallback(
        cfg=cfg, save_path=tmp_path, job_id="12345", min_write_interval_seconds=10.0, time_fn=clock
    )

    callback.on_train_start(trainer)
    callback.update_metrics({"val/acc": 0.9}, trainer=trainer)
    assert len(read_history(callback)) == 1

    trainer.global_step = 1
    clock.advance(1.0)
    callback.on_train_batch_end(trainer, batch=None, batch_idx=0, logs={})

    assert len(read_history(callback)) == 1


def test_status_callback_history_metric_hook_can_filter_metrics(tmp_path):
    class FilteredHistoryCallback(OrchestratorStatusCallback):
        def _history_metrics(self, status: OrchestratorStatus) -> dict[str, float]:
            return {key: value for key, value in status.metrics.items() if key == "val/acc"}

    clock = FakeClock()
    cfg = make_cfg()
    trainer = DummyTrainer(cfg)
    callback = FilteredHistoryCallback(
        cfg=cfg, save_path=tmp_path, job_id="12345", min_write_interval_seconds=0.0, time_fn=clock
    )

    callback.on_train_start(trainer)
    callback.on_epoch_end(trainer, epoch=0, train_logs={"loss": 1.25}, val_logs={"acc": 0.5})

    history = read_history(callback)
    assert history[0].unit == "epoch"
    assert history[0].metrics == {"val/acc": 0.5}

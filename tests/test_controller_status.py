"""Synthetic checks for durable compact controller receipts and ownership."""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

import pytest

import slurminator.controller_status as status_module
from slurminator.controller_status import ControllerStatusPublisher, ledger_fingerprint, status_path
from slurminator.experiments import ExperimentStatus

pytestmark = pytest.mark.unit


def test_receipt_reports_progress_counts_and_missing_metrics_without_histories(tmp_path: Path) -> None:
    ledger = tmp_path / "experiments.yaml"
    ledger.write_text("experiments: []\n")
    rows = [
        {"experiment_id": "done", "status": ExperimentStatus.COMPLETED, "all_metrics": {"test": 0.9}},
        {
            "experiment_id": "active",
            "status": "running",
            "current_step": 5,
            "max_steps": 10,
            "last_metrics_update": 10.0,
            "history": ["must not appear"],
        },
        {"experiment_id": "queued", "status": "queued"},
        {"experiment_id": "failed", "status": "out_of_memory"},
    ]
    with ControllerStatusPublisher(ledger) as publisher:
        receipt = publisher.publish("running", rows)
    assert receipt == json.loads(status_path(ledger).read_text())
    assert receipt["counts"] == {
        "pending": 0,
        "queued": 1,
        "running": 1,
        "completed": 1,
        "failed": 0,
        "partial": 0,
        "timeout": 0,
        "out_of_memory": 1,
        "cancelled": 0,
        "killed": 0,
        "total": 4,
    }
    assert receipt["progress"]["fraction"] == 0.5
    assert receipt["progress"]["excluded_failed_runs"] == 1
    assert receipt["runs"][1]["last_metrics_update"] == 10.0
    assert receipt["missing_metrics"] == ["active", "queued", "failed"]
    assert "history" not in status_path(ledger).read_text()
    assert receipt["ledger_fingerprint"] is None
    assert receipt["results_finalized"] is False


def test_finished_receipt_identifies_durably_saved_ledger(tmp_path: Path, monkeypatch) -> None:
    ledger = tmp_path / "experiments.yaml"
    ledger.write_text("experiments: []\n")
    syncs = []
    monkeypatch.setattr(status_module.os, "fsync", lambda descriptor: syncs.append(descriptor))
    with ControllerStatusPublisher(ledger) as publisher:
        final_fingerprint = ledger_fingerprint(ledger, durable=True)
        receipt = publisher.publish(
            "finished",
            [{"experiment_id": "one", "status": "failed"}],
            finalized=True,
            expected_fingerprint=final_fingerprint,
        )
    assert len(syncs) == 4  # Ledger data/directory, then receipt data/directory.
    assert receipt["outcome"] == "completed_with_failures"
    assert receipt["results_finalized"] is True
    assert receipt["ledger_fingerprint"] == {
        "sha256": hashlib.sha256(ledger.read_bytes()).hexdigest(),
        "size": ledger.stat().st_size,
        "mtime_ns": ledger.stat().st_mtime_ns,
    }
    ledger.write_text("experiments: []\n# changed\n")
    assert ledger_fingerprint(ledger) != receipt["ledger_fingerprint"]


def test_cadence_is_bounded_and_state_changes_are_immediate(tmp_path: Path, monkeypatch) -> None:
    now = [0.0]
    monkeypatch.setattr(status_module.time, "monotonic", lambda: now[0])
    with ControllerStatusPublisher(tmp_path / "ledger.yaml", heartbeat_interval_seconds=10) as publisher:
        assert publisher.publish("running", []) is not None
        now[0] = 9.9
        assert publisher.publish("running", []) is None
        now[0] = 10.0
        assert publisher.publish("running", []) is not None
        assert publisher.publish("finalizing", []) is not None
        assert publisher.publish("error", [], error="broken") is not None


def test_duplicate_controller_does_not_overwrite_receipt_and_lock_is_reusable(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.yaml"
    with ControllerStatusPublisher(ledger) as first:
        first.publish("running", [])
        original = first.path.read_bytes()
        with pytest.raises(RuntimeError, match="already holds"):
            with ControllerStatusPublisher(ledger):
                pytest.fail("duplicate acquired lock")
        assert first.path.read_bytes() == original
    with ControllerStatusPublisher(ledger) as second:
        assert second.run_id != first.run_id
        second.publish("starting", [])
    assert json.loads(status_path(ledger).read_text())["run_id"] == second.run_id


def test_failed_atomic_replace_keeps_previous_snapshot(tmp_path: Path, monkeypatch) -> None:
    with ControllerStatusPublisher(tmp_path / "ledger.yaml") as publisher:
        publisher.publish("running", [])
        previous = publisher.path.read_bytes()
        monkeypatch.setattr(status_module.os, "replace", lambda *_args: (_ for _ in ()).throw(OSError("no space")))
        with pytest.raises(OSError, match="no space"):
            publisher.publish("finalizing", [])
        assert publisher.path.read_bytes() == previous
        assert not list(publisher.path.parent.glob(f".{publisher.path.name}.*"))


@pytest.mark.parametrize("state,finalized", [("finished", False), ("running", True), ("unknown", False)])
def test_invalid_finalization_contract_is_rejected(tmp_path: Path, state: str, finalized: bool) -> None:
    with ControllerStatusPublisher(tmp_path / "ledger.yaml") as publisher:
        with pytest.raises(ValueError):
            publisher.publish(state, [], finalized=finalized)


def test_nonfinite_or_invalid_optional_counters_remain_strict_json(tmp_path: Path) -> None:
    with ControllerStatusPublisher(tmp_path / "ledger.yaml") as publisher:
        receipt = publisher.publish(
            "running",
            [
                {
                    "experiment_id": "one",
                    "status": "running",
                    "current_step": float("nan"),
                    "max_steps": float("inf"),
                    "current_epoch": 1,
                    "max_epochs": 2,
                    "last_metrics_update": "invalid",
                }
            ],
        )
    assert receipt["progress"]["fraction"] == 0.5
    assert receipt["runs"][0]["current_step"] is None
    assert receipt["runs"][0]["max_steps"] is None
    assert receipt["runs"][0]["last_metrics_update"] is None


def test_independent_heartbeat_preserves_snapshot_and_does_not_starve_progress(tmp_path: Path, monkeypatch) -> None:
    heartbeat_seen = threading.Event()
    heartbeats = []
    publisher = ControllerStatusPublisher(tmp_path / "ledger.yaml", heartbeat_interval_seconds=0.01)
    write = publisher._write_payload

    def observe_write(payload: str) -> None:
        write(payload)
        snapshot = json.loads(payload)
        if snapshot["heartbeat_at"] != snapshot["progress_updated_at"]:
            heartbeats.append(snapshot)
            if len(heartbeats) >= 3:
                heartbeat_seen.set()

    monkeypatch.setattr(publisher, "_write_payload", observe_write)
    rows = [{"experiment_id": "one", "status": "running", "current_step": 1, "max_steps": 10}]
    with publisher:
        initial = publisher.publish("running", rows)
        original_progress_time = initial["progress_updated_at"]
        initial["counts"]["total"] = 999  # Caller mutation must not alter the heartbeat's cached snapshot.
        rows[0]["current_step"] = 8  # Controller work has not published its next snapshot yet.
        assert heartbeat_seen.wait(1), "heartbeat stopped while controller was busy"
        heartbeat = heartbeats[-1]
        assert heartbeat["counts"]["total"] == 1
        assert heartbeat["progress"]["fraction"] == 0.1
        assert heartbeat["progress_updated_at"] == original_progress_time
        assert heartbeat["updated_at"] == heartbeat["heartbeat_at"] > original_progress_time
        refreshed = publisher.publish("running", rows)
        assert refreshed is not None  # Heartbeat must not reset substantive publication cadence.
        assert refreshed["progress"]["fraction"] == 0.8
        assert refreshed["progress_updated_at"] > original_progress_time
    assert not publisher._heartbeat_thread.is_alive()


@pytest.mark.parametrize("terminal_state", ["finished", "stopped", "error"])
def test_terminal_publication_waits_for_active_heartbeat_and_stops_it(
    tmp_path: Path, monkeypatch, terminal_state
) -> None:
    ledger = tmp_path / "ledger.yaml"
    ledger.write_text("experiments: []\n")
    publisher = ControllerStatusPublisher(ledger, heartbeat_interval_seconds=0.01)
    heartbeat_entered = threading.Event()
    release_heartbeat = threading.Event()
    terminal_started = threading.Event()
    terminal_done = threading.Event()
    write = publisher._write_payload
    failures = []

    def block_heartbeat(payload: str) -> None:
        if json.loads(payload)["heartbeat_at"] != json.loads(payload)["progress_updated_at"]:
            heartbeat_entered.set()
            assert release_heartbeat.wait(1)
        write(payload)

    def finish() -> None:
        terminal_started.set()
        try:
            publisher.publish(
                terminal_state,
                [],
                finalized=terminal_state == "finished",
                expected_fingerprint=ledger_fingerprint(ledger, durable=True) if terminal_state == "finished" else None,
            )
        except BaseException as exc:
            failures.append(exc)
        finally:
            terminal_done.set()

    monkeypatch.setattr(publisher, "_write_payload", block_heartbeat)
    with publisher:
        publisher.publish("running", [])
        assert heartbeat_entered.wait(1)
        finisher = threading.Thread(target=finish)
        finisher.start()
        try:
            assert terminal_started.wait(1)
            assert not terminal_done.is_set()  # Serialized behind the in-flight heartbeat write.
        finally:
            release_heartbeat.set()
            finisher.join(1)
        assert not finisher.is_alive()
        assert not failures
        publisher._heartbeat_thread.join(1)
        assert not publisher._heartbeat_thread.is_alive()
        assert json.loads(publisher.path.read_text())["state"] == terminal_state
        with pytest.raises(RuntimeError, match="terminal controller receipt"):
            publisher.publish("running", [])
    assert json.loads(publisher.path.read_text())["state"] == terminal_state


def test_background_heartbeat_failure_propagates_on_publication_and_exit(tmp_path: Path, monkeypatch) -> None:
    publisher = ControllerStatusPublisher(tmp_path / "ledger.yaml", heartbeat_interval_seconds=0.01)
    with pytest.raises(RuntimeError, match="heartbeat failed"):
        with publisher:
            publisher.publish("running", [])
            monkeypatch.setattr(
                publisher, "_write_payload", lambda _payload: (_ for _ in ()).throw(OSError("disk full"))
            )
            assert publisher._heartbeat_stop.wait(1)
            with pytest.raises(RuntimeError, match="heartbeat failed"):
                publisher.publish("running", [])
    assert not publisher._heartbeat_thread.is_alive()


def test_finished_receipt_rejects_an_edit_since_final_save(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.yaml"
    ledger.write_text("experiments: []\n")
    final_fingerprint = ledger_fingerprint(ledger, durable=True)
    with ControllerStatusPublisher(ledger) as publisher:
        publisher.publish("finalizing", [])
        ledger.write_text("experiments: []\n# external edit\n")
        with pytest.raises(RuntimeError, match="changed after the final controller save"):
            publisher.publish("finished", [], finalized=True, expected_fingerprint=final_fingerprint)
        assert json.loads(publisher.path.read_text())["state"] == "finalizing"

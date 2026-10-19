"""Observer readiness, freshness and silent-wait regression tests."""

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import pytest

from slurminator import observer
from slurminator.agent import main
from slurminator.controller_status import ControllerStatusPublisher, ledger_fingerprint, status_path
from slurminator.experiments.status_enum import ExperimentStatus

pytestmark = pytest.mark.unit


@pytest.fixture
def ledger(tmp_path: Path) -> Path:
    """A real persisted ledger with one unfinished experiment."""
    path = tmp_path / 'sweep.yaml'
    path.write_text('experiments:\n- experiment_id: a\n  status: running\n  current_step: 25\n  max_steps: 100\n')
    return path


def publish(ledger: Path, *, state: str = 'running', age: float = 0, run_id: str = 'first') -> dict:
    """Write a synthetic controller receipt without constructing a controller."""
    counts = {status.value: 0 for status in ExperimentStatus}
    counts['running' if state != 'finished' else 'completed'] = 1
    counts['total'] = 1
    result = {
        'schema_version': 1,
        'ledger': str(ledger),
        'run_id': run_id,
        'state': state,
        'controller': {'host': 'cpu-test', 'pid': 123},
        'updated_at': (datetime.now(timezone.utc) - timedelta(seconds=age)).isoformat(),
        'stale_after_seconds': 120,
        'counts': counts,
        'progress': {'fraction': 0.25},
        'results_finalized': state == 'finished',
        'runs': [{'experiment_id': 'a', 'current_step': 25}],
        'ledger_fingerprint': (
            {'sha256': hashlib.sha256(ledger.read_bytes()).hexdigest()} if state == 'finished' else None
        ),
    }
    path = status_path(ledger)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result))
    return result


def test_explicit_ledger_source_reuses_progress_without_claiming_finality(ledger: Path) -> None:
    result = observer.read_status(ledger, source='ledger', include_runs=True)
    assert result['progress']['percent'] == 25
    assert result['counts']['queued'] == 0
    assert result['runs'][0]['max_steps'] == 100
    assert result['results_finalized'] is False
    assert result['source'] == 'ledger'


def test_missing_controller_does_not_silently_fall_back(ledger: Path) -> None:
    with pytest.raises(FileNotFoundError, match='source=ledger explicitly'):
        observer.read_status(ledger)


def test_stale_controller_is_inspectable_but_cannot_satisfy_wait(ledger: Path) -> None:
    publish(ledger, age=121)
    result = observer.read_status(ledger)
    assert result['observation']['stale'] is True
    assert 'runs' not in result
    assert observer.wait_for_event(ledger)['event'] == 'controller_stale'


def test_finalized_receipt_survives_exit_but_not_ledger_edits(ledger: Path) -> None:
    publish(ledger, state='finished', age=86400)
    assert observer.wait_for_event(ledger)['event'] == 'finalized'
    ledger.write_text(ledger.read_text() + '\n# changed after completion\n')
    assert observer.wait_for_event(ledger)['event'] == 'ledger_changed'
    assert observer.read_status(ledger)['results_finalized'] is False


def test_replaced_receipt_during_finalization_check_is_not_ready(ledger: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    document = publish(ledger, state='finished')

    def restart(path: Path) -> dict:
        publish(path, run_id='replacement')
        return document['ledger_fingerprint']

    monkeypatch.setattr(observer, 'ledger_fingerprint', restart)
    with pytest.raises(RuntimeError, match='Controller status changed'):
        observer.read_status(ledger)


def test_changed_ledger_during_fingerprint_is_a_json_error(
    ledger: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    publish(ledger, state='finished')

    def changing(path: Path) -> dict:
        raise RuntimeError('Ledger changed while reading completion fingerprint')

    monkeypatch.setattr(observer, 'ledger_fingerprint', changing)
    with pytest.raises(SystemExit) as error:
        main(['status', str(ledger)])
    assert error.value.code == 2
    output = capsys.readouterr()
    assert output.out == ''
    assert json.loads(output.err)['error'].startswith('Ledger changed')


@pytest.mark.parametrize('state,event', [('stopped', 'controller_stopped'), ('error', 'controller_error')])
def test_interrupted_controller_is_not_complete(ledger: Path, state: str, event: str) -> None:
    publish(ledger, state=state)
    assert observer.wait_for_event(ledger)['event'] == event


def test_wait_silent_until_finalization(ledger: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    publish(ledger)
    delays = []

    def complete(delay: float) -> None:
        delays.append(delay)
        publish(ledger, state='finished')

    monkeypatch.setattr(observer.time, 'sleep', complete)
    assert observer.wait_for_event(ledger)['event'] == 'finalized'
    assert delays == [15]
    assert capsys.readouterr().out == ''


def test_restart_is_not_followed_silently(ledger: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    publish(ledger)
    monkeypatch.setattr(observer.time, 'sleep', lambda _: publish(ledger, state='finished', run_id='replacement'))
    assert observer.wait_for_event(ledger)['event'] == 'controller_restarted'


def test_training_at_one_hundred_percent_does_not_trigger_finalization(ledger: Path) -> None:
    document = publish(ledger)
    document['progress']['fraction'] = 1
    status_path(ledger).write_text(json.dumps(document))
    assert observer.read_status(ledger)['results_finalized'] is False


@pytest.mark.parametrize(
    'change',
    [
        {'schema_version': 9},
        {'ledger': '/wrong.yaml'},
        {'state': 'ready'},
        {'results_finalized': True},
        {'progress': {'fraction': float('nan')}},
        {'updated_at': '2026-01-01T00:00:00'},
        {'stale_after_seconds': -1},
    ],
)
def test_malformed_receipts_fail_clearly(ledger: Path, change: dict) -> None:
    document = publish(ledger)
    document.update(change)
    status_path(ledger).write_text(json.dumps(document))
    with pytest.raises(ValueError):
        observer.read_status(ledger)


def test_failure_wait_returns_without_whole_sweep_finishing(ledger: Path) -> None:
    document = publish(ledger)
    document['counts'].update(running=0, failed=1)
    status_path(ledger).write_text(json.dumps(document))
    assert observer.wait_for_event(ledger, until='failure')['event'] == 'failure'


def test_timeout_is_one_event_and_cli_nonzero(ledger: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    publish(ledger)
    ticks = iter([0, 2])
    monkeypatch.setattr(observer.time, 'monotonic', lambda: next(ticks))
    with pytest.raises(SystemExit) as error:
        main(['wait', str(ledger), '--timeout', '1'])
    assert error.value.code == 2
    assert json.loads(capsys.readouterr().out)['event'] == 'timeout'


@pytest.mark.parametrize('option,value', [('poll_interval', 0), ('timeout', -1), ('max_age', float('inf'))])
def test_invalid_wait_bounds(ledger: Path, option: str, value: float) -> None:
    publish(ledger)
    with pytest.raises(ValueError):
        observer.wait_for_event(ledger, **{option: value})


def test_cli_status_outputs_one_json_record(ledger: Path, capsys) -> None:
    main(['status', str(ledger), '--source', 'ledger'])
    output = capsys.readouterr().out
    assert len(output.splitlines()) == 1
    assert json.loads(output)['progress']['fraction'] == 0.25


def test_malformed_ledger_is_a_json_error(ledger: Path, capsys) -> None:
    ledger.write_text('experiments: [')
    with pytest.raises(SystemExit) as error:
        main(['status', str(ledger), '--source', 'ledger'])
    assert error.value.code == 2
    output = capsys.readouterr()
    assert output.out == ''
    assert json.loads(output.err)['error']


def test_real_publisher_completion_is_consumed_once(ledger: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    with ControllerStatusPublisher(ledger) as publisher:
        publisher.publish('running', [{'experiment_id': 'a', 'status': ExperimentStatus.RUNNING}])

        def complete(_: float) -> None:
            ledger.write_text('experiments:\n- experiment_id: a\n  status: completed\n')
            publisher.publish(
                'finished',
                [{'experiment_id': 'a', 'status': ExperimentStatus.COMPLETED}],
                finalized=True,
                expected_fingerprint=ledger_fingerprint(ledger, durable=True),
            )

        monkeypatch.setattr(observer.time, 'sleep', complete)
        main(['wait', str(ledger)])
    output = capsys.readouterr()
    assert output.err == ''
    assert len(output.out.splitlines()) == 1
    result = json.loads(output.out)
    assert result['event'] == 'finalized'
    assert result['status']['progress']['percent'] == 100
    assert result['status']['counts']['completed'] == 1


def test_cli_prepare_only_produces_an_unapproved_proposal(tmp_path: Path, capsys) -> None:
    ledger = tmp_path / 'pending.yaml'
    ledger.write_text('experiments:\n- experiment_id: a\n  status: pending\n')
    config = tmp_path / 'hpc.yaml'
    config.write_text(
        'clusters:\n  OLIVIA:\n    partition: normal\n    account: test\n    hostname: example.invalid\n    username: test\n'
    )
    main(
        [
            'prepare',
            str(ledger),
            '--cluster',
            'olivia',
            '--hpc-config-file',
            str(config),
            '--gpus-per-job',
            '1',
            '--ram-gb-per-job',
            '64',
            '--time-hours',
            '4',
            '--gpu-reason',
            'Prior batch fits one GPU.',
            '--ram-reason',
            'Measured peak plus margin.',
            '--time-reason',
            'Measured throughput plus evaluation margin.',
        ]
    )
    result = json.loads(capsys.readouterr().out)
    assert result['approval_status'] == 'awaiting_user'
    assert result['concurrency'] == 1
    assert '--headless' in result['argv']
    assert set(tmp_path.iterdir()) == {ledger, config}

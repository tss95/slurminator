"""Read-only progress queries and quiet waits; no scheduler or controller startup."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Any

from slurminator.controller_status import ledger_fingerprint, status_path
from slurminator.experiments.status_enum import ExperimentStatus
from slurminator.progress import overall_run_progress
from slurminator.result_ledger import json_value, read_ledgers

STATES = frozenset({'starting', 'running', 'finalizing', 'finished', 'stopped', 'error'})
FAILED_STATUSES = ('failed', 'cancelled', 'timeout', 'out_of_memory', 'killed')


def _positive(value: float, name: str) -> float:
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be a positive finite number')
    return value


def _timestamp(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError('Controller status requires a UTC updated_at timestamp')
    try:
        result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError as error:
        raise ValueError('Invalid controller updated_at timestamp') from error
    if result.tzinfo is None:
        raise ValueError('Controller updated_at timestamp must include a timezone')
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f'Invalid JSON numeric constant: {value}')


def _validate_snapshot(document: Any, ledger: Path) -> dict[str, Any]:
    if not isinstance(document, dict) or document.get('schema_version') != 1:
        raise ValueError('Unsupported or missing controller status schema_version')
    if document.get('ledger') != str(ledger):
        raise ValueError('Controller status belongs to a different ledger')
    if document.get('state') not in STATES:
        raise ValueError('Unknown controller state')
    if not isinstance(document.get('run_id'), str) or not document['run_id']:
        raise ValueError('Missing controller run_id')
    counts = document.get('counts')
    statuses = {status.value for status in ExperimentStatus}
    if not isinstance(counts, dict) or not statuses.issubset(counts):
        raise ValueError('Controller status must include every experiment status count')
    if any(type(value) is not int or value < 0 for value in counts.values()):
        raise ValueError('Controller counts must be nonnegative integers')
    if counts.get('total') != sum(counts[status] for status in statuses):
        raise ValueError('Controller status counts do not sum to total')
    progress = document.get('progress')
    fraction = progress.get('fraction') if isinstance(progress, dict) else None
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 <= fraction <= 1:
        raise ValueError('Controller progress fraction must be between zero and one')
    if type(document.get('results_finalized')) is not bool:
        raise ValueError('Controller status requires a boolean results_finalized')
    if document['results_finalized'] != (document['state'] == 'finished'):
        raise ValueError('Only finished controller status may declare finalized results')
    return document


def read_status(
    ledger: Path, *, source: str = 'controller', include_runs: bool = False, max_age: float | None = None
) -> dict[str, Any]:
    """Read explicit controller or ledger state, preserving freshness and readiness limits.

    A ledger query never claims finalization. Controller receipts bind finalization
    to the exact saved ledger; stale live snapshots remain inspectable but cannot
    satisfy a wait. No scheduler query, source mutation, or controller is started.
    """
    ledger = Path(ledger).resolve(strict=True)
    if max_age is not None:
        _positive(max_age, 'max_age')
    if source == 'ledger':
        loaded = read_ledgers([ledger])
        experiments = [run.data for run in loaded.runs]
        counts = {status.value: 0 for status in ExperimentStatus}
        for experiment in experiments:
            counts[experiment['status']] += 1
        counts['total'] = len(experiments)
        equivalent, eligible, fraction = overall_run_progress(experiments)
        result = {
            'schema_version': 1,
            'source': 'ledger',
            'ledger': str(ledger),
            'state': 'unverified',
            'counts': counts,
            'progress': {
                'fraction': fraction,
                'percent': fraction * 100,
                'completed_equivalent_runs': equivalent,
                'eligible_runs': eligible,
                'excluded_failed_runs': len(experiments) - eligible,
            },
            'results_finalized': False,
            'sources': loaded.sources,
            'warnings': loaded.warnings,
        }
        if include_runs:
            fields = (
                'experiment_id',
                'status',
                'current_step',
                'max_steps',
                'current_epoch',
                'max_epochs',
                'last_metrics_update',
            )
            result['runs'] = [{field: run.data.get(field) for field in fields} for run in loaded.runs]
        return json_value(result)
    if source != 'controller':
        raise ValueError('source must be controller or ledger')
    path = status_path(ledger)
    try:
        receipt = path.read_text()
        result = _validate_snapshot(json.loads(receipt, parse_constant=_reject_constant), ledger)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f'No controller status at {path}; start an approved headless controller, '
            'or query source=ledger explicitly for a persisted snapshot'
        ) from error
    now = datetime.now(timezone.utc)
    age = (now - _timestamp(result.get('updated_at'))).total_seconds()
    if age < -5:
        raise ValueError('Controller timestamp is in the future; check node clock synchronization')
    threshold = max_age if max_age is not None else result.get('stale_after_seconds')
    if not isinstance(threshold, (int, float)):
        raise ValueError('Controller status lacks a stale_after_seconds bound')
    _positive(threshold, 'stale_after_seconds')
    ended = result['state'] in {'finished', 'stopped', 'error'}
    result['source'] = 'controller'
    result['observation'] = {
        'read_at': now.isoformat(),
        'age_seconds': max(0.0, age),
        'stale': not ended and age > threshold,
        'ledger_matches_finalized_snapshot': None,
    }
    if result['state'] == 'finished':
        fingerprint = result.get('ledger_fingerprint')
        if not isinstance(fingerprint, dict) or not isinstance(fingerprint.get('sha256'), str):
            raise ValueError('Finalized controller status lacks its ledger fingerprint')
        matches = ledger_fingerprint(ledger)['sha256'] == fingerprint['sha256']
        if path.read_text() != receipt:
            raise RuntimeError('Controller status changed while checking completion; read the current status again')
        result['observation']['ledger_matches_finalized_snapshot'] = matches
        # A completed record is historical evidence, not readiness for a new edit.
        result['results_finalized'] = matches
    if not include_runs:
        result.pop('runs', None)
    return result


def wait_for_event(
    ledger: Path,
    *,
    until: str = 'finalized',
    timeout: float | None = None,
    poll_interval: float = 15,
    max_age: float | None = None,
) -> dict[str, Any]:
    """Wait silently in ordinary code and return one event with the latest status.

    ``until='failure'`` returns at the first observed unsuccessful terminal run,
    or at finalization if none fail. Controller interruption, stale state, restart,
    and obsolete completion receipts return distinct non-success events. A single
    watcher is bound to one controller run; it never follows a replacement silently.
    """
    if until not in {'finalized', 'failure'}:
        raise ValueError('until must be finalized or failure')
    _positive(poll_interval, 'poll_interval')
    if timeout is not None:
        _positive(timeout, 'timeout')
    start = time.monotonic()
    run_id = None
    while True:
        status = read_status(ledger, max_age=max_age)
        event = None
        if run_id is not None and status['run_id'] != run_id:
            event = 'controller_restarted'
        elif status['observation']['stale']:
            event = 'controller_stale'
        elif status['state'] in {'stopped', 'error'}:
            event = f"controller_{status['state']}"
        elif status['state'] == 'finished' and not status['results_finalized']:
            event = 'ledger_changed'
        elif until == 'failure' and any(status['counts'][key] for key in FAILED_STATUSES):
            event = 'failure'
        elif status['results_finalized']:
            event = 'finalized'
        elif timeout is not None and time.monotonic() - start >= timeout:
            event = 'timeout'
        if event is not None:
            return {'schema_version': 1, 'event': event, 'status': status}
        run_id = status['run_id']
        delay = poll_interval
        if timeout is not None:
            delay = min(delay, max(0, timeout - (time.monotonic() - start)))
        time.sleep(delay)

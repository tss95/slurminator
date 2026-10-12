"""Read-only ledger queries, with explicit metric sources and seed coverage.

This module does not instantiate an orchestrator, replay journals, contact a
scheduler, load checkpoints, or modify a ledger.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timezone
from enum import Enum
from fnmatch import fnmatchcase
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Sequence

import yaml

from slurminator.cli.override_parser import parse_override_list
from slurminator.experiments.experiment_record import coerce_experiment_status
from slurminator.experiments.yaml_utils import ExperimentYAMLLoader


class LedgerLoader(ExperimentYAMLLoader):
    """Use production tags/types while rejecting ambiguous duplicate YAML keys."""

    def construct_mapping(self, node: yaml.Node, deep: bool = False) -> dict:
        seen = set()
        for key_node, _ in node.value:
            if key_node.tag == 'tag:yaml.org,2002:merge':
                continue
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, (str, int, float, bool, type(None))):
                raise ValueError('Ledger mapping keys must be scalar values')
            if key in seen:
                raise ValueError(f'Duplicate YAML key {key!r} at line {key_node.start_mark.line + 1}')
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def json_value(value: Any) -> Any:
    """Convert YAML scalars to strict JSON, retaining null for non-finite data."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    return value


def numeric(value: Any) -> float | None:
    """Accept finite numeric scalars, including quoted numbers, but not booleans."""
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        result = float(value)
    except (ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _mapping(value: Any, label: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f'{label} must be a mapping')
    return value


@dataclass
class Run:
    """One ledger row; its source path and metric surface remain identifiable."""

    ledger: str
    data: dict[str, Any]
    params: dict[str, Any]
    metrics: dict[str, Any]
    metric_source: str
    history_count: int

    @property
    def experiment_id(self) -> str:
        """Return the source row's stable identifier."""
        return self.data['experiment_id']

    def field(self, selector: str) -> Any:
        """Resolve raw dotted fields or exact ``param:`` / ``metric:`` keys."""
        if selector == 'ledger':
            return self.ledger
        if selector == 'metric_source':
            return self.metric_source
        if selector == 'history_count':
            return self.history_count
        if selector == 'seed':
            candidates = [self.data.get('seed'), self.data.get('metadata', {}).get('seed'), self.params.get('seed')]
            present = [value for value in candidates if value is not None]
            if len({str(value) for value in present}) > 1:
                raise ValueError(f'{self.experiment_id}: conflicting seed values in row, metadata, or sweep_params')
            return present[0] if present else None
        if selector.startswith('param:'):
            return self.params.get(selector[6:])
        if selector.startswith('metric:'):
            return self.metrics.get(selector[7:])
        value: Any = self.data
        for part in selector.split('.'):
            if not isinstance(value, dict):
                return None
            value = value.get(part)
        return value


@dataclass
class LedgerSet:
    """Loaded rows and provenance for the exact files that were read."""

    runs: list[Run]
    sources: list[dict[str, Any]]
    warnings: list[str]


def read_ledgers(paths: Sequence[Path], *, include_history: bool = False) -> LedgerSet:
    """Read each file once using the native safe loader; fail on malformed rows."""
    runs: list[Run] = []
    sources: list[dict[str, Any]] = []
    warnings = ['Ledger snapshots only: scheduler state and saved artifacts have not been verified.']
    seen_paths: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen_paths:
            raise ValueError(f'Ledger supplied more than once: {path}')
        seen_paths.add(resolved)
        with path.open('rb') as handle:
            payload = handle.read()
            stat = os.fstat(handle.fileno())
        document = _mapping(yaml.load(payload, Loader=LedgerLoader), str(path))
        if not isinstance(document.get('experiments'), list):
            raise ValueError(f'{path}: expected an experiments list; a sweepfile is not a result ledger')
        metadata = _mapping(document.get('metadata', {}), f'{path}: metadata')
        sources.append(
            {
                'path': str(resolved),
                'bytes': len(payload),
                'mtime_utc': datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                'sha256': hashlib.sha256(payload).hexdigest(),
                'experiment_name': metadata.get('experiment_name'),
                'experiment_description': metadata.get('experiment_description'),
                'runs': len(document['experiments']),
            }
        )
        for folder in ('_submission_receipts', '_scheduler_lifecycle'):
            journal = path.parent / '.orchestrator_status' / folder / f'{path.name}.jsonl'
            if journal.exists():
                warnings.append(f'Journal not replayed; it may contain newer state: {journal}')
        seen_ids: set[str] = set()
        for index, data in enumerate(document['experiments']):
            data = _mapping(data, f'{path}: row {index}')
            identity = data.get('experiment_id')
            if not isinstance(identity, str) or not identity.strip() or identity in seen_ids:
                raise ValueError(f'{path}: missing or duplicate experiment_id at row {index}: {identity!r}')
            seen_ids.add(identity)
            data['status'] = coerce_experiment_status(data.get('status', 'pending')).value
            data['metadata'] = _mapping(data.get('metadata', {}), f'{identity}: metadata')
            raw_params = data.get('sweep_params') or ''
            if not isinstance(raw_params, (str, dict)):
                raise ValueError(f'{identity}: sweep_params must be a string or mapping')
            params = parse_override_list(raw_params) if isinstance(raw_params, str) else raw_params
            surface = 'all_metrics' if 'all_metrics' in data else ('metrics' if 'metrics' in data else 'absent')
            metrics = _mapping(data.get(surface, {}), f'{identity}: {surface}')
            if any(not isinstance(key, str) for key in metrics):
                raise ValueError(f'{identity}: metric names must be strings')
            history = data.get('history', [])
            if not isinstance(history, list):
                raise ValueError(f'{identity}: history must be a list')
            if not include_history:
                data.pop('history', None)
            runs.append(Run(str(resolved), data, params, metrics, surface, len(history)))
    return LedgerSet(runs, sources, warnings)


def metric_keys(runs: Sequence[Run], patterns: Sequence[str], *, history: bool = False) -> list[str]:
    """Resolve exact keys or glob patterns; reject a selector that matches no key."""
    available = set()
    for run in runs:
        surfaces = [run.metrics]
        if history:
            surfaces = []
            for point in run.data.get('history', []):
                point = _mapping(point, f'{run.experiment_id}: history point')
                surfaces.append(_mapping(point.get('metrics', {}), f'{run.experiment_id}: history metrics'))
        for surface in surfaces:
            if any(not isinstance(key, str) for key in surface):
                raise ValueError(f'{run.experiment_id}: metric names must be strings')
            available.update(surface)
    selected: set[str] = set()
    for pattern in patterns:
        matches = {pattern} if pattern in available else {key for key in available if fnmatchcase(key, pattern)}
        if not matches:
            raise ValueError(f'Metric selector {pattern!r} matched no keys; inspect the ledger first')
        selected.update(matches)
    return sorted(selected if patterns else available)


def filter_runs(runs: Sequence[Run], statuses: Sequence[str], conditions: Sequence[str]) -> list[Run]:
    """Filter snapshots by status and typed FIELD=VALUE equality conditions."""
    parsed = []
    for condition in conditions:
        field, separator, raw = condition.partition('=')
        if not separator or not field:
            raise ValueError(f'Filter must be FIELD=VALUE: {condition!r}')
        if not any(run.field(field) is not None for run in runs):
            raise ValueError(f'Filter field is absent in every row: {field!r}')
        parsed.append((field, yaml.safe_load(raw)))
    allowed = {coerce_experiment_status(status).value for status in statuses}
    return [
        run
        for run in runs
        if (not allowed or run.data['status'] in allowed)
        and all(json_value(run.field(field)) == expected for field, expected in parsed)
    ]


def inspect_runs(
    runs: Sequence[Run], keys: Sequence[str], *, limit: int, parameter_patterns: Sequence[str] = ()
) -> dict[str, Any]:
    """Describe field/metric coverage and varying overrides without dumping rows."""
    coverage = [
        {
            'key': key,
            'present': sum(key in run.metrics for run in runs),
            'numeric': sum(numeric(run.metrics.get(key)) is not None for run in runs),
        }
        for key in keys
    ]
    parameters = []
    parameter_keys = {key for run in runs for key in run.params}
    for pattern in parameter_patterns:
        if not any(key == pattern or fnmatchcase(key, pattern) for key in parameter_keys):
            raise ValueError(f'Parameter selector {pattern!r} matched no keys')
    for key in sorted(parameter_keys):
        if parameter_patterns and not any(key == p or fnmatchcase(key, p) for p in parameter_patterns):
            continue
        values = []
        seen = set()
        for run in runs:
            value = json_value(run.params.get(key))
            identity = json.dumps(value, sort_keys=True, allow_nan=False)
            if identity not in seen:
                seen.add(identity)
                values.append(value)
        if len(values) > 1 or parameter_patterns:
            parameters.append({'field': f'param:{key}', 'distinct': len(values), 'sample': values[:5]})
    return {
        'statuses': dict(sorted(Counter(run.data['status'] for run in runs).items())),
        'datasets': dict(sorted(Counter(str(run.data.get('dataset_name')) for run in runs).items())),
        'metric_sources': dict(Counter(run.metric_source for run in runs)),
        'runs_without_metrics': sum(not run.metrics for run in runs),
        'runs_with_history': sum(run.history_count > 0 for run in runs),
        'runs_with_truncated_history': sum(bool(run.data.get('history_truncated')) for run in runs),
        'available_fields': sorted({key for run in runs for key in run.data if key not in ('all_metrics', 'metrics')}),
        'metric_keys_total': len(coverage),
        'metric_keys': coverage[:limit] if limit else coverage,
        'parameters_selection': list(parameter_patterns) or 'varying overrides',
        'parameters_total': len(parameters),
        'parameters': parameters[:limit] if limit else parameters,
    }


def summarize_runs(
    runs: Sequence[Run],
    keys: Sequence[str],
    group_by: Sequence[str],
    *,
    seed_field: str = 'seed',
    expected_seeds: Sequence[str] = (),
    scale: float = 1.0,
) -> list[dict[str, Any]]:
    """Report one independent value per seed, with missing/invalid coverage."""
    groups: dict[tuple, list[Run]] = {}
    for run in runs:
        values = tuple(json_value(run.field(field)) for field in group_by)
        if any(value is None or isinstance(value, (list, dict)) for value in values):
            raise ValueError(f'{run.experiment_id}: group fields must be present scalars: {group_by}')
        groups.setdefault(values, []).append(run)
    result = []
    for group, members in sorted(groups.items(), key=lambda item: str(item[0])):
        seeds: dict[str, Run] = {}
        for run in members:
            seed = run.field(seed_field)
            if seed is None or isinstance(seed, (bool, list, dict)):
                raise ValueError(f'{run.experiment_id}: missing/scalar seed required ({seed_field})')
            label = str(seed)
            if label in seeds:
                raise ValueError(
                    f'Duplicate seed {label} in group {group}: {seeds[label].experiment_id}, {run.experiment_id}. '
                    'Add grouping fields or explicitly select the intended retry; attempts are not extra seeds.'
                )
            seeds[label] = run
        for key in keys:
            observations = {seed: numeric(run.metrics.get(key)) for seed, run in seeds.items()}
            valid = {seed: value * scale for seed, value in observations.items() if value is not None}
            if any(not math.isfinite(value) for value in valid.values()):
                raise ValueError(f'Metric {key!r} overflows with scale {scale}')
            values = list(valid.values())
            result.append(
                {
                    **dict(zip(group_by, group)),
                    'metric': key,
                    'scale': scale,
                    'runs': len(members),
                    'n': len(values),
                    'missing_metric_seeds': [seed for seed, run in seeds.items() if key not in run.metrics],
                    'invalid_metric_seeds': [
                        seed for seed, run in seeds.items() if key in run.metrics and observations[seed] is None
                    ],
                    'missing_seeds': sorted(set(expected_seeds) - seeds.keys()),
                    'unexpected_seeds': sorted(seeds.keys() - set(expected_seeds)) if expected_seeds else [],
                    'mean': mean(values) if values else None,
                    'sample_sd': stdev(values) if len(values) > 1 else None,
                    'min': min(values) if values else None,
                    'max': max(values) if values else None,
                    'per_seed': valid,
                }
            )
    return result


def history_rows(
    runs: Sequence[Run], keys: Sequence[str], *, mode: str, direction: str | None, attempt: int | None = None
) -> list[dict[str, Any]]:
    """Select observed history within one attempt, retaining coordinates and gaps.

    ``last`` means last finite observation, not a certified final endpoint.
    ``best`` selects each metric independently, with earliest-coordinate ties.
    """
    result = []
    for run in runs:
        points = run.data.get('history', [])
        attempts = []
        for point in points:
            _mapping(point, f'{run.experiment_id}: history point')
            value = point.get('attempt', 1)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f'{run.experiment_id}: history attempt must be a positive integer')
            attempts.append(value)
        recorded_attempt = run.data.get('history_attempt_max', 1)
        if not isinstance(recorded_attempt, int) or isinstance(recorded_attempt, bool) or recorded_attempt < 1:
            raise ValueError(f'{run.experiment_id}: history_attempt_max must be a positive integer')
        selected_attempt = attempt if attempt is not None else max([recorded_attempt, *attempts])
        selected = [point for point, number in zip(points, attempts) if number == selected_attempt]
        for key in keys:
            observed = [point for point in selected if numeric(point.get('metrics', {}).get(key)) is not None]

            def coordinates(point: dict) -> tuple:
                return tuple(
                    value if (value := numeric(point.get(field))) is not None else -1
                    for field in ('step', 'epoch', 'timestamp')
                )

            observed.sort(key=coordinates)
            chosen = observed
            if mode == 'last':
                chosen = observed[-1:]
            elif mode == 'best':
                if direction not in ('min', 'max'):
                    raise ValueError('Best history selection requires --direction min or max')
                choose = max if direction == 'max' else min
                chosen = [choose(observed, key=lambda p: numeric(p['metrics'][key]))] if observed else []
            for point in chosen or [None]:
                result.append(
                    {
                        'ledger': run.ledger,
                        'experiment_id': run.experiment_id,
                        'status': run.data['status'],
                        'seed': run.field('seed'),
                        'metric': key,
                        'selection': mode,
                        'attempt': selected_attempt,
                        'observations': len(observed),
                        'history_truncated': bool(run.data.get('history_truncated')),
                        'value': numeric(point['metrics'][key]) if point else None,
                        **{field: point.get(field) if point else None for field in ('step', 'epoch', 'timestamp')},
                    }
                )
    return result

"""Compact read-only experiment reports: ``python -m slurminator.results``."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import yaml

from slurminator.result_ledger import (
    filter_runs,
    history_rows,
    inspect_runs,
    json_value,
    metric_keys,
    numeric,
    read_ledgers,
    summarize_runs,
)

DEFAULT_FIELDS = ('ledger', 'experiment_id', 'status', 'seed', 'job_id', 'current_step', 'max_steps')


def build_parser() -> argparse.ArgumentParser:
    """Build an analysis-only CLI, separate from controller launch arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name, help_text in (
        ('inspect', 'List status, metric coverage, and varying parameters'),
        ('rows', 'Export selected fields and exact metric names'),
        ('summarize', 'Aggregate independent seeds, retaining coverage and dispersion'),
        ('history', 'Query observed history within one attempt'),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument('ledgers', nargs='+', type=Path)
        command.add_argument(
            '--metric', action='append', default=[], help='Exact metric key or quoted glob; repeatable'
        )
        command.add_argument('--status', action='append', default=[], help='Repeatable status filter')
        command.add_argument('--where', action='append', default=[], metavar='FIELD=VALUE')
        command.add_argument(
            '--limit', type=int, default=20, help='Output rows/catalog entries; 0 means all (default: 20)'
        )
        command.add_argument('--format', choices=['json', 'markdown', 'csv'], default='json')
        command.add_argument(
            '--output', type=Path, help='Write the report instead of stdout; cannot overwrite an input'
        )
        if name == 'inspect':
            command.add_argument(
                '--param', action='append', default=[], help='Override key/glob; includes constant values'
            )
        if name == 'rows':
            command.add_argument(
                '--field', action='append', default=[], help='Raw dotted field, param:KEY, or metric:KEY'
            )
        if name == 'summarize':
            command.add_argument('--group-by', action='append', required=True)
            command.add_argument('--seed-field', default='seed')
            command.add_argument('--expect-seeds', help='Comma-separated expected seed labels')
            command.add_argument(
                '--scale', type=float, default=1.0, help='Explicit metric scaling, e.g. 100 for percent'
            )
            command.add_argument(
                '--require-complete', action='store_true', help='Require expected seeds and finite metrics'
            )
            command.add_argument(
                '--include-incomplete', action='store_true', help='Include noncompleted runs explicitly'
            )
        if name == 'history':
            command.add_argument('--mode', choices=['all', 'last', 'best'], default='last')
            command.add_argument(
                '--direction', choices=['min', 'max'], help='Required for best; never inferred from a name'
            )
            command.add_argument('--attempt', type=int, help='Attempt number; defaults to latest recorded attempt')
    return parser


def make_report(args: argparse.Namespace) -> dict[str, Any]:
    """Execute a query and return a strict-JSON-compatible report with provenance."""
    if args.limit < 0:
        raise ValueError('--limit must be nonnegative')
    if args.command in ('summarize', 'history') and not args.metric:
        raise ValueError(f'{args.command} requires at least one --metric selector')
    if args.command == 'history':
        if args.mode == 'best' and not args.direction:
            raise ValueError('--mode best requires --direction min or max')
        if args.attempt is not None and args.attempt < 1:
            raise ValueError('--attempt must be positive')
    if args.command == 'summarize':
        if not math.isfinite(args.scale) or args.scale <= 0:
            raise ValueError('--scale must be finite and positive')
        if args.require_complete and not args.expect_seeds:
            raise ValueError('--require-complete requires --expect-seeds')
        if args.status and set(args.status) != {'completed'} and not args.include_incomplete:
            raise ValueError('Summarizing noncompleted runs requires --include-incomplete')
    ledgers = read_ledgers(args.ledgers, include_history=args.command == 'history')
    statuses = args.status
    if args.command == 'summarize' and not statuses and not args.include_incomplete:
        statuses = ['completed']
    runs = filter_runs(ledgers.runs, statuses, args.where)
    if not runs and ledgers.runs:
        raise ValueError('No runs matched the selection; inspect status and filter fields')
    report: dict[str, Any] = {
        'schema_version': 1,
        'command': args.command,
        'sources': ledgers.sources,
        'source_runs': len(ledgers.runs),
        'source_statuses': dict(sorted(Counter(run.data['status'] for run in ledgers.runs).items())),
        'selected_runs': len(runs),
        'selection': {'status': statuses, 'where': args.where},
        'warnings': ledgers.warnings,
    }
    keys = (
        metric_keys(runs, args.metric, history=args.command == 'history')
        if args.metric or args.command == 'inspect'
        else []
    )
    if args.command == 'inspect':
        report['inspection'] = inspect_runs(runs, keys, limit=args.limit, parameter_patterns=args.param)
        return json_value(report)
    if args.command == 'rows':
        fields = list(dict.fromkeys([*DEFAULT_FIELDS, *args.field]))
        for field in args.field:
            if not any(run.field(field) is not None for run in runs):
                raise ValueError(f'Field is absent/null in every selected row: {field!r}')
        report['rows'] = [
            {
                **{field: run.field(field) for field in fields},
                'metric_source': run.metric_source,
                **{f'metric:{key}': run.metrics.get(key) for key in keys},
                'invalid_metrics': [key for key in keys if key in run.metrics and numeric(run.metrics[key]) is None],
                'missing_metrics': [key for key in keys if key not in run.metrics],
            }
            for run in runs
        ]
    elif args.command == 'summarize':
        expected = [seed.strip() for seed in args.expect_seeds.split(',')] if args.expect_seeds else []
        if len(set(expected)) != len(expected) or any(not seed for seed in expected):
            raise ValueError('--expect-seeds requires distinct nonempty seed labels')
        report['rows'] = summarize_runs(
            runs, keys, args.group_by, seed_field=args.seed_field, expected_seeds=expected, scale=args.scale
        )
        report['warnings'].append('Metrics retain their recorded semantics; global_best is not a final score.')
        if args.require_complete and any(
            row[field]
            for row in report['rows']
            for field in ('missing_seeds', 'unexpected_seeds', 'missing_metric_seeds', 'invalid_metric_seeds')
        ):
            raise ValueError('Incomplete seed/metric coverage; rerun without --require-complete to inspect the gaps')
    else:
        report['rows'] = history_rows(runs, keys, mode=args.mode, direction=args.direction, attempt=args.attempt)
        report['warnings'].append(
            'History selections are observed values within one attempt, not certified final endpoints. '
            'Best is selected separately for each metric; companion metrics may come from different checkpoints.'
        )
    report['rows_total'] = len(report['rows'])
    if args.limit:
        report['rows'] = report['rows'][: args.limit]
    report['rows_omitted'] = report['rows_total'] - len(report['rows'])
    return json_value(report)


def _cell(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    return str(value)


def render_report(report: dict[str, Any], output_format: str) -> str:
    """Render JSON, a compact Markdown table, or rows-only CSV."""
    if output_format == 'json':
        return json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + '\n'
    rows = report.get('rows', [])
    if output_format == 'csv':
        if report['command'] == 'inspect':
            raise ValueError('inspect supports json or markdown; use rows for CSV exports')
        stream = io.StringIO()
        columns = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: _cell(value) for key, value in row.items()} for row in rows)
        return stream.getvalue()
    lines = [
        f"Selected {report['selected_runs']} / {report['source_runs']} runs. "
        f"Source statuses: {_cell(report['source_statuses'])}.",
        '',
    ]
    for source in report['sources']:
        lines.append(f"Source: {source['path']} (modified {source['mtime_utc']}; SHA-256 {source['sha256']}).")
    lines.extend(['', *[f"Note: {warning}" for warning in report['warnings']], ''])
    if report['command'] == 'inspect':
        lines.extend(['```json', json.dumps(report['inspection'], ensure_ascii=False, indent=2), '```'])
    elif rows:
        columns = list(dict.fromkeys(key for row in rows for key in row))
        lines.extend(['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join('---' for _ in columns) + ' |'])
        for row in rows:
            lines.append(
                '| '
                + ' | '.join(_cell(row.get(key)).replace('|', '\\|').replace('\n', '<br>') for key in columns)
                + ' |'
            )
        lines.extend(['', f"Shown {len(rows)} / {report['rows_total']} rows; {report['rows_omitted']} omitted."])
    return '\n'.join(lines) + '\n'


def main(argv: Sequence[str] | None = None) -> None:
    """Run read-only analysis; bad input exits nonzero without a partial report."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.output:
            for path in args.ledgers:
                if args.output.resolve() == path.resolve() or (args.output.exists() and args.output.samefile(path)):
                    raise ValueError('--output cannot overwrite an input ledger')
        report = make_report(args)
        rendered = render_report(report, args.format)
        if args.format == 'csv':
            print(json.dumps({key: value for key, value in report.items() if key != 'rows'}), file=sys.stderr)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding='utf-8')
        else:
            print(rendered, end='')
    except (OSError, ValueError, KeyError, yaml.YAMLError) as exc:
        parser.exit(2, f'slurminator-results: {exc}\n')


if __name__ == '__main__':
    main()

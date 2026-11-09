"""Hermetic checks for ledger analysis; no scheduler, API or training artifacts."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from slurminator.result_ledger import numeric, read_ledgers
from slurminator.results import build_parser, main, make_report, render_report

pytestmark = pytest.mark.unit


def write_ledger(tmp_path: Path, rows: list[dict], name: str = 'runs.yaml') -> Path:
    """Write a synthetic ledger with ordinary YAML scalar types."""
    path = tmp_path / name
    path.write_text(yaml.safe_dump({'metadata': {'experiment_name': 'Synthetic study'}, 'experiments': rows}))
    return path


def row(identity: str, seed: int, score: object = 0.0, **extra: object) -> dict:
    """Build one completed, seeded synthetic result."""
    return {
        'experiment_id': identity,
        'dataset_name': 'toy',
        'status': 'completed',
        'metadata': {'seed': seed},
        'sweep_params': 'model.name=encoder;loss.weight=1.5;labels=[0.01, 0.05];mode=none;resume=None',
        'all_metrics': {'probe/lp0.010/test_acc': score},
        **extra,
    }


def query(command: str, path: Path, *options: str) -> dict:
    """Execute the same argument parsing and report path as the CLI."""
    return make_report(build_parser().parse_args([command, str(path), *options]))


def test_native_tags_types_aliases_and_overrides(tmp_path: Path) -> None:
    path = tmp_path / 'tagged.yaml'
    path.write_text(
        'metadata: &meta {seed: 42}\nexperiments:\n'
        '- experiment_id: a\n  metadata: *meta\n  status: !ExperimentStatus completed\n'
        '  hpc_assignment: !HPCType OLIVIA\n'
        '  sweep_params: "model.name=encoder;labels=[0.01, 0.05];flag=False;value=None;mode=none"\n'
        '  all_metrics: {score: 0.0}\n'
    )
    data = read_ledgers([path])
    (run,) = data.runs
    assert run.field('status') == 'completed' and run.field('seed') == 42
    assert run.field('param:labels') == [0.01, 0.05]
    assert run.field('param:flag') is False and run.field('param:value') is None
    assert run.field('param:mode') == 'none'
    assert run.field('metric:score') == 0.0
    assert data.sources[0]['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    'text',
    [
        'custom_sweeps: []',
        'experiments: {}',
        'experiments: [not_a_mapping]',
        'experiments: [{}]',
        'experiments: [{experiment_id: a}, {experiment_id: a}]',
        'experiments: [{experiment_id: a, status: invalid}]',
        'experiments: [{experiment_id: a, status: pending, status: completed}]',
        'experiments: [{experiment_id: a, all_metrics: []}]',
        'experiments: [{experiment_id: a, history: {}}]',
        'experiments: [{experiment_id: a, sweep_params: "invalid"}]',
        'experiments: !Unknown []',
        'experiments: !!python/object/apply:os.system ["touch SHOULD_NOT_EXIST"]',
        'experiments: [',
    ],
)
def test_invalid_or_unsafe_yaml_fails_without_writes(tmp_path: Path, text: str) -> None:
    path = tmp_path / 'invalid.yaml'
    path.write_text(text)
    before = path.read_bytes()
    with pytest.raises((ValueError, KeyError, yaml.YAMLError)):
        read_ledgers([path])
    assert path.read_bytes() == before and list(tmp_path.iterdir()) == [path]


def test_repeated_paths_and_symlinks_are_not_counted_twice(tmp_path: Path) -> None:
    path = write_ledger(tmp_path, [row('a', 42)])
    alias = tmp_path / 'alias.yaml'
    alias.symlink_to(path)
    with pytest.raises(ValueError, match='more than once'):
        read_ledgers([path, alias])


def test_conflicting_seed_sources_are_not_silently_grouped(tmp_path: Path) -> None:
    path = write_ledger(tmp_path, [row('a', 42) | {'seed': 7}])
    with pytest.raises(ValueError, match='conflicting seed'):
        query('rows', path)


def test_snapshot_discloses_pending_journals_without_replaying(tmp_path: Path) -> None:
    path = write_ledger(tmp_path, [row('a', 42, status='running')])
    journal = tmp_path / '.orchestrator_status/_scheduler_lifecycle/runs.yaml.jsonl'
    journal.parent.mkdir(parents=True)
    journal.write_text('deliberately not loaded')
    report = query('inspect', path)
    assert report['source_statuses'] == {'running': 1}
    assert any(str(journal) in warning for warning in report['warnings'])


@pytest.mark.parametrize('value', [None, True, False, float('nan'), float('inf'), '-inf', 'bad', {}, []])
def test_missing_nonfinite_and_boolean_metrics_are_not_scores(value: object) -> None:
    assert numeric(value) is None


def test_metric_surface_coverage_and_compact_inspection(tmp_path: Path) -> None:
    path = write_ledger(
        tmp_path,
        [row('a', 42, 0.0, metrics={'incorrect': 9}), row('b', 7, all_metrics={}), row('c', 123, float('nan'))],
    )
    report = query('inspect', path)
    assert report['inspection']['metric_keys'] == [{'key': 'probe/lp0.010/test_acc', 'present': 2, 'numeric': 1}]
    assert report['inspection']['runs_without_metrics'] == 1
    assert 'incorrect' not in json.dumps(report)
    json.dumps(report, allow_nan=False)


def test_parameter_discovery_can_select_constant_fields(tmp_path: Path) -> None:
    path = write_ledger(tmp_path, [row('a', 42)])
    report = query('inspect', path, '--param', 'model.*')
    assert report['inspection']['parameters'] == [{'field': 'param:model.name', 'distinct': 1, 'sample': ['encoder']}]
    with pytest.raises(ValueError, match='Parameter selector'):
        query('inspect', path, '--param', 'typo')


def test_rows_filters_globs_and_output_limits(tmp_path: Path) -> None:
    path = write_ledger(tmp_path, [row('a', 42), row('b', 7), row('c', 123, status='running')])
    report = query('rows', path, '--metric', 'probe/*/test_acc', '--status', 'completed', '--limit', '1')
    assert report['source_runs'] == 3 and report['selected_runs'] == 2
    assert len(report['rows']) == 1 and report['rows_omitted'] == 1
    assert report['rows'][0]['metric:probe/lp0.010/test_acc'] == 0.0
    filtered = query('rows', path, '--where', 'param:loss.weight=1.5', '--where', 'seed=7')
    assert [run['experiment_id'] for run in filtered['rows']] == ['b']
    for options in [('--metric', 'typo'), ('--where', 'typo=1'), ('--field', 'typo'), ('--where', 'invalid')]:
        with pytest.raises(ValueError):
            query('rows', path, *options)


def test_seed_summary_reports_dispersion_and_all_coverage(tmp_path: Path) -> None:
    path = write_ledger(
        tmp_path,
        [
            row('a', 42, 0.1),
            row('b', 7, 0.3),
            row('c', 123, all_metrics={}),
            row('d', 9, float('nan')),
            row('e', 99, status='running'),
        ],
    )
    options = (
        '--metric',
        'probe/*/test_acc',
        '--group-by',
        'dataset_name',
        '--expect-seeds',
        '42,7,123,8',
        '--scale',
        '100',
    )
    report = query('summarize', path, *options)
    (result,) = report['rows']
    assert report['selected_runs'] == 4 and report['source_statuses'] == {'completed': 4, 'running': 1}
    assert result['mean'] == pytest.approx(20) and result['sample_sd'] == pytest.approx(14.1421356237)
    assert result['per_seed'] == {'42': 10, '7': 30}
    assert result['missing_metric_seeds'] == ['123'] and result['invalid_metric_seeds'] == ['9']
    assert result['missing_seeds'] == ['8'] and result['unexpected_seeds'] == ['9']
    with pytest.raises(ValueError, match='Incomplete'):
        query('summarize', path, *options, '--require-complete')


def test_summary_preserves_global_best_semantics_and_missing_sample_sd(tmp_path: Path) -> None:
    path = write_ledger(tmp_path, [row('a', 42, all_metrics={'probe/global_best_acc': 0.9, 'probe/final_acc': 0.6})])
    report = query('summarize', path, '--metric', 'probe/*acc', '--group-by', 'dataset_name')
    assert {r['metric']: r['mean'] for r in report['rows']} == {'probe/final_acc': 0.6, 'probe/global_best_acc': 0.9}
    assert all(r['sample_sd'] is None for r in report['rows'])
    assert any('not a final' in warning for warning in report['warnings'])


def test_duplicate_seeds_across_ledgers_require_explicit_selection(tmp_path: Path) -> None:
    first = write_ledger(tmp_path, [row('original', 42)])
    second = write_ledger(tmp_path, [row('retry', 42)], 'retry.yaml')
    options = ['summarize', str(first), str(second), '--metric', '*acc', '--group-by', 'dataset_name']
    with pytest.raises(ValueError, match='Duplicate seed'):
        make_report(build_parser().parse_args(options))
    report = make_report(build_parser().parse_args([*options, '--where', 'experiment_id=retry']))
    assert report['selected_runs'] == 1


@pytest.mark.parametrize(
    'options',
    [
        ('--scale', 'nan'),
        ('--scale', '0'),
        ('--status', 'running'),
        ('--expect-seeds', '42,42'),
        ('--require-complete',),
    ],
)
def test_invalid_summary_options_fail(tmp_path: Path, options: tuple[str, ...]) -> None:
    path = write_ledger(tmp_path, [row('a', 42)])
    with pytest.raises(ValueError):
        query('summarize', path, '--metric', '*acc', '--group-by', 'dataset_name', *options)


def test_history_latest_attempt_ties_gaps_and_last_observed(tmp_path: Path) -> None:
    points = [
        {'attempt': 1, 'epoch': 50, 'step': 500, 'metrics': {'score': 1.0}},
        {'attempt': 2, 'epoch': 1, 'step': 10, 'metrics': {'score': 0.9}},
        {'attempt': 2, 'epoch': 2, 'step': 20, 'metrics': {'score': 0.9}},
        {'attempt': 2, 'epoch': 3, 'step': 30, 'metrics': {'score': 0.5}},
        {'attempt': 2, 'epoch': 4, 'step': 40, 'metrics': {'score': float('nan')}},
    ]
    path = write_ledger(tmp_path, [row('a', 42, history=points, history_attempt_max=2, history_truncated=True)])
    best = query('history', path, '--metric', 'score', '--mode', 'best', '--direction', 'max')['rows'][0]
    assert best['value'] == 0.9 and best['epoch'] == 1 and best['attempt'] == 2
    assert best['observations'] == 3 and best['history_truncated']
    last = query('history', path, '--metric', 'score')['rows'][0]
    assert last['value'] == 0.5 and last['epoch'] == 3 and last['selection'] == 'last'
    original = query('history', path, '--metric', 'score', '--attempt', '1')['rows'][0]
    assert original['value'] == 1.0
    with pytest.raises(ValueError, match='direction'):
        query('history', path, '--metric', 'score', '--mode', 'best')
    assert 'history' not in read_ledgers([path]).runs[0].data


def test_new_attempt_without_observations_does_not_reuse_old_history(tmp_path: Path) -> None:
    path = write_ledger(
        tmp_path, [row('a', 42, history=[{'attempt': 1, 'metrics': {'score': 0.9}}], history_attempt_max=2)]
    )
    result = query('history', path, '--metric', 'score')['rows'][0]
    assert result['attempt'] == 2 and result['observations'] == 0 and result['value'] is None


def test_cli_formats_source_preservation_and_overwrite_guard(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    path = write_ledger(tmp_path, [row('a', 42, 0.7)])
    before = path.read_bytes()
    main(['rows', str(path), '--metric', '*acc', '--format', 'csv'])
    output = capsys.readouterr()
    parsed = list(csv.DictReader(io.StringIO(output.out)))
    assert parsed[0]['metric:probe/lp0.010/test_acc'] == '0.7'
    assert json.loads(output.err)['source_runs'] == 1
    report = query('rows', path, '--metric', '*acc')
    assert '| experiment_id ' in render_report(report, 'markdown')
    with pytest.raises(SystemExit) as exc:
        main(['inspect', str(path), '--output', str(path)])
    assert exc.value.code == 2 and path.read_bytes() == before
    alias = tmp_path / 'hardlink.yaml'
    alias.hardlink_to(path)
    with pytest.raises(SystemExit) as exc:
        main(['inspect', str(path), '--output', str(alias)])
    assert exc.value.code == 2 and path.read_bytes() == before


def test_cli_help_is_lightweight_and_does_not_launch_controller(tmp_path: Path) -> None:
    script = (
        'import sys\nfrom slurminator.results import main\n'
        'try: main(["--help"])\nexcept SystemExit as e: assert e.code == 0\n'
        'assert not any(k in sys.modules for k in ("torch", "wandb", "paramiko", "textual", "slurminator.cli.orchestrator"))\n'
    )
    result = subprocess.run([sys.executable, '-c', script], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert all(command in result.stdout for command in ['inspect', 'rows', 'summarize', 'history'])

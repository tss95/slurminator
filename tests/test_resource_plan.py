"""Read-only launch proposals bind resources to evidence and require approval."""

from __future__ import annotations

import hashlib
import shlex
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from slurminator.config.cluster_registry import HPC_CONFIGS, HPCType
from slurminator.config.user_config_loader import load_hpc_config_file
from slurminator.result_ledger import read_ledgers
from slurminator.resource_plan import prepare_launch_plan

pytestmark = pytest.mark.unit


def _ledger(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / 'ledger with spaces.yaml'
    path.write_text(yaml.safe_dump({'experiments': rows}))
    return path


def _config(tmp_path: Path, **extra: object) -> Path:
    path = tmp_path / 'hpc.yaml'
    cluster = {'partition': 'normal', 'account': 'test', 'hostname': 'example.invalid', 'username': 'test'}
    path.write_text(yaml.safe_dump({'clusters': {'OLIVIA': {**cluster, **extra}, 'FOX': cluster}}))
    return path


def _prepare(path: Path, config_path: Path, **overrides: object) -> dict:
    arguments = {
        'cluster': 'olivia',
        'gpus_per_job': 2,
        'ram_gb_per_job': 64,
        'time_hours': 12,
        'gpu_reason': 'Two GPUs fit the chosen batch and model.',
        'ram_reason': 'Prior run used 45 GB; request includes headroom.',
        'time_reason': 'Prior comparable run took 9 hours; allow 12.',
        'hpc_config_file': config_path,
        **overrides,
    }
    return prepare_launch_plan(path, **arguments)


def test_proposal_is_read_only_source_bound_and_awaits_approval(tmp_path: Path, monkeypatch) -> None:
    path = _ledger(tmp_path, [{'experiment_id': f'run-{i}', 'status': 'pending'} for i in range(100)])
    config = _config(tmp_path)
    original = path.read_bytes()
    registry_before = dict(HPC_CONFIGS)

    def denied(*args, **kwargs):
        raise AssertionError('Planning must not launch subprocesses or contact a scheduler')

    monkeypatch.setattr(subprocess, 'Popen', denied)
    result = _prepare(path, config, evidence=['reports/comparable-run.json'])
    assert result['approval_status'] == 'awaiting_user'
    assert result['sources'][0]['sha256'] == hashlib.sha256(original).hexdigest()
    assert result['hpc_config_source']['sha256'] == hashlib.sha256(config.read_bytes()).hexdigest()
    assert result['concurrency'] == 100
    assert result['counts']['total'] == result['counts']['pending'] == 100
    assert result['counts']['running'] == result['counts']['completed'] == 0
    assert result['maximum_concurrent_requests']['gpus'] == 200
    assert result['maximum_concurrent_requests']['ram_gb'] == 6400
    assert result['evidence'] == ['reports/comparable-run.json']
    assert result['resource_rationale']['walltime'].startswith('Prior comparable')
    assert shlex.split(result['shell_command']) == result['argv']
    assert result['argv'] == [
        'python',
        '-m',
        'slurminator',
        '--yaml',
        str(path.resolve()),
        '--headless',
        '--olivia-limit',
        '100',
        '--n-gpus',
        '2',
        '--job-ram-gb',
        '64',
        '--job-time-hours',
        '12',
        '--hpc-config-file',
        str(config.resolve()),
    ]
    assert path.read_bytes() == original
    assert HPC_CONFIGS == registry_before
    assert set(tmp_path.iterdir()) == {path, config}


@pytest.mark.parametrize(
    'overrides,match',
    [
        ({'gpus_per_job': -1}, 'gpus_per_job'),
        ({'gpus_per_job': True}, 'gpus_per_job'),
        ({'gpus_per_job': 1.5}, 'gpus_per_job'),
        ({'ram_gb_per_job': 0}, 'ram_gb_per_job'),
        ({'ram_gb_per_job': float('inf')}, 'ram_gb_per_job'),
        ({'time_hours': -1}, 'time_hours'),
        ({'time_hours': float('nan')}, 'time_hours'),
        ({'concurrency': 0}, 'concurrency'),
        ({'concurrency': True}, 'concurrency'),
        ({'gpu_reason': ' '}, 'gpu_reason'),
        ({'ram_reason': None}, 'ram_reason'),
        ({'time_reason': ''}, 'time_reason'),
        ({'cluster': 'unknown'}, 'unsupported cluster'),
        ({'evidence': ['']}, 'evidence reference'),
        ({'evidence': 'report.json'}, 'evidence must be a list'),
    ],
)
def test_invalid_resources_reasons_and_evidence_fail(tmp_path: Path, overrides: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _prepare(_ledger(tmp_path, [{'experiment_id': 'pending'}]), _config(tmp_path), **overrides)


@pytest.mark.parametrize('status', ['completed', 'failed', 'timeout', 'queued', 'running', None])
def test_no_launchable_jobs_fail_clearly(tmp_path: Path, status: str | None) -> None:
    rows = [{'experiment_id': 'done', 'status': status}] if status else []
    with pytest.raises(ValueError, match='No launchable'):
        _prepare(_ledger(tmp_path, rows), _config(tmp_path))


def test_cpu_request_and_explicit_concurrency_with_uncertainty(tmp_path: Path) -> None:
    rows = [{'experiment_id': 'first', 'status': 'partial'}, {'experiment_id': 'second'}]
    result = _prepare(_ledger(tmp_path, rows), _config(tmp_path), gpus_per_job=0, concurrency=1)
    assert result['concurrency'] == 1
    assert result['concurrency_source'] == 'explicit'
    assert result['maximum_concurrent_requests']['gpus'] == 0
    assert result['maximum_concurrent_requests']['ram_gb'] == 64
    assert result['counts']['partial'] == result['counts']['pending'] == 1
    assert any('No measurement evidence' in warning for warning in result['warnings'])
    assert any('not a prediction' in warning for warning in result['warnings'])


@pytest.mark.parametrize(
    'extra',
    [
        {'pinned_hpc': 'FOX'},
        {'required_hpc': 'FOX'},
        {'hpc_assignment': 'FOX'},
        {'metadata': {'pinned_hpc': 'FOX'}},
        {'pinned_hpc': 'unknown'},
    ],
)
def test_conflicting_cluster_pin_or_assignment_fails(tmp_path: Path, extra: dict) -> None:
    with pytest.raises(ValueError, match='cluster'):
        _prepare(_ledger(tmp_path, [{'experiment_id': 'pinned', **extra}]), _config(tmp_path))


def test_dataset_cluster_pin_is_read_from_config(tmp_path: Path) -> None:
    path = _ledger(tmp_path, [{'experiment_id': 'pinned', 'dataset_name': 'sleep'}])
    with pytest.raises(ValueError, match='cluster'):
        _prepare(path, _config(tmp_path, pinned_datasets=['sleep']), cluster='FOX')


def test_known_but_unconfigured_cluster_fails(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='absent'):
        _prepare(_ledger(tmp_path, [{'experiment_id': 'run'}]), _config(tmp_path), cluster='LUMI')


@pytest.mark.parametrize('override', [24, '24', None, 'invalid', 0, float('inf')])
def test_retry_walltime_conflict_fails_except_absent(tmp_path: Path, override: object) -> None:
    path = _ledger(tmp_path, [{'experiment_id': 'retry', 'status': 'partial', 'time_hours_override': override}])
    config = _config(tmp_path)
    if override is None:
        assert _prepare(path, config)['resources_per_new_job']['walltime_hours'] == 12
    else:
        with pytest.raises(ValueError, match='time_hours_override'):
            _prepare(path, config)


def test_cli_precedence_is_explicit_and_matching_retry_is_supported(tmp_path: Path) -> None:
    path = _ledger(
        tmp_path,
        [
            {
                'experiment_id': 'overrides',
                'dataset_name': 'toy',
                'time_hours_override': 12,
                'resources': {'gpu_count': 8, 'memory_gb': 160, 'time_hours': 36, 'gpu_mem_gb': 40},
            }
        ],
    )
    config = _config(tmp_path, dataset_resource_overrides={'toy': {'gpu_count': 4}})
    result = _prepare(path, config)
    assert result['resources_per_new_job'] == {'gpus': 2, 'ram_gb': 64, 'walltime_hours': 12}
    assert result['superseded_resource_overrides'] == [
        {
            'experiment_id': 'overrides',
            'resource_overrides': {'gpu_count': 8, 'mem_gb': 160, 'time_hours': 36, 'mem_per_gpu_gb': 40},
        }
    ]
    # Compare the proposal against the real submission resolver, without a
    # controller, SSH connection, scheduler command or output directory.
    from slurminator.submission import resolve_submission_resources

    context = SimpleNamespace(
        hpc_configs=load_hpc_config_file(config), max_gpus_per_job=2, time_hours_override=12, memory_gb_override=64
    )
    actual = resolve_submission_resources(read_ledgers([path]).runs[0].data, HPCType.OLIVIA, context)
    assert result['resources_per_new_job'] == {
        'gpus': actual.gpu_count,
        'ram_gb': actual.requested_ram_gb,
        'walltime_hours': actual.time_hours,
    }


@pytest.mark.parametrize('key', ['time_hours', 'mem_gb'])
def test_invalid_base_overrides_fail_before_proposal(tmp_path: Path, key: str) -> None:
    path = _ledger(tmp_path, [{'experiment_id': 'bad', 'resources': {key: 'invalid'}}])
    with pytest.raises(ValueError, match='invalid resource override'):
        _prepare(path, _config(tmp_path))


def test_active_requests_retained_in_totals(tmp_path: Path) -> None:
    path = _ledger(
        tmp_path,
        [
            {'experiment_id': 'pending'},
            {'experiment_id': 'partial', 'status': 'partial'},
            {
                'experiment_id': 'active',
                'status': 'running',
                'hpc_assignment': 'OLIVIA',
                'requested_gpu_count': 8,
                'requested_ram_gb': 512,
            },
            {'experiment_id': 'done', 'status': 'completed'},
        ],
    )
    config = _config(tmp_path)
    default = _prepare(path, config)
    assert default['concurrency'] == 3
    assert default['maximum_concurrent_requests']['gpus'] == 12
    assert default['maximum_concurrent_requests']['ram_gb'] == 640
    limited = _prepare(path, config, concurrency=1)
    assert limited['maximum_concurrent_requests']['gpus'] == 8
    assert limited['maximum_concurrent_requests']['ram_gb'] == 512


def test_active_jobs_over_limit_are_not_hidden(tmp_path: Path) -> None:
    path = _ledger(
        tmp_path,
        [
            {'experiment_id': 'pending'},
            *[
                {
                    'experiment_id': f'active-{i}',
                    'status': 'queued',
                    'hpc_assignment': 'OLIVIA',
                    'requested_gpu_count': 4,
                    'requested_ram_gb': 128,
                }
                for i in range(3)
            ],
        ],
    )
    result = _prepare(path, _config(tmp_path), concurrency=1)
    assert result['maximum_concurrent_requests']['jobs'] == 3
    assert result['maximum_concurrent_requests']['gpus'] == 12
    assert result['maximum_concurrent_requests']['ram_gb'] == 384
    assert any('already exceed' in warning for warning in result['warnings'])


@pytest.mark.parametrize('missing', ['hpc_assignment', 'requested_gpu_count', 'requested_ram_gb'])
def test_missing_active_resource_records_fail(tmp_path: Path, missing: str) -> None:
    active = {
        'experiment_id': 'active',
        'status': 'running',
        'hpc_assignment': 'OLIVIA',
        'requested_gpu_count': 2,
        'requested_ram_gb': 64,
    }
    active.pop(missing)
    with pytest.raises(ValueError, match=missing):
        _prepare(_ledger(tmp_path, [{'experiment_id': 'pending'}, active]), _config(tmp_path))


def test_config_change_during_read_rejects_unbound_proposal(tmp_path: Path, monkeypatch) -> None:
    import slurminator.resource_plan as resource_plan

    config = _config(tmp_path)
    original_loader = resource_plan.load_hpc_config_file

    def changed(path):
        loaded = original_loader(path)
        path.write_text(path.read_text() + '\n# changed\n')
        return loaded

    monkeypatch.setattr(resource_plan, 'load_hpc_config_file', changed)
    with pytest.raises(ValueError, match='changed while preparing'):
        _prepare(_ledger(tmp_path, [{'experiment_id': 'run'}]), config)


def test_environment_config_discovery_is_bound_into_command(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path)
    monkeypatch.setenv('SLURMINATOR_HPC_CONFIG_FILE', str(config))
    result = _prepare(_ledger(tmp_path, [{'experiment_id': 'run'}]), config, hpc_config_file=None)
    assert result['argv'][-2:] == ['--hpc-config-file', str(config.resolve())]

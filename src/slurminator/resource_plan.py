"""Prepare reviewable resource requests without launching a controller or jobs."""

from __future__ import annotations

import hashlib
import os
import shlex
from collections import Counter
from pathlib import Path
from typing import Any

from slurminator.config.cluster_registry import HPCType, coerce_hpc_type
from slurminator.config.user_config_loader import (
    HPC_CONFIG_FILE,
    HPC_CONFIG_FILE_ENV,
    REPO_ROOT_ENV,
    find_user_config,
    load_hpc_config_file,
)
from slurminator.experiment_policy import resolve_pinned_hpc, resolve_resource_overrides
from slurminator.experiments.status_enum import ExperimentStatus
from slurminator.result_ledger import json_value, read_ledgers

_LAUNCHABLE = {'pending', 'partial'}
_ACTIVE = {'queued', 'running'}


def _integer(value: Any, name: str, *, minimum: int = 1) -> int:
    """Reject truncation, booleans and nonfinite values in explicit requests."""
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} requires a nonempty explanation')
    return value.strip()


def _cluster(value: Any, *, row: str = 'cluster') -> HPCType:
    try:
        return coerce_hpc_type(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f'{row}: unsupported cluster {value!r}') from exc


def prepare_launch_plan(
    ledger: Path,
    *,
    cluster: str,
    gpus_per_job: int,
    ram_gb_per_job: int,
    time_hours: int,
    gpu_reason: str,
    ram_reason: str,
    time_reason: str,
    concurrency: int | None = None,
    evidence: list[str] | None = None,
    hpc_config_file: Path | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Return an unapproved, source-bound launch proposal with explicit estimates.

    Values and rationale come from the caller; this function does not infer
    resource needs from dataset/model names. Existing cluster policy is read
    without updating the global registry. Row retry walltimes and cluster pins
    must agree with the proposal. Queued/running jobs retain their recorded
    allocations, which must be present to compute an honest aggregate bound.
    No scheduler, network, controller, subprocess or source mutation is used.
    """
    selected_cluster = _cluster(cluster)
    gpus = _integer(gpus_per_job, 'gpus_per_job', minimum=0)
    ram = _integer(ram_gb_per_job, 'ram_gb_per_job')
    hours = _integer(time_hours, 'time_hours')
    reasons = {
        'gpus': _text(gpu_reason, 'gpu_reason'),
        'ram': _text(ram_reason, 'ram_reason'),
        'walltime': _text(time_reason, 'time_reason'),
    }
    if concurrency is not None:
        _integer(concurrency, 'concurrency')
    if evidence is not None and not isinstance(evidence, list):
        raise ValueError('evidence must be a list of references')
    references = [_text(item, 'evidence reference') for item in evidence or []]

    loaded = read_ledgers([Path(ledger)])
    launchable = [run for run in loaded.runs if run.data['status'] in _LAUNCHABLE]
    active = [run for run in loaded.runs if run.data['status'] in _ACTIVE]
    if not launchable:
        raise ValueError('No launchable pending/partial runs; use status or wait to observe existing jobs')

    config_path = find_user_config(
        HPC_CONFIG_FILE,
        override_path=hpc_config_file or os.environ.get(HPC_CONFIG_FILE_ENV),
        repo_root=repo_root or os.environ.get(REPO_ROOT_ENV),
        required=True,
    )
    assert config_path is not None
    config_path = config_path.resolve()
    config_bytes = config_path.read_bytes()
    configs = load_hpc_config_file(config_path)
    if config_path.read_bytes() != config_bytes:
        raise ValueError('HPC config changed while preparing the proposal; retry against a stable config')
    if selected_cluster not in configs:
        raise ValueError(f'{selected_cluster.value} is absent from {config_path}')

    warnings = list(loaded.warnings)
    warnings.append(
        'Resource estimates and rationale are caller supplied; no profiling or scheduler validation occurred.'
    )
    warnings.append('Walltime is a per-job request limit, not a prediction of sweep completion or queue delay.')
    if not references:
        warnings.append('No measurement evidence supplied; resource estimates are unverified and uncertain.')
    if active:
        warnings.append(
            'Existing queued/running jobs keep their recorded requests; CLI flags only affect new submissions.'
        )

    superseded: list[dict[str, Any]] = []
    for run in [*launchable, *active]:
        pin = resolve_pinned_hpc(run.data, configs)
        assignment = run.data.get('hpc_assignment')
        # Submission honors a pin first, then a prior assignment. Active jobs
        # cannot move merely because the new controller selects another host.
        effective = assignment if run.data['status'] in _ACTIVE else pin or assignment
        if effective is not None and _cluster(effective, row=run.experiment_id) != selected_cluster:
            raise ValueError(f'{run.experiment_id}: cluster pin/assignment conflicts with {selected_cluster.value}')
        if pin is not None and _cluster(pin, row=run.experiment_id) != selected_cluster:
            raise ValueError(f'{run.experiment_id}: pinned cluster conflicts with {selected_cluster.value}')

    for run in launchable:
        retry_hours = run.data.get('time_hours_override')
        if retry_hours is not None:
            try:
                retry_hours = int(retry_hours)
            except (ValueError, TypeError, OverflowError) as exc:
                raise ValueError(
                    f'{run.experiment_id}: invalid time_hours_override; resolve it before approval'
                ) from exc
            if retry_hours <= 0 or retry_hours != hours:
                raise ValueError(
                    f'{run.experiment_id}: time_hours_override={run.data["time_hours_override"]!r} '
                    f'conflicts with proposed {hours} hours; row retry walltime takes precedence'
                )
        overrides = resolve_resource_overrides(run.data, hpc_type=selected_cluster, cluster_configs=configs)
        # Submission constructs these base values before applying CLI overrides.
        # Invalid values would therefore fail even when a global request wins.
        for key in ('time_hours', 'mem_gb'):
            if key in overrides:
                try:
                    int(overrides[key])
                except (ValueError, TypeError, OverflowError) as exc:
                    raise ValueError(f'{run.experiment_id}: invalid resource override {key}') from exc
        if overrides:
            superseded.append({'experiment_id': run.experiment_id, 'resource_overrides': overrides})

    active_allocations = []
    for run in active:
        if run.data.get('hpc_assignment') is None:
            raise ValueError(
                f'{run.experiment_id}: active job lacks hpc_assignment; cannot bound selected-cluster usage'
            )
        allocated_gpus = _integer(
            run.data.get('requested_gpu_count'), f'{run.experiment_id}: requested_gpu_count', minimum=0
        )
        allocated_ram = _integer(run.data.get('requested_ram_gb'), f'{run.experiment_id}: requested_ram_gb')
        active_allocations.append({'experiment_id': run.experiment_id, 'gpus': allocated_gpus, 'ram_gb': allocated_ram})

    limit = concurrency if concurrency is not None else len(launchable) + len(active)

    # For either resource, the maximum over currently active subsets occurs at
    # the number of active jobs initially present or at the chosen job limit.
    # Each remaining slot can be filled by a homogeneous proposed new job.
    def maximum_request(field: str, per_job: int) -> int:
        ordered = sorted((job[field] for job in active_allocations), reverse=True)
        prefix = [0]
        for value in ordered:
            prefix.append(prefix[-1] + value)
        return max(
            prefix[count] + min(len(launchable), max(0, limit - count)) * per_job for count in range(len(active) + 1)
        )

    if limit < len(active):
        warnings.append('Active jobs already exceed proposed concurrency; lowering the limit does not cancel them.')
    if superseded:
        warnings.append('Explicit GPU/RAM/walltime CLI requests supersede the listed dataset/row resource overrides.')

    argv = [
        'python',
        '-m',
        'slurminator',
        '--yaml',
        loaded.sources[0]['path'],
        '--headless',
        f'--{selected_cluster.value.lower()}-limit',
        str(limit),
        '--n-gpus',
        str(gpus),
        '--job-ram-gb',
        str(ram),
        '--job-time-hours',
        str(hours),
        '--hpc-config-file',
        str(config_path),
    ]
    if repo_root is not None:
        argv.extend(['--repo-root', str(Path(repo_root).resolve())])
    statuses = Counter(run.data['status'] for run in loaded.runs)
    return json_value(
        {
            'schema_version': 1,
            'command': 'prepare',
            'approval_status': 'awaiting_user',
            'approval_required_for': 'Every controller launch, deployment or redeployment requires explicit user approval.',
            'sources': loaded.sources,
            'hpc_config_source': {'path': str(config_path), 'sha256': hashlib.sha256(config_bytes).hexdigest()},
            'cluster': selected_cluster.value,
            'counts': {
                'total': len(loaded.runs),
                **{status.value: statuses[status.value] for status in ExperimentStatus},
            },
            'launchable_runs': len(launchable),
            'active_runs': len(active),
            'concurrency': limit,
            'concurrency_source': 'explicit' if concurrency is not None else 'all_launchable_plus_active',
            'resources_per_new_job': {'gpus': gpus, 'ram_gb': ram, 'walltime_hours': hours},
            'resource_rationale': reasons,
            'evidence': references,
            'maximum_concurrent_requests': {
                'jobs': max(len(active), min(limit, len(launchable) + len(active))),
                'gpus': maximum_request('gpus', gpus),
                'ram_gb': maximum_request('ram_gb', ram),
                'scope': 'Ledger jobs on the selected cluster; componentwise upper bounds, not reserved capacity.',
            },
            'active_allocations': active_allocations,
            'superseded_resource_overrides': superseded,
            'argv': argv,
            'shell_command': shlex.join(argv),
            'warnings': warnings,
        }
    )


__all__ = ['prepare_launch_plan']

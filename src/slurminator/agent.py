"""Display-free agent entry: status, quiet waiting, and unapproved launch proposals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

import yaml

from slurminator.observer import read_status, wait_for_event


def build_parser() -> argparse.ArgumentParser:
    """Build a read-only CLI with no live launch or scheduler actions."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    status = commands.add_parser('status', help='Read progress and counts as compact JSON')
    status.add_argument('ledger', type=Path)
    status.add_argument('--source', choices=['controller', 'ledger'], default='controller')
    status.add_argument('--runs', action='store_true', help='Include per-run progress counters')
    status.add_argument('--max-age', type=float, help='Override stale-controller threshold in seconds')
    wait = commands.add_parser('wait', help='Stay silent until one event occurs; suitable for a background task')
    wait.add_argument('ledger', type=Path)
    wait.add_argument('--until', choices=['finalized', 'failure'], default='finalized')
    wait.add_argument('--timeout', type=float, help='Maximum wait in seconds; default is no deadline')
    wait.add_argument('--poll-interval', type=float, default=15, help='Local sidecar checks in seconds (default: 15)')
    wait.add_argument('--max-age', type=float)
    prepare = commands.add_parser('prepare', help='Produce a resource proposal; always requires user approval')
    prepare.add_argument('ledger', type=Path)
    prepare.add_argument('--cluster', required=True)
    prepare.add_argument('--gpus-per-job', type=int, required=True)
    prepare.add_argument('--ram-gb-per-job', type=int, required=True)
    prepare.add_argument('--time-hours', type=int, required=True)
    prepare.add_argument('--gpu-reason', required=True)
    prepare.add_argument('--ram-reason', required=True)
    prepare.add_argument('--time-reason', required=True)
    prepare.add_argument('--concurrency', type=int)
    prepare.add_argument('--evidence', action='append', default=[])
    prepare.add_argument('--hpc-config-file', type=Path)
    prepare.add_argument('--repo-root', type=Path)
    prepare.add_argument('--isolated-submission-env', action='store_true')
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Print one JSON result, with explicit errors and non-success wait exit codes."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == 'status':
            result = read_status(args.ledger, source=args.source, include_runs=args.runs, max_age=args.max_age)
        elif args.command == 'wait':
            result = wait_for_event(
                args.ledger,
                until=args.until,
                timeout=args.timeout,
                poll_interval=args.poll_interval,
                max_age=args.max_age,
            )
        else:
            from slurminator.resource_plan import prepare_launch_plan

            result = prepare_launch_plan(
                args.ledger,
                cluster=args.cluster,
                gpus_per_job=args.gpus_per_job,
                ram_gb_per_job=args.ram_gb_per_job,
                time_hours=args.time_hours,
                gpu_reason=args.gpu_reason,
                ram_reason=args.ram_reason,
                time_reason=args.time_reason,
                concurrency=args.concurrency,
                evidence=args.evidence,
                hpc_config_file=args.hpc_config_file,
                repo_root=args.repo_root,
                isolated_submission_env=args.isolated_submission_env,
            )
        print(json.dumps(result, allow_nan=False, separators=(',', ':')), flush=True)
    except (OSError, ValueError, RuntimeError, yaml.YAMLError) as error:
        print(json.dumps({'error': str(error)}, allow_nan=False), file=sys.stderr)
        raise SystemExit(2) from error
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    if args.command == 'wait' and result['event'] not in {'finalized', 'failure'}:
        raise SystemExit(2)


if __name__ == '__main__':
    main()

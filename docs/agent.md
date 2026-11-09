# Agent interface

The CPU-first agent interface separates scheduling from read-only observation.
An approved controller runs with `--headless`; observers read its compact status
file without importing a dashboard, connecting to Slurm or starting another
controller. The ordinary dashboard remains the default.

## Status and progress

```bash
python -m slurminator.agent status path/to/ledger.yaml
python -m slurminator.agent status path/to/ledger.yaml --runs
```

Each call returns one JSON object with all status counts (including zeros),
aggregate progress, controller state and freshness. `--runs` includes per-run
step/epoch counters and telemetry timestamps without histories. Slurminator's
`pending` means awaiting submission; `queued` means submitted and waiting in
Slurm. Failed, cancelled, timed-out and out-of-memory runs remain separate from
successful `completed` runs.

Progress shares the dashboard calculation: average the individual run fractions,
preferring steps over epochs; completed is one and pending/queued are zero.
Failed terminal runs are excluded and reported separately. Every eligible run
has equal weight, including when budgets differ. Nonfinite or boolean counters
are unusable; valid epoch counters can supply progress instead. Unknown progress
contributes zero. This percentage measures recorded training progress, not ETA
or proof that final evaluation artifacts exist.

For a ledger without a headless controller, request the persisted source explicitly:

```bash
python -m slurminator.agent status path/to/ledger.yaml --source ledger --runs
```

This reuses the [results reader](result_analysis.md), retains its journal/freshness
warnings, and never claims results are finalized. Ordinary controller snapshots
may be newer than the saved ledger. Detailed metrics, filtering, seed summaries
and histories remain available through `python -m slurminator.results`.

The corresponding Python APIs are `slurminator.observer.read_status(Path(...))`
and `slurminator.observer.wait_for_event(Path(...))`.

## Quiet waiting

```bash
python -m slurminator.agent wait path/to/ledger.yaml --until finalized
```

Run this as one background shell task in the agent client. It stays silent while
ordinary code checks the small status file every 15 seconds; there are no periodic
model calls or scheduler queries. It prints one JSON event and exits. Add
`--until failure` to return on the first observed failed terminal run, or on
finalization if there are no failures. Existing failures also satisfy that query.
`--timeout SECONDS` bounds the wait; it is not a completion estimate.

Events `finalized` and `failure` exit zero (a matching event, not a claim that all
experiments succeeded). `controller_stale`, `controller_stopped`, `controller_error`,
`controller_restarted`, `ledger_changed` and `timeout` exit 2. Invalid/missing
state also exits 2 with a JSON error on stderr; interruption exits 130. A watcher
binds to one controller run and does not silently follow its replacement.

Attach after the intended controller has published its first receipt: startup
may spend time connecting before publication. A missing receipt fails explicitly;
an old finished receipt describes the previous run, even while a new process is
starting. Check `run_id` and controller identity before attaching to a replacement.
Concurrent ledger/receipt changes during completion verification fail explicitly;
query the current state again before deciding what to analyze.

`results_finalized` means retries/submissions have settled, all runs are terminal,
bounded final telemetry collection has settled, and the final ledger was saved.
The response exposes missing metrics; it does not certify downstream artifacts.
Partial runs remain eligible for scheduling. A 100% progress bar does not satisfy
this condition. The saved ledger hash must still match the completion receipt.

Only headless controllers publish this contract in the first version. The sidecar
lives at `.orchestrator_status/agent/<ledger filename>.json` beside the ledger.
Publication is atomic and bounded; a stale heartbeat is reported explicitly.
`updated_at`/`heartbeat_at` describe controller liveness. `progress_updated_at`
describes the last substantive status snapshot; heartbeat renewal during long
operations does not imply fresh training telemetry. Per-run telemetry timestamps
remain available with `--runs`.
Only one controller may own a ledger. The headless lock prevents competing
headless writers, but does not protect against a separately launched legacy UI
controller. Do not run both on the same ledger.

Background process support alone does not prove automatic model continuation in
every client. Claude Code can run the command as a background Bash task. For a
Codex app-server integration, deliver the completed event as `turn/start.toolOutput`
to the existing open thread. PMT provides a local adapter; see its
[agent guide](https://github.com/tss95/PMT/blob/pmma/docs/hpc/slurminator_agent.md). A loaded thread is only a
proxy for an open session; clients may retain loaded threads briefly after a UI
disconnects. No adapter starts a closed session, and RPC acceptance does not
certify that subsequent analysis completed.
MCP is not required for these APIs.

## Prepare resources and request approval

`prepare` produces a proposal and never executes its command:

```bash
python -m slurminator.agent prepare path/to/ledger.yaml \
  --cluster olivia --gpus-per-job 1 --ram-gb-per-job 80 --time-hours 4 \
  --gpu-reason 'Comparable recipe fits on one GPU; no demonstrated need for distributed training.' \
  --ram-reason 'Comparable measured host-memory peak plus explained headroom.' \
  --time-reason 'Comparable step throughput, full evaluation cost and explained margin.' \
  --evidence path/to/comparable-results.json
```

These numbers are illustrative, not recommended defaults. The calling agent must
derive estimates for the actual recipe:

- Walltime: use comparable observed runtime/throughput and the planned budget,
  including initialization, evaluations, probes and checkpoint work. Explain the
  margin and uncertainty. Queue wait is not job walltime.
- GPUs: distinguish model/batch fit from a speedup assumption; require evidence
  before increasing GPUs for distributed scaling.
- Host RAM: use comparable peak resident memory and account for dataset loading,
  workers and process count. Host RAM is separate from GPU memory.
- Concurrency: normally accommodate all eligible runs. Use a smaller limit only
  for an explicit user choice or concrete resource constraint; explain it.

For a local controller running inside its own allocation, add
`--isolated-submission-env` to `prepare` so the exact proposed launch includes
environment isolation. This selects direct local sbatch execution with only a
minimal identity/process environment and explicit cluster `sbatch_env` values.
It retains `--export=ALL` over that clean environment to avoid implicit login
environment reconstruction. The current shared checkout supplies the child
script; the controller does not run remote repository preparation in this mode.
The child wrapper must initialize its runtime and set `SLURM_EXPORT_ENV=ALL`
before launching job steps. PMT's `universal_job.sh` does this.

Slurm always carries submission-process `SLURM_*` variables, including with
restricted export modes. Sanitizing only the `--export` argument is insufficient;
the process environment must be isolated too. See the official
[sbatch export documentation](https://slurm.schedmd.com/sbatch.html#OPT_export).

The tool records explanations, optional evidence references, ledger/config hashes,
resource precedence, peak concurrent requests and an exact command. Estimates are
supplied by the agent; the tool does not infer reliable resource needs from YAML
alone. Default concurrency covers pending/partial and active runs. Incompatible
cluster pins or retry walltimes fail explicitly. Existing active jobs retain
their recorded allocations. Use `--hpc-config-file` and `--repo-root` when needed.

Every proposal has `approval_status: awaiting_user`. Present the proposal and ask
for explicit approval before launch or deployment. A phone connection, permission
to develop, or approval of an earlier sweep is not consent. Reprepare if the ledger,
configuration or resources change; obtain new approval for material changes.
There is deliberately no `launch` subcommand and no self-approval flag.

After approval, the proposed command uses the normal controller entry with
`--headless`. This does not bypass site launch guards, sandbox restrictions or
submission-environment requirements. Controller CPU/RAM is a separate allocation
from the training resources in the proposal. No live controller needs to run just
to prepare or inspect a plan.

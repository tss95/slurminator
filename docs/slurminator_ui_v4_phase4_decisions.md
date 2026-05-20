# Slurminator UI v4 Phase 4 Decisions

This document records implementation decisions made while landing Phase 4. It
is not a replacement for the technical spec; it captures choices made where the
spec left room for implementation detail.

## Slice 0: Git Provenance Capture

- Provenance is captured both when generating the experiment YAML and at submit
  time. The generation-time fields document the sweep source; the
  submission-time `git_sha_at_submission` field documents the exact code state
  for each queued job.
- Provenance capture is best-effort. Git lookup failures return missing values
  instead of aborting orchestration, because provenance should not prevent
  existing sweep submission workflows from running.

## Slice 1: Status Schema v1.1 And History Files

- Schema v1.1 is forward-asymmetric by design: v1.1 readers accept v1.0 status
  files, but v1.0 readers are not expected to accept v1.1 files.
- `attempt` defaults to `1` when reading older v1.0 files.
- History writes use append mode rather than atomic replacement, matching the
  JSONL contract. The existing status-write throttle controls history append
  cadence as well.

## Slice 2: History Ingestion

- Incremental history reads use `stat` followed by `tail -c` from the previous
  byte offset. This keeps polling cheap and simple.
- If the history file is smaller than the stored offset, the reader treats it as
  truncated or rotated and refetches from the beginning.
- Active runs keep full in-memory history. Terminal runs are bounded to the last
  `100` entries to cap dashboard memory growth.
- The actual OLIVIA sweep gate was intentionally skipped during implementation;
  Slice 2 was validated with unit tests, full package tests, and dry-run
  orchestration checks.

## Slice 3: Command Queue Infrastructure

- Command queue roots are discovered from, in order: the experiment file
  directory, `SAVE_PATH`, experiment-row `save_path` values, and enabled cluster
  `save_path` values. The experiment file directory is always included so v4
  dashboard commands and orchestrator polling share a local queue root.
- `pause_submissions` gates new submissions and reassignment. Scheduler/status
  polling continues while paused.
- Slice 3 implements only the initial handler set:
  `cancel_run`, `cancel_all`, `pause_submissions`, `resume_submissions`, and
  `set_concurrency_limit`.
- Failed commands move to `failed/` with a `<command_id>.error.txt` sidecar; the
  orchestrator logs a warning and keeps running.

## Slice 4: Textual Dashboard Skeleton

- Textual is optional and lives in the `v4` extra. The `dev` extra includes it
  so tests can cover v4, but default v2/v3 users do not need to install Textual.
- The v4 dashboard uses the threaded integration model from the spec. The
  synchronous orchestrator loop continues to poll; after each poll it publishes a
  deep-copied `_dashboard_snapshot` for the Textual app.
- Pressing `q` requests a graceful orchestrator shutdown. Earlier Slice 4
  implementation exited only the Textual dashboard and left Slurminator running
  headless; live use showed that this was surprising because users still had to
  press Ctrl+C to terminate the process.
- E.1 future modules are present as inert placeholders so later slices can fill
  them without changing package layout.
- The Slice 4 quota footer is a lightweight skeleton showing submission state,
  active limits, and assigned HPCs. Provider-backed quota detail remains in the
  existing Rich dashboard until a later v4 slice.

## Slice 5: Per-run Plot Screen

- The plot screen uses plain `plotext` rendered into a Textual `Static` widget
  rather than `textual-plotext`. `textual-plotext` was checked during
  implementation and looked riskier for this gate because its latest PyPI
  release was `1.0.1` from 2024-11-30 and its repository had open issues plus a
  documented repeated-build/log-scale limitation. Direct `plotext` keeps
  log-scale behavior under Slurminator control.
- `plotext` is added only to the `v4` and `dev` extras.
- `HPCOrchestrator.force_read_full_history(exp)` owns construction of
  `StatusIngestContext`, so Textual screens do not need to know status-ingest
  internals.
- The plot screen force-reads history on mount, then refreshes from the same
  dashboard snapshot path as the home screen.
- Best-overlay direction uses `display_metric_info[metric].higher_better` when
  present and defaults to higher-is-better when unknown.
- Multi-run overlays, attempt filtering/coloring, detail screens, and log
  screens remain out of scope for Slice 5.

## Slice 7.5: Home Screen Parity Polish

- No formal Slice 7.5 contract file is present in this checkout. This pass was
  scoped as a narrow parity follow-up to design decision D2 and live review
  feedback: restore the high-signal home-screen header/progress/footer elements
  that v3 users rely on.
- The v4 home screen now has a status summary line, three top progress bars
  (`Completed`, aggregate run `Progress`, and `Running` slots), and a multi-line
  footer with completion, sweep, update time, submission state, active limits,
  host, experiment-file name, Slurm resource summary, and quota information.
- Provider-backed quota footer rendering is enabled in v4 with the same
  five-minute cache cadence as the Rich dashboard. If no provider-backed quota
  can be read, v4 shows the provider's unavailable hint rather than hiding the
  footer line.
- The experiment table now applies v3-style status colors, metric threshold
  colors, step-first progress formatting, and v3-like row ordering. Metric
  shortforms belong in the column headers; row cells contain only values.
  Selection is preserved by experiment id across refreshes so sorting does not
  make the cursor jump to a different run.
- The Textual layout gives the footer a fixed three-line area and leaves the
  experiment table as the remaining flexible region, so terminal resizes should
  reallocate table height instead of treating the table as a fixed block.
- `q` is bound on every v4 screen and sets both the Textual dashboard flag and
  an orchestrator-side exit flag. This avoids the earlier failure mode where a
  modal screen could close or ignore the UI without the orchestrator loop seeing
  a graceful shutdown request.

## Known Terminal Compatibility

### tmux + TERM requirement

The Textual dashboard requires `tmux-256color` (preferred) or at minimum
`xterm-256color` to receive resize events and render correctly. The legacy
`screen-256color` default that ships with most tmux installations causes the v4
dashboard to skip resize handling. Users running tmux must add the following to
`~/.tmux.conf`:

```tmux
set-option -g default-terminal "tmux-256color"
set-option -ga terminal-overrides ",xterm-256color:RGB"
```

After updating, restart tmux (`tmux kill-server && tmux`) or export
`TERM=tmux-256color` inside the existing session before launching the
dashboard. Verify with `echo $TERM`; it must show `tmux-256color` or
`xterm-256color`, not `screen-256color`.

### Outer terminal compatibility

On Windows, use Windows Terminal or another modern terminal emulator such as
WezTerm rather than the legacy `powershell.exe` console host. The legacy host
does not reliably propagate `SIGWINCH` through SSH, which prevents resize
events from reaching the remote tmux + Textual stack regardless of TERM
configuration.

If `tmux-256color` is unavailable on the remote system, which is rare but
possible on older clusters, users can install it locally with `tic`. See the
Textual terminal support documentation for the workaround.

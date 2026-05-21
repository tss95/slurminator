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

## Plot Axis Resolution

- Schema v1.1 includes optional `HistoryEntry.unit`. This field was folded into
  v1.1 while the schema is still in development; readers accept earlier v1.1
  history lines that do not yet contain the field by treating the unit as
  unknown.
- The plot screen resolves its canonical x-axis from, in order: the most recent
  populated history-entry `unit`, the projected status row (`progress.unit` or
  `progress_unit`), a shape fallback that prefers step when step values vary
  meaningfully more than epoch values, and finally `epoch`.
- Slurminator does not know about PMT pseudo-epochs. PMT-specific callbacks map
  step-budget runs to the general schema contract `progress.unit="step"`;
  dashboard plotting only consumes that declared unit.
- Step-based plots prefer `step` values with an explicit `Step` x-axis label.
  Epoch-based plots prefer `epoch` values with an explicit `Epoch` label.
- The plot renderer requests distributed x-axis ticks and linear y-axis ticks
  when appropriate so short metric trajectories remain readable in
  terminal-sized panels.
- Live review later showed that dense gridlines, colored ANSI output, and
  maximal plot sizing made plotext hard to read inside Textual panels. The v4
  plot screen now uses a simpler clear theme, no grid, clipped plain-text output,
  and capped dimensions so plots stay inside the widget bounds.

## Slice 7.5: Home Screen Parity Polish

- No formal Slice 7.5 contract file is present in this checkout. This pass was
  scoped as a narrow parity follow-up to design decision D2 and live review
  feedback: restore the high-signal home-screen header/progress/footer elements
  that v3 users rely on.
- The v4 home screen now has a status summary line, three top progress bars
  (`Completed`, aggregate run `Progress`, and `Running` slots), and a multi-line
  footer with completion, sweep, update time, submission state, active limits,
  host, experiment-file name, Slurm resource summary, and quota information.
- Footer content is rendered as Rich `Text` with explicit style spans rather
  than prejoined plain strings, so labels and quota numbers keep the color cues
  from the Rich dashboard.
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

## Slice 8: Global Menu

- No formal Slice 8 contract file is present in this checkout. This pass was
  scoped from the Phase 4 sequence notes: add the `g` shortcut and a
  `GlobalMenuScreen` for dashboard-wide actions.
- The global menu is available from the home screen and from per-run plot,
  detail, log, and menu screens. `Esc` closes the menu and `q` keeps its global
  graceful shutdown behavior.
- Initial global actions are command-queue-backed `pause_submissions`,
  `resume_submissions`, and `cancel_all`. Concurrency limit editing is kept out
  of this slice because the earlier plan calls out a dedicated concurrency modal
  in Slice 11.
- `cancel_all` submits the existing command action with `{"scope": "session"}`.
  The existing command handler only cancels queued/running jobs, so terminal
  runs are not touched.
- Review adjustment: pause/resume is only exposed through the global menu, not
  a direct home-screen `p` key, to avoid duplicate controls. Quit remains a
  main summary view action only; nested menus and per-run screens use `Esc` to
  return instead of offering a `q` shortcut.

## Slice 9: Per-Run Cancel And Relaunch

- No formal Slice 9 contract file is present in this checkout. This pass was
  scoped from the Phase 4 sequence notes and existing placeholders:
  per-run actions for cancelling an active run and relaunching a terminal run.
- `Cancel selected run` writes the existing `cancel_run` command queue action.
  The existing handler only issues `scancel` when the run is still
  queued/running.
- `Relaunch` opens a confirmation modal and writes a new `relaunch_run` command.
  The command handler refuses queued/running/pending runs to avoid duplicate
  active jobs; users should cancel active jobs first, then relaunch after the
  scheduler marks them terminal.
- `Return` is present as an explicit non-destructive menu item for leaving the
  per-run menu; `Esc` remains available as a shortcut.
- A relaunch resets the selected terminal experiment to `PENDING`, clears stale
  job/log/history fields, keeps the experiment assignment and sweep parameters,
  and records lightweight audit fields (`manual_relaunch_count`,
  `relaunch_requested_at`, `relaunch_previous_status`, and the previous job id
  when available). The next orchestrator poll performs the actual sbatch submit
  through the normal submission path.

## Slice 10: Per-Run Settings

- No formal Slice 10 contract file is present in this checkout. This pass was
  scoped from the existing per-run `Settings` placeholder and submission
  contracts.
- Settings are explicitly "next submission" settings. They do not mutate an
  already-running Slurm allocation. The form writes an `update_run_settings`
  command, and the orchestrator applies the changes on its next command-queue
  pass.
- Editable fields are walltime override, memory override, GPU count override,
  and pinned HPC. Walltime is stored as `time_hours_override`, because that
  field already has highest precedence in Slurminator's submission resource
  resolution. Memory and GPU count are stored under `resource_overrides`, and
  pinned HPC is stored as `pinned_hpc`.
- Blank fields clear the corresponding override. The form also exposes a
  `Clear overrides` action for clearing all four fields at once, plus `Return`
  as a non-destructive exit path.

## Slice 11: Concurrency Modal And Help Overlay

- No formal Slice 11 contract file is present in this checkout. This pass was
  scoped from the remaining v4 placeholders and live review feedback: add a
  global concurrency modal, add a lightweight help overlay, and set the app
  title to `Slurminator`.
- The concurrency modal writes one existing `set_concurrency_limit` command per
  configured HPC. It validates non-negative integer values locally and relies on
  the orchestrator's existing command-queue pass to apply the limits on the next
  poll. This keeps the dashboard write path consistent with pause/resume and
  cancel actions.
- Review adjustment: the modal now shows explicit `Apply limits` and `Return`
  buttons. Pressing Enter in a limit field also applies the form; `Esc` returns
  without saving to avoid accidental concurrency changes.
- Only HPCs already connected in the current orchestrator session are editable.
  The command handler also rejects `set_concurrency_limit` commands for
  disconnected HPCs, so an operator cannot enable a cluster that the
  orchestrator failed to connect to at startup.
- Help is available through `?` and through the global menu. It is intentionally
  informational only; command behavior remains owned by the individual screens
  and menus.
- The app title is assigned after Textual app construction rather than passed as
  `title=` to `App.__init__`, because the installed Textual version on OLIVIA
  rejects that keyword argument.
- The home-screen trajectory column now resolves metric history through the raw
  metric key, the display shortform, and finally the available history metric
  keys. Live runs exposed that exact-only lookup could leave the trajectory
  blank even when the history file contained metric points.
- Follow-up live review exposed that falling through to arbitrary history keys
  made the trajectory column change meaning over time, for example showing
  total loss until a sparse probe metric appeared. The trajectory column now
  tracks only the declared primary metric, resolves shortforms to raw history
  keys, and coalesces consecutive repeated values so probe metrics are not
  visually duplicated on every status write between probe updates.
- `MetricInfo.best_key` is part of the in-development v1.1 status schema so v4
  can restore the v3-style `current (best)` metric cells for primary and
  secondary metrics. For older in-flight rows missing `best_key`, v4 infers the
  common `step_best` -> `global_best` key pattern.
- Live PMT review also exposed the opposite failure mode: no history files at
  all because Slurm jobs inherited `PMT_ENV_LOADED=1` from the launcher and
  `step_0.sh` could early-return inside `universal_job.sh`. PMT's job wrapper
  now forces `PMT_FORCE_RELOAD=1` when sourcing the env script so newly
  submitted jobs pick up the sibling Slurminator source and write current
  status/history schema files.

## Known Terminal Compatibility

### tmux + TERM compatibility

The v4 dashboard uses app-side terminal-size polling as a fallback when tmux or
SSH does not deliver resize events reliably. Because of that fallback, users do
not need to change global tmux settings just to get the adaptive dashboard
layout.

Live review showed that forcing `tmux-256color`/RGB globally can make normal
interactive shells redraw poorly, including readline history appearing
additive while scrolling through commands. If that happens, revert the tmux
changes, restart tmux, and keep the dashboard on the default `screen-256color`
path. The dashboard uses resize polling rather than refusing to start or logging
a TERM compatibility warning.

For an already-running tmux server, removing the lines from `~/.tmux.conf` is
not enough. Reset the live server:

```bash
tmux set-option -g default-terminal "screen-256color"
tmux set-option -gu terminal-overrides
```

Existing panes keep the `TERM` value they launched with. Open a new pane/window,
or restart the affected shell with `TERM=screen-256color exec bash -l`.

If resize handling is still unreliable in a dedicated dashboard tmux
session/pane, opt that session into `tmux-256color`:

```tmux
set-option -g default-terminal "tmux-256color"
set-option -ga terminal-overrides ",xterm-256color:RGB"
```

After updating, restart tmux (`tmux kill-server && tmux`). Avoid exporting a
different `TERM` in a long-lived shell unless it is a dedicated dashboard pane;
mismatched tmux/TERM settings can break shell redraw even if the dashboard
itself looks better.

### tmux + clipboard compatibility

The v4 dashboard's copy actions use OSC 52 terminal clipboard sequences.
Slurminator writes both the normal Textual OSC 52 sequence and, when `$TMUX` is
set, a tmux DCS passthrough-wrapped OSC 52 sequence. This preserves mouse
support in the dashboard while giving users a deterministic way to copy the
dashboard experiment-list ID, such as `experiments_20260521_133111`.

tmux must still be configured to forward clipboard sequences to the outer
terminal. For the current tmux server:

```bash
tmux set-option -g set-clipboard on
tmux set-option -g allow-passthrough on
```

For persistent setup in `~/.tmux.conf`:

```tmux
set-option -g set-clipboard on
set-option -g allow-passthrough on
```

These clipboard settings are independent of the `TERM`/resize settings above
and should not require switching the session to `tmux-256color`. The outer
terminal must also allow OSC 52 clipboard writes. Windows Terminal and WezTerm
support this path; some terminal emulators, SSH clients, or managed terminal
policies may block it.

### Outer terminal compatibility

On Windows, use Windows Terminal or another modern terminal emulator such as
WezTerm rather than the legacy `powershell.exe` console host. The legacy host
does not reliably propagate `SIGWINCH` through SSH, which prevents resize
events from reaching the remote tmux + Textual stack regardless of TERM
configuration.

If `tmux-256color` is unavailable on the remote system, which is rare but
possible on older clusters, users can install it locally with `tic`. See the
Textual terminal support documentation for the workaround.

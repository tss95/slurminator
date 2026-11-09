# Read-only experiment result queries

Use `python -m slurminator.results` to inspect experiment-state YAML files without
starting a controller. Installed packages also expose `slurminator-results`.
The reader uses Slurminator's safe YAML loader and override parser. It imports
neither the controller nor Torch, W&B, SSH, or dashboard packages.

This replaces repeated scripts for status counts, metric extraction, seed
tables, and history queries. Inputs must contain an `experiments` list; generator
`--sweepfile` YAML describes planned runs and is not a results ledger.

## Inspect before selecting metrics

```bash
python -m slurminator.results inspect experiment_lists/experiments_example.yaml
python -m slurminator.results inspect experiment_lists/experiments_example.yaml \
  --metric 'probe/*acc*' --limit 40
python -m slurminator.results inspect experiment_lists/experiments_example.yaml \
  --param 'model.*' --param 'loss.*'
```

Inspection returns the experiment title, description, source hash and modification
time, status/dataset counts, metric coverage, and varying sweep parameters.
`--param` filters parameter keys and includes constants as well as differences. Lists
default to 20 entries and report their total size. `--limit 0` requests all entries.
The tool reads each supplied YAML once per invocation, using LibYAML when available.
It parses one complete ledger at a time; histories are released after counting
unless requested by the `history` command. It does not recursively discover files
or cache snapshots, so pass the intended ledgers explicitly.

## Export only useful fields

```bash
python -m slurminator.results rows experiment_lists/experiments_example.yaml \
  --status completed --field dataset_name --field param:model.name \
  --field metadata.config_profile --field output_dir --field wandb_run_url \
  --metric 'probe/*/test_acc' --format csv --limit 0 --output reports/runs.csv
```

Selectors have these meanings:

| Selector | Meaning |
| --- | --- |
| `status`, `job_id`, `progress.current_step`, `metadata.config_profile` | Exact row field, with dots for nested mappings |
| `param:model.name` | Exact key in parsed `sweep_params`; dots belong to the key |
| `metric:probe/lp0.010/test_acc` | Exact metric name; decimal points and slashes are preserved |
| `seed` | Row, metadata, or sweep seed; conflicting recorded values are rejected |
| `ledger`, `metric_source`, `history_count` | Source path, metric surface, or saved history count |

`--metric` accepts an exact name or quoted shell glob and can be repeated. Exact
matches take precedence over glob syntax. A selector matching no metric fails
instead of silently returning an empty table. `--where FIELD=VALUE` uses typed
YAML equality, for example `--where seed=42` or
`--where param:loss.lambda_trd=1.5`. Multiple conditions are ANDed; repeated
`--status` options select any of those statuses.

JSON is the default. Markdown retains report context; CSV contains rows only
and writes provenance, selection, warnings, and omitted-row counts to stderr.
All formats are bounded by `--limit`; use `--limit 0` for a complete export.
Missing and invalid metrics are recorded separately. Non-finite values serialize
as JSON null, never as fabricated zeros. Output cannot overwrite an input ledger.

## Aggregate independent seeds

```bash
python -m slurminator.results summarize experiment_lists/experiments_example.yaml \
  --metric 'probe/*/test_acc' \
  --group-by dataset_name --group-by param:model.name --group-by param:loss.lambda_trd \
  --expect-seeds 42,7,123 --require-complete --scale 100
```

Grouping fields are explicit: include every varying recipe/protocol dimension
that should remain separate. The tool cannot infer scientific equivalence.
Only completed rows are summarized by default; the report always includes status
counts for the entire input. `--include-incomplete` explicitly allows partial runs.

Each group/metric reports finite-value count, mean, sample standard deviation
(`ddof=1`), min/max, per-seed values, missing/invalid metric seeds, and expected
seed coverage. One seed has null sample SD. `--scale 100` is an explicit conversion;
the tool never guesses whether a metric is a fraction or percentage.
`--require-complete` requires `--expect-seeds` and rejects incomplete selected
groups. It cannot detect an entirely absent group without an external protocol.

Duplicate seeds within a group fail, including across input files. Add grouping
fields for different conditions, or explicitly select the intended retry. The
tool never picks the highest-scoring attempt or treats retries as extra seeds.

## Query observed history

```bash
python -m slurminator.results history experiment_lists/experiments_example.yaml \
  --metric 'probe/lp0.010/step_best_top1_acc' --mode last
python -m slurminator.results history experiment_lists/experiments_example.yaml \
  --metric 'probe/lp0.010/step_best_top1_acc' --mode best --direction max
python -m slurminator.results history experiment_lists/experiments_example.yaml \
  --metric 'train/avg_total_loss' --mode all --attempt 2 --limit 0 --format csv
```

The default is the latest recorded attempt, using `history_attempt_max` as well
as saved points. An attempt with no observations stays empty; earlier attempts
are not substituted. Last means the last finite observation, ordered by step,
epoch, then timestamp. Best requires an explicit direction and breaks ties at
the earliest coordinates. Each result retains step, epoch, timestamp, attempt,
observation count, and the ledger's `history_truncated` flag. Best is selected
independently per metric; these rows do not certify a shared selected checkpoint.

## Evidence and limits

- Ledger metrics use `all_metrics` when present, otherwise legacy `metrics`.
  Dashboard aliases are not renamed or combined with canonical keys. Choose the
  canonical metric for analysis. History has its own explicitly selected surface.
- A recorded `global_best_*` metric keeps that meaning. It is not a final score.
  Last observed history also does not prove the requested endpoint completed.
- This is a snapshot reader, not a scheduler query or a saved-artifact audit.
  It reports durable journals that may contain newer state but does not replay
  them or reconcile retries. Old running rows can remain after replacement jobs.
- It does not infer split, label count, score direction, checkpoint-selection
  policy, or missing scores. Resolve those from the experiment protocol and,
  when needed, saved configs/manifests/logs. It never loads pickle checkpoints.
- Malformed YAML, unsafe/unknown tags, duplicate keys/row IDs, unmatched metric
  selectors, and ambiguous seed groups fail with a nonzero exit.

The discoverable [slurminator-results skill](../skills/slurminator-results/SKILL.md)
provides the agent workflow. Removing that skill link and these new query modules
rolls back the feature without changing ledgers or controller behavior.

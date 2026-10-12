---
name: slurminator-results
description: Inspect and summarize Slurminator experiment-state YAML with the reusable results CLI. Use for run progress, metric extraction, seed comparisons, or observed best/last histories, before writing a new YAML-analysis script.
---

# Slurminator results

Use the existing reader for ledger mechanics:

```bash
python -m slurminator.results inspect path/to/experiments.yaml
```

It accepts multiple explicit ledgers, reads production YAML tags/override syntax,
and reports compact JSON with source hashes and coverage. No scheduler, W&B,
checkpoint loading, or source writes occur. Do not read its implementation just
to run it. Use `--help` for a command's options and the
[query guide](../../docs/result_analysis.md) for less common cases.

For PMT/Olivia setup and metric/artifact conventions, read
[references/pmt.md](references/pmt.md). For another project, use its existing
Python environment with Slurminator available; do not create an environment just
to run this reader.

Start with the ledger identified by the request or the relevant tracker. If the
path is unknown, inspect the nearest campaign record or list filenames in the
known experiment directory. Inspect one candidate before expanding the search.
Avoid dumping full YAML or embedding hundreds of metrics in tool output.

Choose the query that answers the request:

- **Progress:** `inspect` gives persisted status/coverage. `rows --status running`
  or `rows --status failed --field output_dir` narrows follow-up to relevant jobs.
- **Metric discovery:** `inspect --metric 'probe/*' --limit 30`; refine the glob
  until the exact metric and split are clear. `--limit 0` is an explicit full export.
  Use `inspect --param 'model.*' --param 'loss.*'` to inspect a targeted set of
  overrides, including values that are constant across this ledger.
- **Per-run results:** `rows --metric 'EXACT_KEY' --field param:model.name`.
  `--field` also accepts nested row fields and exact `param:KEY` / `metric:KEY`.
- **Seed tables:** `summarize --metric 'EXACT_KEY' --group-by dataset_name
  --group-by param:model.name --expect-seeds 42,7,123 --require-complete`.
  Add all recipe and protocol dimensions that must remain separate. Use the
  actual expected seeds from the plan; the example seed set is not a default.
- **History:** `history --metric 'EXACT_KEY' --mode last`, or `--mode best
  --direction max` (or `min`). These operate within the latest recorded attempt;
  `--attempt N` selects another explicitly.

Keep metric keys, source files, split, selection rule, finite count, and seed
coverage visible in the answer. `--scale 100` explicitly converts fractional
metrics; do not assume all metrics are fractions. Summaries use sample SD and
retain per-seed values. Best/last/history and snapshot/global-best are different
sources; do not rename a dashboard best score as a final endpoint.

Missing metrics are missing evidence, including on completed jobs. Preserve
gaps and truncated-history flags. Duplicate seeds fail intentionally: use a
known retry mapping or additional grouping fields, never the best-scoring
attempt. Ledger status may lag the scheduler or a replacement ledger; report
the snapshot time and consult the established tracker when they disagree.

For artifact verification, protocol-specific derived scores, paired statistical
tests, or absent ledger metrics, reuse the campaign's existing audit/collector
and inspect only its referenced artifacts. Do not claim this reader verifies
checkpoint contents, logs, actual label counts, or raw predictions. If recurring
query mechanics are missing, extend the reusable reader with a regression test
instead of copying another campaign's parser. Scientific audits can remain
campaign-specific.

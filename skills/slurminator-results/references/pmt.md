# PMT and Olivia

From the PMT root, source `step_0.sh` as required by the repository. It exposes
the sibling Slurminator source path and the existing architecture-matched
container Python. Then use `python -m slurminator.results`; this module does not
launch the `slurminator` controller. No installation or scheduler request is
needed. Keep generated reports in ignored `reports/` or allocation-local scratch.
If host scratch is not mounted inside the container, use host shell redirection
(`python -m slurminator.results ... > "$TMPDIR/report.json"`) instead of passing
that host path to `--output`.

```bash
source step_0.sh
python -m slurminator.results inspect experiment_lists/experiments_20260919_030458.yaml \
  --metric 'probe/ppick/*/val_macro_pick_f1'
python -m slurminator.results summarize experiment_lists/experiments_20260919_030458.yaml \
  --metric 'probe/ppick/*/val_macro_pick_f1' \
  --group-by dataset_name --group-by param:model.backbone_name \
  --group-by param:loss.lambda_trd --expect-seeds 42,7,123 --require-complete --scale 100
```

That file is a reproducible six-run CREW example, not the default target for new
requests. Use the ledger in the user's current campaign record. In the related
18-row ledger, some completed checkpoint-probe jobs have no metric telemetry;
reuse its saved-artifact audit for those results instead of averaging only the
rows that happened to report metrics.

Useful PMT selectors:

- `param:model.backbone_name`, `param:loss.lambda_trd`, `metadata.config_profile`,
  `param:probe_parameters.eval_split`, `param:training_configs.max_train_steps`.
- `output_dir`, `job_id`, `wandb_run_url` identify logs/tracking. Recorded job
  logs are normally `output_dir/slurm-<job_id>.out` and `.err`.
- `param:paths.checkpoint_path` and `param:paths.output_path` locate saved
  manifests/configs. They can contain `${run_name}` and `${seed}`; resolve those
  against `metadata.run_name` and `seed`, never by evaluating shell expressions.

Probe proportions are part of the exact metric key: `lp0.00005`, `lp0.0001`,
`lp0.010`, etc. Never round them to a common label or infer actual trace counts
from that suffix alone. Keep `val`, `test`, and tuning/calibration roles separate.

The latest scalar dump, retained history subset, and dashboard columns have
different coverage; see Slurminator's `MetricLayout` and PMT's metric-layout
adapter if diagnosing missing telemetry. Do not assume a displayed shortform
is the canonical scientific metric. `global_best_*` retains its selection
meaning; final-versus-best claims need the campaign's endpoint/selection rule.

W&B lookups use the Python API and project scripts if separately needed, following
PMT's repository rules. Normal ledger inspection requires no network calls.

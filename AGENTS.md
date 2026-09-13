# Agent Instructions

## Experiment Config Context

When creating or materially revising experiment/sweep configs, populate the
optional `experiment_name` and `experiment_description` fields when the
evaluation intent is known. Use a short title of a few words that identifies
what is being evaluated. Expand it in one to three description sentences
covering the objective, main comparison, and relevant dataset/setup. Describe
the whole experiment list when it contains multiple sweeps or datasets.
Avoid opaque IDs, timestamps, parameter dumps, and invented context.

Put these fields at the root of a `--sweepfile` YAML. In a hand-authored
`--yaml` ledger, put them under the top-level `metadata` mapping. They are
display context; `experiment_prefix`, row IDs, and run names retain their
existing roles. See the [YAML guide](docs/sweep_yaml.md#experiment-title-and-description)
for examples, persistence, and dashboard behavior.

# Agent Instructions

## Experiment Results

For run progress, metric extraction, seed summaries, and observed best/last
histories, start with `python -m slurminator.results inspect <ledger.yaml>` and
the [results skill](skills/slurminator-results/SKILL.md) before writing another
YAML parser. The [query guide](docs/result_analysis.md) documents the available
commands. Keep protocol-specific artifact audits separate from ledger summaries.

## Agent Observation And Launch Approval

Use the [agent interface](docs/agent.md) for compact progress/status queries,
quiet completion waits, and resource proposals. Observation and preparation
must remain read-only; do not start a controller merely to inspect a ledger.

Before each live controller launch, deployment, or redeployment, present the
concrete ledger, cluster, GPU count per job, host RAM in GB, walltime, concurrency,
and aggregate resource request, then obtain explicit user approval. Estimate
resources from comparable measurements or documented assumptions; explain
headroom and uncertainty. Normally allow all eligible runs to fit within the
concurrency limit unless the user or a concrete resource constraint warrants a
smaller limit. Controller allocation resources are separate from training-job
requests.

Use `python -m slurminator.agent prepare` to bind the proposal to its input
ledger/configuration and retain the reasons and evidence. A phone connection,
permission to implement a feature, or approval of a previous launch does not
authorize another launch. Reprepare and request approval when the scope or
resources materially change. Follow the project's site-specific launcher and
sandbox restrictions after approval; never bypass them. The interface and
approval process do not themselves establish compute-node submission support.

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

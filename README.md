# slurminator

Reusable SLURM/HPC experiment orchestration package extracted from PMT.

This repository is being bootstrapped as part of PMT's orchestrator extraction
Phase 3. The first commits are intentionally mechanical so code movement and
behavior changes can be reviewed separately.

## Dashboard Quota Providers

The terminal dashboard can render cluster budget/quota information through
optional quota providers. Slurminator ships with an OLIVIA/Sigma2 provider and
custom providers can be registered for other clusters. See
`docs/quota_providers.md`.

## Project Plugins

Project-specific command building, validation, parser extensions, and tracker
integration can be provided through a plugin:

```bash
export SLURMINATOR_PLUGIN="my_project.orchestrator:MyOrchestratorPlugin"
slurminator --yaml experiment_lists/small.yaml --olivia-limit 1 --dry-run
```

The value may use `module:ClassName` or `module.ClassName` syntax. If unset,
Slurminator uses the generic default plugin; experiments must then provide an
explicit `extra_command`/`command` or use `--simple-command-entrypoint`.

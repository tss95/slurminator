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

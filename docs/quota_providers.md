# Quota Providers

Slurminator can show an HPC budget footer in the terminal dashboard. The
dashboard owns the generic rendering and the worst-case sweep-spend estimate;
cluster-specific quota probes live behind narrow quota providers.

The package ships with `OliviaQuotaProvider`, which parses Sigma2/OLIVIA
`sshare` output and adds the current Sigma2 allocation-period context. It is
registered by default for `HPCType.OLIVIA`.

## Provider Contract

A provider implements `slurminator.quota.QuotaProvider`:

```python
from datetime import date
from typing import Any

from slurminator.config import HPCType
from slurminator.quota import QuotaSnapshot


class MyQuotaProvider:
    hpc_type = HPCType.OLIVIA
    unavailable_hint = "quota command unavailable"

    def fetch_snapshot(self, *, account: str, connection_manager: Any) -> QuotaSnapshot | None:
        ...

    def period_bounds(self, *, today: date | None = None) -> tuple[date, date] | None:
        return None
```

`fetch_snapshot()` should return `None` when quota probing is unavailable. The
dashboard will keep running and render the quota as unavailable.

## Registering A Provider

Register a provider during your project bootstrap:

```python
from slurminator.quota import register_quota_provider

register_quota_provider(MyQuotaProvider())
```

The registry is keyed by `HPCType`, so a custom provider replaces the bundled
provider for that cluster.

## Worst-Case Spend

`QuotaSnapshot.worst_case_unit="gpu_hours"` tells the dashboard that it can
compare the active sweep's worst-case remaining GPU-hour spend against the
quota. Providers for CPU-hour, node-hour, or site-specific charge models should
set `worst_case_unit=None` until they provide a matching estimator.

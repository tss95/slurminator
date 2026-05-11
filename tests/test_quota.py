from datetime import date

import pytest

from slurminator.config import HPCType
from slurminator.quota import OliviaQuotaProvider, QuotaSnapshot, get_quota_provider

pytestmark = pytest.mark.unit


def test_olivia_provider_is_registered_by_default() -> None:
    provider = get_quota_provider(HPCType.OLIVIA)

    assert isinstance(provider, OliviaQuotaProvider)


def test_olivia_parse_sshare_output() -> None:
    raw = (
        "nn8104k||billing=60000,gres/gpu=9000000|"
        "cpu=58541523,mem=88443030459,energy=0,node=349293,billing=1042,"
        "fs/disk=0,vmem=0,pages=0,gres/gpu=1178031,gres/gpu:h200=1178031,gres/gpumem=0,gres/gpuutil=0\n"
        "nn8104k|tordss||cpu=536680,mem=10962942293,energy=0,node=67085,billing=0,"
        "fs/disk=0,vmem=0,pages=0,gres/gpu=134116,gres/gpu:h200=134116,gres/gpumem=0,gres/gpuutil=0\n"
    )

    parsed = OliviaQuotaProvider.parse_sshare_output(raw, "nn8104k", today=date(2026, 4, 28))

    assert isinstance(parsed, QuotaSnapshot)
    assert parsed.hpc_type == HPCType.OLIVIA
    assert parsed.limit == pytest.approx(150_000.0)
    assert parsed.used == pytest.approx(19_633.85)
    assert parsed.remaining == pytest.approx(130_366.15)
    assert parsed.period_start == date(2026, 4, 1)
    assert parsed.period_end == date(2026, 9, 30)


def test_olivia_sigma2_period_boundaries() -> None:
    assert OliviaQuotaProvider.allocation_period_end_date(date(2026, 4, 28)) == date(2026, 9, 30)
    assert OliviaQuotaProvider.allocation_period_end_date(date(2026, 10, 1)) == date(2027, 3, 31)
    assert OliviaQuotaProvider.allocation_period_end_date(date(2026, 3, 15)) == date(2026, 3, 31)


def test_olivia_sigma2_elapsed_pct_boundaries() -> None:
    assert OliviaQuotaProvider.allocation_period_elapsed_pct(date(2026, 4, 1)) == pytest.approx(0.0)
    assert OliviaQuotaProvider.allocation_period_elapsed_pct(date(2026, 9, 30)) == pytest.approx(100.0)
    assert OliviaQuotaProvider.allocation_period_elapsed_pct(date(2026, 4, 28)) == pytest.approx((27.0 / 182.0) * 100.0)

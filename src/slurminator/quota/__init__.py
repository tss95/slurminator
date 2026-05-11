"""Quota-provider extension points and bundled providers."""

from slurminator.quota.base import (
    QuotaProvider,
    QuotaSnapshot,
    clear_quota_providers,
    get_quota_provider,
    get_quota_providers,
    register_quota_provider,
)
from slurminator.quota.olivia import OliviaQuotaProvider

register_quota_provider(OliviaQuotaProvider())

__all__ = [
    "OliviaQuotaProvider",
    "QuotaProvider",
    "QuotaSnapshot",
    "clear_quota_providers",
    "get_quota_provider",
    "get_quota_providers",
    "register_quota_provider",
]

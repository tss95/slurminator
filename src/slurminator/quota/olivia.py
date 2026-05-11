"""Built-in OLIVIA quota provider."""

from __future__ import annotations

import logging
from datetime import date, datetime as dt
from typing import Any

from slurminator.config import HPCType
from slurminator.quota.base import QuotaSnapshot

logger = logging.getLogger("slurminator")


class OliviaQuotaProvider:
    """Fetch and parse Sigma2/OLIVIA GPU-hour quota from ``sshare``."""

    hpc_type = HPCType.OLIVIA
    cluster_name = "OLIVIA"
    resource_label = "GPU quota"
    unavailable_hint = "missing `sshare` in runtime?"

    def fetch_snapshot(self, *, account: str, connection_manager: Any) -> QuotaSnapshot | None:
        """Return the account-level OLIVIA GPU quota snapshot."""
        account = str(account or "").strip()
        if not account:
            return None
        cmd = f"sshare -P -h -A {account} -o Account,User,GrpTRESMins,GrpTRESRaw"
        try:
            stdout, stderr = connection_manager.run_command(self.hpc_type, cmd)
        except Exception as exc:  # pragma: no cover - defensive path
            logger.debug("Olivia quota probe failed: %s", exc)
            return None

        if stderr and stderr.strip():
            logger.debug("Olivia quota probe stderr: %s", stderr.strip())
        return self.parse_sshare_output(stdout, account)

    def period_bounds(self, *, today: date | None = None) -> tuple[date, date]:
        """Return the active Sigma2 allocation period."""
        return self.allocation_period_bounds(today)

    @classmethod
    def parse_sshare_output(cls, raw_output: str, account: str, *, today: date | None = None) -> QuotaSnapshot | None:
        """Parse an ``sshare -P`` response into a quota snapshot."""
        if not raw_output:
            return None
        account_norm = str(account or "").strip()
        if not account_norm:
            return None

        for raw_line in raw_output.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 4:
                continue
            account_field = parts[0].strip()
            user_field = parts[1].strip()
            if account_field != account_norm or user_field:
                continue
            limit_map = cls.parse_tres_map(parts[2])
            used_map = cls.parse_tres_map(parts[3])
            limit_minutes = cls.extract_gpu_tres_value(limit_map)
            used_minutes = cls.extract_gpu_tres_value(used_map)
            if limit_minutes is None or used_minutes is None or limit_minutes <= 0:
                continue
            period_start, period_end = cls.allocation_period_bounds(today)
            return QuotaSnapshot(
                hpc_type=cls.hpc_type,
                cluster_name=cls.cluster_name,
                resource_label=cls.resource_label,
                used=used_minutes / 60.0,
                limit=limit_minutes / 60.0,
                unit="h",
                period_start=period_start,
                period_end=period_end,
                worst_case_unit="gpu_hours",
            )
        return None

    @staticmethod
    def parse_tres_map(raw_tres: str) -> dict[str, float | None]:
        """Parse Slurm TRES CSV (``k=v,...``) into a key/value dict."""
        parsed: dict[str, float | None] = {}
        if not raw_tres:
            return parsed
        for token in raw_tres.split(","):
            token = token.strip()
            if not token or "=" not in token:
                continue
            key, value = token.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                continue
            if value.upper() == "N":
                parsed[key] = None
                continue
            try:
                parsed[key] = float(value)
            except ValueError:
                continue
        return parsed

    @staticmethod
    def extract_gpu_tres_value(tres_map: dict[str, float | None]) -> float | None:
        """Return the first concrete GPU TRES value from a parsed Slurm map."""
        preferred_keys = ("gres/gpu", "gres/gpu:h200")
        for key in preferred_keys:
            value = tres_map.get(key)
            if value is not None:
                return value
        for key, value in tres_map.items():
            if key.startswith("gres/gpu") and value is not None:
                return value
        return None

    @staticmethod
    def allocation_period_bounds(today: date | None = None) -> tuple[date, date]:
        """Return (start_date, end_date) for the active Sigma2 allocation period."""
        if today is None:
            today = dt.now().date()
        year = today.year
        month = today.month
        if 4 <= month <= 9:
            return (date(year, 4, 1), date(year, 9, 30))
        if month >= 10:
            return (date(year, 10, 1), date(year + 1, 3, 31))
        return (date(year - 1, 10, 1), date(year, 3, 31))

    @classmethod
    def allocation_period_end_date(cls, today: date | None = None) -> date:
        """Return Sigma2 allocation-period end date for ``today``."""
        _start, end = cls.allocation_period_bounds(today)
        return end

    @classmethod
    def allocation_period_elapsed_pct(cls, today: date | None = None) -> float:
        """Return elapsed percentage for the active Sigma2 allocation period."""
        if today is None:
            today = dt.now().date()
        start, end = cls.allocation_period_bounds(today)
        total_days = max((end - start).days, 1)
        elapsed_days = max(min((today - start).days, total_days), 0)
        return (elapsed_days / float(total_days)) * 100.0


__all__ = ["OliviaQuotaProvider"]

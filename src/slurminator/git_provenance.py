"""Capture git SHAs of relevant directories for ledger provenance."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger("slurminator")

_GIT_SHA_TIMEOUT_SECONDS = 5.0


def get_git_sha(directory: str | Path) -> str | None:
    """Return the git HEAD SHA for ``directory``, or ``None`` if unavailable."""
    try:
        path = Path(directory).resolve()
        if not path.exists():
            return None

        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=_GIT_SHA_TIMEOUT_SECONDS,
        )
        if result.returncode != 0:
            return None

        sha = result.stdout.strip()
        if len(sha) != 40 or not all(char in "0123456789abcdef" for char in sha):
            return None
        return sha
    except Exception as exc:
        logger.debug("Failed to read git SHA from %s: %s", directory, exc)
        return None


def get_slurminator_sha() -> str | None:
    """Return Slurminator's own git SHA by introspecting the package location."""
    here = Path(__file__).resolve().parent
    for parent in [here, *here.parents]:
        if (parent / ".git").exists():
            return get_git_sha(parent)
    return None


def get_project_sha(cwd: str | Path | None = None) -> str | None:
    """Return the project directory's git SHA, defaulting to the current working directory."""
    return get_git_sha(cwd or Path.cwd())


def capture_provenance(project_dir: str | Path | None = None) -> dict[str, str | None]:
    """Return provenance metadata suitable for experiment ledgers."""
    return {"project": get_project_sha(project_dir), "slurminator": get_slurminator_sha()}


__all__ = ["capture_provenance", "get_git_sha", "get_project_sha", "get_slurminator_sha"]

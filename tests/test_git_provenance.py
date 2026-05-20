from __future__ import annotations

import subprocess
import shutil
from types import SimpleNamespace

import pytest

from slurminator import git_provenance

pytestmark = pytest.mark.unit


def _require_git_available() -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is not available to Python subprocesses")


def _git(repo, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _init_git_repo(repo) -> str:
    _require_git_available()
    repo.mkdir()
    _git(repo, "init")
    (repo / "file.txt").write_text("content\n", encoding="utf-8")
    _git(repo, "add", "file.txt")
    _git(repo, "-c", "user.name=Slurminator Tests", "-c", "user.email=tests@example.com", "commit", "-m", "init")
    return _git(repo, "rev-parse", "HEAD")


def test_get_git_sha_returns_head_for_git_repo(tmp_path) -> None:
    expected_sha = _init_git_repo(tmp_path / "repo")

    assert git_provenance.get_git_sha(tmp_path / "repo") == expected_sha


def test_get_git_sha_accepts_valid_git_output(monkeypatch, tmp_path) -> None:
    expected_sha = "a" * 40

    def fake_run(args, **kwargs):
        assert args == ["git", "-C", str(tmp_path.resolve()), "rev-parse", "HEAD"]
        assert kwargs["timeout"] == 5.0
        return SimpleNamespace(returncode=0, stdout=f"{expected_sha}\n")

    monkeypatch.setattr(git_provenance.subprocess, "run", fake_run)

    assert git_provenance.get_git_sha(tmp_path) == expected_sha


def test_get_git_sha_returns_none_for_non_git_or_missing_directory(tmp_path) -> None:
    non_git = tmp_path / "non_git"
    non_git.mkdir()

    assert git_provenance.get_git_sha(non_git) is None
    assert git_provenance.get_git_sha(tmp_path / "missing") is None


def test_get_git_sha_returns_none_on_subprocess_errors(monkeypatch, tmp_path) -> None:
    def fail_run(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd="git", timeout=5.0)

    monkeypatch.setattr(git_provenance.subprocess, "run", fail_run)

    assert git_provenance.get_git_sha(tmp_path) is None


def test_capture_provenance_uses_project_directory_and_slurminator_sha(monkeypatch, tmp_path) -> None:
    project_sha = _init_git_repo(tmp_path / "project")
    slurminator_sha = "b" * 40
    monkeypatch.setattr(git_provenance, "get_slurminator_sha", lambda: slurminator_sha)

    assert git_provenance.capture_provenance(tmp_path / "project") == {
        "project": project_sha,
        "slurminator": slurminator_sha,
    }


def test_get_slurminator_sha_returns_none_when_package_is_not_in_git_checkout(monkeypatch, tmp_path) -> None:
    package_dir = tmp_path / "site-packages" / "slurminator"
    package_dir.mkdir(parents=True)
    module_file = package_dir / "git_provenance.py"
    module_file.write_text("# installed package copy\n", encoding="utf-8")
    monkeypatch.setattr(git_provenance, "__file__", str(module_file))

    assert git_provenance.get_slurminator_sha() is None

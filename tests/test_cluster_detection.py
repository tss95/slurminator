import socket

import pytest

from slurminator.config import HPCType, determine_current_hpc, is_current_hpc

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("hostname", "fqdn", "expected"),
    [
        ("foxlogin1", "foxlogin1.fox.educloud.no", HPCType.FOX),
        ("login-002", "login-002.lumi.csc.fi", HPCType.LUMI),
        ("nid001234", "nid001234", HPCType.LUMI),
        ("login-1", "login-1.saga.sigma2.no", HPCType.SAGA),
        ("uan02", "uan02.head.cm.americas.sgi.com", HPCType.OLIVIA),
        ("localhost", "localhost", None),
    ],
)
def test_determine_current_hpc_from_hostname(monkeypatch, hostname: str, fqdn: str, expected: HPCType | None) -> None:
    monkeypatch.delenv("CLUSTER", raising=False)
    monkeypatch.delenv("LUMI_STACK_NAME", raising=False)
    monkeypatch.delenv("SLURM_CLUSTER_NAME", raising=False)
    monkeypatch.setattr(socket, "gethostname", lambda: hostname)
    monkeypatch.setattr(socket, "getfqdn", lambda: fqdn)

    assert determine_current_hpc() == expected


def test_determine_current_hpc_prefers_explicit_cluster_env(monkeypatch) -> None:
    monkeypatch.setenv("CLUSTER", "olivia")
    monkeypatch.setattr(socket, "gethostname", lambda: "localhost")
    monkeypatch.setattr(socket, "getfqdn", lambda: "localhost")

    assert determine_current_hpc() == HPCType.OLIVIA
    assert is_current_hpc(HPCType.OLIVIA)


def test_determine_current_hpc_uses_slurm_cluster_name(monkeypatch) -> None:
    monkeypatch.delenv("CLUSTER", raising=False)
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "saga")
    monkeypatch.setattr(socket, "gethostname", lambda: "compute")
    monkeypatch.setattr(socket, "getfqdn", lambda: "compute")

    assert determine_current_hpc() == HPCType.SAGA

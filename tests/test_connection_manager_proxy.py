import sys
from types import ModuleType, SimpleNamespace

import pytest

pytestmark = pytest.mark.unit

# Minimal paramiko stub tailored for this test
_dummy = ModuleType("paramiko")


class DummyTransportJump:
    def open_channel(self, kind, dest, src):
        assert kind == "direct-tcpip"
        return object()  # dummy channel


class DummyJumpClient:
    def __init__(self):
        self._t = DummyTransportJump()

    def get_transport(self):
        return self._t


class RecordingSSHClient:
    def __init__(self):
        self.kwargs = None
        self._transport = None

    def set_missing_host_key_policy(self, _):
        pass

    def connect(self, **kwargs):
        # Record that we used the channel-based standard connect path
        self.kwargs = kwargs
        assert "sock" in kwargs, "expected to connect through tunnel using standard auth"

    def get_transport(self):
        return None


# attach stubs to dummy paramiko
_dummy.SSHClient = RecordingSSHClient
_dummy.AutoAddPolicy = lambda: object()


# If interactive path is used erroneously, constructing Transport should raise
def _forbid_transport(*args, **kwargs):
    raise AssertionError("Interactive Transport path should not be used for OLIVIA when two_factor=False")


_dummy.Transport = _forbid_transport
_dummy.Ed25519Key = type("Ed25519Key", (), {"from_private_key_file": staticmethod(lambda *a, **k: None)})
_dummy.RSAKey = type("RSAKey", (), {"from_private_key_file": staticmethod(lambda *a, **k: None)})
_dummy.SSHException = Exception
_dummy.ssh_exception = SimpleNamespace(PasswordRequiredException=Exception, SSHException=Exception)
sys.modules.setdefault("paramiko", _dummy)
sys.modules.setdefault("paramiko.ssh_exception", _dummy.ssh_exception)


def test_proxy_olivia_uses_standard_auth(monkeypatch):
    # Ensure the orchestrator module uses our stubbed paramiko even if it was
    # imported earlier in the test session (e.g., when running the full suite).
    import slurminator.connection_manager as hpc_conn

    monkeypatch.setattr(hpc_conn, "paramiko", _dummy, raising=False)

    # Import after patching the module-level reference
    from slurminator.config import HPCClusterConfig, HPCPartition, HPCType, HPC_CONFIGS
    from slurminator.connection_manager import HPCConnectionConfig, HPCConnectionManager

    # Ensure OLIVIA is configured without two_factor and with proxy_jump via SAGA
    olivia_cfg = HPCClusterConfig(
        cluster_type=HPCType.OLIVIA,
        partition=HPCPartition.ACCEL,
        account="demo_account",
        hostname="olivia.example.org",
        username="demo_user",
        repo_path=None,
        two_factor=False,
        proxy_jump="SAGA",
    )
    saga_cfg = HPCClusterConfig(
        cluster_type=HPCType.SAGA,
        partition=HPCPartition.A100,
        account="demo_account",
        hostname="saga.example.org",
        username="demo_user",
        repo_path=None,
        two_factor=True,
    )
    monkeypatch.setitem(HPC_CONFIGS, HPCType.OLIVIA, olivia_cfg)
    monkeypatch.setitem(HPC_CONFIGS, HPCType.SAGA, saga_cfg)

    # Manager with both configs; OLIVIA through SAGA
    mgr = HPCConnectionManager(
        configs={
            HPCType.OLIVIA: HPCConnectionConfig(
                hostname=olivia_cfg.hostname, username=olivia_cfg.username, proxy_jump="SAGA", two_factor=False
            ),
            HPCType.SAGA: HPCConnectionConfig(hostname=saga_cfg.hostname, username=saga_cfg.username, two_factor=True),
        }
    )

    # Pretend we are not running on either cluster
    monkeypatch.setattr(mgr, "is_local_hpc", lambda *_: False)

    # Pretend SAGA is already connected and provides a transport with open_channel
    mgr._connected[HPCType.SAGA] = True
    mgr._clients[HPCType.SAGA] = DummyJumpClient()

    # Avoid a blocking password prompt if standard path falls back to password
    monkeypatch.setattr("slurminator.connection_manager._safe_getpass", lambda p: "pw")

    # Should not raise; and must not invoke paramiko.Transport constructor (assertion)
    mgr.connect(HPCType.OLIVIA)


def test_direct_connection_keyboard_interactive_fallback(monkeypatch):
    # Ensure we fall back to keyboard-interactive when the server only allows it
    from slurminator.connection_manager import HPCConnectionManager, HPCConnectionConfig, BadAuthenticationType

    called = {"kbd": False}

    def fake_interactive(self, client, cfg):
        called["kbd"] = True

    # Replace interactive auth with a cheap stub so the test remains fast/non-blocking
    monkeypatch.setattr(HPCConnectionManager, "_interactive_auth", fake_interactive, raising=False)

    class DummySSHClient:
        def set_missing_host_key_policy(self, _):
            pass

        def connect(self, **kwargs):
            raise BadAuthenticationType(["keyboard-interactive"], "keyboard-interactive required")

        def close(self):
            pass

    mgr = HPCConnectionManager(configs={})
    cfg = HPCConnectionConfig(hostname="olivia.example.org", username="demo_user", two_factor=False, use_key=False)

    mgr._standard_auth(DummySSHClient(), cfg)

    assert called["kbd"] is True

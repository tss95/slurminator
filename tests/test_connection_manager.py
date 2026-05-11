# ruff: noqa: E402
import builtins
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

pytestmark = pytest.mark.unit

# Provide a dummy paramiko module so the import in hpc_connection succeeds
_dummy = ModuleType("paramiko")
_dummy.SSHClient = type("SSHClient", (), {})
_dummy.AutoAddPolicy = type("AutoAddPolicy", (), {})
_dummy.Transport = type("Transport", (), {})
_dummy.Ed25519Key = type("Ed25519Key", (), {"from_private_key_file": lambda *a, **k: None})
_dummy.RSAKey = type("RSAKey", (), {"from_private_key_file": lambda *a, **k: None})
_dummy.SSHException = Exception
_dummy.ssh_exception = SimpleNamespace(PasswordRequiredException=Exception, SSHException=Exception)
sys.modules.setdefault("paramiko", _dummy)
sys.modules.setdefault("paramiko.ssh_exception", _dummy.ssh_exception)

from slurminator.connection_manager import (
    HPCConnectionConfig,
    HPCConnectionManager,
    HPCType,
    _env,
    _env_prefixes,
    _safe_getpass,
    _safe_input,
    UserCancelledError,
)


@pytest.fixture
def minimal_manager():
    cfg = HPCConnectionConfig(hostname="local", username="user")
    return HPCConnectionManager({HPCType.FOX: cfg})


def test_env_prefixes_can_be_extended_from_environment(monkeypatch):
    monkeypatch.setenv("SLURMINATOR_ENV_PREFIXES", "PMT,SLURMINATOR")
    monkeypatch.setenv("PMT_SSH_CONNECT_TIMEOUT_FOX", "44")
    monkeypatch.setenv("SLURMINATOR_SSH_CONNECT_TIMEOUT_FOX", "55")

    assert _env_prefixes() == ("PMT", "SLURMINATOR")
    assert _env("SSH_CONNECT_TIMEOUT", hpc_name="FOX") == "44"


def test_env_prefixes_keep_slurminator_fallback(monkeypatch):
    monkeypatch.setenv("SLURMINATOR_ENV_PREFIXES", "PMT")
    monkeypatch.setenv("SLURMINATOR_SSH_CONNECT_TIMEOUT_FOX", "55")

    assert _env_prefixes() == ("PMT", "SLURMINATOR")
    assert _env("SSH_CONNECT_TIMEOUT", hpc_name="FOX") == "55"


def test_is_local_hpc_detects(monkeypatch, minimal_manager):
    # Ensure env-based overrides don't interfere with hostname-based detection
    monkeypatch.setenv("CLUSTER", "", prepend=False)
    monkeypatch.setattr("socket.gethostname", lambda: "foxlogin01")
    monkeypatch.setattr("socket.getfqdn", lambda: "foxlogin01.fox.educloud.no")
    assert minimal_manager.is_local_hpc(HPCType.FOX)
    assert not minimal_manager.is_local_hpc(HPCType.LUMI)


def test_run_command_local(monkeypatch, minimal_manager):
    monkeypatch.setattr(minimal_manager, "is_local_hpc", lambda *_: True)

    def fake_run(cmd, shell, capture_output, text):
        assert cmd == "echo hi"
        return SimpleNamespace(stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    out, err = minimal_manager.run_command(HPCType.FOX, "echo hi")
    assert out == "ok"
    assert err == ""


def test_run_command_prefer_remote_local_non_slurm_falls_back_local(monkeypatch, minimal_manager):
    monkeypatch.setattr(minimal_manager, "is_local_hpc", lambda *_: True)

    def fake_run(cmd, shell, capture_output, text):
        assert cmd == "echo hi"
        return SimpleNamespace(stdout="ok-remote-fallback", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    out, err = minimal_manager.run_command(HPCType.FOX, "echo hi", prefer_remote=True)
    assert out == "ok-remote-fallback"
    assert err == ""


def test_run_command_prefer_remote_skips_local_for_missing_slurm(monkeypatch, minimal_manager):
    monkeypatch.setattr(minimal_manager, "is_local_hpc", lambda *_: True)
    monkeypatch.setattr(
        "slurminator.connection_manager.shutil.which",
        lambda binary: None if binary == "sbatch" else f"/usr/bin/{binary}",
    )

    def fail_local_run(*_args, **_kwargs):
        raise AssertionError("local subprocess fallback should not run when required Slurm binary is missing")

    monkeypatch.setattr(subprocess, "run", fail_local_run)

    client = DummySSHClient()
    client.exec_responses = ["Submitted batch job 42"]
    calls = []

    def fake_connect(ht, force_remote=False):
        calls.append((ht, force_remote))
        minimal_manager._clients[ht] = client
        minimal_manager._connected[ht] = True

    monkeypatch.setattr(minimal_manager, "connect", fake_connect)

    out, err = minimal_manager.run_command(HPCType.FOX, "sbatch --version", prefer_remote=True)
    assert out == "Submitted batch job 42"
    assert err == ""
    assert calls == [(HPCType.FOX, True)]
    assert client.exec_commands == ["sbatch --version"]


def test_run_command_prefer_remote_localhost_submission_uses_local_when_slurm_present(monkeypatch):
    cfg = HPCConnectionConfig(hostname="h", username="u", submission_host="localhost")
    mgr = HPCConnectionManager({HPCType.FOX: cfg})
    monkeypatch.setattr(mgr, "is_local_hpc", lambda *_: True)
    monkeypatch.setattr("slurminator.connection_manager.shutil.which", lambda _binary: "/usr/bin/sbatch")

    local_calls = []

    def fake_run(cmd, shell, capture_output, text):  # noqa: ARG001
        local_calls.append(cmd)
        return SimpleNamespace(stdout="local-sbatch-ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        mgr,
        "_connect_submission",
        lambda _h: (_ for _ in ()).throw(
            AssertionError("submission SSH should not be used for localhost with local Slurm")
        ),
    )

    out, err = mgr.run_command(HPCType.FOX, "sbatch --version", prefer_remote=True)
    assert out == "local-sbatch-ok"
    assert err == ""
    assert local_calls == ["sbatch --version"]


def test_run_command_prefer_remote_localhost_submission_raises_when_slurm_missing(monkeypatch):
    cfg = HPCConnectionConfig(hostname="h", username="u", submission_host="localhost")
    mgr = HPCConnectionManager({HPCType.FOX: cfg})
    monkeypatch.setattr(mgr, "is_local_hpc", lambda *_: True)
    monkeypatch.setattr("slurminator.connection_manager.shutil.which", lambda _binary: None)
    with pytest.raises(RuntimeError, match="Refusing self-SSH"):
        mgr.run_command(HPCType.FOX, "sbatch --version", prefer_remote=True)


def test_run_command_prefer_remote_localhost_submission_uses_submission_with_env_override(monkeypatch):
    cfg = HPCConnectionConfig(hostname="h", username="u", submission_host="localhost")
    mgr = HPCConnectionManager({HPCType.FOX: cfg})
    monkeypatch.setattr(mgr, "is_local_hpc", lambda *_: True)
    monkeypatch.setattr("slurminator.connection_manager.shutil.which", lambda _binary: None)
    monkeypatch.setenv("SLURMINATOR_ALLOW_SELF_SSH_LOCAL", "1")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("local subprocess should not run when Slurm is missing")
        ),
    )

    submission_client = DummySSHClient()
    submission_client.exec_responses = ["remote-sbatch-ok"]

    def fake_connect_submission(_h):
        mgr._sub_clients[HPCType.FOX] = submission_client
        mgr._sub_connected[HPCType.FOX] = True

    monkeypatch.setattr(mgr, "_connect_submission", fake_connect_submission)

    out, err = mgr.run_command(HPCType.FOX, "sbatch --version", prefer_remote=True)
    assert out == "remote-sbatch-ok"
    assert err == ""
    assert submission_client.exec_commands == ["sbatch --version"]


def test_safe_input_keyboard_interrupt(monkeypatch):
    monkeypatch.setattr(builtins, "input", lambda prompt: (_ for _ in ()).throw(KeyboardInterrupt))
    with pytest.raises(UserCancelledError):
        _safe_input("foo: ")


def test_safe_getpass_keyboard_interrupt(monkeypatch):
    import getpass

    monkeypatch.setattr(getpass, "getpass", lambda prompt: (_ for _ in ()).throw(KeyboardInterrupt))
    with pytest.raises(UserCancelledError):
        _safe_getpass("pw: ")


class DummyStream:
    def __init__(self, data: str = ""):
        self._data = data.encode()

    def read(self):
        return self._data


class DummyTransport:
    def __init__(self):
        self.interval = None

    def set_keepalive(self, interval: int):
        self.interval = interval


class DummySSHClient:
    def __init__(self):
        self.transport = DummyTransport()
        self.exec_commands = []
        self.exec_responses = []

    def set_missing_host_key_policy(self, _):
        pass

    def connect(self, **_):
        self.connected = True

    def get_transport(self):
        return self.transport

    def exec_command(self, cmd: str):
        self.exec_commands.append(cmd)
        resp = self.exec_responses.pop(0) if self.exec_responses else ""
        if isinstance(resp, Exception):
            raise resp
        return None, DummyStream(resp), DummyStream()

    def close(self):
        self.closed = True


def test_connect_force_remote_upgrades_local_marker(monkeypatch):
    from slurminator.config import HPC_CONFIGS, HPCClusterConfig, HPCPartition

    cluster = HPCClusterConfig(
        cluster_type=HPCType.FOX, partition=HPCPartition.ACCEL, account="a", hostname="h", username="u", repo_path=None
    )
    monkeypatch.setitem(HPC_CONFIGS, HPCType.FOX, cluster)

    dummy = DummySSHClient()
    monkeypatch.setattr(sys.modules["paramiko"], "SSHClient", lambda: dummy)
    monkeypatch.setattr(sys.modules["paramiko"], "AutoAddPolicy", lambda: object())

    cfg = HPCConnectionConfig(hostname="h", username="u", two_factor=False, use_key=False)
    mgr = HPCConnectionManager({HPCType.FOX: cfg})
    monkeypatch.setattr(mgr, "is_local_hpc", lambda *_: True)
    monkeypatch.setattr(mgr, "_standard_auth", lambda *_args, **_kwargs: None)

    mgr.connect(HPCType.FOX, force_remote=False)
    assert mgr._connected[HPCType.FOX] is True
    assert mgr._clients[HPCType.FOX] is None

    mgr.connect(HPCType.FOX, force_remote=True)
    assert mgr._clients[HPCType.FOX] is dummy
    assert mgr._connected[HPCType.FOX] is True


def test_connect_remote(monkeypatch):
    from slurminator.config import HPC_CONFIGS, HPCClusterConfig, HPCPartition

    dummy = DummySSHClient()
    monkeypatch.setattr(sys.modules["paramiko"], "SSHClient", lambda: dummy)
    monkeypatch.setattr(sys.modules["paramiko"], "AutoAddPolicy", lambda: object())
    monkeypatch.setattr("slurminator.connection_manager._safe_getpass", lambda p: "pw")

    cluster = HPCClusterConfig(
        cluster_type=HPCType.FOX,
        partition=HPCPartition.ACCEL,
        account="a",
        hostname="h",
        username="u",
        repo_path="/repo",
    )
    monkeypatch.setitem(HPC_CONFIGS, HPCType.FOX, cluster)

    cfg = HPCConnectionConfig(hostname="h", username="u", keep_alive=True, keep_alive_interval=10)
    mgr = HPCConnectionManager({HPCType.FOX: cfg})
    monkeypatch.setattr(mgr, "is_local_hpc", lambda *_: False)

    cmds = []

    def fake_run(ht, cmd):
        cmds.append(cmd)
        return "", ""

    monkeypatch.setattr(mgr, "run_command", fake_run)

    mgr.connect(HPCType.FOX)
    assert mgr._connected[HPCType.FOX]
    assert cmds == ["cd /repo"]
    assert dummy.transport.interval == 10


def test_run_command_reconnect(monkeypatch):
    client = DummySSHClient()
    client.exec_responses = [Exception("fail"), "ok"]

    cfg = HPCConnectionConfig(hostname="h", username="u")
    mgr = HPCConnectionManager({HPCType.FOX: cfg})
    mgr._clients[HPCType.FOX] = client
    mgr._connected[HPCType.FOX] = True

    monkeypatch.setattr(mgr, "is_local_hpc", lambda *_: False)

    called = []

    def fake_connect(ht):
        called.append(ht)
        mgr._clients[ht] = client
        mgr._connected[ht] = True

    monkeypatch.setattr(mgr, "connect", fake_connect)
    monkeypatch.setattr(mgr, "close", lambda *_: None)

    out, err = mgr.run_command(HPCType.FOX, "echo")
    assert called == [HPCType.FOX]
    assert out == "ok"
    assert client.exec_commands == ["echo", "echo"]


def test_interactive_auth_retries_with_pam_after_default_failure(monkeypatch):
    from slurminator.connection_manager import HPCConnectionManager, HPCConnectionConfig

    calls = []
    prompt_calls = []

    class DummyInteractiveTransport:
        def __init__(self, *_args, **_kwargs):
            self.closed = False
            self.banner_timeout = None
            self.auth_timeout = None

        def set_keepalive(self, _interval):
            return None

        def start_client(self, timeout=None):  # noqa: ARG002
            return None

        def auth_interactive(self, _username, handler, submethods=None):
            calls.append(submethods)
            # Exercise prompt handler path
            prompt_calls.append(handler("", "", [("One-time password (OATH) for `demo_user':", False)]))
            if submethods is None:
                raise Exception("default interactive failed")
            return None

        def close(self):
            self.closed = True

    monkeypatch.setattr("slurminator.connection_manager.paramiko.Transport", DummyInteractiveTransport)
    monkeypatch.setattr("slurminator.connection_manager._safe_input", lambda _p: "123456")
    monkeypatch.setattr("slurminator.connection_manager._safe_getpass", lambda _p: "secret")

    mgr = HPCConnectionManager({HPCType.FOX: HPCConnectionConfig(hostname="h", username="u")})
    client = DummySSHClient()

    mgr._interactive_auth(client, HPCConnectionConfig(hostname="h", username="u", keep_alive_interval=15))

    assert calls == [None, "pam"]
    assert prompt_calls == [["123456"], ["123456"]]
    assert client._transport is not None

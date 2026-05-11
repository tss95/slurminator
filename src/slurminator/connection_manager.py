"""SSH and local command execution helpers for Slurminator clusters."""

from __future__ import annotations

import logging
import os
import re
import shlex
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

try:
    import paramiko  # type: ignore
except Exception:  # pragma: no cover - allow import in minimal test environments without paramiko
    paramiko = None  # type: ignore
# Be resilient to minimal paramiko stubs used in unit tests that may not expose
# AuthenticationException. Fall back to defining it as a subclass of SSHException.
try:
    from paramiko.ssh_exception import (
        PasswordRequiredException,
        SSHException,
        AuthenticationException,
        BadAuthenticationType,
    )
except Exception:  # pragma: no cover - exercised in tests with stubs
    from paramiko.ssh_exception import PasswordRequiredException, SSHException  # type: ignore

    class AuthenticationException(SSHException):  # type: ignore
        pass

    class BadAuthenticationType(SSHException):  # type: ignore
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.allowed_types = []


from slurminator.config import HPC_CONFIGS, HPCType, is_current_hpc

logger = logging.getLogger("slurminator")

# Compatibility wrappers may prepend project-specific prefixes. Users can also
# set SLURMINATOR_ENV_PREFIXES=PROJECT,SLURMINATOR to opt into project-local
# environment-variable names without wrapping this module.
ENV_PREFIXES: tuple[str, ...] = ("SLURMINATOR",)


def _env_prefixes() -> tuple[str, ...]:
    """Return environment-variable prefixes in lookup order."""
    raw = os.getenv("SLURMINATOR_ENV_PREFIXES", "")
    if not raw.strip():
        return ENV_PREFIXES

    prefixes: list[str] = []
    for item in raw.split(","):
        prefix = item.strip().upper()
        if prefix and prefix not in prefixes:
            prefixes.append(prefix)

    for prefix in ENV_PREFIXES:
        if prefix not in prefixes:
            prefixes.append(prefix)

    return tuple(prefixes) or ENV_PREFIXES


def _env(name: str, default: str | None = None, *, hpc_name: str | None = None) -> str | None:
    """Return the first matching prefixed environment variable."""
    suffix = f"{name}_{hpc_name}" if hpc_name else name
    for prefix in _env_prefixes():
        value = os.getenv(f"{prefix}_{suffix}")
        if value is not None:
            return value
    return default


def _int_env(name: str, default: int, *, hpc_name: str | None = None) -> int:
    """Return a prefixed integer environment variable or ``default``."""
    try:
        return int(_env(name, str(default), hpc_name=hpc_name) or str(default))
    except Exception:
        return default


def _truthy_env(name: str, default: str = "0", *, hpc_name: str | None = None) -> bool:
    """Return True for common truthy environment values."""
    return str(_env(name, default, hpc_name=hpc_name)).lower() in {"1", "true", "yes"}


def _ssh_connect_timeout() -> int:
    return _int_env("SSH_CONNECT_TIMEOUT", 30)


def _ssh_tunnel_timeout() -> int:
    return _int_env("SSH_TUNNEL_TIMEOUT", 15)


def _ssh_banner_timeout() -> int:
    return _int_env("SSH_BANNER_TIMEOUT", 240)


def _ssh_auth_timeout() -> int:
    return _int_env("SSH_AUTH_TIMEOUT", 240)


def _ssh_proxy_retries() -> int:
    return _int_env("SSH_PROXY_RETRIES", 3)


def _ssh_proxy_backoff_base() -> int:
    return _int_env("SSH_PROXY_BACKOFF_BASE", 2)


# Reuse user's OpenSSH ControlMaster when possible to avoid repeated OTP/2FA on
# jump hosts. We detect master state via `ssh -O check <alias>`.
def _openssh_master_running(alias: str, timeout: int = 2) -> bool:
    try:
        proc = subprocess.run(
            ["ssh", "-O", "check", alias], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout
        )
        return proc.returncode == 0
    except Exception:
        return False


_SLURM_BINARIES = ("sbatch", "squeue", "sacct", "scancel", "scontrol", "sinfo", "srun")
_SLURM_BINARY_PATTERNS = {
    binary: re.compile(rf"(?<![A-Za-z0-9_./-]){re.escape(binary)}(?![A-Za-z0-9_./-])") for binary in _SLURM_BINARIES
}


def _missing_local_slurm_binaries(command: str) -> list[str]:
    """Return Slurm binaries referenced by `command` that are absent from local PATH."""
    missing: list[str] = []
    for binary, pattern in _SLURM_BINARY_PATTERNS.items():
        if pattern.search(command) and shutil.which(binary) is None:
            missing.append(binary)
    return missing


def _is_loopback_or_current_host(host: str) -> bool:
    """Return True when `host` targets this machine (loopback/current hostname)."""
    host_norm = (host or "").strip().lower()
    if host_norm in {"localhost", "127.0.0.1", "::1"}:
        return True

    try:
        current_short = socket.gethostname().strip().lower()
    except Exception:
        current_short = ""
    try:
        current_fqdn = socket.getfqdn().strip().lower()
    except Exception:
        current_fqdn = ""

    host_short = host_norm.split(".", 1)[0]
    current_short_only = current_short.split(".", 1)[0] if current_short else ""
    current_fqdn_short = current_fqdn.split(".", 1)[0] if current_fqdn else ""

    candidates = {current_short, current_fqdn, current_short_only, current_fqdn_short}
    return host_norm in candidates or host_short in candidates


class UserCancelledError(Exception):
    """Raised when user presses Ctrl+C during input or authentication."""


@dataclass
class HPCConnectionConfig:
    """
    Internal paramiko connection config derived from HPCClusterConfig.
    This class is specialized for paramiko usage (port, key usage, 2FA, etc.).
    """

    hostname: str
    username: str
    port: int = 22
    use_key: bool = False
    key_path: Optional[str] = None
    two_factor: bool = False
    keep_alive: bool = True
    keep_alive_interval: int = 30
    proxy_jump: Optional[str] = None  # Name of HPC to use as jump host
    proxy_jump_username: Optional[str] = None
    proxy_jump_port: int = 22
    # Optional OpenSSH alias for the jump host (e.g., "saga"). When present and
    # a ControlMaster is active, we prefer ProxyCommand (ssh -O check <alias>)
    # to reuse your existing SAGA session.
    proxy_jump_alias: Optional[str] = None

    # Optional alternate submission endpoint on same HPC
    submission_host: Optional[str] = None
    submission_username: Optional[str] = None
    submission_port: Optional[int] = None
    submission_use_key: Optional[bool] = None
    submission_key_path: Optional[str] = None
    submission_two_factor: Optional[bool] = None

    def __post_init__(self):
        # Default key path if use_key is True but none provided
        if self.use_key and not self.key_path:
            self.key_path = os.path.expanduser("~/.ssh/id_rsa")


class HPCConnectionManager:
    """Manages SSH connections to multiple HPCs."""

    def __init__(self, configs: Dict[HPCType, HPCConnectionConfig]):
        self.configs = configs
        self._clients = {}  # Dict[HPCType, paramiko.SSHClient]
        self._connected = {}  # Dict[HPCType, bool]
        # Dedicated connections to alternate submission endpoints
        self._sub_clients = {}  # Dict[HPCType, paramiko.SSHClient]
        self._sub_connected = {}  # Dict[HPCType, bool]
        self._failed_connections = set()  # Track HPCs that failed to connect

        # Initialize connection status
        for hpc_type in self.configs.keys():
            self._connected[hpc_type] = False
            self._clients[hpc_type] = None
            self._sub_connected[hpc_type] = False
            self._sub_clients[hpc_type] = None

    def connect_all(self) -> None:
        """Connect to all HPCs in self.configs."""
        logger.info(f"Connecting to HPCs: {[hpc.name for hpc in self.configs.keys()]}")
        for hpc_type in self.configs.keys():
            try:
                self.connect(hpc_type)
            except UserCancelledError:
                logger.info("User cancelled authentication process.")
                break
            except Exception as e:
                logger.error(f"Connection to {hpc_type.name} failed: {e}")
                # Keep trying other clusters even if one fails
                continue

    def is_local_hpc(self, hpc_type: HPCType) -> bool:
        """Return True if running on the given HPC type."""
        return is_current_hpc(hpc_type)

    def connect(self, hpc_type: HPCType, force_remote: bool = False) -> None:
        """Connect to a single HPC."""
        # If we've already established a connection in this session, avoid
        # re-connecting (which could trigger an unnecessary 2FA/password prompt).
        # Note: when force_remote=True, we intentionally bypass the local short-circuit
        # and establish a real SSH connection even if we're currently on that HPC.
        if self._connected.get(hpc_type, False):
            has_remote_client = self._clients.get(hpc_type) is not None
            # If a remote session is already established, never re-prompt just because
            # force_remote=True. `force_remote` only bypasses the local short‑circuit
            # (when we're physically on that HPC) – it should not tear down and
            # recreate a live SSH session.
            if has_remote_client or not force_remote:
                logger.debug(
                    "Already connected to %s; skipping reconnect (force_remote=%s).", hpc_type.name, force_remote
                )
                return
            # Local marker without an SSH client + force_remote=True => upgrade to
            # a real SSH session now.
            logger.debug(
                "Upgrading local connection marker to remote SSH session for %s (force_remote=True).", hpc_type.name
            )

        if self.is_local_hpc(hpc_type) and not force_remote:
            logger.info(f"Running locally on {hpc_type.name}")
            # repo_path = HPC_CONFIGS[hpc_type].repo_path
            # logger.debug(f"Changing to repo directory: {repo_path}")
            # os.chdir(repo_path)
            self._connected[hpc_type] = True
            return

        cfg = self.configs[hpc_type]

        # Handle proxy jump if specified
        if cfg.proxy_jump:
            logger.info(f"Connecting to {hpc_type.name} via jump host {cfg.proxy_jump}")
            self._connect_via_proxy(hpc_type, cfg)
            return

        logger.info(f"Connecting to {hpc_type.name} ({cfg.hostname})")
        logger.info("If prompted, enter your password/2FA for %s; prompts follow below.", hpc_type.name)
        logger.debug(f"Connection details: username={cfg.username}, port={cfg.port}, use_key={cfg.use_key}")

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        try:
            if cfg.two_factor:
                logger.debug("Using interactive 2FA authentication")
                self._interactive_auth(client, cfg)
            else:
                logger.debug("Using standard authentication")
                self._standard_auth(client, cfg)

            self._clients[hpc_type] = client
            self._connected[hpc_type] = True

            transport = client.get_transport()
            if transport and cfg.keep_alive:
                transport.set_keepalive(cfg.keep_alive_interval)
                logger.debug(f"Set keepalive to {cfg.keep_alive_interval}s")

            # Switch to the HPC's repository directory
            repo_path = HPC_CONFIGS[hpc_type].repo_path
            if repo_path:
                out, err = self.run_command(hpc_type, f"cd {repo_path}")
                if err.strip():
                    # Non-fatal – log & continue, preserving the live SSH
                    # session so the user is **not** prompted for credentials
                    # a second time when the orchestrator reconnects.
                    logger.warning(
                        "Could not change to repo directory '%s' on %s: %s – continuing without chdir.",
                        repo_path,
                        hpc_type.name,
                        err.strip(),
                    )

            logger.info(f"Successfully connected to {hpc_type.name}")

        except AuthenticationException as e:
            # Authentication failures are recoverable (e.g., OTP typos). Do not
            # mark the host as permanently failed; just propagate so the caller
            # can re-prompt on next attempt.
            logger.error(f"Connection failed: AuthenticationException: {e}")
            self._connected[hpc_type] = False
            self._clients[hpc_type] = None
            raise
        except Exception as e:
            logger.error(f"Connection failed: {type(e).__name__}: {e}")
            self._connected[hpc_type] = False
            self._clients[hpc_type] = None
            self._failed_connections.add(hpc_type)  # Mark as failed
            raise

    def _connect_via_proxy(self, hpc_type: HPCType, cfg: HPCConnectionConfig) -> None:
        """Connect to HPC via a proxy jump host."""
        # First ensure jump host is connected
        jump_hpc_type = HPCType[cfg.proxy_jump.upper()]

        # If jump host had a non-auth failure earlier, skip (auth failures are retriable)
        if jump_hpc_type in self._failed_connections:
            logger.warning(
                f"Skipping connection to {hpc_type.name} because jump host {jump_hpc_type.name} failed to connect"
            )
            self._failed_connections.add(hpc_type)
            raise ConnectionError(f"Cannot connect to {hpc_type.name}: jump host {jump_hpc_type.name} is not available")

        # Ensure jump host session exists
        if not self._connected.get(jump_hpc_type, False):
            logger.info(f"Connecting to jump host {jump_hpc_type.name} first...")
            self.connect(jump_hpc_type)

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # Optional: prefer OpenSSH ProxyCommand via ControlMaster when alias is configured
        proxy_alias = cfg.proxy_jump_alias or _env("PROXYJUMP_ALIAS", hpc_name=hpc_type.name)
        use_proxycommand_env = str(_env("USE_PROXYCOMMAND", "auto")).lower()
        try_proxycommand = False
        if proxy_alias:
            if use_proxycommand_env in {"1", "true", "yes", "force"}:
                try_proxycommand = True
            elif use_proxycommand_env in {"0", "false", "no"}:
                try_proxycommand = False
            else:
                try_proxycommand = _openssh_master_running(proxy_alias)

        try:
            if try_proxycommand:
                try:
                    from paramiko.proxy import ProxyCommand  # type: ignore
                except Exception:
                    ProxyCommand = None  # type: ignore
                if ProxyCommand is not None:
                    logger.info("Using OpenSSH ProxyCommand via alias '%s' to reach %s", proxy_alias, hpc_type.name)
                    # Allow overriding destination for the inner hop
                    _dest_host = _env("TUNNEL_HOST", hpc_name=hpc_type.name) or cfg.hostname
                    _dest_port = _int_env("TUNNEL_PORT", cfg.port, hpc_name=hpc_type.name)
                    proxy_cmd = f"ssh -q -W {_dest_host}:{_dest_port} {proxy_alias}"
                    # Try agent/default keys first, then password
                    try:
                        client.connect(
                            hostname=_dest_host,
                            port=_dest_port,
                            username=cfg.username,
                            sock=ProxyCommand(proxy_cmd),
                            look_for_keys=True,
                            allow_agent=True,
                            timeout=_ssh_connect_timeout(),
                            banner_timeout=_ssh_banner_timeout(),
                            auth_timeout=_ssh_auth_timeout(),
                        )
                    except SSHException:
                        if cfg.use_key and cfg.key_path:
                            key_path = os.path.expanduser(cfg.key_path)
                            try:
                                pkey = paramiko.Ed25519Key.from_private_key_file(key_path)
                            except Exception:
                                try:
                                    pkey = paramiko.RSAKey.from_private_key_file(key_path)
                                except PasswordRequiredException:
                                    passphrase = _safe_getpass(f"Enter passphrase for key '{key_path}': ")
                                    pkey = paramiko.RSAKey.from_private_key_file(key_path, password=passphrase)
                            client.connect(
                                hostname=_dest_host,
                                port=_dest_port,
                                username=cfg.username,
                                pkey=pkey,
                                sock=ProxyCommand(proxy_cmd),
                                look_for_keys=False,
                                allow_agent=False,
                                timeout=_ssh_connect_timeout(),
                                banner_timeout=_ssh_banner_timeout(),
                                auth_timeout=_ssh_auth_timeout(),
                            )
                        else:
                            password = _safe_getpass(f"Enter password for {cfg.username}@{cfg.hostname}: ")
                            client.connect(
                                hostname=_dest_host,
                                port=_dest_port,
                                username=cfg.username,
                                password=password,
                                sock=ProxyCommand(proxy_cmd),
                                look_for_keys=False,
                                allow_agent=False,
                                timeout=_ssh_connect_timeout(),
                                banner_timeout=_ssh_banner_timeout(),
                                auth_timeout=_ssh_auth_timeout(),
                            )

                    self._clients[hpc_type] = client
                    self._connected[hpc_type] = True
                    transport = client.get_transport()
                    if transport and cfg.keep_alive:
                        transport.set_keepalive(cfg.keep_alive_interval)
                        logger.debug(f"Set keepalive to {cfg.keep_alive_interval}s")
                    logger.info("Successfully connected to %s via ProxyCommand (alias %s)", hpc_type.name, proxy_alias)
                    return

            # Helper: try an explicit ProxyCommand using jump host hostname/port
            def _try_proxycommand_direct() -> bool:
                """Attempt connection via an explicit ProxyCommand using the jump host's hostname/port.

                Returns True on success, False if ProxyCommand is unavailable.
                Raises on non-auth fatal errors to preserve original behaviour.
                """
                try:
                    from paramiko.proxy import ProxyCommand  # type: ignore
                except Exception:
                    return False

                # Resolve jump host connection details from the loaded HPC config
                jump_cfg = HPC_CONFIGS.get(jump_hpc_type)
                # Avoid interactive ProxyCommand when jump host requires 2FA and there is no active ControlMaster
                allow_interactive_pc = _truthy_env("ALLOW_INTERACTIVE_PROXYCOMMAND")
                alias_has_master = _openssh_master_running(proxy_alias) if proxy_alias else False
                if jump_cfg and jump_cfg.two_factor and not allow_interactive_pc and not alias_has_master:
                    logger.debug(
                        "Skipping ProxyCommand fallback; %s requires 2FA and no ControlMaster is active.",
                        jump_hpc_type.name,
                    )
                    return False
                j_user = cfg.proxy_jump_username or (jump_cfg.username if jump_cfg else None) or cfg.username
                j_host = jump_cfg.hostname if jump_cfg else cfg.proxy_jump or ""
                j_port = int(cfg.proxy_jump_port or (jump_cfg.port if jump_cfg else 22))
                if not j_host:
                    return False

                logger.info(
                    "Falling back to OpenSSH ProxyCommand via %s@%s:%s → %s:%s",
                    j_user,
                    j_host,
                    j_port,
                    _env("TUNNEL_HOST", hpc_name=hpc_type.name) or cfg.hostname,
                    _int_env("TUNNEL_PORT", cfg.port, hpc_name=hpc_type.name),
                )
                _dest_host2 = _env("TUNNEL_HOST", hpc_name=hpc_type.name) or cfg.hostname
                _dest_port2 = _int_env("TUNNEL_PORT", cfg.port, hpc_name=hpc_type.name)
                proxy_cmd = (
                    f"ssh -q -l {shlex.quote(j_user)} -p {j_port} {shlex.quote(j_host)} -W {_dest_host2}:{_dest_port2}"
                )

                # Prefer agent/default keys first; then fall back to password for the destination
                try:
                    client.connect(
                        hostname=_dest_host2,
                        port=_dest_port2,
                        username=cfg.username,
                        sock=ProxyCommand(proxy_cmd),
                        look_for_keys=True,
                        allow_agent=True,
                        timeout=_ssh_connect_timeout(),
                        banner_timeout=_ssh_banner_timeout(),
                        auth_timeout=_ssh_auth_timeout(),
                    )
                    self._clients[hpc_type] = client
                    self._connected[hpc_type] = True
                    return True
                except SSHException:
                    password = _safe_getpass(f"Enter password for {cfg.username}@{cfg.hostname}: ")
                    client.connect(
                        hostname=_dest_host2,
                        port=_dest_port2,
                        username=cfg.username,
                        password=password,
                        sock=ProxyCommand(proxy_cmd),
                        look_for_keys=False,
                        allow_agent=False,
                        timeout=_ssh_connect_timeout(),
                        banner_timeout=_ssh_banner_timeout(),
                        auth_timeout=_ssh_auth_timeout(),
                    )
                    self._clients[hpc_type] = client
                    self._connected[hpc_type] = True
                    return True

            # Open a direct-tcpip channel through the jump host with bounded retries
            def _open_tunnel_with_retries():
                last_exc: Exception | None = None
                retries = max(1, _ssh_proxy_retries())
                for attempt in range(1, retries + 1):
                    jump_client = self._clients.get(jump_hpc_type)
                    if not jump_client:
                        raise ValueError(f"Jump host {jump_hpc_type.name} client not available")

                    jump_transport = jump_client.get_transport()
                    # If transport is missing or inactive, try to reconnect once without
                    # forcing teardown that would re-prompt OTP unless needed.
                    if not jump_transport or getattr(jump_transport, "is_active", lambda: True)() is False:
                        logger.warning(
                            "Jump transport unavailable for %s; reconnecting (attempt %d/%d)",
                            jump_hpc_type.name,
                            attempt,
                            retries,
                        )
                        try:
                            self.connect(jump_hpc_type)
                        except Exception:
                            try:
                                self.close(jump_hpc_type)
                            except Exception:
                                pass
                            self.connect(jump_hpc_type)
                        jump_transport = self._clients[jump_hpc_type].get_transport()

                    # Allow overriding the destination of the tunnel
                    _dest_host3 = _env("TUNNEL_HOST", hpc_name=hpc_type.name) or cfg.hostname
                    _dest_port3 = _int_env("TUNNEL_PORT", cfg.port, hpc_name=hpc_type.name)
                    logger.debug(
                        "Creating SSH tunnel through %s to %s:%d (attempt %d/%d)",
                        jump_hpc_type.name,
                        _dest_host3,
                        _dest_port3,
                        attempt,
                        retries,
                    )
                    dest_addr = (_dest_host3, _dest_port3)
                    src_addr = ("127.0.0.1", 0)
                    try:
                        try:
                            return jump_transport.open_channel(
                                "direct-tcpip", dest_addr, src_addr, timeout=_ssh_tunnel_timeout()
                            )
                        except TypeError:
                            return jump_transport.open_channel("direct-tcpip", dest_addr, src_addr)
                    except Exception as e:
                        last_exc = e
                        emsg = str(e).lower()
                        transient = any(
                            s in emsg
                            for s in (
                                "timeout opening channel",
                                "open failed",
                                "connection reset",
                                "connection refused",
                            )
                        )
                        if attempt >= retries or not transient:
                            break
                        sleep_s = min(30, _ssh_proxy_backoff_base() ** (attempt - 1))
                        logger.warning(
                            "Tunnel open failed (%s). Retrying in %ss (attempt %d/%d)…",
                            e,
                            sleep_s,
                            attempt + 1,
                            retries,
                        )
                        # Keep jump session to avoid repeated OTP prompts
                        time.sleep(sleep_s)
                assert last_exc is not None
                raise last_exc

            channel = _open_tunnel_with_retries()

            logger.debug(f"Connecting to {hpc_type.name} through the tunnel")

            # Use keyboard-interactive when two_factor is enabled, or when explicitly
            # requested via env (e.g., servers that expose only keyboard-interactive).
            # Tests expect OLIVIA without two_factor to use standard auth by default,
            # which remains true unless the env override is set.
            env_kbd = _truthy_env("KBDINT", hpc_name=hpc_type.name)
            use_kbd = bool(cfg.two_factor) or env_kbd

            if use_kbd:
                # Interactive auth on the tunneled channel
                transport = paramiko.Transport(channel)
                try:
                    transport.banner_timeout = _ssh_auth_timeout()  # type: ignore[attr-defined]
                except Exception:
                    pass
                transport.set_keepalive(cfg.keep_alive_interval)
                transport.start_client(timeout=_ssh_auth_timeout())

                def handler(title, instructions, prompt_list):
                    responses = []
                    for prompt, show_echo in prompt_list:
                        p = prompt.strip()
                        p_lower = p.lower()
                        if (
                            'verification code' in p_lower
                            or '2fa' in p_lower
                            or 'one-time' in p_lower
                            or 'oath' in p_lower
                            or 'otp' in p_lower
                            or 'token' in p_lower
                        ):
                            val = _safe_input(f"[{hpc_type.name}] {p} ")
                        elif 'password' in p_lower:
                            val = _safe_getpass(f"[{hpc_type.name}] {p} ")
                        else:
                            val = _safe_input(f"[{hpc_type.name}] {p} ")
                        responses.append(val)
                    return responses

                try:
                    transport.auth_interactive(cfg.username, handler, submethods='pam')
                except TypeError:
                    transport.auth_interactive(cfg.username, handler)
                client._transport = transport
            else:
                # Standard auth through the tunnel with retry on handshake/banner timeouts
                connect_retries = max(1, _int_env("SSH_PROXY_CONNECT_RETRIES", 3))
                backoff_base = max(1, _int_env("SSH_PROXY_CONNECT_BACKOFF_BASE", 2))

                def _connect_through_channel():
                    nonlocal channel
                    last_exc: Exception | None = None
                    for attempt in range(1, connect_retries + 1):
                        try:
                            if cfg.use_key and cfg.key_path:
                                key_path = os.path.expanduser(cfg.key_path)
                                try:
                                    pkey = paramiko.Ed25519Key.from_private_key_file(key_path)
                                except Exception:
                                    try:
                                        pkey = paramiko.RSAKey.from_private_key_file(key_path)
                                    except PasswordRequiredException:
                                        passphrase = _safe_getpass(f"Enter passphrase for key '{key_path}': ")
                                        pkey = paramiko.RSAKey.from_private_key_file(key_path, password=passphrase)
                                client.connect(
                                    hostname=_env("TUNNEL_HOST", hpc_name=hpc_type.name) or cfg.hostname,
                                    port=_int_env("TUNNEL_PORT", cfg.port, hpc_name=hpc_type.name),
                                    username=cfg.username,
                                    pkey=pkey,
                                    sock=channel,
                                    look_for_keys=False,
                                    allow_agent=False,
                                    timeout=_ssh_connect_timeout(),
                                    banner_timeout=_ssh_banner_timeout(),
                                    auth_timeout=_ssh_auth_timeout(),
                                )
                            else:
                                try:
                                    client.connect(
                                        hostname=_env("TUNNEL_HOST", hpc_name=hpc_type.name) or cfg.hostname,
                                        port=_int_env("TUNNEL_PORT", cfg.port, hpc_name=hpc_type.name),
                                        username=cfg.username,
                                        sock=channel,
                                        look_for_keys=True,
                                        allow_agent=True,
                                        timeout=_ssh_connect_timeout(),
                                        banner_timeout=_ssh_banner_timeout(),
                                        auth_timeout=_ssh_auth_timeout(),
                                    )
                                except AuthenticationException:
                                    # Re-open a fresh tunnel before password/interactive fallback
                                    try:
                                        channel = _open_tunnel_with_retries()
                                    except Exception:
                                        raise
                                    password = _safe_getpass(f"Enter password for {cfg.username}@{cfg.hostname}: ")
                                    client.connect(
                                        hostname=_env("TUNNEL_HOST", hpc_name=hpc_type.name) or cfg.hostname,
                                        port=_int_env("TUNNEL_PORT", cfg.port, hpc_name=hpc_type.name),
                                        username=cfg.username,
                                        password=password,
                                        sock=channel,
                                        look_for_keys=False,
                                        allow_agent=False,
                                        timeout=_ssh_connect_timeout(),
                                        banner_timeout=_ssh_banner_timeout(),
                                        auth_timeout=_ssh_auth_timeout(),
                                    )
                                except SSHException:
                                    # Handshake/banner issues: bubble up to outer retry which re-opens tunnel
                                    raise
                            return
                        except Exception as e:
                            last_exc = e
                            emsg = str(e).lower()
                            transient = any(
                                s in emsg
                                for s in ("error reading ssh protocol banner", "banner", "timeout", "eof", "handshake")
                            )
                            if attempt >= connect_retries or not transient:
                                break
                            sleep_s = min(30, (backoff_base ** (attempt - 1)))
                            logger.warning(
                                "Handshake failed (%s). Re-opening tunnel and retrying in %ss (attempt %d/%d)…",
                                e,
                                sleep_s,
                                attempt + 1,
                                connect_retries,
                            )
                            try:
                                channel = _open_tunnel_with_retries()
                            except Exception as e2:
                                last_exc = e2
                                break
                            time.sleep(sleep_s)
                    assert last_exc is not None
                    raise last_exc

                try:
                    _connect_through_channel()
                except AuthenticationException as e_auth:
                    # If server disallows password and requires keyboard-interactive,
                    # automatically retry using interactive auth on a fresh tunnel.
                    emsg = str(e_auth).lower()
                    if "keyboard-interactive" in emsg or "keyboardinteractive" in emsg:
                        try:
                            # Open a fresh tunnel (do not reuse previous channel)
                            channel = _open_tunnel_with_retries()
                            transport = paramiko.Transport(channel)
                            try:
                                transport.banner_timeout = _ssh_auth_timeout()  # type: ignore[attr-defined]
                            except Exception:
                                pass
                            transport.set_keepalive(cfg.keep_alive_interval)
                            transport.start_client(timeout=_ssh_auth_timeout())

                            def handler(title, instructions, prompt_list):
                                responses = []
                                for prompt, show_echo in prompt_list:
                                    p = prompt.strip()
                                    p_lower = p.lower()
                                    if (
                                        'verification code' in p_lower
                                        or '2fa' in p_lower
                                        or 'one-time' in p_lower
                                        or 'oath' in p_lower
                                        or 'otp' in p_lower
                                        or 'token' in p_lower
                                    ):
                                        val = _safe_input(f"[{hpc_type.name}] {p} ")
                                    elif 'password' in p_lower:
                                        val = _safe_getpass(f"[{hpc_type.name}] {p} ")
                                    else:
                                        val = _safe_input(f"[{hpc_type.name}] {p} ")
                                    responses.append(val)
                                return responses

                            try:
                                transport.auth_interactive(cfg.username, handler, submethods='pam')
                            except TypeError:
                                transport.auth_interactive(cfg.username, handler)
                            client._transport = transport
                            # success path mirrors normal connect
                            self._clients[hpc_type] = client
                            self._connected[hpc_type] = True
                            t2 = client.get_transport()
                            if t2 and cfg.keep_alive:
                                t2.set_keepalive(cfg.keep_alive_interval)
                            logger.info(
                                "Successfully connected to %s via keyboard-interactive through %s",
                                hpc_type.name,
                                jump_hpc_type.name,
                            )
                            return
                        except Exception:
                            # Fall through to generic handler below
                            pass
                    # Not an allowed-type switch case; re-raise to generic flow
                    raise e_auth
                except Exception as e_first:
                    # If handshake failed very early (no SSH banner/EOF/timeout), try a
                    # resilient explicit ProxyCommand as a fallback before giving up.
                    emsg = str(e_first).lower()
                    is_bannerish = any(
                        s in emsg
                        for s in ("error reading ssh protocol banner", "banner", "eof", "handshake", "timeout")
                    )
                    if is_bannerish:
                        try:
                            if _try_proxycommand_direct():
                                transport = client.get_transport()
                                if transport and cfg.keep_alive:
                                    transport.set_keepalive(cfg.keep_alive_interval)
                                logger.info(
                                    "Successfully connected to %s via explicit ProxyCommand fallback through %s",
                                    hpc_type.name,
                                    jump_hpc_type.name,
                                )
                                self._clients[hpc_type] = client
                                self._connected[hpc_type] = True
                                return
                        except Exception as e_pc:
                            # If ProxyCommand fallback also fails, re-raise the original error to preserve semantics
                            logger.debug("ProxyCommand fallback failed: %s", e_pc)
                            raise e_first
                    # Non-banner or fallback failed: re-raise original
                    raise e_first

            self._clients[hpc_type] = client
            self._connected[hpc_type] = True

            transport = client.get_transport()
            if transport and cfg.keep_alive:
                transport.set_keepalive(cfg.keep_alive_interval)
                logger.debug(f"Set keepalive to {cfg.keep_alive_interval}s")

            logger.info(f"Successfully connected to {hpc_type.name} via {jump_hpc_type.name}")

        except AuthenticationException as e:
            # Treat auth as retriable: do not mark as failed
            logger.error(f"Proxy connection failed: AuthenticationException: {e}")
            self._connected[hpc_type] = False
            self._clients[hpc_type] = None
            raise
        except Exception as e:
            logger.error(f"Proxy connection failed: {type(e).__name__}: {e}")
            self._connected[hpc_type] = False
            self._clients[hpc_type] = None
            self._failed_connections.add(hpc_type)
            raise

    def _interactive_auth(self, client: paramiko.SSHClient, cfg: HPCConnectionConfig):
        """
        Keyboard-interactive approach for HPC that has separate 2FA + password steps.
        If HPC doesn't need interactive 2FA, set two_factor=False.
        """

        def handler(title, instructions, prompt_list):
            responses = []
            for prompt, show_echo in prompt_list:
                prompt_clean = prompt.strip()
                lower_prompt = prompt_clean.lower()
                if (
                    "verification code" in lower_prompt
                    or "2fa" in lower_prompt
                    or "one-time" in lower_prompt
                    or "oath" in lower_prompt
                    or "otp" in lower_prompt
                    or "token" in lower_prompt
                ):
                    val = _safe_input(prompt_clean + " ")
                elif "password" in lower_prompt:
                    val = _safe_getpass(prompt_clean + " ")
                else:
                    val = _safe_input(prompt_clean + " ")
                responses.append(val)
            return responses

        # Some servers require `pam`, others fail when it is forced.
        # Try generic keyboard-interactive first, then retry with `pam`.
        last_exc: Optional[Exception] = None
        for submethod in (None, "pam"):
            transport = paramiko.Transport((cfg.hostname, cfg.port))
            # Increase timeouts to avoid premature auth timeouts during 2FA.
            try:
                transport.banner_timeout = _ssh_banner_timeout()  # type: ignore[attr-defined]
            except Exception:
                pass
            try:
                transport.auth_timeout = _ssh_auth_timeout()  # type: ignore[attr-defined]
            except Exception:
                pass
            transport.set_keepalive(cfg.keep_alive_interval)
            try:
                transport.start_client(timeout=_ssh_auth_timeout())
                if submethod is None:
                    transport.auth_interactive(cfg.username, handler)
                else:
                    try:
                        transport.auth_interactive(cfg.username, handler, submethods=submethod)
                    except TypeError:
                        # Fallback for older Paramiko that doesn't support submethods kwarg
                        transport.auth_interactive(cfg.username, handler)

                # Attach the successful transport to the paramiko client
                client._transport = transport
                return
            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt during interactive auth.")
                transport.close()
                raise UserCancelledError("User cancelled authentication.")
            except Exception as e:
                last_exc = e
                try:
                    transport.close()
                except Exception:
                    pass
                logger.debug(
                    "Interactive auth attempt failed for %s (submethod=%s): %s",
                    cfg.hostname,
                    submethod if submethod is not None else "<default>",
                    e,
                )

        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"Interactive authentication failed for {cfg.username}@{cfg.hostname}")

    def _standard_auth(self, client: paramiko.SSHClient, cfg: HPCConnectionConfig):
        """Standard SSH auth with optional key-based auth.

        Falls back to keyboard-interactive when the server requires it, even if
        two_factor is False in the config (e.g., Olivia direct login).
        """

        def _requires_kbdint(exc: Exception) -> bool:
            """Return True when the exception indicates keyboard-interactive is required."""
            try:
                allowed = [s.lower() for s in getattr(exc, "allowed_types", []) if isinstance(s, str)]
            except Exception:
                allowed = []
            msg = str(exc).lower()
            return (
                "keyboard-interactive" in msg
                or "keyboardinteractive" in msg
                or any("keyboard-interactive" in s or "keyboardinteractive" in s for s in allowed)
            )

        try:
            if cfg.use_key:
                logger.debug(f"Attempting key-based auth for {cfg.hostname}")
                key_path = os.path.expanduser(cfg.key_path)
                # Try Ed25519 first
                try:
                    logger.debug("Trying Ed25519 key...")
                    pkey = paramiko.Ed25519Key.from_private_key_file(key_path)
                except PasswordRequiredException:
                    logger.debug("Ed25519 key requires passphrase")
                    passphrase = _safe_getpass(f"Enter passphrase for key '{key_path}': ")
                    pkey = paramiko.Ed25519Key.from_private_key_file(key_path, password=passphrase)
                    logger.debug("Successfully loaded Ed25519 key with passphrase")
                except Exception:
                    # Fallback to RSA
                    try:
                        logger.debug("Trying RSA key...")
                        pkey = paramiko.RSAKey.from_private_key_file(key_path)
                    except PasswordRequiredException:
                        logger.debug("RSA key requires passphrase")
                        passphrase = _safe_getpass(f"Enter passphrase for key '{key_path}': ")
                        pkey = paramiko.RSAKey.from_private_key_file(key_path, password=passphrase)
                        logger.debug("Successfully loaded RSA key with passphrase")

                logger.debug("Attempting connection with key")
                try:
                    client.connect(
                        hostname=cfg.hostname,
                        port=cfg.port,
                        username=cfg.username,
                        pkey=pkey,
                        look_for_keys=False,
                        allow_agent=False,
                        timeout=_ssh_connect_timeout(),
                        banner_timeout=_ssh_banner_timeout(),
                        auth_timeout=_ssh_auth_timeout(),
                    )
                    logger.debug("Successfully connected with key authentication")
                except BadAuthenticationType as e_bad:
                    if _requires_kbdint(e_bad):
                        logger.info("Server requires keyboard-interactive; retrying interactively")
                        self._interactive_auth(client, cfg)
                        return
                    raise
                except SSHException:
                    # Fallback to agent-based auth if available
                    logger.debug("Key auth failed; trying SSH agent / default keys")
                    client.connect(
                        hostname=cfg.hostname,
                        port=cfg.port,
                        username=cfg.username,
                        allow_agent=True,
                        look_for_keys=True,
                        timeout=_ssh_connect_timeout(),
                        banner_timeout=_ssh_banner_timeout(),
                        auth_timeout=_ssh_auth_timeout(),
                    )
                    logger.debug("Successfully connected using SSH agent/default keys")
            else:
                # Try agent/default keys first to avoid interactive prompts on trusted hosts
                logger.debug(f"Attempting agent/default-key auth for {cfg.hostname}")
                try:
                    client.connect(
                        hostname=cfg.hostname,
                        port=cfg.port,
                        username=cfg.username,
                        allow_agent=True,
                        look_for_keys=True,
                        timeout=_ssh_connect_timeout(),
                    )
                    logger.debug("Successfully connected using SSH agent/default keys")
                except BadAuthenticationType as e_bad:
                    if _requires_kbdint(e_bad):
                        logger.info("Server requires keyboard-interactive; retrying interactively")
                        self._interactive_auth(client, cfg)
                        return
                    raise
                except SSHException as e_agent:
                    # Fallback to password prompt
                    logger.debug("Agent/default keys failed; falling back to password auth")
                    try:
                        password = _safe_getpass(f"Enter password for {cfg.username}@{cfg.hostname}: ")
                        client.connect(
                            hostname=cfg.hostname,
                            port=cfg.port,
                            username=cfg.username,
                            password=password,
                            look_for_keys=False,
                            allow_agent=False,
                            timeout=_ssh_connect_timeout(),
                            banner_timeout=_ssh_banner_timeout(),
                            auth_timeout=_ssh_auth_timeout(),
                        )
                        logger.debug("Successfully connected with password authentication")
                    except BadAuthenticationType as e_bad_pw:
                        if _requires_kbdint(e_bad_pw):
                            logger.info("Server requires keyboard-interactive; retrying interactively")
                            self._interactive_auth(client, cfg)
                            return
                        raise e_bad_pw
                    except SSHException:
                        # Re-raise the original agent failure to keep semantics
                        raise e_agent
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt during authentication")
            raise UserCancelledError("User cancelled authentication.")
        except BadAuthenticationType as e:
            if _requires_kbdint(e):
                logger.info("Server requires keyboard-interactive; retrying interactively")
                self._interactive_auth(client, cfg)
                return
            logger.error(f"SSH authentication error: {e}")
            raise
        except AuthenticationException as e:
            if _requires_kbdint(e):
                logger.info("Server requires keyboard-interactive; retrying interactively")
                self._interactive_auth(client, cfg)
                return
            logger.error(f"SSH authentication error: {e}")
            raise
        except paramiko.SSHException as e:
            logger.error(f"SSH authentication error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error during standard authentication: {type(e).__name__}: {e}")
            raise

    def run_command(self, hpc_type: HPCType, command: str, prefer_remote: bool = False) -> Tuple[str, str]:
        """
        Run a shell command on HPC.
        - If prefer_remote is True, execute via SSH even when running on that HPC locally.
        - Otherwise, run locally when we are on that HPC; use SSH for non-local.
        Returns stdout, stderr as strings.
        """
        # Remote-preferred path (used for Slurm CLI like sbatch/squeue/scancel)
        if prefer_remote:
            # When running on the same HPC, prefer a dedicated submission endpoint if configured.
            # If no submission host and no SSH session, local fallback is allowed only
            # when required Slurm CLIs are visible from the current PATH.
            use_submission = False
            use_local_fallback = False
            cfg = self.configs[hpc_type]
            if self.is_local_hpc(hpc_type):
                sub_host = _env("SUBMISSION_HOST", hpc_name=hpc_type.name) or cfg.submission_host
                missing_slurm = _missing_local_slurm_binaries(command)
                if sub_host:
                    if _is_loopback_or_current_host(sub_host):
                        if not missing_slurm:
                            # Avoid pointless self-SSH prompts when Slurm CLI is already available locally.
                            use_local_fallback = True
                        else:
                            allow_self_ssh = _truthy_env("ALLOW_SELF_SSH_LOCAL")
                            if allow_self_ssh:
                                use_submission = True
                            else:
                                raise RuntimeError(
                                    f"Local Slurm CLI unavailable on {hpc_type.name} "
                                    f"(missing: {', '.join(missing_slurm)}). "
                                    f"Refusing self-SSH to '{sub_host}' by default. "
                                    "Fix local Slurm availability in this environment or set "
                                    f"{_env_prefixes()[0]}_ALLOW_SELF_SSH_LOCAL=1 to explicitly allow self-SSH."
                                )
                    else:
                        use_submission = True
                elif not self._clients.get(hpc_type):
                    if missing_slurm:
                        logger.debug(
                            "Skipping local prefer_remote fallback for %s; missing Slurm binaries in local PATH: %s",
                            hpc_type.name,
                            ", ".join(missing_slurm),
                        )
                    else:
                        # No submission host and no SSH session — run locally
                        use_local_fallback = True
            if use_local_fallback:
                import subprocess as _sp

                logger.debug(f"Running local command (prefer_remote fallback): {command}")
                result = _sp.run(command, shell=True, capture_output=True, text=True)
                return (result.stdout, result.stderr)
            if use_submission:
                if not self._sub_connected.get(hpc_type):
                    logger.debug(f"Connecting to submission host for {hpc_type.name}…")
                    self._connect_submission(hpc_type)
                client = self._sub_clients.get(hpc_type)
            else:
                if not self._clients.get(hpc_type):
                    logger.debug(f"No SSH session to {hpc_type.name}. Connecting (force_remote=True)...")
                    self.connect(hpc_type, force_remote=True)
                client = self._clients.get(hpc_type)
            if client is None:
                logger.warning(
                    "Remote client missing for %s (submission=%s). Forcing reconnect.", hpc_type.name, use_submission
                )
                if use_submission:
                    self._close_submission(hpc_type)
                    self._connect_submission(hpc_type)
                    client = self._sub_clients.get(hpc_type)
                else:
                    self.close(hpc_type)
                    self.connect(hpc_type, force_remote=True)
                    client = self._clients.get(hpc_type)
                if client is None:
                    raise RuntimeError(
                        f"Unable to establish remote command client for {hpc_type.name} (submission={use_submission})."
                    )
            try:
                logger.debug(f"Running remote command on {hpc_type.name}: {command}")
                # Avoid shell wrapping for simple commands so tests can assert exact calls.
                needs_shell = any(tok in command for tok in [';', '&&', '||', '|', '$', '`'])
                wrapped = f"bash -lc {shlex.quote(command)}" if needs_shell else command
                stdin, stdout, stderr = client.exec_command(wrapped)
                out = stdout.read().decode("utf-8", errors="replace")
                err = stderr.read().decode("utf-8", errors="replace")
                return out, err
            except Exception as e:
                logger.error(f"Command failed on {hpc_type.name}: {command} ({type(e).__name__}: {e})")
                if not getattr(self, '_reconnecting', False):
                    logger.info(f"Attempting to reconnect to {hpc_type.name}")
                    self._reconnecting = True
                    if use_submission:
                        self._close_submission(hpc_type)
                        self._connect_submission(hpc_type)
                    else:
                        self.close(hpc_type)
                        self.connect(hpc_type, force_remote=True)
                    self._reconnecting = False
                    return self.run_command(hpc_type, command, prefer_remote=True)
                else:
                    logger.error("Already attempting reconnection")
                    raise

        # Local-or-remote default path
        if self.is_local_hpc(hpc_type):
            # Run command locally
            import subprocess

            logger.debug(f"Running local command: {command}")
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            return (result.stdout, result.stderr)

        # Non-local: run remotely via paramiko
        if not self._connected.get(hpc_type, False):
            logger.debug(f"Not connected to {hpc_type.name}, connecting first...")
            self.connect(hpc_type)
            return self.run_command(hpc_type, command)
        client = self._clients[hpc_type]
        try:
            logger.debug(f"Running remote command on {hpc_type.name}: {command}")
            needs_shell = any(tok in command for tok in [';', '&&', '||', '|', '$', '`'])
            wrapped = f"bash -lc {shlex.quote(command)}" if needs_shell else command
            stdin, stdout, stderr = client.exec_command(wrapped)
            out = stdout.read().decode("utf-8", errors="replace")
            err = stderr.read().decode("utf-8", errors="replace")
            return out, err
        except Exception as e:
            logger.error(f"Command failed on {hpc_type.name}: {command} ({type(e).__name__}: {e})")
            if not getattr(self, '_reconnecting', False):
                logger.info(f"Attempting to reconnect to {hpc_type.name}")
                self._reconnecting = True
                self.close(hpc_type)
                self.connect(hpc_type)
                self._reconnecting = False
                return self.run_command(hpc_type, command)
            else:
                logger.error("Already attempting reconnection")
                raise

    # -------------------------------
    # File transfer helpers (SFTP)
    # -------------------------------
    def upload_file(self, hpc_type: HPCType, local_path: str, remote_path: str) -> None:
        """Upload a local file to the remote HPC via SFTP.

        Creates parent directory if necessary using a remote mkdir -p call.
        """
        import posixpath

        if not self._connected.get(hpc_type, False):
            self.connect(hpc_type)
        client = self._clients[hpc_type]
        # Ensure parent dir exists
        parent = posixpath.dirname(remote_path)
        if parent:
            try:
                self.run_command(hpc_type, f"mkdir -p {parent}")
            except Exception:
                pass
        # SFTP put
        sftp = client.open_sftp()
        try:
            sftp.put(local_path, remote_path)
        finally:
            try:
                sftp.close()
            except Exception:
                pass

    def upload_text(self, hpc_type: HPCType, content: str, remote_path: str, mode: Optional[int] = None) -> None:
        """Create or overwrite a remote file with given text content via SFTP."""
        import posixpath

        if not self._connected.get(hpc_type, False):
            self.connect(hpc_type)
        client = self._clients[hpc_type]
        parent = posixpath.dirname(remote_path)
        if parent:
            try:
                self.run_command(hpc_type, f"mkdir -p {parent}")
            except Exception:
                pass
        sftp = client.open_sftp()
        try:
            with sftp.file(remote_path, 'w') as f:
                f.write(content)
            if mode is not None:
                sftp.chmod(remote_path, mode)
        finally:
            try:
                sftp.close()
            except Exception:
                pass

    def _connect_submission(self, hpc_type: HPCType) -> None:
        """Establish SSH connection to the alternate submission host for the HPC."""
        cfg = self.configs[hpc_type]
        host = _env("SUBMISSION_HOST", hpc_name=hpc_type.name) or cfg.submission_host or cfg.hostname
        user = _env("SUBMISSION_USER", hpc_name=hpc_type.name) or cfg.submission_username or cfg.username
        port = _int_env("SUBMISSION_PORT", cfg.submission_port or cfg.port, hpc_name=hpc_type.name)
        use_key = (
            _truthy_env("SUBMISSION_USE_KEY", hpc_name=hpc_type.name)
            if _env("SUBMISSION_USE_KEY", hpc_name=hpc_type.name) is not None
            else (cfg.submission_use_key if cfg.submission_use_key is not None else cfg.use_key)
        )
        key_path = _env("SUBMISSION_KEY_PATH", hpc_name=hpc_type.name) or cfg.submission_key_path or cfg.key_path
        two_factor = (
            _truthy_env("SUBMISSION_2FA", hpc_name=hpc_type.name)
            if _env("SUBMISSION_2FA", hpc_name=hpc_type.name) is not None
            else (cfg.submission_two_factor if cfg.submission_two_factor is not None else False)
        )

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            if two_factor:
                # Keyboard-interactive flow
                logger.debug(f"Submission host using keyboard-interactive for {hpc_type.name}")
                self._interactive_auth(
                    client,
                    HPCConnectionConfig(
                        hostname=host, username=user, port=port, use_key=use_key, key_path=key_path, two_factor=True
                    ),
                )
            else:
                logger.debug(f"Submission host using standard auth for {hpc_type.name}")
                # Standard auth with key/password
                tmp_cfg = HPCConnectionConfig(
                    hostname=host, username=user, port=port, use_key=use_key, key_path=key_path, two_factor=False
                )
                self._standard_auth(client, tmp_cfg)
            self._sub_clients[hpc_type] = client
            self._sub_connected[hpc_type] = True
            logger.info(f"Connected to submission host '{host}' for {hpc_type.name}")
        except Exception:
            self._sub_connected[hpc_type] = False
            self._sub_clients[hpc_type] = None
            raise

    def _close_submission(self, hpc_type: HPCType) -> None:
        """Close submission host connection for the HPC."""
        try:
            if self._sub_clients.get(hpc_type) and self._sub_connected.get(hpc_type):
                self._sub_clients[hpc_type].close()
        except Exception:
            pass
        self._sub_connected[hpc_type] = False
        self._sub_clients[hpc_type] = None

    def close(self, hpc_type: HPCType):
        """Close a single HPC connection."""
        if hpc_type in self._clients and self._connected[hpc_type]:
            try:
                self._clients[hpc_type].close()
            except Exception:
                pass
        self._connected[hpc_type] = False
        logger.info(f"Closed connection to {hpc_type.name}.")

    def close_all(self):
        """Close all HPC connections."""
        for hpc_type in list(self._clients.keys()):
            self.close(hpc_type)


def _safe_input(prompt: str) -> str:
    """Prompt user for input with potential KeyboardInterrupt handling."""
    try:
        return input(prompt)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt during input.")
        raise UserCancelledError("User cancelled prompt.")


def _safe_getpass(prompt: str) -> str:
    """Prompt user for password with potential KeyboardInterrupt handling."""
    import getpass

    try:
        return getpass.getpass(prompt)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt during getpass.")
        raise UserCancelledError("User cancelled getpass prompt.")


def main():
    """
    Example usage of HPCConnectionManager, building from HPC_CONFIGS in hpc_config.py.
    Run this script directly to test connectivity to the HPCs.
    """
    import logging

    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")

    # Build HPCConnectionConfig from HPCClusterConfig
    # This centralizes the SSH info from HPC_CONFIGS
    connection_configs = {}
    for hpc_type, cluster_cfg in HPC_CONFIGS.items():
        connection_configs[hpc_type] = HPCConnectionConfig(
            hostname=cluster_cfg.hostname,
            username=cluster_cfg.username,
            port=cluster_cfg.port,
            use_key=cluster_cfg.use_key,
            key_path=cluster_cfg.key_path,
            two_factor=cluster_cfg.two_factor,
            keep_alive=True,
            keep_alive_interval=30,
        )

    manager = HPCConnectionManager(configs=connection_configs)

    try:
        # Example: Connect to all HPCs listed in HPC_CONFIGS
        manager.connect_all()

        # Test commands
        for hpc_type in connection_configs.keys():
            out, err = manager.run_command(hpc_type, "hostname")
            print(f"[{hpc_type.name}] out: {out.strip()}")
            if err.strip():
                print(f"[{hpc_type.name}] err: {err.strip()}")

    except UserCancelledError as e:
        logger.info(f"Authentication cancelled by user: {e}")
    except KeyboardInterrupt:
        logger.info("Script interrupted by user at top-level.")
    finally:
        logger.info("Cleaning up and closing all connections.")
        manager.close_all()


__all__ = [
    "BadAuthenticationType",
    "ENV_PREFIXES",
    "_env_prefixes",
    "HPCConnectionConfig",
    "HPCConnectionManager",
    "UserCancelledError",
    "_missing_local_slurm_binaries",
    "_safe_getpass",
    "_safe_input",
]


if __name__ == "__main__":
    main()

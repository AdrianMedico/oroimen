"""Test del EgressFirewall (Fase 0 hardening)."""

import os
from unittest.mock import MagicMock, patch


# No aplicamos iptables real: mockeamos subprocess.run
class FakeRun:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr

    def __call__(self, *args, **kwargs):
        return MagicMock(returncode=self.returncode, stdout=self.stdout, stderr=self.stderr)


def test_disabled_noop():
    """EGRESS_FIREWALL_ENABLED=false: noop, return True."""
    with patch.dict(os.environ, {"EGRESS_FIREWALL_ENABLED": "false"}, clear=False):
        from hermes.security.egress import EgressFirewall

        fw = EgressFirewall.from_env()
        assert fw.enabled is False
        with patch("subprocess.run") as mock:
            assert fw.apply() is True
            mock.assert_not_called()


def test_from_env_parses_correctly():
    """from_env lee env vars correctamente."""
    env = {
        "EGRESS_FIREWALL_ENABLED": "true",
        "EGRESS_DNS_SERVERS": "1.1.1.1,8.8.8.8",
        "EGRESS_ALLOWED_DOMAINS": "example.com,test.com",
        "EGRESS_ALLOWED_IPS": "198.51.100.0/24,203.0.113.0/24",
        "EGRESS_ALLOWED_PORTS": "443,80,8080",
    }
    with patch.dict(os.environ, env, clear=False):
        from hermes.security.egress import EgressFirewall

        fw = EgressFirewall.from_env()
        assert fw.enabled is True
        assert "1.1.1.1" in fw.dns_servers
        assert "8.8.8.8" in fw.dns_servers
        assert "example.com" in fw.allowed_domains
        assert "198.51.100.0/24" in fw.allowed_ips
        assert 443 in fw.allowed_ports
        assert 80 in fw.allowed_ports
        assert 8080 in fw.allowed_ports


def test_apply_runs_iptables():
    """apply() corre iptables con los args correctos."""
    with patch.dict(os.environ, {"EGRESS_FIREWALL_ENABLED": "true"}, clear=False):
        from hermes.security.egress import EgressFirewall

        fw = EgressFirewall.from_env()
        calls = []

        def fake_run(args, **kwargs):
            calls.append(args)
            return MagicMock(returncode=0, stdout="", stderr="")

        with (
            patch("subprocess.run", side_effect=fake_run),
            patch("socket.gethostbyname_ex", return_value=("api.opencode.ai", [], ["203.0.113.1"])),
        ):
            result = fw.apply()
        assert result is True
        # Chain create: iptables -N HERMES_EGRESS
        assert any("-N" in str(c) and "HERMES_EGRESS" in str(c) for c in calls)
        # OUTPUT hook
        assert any("-I" in str(c) and "OUTPUT" in str(c) for c in calls)
        # Final DROP
        assert any("DROP" in str(c) for c in calls)


def test_apply_handles_iptables_failure():
    """Si iptables falla, graceful: no crash, return False."""
    with patch.dict(os.environ, {"EGRESS_FIREWALL_ENABLED": "true"}, clear=False):
        from hermes.security.egress import EgressFirewall

        fw = EgressFirewall.from_env()

        def fake_run(args, **kwargs):
            return MagicMock(returncode=1, stdout="", stderr="permission denied")

        with (
            patch("subprocess.run", side_effect=fake_run),
            patch("socket.gethostbyname_ex", return_value=("api.opencode.ai", [], ["203.0.113.1"])),
        ):
            result = fw.apply()
        # Chain create falla -> apply returns False
        assert result is False


def test_remove_idempotent():
    """remove() es idempotente (puede llamarse 2 veces sin error)."""
    with patch.dict(os.environ, {"EGRESS_FIREWALL_ENABLED": "true"}, clear=False):
        from hermes.security.egress import EgressFirewall

        fw = EgressFirewall.from_env()
        calls = []

        def fake_run(args, **kwargs):
            calls.append(args)
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("subprocess.run", side_effect=fake_run):
            fw.remove()
            fw.remove()  # segunda vez, no debe fallar
        # Ambas llamadas sin excepcion
        assert len(calls) > 0

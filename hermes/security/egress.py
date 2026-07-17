"""Fase 0 Hardening: egress firewall para hermes.

Si hermes es comprometido (prompt injection via open-webui, 0-day en
LLM, bug en dep), el atacante NO debe poder exfiltrar data a
servidores arbitrarios. Solo puede hablar con los servicios que
hermes ya usa legitimamente.

Approach: iptables en el container, default deny, whitelist
configurable via env vars.

Config (env vars):
    EGRESS_FIREWALL_ENABLED: true/false (default: false, opt-in)
    EGRESS_DNS_SERVERS: DNS servers (default: 127.0.0.11 Docker DNS + 1.1.1.1)
    EGRESS_ALLOWED_DOMAINS: comma-separated FQDN (default: ver DEFAULT_DOMAINS)
    EGRESS_ALLOWED_IPS: comma-separated CIDR/range (default: LAN ranges)
    EGRESS_ALLOWED_PORTS: comma-separated (default: 443, 80)

Default allowed (Sprint 9.5+ stack, future-proof for S11+S12):
    - api.minimax.io (LLM primary, MiniMax API)
    - generativelanguage.googleapis.com (STT S11+)
    - api.tavily.com, api.exa.ai (web search S9.3)
    - api.github.com (GitHub MCP S12+)
    - api.githubusercontent.com (assets)
    - mcp.context7.com (current MCP)
    - Docker bridge address ranges configured by the operator
    - Private LAN ranges configured by the operator

NOTA: el default de EGRESS_ALLOWED_IPS esta vacio intencionalmente.
Cada deployment debe configurar su rango LAN via env var (ej.
`EGRESS_ALLOWED_IPS=<internal-cidr>` para una LAN típica). No se incluye
ningún rango privado específico por defecto; cada operador define el suyo.

If EGRESS_FIREWALL_ENABLED=false (default, dev workflow): noop, log info.
If iptables fails (no NET_ADMIN cap): graceful degradation, log warning.
"""

from __future__ import annotations

import logging
import os
import socket
import subprocess
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# Defaults (override via env vars)
DEFAULT_DNS_SERVERS = "127.0.0.11,1.1.1.1"
DEFAULT_ALLOWED_DOMAINS = (
    "api.minimax.io,"
    "generativelanguage.googleapis.com,"
    "api.tavily.com,"
    "api.exa.ai,"
    "api.github.com,"
    "api.githubusercontent.com,"
    "mcp.context7.com"
)
DEFAULT_ALLOWED_IPS = ""
DEFAULT_ALLOWED_PORTS = "443,80"


@dataclass
class EgressFirewall:
    """Sprint 0 hardening: egress firewall que deniega todo el trafico
    saliente EXCEPTO una whitelist configurable.

    Use case: defense-in-depth contra prompt injection y RCE.
    Si hermes es comprometido, el atacante solo puede hablar con los
    servicios que hermes ya usa legitimamente (LLM API, MCP servers,
    LAN de NAS host). No puede exfiltrar data a evil.com.

    NO es un reemplazo de bearer auth ni de network isolation Docker.
    Es la TERCERA capa de defense-in-depth:
    1. Network isolation (Docker networks, internal isolated network)
    2. Bearer auth (HTTP API only)
    3. Egress firewall (este modulo, opcional)
    """

    enabled: bool = False
    dns_servers: list[str] = field(default_factory=list)
    allowed_domains: list[str] = field(default_factory=list)
    allowed_ips: list[str] = field(default_factory=list)
    allowed_ports: list[int] = field(default_factory=list)
    rules_applied: int = 0

    @classmethod
    def from_env(cls) -> EgressFirewall:
        """Lee env vars y construye la config."""
        return cls(
            enabled=os.environ.get("EGRESS_FIREWALL_ENABLED", "false").lower()
            in ("true", "1", "yes"),
            dns_servers=[
                s.strip()
                for s in os.environ.get("EGRESS_DNS_SERVERS", DEFAULT_DNS_SERVERS).split(",")
                if s.strip()
            ],
            allowed_domains=[
                d.strip()
                for d in os.environ.get("EGRESS_ALLOWED_DOMAINS", DEFAULT_ALLOWED_DOMAINS).split(
                    ","
                )
                if d.strip()
            ],
            allowed_ips=[
                i.strip()
                for i in os.environ.get("EGRESS_ALLOWED_IPS", DEFAULT_ALLOWED_IPS).split(",")
                if i.strip()
            ],
            allowed_ports=[
                int(p.strip())
                for p in os.environ.get("EGRESS_ALLOWED_PORTS", DEFAULT_ALLOWED_PORTS).split(",")
                if p.strip() and p.strip().isdigit()
            ],
        )

    def _resolve_domain(self, domain: str) -> list[str]:
        """Resuelve un dominio a sus IPs. Retorna lista vacia si falla."""
        try:
            return list(set(socket.gethostbyname_ex(domain)[2]))
        except (socket.gaierror, OSError) as exc:
            logger.warning(
                "egress_dns_resolve_failed",
                extra={
                    "domain": domain,
                    "error": str(exc),
                },
            )
            return []

    def _run_iptables(self, args: list[str]) -> tuple[bool, str]:
        """Ejecuta iptables. Retorna (success, output)."""
        try:
            result = subprocess.run(
                ["iptables", *args],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                return False, result.stderr.strip()
            return True, result.stdout.strip()
        except FileNotFoundError:
            return False, "iptables not found (no NET_ADMIN cap?)"
        except subprocess.TimeoutExpired:
            return False, "iptables timeout"

    def apply(self) -> bool:
        """Aplica las reglas de firewall. Retorna True si exito."""
        if not self.enabled:
            logger.info("egress_firewall_disabled")
            return True  # disabled = "success" (noop)

        logger.info(
            "egress_firewall_applying",
            extra={
                "dns_servers": self.dns_servers,
                "allowed_domains": self.allowed_domains,
                "allowed_ips": self.allowed_ips,
                "allowed_ports": self.allowed_ports,
            },
        )

        # Limpiar reglas existentes de nuestra chain (idempotente)
        # Chain name: HERMES_EGRESS (custom chain para aislar)
        self._run_iptables(["-F", "HERMES_EGRESS"])
        self._run_iptables(["-X", "HERMES_EGRESS"])
        # Crear chain
        ok, err = self._run_iptables(["-N", "HERMES_EGRESS"])
        if not ok:
            logger.warning("egress_firewall_chain_create_failed", extra={"error": err})
            return False

        # Regla 1: loopback (permitir todo en 127.0.0.0/8)
        ok, _ = self._run_iptables(["-A", "HERMES_EGRESS", "-o", "lo", "-j", "ACCEPT"])
        if not ok:
            logger.warning("egress_firewall_loopback_rule_failed")

        # Regla 2: DNS (UDP/TCP 53 a los DNS servers whitelisteados)
        for dns in self.dns_servers:
            for proto in ("udp", "tcp"):
                self._run_iptables(
                    [
                        "-A",
                        "HERMES_EGRESS",
                        "-p",
                        proto,
                        "-d",
                        dns,
                        "--dport",
                        "53",
                        "-j",
                        "ACCEPT",
                    ]
                )

        # Regla 3: resolver dominios whitelisteados y permitir acceso a sus IPs
        # Solo para puertos whitelisteados
        domain_ips: set[str] = set()
        for domain in self.allowed_domains:
            for ip in self._resolve_domain(domain):
                domain_ips.add(ip)
                for port in self.allowed_ports:
                    self._run_iptables(
                        [
                            "-A",
                            "HERMES_EGRESS",
                            "-p",
                            "tcp",
                            "-d",
                            ip,
                            "--dport",
                            str(port),
                            "-j",
                            "ACCEPT",
                        ]
                    )

        # Regla 4: IPs/CIDRs whitelisteados (LAN ranges, etc)
        for cidr in self.allowed_ips:
            for port in self.allowed_ports:
                self._run_iptables(
                    [
                        "-A",
                        "HERMES_EGRESS",
                        "-p",
                        "tcp",
                        "-d",
                        cidr,
                        "--dport",
                        str(port),
                        "-j",
                        "ACCEPT",
                    ]
                )

        # Regla 5: established/related (return traffic)
        self._run_iptables(
            [
                "-A",
                "HERMES_EGRESS",
                "-m",
                "state",
                "--state",
                "ESTABLISHED,RELATED",
                "-j",
                "ACCEPT",
            ]
        )

        # Regla 6 (final): DROP todo lo demas
        self._run_iptables(["-A", "HERMES_EGRESS", "-j", "DROP"])

        # Insertar chain en OUTPUT (despues de reglas base)
        # Primero remover insercion previa (idempotente)
        self._run_iptables(["-D", "OUTPUT", "-j", "HERMES_EGRESS"])
        ok, err = self._run_iptables(
            [
                "-I",
                "OUTPUT",
                "1",
                "-j",
                "HERMES_EGRESS",
            ]
        )
        if not ok:
            logger.warning("egress_firewall_output_hook_failed", extra={"error": err})
            # Cleanup: borrar chain
            self._run_iptables(["-F", "HERMES_EGRESS"])
            self._run_iptables(["-X", "HERMES_EGRESS"])
            return False

        # Contar reglas aplicadas (excluyendo el chain itself)
        ok, out = self._run_iptables(["-L", "HERMES_EGRESS"])
        rule_count = len([line for line in out.splitlines() if line.strip().startswith("--")])
        # Aproximacion: cada regla es 1-2 lineas en iptables -L
        self.rules_applied = rule_count

        logger.info(
            "egress_firewall_applied",
            extra={
                "rules_count": rule_count,
                "resolved_ips": list(domain_ips),
            },
        )
        return True

    def remove(self) -> bool:
        """Quita el firewall (rollback). Idempotente."""
        if not self.enabled:
            return True
        self._run_iptables(["-D", "OUTPUT", "-j", "HERMES_EGRESS"])
        self._run_iptables(["-F", "HERMES_EGRESS"])
        self._run_iptables(["-X", "HERMES_EGRESS"])
        logger.info("egress_firewall_removed")
        return True

"""Security primitives (Fase 0 hardening).

Modules:
- egress: iptables-based egress firewall (optional, opt-in).
  Use case: if hermes is compromised via prompt injection or 0-day,
  the attacker cannot exfiltrate data to arbitrary servers. Only
  whitelisted services (LLM API, MCP servers, LAN) are reachable.
"""

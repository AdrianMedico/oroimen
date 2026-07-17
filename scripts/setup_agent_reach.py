"""Sprint 4 MVP-2: setup idempotente de Agent-Reach al arranque de Oroimen.

Que hace (todo idempotente, safe to re-run):
1. Crea ~/.agent-reach/config.yaml con template (permisos 0600).
2. Configura yt-dlp JS runtime (YouTube requiere Node.js).
3. Configura Exa MCP via mcporter (no API key necesaria).

Lo que NO hace (se hace en Dockerfile, build time):
- apt-get install gh, nodejs, ffmpeg
- pip install agent-reach
- npm install -g mcporter

Si las deps no estan instaladas, las llamadas subprocess fallan con
FileNotFoundError, que se loggea como warning. No rompe el arranque.

Cuando un tool intenta usar agent-reach, her Oroimen wrapper
(hermes/tools/agent_reach.py) detecta el FileNotFoundError y devuelve
un mensaje user-friendly al LLM.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


CONFIG_TEMPLATE = """\
# Agent-Reach config (Sprint 4 MVP-2)
# Edita y anade tokens. Despues: agent-reach doctor
# Para mas info: https://github.com/Panniantong/Agent-Reach

# github_token: ghp_xxxx
# groq_api_key: gsk_xxxx
# proxy: http://user:pass@host:port
"""


def setup_agent_reach() -> None:
    """Setup idempotente: config dir, yt-dlp JS, Exa MCP."""
    home = Path.home()

    # 1. ~/.agent-reach/config.yaml
    config_dir = home / ".agent-reach"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "config.yaml"
    if not config_file.exists():
        config_file.write_text(CONFIG_TEMPLATE, encoding="utf-8")
        os.chmod(config_file, 0o600)
        logger.info("agent_reach_config_created", extra={"path": str(config_file)})

    # 2. yt-dlp JS runtime (YouTube requiere node como JS engine)
    ytdlp_dir = home / ".config" / "yt-dlp"
    ytdlp_dir.mkdir(parents=True, exist_ok=True)
    ytdlp_config = ytdlp_dir / "config"
    needs_js_config = True
    if ytdlp_config.exists():
        try:
            if "--js-runtimes" in ytdlp_config.read_text(encoding="utf-8"):
                needs_js_config = False
        except OSError:
            pass
    if needs_js_config:
        with ytdlp_config.open("a", encoding="utf-8") as f:
            f.write("--js-runtimes node\n")
        logger.info("yt_dlp_js_runtimes_configured")

    # 3. mcporter Exa MCP (search semantico, no API key)
    try:
        result = subprocess.run(
            ["mcporter", "config", "list"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if "exa" not in result.stdout.lower():
            subprocess.run(
                ["mcporter", "config", "add", "exa", "https://mcp.exa.ai/mcp"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            logger.info("exa_mcp_configured")
    except FileNotFoundError:
        logger.warning("mcporter_not_installed_skipping_exa_setup")
    except subprocess.TimeoutExpired:
        logger.warning("mcporter_config_timeout")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    setup_agent_reach()
    print("agent_reach_setup: OK")

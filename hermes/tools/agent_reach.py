"""Sprint 4 MVP-2 T11b (minimalista): wrapper sobre upstream tools.

Que hace este modulo (T11b minimalista):
1. Enum allowlist de 4 actions (read, youtube_transcript, github_search,
   rss_read). Validado por JSON Schema (provider rechaza valores fuera
   del enum ANTES de invocar la tool).
2. asyncio.create_subprocess_exec - no bloquea el event loop del bot.
3. Devuelve texto raw. El wrap en <tool_output>, truncate y regex de
   prompt injection se aplican en secure_execute (Sprint 5 T49).
4. file_not_found user-friendly si falta alguna upstream tool.

Lo que NO tiene (vs version anterior over-engineered):
- Sin semaforo: 1 user, tools por-turno, no concurrentes en practice.
- Sin LLM validator (capa 2): 1-2s latencia extra NO justificada para
  tools zero-config que el user invoca explicitamente. Si en algun
  momento scrapeamos contenido arbitrario sin supervision, se anade.

Sprint 5 T49: el wrapper ya no aplica wrap/truncate/regex manualmente.
Devuelve texto raw y deja que secure_execute centralice el defense
pipeline (consistencia con builtin tools y futuras read tools).

El LLM NO ve la implementacion. Solo ve 4 tools pre-construidas
con schemas estrictos (action enum + target string). Internamente,
cada action se mapea a un comando upstream:
- read: curl + Jina Reader (cero config, no API key)
- youtube_transcript: yt-dlp --write-auto-sub (extrae subs de YouTube)
- github_search: curl + api.github.com (rate limit 60/h sin auth)
- rss_read: python3 hermes/tools/scripts/rss_read.py (feedparser)
"""

from __future__ import annotations

import asyncio
import enum
import logging
import shutil
import time
from pathlib import Path
from typing import Any

from hermes.tools.security import ToolTimeout

logger = logging.getLogger(__name__)


# ===========================================================================
# Enum allowlist: actions que el LLM puede pedir
# ===========================================================================
# Anadir una action = anadir valor aqui + 1 linea al mapping ACTION_TO_CMD
# debajo. JSON Schema rechaza valores fuera del enum ANTES de invocar
# la tool (primera capa de defensa en provider), Python valida de nuevo
# (defense in depth en codigo).


class AgentReachAction(enum.StrEnum):
    READ = "read"
    YOUTUBE_TRANSCRIPT = "youtube_transcript"
    GITHUB_SEARCH = "github_search"
    RSS_READ = "rss_read"


# Mapping action -> comando upstream. Cada valor es la lista COMPLETA
# de args (sin el target, que se concatena al final). shell=False implica
# que la shell NO interpreta nada: cada arg es independiente.
#
# Notes sobre los comandos:
# - read: Jina Reader (https://r.jina.ai/) extrae contenido limpio de
#   cualquier URL en markdown. Cero config, sin API key, gratis.
# - youtube_transcript: yt-dlp + deno + yt-dlp-ejs (instalados en
#   Dockerfile). Flags:
#     --user-agent: UA moderno para que YouTube no nos trate como bot.
#     --js-runtimes deno: usa deno como JS engine para resolver
#       challenges. Sin deno, yt-dlp puede fallar con "Sign in to
#       confirm you're not a bot".
#     --extractor-args "youtube:player_client=android,web,ios":
#       prueba con android primero (rate limits separados), luego web,
#       luego ios. Cada cliente tiene su propio bucket de rate limit,
#       asi que el fallback ayuda cuando uno se satura.
#     --sleep-interval 5: throttle para no ser flagged.
#     -o /tmp/yt-transcript.%(ext)s: output predecible.
# - github_search: GitHub REST API publica. Rate limit 60/h por IP sin
#   auth (suficiente para uso personal). 10 resultados, ordenados por
#   stars. Para uso intensivo, configurar GH_TOKEN.
# - rss_read: script Python con feedparser. 5 items max, 300 chars
#   summary. JSON output. Errores a stderr.
_CHROME_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

ACTION_TO_CMD: dict[AgentReachAction, list[str]] = {
    AgentReachAction.READ: [
        "curl",
        "-sL",
        "--max-time",
        "20",
        "-H",
        "User-Agent: Oroimen/1.0",
    ],
    AgentReachAction.YOUTUBE_TRANSCRIPT: [
        "yt-dlp",
        "--write-auto-sub",
        "--skip-download",
        "--sub-format",
        "vtt",
        "--no-warnings",
        "--user-agent",
        _CHROME_USER_AGENT,
        "--js-runtimes",
        "deno:/usr/local/bin/deno",
        "--extractor-args",
        "youtube:player_client=android,web,ios",
        "--sleep-interval",
        "5",
        "--max-sleep-interval",
        "15",
        "--cookies",
        "/app/config/yt-cookies.txt",
        "-o",
        "/tmp/yt-transcript.%(ext)s",
    ],
    AgentReachAction.GITHUB_SEARCH: [
        "curl",
        "-sL",
        "--max-time",
        "20",
        "-H",
        "Accept: application/vnd.github.v3+json",
        "-H",
        "User-Agent: Oroimen/1.0",
    ],
    AgentReachAction.RSS_READ: [
        "python3",
        "/app/hermes/tools/scripts/rss_read.py",
    ],
}

# URL templates que necesitan el target inyectado (vs append simple).
# read: prepend "https://r.jina.ai/" al target
# github_search: URL con query params
_ACTION_TARGET_PREFIX: dict[AgentReachAction, str] = {
    AgentReachAction.READ: "https://r.jina.ai/",
    AgentReachAction.GITHUB_SEARCH: (
        "https://api.github.com/search/repositories" "?q={q}&sort=stars&order=desc&per_page=10"
    ),
}


# Schema para OpenAI tool calling: SOLO target.
# El action esta IMPLICITO en el nombre del tool (agent_reach_read
# -> action=read, agent_reach_youtube_transcript -> action=youtube).
# Si pusieramos action en el schema, el LLM lo pasaria y el registry
# hace fn(**arguments) lo que falla porque tool_callable solo acepta
# target. Defense in depth: el nombre del tool ES el allowlist.
AGENT_REACH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "target": {
            "type": "string",
            "description": (
                "URL (for read/youtube_transcript/rss_read) or query " "string (for github_search)."
            ),
        },
    },
    "required": ["target"],
    # _unused previene "function name or parameters is empty (2013)"
    # que el provider lanza cuando properties esta vacio o solo tiene
    # un enum sin nada mas. Es un workaround del provider, no del schema.
    "_unused": {"type": "string", "description": "Internal: prevents empty schema error"},
}


# ===========================================================================
# Configuracion
# ===========================================================================

_DEFAULT_TIMEOUT_S = 25.0
_YT_SUB_PATTERN = "/tmp/yt-transcript.*.vtt"
# Path al cookies file de YouTube dentro del container (bind-mount del
# host en an operator-specific Compose override). Si el archivo no existe, yt-dlp
# se ejecuta sin cookies (fallback) y el LLM reporta el rate limit.
_YT_COOKIES_PATH = "/app/config/yt-cookies.txt"


# ===========================================================================
# Helpers
# ===========================================================================


def _build_cmd(action: AgentReachAction, target: str) -> list[str]:
    """Construye el comando completo para una action + target."""
    base = list(ACTION_TO_CMD[action])  # copia
    if action in _ACTION_TARGET_PREFIX:
        if action == AgentReachAction.GITHUB_SEARCH:
            # URL-encode el query (espacios, etc.)
            from urllib.parse import quote_plus

            url = _ACTION_TARGET_PREFIX[action].format(q=quote_plus(target))
        else:
            url = _ACTION_TARGET_PREFIX[action] + target
        base.append(url)
    else:
        base.append(target)
    return base


def _read_yt_sub_file(target_url: str) -> str | None:
    """Lee el archivo VTT que yt-dlp genero. Retorna None si no existe."""
    matches = sorted(Path("/tmp").glob("yt-transcript.*.vtt"))
    if not matches:
        return None
    # Tomar el mas reciente (por mtime)
    latest = max(matches, key=lambda p: p.stat().st_mtime)
    try:
        content = latest.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    finally:
        # Limpiar: yt-dlp puede dejar varios archivos (.vtt, .en.vtt, etc.)
        import contextlib

        for m in matches:
            with contextlib.suppress(OSError):
                m.unlink()
    return content


# ===========================================================================
# Core: ejecutar upstream tool con subprocess async
# ===========================================================================


async def _run_agent_reach(
    action: AgentReachAction,
    target: str,
    *,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
) -> str:
    """Ejecuta el comando upstream y devuelve texto raw.

    Sprint 5 T49: el wrap en <tool_output>, truncate y regex de
    prompt injection se aplican en secure_execute (no aquí). Esta
    función solo hace el "heavy lifting": validación, subprocess,
    lectura de VTT, cleanup. Devuelve texto raw que secure_execute
    procesa.

    Raises:
        ValueError: si action no es AgentReachAction valido.
        FileNotFoundError: si el binario upstream no está instalado.
        RuntimeError: si el subprocess falla (returncode != 0) o no
            produce output (ej. yt-dlp sin subs).
        ToolTimeout: si el subprocess excede timeout_s.

    Returns:
        Texto raw del output (ej. contenido VTT, JSON de GitHub, etc.).
    """
    # 1. Validar action (defense in depth)
    if not isinstance(action, AgentReachAction):
        raise ValueError(f"Action '{action}' no es AgentReachAction valido")
    if action not in ACTION_TO_CMD:
        raise ValueError(f"Action '{action}' no tiene comando upstream")

    # Verificar que el binario/script existe (fail-fast)
    cmd = _build_cmd(action, target)
    if action != AgentReachAction.YOUTUBE_TRANSCRIPT:
        # yt-dlp produce el binario, lo verificamos al final
        binary = cmd[0]
        if binary == "python3":
            # Para python3 + script, verificar el script
            script = cmd[1]
            if not Path(script).exists():
                raise FileNotFoundError(f"Script no encontrado: {script}")
        elif not shutil.which(binary):
            raise FileNotFoundError(
                f"Tool '{binary}' no esta instalado. " f"Instala la dependencia correspondiente."
            )
    else:
        # youtube_transcript: si el cookies file no existe (bind-mount
        # no aplicado, o user no ha exportado cookies), omitir --cookies
        # para que yt-dlp se ejecute sin cookies (fallback). Si las
        # cookies existen, yt-dlp las usa para evitar el challenge y
        # el rate limit de YouTube. El wrapper siempre incluye el flag
        # en _build_cmd, pero aqui filtramos dinamicamente.
        if "--cookies" in cmd:
            cookies_idx = cmd.index("--cookies")
            cookies_path = cmd[cookies_idx + 1]
            if not Path(cookies_path).exists():
                logger.info(
                    "yt_cookies_not_found_running_without",
                    extra={"path": cookies_path},
                )
                # Quitar --cookies y su valor
                cmd = cmd[:cookies_idx] + cmd[cookies_idx + 2 :]

    logger.info(
        "agent_reach_subprocess_start",
        extra={"action": action.value, "target_preview": target[:80]},
    )

    # 2-3. Subprocess async
    start = time.perf_counter()
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd="/tmp",
    )

    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout_s,
        )
    except TimeoutError as exc:
        proc.kill()
        await proc.wait()
        raise ToolTimeout(f"agent-reach {action.value} excedio {timeout_s}s") from exc

    stdout = stdout_bytes.decode("utf-8", errors="replace").strip()
    stderr = stderr_bytes.decode("utf-8", errors="replace").strip()
    latency_ms = int((time.perf_counter() - start) * 1000)

    logger.info(
        "agent_reach_subprocess_done",
        extra={
            "action": action.value,
            "returncode": proc.returncode,
            "stdout_chars": len(stdout),
            "stderr_chars": len(stderr),
            "latency_ms": latency_ms,
        },
    )

    # 4. youtube_transcript: leer el archivo VTT generado
    if action == AgentReachAction.YOUTUBE_TRANSCRIPT:
        sub_content = _read_yt_sub_file(target)
        if sub_content is not None:
            stdout = sub_content
        elif not stdout:
            # yt-dlp no genero archivo y stdout vacio = fallo
            error_msg = stderr[:500] or "yt-dlp no genero transcript (subs no disponibles?)"
            raise RuntimeError(f"yt_dlp_no_subtitle: {error_msg}")

    # 5. Si fallo, devolver stderr como RuntimeError
    if proc.returncode != 0 and not stdout:
        error_msg = stderr[:500] or f"comando exited {proc.returncode}"
        raise RuntimeError(f"Error ejecutando {action.value}: {error_msg}")

    return stdout


# ===========================================================================
# Tool factories: 4 tools pre-construidas con schemas estrictos
# ===========================================================================


def make_agent_reach_tools() -> list[dict[str, Any]]:
    """Devuelve 4 tool specs pre-construidas para las upstream tools.

    Cada spec tiene name, description, schema (con action enum) y
    el callable wrapped. El callable es una coroutine que recibe
    (target: str) y devuelve el output wrapeado en <tool_output>.

    El LLM solo ve el schema (action enum + target string); internamente
    mapeamos al comando upstream correspondiente.
    """

    def _make_tool(
        action: AgentReachAction,
        description: str,
    ) -> dict[str, Any]:
        async def tool_callable(target: str, **_unused: Any) -> str:
            # El LLM solo envia target. El action viene del closure
            # (cada tool tiene un action FIJO por nombre). Defense in
            # depth: aunque el LLM envie "action" o cualquier otro kwarg,
            # el registry usa fn(**arguments) y los kwargs extra se
            # ignoran silenciosamente con **_unused.
            return await _run_agent_reach(action, target)

        return {
            "name": f"agent_reach_{action.value}",
            "description": description,
            "schema": AGENT_REACH_SCHEMA,
            "callable": tool_callable,
        }

    return [
        _make_tool(
            AgentReachAction.READ,
            "Lee cualquier URL publica via Jina Reader (curl). Devuelve el "
            "contenido en Markdown/texto plano. Util para articulos, "
            "blogs, docs, Wikipedia. Cero config.",
        ),
        _make_tool(
            AgentReachAction.YOUTUBE_TRANSCRIPT,
            "Extrae el transcript (subtitulos auto-generados) de un video "
            "de YouTube via yt-dlp. Devuelve el texto en formato VTT. "
            "Util para resumir videos sin verlos.",
        ),
        _make_tool(
            AgentReachAction.GITHUB_SEARCH,
            "Busca repositorios publicos en GitHub via API REST (curl). "
            "Rate limit 60/h por IP sin auth (suficiente para uso "
            "personal). Devuelve top 10 resultados con name, description, "
            "url, stars.",
        ),
        _make_tool(
            AgentReachAction.RSS_READ,
            "Lee un feed RSS o Atom (feedparser) y devuelve los ultimos "
            "5 items con title, link, published, summary. Util para "
            "monitorizar blogs, podcasts, news.",
        ),
    ]

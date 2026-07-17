"""Pipeline de seguridad para tools (Sprint 4 T6 — Defense in Depth).

Por qué existe: las tools pueden devolver contenido externo (web scrapes,
output de subprocesos) que potencialmente incluye **prompt injection**
(instrucciones ocultas que intentan manipular al LLM).

Defense in depth con 3 capas:

1. **XML delimiters** (`wrap_in_xml`): envuelve output en
   `<tool_output source="..." timestamp="...">...</tool_output>`. El
   LLM aprende a NO ejecutar instrucciones dentro de `<tool_output>`.
   Esto se refuerza en el system prompt (T8).

2. **Truncate 2500 chars** (`truncate_output`): limita el tamaño del
   output. Reduce la viabilidad de payloads largos de jailbreak y
   ahorra tokens. Hardcoded porque 2500 es el límite arquitectónico,
   no configurable.

3. **Pre-filtro regex** (`is_suspicious`): detecta patrones obvios
   de prompt injection SIN llamar al LLM. Cubre ~80% de ataques
   conocidos. Si detecta → marca como unsafe, no se envía al LLM.

Pipeline completo (`secure_execute`):
    1. result = await execute_with_timeout(tool, args, timeout)
    2. result = truncate_output(result, 2500)
    3. if is_suspicious(result): result = "[FILTERED: prompt injection]"
    4. result = wrap_in_xml(tool_name, result)
    5. Return como mensaje role: "tool" (quarantine, no en system prompt)
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hermes.config import Settings
    from hermes.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

_MAX_OUTPUT_CHARS = 2500

# Patrones de prompt injection. Cubren los ataques más comunes según
# https://owasp.org/www-project-top-10-for-large-language-model-applications/
# y https://github.com/agentica-project/llm-prompt-injection-dataset
_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    # "Ignore/forget/disregard [previous/all/everything] [instructions/prompts/context/above]"
    # Cubre: "ignore previous instructions", "forget everything above",
    # "disregard all prior context", etc.
    re.compile(
        r"(?i)(ignore|forget|disregard)\s+"
        r"(all|previous|prior|above|earlier|everything)\s+"
        r"(instructions?|prompts?|context|above|before)?"
    ),
    # "You are now X" / "Act as X"
    re.compile(r"(?i)(you\s+are\s+now|act\s+as)\s+(a|an)\s+"),
    # Delimiters de modelos de chat (ChatML, etc.)
    re.compile(r"<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>"),
    # System prompt injection
    re.compile(r"(?i)system\s*prompt\s*:"),
    # "New instructions:" / "Updated instructions:"
    re.compile(r"(?i)\bnew\s+instructions?\b\s*:"),
    # "From now on..." que intenta resetear comportamiento
    re.compile(r"(?i)from\s+now\s+on\s*,?\s+(you|your|the\s+assistant)"),
    # Template injection con system/prompt
    re.compile(r"\{\{.*?(system|prompt|instructions?).*?\}\}"),
)


class ToolTimeout(Exception):
    """La tool excedió el timeout configurado."""


def wrap_in_xml(source: str, content: str, **meta: Any) -> str:
    """Envuelve content en delimitadores XML para aislar del LLM.

    Formato:
        <tool_output source="<source>" timestamp="<iso>" <meta>>
        <content>
        </tool_output>

    El system prompt (T8) instruye al LLM a NO ejecutar instrucciones
    dentro de <tool_output>.

    Args:
        source: nombre de la tool (e.g. "get_weather", "agent_reach").
        content: contenido a envolver.
        **meta: atributos adicionales (e.g. url="...", latency_ms=42).

    Returns:
        String con el contenido envuelto en XML.
    """
    timestamp = datetime.now(UTC).isoformat()
    meta_attrs = " ".join(f'{k}="{_escape_attr(str(v))}"' for k, v in meta.items())
    meta_section = f" {meta_attrs}" if meta_attrs else ""
    escaped_content = _escape_xml(content)
    return (
        f'<tool_output source="{_escape_attr(source)}" timestamp="{timestamp}"{meta_section}>\n'
        f"{escaped_content}\n"
        f"</tool_output>"
    )


def _escape_xml(text: str) -> str:
    """Escapa caracteres XML especiales para evitar injection via content."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _escape_attr(text: str) -> str:
    """Escapa atributo XML (sin `<>` pero sí comillas)."""
    return text.replace("&", "&amp;").replace('"', "&quot;")


def truncate_output(output: str, max_chars: int = _MAX_OUTPUT_CHARS) -> tuple[str, bool]:
    """Trunca output si excede max_chars.

    Args:
        output: string a truncar.
        max_chars: límite (default 2500).

    Returns:
        Tupla (contenido_truncado, fue_truncado_bool).
    """
    if len(output) <= max_chars:
        return output, False
    return output[:max_chars], True


def is_suspicious(output: str) -> bool:
    """Detecta patrones de prompt injection en output externo.

    Usa regex (sin LLM) para máxima velocidad. Cobertura ~80% de
    ataques conocidos. NO es bulletproof — por eso hay una Capa 2
    con sub-LLM validator (Sprint 4 MVP-2 T11) para casos sutiles.

    Args:
        output: texto a analizar.

    Returns:
        True si algún patrón de injection matchea.
    """
    return any(p.search(output) for p in _INJECTION_PATTERNS)


async def execute_with_timeout(
    tool_fn: Callable[..., Awaitable[Any]],
    args: dict[str, Any],
    timeout_s: float,
) -> Any:
    """Ejecuta una tool con timeout estricto.

    Args:
        tool_fn: función async de la tool.
        args: kwargs para la función.
        timeout_s: segundos antes de timeout.

    Returns:
        Resultado de la tool.

    Raises:
        ToolTimeout: si la tool excede timeout_s.
    """
    coro = tool_fn(**args) if not _is_bound_with_args(tool_fn) else tool_fn(*(), **args)
    try:
        return await asyncio.wait_for(coro, timeout=timeout_s)
    except TimeoutError as exc:
        raise ToolTimeout(f"Tool {getattr(tool_fn, '__name__', '?')} excedió {timeout_s}s") from exc


def _is_bound_with_args(fn: Callable[..., Any]) -> bool:
    """¿La función ya está parcialmente aplicada (functools.partial)?"""
    return hasattr(fn, "keywords") or hasattr(fn, "args")


@dataclass(frozen=True)
class ToolExecutionResult:
    """Resultado estructurado de secure_execute.

    Por qué dataclass y no string: secure_execute captura internamente
    TODAS las excepciones y las envuelve en XML. Si retornara un string,
    el caller no podría distinguir éxito de error y corrompería la
    telemetría/auditoría (success=True en tool_calls DB sería siempre
    True, incluso en errores).

    Sprint 5 T49.
    """

    content: str  # XML-wrapped output (o error envuelto en XML)
    success: bool  # True si la tool ejecutó sin error
    error: str | None  # Mensaje de error si success=False
    status: str  # "safe" | "timeout" | "error" | "filtered"
    truncated: bool  # True si output fue truncado a max_chars
    latency_ms: int  # Tiempo de ejecución de la tool


def get_max_chars_for_tool(
    tool_name: str,
    registry: ToolRegistry,
    settings: Settings,
) -> int:
    """Devuelve el max_chars apropiado consultando ToolSpec en el registry.

    Categorías (declaradas en ToolSpec.tool_category):
    - "read" (agent_reach_*, web_scrape futuro, etc.):
      settings.read_tool_max_chars (default 150K)
    - "system" (get_weather, search_vault, get_current_time, get_system_status):
      settings.system_tool_max_chars (default 2500)

    Por qué metadata-driven (no name-based): si en el futuro agregamos
    una nueva read tool (ej. web_scrape en builtin.py), no hay que
    tocar esta función. La categoría vive en ToolSpec.

    Fail-safe: tools no registradas o sin tool_category caen en
    system_tool_max_chars (2500, conservador). Mejor truncar de más
    que permitir output sin control.

    Sprint 5 T49.
    """
    tool_spec = registry.get_spec(tool_name)
    if tool_spec is not None and tool_spec.tool_category == "read":
        return settings.read_tool_max_chars
    return settings.system_tool_max_chars


async def secure_execute(
    tool_name: str,
    tool_fn: Callable[..., Awaitable[Any]],
    args: dict[str, Any],
    *,
    timeout_s: float = 30.0,
    max_chars: int = _MAX_OUTPUT_CHARS,
) -> ToolExecutionResult:
    """Pipeline completo de seguridad para una tool.

    Pasos:
    1. Ejecuta con timeout.
    2. Trunca a max_chars.
    3. Detecta prompt injection (regex).
    4. Envoltorio XML (con escape ciego e incondicional del contenido).
    5. Si hubo error o injection, loggea.

    Args:
        tool_name: nombre de la tool (para logs y XML).
        tool_fn: función async de la tool.
        args: kwargs para la función.
        timeout_s: timeout en segundos.
        max_chars: límite de caracteres del output.

    Returns:
        ToolExecutionResult con content (XML-wrapped), success, error,
        status, truncated, latency_ms. NUNCA lanza excepción al caller
        (las excepciones se envuelven en el resultado con success=False).

    Sprint 5 T49: la firma cambió de -> str a -> ToolExecutionResult.
    Esto permite al caller persistir success/error correctamente en DB
    (sin esto, el bloque except Exception en el caller jamás se
    ejecutaría y success=True sería siempre True, bug de telemetría).
    """
    start = time.perf_counter()

    try:
        result = await execute_with_timeout(tool_fn, args, timeout_s)
    except ToolTimeout as exc:
        latency_ms = int((time.perf_counter() - start) * 1000)
        return ToolExecutionResult(
            content=wrap_in_xml(
                tool_name,
                f"[TIMEOUT] {exc}",
                status="timeout",
            ),
            success=False,
            error=str(exc),
            status="timeout",
            truncated=False,
            latency_ms=latency_ms,
        )
    except Exception as exc:
        latency_ms = int((time.perf_counter() - start) * 1000)
        return ToolExecutionResult(
            content=wrap_in_xml(
                tool_name,
                f"[ERROR] {type(exc).__name__}: {exc}",
                status="error",
            ),
            success=False,
            error=f"{type(exc).__name__}: {exc}",
            status="error",
            truncated=False,
            latency_ms=latency_ms,
        )

    result_str = str(result)
    truncated_result, was_truncated = truncate_output(result_str, max_chars)
    latency_ms = int((time.perf_counter() - start) * 1000)

    if is_suspicious(truncated_result):
        logger.warning(
            "secure_execute_output_filtered",
            extra={"tool": tool_name, "output_chars": len(truncated_result)},
        )
        return ToolExecutionResult(
            content=wrap_in_xml(
                tool_name,
                "[FILTERED] prompt injection detectado en output",
                status="filtered",
            ),
            success=False,
            error="prompt_injection_detected",
            status="filtered",
            truncated=was_truncated,
            latency_ms=latency_ms,
        )

    return ToolExecutionResult(
        content=wrap_in_xml(
            tool_name,
            truncated_result,
            status="safe",
            truncated=str(was_truncated).lower(),
        ),
        success=True,
        error=None,
        status="safe",
        truncated=was_truncated,
        latency_ms=latency_ms,
    )


__all__ = [
    "_INJECTION_PATTERNS",
    "_MAX_OUTPUT_CHARS",
    "ToolExecutionResult",
    "ToolTimeout",
    "execute_with_timeout",
    "get_max_chars_for_tool",
    "is_suspicious",
    "secure_execute",
    "truncate_output",
    "wrap_in_xml",
]

"""Prompt templates + sanitize_summary helper para research jobs.

Ver TDD_S14_DEEP_RESEARCH.md §16 y §16.1.
"""

from __future__ import annotations

import json
import logging
import re
from string import Template

_LOG = logging.getLogger(__name__)

PER_SOURCE_PROMPT = Template(
    """You are analyzing a single source for a deep research task.

ORIGINAL QUERY: $query

SOURCE URL: $url
SOURCE CONTENT (cleaned):
\"\"\"
$source
\"\"\"

Write a structured summary of this source that:
1. Identifies the 3-5 most relevant facts/findings for the original query
2. Notes the source's authority/bias if apparent
3. Quotes 1-2 short passages verbatim (under 50 words each) with attribution
4. Outputs in markdown with ## headings

Target: 500-1500 words. Be concise but substantive. Do not editorialize.
"""
)

FINAL_SYNTH_PROMPT = Template(
    """You are synthesizing a deep research report from multiple source summaries.

ORIGINAL QUERY: $query

SOURCE SUMMARIES:
$summaries

Write a coherent research report that:
1. Opens with a 2-3 sentence summary of findings
2. Organizes by theme, not by source
3. Uses [1], [2], etc. citation markers tied to the source number
4. Has sections: ## Summary, ## Key Findings, ## Sources, ## Caveats
5. Notes any disagreements between sources
6. Cites the source number for every non-trivial claim

Target: 1500-3000 words. Be thorough but not bloated. Avoid filler.
"""
)


# Sanitización anti-thinking-blocks (TDD §16.1).
# DeepSeek-V3 y otros modelos reasoning exponen `reasoning_content` separado
# de `content`. Si Phase 3 almacena ambos y Phase 4 concatena summaries
# leyendo en crudo, el FINAL_SYNTH_PROMPT recibe miles de tokens de
# "monólogo interno" del modelo de respaldo → output final emula formato
# de pensamiento o alucina mezclando monólogo con datos.
_THINKING_BLOCK_PATTERNS = [
    re.compile(r"<think>.*?</think>", re.DOTALL),  # deepseek-v3 native
    re.compile(r"<\|thinking\|>.*?</\|thinking\|>", re.DOTALL),  # ChatML tokens
    re.compile(r"<reasoning>.*?</reasoning>", re.DOTALL),  # fallback XML
    re.compile(r"```reasoning.*?```", re.DOTALL),  # markdown fence variant
]

# Bloques JSON con campo `thinking` (formato raw response de algunos modelos).
# Q2 verifier finding: regex simple no maneja `\"escaped\"` ni `{}` anidados.
# El detector ahora es un bracket-counter escrito en Python (ver ``_strip_json_thinking_block``).
_THINKING_KEY_PATTERN = re.compile(r'\{\s*"thinking"\s*:', re.DOTALL)


def _find_matching_close_brace(text: str, start: int) -> int | None:
    """Encuentra el '}' que cierra el '{' en ``start`` respetando strings JSON.

    Tracking de brace depth, in_string state, y backslash escape (``\\\\"`` no
    cierra el string). No maneja comentarios JSONC — el contenido LLM no los
    emite. Devuelve None si la estructura queda desbalanceada (LLM truncó
    la respuesta mid-stream).
    """
    if start < 0 or start >= len(text) or text[start] != "{":
        return None
    depth = 1
    in_string = False
    escape = False
    i = start + 1
    n = len(text)
    while i < n:
        c = text[i]
        if escape:
            escape = False
        elif c == "\\" and in_string:
            escape = True
        elif c == '"':
            in_string = not in_string
        elif not in_string:
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return i
        i += 1
    return None


def _strip_json_thinking_block(text: str) -> str:
    """Strips ``{"thinking": "..."}`` blocks via bracket counting (Q2 fix).

    Reemplaza la regex anterior (``_JSON_THINKING_PATTERN``) que fallaba en
    dos casos reales:
      1. ``"thinking": "long content with \\"escaped\\" quotes"`` —
         el ``.*?"`` no-greedy corta en el primer ``"`` interno.
      2. ``"thinking": "value with {nested} braces"`` — el ``.*?``
         se detiene antes del primer ``}`` mid-string.

    Algoritmo:
      - Encontrar candidatos ``{\"thinking\":`` con regex.
      - Para cada candidato, caminar hacia adelante contando braces y
        respetando string literals (con backslash-escape).
      - Si no se encuentra el ``}`` de cierre, log warning y dejar el
        texto intacto (defensive: la próxima regex candidate retry will
        no entrar en loop porque avanzamos i += 1 en el cursor).

    Después de cada match, también strip una coma colgante y el newline
    + indent del siguiente campo (limpieza cosmética; sin esto quedan
    comas sueltas en el output del Phase 4 final synth).
    """
    result_parts: list[str] = []
    cursor = 0
    n = len(text)
    for match in _THINKING_KEY_PATTERN.finditer(text):
        result_parts.append(text[cursor : match.start()])
        end_idx = _find_matching_close_brace(text, match.start())
        if end_idx is None:
            _LOG.warning(
                "sanitize_summary: JSON thinking block starting at offset %d "
                "has no matching '}' (truncated/malformed). Leaving text intact.",
                match.start(),
            )
            # Mantener el '{' literal y avanzar el cursor un caracter para
            # que finditer no se atasque en bucle infinito en esta posición.
            result_parts.append(text[match.start()])
            cursor = match.start() + 1
            continue
        # Consumir el bloque entero + posible coma colgante + un único
        # whitespace run (newline + indent del campo siguiente).
        cursor = end_idx + 1
        if cursor < n and text[cursor] == ",":
            cursor += 1
        if cursor < n and text[cursor] == "\n":
            cursor += 1
            while cursor < n and text[cursor] in (" ", "\t"):
                cursor += 1
        elif cursor < n and text[cursor] in (" ", "\t"):
            cursor += 1
    result_parts.append(text[cursor:])
    return "".join(result_parts)


def sanitize_summary(raw: str) -> str:
    """Extrae solo el campo `content`-equivalente de un summary.

    Defense in depth (TDD §16.1) contra thinking blocks que contaminan
    el output final. Aplica en:
      - Phase 3 output (antes de almacenar/retornar)
      - Phase 4 input (defense in depth, por si T51 almacena ambos campos)
      - Recovery re-read (por si checkpoint tiene data pre-sanitización)

    Pasos:
      1. Si el input es JSON (formato deepseek raw response con
         reasoning_content), parsear y devolver solo el campo `content`.
      2. Si el input es texto plano, strip thinking blocks via regex.
      3. Whitespace cleanup.

    Args:
        raw: texto crudo del LLM (puede tener thinking blocks o ser JSON).

    Returns:
        texto limpio (sin thinking, con whitespace normalizado).
    """
    if not raw:
        return ""

    # Paso 1: ¿es JSON? Intentar parsear y extraer `content` (deepseek format).
    raw_stripped = raw.strip()
    if raw_stripped.startswith("{") and raw_stripped.endswith("}"):
        try:
            data = json.loads(raw_stripped)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            if "content" in data:
                return str(data["content"]).strip()
            # Si no hay "content", descartar campos sospechosos y quedarnos
            # con el resto del dict como string (aún útil como summary).
            safe = {
                k: v
                for k, v in data.items()
                if k not in ("reasoning_content", "reasoning", "thinking", "thought")
            }
            return str(safe).strip()

    # Paso 2: strip thinking blocks via regex.
    cleaned = raw
    for pattern in _THINKING_BLOCK_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    # Strip bloques JSON `{"thinking": "..."}` via bracket-counting parser
    # (Q2 verifier finding): maneja escaped quotes y nested braces que
    # la regex simple omitia.
    cleaned = _strip_json_thinking_block(cleaned)

    # Paso 3: whitespace cleanup.
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned

"""Telegram message chunking (Sprint 5 T50).

Por que existe: Telegram tiene un limite de 4096 chars por mensaje.
Con T49 (150K read budget), el LLM genera resumenes mas largos que
cruzan ese limite. Sin chunking, el bot se queda en "Pensando..."
para siempre (edit_text y message.answer fallan con MESSAGE_TOO_LONG).

Algoritmo:
1. Walk through text from start, tracking if we're inside a code block.
2. At each "safe boundary" (outside a code block, at a natural
   cut point: paragraph/line/space), record the position.
3. Emit chunks at safe boundaries, ensuring no chunk exceeds max_len.
4. If no safe boundary within max_len, hard cut + close/reopen code
   block to keep ``` balanced.

Limit:
- max_len default 3900 (margen de ~196 chars para overhead de
  formato markdown, expansion de emojis UTF-8, caracteres especiales).
"""

from __future__ import annotations

import re

# Margen conservador: 4096 - 196 = 3900 chars max por chunk.
TELEGRAM_MAX_MSG = 3900

# Margen reservado para "\n```" al hacer hard cut dentro de un code block.
# 4 = "\n" + "```" + posible "\n" extra.
_HARD_CUT_MARGIN = 4


def _find_natural_cut(text: str, max_len: int) -> int:
    """Encuentra el mejor punto de corte natural respetando max_len.

    Heuristica (en orden):
    1. Ultimo parrafo (\\n\\n) antes del limite
    2. Ultima linea (\\n) antes del limite
    3. Ultimo espacio antes del limite
    4. Hard cut en max_len
    """
    cut = text.rfind("\n\n", 0, max_len)
    if cut > 0:
        return cut
    cut = text.rfind("\n", 0, max_len)
    if cut > 0:
        return cut
    cut = text.rfind(" ", 0, max_len)
    if cut > 0:
        return cut
    return max_len


def _find_safe_cut(text: str, start: int, max_len: int) -> tuple[int, bool]:
    """Encuentra un cut point que NO deje el chunk dentro de un code block.

    Args:
        text: texto completo.
        start: posicion donde empieza el chunk actual.
        max_len: tamano maximo del chunk desde start.

    Returns:
        Tupla (safe_cut, in_block_at_target):
        - safe_cut: posicion (en text) donde es seguro cortar, o -1
          si no hay un "fuera de bloque" dentro del rango
        - in_block_at_target: True si al final del rango [start, target)
          estamos dentro de un code block sin cerrar

    IMPORTANTE: in_block_at_target refleja el estado al final del
    RANGO, no al final del texto completo. Esto es importante porque
    el chunk termina en target, no al final del texto.
    """
    target = start + max_len
    n = len(text)
    in_block = False
    last_exit = -1
    i = start
    # Walk through [start, target) y trackear estado
    while i + 2 < target and i + 2 < n:
        if text[i : i + 3] == "```":
            in_block = not in_block
            i += 3
            if not in_block:
                last_exit = i
        else:
            i += 1
    # NOTA: no seguimos trackeando mas alla de target. El estado
    # al final del rango es lo que importa para el chunk actual.
    return last_exit, in_block


def _find_natural_cut_in_range(text: str, start: int, max_len: int) -> int:
    """Encuentra un cut natural (paragraph/line/space) dentro de [start, start+max_len]."""
    end = min(start + max_len, len(text))
    sub = text[start:end]
    cut = sub.rfind("\n\n")
    if cut > 0:
        return start + cut
    cut = sub.rfind("\n")
    if cut > 0:
        return start + cut
    cut = sub.rfind(" ")
    if cut > 0:
        return start + cut
    return start + max_len


def split_message(text: str, max_len: int = TELEGRAM_MAX_MSG) -> list[str]:
    """Divide text en chunks respetando estructura y code blocks.

    Args:
        text: texto a dividir.
        max_len: tamano maximo por chunk (default 3900).

    Returns:
        Lista de chunks. Lista vacia si text es vacio o solo whitespace.
        Cada chunk tiene ``` balanceados (parser de Telegram no rompe).
    """
    if not text or not text.strip():
        return []

    if len(text) <= max_len:
        return [text]

    chunks: list[str] = []
    pos = 0
    n = len(text)

    while pos < n:
        # Si el resto cabe, devolver
        if n - pos <= max_len:
            chunks.append(text[pos:])
            break

        # Intentar encontrar un cut "fuera de code block" dentro del rango
        safe_cut, in_block = _find_safe_cut(text, pos, max_len)

        if safe_cut > pos:
            # Encontramos un cierre de code block. Cortamos ahi.
            chunk = text[pos:safe_cut].rstrip()
            chunks.append(chunk)
            pos = safe_cut
        elif in_block:
            # Estamos dentro de un code block sin cerrar en el rango.
            # Estrategia depende de donde esta la apertura ```.
            open_idx = text.find("```", pos, pos + max_len)
            if open_idx > pos:
                # La apertura esta en medio del rango. Cortamos ANTES
                # de la apertura (asi el chunk no tiene ``` y el
                # remainder empieza con la apertura, manteniendo balance).
                chunk = text[pos:open_idx].rstrip()
                chunks.append(chunk)
                text = text[open_idx:]
                n = len(text)
                pos = 0
            else:
                # La apertura esta al inicio (o no se encontro, edge case).
                # Hard cut + close + reopen.
                cut_pos = pos + max_len - _HARD_CUT_MARGIN
                chunk = text[pos:cut_pos] + "\n```"
                chunks.append(chunk)
                text = "```\n" + text[cut_pos:]
                n = len(text)
                pos = 0
        else:
            # Fuera de code blocks: usar corte natural (paragraph/line/space).
            cut = _find_natural_cut_in_range(text, pos, max_len)
            chunk = text[pos:cut].rstrip()
            chunks.append(chunk)
            pos = cut

    return chunks


def strip_markdown(text: str) -> str:
    """Limpia caracteres de formato Markdown para fallback a texto plano.

    Usado cuando un chunk falla por "can't parse entities" en Telegram.
    Trade-off: el chunk pierde formato (bold, italic, code highlighting,
    links) pero la informacion sigue legible.

    Args:
        text: texto con posible formato Markdown.

    Returns:
        Texto sin formato: bold, italic, code blocks, links, headers.
    """
    # Code blocks: ```code``` → code (multiline flag para que . no matchee \n)
    text = re.sub(r"```(\w*\n)?(.*?)```", r"\2", text, flags=re.DOTALL)
    # Inline code: `code` → code
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Bold: **text** o __text__ → text
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"__(.+?)__", r"\1", text)
    # Italic: *text* o _text_ → text
    # (despues de bold para no capturar **)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"(?<![a-zA-Z0-9])_([^_]+?)_(?![a-zA-Z0-9])", r"\1", text)
    # Links: [text](url) → text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Headers: # text, ## text, etc. → text
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    return text


__all__ = [
    "TELEGRAM_MAX_MSG",
    "split_message",
    "strip_markdown",
]

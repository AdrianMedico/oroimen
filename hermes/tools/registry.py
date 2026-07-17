"""ToolRegistry: despacha tool calls a herramientas registradas."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

logger = logging.getLogger(__name__)

# Categorías de tool. Determinan el max_chars budget en el security pipeline
# (ver get_max_chars_for_tool en hermes/tools/security.py).
# - "read": tools que devuelven contenido externo potencialmente grande
#   (artículos, transcripts, RSS, search). Budget: 150K chars por defecto.
# - "system": tools que devuelven info corta del sistema (status, weather,
#   time, vault search). Budget: 2500 chars por defecto.
ToolCategory = Literal["read", "system"]


@dataclass
class ToolSpec:
    """Especificación de una tool registrada (para enviar al LLM).

    Attributes:
        name: nombre único de la tool.
        description: descripción legible para el LLM (en el system prompt).
        schema: JSON Schema de los parámetros (formato OpenAI). None si
            la tool no acepta parámetros.
        tool_category: categoría de la tool. Default "system" (fail-safe:
            el caller debe declarar explícitamente si es "read"). Usado
            por el security pipeline para determinar el max_chars budget.
    """

    name: str
    description: str = ""
    schema: dict | None = None
    tool_category: ToolCategory = "system"


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Callable[..., Any]] = {}
        self._specs: dict[str, ToolSpec] = {}

    def register(
        self,
        name: str,
        fn: Callable[..., Any],
        *,
        description: str = "",
        schema: dict | None = None,
        tool_category: ToolCategory = "system",
    ) -> None:
        """Registra una tool.

        Args:
            name: nombre único.
            fn: función (sync o async) que implementa la tool.
            description: descripción para el LLM.
            schema: JSON Schema de parámetros. Si es None, la tool se
                omite de `tool_schemas()` (no se envía al LLM).
            tool_category: "read" si devuelve contenido externo grande
                (articles, transcripts, RSS), "system" si devuelve info
                corta del sistema. Default "system" (conservador).
        """
        self._tools[name] = fn
        self._specs[name] = ToolSpec(
            name=name,
            description=description,
            schema=schema,
            tool_category=tool_category,
        )
        # NOTA: NO usamos extra={"name": ...} porque 'name' es campo
        # reservado de LogRecord en Python 3.14+ y KeyError.
        logger.info(
            "tool_registered: %s (has_schema=%s, category=%s)",
            name,
            schema is not None,
            tool_category,
        )

    def has(self, name: str) -> bool:
        return name in self._tools

    def get_tool_fn(self, name: str) -> Callable[..., Any] | None:
        """Devuelve el callable raw de una tool, o None si no existe.

        Usado por el security pipeline (secure_execute) que necesita
        el callable para ejecutarlo con timeout + truncate + wrap.
        Diferencia con execute(): execute() corre la tool; get_tool_fn
        solo la devuelve. secure_execute maneja el ciclo de vida.
        """
        return self._tools.get(name)

    async def execute(self, name: str, arguments: dict) -> Any:
        if not self.has(name):
            raise KeyError(f"Unknown tool: {name}")
        fn = self._tools[name]
        return await fn(**arguments) if _is_coro(fn) else fn(**arguments)

    def get_spec(self, name: str) -> ToolSpec | None:
        """Devuelve la ToolSpec de una tool, o None si no existe.

        Usado por get_max_chars_for_tool para consultar tool_category
        sin acoplamiento al nombre.
        """
        return self._specs.get(name)

    def list_tools(self) -> list[str]:
        return sorted(self._tools.keys())

    def list_specs(self) -> list[ToolSpec]:
        return [self._specs[name] for name in sorted(self._specs.keys())]

    def tool_schemas(self) -> list[dict]:
        """Devuelve schemas en formato OpenAI Chat Completions.

        Solo incluye tools con schema definido. Formato:
        [{"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}]
        """
        result: list[dict] = []
        for spec in self.list_specs():
            if spec.schema is None:
                continue
            result.append(
                {
                    "type": "function",
                    "function": {
                        "name": spec.name,
                        "description": spec.description,
                        "parameters": spec.schema,
                    },
                }
            )
        return result


def _is_coro(fn: Callable[..., Any]) -> bool:
    import inspect

    return inspect.iscoroutinefunction(fn)

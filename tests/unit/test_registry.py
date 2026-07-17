"""Tests para ToolRegistry."""

from __future__ import annotations

import pytest

from hermes.tools.registry import ToolRegistry


@pytest.mark.asyncio
async def test_register_and_execute_sync() -> None:
    reg = ToolRegistry()

    def add(a: int, b: int) -> int:
        return a + b

    reg.register("add", add)
    assert reg.has("add")
    result = await reg.execute("add", {"a": 2, "b": 3})
    assert result == 5


@pytest.mark.asyncio
async def test_register_and_execute_async() -> None:
    reg = ToolRegistry()

    async def greet(name: str) -> str:
        return f"hi {name}"

    reg.register("greet", greet)
    result = await reg.execute("greet", {"name": "ada"})
    assert result == "hi ada"


@pytest.mark.asyncio
async def test_unknown_tool_raises() -> None:
    reg = ToolRegistry()
    with pytest.raises(KeyError):
        await reg.execute("nope", {})

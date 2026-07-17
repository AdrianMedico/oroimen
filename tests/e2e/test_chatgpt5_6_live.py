"""Opt-in live smoke test for the GPT-5.6 Sol frontier route.

This module is excluded from deterministic offline runs. It exercises
the real router path only when frontier mode is explicitly enabled in
the process environment. The request is text-only: tool support remains
unit-tested request plumbing and is not part of the live Build Week
evidence.
"""

from __future__ import annotations

import asyncio

import pytest

from hermes.config import Settings
from hermes.llm.router import LLMRouter

pytestmark = [
    pytest.mark.network,
    pytest.mark.slow,
    pytest.mark.asyncio,
]

EXPECTED_MODEL = "gpt-5.6-sol"
SMOKE_PROMPT = "Reply with exactly: oroimen-smoke-ok"


async def test_gpt_5_6_sol_via_router() -> None:
    """A real frontier request returns content and a GPT-5.6 model."""
    settings = Settings(_env_file=None)
    if not settings.llm_text_frontier_enabled:
        pytest.skip(
            "Frontier mode is not enabled. Configure external access "
            "outside Git before running this opt-in smoke test."
        )

    assert settings.llm_text_frontier_model == EXPECTED_MODEL

    router = LLMRouter(settings)
    try:
        async with asyncio.timeout(60):
            response = await router.chat(
                [{"role": "user", "content": SMOKE_PROMPT}],
                temperature=0.0,
                chain_override=[EXPECTED_MODEL],
            )
    finally:
        await router.aclose()

    assert response.content.strip()
    assert response.model.startswith("gpt-5.6")
    assert response.latency_ms > 0

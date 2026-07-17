"""Integration tests for hermes.jobs.service — 5-phase pipeline.

Anti-regression checks (TDD §6):
- Phase 1 (search): returns URLs from web_search mock.
- Phase 2 (scrape): size guard truncates HTML BEFORE to_thread.
- Phase 3 (per-source synthesis): one failed source doesn't kill others.
- Phase 4 (final synthesis): citations in output format.
- All phases use real DB but mocked LLM / search / network.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hermes.jobs.exceptions import PhaseError
from hermes.jobs.prompts import sanitize_summary
from hermes.jobs.service import (
    _HTML_SIZE_GUARD_BYTES,
    DeepResearchService,
    html_to_text_selectolax,
)


@dataclass
class _FakeLLMResp:
    """Simulated LLMRouter.chat() response."""

    content: str
    tokens_in: int = 1000
    tokens_out: int = 500
    latency_ms: int = 200


@dataclass
class _FakeSearchResult:
    """Simulated hermes_search() result with results list."""

    results: list[dict] = field(default_factory=list)


class _FakeSettings:
    """Minimal settings stub for service tests."""

    deep_research_daily_budget_usd = 100.0  # high cap so budget doesn't trip
    deep_research_max_sources = 5
    deep_research_phase1_timeout_s = 5
    deep_research_phase2_timeout_s = 5
    deep_research_phase3_timeout_s = 10
    deep_research_phase4_timeout_s = 10
    deep_research_phase5_timeout_s = 5
    deep_research_per_source_max_tokens = 3000
    deep_research_output_max_tokens = 10000


@pytest.fixture
def service_with_mocks(db, tmp_path: Path):
    """Service with real DB + mocked LLM/search/notifier/scheduler."""
    settings = _FakeSettings()
    settings.deep_research_data_root = str(tmp_path / "jobs")

    notifier = MagicMock()
    notifier.send_research_complete = AsyncMock(return_value=True)
    notifier.send_research_failed = AsyncMock(return_value=True)

    llm = MagicMock()
    search = MagicMock()
    scheduler = MagicMock()
    scheduler.enqueue = AsyncMock()

    service = DeepResearchService(
        db=db,
        notifier=notifier,
        llm_router=llm,
        web_search=search,
        settings=settings,
        scheduler=scheduler,
    )
    return service, llm, search, notifier


@pytest.mark.asyncio
async def test_phase_search_ok(db, service_with_mocks) -> None:
    """Phase 1: search → list of URLs from web_search mock."""
    service, _llm, search, _notifier = service_with_mocks
    job_id = "searchtest1"
    await db.create_research_job(
        job_id=job_id,
        query="best hiking trails in Spain",
        notify_via_tg=0,
        user_id=0,
    )

    # Mock web_search to return 3 URLs (use AsyncMock so it's awaitable)
    search.return_value = _FakeSearchResult(
        results=[
            {"url": "https://example.com/a"},
            {"url": "https://example.com/b"},
            {"url": "https://example.com/c"},
        ]
    )
    search.side_effect = None  # reset any prior side_effect from fixture
    # Make search an AsyncMock so its call returns a coroutine
    search_async = AsyncMock(
        return_value=_FakeSearchResult(
            results=[
                {"url": "https://example.com/a"},
                {"url": "https://example.com/b"},
                {"url": "https://example.com/c"},
            ]
        )
    )
    service._search = search_async

    urls = await service._phase_search(job_id)
    assert urls == [
        "https://example.com/a",
        "https://example.com/b",
        "https://example.com/c",
    ]


@pytest.mark.slow
@pytest.mark.asyncio
async def test_phase_search_timeout_raises_phase_error(db, service_with_mocks) -> None:
    """Phase 1 timeout → PhaseError(taxonomy='timeout', retryable=True).

    Marcado @slow (5.01s en suite local): espera real al timeout
    configurado. Se ejecuta en nightly-tests.yml, no en CI diaria.
    """
    import asyncio

    service, _llm, _search, _notifier = service_with_mocks
    job_id = "searchtimeout1"
    await db.create_research_job(
        job_id=job_id,
        query="x",
        notify_via_tg=0,
        user_id=0,
    )

    async def _slow(*args, **kwargs):
        await asyncio.sleep(10)
        return _FakeSearchResult(results=[])

    # Override service._search with a slow AsyncMock
    service._search = AsyncMock(side_effect=_slow)

    with pytest.raises(PhaseError) as exc_info:
        await service._phase_search(job_id)
    assert exc_info.value.taxonomy == "timeout"
    assert exc_info.value.retryable is True


@pytest.mark.asyncio
async def test_phase_scrape_size_guard_truncates_before_thread(db, service_with_mocks) -> None:
    """Phase 2: HTML > 2MB → truncated to 2MB BEFORE to_thread.

    Verifies by patching html_to_text_selectolax to capture what it
    receives and asserting the input was ≤ _HTML_SIZE_GUARD_BYTES.
    """
    service, _llm, _search, _notifier = service_with_mocks
    job_id = "sizeguard1"
    await db.create_research_job(
        job_id=job_id,
        query="x",
        notify_via_tg=0,
        user_id=0,
    )

    # Mock httpx with a response whose .text is 3MB of HTML
    oversized = "x" * (3 * 1024 * 1024)

    class _FakeResp:
        status_code = 200
        text = oversized

        def raise_for_status(self):
            pass

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            # Accept whatever httpx.AsyncClient() passes (timeout=, etc.)
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url, **kwargs):
            return _FakeResp()

    received_html_sizes: list[int] = []
    real_html_to_text = html_to_text_selectolax

    def _capture(html: str) -> str:
        received_html_sizes.append(len(html))
        return real_html_to_text(html)

    # Patch at module level because _phase_scrape calls
    # `html_to_text_selectolax` (module-level reference) directly.
    # Also patch sys.modules['httpx'].AsyncClient (which is what `import httpx`
    # inside _phase_scrape resolves to).
    import httpx as _httpx_mod

    from hermes.jobs import service as service_module

    with (
        patch.object(_httpx_mod, "AsyncClient", _FakeClient),
        patch.object(service_module, "html_to_text_selectolax", _capture),
    ):
        results = await service._phase_scrape(job_id, ["https://example.com/big"])

    # Size guard truncated to ≤ 2MB
    assert len(received_html_sizes) == 1
    assert received_html_sizes[0] <= _HTML_SIZE_GUARD_BYTES
    # Output structure
    assert len(results) == 1
    assert results[0]["url"] == "https://example.com/big"


@pytest.mark.asyncio
async def test_phase_per_source_synthesis_with_one_failed(db, service_with_mocks) -> None:
    """Phase 3: one source fails (LLM error) but others succeed → partial summaries.

    TDD §6.4: phase 3 is mandatory, but per-source failures are
    isolated. Service continues with successful summaries.
    """
    service, _llm, _search, _notifier = service_with_mocks
    job_id = "persource1"
    await db.create_research_job(
        job_id=job_id,
        query="x",
        notify_via_tg=0,
        user_id=0,
    )

    sources = [
        {"url": "https://a.com", "success": True, "clean_text": "alpha content"},
        {"url": "https://b.com", "success": True, "clean_text": "beta content"},
        {"url": "https://c.com", "success": False, "error": "too_short"},
    ]

    # Mock LLM: first 2 calls succeed, third never happens (c filtered out)
    llm_mock = MagicMock()
    llm_mock.chat = AsyncMock(
        side_effect=[
            _FakeLLMResp(content="Summary of A: ...", tokens_in=100, tokens_out=50),
            _FakeLLMResp(content="Summary of B: ...", tokens_in=100, tokens_out=50),
        ]
    )
    service._llm = llm_mock

    summaries = await service._phase_per_source_synthesis(job_id, sources)

    # 2 successful summaries (c was filtered because success=False)
    assert len(summaries) == 2
    assert "Summary of A" in summaries[0]
    assert "Summary of B" in summaries[1]
    # Only 2 LLM calls (c didn't trigger)
    assert llm_mock.chat.call_count == 2


@pytest.mark.asyncio
async def test_phase_per_source_synthesis_no_valid_sources_raises(db, service_with_mocks) -> None:
    """Phase 3 with zero valid sources → PhaseError(llm_5xx, retryable=False)."""
    service, _llm, _search, _notifier = service_with_mocks
    job_id = "persource2"
    await db.create_research_job(
        job_id=job_id,
        query="x",
        notify_via_tg=0,
        user_id=0,
    )

    sources = [
        {"url": "https://a.com", "success": False, "error": "too_short"},
        {"url": "https://b.com", "success": False, "error": "timeout"},
    ]

    with pytest.raises(PhaseError) as exc_info:
        await service._phase_per_source_synthesis(job_id, sources)
    assert exc_info.value.taxonomy == "llm_5xx"
    assert exc_info.value.retryable is False


@pytest.mark.asyncio
async def test_phase_final_synthesis_with_citations(db, service_with_mocks) -> None:
    """Phase 4: final synthesis receives summaries, produces report with [1], [2] markers.

    Sanitize step applies: input summaries that have thinking blocks
    should be cleaned before passing to LLM (defense in depth).
    """
    service, llm, _search, _notifier = service_with_mocks
    job_id = "finalsynth1"
    await db.create_research_job(
        job_id=job_id,
        query="x",
        notify_via_tg=0,
        user_id=0,
    )

    # Mock LLM to return a citation-style report
    citation_report = """## Summary
Based on the sources [1] and [2], the answer is X.

## Key Findings
- Finding one [1].
- Finding two [2].

## Sources
1. Source A
2. Source B
"""
    llm.chat = AsyncMock(
        return_value=_FakeLLMResp(
            content=citation_report,
            tokens_in=2000,
            tokens_out=1500,
        )
    )
    service._llm = llm  # ensure service uses the new mock

    summaries = ["Summary A", "Summary B"]
    report = await service._phase_final_synthesis(job_id, summaries)

    # Citations preserved
    assert "[1]" in report
    assert "[2]" in report
    # Sanitized (no thinking blocks leaked)
    assert "<think>" not in report


@pytest.mark.asyncio
async def test_sanitize_summary_strips_thinking_blocks() -> None:
    """sanitize_summary removes <think>...</think> and ChatML <|thinking|>...| blocks."""
    raw_with_think = """<think>
I should consider the user's question carefully...
The answer is 42.
</think>

## Summary
The answer is 42."""

    cleaned = sanitize_summary(raw_with_think)
    assert "<think>" not in cleaned
    assert "I should consider" not in cleaned
    assert "## Summary" in cleaned
    assert "The answer is 42" in cleaned


@pytest.mark.asyncio
async def test_sanitize_summary_handles_json_content_extraction() -> None:
    """If LLM returns JSON with `content` field, sanitize extracts just content."""
    raw_json = json.dumps(
        {
            "thinking": "internal monologue here",
            "content": "## Summary\nThis is the actual response.",
        }
    )
    cleaned = sanitize_summary(raw_json)
    assert "internal monologue" not in cleaned
    assert "## Summary" in cleaned
    assert "This is the actual response" in cleaned

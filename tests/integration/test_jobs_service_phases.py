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
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hermes.jobs.exceptions import PhaseError
from hermes.jobs.models import PhaseName
from hermes.jobs.prompts import sanitize_summary
from hermes.jobs.service import (
    _HTML_SIZE_GUARD_BYTES,
    DeepResearchService,
    _map_search_diagnostic_to_taxonomy,
    _phase_error_from_search_error,
    html_to_text_selectolax,
)
from hermes.services.search.errors import (
    SearchDiagnosticCategory,
    SearchErrorCode,
    _build_structured_error,
)
from hermes.services.search.protocol import SearchResult


@dataclass
class _FakeLLMResp:
    """Simulated LLMRouter.chat() response."""

    content: str
    tokens_in: int = 1000
    tokens_out: int = 500
    latency_ms: int = 200


@dataclass
class _FakeSearchResult:
    """Simulated hermes_search() result with results list.

    PRE2-A1: also carries an optional ``error`` field (SearchError
    or None) so bridge tests can drive the structured failure
    pathway of ``_phase_search``.
    """

    results: list[dict] = field(default_factory=list)
    error: Any | None = None


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


class _FakeFetcher:
    """Controlled fake safe fetcher for Phase 2 tests.

    Records every ``fetch(url)`` invocation and returns scripted bytes.
    The service must use ONLY the returned bounded bytes for local
    decode; there is no other transport.
    """

    def __init__(self, bodies: dict[str, bytes] | None = None) -> None:
        self.calls: list[str] = []
        self._bodies = bodies or {}

    async def fetch(self, url: str) -> _FakeFetchResult:
        self.calls.append(url)
        body = self._bodies.get(url, b"<html>default</html>")
        return _FakeFetchResult(body=body, media_type="text/html", status=200)


@dataclass
class _FakeFetchResult:
    body: bytes
    media_type: str = "text/html"
    status: int = 200
    redirect_count: int = 0


@pytest.fixture
def service_with_mocks(db, tmp_path: Path):
    """Service with real DB + mocked LLM/search/notifier/scheduler/fetcher."""
    settings = _FakeSettings()
    settings.deep_research_data_root = str(tmp_path / "jobs")

    notifier = MagicMock()
    notifier.send_research_complete = AsyncMock(return_value=True)
    notifier.send_research_failed = AsyncMock(return_value=True)

    llm = MagicMock()
    search = MagicMock()
    scheduler = MagicMock()
    scheduler.enqueue = AsyncMock()
    # Slice 1C1b: controlled fake fetcher; tests can override
    # service._fetcher for specific Phase 2 scenarios.
    fetcher = _FakeFetcher()

    service = DeepResearchService(
        db=db,
        notifier=notifier,
        llm_router=llm,
        web_search=search,
        fetcher=fetcher,
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

    Slice 1C1b: replaces the former httpx.AsyncClient seam with a
    controlled safe fetcher. The service must call
    ``await self._fetcher.fetch(url)`` and use only the bounded bytes
    for local decode.

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

    # Body must be local bytes returned by the fetcher (3MB).
    oversized = b"x" * (3 * 1024 * 1024)
    fetcher = _FakeFetcher(
        bodies={"https://example.com/big": oversized},
    )
    service._fetcher = fetcher

    received_html_sizes: list[int] = []
    real_html_to_text = html_to_text_selectolax

    def _capture(html: str) -> str:
        received_html_sizes.append(len(html))
        return real_html_to_text(html)

    from hermes.jobs import service as service_module

    with patch.object(service_module, "html_to_text_selectolax", _capture):
        results = await service._phase_scrape(job_id, ["https://example.com/big"])

    # Safe fetcher was called for the URL.
    assert fetcher.calls == ["https://example.com/big"]
    # Size guard truncated to ≤ 2MB.
    assert len(received_html_sizes) == 1
    assert received_html_sizes[0] <= _HTML_SIZE_GUARD_BYTES
    # Output structure.
    assert len(results) == 1
    assert results[0]["url"] == "https://example.com/big"
    assert results[0]["success"] is True


@pytest.mark.asyncio
async def test_phase_scrape_uses_fake_fetcher(db, service_with_mocks) -> None:
    """Phase 2 calls ``self._fetcher.fetch(url)`` and consumes bounded bytes only.

    Slice 1C1b: proves the safe-fetcher boundary is the only HTTP
    seam. The service must NOT import httpx or instantiate AsyncClient.
    """
    service, _llm, _search, _notifier = service_with_mocks
    job_id = "usesfetch1"
    await db.create_research_job(
        job_id=job_id,
        query="x",
        notify_via_tg=0,
        user_id=0,
    )

    url = "https://example.invalid/page"
    body_html = (
        b"<html><body>" + (b"hello world this is sufficient content. " * 100) + b"</body></html>"
    )
    fetcher = _FakeFetcher(
        bodies={url: body_html},
    )
    service._fetcher = fetcher

    results = await service._phase_scrape(job_id, [url])

    # Fetcher was called exactly once for the URL.
    assert fetcher.calls == [url]
    assert len(results) == 1
    assert results[0]["success"] is True
    assert "hello world" in results[0]["clean_text"]


@pytest.mark.asyncio
async def test_phase_scrape_fetcher_exception_maps_to_redacted_error(
    db, service_with_mocks
) -> None:
    """Phase 2 maps a fetcher exception to a stable redacted error.

    Slice 1C1b: the redacted error must NOT contain the URL,
    hostname, exception text, exception type, or any input
    identifier. The marker is a stable ``safe_fetch_failed`` string.
    """

    class _ExplodingFetcher:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def fetch(self, url: str) -> Any:
            self.calls.append(url)
            # Raise with a value the test can search for. The service
            # MUST NOT include this text in the returned marker.
            raise RuntimeError(
                "SECRET_TOKEN_abc123 example.invalid TypeError: secret internal detail"
            )

    service, _llm, _search, _notifier = service_with_mocks
    job_id = "explode1"
    await db.create_research_job(
        job_id=job_id,
        query="x",
        notify_via_tg=0,
        user_id=0,
    )

    exploding = _ExplodingFetcher()
    service._fetcher = exploding

    results = await service._phase_scrape(job_id, ["https://example.invalid/x"])

    # Fetcher was called.
    assert exploding.calls == ["https://example.invalid/x"]
    assert len(results) == 1
    # The marker must be a stable redacted error.
    assert results[0]["success"] is False
    assert results[0]["error"] == "safe_fetch_failed"
    # The marker must NOT contain the URL, hostname, exception text,
    # exception type, or input identifiers.
    marker = results[0]["error"]
    assert "SECRET_TOKEN_abc123" not in marker
    assert "example.invalid" not in marker
    assert "RuntimeError" not in marker
    assert "TypeError" not in marker
    assert "secret" not in marker


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

    # Oroimen Slice 1C1a: set distinct non-default configured value so
    # the assertion below proves the override is forwarded (not just
    # that the global llm_max_tokens default leaks through).
    service._settings.deep_research_per_source_max_tokens = 4321

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
    # Oroimen Slice 1C1a: phase 3 forwards the configured per-source
    # output token limit, distinct from the global llm_max_tokens default.
    assert llm_mock.chat.await_count == 2
    for call in llm_mock.chat.await_args_list:
        assert call.kwargs.get("max_tokens") == 4321


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

    # Oroimen Slice 1C1a: set distinct non-default configured value so
    # the assertion below proves phase 4 forwards its own setting
    # (not the per-source value or any global default).
    service._settings.deep_research_output_max_tokens = 8765

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
    # Oroimen Slice 1C1a: phase 4 forwards the configured final-output
    # token limit, distinct from the per-source setting.
    llm.chat.assert_awaited_once()
    assert llm.chat.await_args.kwargs.get("max_tokens") == 8765


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


# =====================================================================
# PRE2-A1: Deep Research bridge tests
# =====================================================================
#
# These tests prove that:
#   - SearchResult.error is consumed BEFORE URLs in _phase_search.
#   - The structured SearchError fields reach PhaseError intact.
#   - The broad persisted taxonomy is preserved.
#   - Valid empty result (error=None, results=[]) returns [] without
#     raising a search error.
#   - The _run_phase_with_retry retry decision is taken from
#     PhaseError.retryable, not from taxonomy membership.


# --- bridge: SearchError -> PhaseError structured field carry-over ---


def test_pre2a1_phase_error_carries_search_structured_fields() -> None:
    """PRE2-A1: SearchError structured fields reach PhaseError intact."""
    err = _build_structured_error(
        code=SearchErrorCode.SERVER_ERROR,
        message="Search backend tavily returned HTTP 503.",
        backend="tavily",
        retryable=True,
        breaker_relevant=True,
        http_status=503,
        diagnostic_category=SearchDiagnosticCategory.SERVER_ERROR,
    )
    pe = _phase_error_from_search_error(err)
    # Persisted taxonomy mapped from diagnostic_category.
    assert pe.taxonomy == "search_5xx"
    # Retry authority preserved.
    assert pe.retryable is True
    # PRE2-A1: structured bridge fields.
    assert pe.search_error_code == "SERVER_ERROR"
    assert pe.search_backend == "tavily"
    assert pe.search_breaker_relevant is True
    assert pe.search_http_status == 503
    assert pe.search_diagnostic_category == "server_error"


def test_pre2a1_phase_error_429_maps_to_4xx_with_retryable_true() -> None:
    """PRE2-A1: 429 maps to search_4xx but is retryable (contradictory-case)."""
    err = _build_structured_error(
        code=SearchErrorCode.RATE_LIMITED,
        message="Search backend tavily returned HTTP 429.",
        backend="tavily",
        retryable=True,
        breaker_relevant=False,
        http_status=429,
        diagnostic_category=SearchDiagnosticCategory.RATE_LIMIT,
    )
    pe = _phase_error_from_search_error(err)
    assert pe.taxonomy == "search_4xx"
    assert pe.retryable is True
    # The PhaseError still carries the structured fields for downstream
    # consumers (logs, dashboards) that want the precise cause.
    assert pe.search_error_code == "RATE_LIMITED"
    assert pe.search_http_status == 429
    assert pe.search_diagnostic_category == "rate_limit"


def test_pre2a1_phase_error_5xx_with_retryable_false_contradictory() -> None:
    """PRE2-A1: 5xx with retryable=False (contradictory-case)."""
    err = _build_structured_error(
        code=SearchErrorCode.SERVER_ERROR,
        message="Search backend tavily returned HTTP 503.",
        backend="tavily",
        retryable=False,  # contradictory: 5xx taxonomy but not retryable
        breaker_relevant=True,
        http_status=503,
        diagnostic_category=SearchDiagnosticCategory.SERVER_ERROR,
    )
    pe = _phase_error_from_search_error(err)
    # Persisted taxonomy still says 5xx (broad).
    assert pe.taxonomy == "search_5xx"
    # But the explicit retry authority says NO.
    assert pe.retryable is False


def test_pre2a1_phase_error_4xx_with_retryable_true_contradictory() -> None:
    """PRE2-A1: 4xx with retryable=True (the 429 case above)."""
    err = _build_structured_error(
        code=SearchErrorCode.AUTH_ERROR,
        message="Search backend tavily returned HTTP 401.",
        backend="tavily",
        retryable=True,  # contradictory: 4xx taxonomy but retryable
        breaker_relevant=False,
        http_status=401,
        diagnostic_category=SearchDiagnosticCategory.AUTH,
    )
    pe = _phase_error_from_search_error(err)
    assert pe.taxonomy == "search_4xx"
    assert pe.retryable is True


def test_pre2a1_mapping_diagnostic_to_taxonomy_is_explicit() -> None:
    """PRE2-A1: every diagnostic category maps deterministically.

    The persisted taxonomy is the closed set
    ``search_4xx / search_5xx / timeout / network``. No new
    persisted bucket is added — invalid_response folds into the
    broad ``search_5xx`` group, with the precise cause available
    only in the in-memory ``PhaseError.search_diagnostic_category``
    field.
    """
    assert _map_search_diagnostic_to_taxonomy("timeout") == "timeout"
    assert _map_search_diagnostic_to_taxonomy("network") == "network"
    assert _map_search_diagnostic_to_taxonomy("auth") == "search_4xx"
    assert _map_search_diagnostic_to_taxonomy("client_error") == "search_4xx"
    assert _map_search_diagnostic_to_taxonomy("rate_limit") == "search_4xx"
    assert _map_search_diagnostic_to_taxonomy("local_validation") == "search_4xx"
    assert _map_search_diagnostic_to_taxonomy("budget") == "search_4xx"
    assert _map_search_diagnostic_to_taxonomy("server_error") == "search_5xx"
    assert _map_search_diagnostic_to_taxonomy("circuit") == "search_5xx"
    assert (
        _map_search_diagnostic_to_taxonomy("all_backends_failed") == "search_5xx"
    )
    # invalid_response folds into the broad search_5xx bucket.
    assert (
        _map_search_diagnostic_to_taxonomy("invalid_response") == "search_5xx"
    )


# --- _phase_search: error before URLs ---


def _make_search_result_with_error(
    code: SearchErrorCode,
    backend: str,
    retryable: bool,
    breaker_relevant: bool,
    http_status: int | None,
    category: SearchDiagnosticCategory,
) -> Any:
    """Build a real ``SearchResult`` with a structured error.

    PRE2-A1 contract: ``_phase_search`` only treats a real
    ``SearchResult`` instance as carrying a structured failure.
    A ``_FakeSearchResult`` (duck-typed, with just ``results`` and
    ``error``) is not a real ``SearchResult`` and the phase would
    fall through to the legacy ``hasattr(result, "results")`` path.
    """
    err = _build_structured_error(
        code=code,
        message=f"Search backend {backend} returned HTTP {http_status or 'N/A'}.",
        backend=backend,
        retryable=retryable,
        breaker_relevant=breaker_relevant,
        http_status=http_status,
        diagnostic_category=category,
    )
    return SearchResult(
        results=[{"url": "https://example.com/should-not-be-read"}],
        backend_used=backend,
        query="q",
        content_mode="snippet",
        original_content_mode="snippet",
        format_fallback=False,
        size_guard_chars=50000,
        truncated=False,
        error=err,
    )


@pytest.mark.asyncio
async def test_pre2a1_phase_search_consumes_error_before_urls(db, service_with_mocks) -> None:
    """PRE2-A1: _phase_search raises PhaseError from SearchResult.error."""
    service, _llm, _search, _notifier = service_with_mocks
    job_id = "pre2a1-bridge-1"
    await db.create_research_job(
        job_id=job_id,
        query="q",
        notify_via_tg=0,
        user_id=0,
    )

    # Build a SearchResult with structured error + a URL that must NOT
    # be read (the error is consumed first).
    result = _make_search_result_with_error(
        code=SearchErrorCode.SERVER_ERROR,
        backend="tavily",
        retryable=True,
        breaker_relevant=True,
        http_status=503,
        category=SearchDiagnosticCategory.SERVER_ERROR,
    )
    service._search = AsyncMock(return_value=result)

    with pytest.raises(PhaseError) as exc_info:
        await service._phase_search(job_id)
    pe = exc_info.value
    # Persisted taxonomy mapped from diagnostic_category.
    assert pe.taxonomy == "search_5xx"
    assert pe.retryable is True
    # PRE2-A1: structured fields reach PhaseError.
    assert pe.search_error_code == "SERVER_ERROR"
    assert pe.search_backend == "tavily"
    assert pe.search_breaker_relevant is True
    assert pe.search_http_status == 503
    assert pe.search_diagnostic_category == "server_error"


@pytest.mark.asyncio
async def test_pre2a1_phase_search_400_401_403_non_retryable(
    db, service_with_mocks
) -> None:
    """PRE2-A1: 400/401/403 surface as non-retryable PhaseError."""
    service, _llm, _search, _notifier = service_with_mocks
    for status, code, cat in (
        (400, SearchErrorCode.CLIENT_ERROR, SearchDiagnosticCategory.CLIENT_ERROR),
        (401, SearchErrorCode.AUTH_ERROR, SearchDiagnosticCategory.AUTH),
        (403, SearchErrorCode.AUTH_ERROR, SearchDiagnosticCategory.AUTH),
    ):
        await db.create_research_job(
            job_id=f"pre2a1-4xx-{status}",
            query="q",
            notify_via_tg=0,
            user_id=0,
        )
        result = _make_search_result_with_error(
            code=code,
            backend="tavily",
            retryable=False,
            breaker_relevant=False,
            http_status=status,
            category=cat,
        )
        service._search = AsyncMock(return_value=result)
        with pytest.raises(PhaseError) as exc_info:
            await service._phase_search(f"pre2a1-4xx-{status}")
        pe = exc_info.value
        assert pe.retryable is False
        assert pe.search_http_status == status
        # 4xx taxonomy group.
        assert pe.taxonomy == "search_4xx"


@pytest.mark.asyncio
async def test_pre2a1_phase_search_429_retryable(db, service_with_mocks) -> None:
    """PRE2-A1: 429 surfaces as retryable PhaseError (search_4xx, retryable=True)."""
    service, _llm, _search, _notifier = service_with_mocks
    await db.create_research_job(
        job_id="pre2a1-429",
        query="q",
        notify_via_tg=0,
        user_id=0,
    )
    result = _make_search_result_with_error(
        code=SearchErrorCode.RATE_LIMITED,
        backend="tavily",
        retryable=True,
        breaker_relevant=False,
        http_status=429,
        category=SearchDiagnosticCategory.RATE_LIMIT,
    )
    service._search = AsyncMock(return_value=result)
    with pytest.raises(PhaseError) as exc_info:
        await service._phase_search("pre2a1-429")
    pe = exc_info.value
    assert pe.taxonomy == "search_4xx"
    assert pe.retryable is True
    assert pe.search_http_status == 429


@pytest.mark.asyncio
async def test_pre2a1_phase_search_5xx_timeout_network_retryable(
    db, service_with_mocks
) -> None:
    """PRE2-A1: 5xx, timeout, network surface as retryable PhaseError."""
    service, _llm, _search, _notifier = service_with_mocks
    cases = [
        (
            "5xx",
            SearchErrorCode.SERVER_ERROR,
            SearchDiagnosticCategory.SERVER_ERROR,
            503,
            True,
        ),
        (
            "timeout",
            SearchErrorCode.TIMEOUT,
            SearchDiagnosticCategory.TIMEOUT,
            None,
            True,
        ),
        (
            "network",
            SearchErrorCode.NETWORK_ERROR,
            SearchDiagnosticCategory.NETWORK,
            None,
            True,
        ),
    ]
    for tag, code, cat, status, retryable in cases:
        await db.create_research_job(
            job_id=f"pre2a1-{tag}",
            query="q",
            notify_via_tg=0,
            user_id=0,
        )
        result = _make_search_result_with_error(
            code=code,
            backend="tavily",
            retryable=retryable,
            breaker_relevant=True,
            http_status=status,
            category=cat,
        )
        service._search = AsyncMock(return_value=result)
        with pytest.raises(PhaseError) as exc_info:
            await service._phase_search(f"pre2a1-{tag}")
        pe = exc_info.value
        assert pe.retryable is True
        assert pe.search_breaker_relevant is True
        assert pe.search_error_code == code.value


@pytest.mark.asyncio
async def test_pre2a1_phase_search_valid_empty_returns_empty_list(
    db, service_with_mocks
) -> None:
    """PRE2-A1: valid empty result returns [] without raising."""
    service, _llm, _search, _notifier = service_with_mocks
    job_id = "pre2a1-empty"
    await db.create_research_job(
        job_id=job_id,
        query="q",
        notify_via_tg=0,
        user_id=0,
    )
    # Valid empty SearchResult: error=None, results=[]
    result = _FakeSearchResult(results=[], error=None)
    service._search = AsyncMock(return_value=result)
    urls = await service._phase_search(job_id)
    assert urls == []


# --- PRE2-A1: retry authority ---


@pytest.mark.asyncio
async def test_pre2a1_run_phase_with_retry_uses_retryable_not_taxonomy(
    db, service_with_mocks
) -> None:
    """PRE2-A1: PhaseError.retryable is authoritative.

    The contradictory cases:
      - PhaseError("search_5xx", retryable=False) -> 1 attempt.
      - PhaseError("search_4xx", retryable=True)  -> 3 attempts.

    Patches _RETRY_BACKOFF_SCHEDULE to zero so the test does not
    wait real backoff durations.
    """
    import hermes.jobs.service as service_mod

    original_schedule = service_mod._RETRY_BACKOFF_SCHEDULE
    service_mod._RETRY_BACKOFF_SCHEDULE = (0, 0, 0)
    try:
        service, _llm, _search, _notifier = service_with_mocks

        # Case 1: search_5xx with retryable=False → exactly 1 attempt.
        attempts_5xx_not_retryable = 0

        async def raise_5xx_not_retryable() -> None:
            nonlocal attempts_5xx_not_retryable
            attempts_5xx_not_retryable += 1
            raise PhaseError("search_5xx", "forced", retryable=False)

        with pytest.raises(PhaseError) as exc:
            await service._run_phase_with_retry(
                "job-5xx-not-retryable", PhaseName.SEARCH, raise_5xx_not_retryable
            )
        assert exc.value.retryable is False
        assert attempts_5xx_not_retryable == 1

        # Case 2: search_4xx with retryable=True → up to 3 attempts.
        attempts_4xx_retryable = 0

        async def raise_4xx_retryable() -> None:
            nonlocal attempts_4xx_retryable
            attempts_4xx_retryable += 1
            raise PhaseError("search_4xx", "forced", retryable=True)

        with pytest.raises(PhaseError) as exc:
            await service._run_phase_with_retry(
                "job-4xx-retryable", PhaseName.SEARCH, raise_4xx_retryable
            )
        assert exc.value.retryable is True
        assert attempts_4xx_retryable == 3

        # Case 3 (control): search_5xx with retryable=True → 3 attempts
        # (matches the old RETRYABLE_ERRORS membership behavior).
        attempts_5xx_retryable = 0

        async def raise_5xx_retryable() -> None:
            nonlocal attempts_5xx_retryable
            attempts_5xx_retryable += 1
            raise PhaseError("search_5xx", "forced", retryable=True)

        with pytest.raises(PhaseError):
            await service._run_phase_with_retry(
                "job-5xx-retryable", PhaseName.SEARCH, raise_5xx_retryable
            )
        assert attempts_5xx_retryable == 3
    finally:
        service_mod._RETRY_BACKOFF_SCHEDULE = original_schedule


# --- PRE2-A1: never parses str(exc) in the phase ---


@pytest.mark.asyncio
async def test_pre2a1_phase_search_no_exception_text_parsing(
    db, service_with_mocks
) -> None:
    """PRE2-A1: a 401/403 from the search backend reaches the LLM as
    a structured search failure even when the exception text is
    misleading (e.g. contains 'api key' or 'unauthorized' or '403').

    The phase must not parse str(exc); it must use the structured
    fields from the search result. The router's classifier already
    handled the typed exception.
    """
    service, _llm, _search, _notifier = service_with_mocks
    job_id = "pre2a1-no-str-parsing"
    await db.create_research_job(
        job_id=job_id,
        query="q",
        notify_via_tg=0,
        user_id=0,
    )
    # The mock simulates a backend that returns a SearchResult with
    # error.code=AUTH_ERROR (e.g. Tavily returned 403 with body
    # containing the misleading phrase "api key" — but the router
    # already classified it as AUTH_ERROR structurally).
    err = _build_structured_error(
        code=SearchErrorCode.AUTH_ERROR,
        # This message contains '403' and 'api key' but the phase
        # bridge does NOT inspect it; it uses code/backend/http_status.
        message="Backend 403: api key invalid",
        backend="tavily",
        retryable=False,
        breaker_relevant=False,
        http_status=403,
        diagnostic_category=SearchDiagnosticCategory.AUTH,
    )
    result = SearchResult(
        results=[],
        backend_used="tavily",
        query="q",
        content_mode="snippet",
        original_content_mode="snippet",
        format_fallback=False,
        size_guard_chars=50000,
        truncated=False,
        error=err,
    )
    service._search = AsyncMock(return_value=result)
    with pytest.raises(PhaseError) as exc_info:
        await service._phase_search(job_id)
    pe = exc_info.value
    assert pe.taxonomy == "search_4xx"
    assert pe.retryable is False
    assert pe.search_error_code == "AUTH_ERROR"
    assert pe.search_http_status == 403


# --- PRE2-A1: invalid-response persistence compatibility ---


@pytest.mark.asyncio
async def test_pre2a1_invalid_response_search_failure_persists_search_5xx(
    db, service_with_mocks
) -> None:
    """PRE2-A1: a structured invalid-response search failure reaches
    a terminal ``failed`` state in the DB with the broad
    ``error_taxonomy='search_5xx'``.

    Exercises the full job-failure transition (not just
    ``_phase_search`` in isolation) so persistence compatibility is
    proven end-to-end:

      1. exactly one search attempt is made;
      2. no retry happens (retryable=False at the phase boundary);
      3. the row reaches terminal ``failed``;
      4. ``error_taxonomy='search_5xx'`` is persisted (the broad
         bucket — no new persisted column, no schema change, no
         DTO change);
      5. ``retryable=False`` is preserved at the phase boundary;
      6. the row is NOT left in ``running``;
      7. no provider or network call is made beyond the in-process
         search mock.

    The point of going through the full ``_run_research`` loop
    (rather than calling ``_phase_search`` directly) is to prove
    the broad ``search_5xx`` taxonomy is acceptable to the existing
    ``error_taxonomy`` CHECK constraint and that the conditional
    ``running -> failed`` transition writes both
    ``error_taxonomy`` and ``error_message`` to the same row.
    """
    service, _llm, _search, _notifier = service_with_mocks
    job_id = "pre2a1-invalid-response-persistence"
    await db.create_research_job(
        job_id=job_id,
        query="q",
        notify_via_tg=0,
        user_id=0,
    )

    # Build a real ``SearchResult`` with a structured
    # ``INVALID_RESPONSE`` ``SearchError``. ``retryable=False`` so
    # the retry loop MUST NOT re-attempt the search.
    invalid_response_error = _build_structured_error(
        code=SearchErrorCode.INVALID_RESPONSE,
        message="Search backend tavily returned invalid response.",
        backend="tavily",
        retryable=False,
        breaker_relevant=False,
        http_status=None,
        diagnostic_category=SearchDiagnosticCategory.INVALID_RESPONSE,
    )
    real_search_result = SearchResult(
        results=[],
        backend_used="tavily",
        query="q",
        content_mode="snippet",
        original_content_mode="snippet",
        format_fallback=False,
        size_guard_chars=50000,
        truncated=False,
        error=invalid_response_error,
    )

    # Track how many times the search callable is invoked.
    # PRE2-A1: ``retryable=False`` MUST result in a single search
    # attempt — the retry loop sees ``exc.retryable is False`` and
    # re-raises without sleeping or re-calling.
    invocation_count = 0

    async def _counting_search(**kwargs: Any) -> SearchResult:
        nonlocal invocation_count
        invocation_count += 1
        return real_search_result

    service._search = _counting_search

    # Run the full research loop synchronously. The job starts in
    # 'pending'; the startup CAS transitions it to 'running'; phase 1
    # raises ``PhaseError(taxonomy='search_5xx', retryable=False)``;
    # the inner ``except PhaseError`` branch transitions the row to
    # 'failed' with the broad taxonomy.
    await service._run_research(job_id)

    # 1. The search was called exactly once (no retry).
    assert invocation_count == 1
    # 2. The row is in terminal 'failed' state, NOT 'running'.
    row = await db.get_research_job(job_id)
    assert row["status"] == "failed"
    assert row["status"] != "running"
    # 3. The persisted ``error_taxonomy`` is the broad ``search_5xx``.
    #    No new persisted bucket (``search_invalid``) is introduced.
    assert row["error_taxonomy"] == "search_5xx"
    # 4. The ``error_message`` is the safe static string from the
    #    structured ``SearchError`` (NOT ``str(exc)`` and NOT a raw
    #    body / header).
    assert row["error_message"] == (
        "Search backend tavily returned invalid response."
    )
    # 5. ``completed_at`` is set (terminal transition recorded).
    assert row["completed_at"] is not None
    # 6. The fetcher (phase 2) was never called: the invalid-response
    #    failure halts the pipeline at phase 1 so no URL is produced
    #    for it to fetch. (The LLM, scheduler, and notifier are
    #    all in-process mocks; no real provider / network call is
    #    made by the test itself.)
    assert service._fetcher.calls == []

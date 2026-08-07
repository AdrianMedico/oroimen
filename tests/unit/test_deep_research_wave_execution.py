"""Deterministic tests for one SearchWave execution."""

from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from hermes.deep_research.planning import (
    MAX_QUERIES_PER_WAVE,
    MAX_QUERY_CHARS,
    CapabilitySnapshot,
    PlannedSearchQuery,
    PlanningLimits,
    SearchPlan,
    compute_query_id,
    compute_research_brief_sha256,
    normalize_query_text,
)
from hermes.deep_research.wave_execution import (
    SearchWaveExecutor,
    WaveExecutionOutcome,
)
from hermes.services.search.errors import (
    SearchDiagnosticCategory,
    SearchError,
    SearchErrorCode,
)
from hermes.services.search.protocol import SearchResult


def _query(ordinal: int, text: str) -> PlannedSearchQuery:
    brief_sha = compute_research_brief_sha256("brief" * 1000)
    return PlannedSearchQuery(
        query_id=compute_query_id(
            schema_version=1,
            wave_index=0,
            ordinal=ordinal,
            normalized_text=normalize_query_text(text),
            research_brief_sha256=brief_sha,
        ),
        text=text,
        purpose="find evidence",
        dimension_ids=("coverage",),
        ordinal=ordinal,
    )


def _plan(queries: tuple[PlannedSearchQuery, ...]) -> SearchPlan:
    limits = PlanningLimits(
        max_queries_per_wave=MAX_QUERIES_PER_WAVE,
        max_query_chars=MAX_QUERY_CHARS,
    )
    snapshot = CapabilitySnapshot(
        planner_kind="c1a-deterministic-stub",
        planner_version="0.1.0",
        max_queries_per_wave=MAX_QUERIES_PER_WAVE,
        max_query_chars=MAX_QUERY_CHARS,
    )
    return SearchPlan(
        schema_version=1,
        planner_kind="c1a-deterministic-stub",
        planner_version="0.1.0",
        research_brief_sha256=compute_research_brief_sha256("brief" * 1000),
        wave_index=0,
        queries=queries,
        planning_limits=limits,
        capability_snapshot=snapshot,
        created_at="2026-08-07T00:00:00Z",
    )


def _result(urls: list[str], backend: str = "fake") -> SearchResult:
    return SearchResult(
        results=[{"url": url} for url in urls],
        backend_used=backend,
        query="query",
        content_mode="snippet",
        original_content_mode="snippet",
        format_fallback=False,
        size_guard_chars=200000,
        truncated=False,
    )


def _error_result() -> SearchResult:
    return SearchResult(
        results=[],
        backend_used="fake",
        query="query",
        content_mode="snippet",
        original_content_mode="snippet",
        format_fallback=False,
        size_guard_chars=200000,
        truncated=False,
        error=SearchError(
            code=SearchErrorCode.RATE_LIMITED,
            message="must not be copied",
            backend="fake",
            retryable=True,
            suggestion="retry",
            breaker_relevant=False,
            http_status=429,
            diagnostic_category=SearchDiagnosticCategory.RATE_LIMIT,
        ),
    )


class _FakeSearch:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, str, str, int]] = []

    async def __call__(
        self,
        *,
        query: str,
        intent: str,
        content: str,
        num_results: int,
    ) -> Any:
        self.calls.append((query, intent, content, num_results))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [1, 2, 4])
async def test_queries_execute_once_in_ordinal_order(count: int) -> None:
    queries = tuple(_query(ordinal, f"q{ordinal}") for ordinal in reversed(range(count)))
    fake = _FakeSearch([_result([f"https://example.test/{i}"]) for i in range(count)])
    result = await SearchWaveExecutor(fake).execute(_plan(queries))

    assert [call[0] for call in fake.calls] == [f"q{i}" for i in range(count)]
    assert [obs.query_id for obs in result.observations] == [
        _query(i, f"q{i}").query_id for i in range(count)
    ]


@pytest.mark.asyncio
async def test_global_dedupe_cap_and_first_provenance() -> None:
    plan = _plan((_query(0, "first"), _query(1, "second")))
    fake = _FakeSearch(
        [
            _result(
                [
                    " https://example.test/a/// ",
                    "ftp://unsafe.test/a",
                    "javascript:alert(1)",
                    "https://example.test/b/",
                ]
            ),
            _result(["https://example.test/a", "https://example.test/c/"]),
        ]
    )
    result = await SearchWaveExecutor(fake, max_unique_sources=2).execute(plan)

    assert result.unique_source_refs == (
        "https://example.test/a",
        "https://example.test/b",
    )
    assert dict(result.source_query_ids) == {
        "https://example.test/a": plan.queries[0].query_id,
        "https://example.test/b": plan.queries[0].query_id,
    }
    assert result.unique_source_cap == 2


@pytest.mark.asyncio
async def test_all_deduplicated_results_are_successful_not_empty() -> None:
    fake = _FakeSearch(
        [_result(["https://example.test/a"]), _result(["https://example.test/a"])]
    )
    result = await SearchWaveExecutor(fake).execute(
        _plan((_query(0, "first"), _query(1, "duplicate")))
    )
    assert result.outcome == WaveExecutionOutcome.ALL_SUCCESS
    assert result.unique_source_refs == ("https://example.test/a",)
    assert all(observation.structured_error is None for observation in result.observations)


@pytest.mark.asyncio
async def test_url_normalization_rejects_userinfo_and_scheme_relative_urls() -> None:
    fake = _FakeSearch(
        [
            _result(
                [
                    "https://user:password@example.test/private",
                    "//example.test/scheme-relative",
                    "https://example.test/accepted",
                ]
            )
        ]
    )
    result = await SearchWaveExecutor(fake).execute(_plan((_query(0, "safe"),)))
    assert result.unique_source_refs == ("https://example.test/accepted",)


@pytest.mark.asyncio
async def test_partial_success_preserves_successful_evidence() -> None:
    fake = _FakeSearch([_result(["https://ok.test/"], "one"), _error_result()])
    result = await SearchWaveExecutor(fake).execute(
        _plan((_query(0, "ok"), _query(1, "bad")))
    )
    assert result.outcome == WaveExecutionOutcome.PARTIAL_SUCCESS
    assert result.unique_source_refs == ("https://ok.test",)
    assert result.observations[1].structured_error is not None


@pytest.mark.asyncio
async def test_all_empty_and_all_failed_outcomes() -> None:
    empty = await SearchWaveExecutor(_FakeSearch([_result([])])).execute(
        _plan((_query(0, "empty"),))
    )
    failed = await SearchWaveExecutor(_FakeSearch([_error_result()])).execute(
        _plan((_query(0, "failed"),))
    )
    assert empty.outcome == WaveExecutionOutcome.ALL_EMPTY
    assert failed.outcome == WaveExecutionOutcome.ALL_FAILED


@pytest.mark.asyncio
async def test_empty_plus_failed_without_evidence_is_not_partial_success() -> None:
    result = await SearchWaveExecutor(
        _FakeSearch([_result([]), _error_result()])
    ).execute(_plan((_query(0, "empty"), _query(1, "failed"))))

    assert result.unique_source_refs == ()
    assert result.outcome == WaveExecutionOutcome.PARTIAL_NO_EVIDENCE
    assert result.outcome != WaveExecutionOutcome.PARTIAL_SUCCESS


@pytest.mark.asyncio
async def test_malformed_and_callable_failure_are_structured_and_continue() -> None:
    fake = _FakeSearch([{"unexpected": True}, RuntimeError("secret traceback")])
    result = await SearchWaveExecutor(fake).execute(
        _plan((_query(0, "malformed"), _query(1, "raises")))
    )
    assert result.outcome == WaveExecutionOutcome.ALL_FAILED
    assert json.loads(result.observations[0].structured_error or "{}") == {
        "code": "malformed_result"
    }
    assert json.loads(result.observations[1].structured_error or "{}") == {
        "code": "search_callable_failed"
    }
    assert "secret" not in (result.observations[1].structured_error or "")


@pytest.mark.asyncio
async def test_iterable_materialization_failure_is_malformed_and_wave_continues() -> None:
    class ExplodingRows:
        backend_used = "malformed"

        @property
        def results(self):
            def rows():
                raise RuntimeError("iterator detail must not escape")
                yield {"url": "https://never.test"}

            return rows()

    result = await SearchWaveExecutor(
        _FakeSearch([ExplodingRows(), _result(["https://ok.test/source"])])
    ).execute(_plan((_query(0, "malformed iterable"), _query(1, "ok"))))

    assert json.loads(result.observations[0].structured_error or "{}") == {
        "code": "malformed_result"
    }
    assert result.unique_source_refs == ("https://ok.test/source",)
    assert result.outcome == WaveExecutionOutcome.PARTIAL_SUCCESS


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", [asyncio.CancelledError, SystemExit])
async def test_terminal_iterable_semantics_are_not_swallowed(terminal: type[BaseException]) -> None:
    class TerminalRows:
        backend_used = "terminal"

        @property
        def results(self):
            def rows():
                raise terminal()
                yield {"url": "https://never.test"}

            return rows()

    with pytest.raises(terminal):
        await SearchWaveExecutor(_FakeSearch([TerminalRows()])).execute(
            _plan((_query(0, "terminal"),))
        )


@pytest.mark.asyncio
async def test_cancellation_is_not_swallowed_as_search_failure() -> None:
    class CancelledSearch:
        async def __call__(self, **_: Any) -> SearchResult:
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await SearchWaveExecutor(CancelledSearch()).execute(
            _plan((_query(0, "cancelled"),))
        )


@pytest.mark.asyncio
async def test_structured_error_and_observation_fields_are_truthful() -> None:
    fake = _FakeSearch([_error_result()])
    result = await SearchWaveExecutor(fake).execute(_plan((_query(0, "error"),)))
    observation = result.observations[0]
    payload = json.loads(observation.structured_error or "{}")

    assert payload == {
        "backend": "fake",
        "code": "RATE_LIMITED",
        "diagnostic_category": "rate_limit",
        "http_status": 429,
        "retryable": True,
    }
    assert observation.backend == "fake"
    assert observation.result_refs == ()
    assert observation.attempt_count == 1
    assert isinstance(observation.duration_ms, int)
    assert observation.duration_ms >= 0
    assert dict(observation.local_usage) == {"search_calls": 1}


@pytest.mark.asyncio
async def test_plan_hash_and_object_are_not_mutated() -> None:
    plan = _plan((_query(0, "long brief"),))
    before = plan.research_brief_sha256
    result = await SearchWaveExecutor(_FakeSearch([_result([])])).execute(plan)
    assert result.plan is plan
    assert plan.research_brief_sha256 == before
    with pytest.raises(FrozenInstanceError):
        result.outcome = WaveExecutionOutcome.ALL_SUCCESS  # type: ignore[misc]


def test_executor_has_no_provider_implementation_imports() -> None:
    source = inspect.getsource(__import__(
        "hermes.deep_research.wave_execution", fromlist=["SearchWaveExecutor"]
    ))
    assert "hermes.services.search.tavily" not in source
    assert "hermes.services.search.exa" not in source
    assert "hermes.services.search.searxng" not in source


def test_positive_cap_is_required() -> None:
    fake = _FakeSearch([])
    with pytest.raises(ValueError):
        SearchWaveExecutor(fake, max_unique_sources=0)
    with pytest.raises(ValueError):
        SearchWaveExecutor(fake, max_unique_sources=True)  # type: ignore[arg-type]

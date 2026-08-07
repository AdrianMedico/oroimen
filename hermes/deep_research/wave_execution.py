"""Deterministic execution of one bounded Deep Research SearchWave.

The executor owns wave semantics, not provider selection.  It runs a frozen
``SearchPlan`` sequentially, records one observation per query, and returns
globally deduplicated source references with first-query provenance.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Protocol
from urllib.parse import urlsplit

from hermes.deep_research.planning import (
    PlannedSearchQuery,
    SearchObservation,
    SearchPlan,
)
from hermes.services.search.protocol import SearchResult


class WaveExecutionOutcome(StrEnum):
    """Aggregate result for one SearchWave."""

    ALL_SUCCESS = "all_success"
    PARTIAL_SUCCESS = "partial_success"
    PARTIAL_NO_EVIDENCE = "partial_no_evidence"
    ALL_EMPTY = "all_empty"
    ALL_FAILED = "all_failed"


@dataclass(frozen=True)
class WaveExecutionResult:
    """Immutable wave evidence and first-query source provenance."""

    plan: SearchPlan
    observations: tuple[SearchObservation, ...]
    unique_source_refs: tuple[str, ...]
    source_query_ids: Mapping[str, str]
    outcome: WaveExecutionOutcome
    unique_source_cap: int


class SearchCallable(Protocol):
    """Minimal provider-replaceable search boundary."""

    async def __call__(
        self,
        *,
        query: str,
        intent: str,
        content: str,
        num_results: int,
    ) -> Any: ...


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _structured_error(error: Any) -> str:
    """Serialize only stable, non-sensitive SearchError fields."""

    payload = {
        "backend": getattr(error, "backend", None),
        "code": _enum_value(getattr(error, "code", None)),
        "diagnostic_category": _enum_value(
            getattr(error, "diagnostic_category", None)
        ),
        "breaker_relevant": bool(getattr(error, "breaker_relevant", False)),
        "http_status": getattr(error, "http_status", None),
        "retryable": bool(getattr(error, "retryable", False)),
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _failure(code: str) -> str:
    return json.dumps({"code": code}, separators=(",", ":"), sort_keys=True)


def _normalize_url(raw: Any) -> str | None:
    if not isinstance(raw, str):
        return None
    value = raw.strip().rstrip("/")
    if not value:
        return None
    try:
        parsed = urlsplit(value)
    except ValueError:
        return None
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return None
    if parsed.username is not None or parsed.password is not None:
        return None
    return value


def _rows(result: Any) -> tuple[Iterable[Any], str | None] | None:
    if isinstance(result, SearchResult):
        return result.results, result.backend_used
    if isinstance(result, list):
        return result, None
    try:
        raw_rows = result.results
    except AttributeError:
        return None
    except Exception:
        return None
    try:
        backend = result.backend_used
    except AttributeError:
        backend = None
    except Exception:
        return None
    return raw_rows, backend


def _candidate_urls(result: Any) -> tuple[tuple[str, ...], str | None] | None:
    extracted = _rows(result)
    if extracted is None:
        return None
    raw_rows, backend = extracted
    try:
        rows = tuple(raw_rows)
    except Exception:
        return None

    refs: list[str] = []
    for row in rows:
        raw_url: Any
        if isinstance(result, list):
            if not isinstance(row, str):
                return None
            raw_url = row
        else:
            if not isinstance(row, dict):
                return None
            raw_url = row.get("url")
        normalized = _normalize_url(raw_url)
        if normalized is not None:
            refs.append(normalized)
    return tuple(refs), backend


class SearchWaveExecutor:
    """Execute one valid SearchPlan in ordinal order, without concurrency."""

    def __init__(
        self,
        search: SearchCallable,
        *,
        max_unique_sources: int = 5,
        content_mode: str = "snippet",
        num_results: int = 5,
    ) -> None:
        if (
            not isinstance(max_unique_sources, int)
            or isinstance(max_unique_sources, bool)
            or max_unique_sources <= 0
        ):
            raise ValueError("max_unique_sources must be a positive int")
        if not isinstance(content_mode, str) or not content_mode:
            raise ValueError("content_mode must be a non-empty string")
        if not isinstance(num_results, int) or isinstance(num_results, bool) or num_results <= 0:
            raise ValueError("num_results must be a positive int")
        self._search = search
        self._max_unique_sources = max_unique_sources
        self._content_mode = content_mode
        self._num_results = num_results

    async def execute(self, plan: SearchPlan) -> WaveExecutionResult:
        observations: list[SearchObservation] = []
        unique_refs: list[str] = []
        source_query_ids: dict[str, str] = {}
        failures = 0

        for query in sorted(plan.queries, key=lambda item: item.ordinal):
            observation, refs = await self._execute_query(plan.wave_index, query)
            observations.append(observation)
            if observation.structured_error is not None:
                failures += 1
                continue
            for ref in refs:
                if ref in source_query_ids:
                    continue
                if len(unique_refs) >= self._max_unique_sources:
                    break
                unique_refs.append(ref)
                source_query_ids[ref] = query.query_id

        if failures == len(observations):
            outcome = WaveExecutionOutcome.ALL_FAILED
        elif not unique_refs:
            outcome = (
                WaveExecutionOutcome.PARTIAL_NO_EVIDENCE
                if failures
                else WaveExecutionOutcome.ALL_EMPTY
            )
        elif failures:
            outcome = WaveExecutionOutcome.PARTIAL_SUCCESS
        elif unique_refs:
            outcome = WaveExecutionOutcome.ALL_SUCCESS
        else:
            outcome = WaveExecutionOutcome.ALL_EMPTY

        return WaveExecutionResult(
            plan=plan,
            observations=tuple(observations),
            unique_source_refs=tuple(unique_refs),
            source_query_ids=MappingProxyType(source_query_ids),
            outcome=outcome,
            unique_source_cap=self._max_unique_sources,
        )

    async def _execute_query(
        self,
        wave_index: int,
        query: PlannedSearchQuery,
    ) -> tuple[SearchObservation, tuple[str, ...]]:
        started = time.monotonic()
        try:
            result = await self._search(
                query=query.text,
                intent="deep_research",
                content=self._content_mode,
                num_results=self._num_results,
            )
        except Exception:
            return (
                self._observation(
                    wave_index=wave_index,
                    query_id=query.query_id,
                    structured_error=_failure("search_callable_failed"),
                    duration_ms=self._duration_ms(started),
                ),
                (),
            )

        if isinstance(result, SearchResult) and result.error is not None:
            return (
                self._observation(
                    wave_index=wave_index,
                    query_id=query.query_id,
                    backend=result.backend_used,
                    structured_error=_structured_error(result.error),
                    duration_ms=self._duration_ms(started),
                ),
                (),
            )

        candidates = _candidate_urls(result)
        if candidates is None:
            return (
                self._observation(
                    wave_index=wave_index,
                    query_id=query.query_id,
                    structured_error=_failure("malformed_result"),
                    duration_ms=self._duration_ms(started),
                ),
                (),
            )
        refs, backend = candidates
        return (
            self._observation(
                wave_index=wave_index,
                query_id=query.query_id,
                backend=backend,
                result_refs=refs,
                duration_ms=self._duration_ms(started),
            ),
            refs,
        )

    @staticmethod
    def _duration_ms(started: float) -> int:
        return max(0, int((time.monotonic() - started) * 1000))

    @staticmethod
    def _observation(
        *,
        wave_index: int,
        query_id: str,
        backend: str | None = None,
        result_refs: tuple[str, ...] = (),
        structured_error: str | None = None,
        duration_ms: int | None = None,
    ) -> SearchObservation:
        return SearchObservation(
            wave_index=wave_index,
            query_id=query_id,
            backend=backend,
            result_refs=result_refs,
            structured_error=structured_error,
            attempt_count=1,
            duration_ms=duration_ms,
            local_usage={"search_calls": 1},
        )


__all__ = [
    "SearchCallable",
    "SearchWaveExecutor",
    "WaveExecutionOutcome",
    "WaveExecutionResult",
]

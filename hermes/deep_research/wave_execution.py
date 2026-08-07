"""Deterministic execution of one bounded Deep Research SearchWave.

The executor owns wave semantics, not provider selection.  It runs a frozen
``SearchPlan`` sequentially, records one observation per query, and returns
globally deduplicated source references with first-query provenance.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Protocol
from urllib.parse import parse_qsl, urlsplit

from hermes.deep_research.planning import (
    MAX_QUERY_CHARS,
    EvidenceItem,
    PlannedSearchQuery,
    SearchObservation,
    SearchPlan,
    compute_evidence_digest,
)
from hermes.services.search.errors import (
    SearchDiagnosticCategory,
    SearchError,
    SearchErrorCode,
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


MAX_RESULT_ROWS = 32
MAX_SOURCE_REF_CHARS = 2_048
MAX_BACKEND_CHARS = 128
MAX_ERROR_TEXT_CHARS = 512


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


class WaveExecutionCancelled(Exception):
    """Cooperative cancellation observed between query dispatches."""


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


def _safe_search_error(error: Any) -> bool:
    if type(error) is not SearchError:
        return False
    return (
        type(error.code) is SearchErrorCode
        and type(error.message) is str
        and len(error.message) <= MAX_ERROR_TEXT_CHARS
        and (error.backend is None or type(error.backend) is str)
        and (error.backend is None or len(error.backend) <= MAX_BACKEND_CHARS)
        and type(error.retryable) is bool
        and type(error.suggestion) is str
        and len(error.suggestion) <= MAX_ERROR_TEXT_CHARS
        and type(error.backends_tried) is list
        and len(error.backends_tried) <= 16
        and all(
            type(value) is str and len(value) <= MAX_BACKEND_CHARS
            for value in error.backends_tried
        )
        and type(error.reasons) is dict
        and len(error.reasons) <= 16
        and all(
            type(key) is str
            and len(key) <= MAX_BACKEND_CHARS
            and type(value) is str
            and len(value) <= MAX_ERROR_TEXT_CHARS
            for key, value in error.reasons.items()
        )
        and type(error.breaker_relevant) is bool
        and (
            error.http_status is None
            or (type(error.http_status) is int and 100 <= error.http_status <= 599)
        )
        and type(error.diagnostic_category) is SearchDiagnosticCategory
    )


def _valid_search_result_envelope(result: Any, expected_query: str) -> bool:
    """Validate the exact provider result envelope before any branch uses it."""

    if type(result) is not SearchResult:
        return False
    return (
        type(result.backend_used) is str
        and len(result.backend_used) <= MAX_BACKEND_CHARS
        and type(result.query) is str
        and len(result.query) <= MAX_QUERY_CHARS
        and result.query == expected_query
        and type(result.results) is list
        and (result.error is None or _safe_search_error(result.error))
    )


def _failure(code: str) -> str:
    return json.dumps({"code": code}, separators=(",", ":"), sort_keys=True)


def _normalize_url(raw: Any) -> str | None:
    if type(raw) is not str:
        return None
    if len(raw) > MAX_SOURCE_REF_CHARS:
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
    # Persisted source refs are intentionally allow-listed to origin/path.
    # Query strings and fragments may contain signed URLs or session tokens.
    if parsed.query or parsed.fragment:
        return None
    try:
        if parse_qsl(parsed.query, keep_blank_values=True, max_num_fields=64):
            return None
    except ValueError:
        return None
    return value


def _rows(result: Any) -> tuple[Iterable[Any], str | None] | None:
    # Only exact built-in/domain values cross this synchronous materializer.
    # Reject subclasses before touching properties or overridden methods;
    # malformed provider objects must not block the event loop.
    if type(result) is SearchResult:
        if (
            type(result.backend_used) is not str
            or len(result.backend_used) > MAX_BACKEND_CHARS
        ):
            return None
        return result.results, result.backend_used
    if type(result) is list:
        return result, None
    return None


def _evidence_digest(raw_row: Any, normalized_url: str) -> str:
    return compute_evidence_digest(
        normalized_url,
        _bounded_text(raw_row, "title", 512),
        _bounded_text(raw_row, "snippet", 1_024),
    )


def _bounded_text(raw_row: Any, key: str, max_chars: int) -> str:
    if type(raw_row) is not dict:
        return ""
    value = raw_row.get(key)
    if type(value) is not str:
        return ""
    return " ".join(value[:max_chars].split())[:max_chars]


def _candidate_urls(
    result: Any,
) -> tuple[tuple[tuple[str, str, str, str], ...], str | None] | None:
    extracted = _rows(result)
    if extracted is None:
        return None
    raw_rows, backend = extracted
    # A concrete list/tuple is already bounded by the provider boundary. An
    # arbitrary synchronous iterator cannot be preempted safely in-process;
    # reject it as malformed so a job deadline cannot leave an unbounded
    # worker behind.
    if type(raw_rows) not in (list, tuple):
        return None
    try:
        iterator = iter(raw_rows)
    except Exception:
        return None

    refs: list[tuple[str, str, str, str]] = []
    try:
        for index, row in enumerate(iterator):
            if index >= MAX_RESULT_ROWS:
                return None
            raw_url: Any
            if type(result) is list:
                if type(row) is not str:
                    return None
                raw_url = row
            else:
                if type(row) is not dict:
                    return None
                raw_url = row.get("url")
            normalized = _normalize_url(raw_url)
            if normalized is not None:
                refs.append(
                    (
                        normalized,
                        _evidence_digest(row, normalized),
                        _bounded_text(row, "title", 512),
                        _bounded_text(row, "snippet", 1_024),
                    )
                )
    except Exception:
        return None
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

    async def execute(
        self,
        plan: SearchPlan,
        *,
        completed_observations: tuple[SearchObservation, ...] = (),
        on_observation: Callable[[SearchObservation], None] | None = None,
        on_dispatch: Callable[[PlannedSearchQuery], None] | None = None,
        cancellation: Callable[[], bool] | None = None,
    ) -> WaveExecutionResult:
        ordered_queries = sorted(plan.queries, key=lambda item: item.ordinal)
        if not isinstance(completed_observations, tuple):
            raise ValueError("completed_observations must be a tuple")
        if len(completed_observations) > len(ordered_queries):
            raise ValueError("completed observations exceed the plan")

        observations: list[SearchObservation] = []
        unique_refs: list[str] = []
        source_query_ids: dict[str, str] = {}
        failures = 0

        for query, observation in zip(
            ordered_queries[: len(completed_observations)],
            completed_observations,
            strict=True,
        ):
            if observation.wave_index != plan.wave_index or observation.query_id != query.query_id:
                raise ValueError("completed observations must be an ordinal plan prefix")
            if observation.local_usage.get("search_calls") != 1:
                raise ValueError("completed observations must report one search call")
            observations.append(observation)
            if observation.structured_error is not None:
                failures += 1
            self._record_refs(
                query_id=query.query_id,
                refs=observation.result_refs,
                unique_refs=unique_refs,
                source_query_ids=source_query_ids,
            )

        for query in ordered_queries[len(completed_observations) :]:
            if cancellation is not None and cancellation():
                raise WaveExecutionCancelled
            if on_dispatch is not None:
                on_dispatch(query)
            observation, refs = await self._execute_query(plan.wave_index, query)
            observations.append(observation)
            if observation.structured_error is not None:
                failures += 1
            self._record_refs(
                query_id=query.query_id,
                refs=refs,
                unique_refs=unique_refs,
                source_query_ids=source_query_ids,
            )
            if on_observation is not None:
                on_observation(observation)

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

    def _record_refs(
        self,
        *,
        query_id: str,
        refs: Iterable[Any],
        unique_refs: list[str],
        source_query_ids: dict[str, str],
    ) -> None:
        for raw_ref in refs:
            ref = _normalize_url(raw_ref)
            if ref is None or ref in source_query_ids:
                continue
            if len(unique_refs) >= self._max_unique_sources:
                break
            unique_refs.append(ref)
            source_query_ids[ref] = query_id

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

        try:
            if type(result) is SearchResult and not _valid_search_result_envelope(
                result, query.text
            ):
                raise ValueError("malformed search result envelope")
            if type(result) is SearchResult and result.error is not None:
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
                raise ValueError("malformed search result")
            candidates_with_digests, backend = candidates
            refs = tuple(ref for ref, _digest, _title, _snippet in candidates_with_digests)
            evidence_digests = tuple(
                digest for _ref, digest, _title, _snippet in candidates_with_digests
            )
            evidence_items = tuple(
                EvidenceItem(
                    query_id=query.query_id,
                    source_ref=ref,
                    title=title,
                    snippet=snippet,
                    digest=digest,
                )
                for ref, digest, title, snippet in candidates_with_digests
            )
            return (
                self._observation(
                    wave_index=wave_index,
                    query_id=query.query_id,
                    backend=backend,
                    result_refs=refs,
                    evidence_items=evidence_items,
                    evidence_digests=evidence_digests,
                    duration_ms=self._duration_ms(started),
                ),
                refs,
            )
        except Exception:
            return (
                self._observation(
                    wave_index=wave_index,
                    query_id=query.query_id,
                    structured_error=_failure("malformed_result"),
                    duration_ms=self._duration_ms(started),
                ),
                (),
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
        evidence_items: tuple[EvidenceItem, ...] = (),
        evidence_digests: tuple[str, ...] = (),
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
            evidence_items=evidence_items,
            evidence_digests=evidence_digests,
            local_usage={"search_calls": 1},
        )


__all__ = [
    "SearchCallable",
    "SearchWaveExecutor",
    "WaveExecutionCancelled",
    "WaveExecutionOutcome",
    "WaveExecutionResult",
]

# Deep Research query-planning design

- Status: Accepted
- Date: 2026-08-03
- Baseline: `8f08c42e316a6044390e1f6ba4915ce053d919cb`
- Scope: design only; no production behavior, schema, provider, or benchmark change
- Architectural decision: `docs/ADR_DEEP_RESEARCH_QUERY_PLANNING.md`

> **Governance alignment.** The associated ADR
> (`docs/ADR_DEEP_RESEARCH_QUERY_PLANNING.md`) is accepted for
> architecture and v1 design contracts. This design document
> remains non-runtime and unimplemented; it introduces design
> truth only and does not change runtime behavior. Implementation
> planning requires a fresh baseline check and a bounded
> implementation mission. ADR acceptance is not implementation
> authorization, merge authorization, provider authorization,
> or product acceptance.

## 1. Purpose

This document defines a bounded, provider-independent design for turning a
Deep Research question into one direct search or a small auditable set of
focused searches.

It also repairs the search-error boundary that currently converts structured
router failures into retryable `search_5xx/no_results`.

The design is intentionally a modular-monolith change. It adds no service, no
agent framework, no autonomous loop, and no new external dependency.

## 2. Current baseline facts

The following statements are facts at the pinned baseline.

### 2.1 Research-job input

`CreateJobRequest.query` accepts a natural-language string between 3 and 2,000
characters. The original string is stored in the research job.

### 2.2 Current phase-1 shape

`DeepResearchService._phase_search`:

1. reads the original job query;
2. invokes the injected search callable once with
   `intent="deep_research"`;
3. extracts URLs from one result;
4. treats an empty URL list as retryable `search_5xx/no_results`; and
5. returns at most `deep_research_max_sources` URLs.

There is no plan object, no query fan-out, and no per-source query provenance.

### 2.3 Current router and backend shape

The common `BackendProtocol` declares:

- backend name;
- supported content modes;
- `search`;
- `has_budget`; and
- `health_check`.

It does not declare query-length constraints.

The router has one generic `_MAX_QUERY_CHARS = 2000` guard and silently slices
queries above that value. For `deep_research`, the intent map selects Tavily
when configured. SearXNG and Exa adapters are already implemented.

Tavily forwards the query unchanged in a JSON POST. SearXNG URL-encodes the
query and forwards it through its self-hosted meta-search endpoint. Exa
forwards the query unchanged in a JSON POST.

### 2.4 Current error-boundary loss

The router always returns `SearchResult`. Its `error` field carries a
structured `SearchError`.

On a generic backend exception, the router currently:

- records a circuit-breaker failure; and
- returns `SearchErrorCode.INVALID_RESPONSE` with `retryable=False`.

`_phase_search` does not inspect `SearchResult.error`. Because the result has no
URLs, it raises `search_5xx/no_results` with `retryable=True`.

Therefore, the current observed retry behavior is not explained only by the
exception-text classifier. The structured router error is lost before phase
retry classification.

### 2.5 Current retry contract

`PhaseError` contains both `taxonomy` and `retryable`.

The retry helper currently decides by membership in the static
`RETRYABLE_ERRORS` taxonomy set rather than by `PhaseError.retryable`.
This prevents the system from expressing exceptions such as a retryable 429
without overloading a broad taxonomy.

### 2.6 Current search accounting

The current search-budget behavior is a pre-dispatch local debit serialized
by the selected backend semaphore. It records Oroimen-local allowance
consumption and is not provider billing or invoice truth. The current
`has_budget()` plus later multi-unit debit is not count-aware admission: a
request can theoretically cross the configured local limit when the remaining
count is below the required debit. A count-aware
`reserve_if_remaining(count)` or equivalent is an open implementation
decision.

### 2.7 Current preflight and documentation

The public preflight contract contains
`dr.architecture.query_decomposition`. At the baseline it is an optional WARN
because decomposition is not implemented.

The public capability ledger treats documented-but-unimplemented capabilities
as `Design only` and distinguishes code, runtime wiring, deterministic
integration, live-provider evidence, quality measurement, and product support.

The current Deep Research egress ADR and Preflight design are historical slice
documents. Their still-supported statements include: query decomposition was
previously deferred, preflight has an architecture marker for decomposition,
offline and live proof are distinct, and safe egress remains explicit. Their
slice-specific runtime statements are NOT current runtime truth and must not
be copied into the new documents.

## 3. Goals

The design must:

1. preserve the original research question byte-for-byte after API
   normalization;
2. support explicit direct and decomposed search plans;
3. keep planning independent from the selected backend;
4. produce bounded, focused, auditable search queries;
5. validate each query against the selected backend contract before dispatch;
6. preserve search-query-to-source provenance;
7. keep the final unique-source cap global;
8. distinguish validation errors, provider errors, and valid empty results;
9. apply retry and circuit-breaker behavior from structured semantics;
10. make local search accounting truthful about what it measures;
11. remain deterministically testable without external providers; and
12. remain maintainable by one developer.

## 4. Non-goals

The initial implementation does not include:

- recursive planning;
- reflection or autonomous re-query loops;
- an LLM planner unless the offline design gate justifies it;
- broad source-authority scoring;
- claim verification;
- a new worker or microservice;
- a new database table by default;
- provider changes;
- benchmark text changes;
- live provider calls in tests;
- parallel fan-out in the first implementation;
- a change to the public `POST /v1/jobs` request shape; or
- a claim of improved research quality before measurement.

## 5. Conceptual architecture

```text
Persisted original research query (normalized string)
        |
        v
PlanningDecision
    |             |
    | DIRECT      | DECOMPOSE
    v             v
SearchPlan with 1..N bounded PlannedSearchQuery items
          |
          v
Sequential SearchExecutor
          |
          v
Search router: one query, one selected backend, structured result/error
          |
          v
SearchObservation[] with query_id and backend provenance
          |
          v
Normalize -> sanitize -> cross-query deduplicate -> globally select
          |
          v
EvidenceSet (max unique URLs = deep_research_max_sources)
          |
          v
Existing scrape and synthesis phases using the original research query
```

Planning belongs to Deep Research orchestration. The generic router remains a
single-query execution boundary.

## 6. Proposed domain contracts

Names below describe required semantics. Exact Python names may change during
implementation if existing repository conventions support a clearer fit. None
of these are public HTTP DTOs.

### 6.1 Research question

The original research question is the normalized query string already
persisted with the job. It is not a separate wrapper class.

The question text is never replaced or rewritten by planning. A hash may be
derived only at an evidence or checkpoint boundary. Language detection or
declaration is an open decision; v1 does not require it.

### 6.2 `SearchPlan`

| Field | Meaning |
| --- | --- |
| `schema_version` | Versioned plan contract |
| `planner_kind` | `direct` or the accepted bounded planner identifier |
| `planner_version` | Stable implementation version |
| `research_question_sha256` | Binds the plan to the original question |
| `queries` | Ordered, non-empty list of `PlannedSearchQuery` |
| `limits` | Effective planning and provider constraints |
| `created_at` | Evidence timestamp, not an authorization timestamp |

### 6.3 `PlannedSearchQuery`

| Field | Meaning |
| --- | --- |
| `query_id` | Stable deterministic ID within the plan |
| `text` | Concrete string passed to the router |
| `purpose` | Short human-readable evidence need |
| `dimensions` | Stable labels for original-question dimensions covered |
| `ordinal` | Deterministic execution order |

The query text must be non-empty and valid for the selected backend before any
network call.

### 6.4 `BackendQueryCapabilities`

Initial scope contains only what the current problem requires:

| Field | Type | Meaning |
| --- | --- | --- |
| `max_query_chars` | `int \| None` | Provider-specific operational limit enforced locally before dispatch when known. `None` does NOT mean unlimited; it means no Oroimen-side provider-specific local rejection is applied. |

Content-mode capabilities remain where they are today.

The initial change must not grow this into a generic feature-negotiation
framework. Provider capability truth and Oroimen API input validation are
separate contracts; the existing general research-question/API limit remains
independently applicable.

### 6.5 `SearchObservation`

| Field | Meaning |
| --- | --- |
| `query_id` | Planned query that produced this observation |
| `backend_used` | Actual backend selected by the router |
| `results` | Normalized backend results |
| `error` | Structured error, if present |
| `attempt_count` | Local attempts for this query |
| `duration_ms` | Local elapsed time |
| `local_usage` | Local accounting facts, explicitly labeled |

### 6.6 Evidence candidate (conceptual aggregation)

Each candidate source carries:

- normalized URL;
- title and snippet/content returned by search;
- backend identity;
- one or more producing `query_id` values;
- original result rank per query; and
- deterministic merge order.

A URL returned by multiple queries is one candidate with multiple provenance
links, not multiple sources. `EvidenceCandidate` is a conceptual aggregation
shape, not a mandatory class.

## 7. Planning decision

### 7.1 Direct plan

The direct path emits one query when the question is already focused and the
query satisfies the selected backend contract.

The direct plan is still explicit and auditable. It is not a bypass around
validation or provenance.

### 7.2 Decomposed plan

The decomposed path emits a small ordered set of focused queries when the
question contains independent evidence needs.

Relevant signals include:

- explicit enumerated subquestions;
- comparison of multiple named entities across several axes;
- distinct time periods or jurisdictions;
- requirements to surface conflicting positions;
- a timeline plus current status plus enforcement or implementation evidence;
- several source classes;
- a query that cannot satisfy the selected backend contract directly.

Length alone is not the complexity definition.

### 7.3 Initial boundedness

The exact limit is an owner-visible open decision resolved by offline
measurement. The implementation contract nevertheless requires:

- a fixed maximum number of queries;
- a fixed maximum query length from backend capabilities;
- no recursive expansion;
- deterministic ordering;
- duplicate-query rejection;
- rejection of an empty or invalid plan; and
- a direct fallback only when it preserves the required dimensions.

The planner must fail closed rather than silently discard dimensions.

### 7.4 Planner strategy gate

The first gate is offline and provider-free.

Candidate deterministic strategies may use explicit structure already present
in the question, such as enumerated clauses, comparison entities, named axes,
time ranges, and source constraints.

The gate measures at least:

- coverage of material dimensions;
- preservation of named entities;
- preservation of dates, jurisdictions, and constraints;
- query validity;
- query count;
- duplicate rate;
- stability across repeated runs;
- legibility; and
- estimated search-call multiplier.

An LLM planner remains `OPEN — pending measurement`.

It becomes eligible only if the smallest maintainable deterministic strategy
fails a predeclared coverage threshold on the approved corpus. Any later LLM
planner requires structured output, hard query-count and length validation,
bounded tokens, deterministic fallback, separate cost accounting, and its own
authorization for live measurement.

## 8. Provider-capability semantics

### 8.1 Tavily

Initial declaration: `max_query_chars = 399`.

Rationale: current official guidance says queries should remain under 400
characters. 399 is Oroimen's conservative operational cap. It is a local
safety and compatibility contract, not a claim that every query of 399
characters will succeed.

The router validates locally before dispatch. An over-limit query becomes a
structured validation error and causes no backend request. The deterministic
test boundary is:

```text
398 characters: allowed
399 characters: allowed
400 characters: rejected locally
```

### 8.2 SearXNG

SearXNG itself accepts a `q` parameter and forwards terms to configured
upstream engines. Those engines can have different syntax and limits.

Initial declaration: `max_query_chars = None`.

This means no Oroimen-side universal provider-specific local hard cap has
been declared. It does NOT mean that the configured upstream engines accept
arbitrary query sizes. Focused planning remains useful for relevance,
coverage, and engine portability.

The local SearXNG adapter must not claim that its upstream engine set is
unlimited.

### 8.3 Exa and future providers

Initial declaration: `max_query_chars = None`.

A finite limit is NOT invented without primary evidence. A provider declares
a hard or operational limit only when supported by current primary
documentation or deterministic provider-contract evidence. Unknown values
remain explicit and are not silently treated as unlimited.

### 8.4 Capability drift

Provider limits can change. The capability constant must:

- name its primary source in code or adjacent documentation;
- be covered by a deterministic test;
- be reviewed when provider documentation changes; and
- fail safely if the provider rejects a supposedly valid query.

A declared limit is a local guard, not proof that every request below it is
valid.

## 9. Execution semantics

### 9.1 Sequential first implementation

The first implementation executes planned queries sequentially.

This is deliberate:

- simpler retry and accounting;
- deterministic ordering;
- lower burst risk;
- easier evidence review; and
- no need to introduce additional concurrency policy before quality and
  latency are measured.

Bounded parallel execution can be proposed later from measured latency and
provider limits.

### 9.2 Candidate request bound

Each query requests a bounded number of candidates. The exact value is
configuration or an internal constant selected during implementation review.

It must not cause the final source cap to multiply.

### 9.3 Merge order

The initial deterministic merge policy is:

1. normalize and sanitize each result using the existing router behavior;
2. preserve query order;
3. preserve provider rank within each query;
4. merge candidates round-robin by query to avoid one query consuming the
   entire source budget;
5. deduplicate by normalized URL;
6. aggregate all producing `query_id` values; and
7. stop after the global unique-source cap is reached.

A later relevance reranker is outside this slice.

### 9.4 `max_sources`

`deep_research_max_sources` is the global unique-source cap applied to the
final set of unique URLs passed from phase 1 to the later Deep Research
phases, after merge and deduplication. It is not multiplied by the number of
search queries.

It does not mean:

- maximum per subquery;
- total raw results retrieved; or
- number of search calls.

### 9.5 Partial success

A query-level deterministic validation failure indicates a planner bug and
fails the plan before execution.

During execution:

- successful query with results: contributes candidates;
- successful query with zero results: recorded as valid empty;
- non-retryable provider error: recorded and execution continues to remaining
  independent queries;
- retryable provider error: receives the bounded retry policy, then is
  recorded if exhausted.

After all queries:

- if at least one unique source exists, phase 1 may continue in a degraded
  state with the failed/empty query observations preserved;
- if no unique source exists, phase 1 fails with the most truthful aggregate
  error;
- a valid all-empty result is distinct from provider failure.

Whether degraded coverage is sufficient for synthesis is measured and may
later require a minimum-dimension gate. It is not silently assumed.

## 10. Error, retry, and circuit-breaker contract

### 10.1 Structured error preservation

`_phase_search` must inspect `SearchResult.error` before reading URLs.

The router-to-job bridge maps structured fields rather than parsing exception
strings.

### 10.2 Retry authority

`PhaseError.retryable` is authoritative for the phase retry loop.

Taxonomy remains useful for persistence and reporting, but a static taxonomy
membership list must not override an explicit structured retry decision.
Persisted taxonomy does not override explicit retryability.

Backward-compatible behavior for existing transient errors requires focused
regression tests.

### 10.3 Structured search failure contract

The internal search failure carries, at minimum:

```text
stable error code
backend identity
retryable boolean
breaker-relevant boolean
optional HTTP status
safe diagnostic category
```

The router preserves at minimum the following classes, which the contract
fields express:

- local query validation failure;
- HTTP status code when available;
- authentication or authorization failure;
- rate limiting;
- timeout;
- network failure;
- provider 5xx;
- invalid successful response; and
- valid empty results.

Raw provider response bodies and exception text are NOT part of the safe
contract. HTTP status is optional because not every transport failure has
one. No new persisted job taxonomy is required merely to preserve these
fields in memory. The router-to-job bridge must consume structured fields
and must not classify by parsing exception text.

A broad job taxonomy may map several router codes to `search_4xx` or
`search_5xx`, provided status, retryability, and safe diagnostic details are
not lost. A new persisted job taxonomy is not required unless the existing
schema proves insufficient.

### 10.4 Frozen v1 retry and breaker policy

| Condition                             | Retryable                                  | Circuit-breaker relevant |
| ------------------------------------- | ------------------------------------------ | ------------------------ |
| Local validation failure              | no                                         | no                       |
| HTTP 400 or 422                       | no                                         | no                       |
| HTTP 401 or 403                       | no                                         | no                       |
| HTTP 429                              | yes, using the existing bounded phase retry limit | no                |
| Provider HTTP 5xx                     | yes, bounded                               | yes                      |
| Timeout or network failure            | yes, bounded                               | yes                      |
| Successful response with zero results | not an error                               | no                       |
| Invalid HTTP-2xx provider response    | no by default                              | no by default            |

Additional notes:

- `Retry-After` handling is deferred unless later implementation evidence
  justifies it.
- HTTP 429 represents provider capacity or policy, not proof of provider
  outage; it is therefore not breaker-relevant.
- Invalid HTTP-2xx provider response behavior may be revisited from measured
  evidence.
- Deterministic request errors (validation, 4xx) must not open the breaker.

### 10.5 Circuit breaker

Only failures that indicate backend health degradation count toward the
breaker. The frozen policy in section 10.4 is authoritative.

Do not record breaker failure for:

- local validation;
- deterministic HTTP 400 or 422;
- authentication or authorization configuration errors (401, 403);
- rate limiting (429);
- valid empty results; or
- planner validation errors.

### 10.6 Safe diagnostics

Public API and preflight output must not expose:

- raw research questions;
- raw provider response bodies or exception text;
- credentials;
- headers;
- internal addresses; or
- local filesystem paths.

Evidence logs use hashes, stable error codes, lengths, counts, and redacted
metadata.

## 11. Search accounting design

The current search-budget behavior is a pre-dispatch local debit serialized
by the selected backend semaphore. It records Oroimen-local allowance
consumption and is not provider billing or invoice truth.

The current `has_budget()` plus later multi-unit debit is NOT count-aware
admission: a request can theoretically cross the configured local limit when
the remaining count is below the required debit. A count-aware
`reserve_if_remaining(count)` or equivalent is an open implementation
decision and is NOT selected by this documentation mission.

Local allowance accounting rules:

1. record Oroimen-local allowance consumption;
2. do not label the local allowance counter as provider-billed credits;
3. do not claim invoice truth from local counts;
4. reconcile provider usage only through a separately authorized provider
   usage surface.

Accounting needed for multiple planned searches belongs to PRE2-C unless
PRE2-A1 or PRE2-A2 evidence proves a smaller prerequisite is necessary. A
mandatory standalone PRE2-A3 accounting slice is NOT selected here.

The Deep Research cost estimator must expose the search-call assumption when
planning can emit more than one query. LLM paygo-equivalent cost and search
provider usage remain separate measures.

## 12. Provenance and persistence

### 12.1 Minimum evidence (for the planning slice)

A completed or failed job must make it possible for an authorized operator to
reconstruct:

- original research-question hash;
- planner kind and version;
- plan limits;
- ordered query IDs and query lengths;
- backend used for each query;
- result counts;
- normalized selected URLs or their approved evidence representation;
- query-to-source mapping;
- per-query errors and attempts;
- merge and dedup counts; and
- final unique-source count.

### 12.2 Persistence seam

The existing checkpoint writer is a candidate persistence seam. Current
phase-one checkpoint data contains URLs, not a reusable plan. Current startup
recovery re-enqueues work; it does not prove plan reuse.

PRE2-C must define plan write, load-before-dispatch, validation, version
mismatch policy, backend-capability snapshot, and reuse semantics. Merely
storing a plan does not prove deterministic recovery.

Preferred order:

1. typed in-memory plan and observations;
2. versioned checkpoint or job-evidence artifact using existing confined
   storage;
3. additive database persistence only if restart correctness, API behavior, or
   required auditability cannot be met otherwise.

No new table is selected unless the existing confined checkpoint or evidence
seam is proven insufficient.

No private run path or benchmark artifact enters the public repository.

### 12.3 Recovery

PRE2-C must define recovery reuse semantics. If phase 1 is checkpointed,
recovery must not regenerate a different plan for the same job without an
explicit version mismatch policy.

A persisted plan binds to:

- research-question hash;
- planner version;
- backend capability snapshot; and
- relevant planning limits.

## 13. Preflight and capability ledger

### 13.1 Preflight

Before runtime implementation:

- `dr.architecture.query_decomposition` remains WARN/not implemented.

After deterministic runtime wiring and tests:

- the capability can report PASS only when the actual composition root wires
  the accepted planner and its hard limits;
- offline preflight performs no provider call; and
- provider capability values need not be exposed if they reveal deployment
  policy, but the check must state whether a valid contract is configured.

A separate check may be warranted for backend query-contract availability if
it represents one stable condition. Do not overload the existing
decomposition check.

### 13.2 Public capability ledger

The documentation PR must amend the existing `Deep Research query
decomposition` row in `docs/CAPABILITY_LEDGER.md`. The row's table structure
and column count are preserved; the row is updated in place, not duplicated.

The amended row must communicate, using concise wording consistent with
adjacent rows:

- capability: Deep Research query decomposition / query planning;
- public evidence: this ADR and technical design;
- status: `Design only`;
- current runtime fact: phase 1 still performs one search query and has no
  planning fan-out or query-to-source provenance;
- disposition: retain as accepted design, with implementation and
  measurement deferred to separately scoped PRE2 implementation and
  measurement work;
- risks or dependencies: structured search-error truth, backend query
  capabilities, offline planner selection, bounded accounting, recovery
  semantics, and provider-specific live validation;
- acceptance: deterministic direct and decomposed planning, global source
  cap, query-to-source provenance, recovery reuse, and separate live and
  quality evidence.

The amended row must not claim implementation, runtime availability,
provider proof, research-quality improvement, or product support.

After implementation, its status advances only with matching evidence:

```text
Design only
-> Implemented, runtime unavailable
-> Implemented, runtime available behind opt-in
-> Deterministic vertical integration is proven
-> Live provider behavior is proven
-> Research quality is measured
```

The public ledger at the implementation commit must also update its baseline
header.

## 14. Security and privacy

- The planner receives only the research question already authorized for the
  Deep Research job.
- Planning does not expand egress authorization.
- Only concrete search-query text selected by the plan crosses the search
  provider boundary.
- The final synthesis provider still receives content according to the
  existing explicit egress contract.
- No query, source URL, provider body, or plan text is exposed through
  unauthenticated diagnostics.
- SearXNG remains a local service, but its configured upstream engines are
  external egress; local hosting does not make the underlying searches local.
- Provider fallback must remain visible in evidence.
- Planning must not introduce external model calls by default.

## 15. Implementation slices

### PRE2-A1 — truthful search-error bridge and retry/breaker semantics

No planning fan-out.

Required behavior:

- structured search failure (stable code, backend identity, retryable,
  breaker-relevant, optional HTTP status, safe diagnostic category);
- router-to-job preservation of structured fields without parsing exception
  text;
- `PhaseError.retryable` authoritative for the phase retry loop;
- frozen 400 / 401 / 403 / 429 / 5xx / timeout / network breaker policy;
- valid empty-result semantics distinct from provider failure;
- truthful local accounting labels (pre-dispatch local debit, not invoice
  truth); and
- deterministic tests.

This slice intentionally causes an incompatible long direct query to fail
once and truthfully. It does not yet make that question complete.

### PRE2-A2 — backend query capability and local validation

Depends on PRE2-A1.

Required behavior:

- minimal `BackendQueryCapabilities.max_query_chars: int | None`;
- Tavily operational cap 399 with the 398/399/400 deterministic test
  boundary;
- SearXNG and Exa initial `max_query_chars = None` (unknown-cap semantics;
  not unlimited);
- zero dispatch on locally invalid query;
- Deep Research no longer uses silent query slicing as repair; and
- deterministic tests.

### PRE2-B — offline planner gate

No provider calls, no production mutation.

Required output:

- candidate deterministic planning strategy;
- frozen measurement inputs approved for the experiment;
- dimension-coverage results;
- query-count and length distributions;
- duplicate rate;
- stability;
- multilingual observations;
- estimated search-call multiplier; and
- explicit `ACCEPT_DETERMINISTIC`, `JUSTIFY_LLM_PLANNER`, or
  `MEASURE_MORE` adjudication.

### PRE2-C — bounded planning implementation

Only after PRE2-B selection.

Required behavior:

- direct and decomposed plans;
- minimal plan and observation types (`SearchPlan`, `PlannedSearchQuery`,
  `SearchObservation`);
- sequential bounded execution;
- query-to-source provenance;
- deterministic merge and URL deduplication;
- global unique-source cap;
- partial, all-empty, and all-failed semantics;
- checkpoint plan write, load, validation, version mismatch, backend
  capability snapshot, and reuse semantics;
- N-search accounting truth and estimator assumptions;
- preflight and capability-ledger advancement only when evidence supports
  it; and
- deterministic vertical integration.

A mandatory standalone PRE2-A3 accounting slice is NOT selected. Accounting
needed for multiple planned searches belongs to PRE2-C unless PRE2-A1 or
PRE2-A2 evidence proves a smaller prerequisite is necessary.

### PRE2-D — live-provider and quality measurement

Separate sensitive authorization.

Required evidence:

- backend-specific contract behavior;
- real latency and provider usage reconciliation;
- first-wave reports;
- source coverage and duplicate rate;
- 17-dimension or successor rubric results;
- quality comparison against the pre-change baseline; and
- explicit owner decision before continuation.

## 16. Deterministic test matrix

### Search contract

- direct query below a backend limit dispatches unchanged;
- Tavily boundary: 398 characters allowed, 399 characters allowed, 400
  characters rejected locally with zero backend calls;
- query above the declared limit fails locally with zero backend calls;
- `max_query_chars = None` causes no provider-specific local rejection;
- independent generic/API input limits continue to apply; and
- capability, error, preflight, and diagnostic surfaces make no claim
  that the provider supports unlimited query length.
- Deep Research does not use generic silent truncation as repair;
- Tavily HTTP 4xx fixture retains status and structured error;
- 401 / 403 are non-retryable and do not affect breaker health;
- HTTP 429 follows its explicit retry policy and does not affect breaker
  health;
- provider HTTP 5xx, timeout, and network failures are retryable, bounded,
  and ARE breaker-relevant;
- valid empty HTTP 200 is not a 5xx.

### Retry bridge

- `_phase_search` inspects `SearchResult.error`;
- `PhaseError.retryable=False` prevents retries independently of taxonomy;
- transient errors still receive the configured attempt bound;
- exception text containing misleading digits cannot change taxonomy;
- breaker receives only health-relevant failures.

### Planning

- simple question produces one direct query;
- composite question produces an ordered bounded plan;
- original question remains unchanged;
- each query is non-empty and within backend limits;
- all material dimensions are represented or plan validation fails;
- duplicate queries are rejected;
- repeated deterministic planning is byte-stable;
- invalid planner output causes zero backend calls.

### Fan-out and evidence

- queries execute sequentially in deterministic order;
- URL normalization and dangerous-scheme filtering remain active;
- same URL from several queries becomes one source with several provenance
  links;
- round-robin merge prevents one query from consuming the global cap;
- final source count never exceeds `deep_research_max_sources`;
- one failed query plus successful evidence follows degraded semantics;
- all failures produce the truthful aggregate error;
- all valid-empty results remain distinguishable.

### Accounting and recovery

- local allowance remains a pre-dispatch local debit serialized by the
  selected backend semaphore;
- one direct query and N planned queries record truthful local attempts;
- failure and success observations are distinguishable;
- no local field claims actual provider billing;
- estimator exposes the search-call assumption;
- checkpoint recovery reuses the same plan (PRE2-C contract);
- plan-version mismatch fails safely or follows a documented migration
  policy.

### Boundaries

- default tests block non-loopback network;
- no credentials are read;
- no provider calls occur;
- public DTOs do not leak queries, paths, credentials, or raw errors;
- preflight remains offline; and
- existing short-query and deterministic Deep Research vertical tests remain
  green.

## 17. Open decisions

These decisions require measured evidence or implementation-seam inspection:

1. maximum planned queries per job;
2. candidate results requested per query;
3. deterministic decomposition algorithm;
4. dimension-coverage threshold;
5. degraded-evidence threshold;
6. exact persisted representation for the plan and observations;
7. whether count-aware local reservation (`reserve_if_remaining(count)` or
   equivalent) is required;
8. whether later bounded parallelism is justified;
9. whether an LLM planner is necessary;
10. `Retry-After` handling for HTTP 429 (deferred unless later implementation
    evidence justifies it);
11. whether invalid HTTP-2xx provider response behavior should be revised
    from measured evidence; and
12. whether the existing checkpoint / evidence seam is sufficient for plan
    persistence, or whether an additive change is required (PRE2-C decision).

None of these open items authorizes implementation scope expansion. Do not let
an open decision contradict a contract already frozen in this design.

## 18. Acceptance

The technical design is ready for implementation planning when:

- the associated ADR is accepted for architecture and v1 design contracts
  (this ADR-level gate is now satisfied; ADR acceptance is not
  implementation authorization);
- public/private publication boundaries are reviewed;
- every open decision needed by Slice A is resolved;
- Slice A has a bounded writable scope and deterministic tests;
- the offline planner measurement contract is separately frozen;
- no implementation mission silently includes provider calls, spending,
  schema changes, or benchmark changes; and
- the public repository baseline is reverified immediately before work.

## 19. Public references

- `docs/ADR_DEEP_RESEARCH_EGRESS.md` (historical slice document; do not
  treat its slice-specific runtime statements as current truth)
- `docs/DEEP_RESEARCH_PREFLIGHT_DESIGN.md` (historical slice document; the
  decomposition architecture marker is still supported)
- `docs/DR_Q1A_BASELINE_CALIBRATION_PLAN.md`
- `docs/CAPABILITY_LEDGER.md`
- `hermes/jobs/models.py`
- `hermes/jobs/exceptions.py`
- `hermes/jobs/service.py`
- `hermes/jobs/preflight.py`
- `hermes/services/search/protocol.py`
- `hermes/services/search/router.py`
- `hermes/services/search/errors.py`
- `hermes/services/search/budget.py`
- `hermes/services/search/resilience.py`
- `hermes/services/search/tavily.py`
- `hermes/services/search/searxng.py`
- `hermes/services/search/exa.py`
- Tavily Search best practices:
  `https://docs.tavily.com/documentation/best-practices/best-practices-search`
- Tavily Search API:
  `https://docs.tavily.com/documentation/api-reference/endpoint/search`
- SearXNG Search API:
  `https://docs.searxng.org/dev/search_api.html`

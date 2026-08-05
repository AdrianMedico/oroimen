# ADR: provider-independent Deep Research query planning

- Status: Accepted
- Date: 2026-08-03
- Baseline: `8f08c42e316a6044390e1f6ba4915ce053d919cb`
- Scope: design only; this document changes no runtime behavior

> **Acceptance scope.** The architecture and the documented v1
> contracts in this ADR are accepted. Acceptance is a governance
> decision about the design; it is not a runtime claim. It does
> not mean the design is implemented, does not prove deterministic
> integration, live-provider behavior, research-quality improvement,
> or product support. Implementation remains divided into
> separately scoped PRE2 stages. Changing a frozen v1 design
> contract requires a later bounded decision and must not be
> silently relaxed.

## Context

Oroimen currently accepts a natural-language Deep Research question of up to
2,000 characters, stores that question unchanged, and sends the same string to
the search router during phase 1. The router selects one backend for the
`deep_research` intent and executes one search call.

That shape conflates two different concepts:

1. the user's research question, which may contain several entities, time
   ranges, jurisdictions, comparison axes, evidence requirements, and output
   constraints; and
2. a search-engine query, which should be focused enough for a particular
   backend and its upstream engines.

The distinction matters even when a backend accepts a long string. A
multi-aspect research question can produce weak or uneven retrieval when it is
treated as one undifferentiated search query.

The concrete backends already expose different behavior:

- Tavily is selected for the current `deep_research` intent when configured.
  Its official search guidance requires concise queries under 400 characters
  and recommends separate focused searches for complex questions.
- SearXNG is already implemented as the privacy-first self-hosted backend. It
  forwards a query to one or more configured upstream engines, whose syntax,
  limits, and interpretation can differ.
- Exa is already implemented for semantic search and has its own request
  contract.

The current common backend protocol declares supported content modes but does
not declare a query-length capability. The router applies one generic
2,000-character limit and silently truncates only above that limit.

The current Deep Research preflight contract already reports
`dr.architecture.query_decomposition` as not implemented. The existing Deep
Research egress ADR (a historical slice document) explicitly deferred
decomposition for the earlier slice. The deferred decision is now material to
retrieval quality, provider replaceability, truthful failure handling, and
calibration.

The current search error contract also has a boundary loss: the router returns
a structured `SearchResult.error`, but Deep Research phase 1 does not inspect
that field. An empty result is therefore converted to `search_5xx/no_results`
even when the router already classified a non-retryable client error.

## Decision

### 1. Keep the research question distinct from search queries

The original research question is the immutable objective of the job. It is
stored and supplied to synthesis unchanged.

Search queries are derived execution artifacts. They may be shorter and more
focused, but they never replace or silently rewrite the research question.

### 2. Put planning in Deep Research, above the search router

Deep Research owns the decision to execute either:

- one direct search query; or
- a bounded plan containing multiple focused search queries.

The shared search router remains responsible for executing one concrete query
against one selected backend, enforcing that backend's declared contract, and
returning a structured result or error.

Provider adapters do not perform hidden decomposition or semantic truncation.

### 3. Make planning provider-independent

Query planning is a general Deep Research capability. It is not a Tavily-only
repair.

A plan may use backend capabilities as constraints, but the planner must not
encode provider-specific behavior into the meaning of the research question.
Switching among Tavily, SearXNG, Exa, or a future backend must not change the
planning contract.

### 4. Preserve a direct path for simple questions

Not every research question requires decomposition. The initial design must
support an explicit direct-search plan for questions that are already focused.

Planning is conditional and bounded. It must not turn a simple lookup into an
autonomous search loop.

### 5. Declare only the backend capabilities that are needed

The common backend contract will expose a provider-specific maximum query
length when a hard or operationally required limit is known.

The capability is typed `max_query_chars: int | None`. An integer value is a
provider-specific operational limit enforced locally before dispatch. A
`None` value means no Oroimen-side provider-specific local rejection is
applied; it does NOT mean or claim that the provider is unlimited. Provider
capability truth and Oroimen API input validation are separate contracts;
the existing general research-question/API limit remains independently
applicable.

The design must avoid a broad capability framework until additional concrete
provider differences require one.

### 6. Forbid silent semantic truncation

The system must not make an over-length provider query appear successful by
silently slicing its text.

A query that violates a backend contract is either:

- transformed by the explicit, auditable planning step before dispatch; or
- rejected locally with a structured, non-retryable error.

The existing generic 2,000-character truncation must not be used as a semantic
repair for Deep Research.

### 7. Preserve query-to-source provenance

Every executed search query has a stable identity. Retrieved sources retain
the identity of the query or queries that produced them.

After fan-out, Oroimen merges and normalizes results, deduplicates URLs, and
selects a bounded global evidence set. The final report is synthesized against
the original research question, not against any one subquery.

### 8. Keep `max_sources` global

For a decomposed plan, `deep_research_max_sources` means the maximum number of
unique source URLs passed from phase 1 into the later Deep Research phases
after merge and deduplication.

It is not multiplied by the number of search queries.

Candidate results requested from each backend may be separately bounded, but
that bound is an implementation parameter and does not redefine
`max_sources`.

### 9. Preserve structured errors and truthful retry semantics

Deep Research phase 1 must consume structured search errors rather than infer
failure type from an empty result or exception text.

The internal search failure carries, at minimum:

```text
stable error code
backend identity
retryable boolean
breaker-relevant boolean
optional HTTP status
safe diagnostic category
```

Raw provider response bodies and exception text are not part of the safe
contract. HTTP status is optional because not every transport failure has
one. No new persisted job taxonomy is required merely to preserve these
fields in memory. The router-to-job bridge must consume structured fields
and must not classify by parsing exception text.

`PhaseError.retryable` is authoritative for the phase retry loop. Persisted
taxonomy remains useful for reporting but does not override explicit
retryability.

Deterministic request errors do not retry and do not increment a provider
circuit breaker. Transient network, timeout, rate-limit, and server failures
retain bounded behavior appropriate to their classified error.

A successful response with zero results is distinct from an HTTP or local
validation failure.

### 10. Separate admission, local accounting, and provider billing truth

Search budget handling must distinguish:

- local admission or reservation;
- attempted requests;
- successful provider responses;
- provider-reported usage when available; and
- actual provider billing, which remains unknown unless reconciled from an
  authoritative provider surface.

Local counters must not be described as invoice truth. The current
search-budget behavior is a pre-dispatch local debit serialized by the
selected backend semaphore; it is not a count-aware reservation. A
count-aware `reserve_if_remaining(count)` or equivalent is an open
implementation decision and is not selected by this documentation mission.

### 11. Measure planner strategy before selecting it

The architecture requires bounded planning, but does not yet select a learned
planner.

The first planner gate is an offline, deterministic evaluation over approved
research questions. It compares the smallest maintainable deterministic
approach against the required coverage and constraint preservation.

An LLM-based planner is considered only if measured deterministic approaches
systematically lose material dimensions. It must not be introduced merely
because an LLM is available.

## Frozen v1 contracts

The contracts below are frozen for v1. Later changes require a new bounded
mission and must not be silently relaxed.

### Frozen structured search failure contract

The internal search failure carries, at minimum:

```text
stable error code
backend identity
retryable boolean
breaker-relevant boolean
optional HTTP status
safe diagnostic category
```

Raw provider response bodies and exception text are not part of the safe
contract. HTTP status is optional because not every transport failure has
one. No new persisted job taxonomy is required merely to preserve these
fields in memory. The router-to-job bridge must consume structured fields
and must not classify by parsing exception text.

`PhaseError.retryable` is authoritative for the phase retry loop. Persisted
taxonomy remains useful for reporting but does not override explicit
retryability.

### Frozen retry and breaker policy

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

`Retry-After` handling is deferred unless later implementation evidence
justifies it. HTTP 429 represents provider capacity or policy, not proof of
provider outage, and is therefore not breaker-relevant. Invalid HTTP-2xx
provider response behavior may be revisited from measured evidence.
Deterministic request errors must not open the breaker.

### Frozen `max_query_chars` initial values

- Tavily: `max_query_chars = 399`. Rationale: current official guidance says
  queries should remain under 400 characters. 399 is Oroimen's conservative
  operational cap. It is a local safety and compatibility contract, not a
  claim that every query of 399 characters will succeed.
- SearXNG: `max_query_chars = None` (no universal provider-specific hard cap
  declared; configured upstream engines are not unlimited).
- Exa: `max_query_chars = None` (no limit is invented without primary
  evidence).

The deterministic test boundary for Tavily is:

```text
398 characters: allowed
399 characters: allowed
400 characters: rejected locally
```

## Consequences

### Positive

- Deep Research can search complex questions more deliberately across
  different providers.
- The original user intent remains visible and correctable.
- Search-provider constraints become explicit instead of hidden in adapters.
- Query-to-source provenance improves auditability.
- Deterministic request failures stop contaminating retries and circuit
  breakers.
- Retrieval behavior can be measured independently from synthesis quality.
- Provider replacement does not require redesigning the research-question
  contract.

### Costs and risks

- A decomposed plan can increase search calls, latency, and provider usage.
- Multiple queries can return duplicate or uneven evidence.
- Partial failures require explicit semantics.
- Planner behavior becomes another surface that must be tested and versioned.
- An LLM planner, if later justified, adds cost and variability.
- Backend capability declarations can become stale and require maintenance.

### Deliberate limits

- No recursive research loop.
- No unbounded query generation.
- No new microservice.
- No new database schema unless the existing checkpoint and evidence seams
  prove insufficient.
- No provider-specific decomposition hidden inside an adapter.
- No benchmark rewrite merely to fit a provider.
- No claim of improved research quality until a frozen evaluation measures it.

## Alternatives considered

### Shorten or re-freeze research questions

Rejected as the default repair. It changes the question being measured and can
remove dimensions that a real user would reasonably ask Oroimen to research.

### Blind truncation

Rejected. It silently changes search intent and can discard the later parts of
a composite question.

### Derive one shorter query

Retained as a possible direct-plan strategy for some questions, but not as a
general substitute for multi-branch planning.

### Implement decomposition only in the Tavily adapter

Rejected. It creates hidden provider-specific semantics and does not improve
SearXNG, Exa, or future backends.

### Change the Deep Research backend

Not selected. A backend change affects source sets, cost, privacy, and
calibration, and does not remove the general value of planning.

### Use an LLM planner immediately

Deferred pending measurement. It adds another provider-dependent and
non-deterministic step before the smallest maintainable option has been tested.

## Verification

Acceptance of the eventual implementation requires deterministic, offline
evidence for:

- direct-search behavior for simple questions;
- preservation of the original research question;
- bounded and valid search plans;
- provider-capability enforcement at the router boundary;
- query-to-source provenance;
- URL deduplication across queries;
- a global unique-source cap;
- truthful empty-result and structured-error handling;
- no retry and no circuit-breaker increment for deterministic request errors;
- bounded retry for transient failures;
- explicit partial- and total-failure semantics;
- search accounting for one and multiple queries;
- no external provider calls in the default test suite;
- an updated preflight state and public capability ledger;
- a separate, explicitly authorized live-provider validation; and
- a separate frozen quality evaluation before any product-support claim.

## Related public sources

- `docs/ADR_DEEP_RESEARCH_EGRESS.md` (historical slice document; do not
  treat its slice-specific runtime statements as current truth)
- `docs/DEEP_RESEARCH_PREFLIGHT_DESIGN.md` (historical slice document; the
  decomposition architecture marker is still supported)
- `docs/DR_Q1A_BASELINE_CALIBRATION_PLAN.md`
- `docs/CAPABILITY_LEDGER.md`
- `hermes/jobs/service.py`
- `hermes/jobs/preflight.py`
- `hermes/services/search/protocol.py`
- `hermes/services/search/router.py`
- `hermes/services/search/tavily.py`
- `hermes/services/search/searxng.py`
- `hermes/services/search/exa.py`
- Tavily Search best practices:
  `https://docs.tavily.com/documentation/best-practices/best-practices-search`
- Tavily Search API:
  `https://docs.tavily.com/documentation/api-reference/endpoint/search`
- SearXNG Search API:
  `https://docs.searxng.org/dev/search_api.html`

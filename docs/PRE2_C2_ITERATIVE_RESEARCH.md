# PRE2-C2 bounded iterative Deep Research

PRE2-C2 adds an internal, provider-neutral coordination seam for evolving
the C1B one-wave foundation into a bounded multi-wave loop. It is component
implementation only: the production Deep Research service, public Research
Brief API, schema, deploy path, and provider configuration are unchanged.

## Authorized flow

```text
ResearchBrief
  -> SemanticPlanner
  -> validated DIRECT | DECOMPOSE SearchPlan
  -> SearchWave execution
  -> evidence and provenance
  -> GapAssessment
  -> next SearchWave or terminal STOP
```

The original Research Brief is supplied to the replaceable semantic planner
on every planning pass. The controller does not rewrite it and persists only
its SHA-256 binding plus bounded derived context. `DIRECT` means that the
semantic planner selected one query for that wave; it is not a lexical or
deterministic bypass around the planner.

The planner and assessor may propose. `ResearchController` deterministically
authorizes the proposal after local validation:

- every plan is structured, brief-bound, capability-bound, and provenance-
  bound;
- each wave contains 1..4 locally identified queries, each at most 399
  characters, with duplicate rejection;
- later planner requests carry prior source references, open gaps, and
  exhausted query/source identifiers;
- a repeated normalized query is rejected before another search call;
- continuation requires remaining gaps and an explicit material-gain
  proposal from the assessment boundary; a new URL is only an evidence
  frontier signal, not proof of semantic gain;
- `STOP_COVERED` requires no remaining gaps and material gain, while an
  explicit no-gain proposal deterministically stops with `NO_MATERIAL_GAIN`.

## Job bounds and STOP reasons

The internal hard ceiling is eight waves and 32 search calls. The configured
job limits also cap elapsed local time and a provider-independent call-unit
counter (planner + search + assessment). Evidence, gap context, source URLs,
and observations have explicit bounded sizes. Cooperative cancellation is a
persisted `CANCELLED` STOP; `asyncio.CancelledError`, `SystemExit`, and other
control-flow semantics are not swallowed by the one-wave executor.

The controller can stop with:

- `OBJECTIVE_COVERED` — the assessor found no remaining gaps with evidence;
- `BUDGET_EXHAUSTED` — waves, searches, elapsed time, or local call units are
  exhausted;
- `NO_MATERIAL_GAIN` — continuation would repeat the evidence frontier or the
  assessor explicitly reports no material gain; or
- `CANCELLED` — an owner/system cancellation probe was observed.

No empty-evidence result is promoted to success by the iteration layer. The
underlying wave executor retains its truthful `ALL_EMPTY`, `ALL_FAILED`, and
`PARTIAL_NO_EVIDENCE` outcomes.

## Checkpoints and recovery

`LocalIterationStateStore` writes versioned JSON atomically inside a confined
`research_iterations/` directory. Checkpoints are written at these seams:

1. `READY_TO_PLAN` before the planner call, with the planner dispatch counted;
2. `PLAN_PERSISTED` immediately after a validated plan and before search;
3. a durable in-flight query checkpoint before each search dispatch, followed by
   a bounded partial observation checkpoint after each completed query;
4. `ASSESSMENT_PENDING` after the complete wave evidence is recorded; and
5. terminal `STOPPED` after deterministic assessment or budget/cancellation.

If execution is interrupted after `PLAN_PERSISTED`, recovery reuses that
exact plan and does not call the planner again. Completed query observations
are checkpointed by ordinal. A call that was already dispatched but did not
produce an observation is never replayed silently because this provider-neutral
seam has no idempotency key; recovery converts that uncertain boundary into a
truthful `CANCELLED` STOP. Planner and assessor dispatches follow the same
rule. Terminal states are idempotent reads and make no further planner,
search, or assessment calls. Corrupt, mismatched, capability-drifted, or
brief-drifted state fails closed, including nested plan/observation fields and
phase/accounting equations. The assessor boundary is async and cancellable;
search results accept only concrete list/tuple containers, so an arbitrary
synchronous iterator cannot survive a job deadline as a background worker.

Each persisted source reference retains first-query provenance and is derived
from observed evidence; provenance must be complete, and wave outcome must
match the materialized observations. Each bounded evidence item includes a
sanitized title/snippet plus a digest tied to its source/query observation;
raw provider content beyond those caps is not persisted. Coverage requires
substantive bounded evidence, not merely a new URL or opaque digest.
Signed/query-secret URLs, query strings, and URL fragments are rejected before
checkpointing. Accounting counts planner,
search, and assessment dispatches before the call boundary and is explicitly
local-call truth; it is not provider billing or spending truth.

## Evidence and nonclaims

The deterministic tests use fakes and prove multiple waves, DIRECT versus
DECOMPOSE persistence, multilingual-safe planner input, gap continuation,
all STOP paths, partial-query cancellation/recovery, provenance, accounting,
brief privacy, bounded malformed iterables, and strict corrupt-state
rejection. They do not prove semantic planner
quality, search quality, live provider behavior, or multi-wave product
runtime wiring. This slice introduces no public API expansion, no database
schema migration, no deploy/release, no credentials, no PayGo/spending, and
no production schema or service composition change.

# Deep Research preflight design

This document defines the contract for Slice 1. No endpoint or command
implements it at the Slice 0 baseline.

## Purpose

Preflight answers two different questions without exposing sensitive values:

1. Is the selected Deep Research mode configured and locally safe?
2. When explicitly requested, can its external dependencies be reached and
   authorized?

Preflight does not submit a research query, fetch an arbitrary source, or send
user content to an LLM.

## Contract

```json
{
  "schema_version": 1,
  "status": "blocked",
  "mode": "offline",
  "checks": [
    {
      "code": "dr.runtime.service_wiring",
      "state": "fail",
      "required": true,
      "message": "Deep Research runtime service is not available.",
      "remediation": "Enable the reviewed Deep Research runtime integration."
    }
  ],
  "limits": [
    {
      "name": "final_model_output",
      "classification": "estimate",
      "configured": true
    }
  ]
}
```

Fields:

- `schema_version`: integer contract version.
- `status`: `ready`, `degraded`, `blocked`, or `disabled`.
- `mode`: `offline` or `live`.
- `checks`: deterministic, ordered check results.
- `limits`: redacted limit classification.

Each check contains:

- `code`: stable machine-readable identifier;
- `state`: `pass`, `warn`, `fail`, or `skip`;
- `required`: whether failure blocks the selected mode;
- `message`: safe human summary;
- `remediation`: safe next action.

Provider names may be returned when they are part of the public configuration
contract. Keys, values, headers, raw URLs, queries, internal addresses, resolved
addresses, and exception text are forbidden.

## Overall status

- `disabled`: the operator has not opted into Deep Research. This state takes
  precedence over all readiness checks; required runtime and provider checks
  are reported as `skip` because no supported execution was requested.
- `blocked`: a required check fails or its state is unknown.
- `degraded`: every required check passes, while an optional dependency is
  unavailable or a non-blocking warning exists.
- `ready`: every required check for the selected mode passes.

Offline `ready` means **configured and locally policy-valid**, not network
reachable. Live `ready` additionally requires explicit reachability and, where
the provider supports a safe authenticated probe, authorization.

## Stable diagnostic codes

| Code | Mode | Required | Meaning at the Slice 0 baseline |
| --- | --- | ---: | --- |
| `dr.feature.opt_in` | offline | yes | Operator opt-in state |
| `dr.runtime.service_wiring` | offline | yes | Production service and scheduler registration; currently absent |
| `dr.runtime.recovery_wiring` | offline | yes | Startup recovery registration; currently absent |
| `dr.search.backend_configured` | offline | yes | Selected backend configuration is present without revealing its value |
| `dr.search.backend_reachable` | live | yes | Explicit backend reachability probe |
| `dr.search.backend_authorized` | live | conditional | Required only when the provider offers a safe authenticated probe; otherwise `skip` is readiness-neutral and the result cannot claim authorization |
| `dr.fetch.policy_available` | offline | yes | Reviewed request-level fetch policy exists; currently absent |
| `dr.fetch.external_enabled` | offline | yes | External fetching is enabled only after policy acceptance |
| `dr.llm.provider_configured` | offline | yes | Selected research provider is configured |
| `dr.llm.provider_reachable` | live | yes | Explicit provider reachability probe |
| `dr.llm.provider_authorized` | live | conditional | Required only when the provider offers a safe authenticated probe; otherwise `skip` is readiness-neutral and the result cannot claim authorization |
| `dr.report.retrieval_available` | offline | yes | Authenticated report-content retrieval exists; currently absent |
| `dr.egress.firewall_enabled` | offline | no | Optional container defense-in-depth state |
| `dr.limits.model_output_enforced` | offline | yes | Phase 3 and 4 calls enforce model-output caps; currently false |
| `dr.limits.job_budget_enforced` | offline | no | Reports whether per-job budget is a hard stop or a soft warning |
| `dr.architecture.query_decomposition` | offline | no | Reports `not_implemented`; it is not a readiness requirement for Slice 1 |

Codes describe one condition each. Implementation may add codes without
changing `schema_version`; changing the meaning or shape of an existing code
requires a versioned contract change.

A conditional check is evaluated as required only when its provider declares
that capability. A conditional `skip` is readiness-neutral, but preflight must
then describe the capability as unverified rather than authorized.

## Offline mode

Offline mode is the default for local diagnostics and CI. It:

- reads validated settings without returning their values;
- checks runtime registration;
- validates the fetch-policy configuration;
- classifies limits as `hard`, `admission_control`, `soft_warning`, or
  `estimate`;
- verifies report-retrieval registration;
- marks every live check `skip`.

It performs no DNS lookup, HTTP request, search, arbitrary fetch, or LLM call.

## Live mode

Live mode requires an explicit confirmation such as
`--mode live --confirm-egress`. The caller must be locally privileged or use
the same authenticated operator boundary selected for other diagnostics.

Live checks:

- use fixed, provider-safe probe content rather than a user query;
- have strict time and cost bounds;
- do not follow arbitrary redirects;
- distinguish reachability from authorization;
- redact exception details before producing the contract;
- never fetch a user-supplied source URL as a health check.

## Limit classification

| Classification | Meaning |
| --- | --- |
| `hard` | Runtime rejects or stops work at the stated boundary |
| `admission_control` | Checked before a job starts; later work may still exceed the estimate |
| `soft_warning` | Records or reports an overage but does not stop work |
| `estimate` | Used for planning or cost prediction only |

Each limit entry returns its name, classification, whether it is configured,
and a public-safe unit. Values may be omitted when revealing them would expose
deployment policy.

## Slice 1 implementation order

1. Add the typed contract, offline evaluator, and deterministic unit tests.
2. Implement and adversarially test the shared safe external fetcher described
   by `ADR_DEEP_RESEARCH_EGRESS.md`.
3. Wire the fetcher into phase 2 behind explicit opt-in; keep external fetching
   disabled by default until the safety gate passes.
4. Pass enforced model-output limits to phase 3 and phase 4 LLM calls; classify
   daily and per-job budgets accurately.
5. Add production service, scheduler, and recovery wiring behind the preflight
   gate.
6. Add authenticated, owner-scoped report-content retrieval with path
   confinement and retention behavior.
7. Add bounded live probes and document create, status, cancel, retry, and
   report retrieval.
8. Run a clean supported vertical-slice smoke and independent R1 review.

## Acceptance evidence for Slice 1

| Area | Required evidence |
| --- | --- |
| Contract | Schema and diagnostic-code tests; stable ordering; version behavior |
| Privacy | Redaction tests for keys, headers, URLs, addresses, and exceptions |
| Offline boundary | Tests proving zero DNS, HTTP, search, fetch, and LLM calls |
| Fetch safety | Initial URL and redirect validation, IPv4/IPv6 special-use rejection, rebinding or proxy control, timeout and streamed-size enforcement |
| Limits | Tests proving model-output arguments reach both LLM call sites and that every published limit has the correct classification |
| Runtime wiring | Real application startup test, service availability, scheduler lifecycle, and recovery |
| Ownership | Cross-user report retrieval returns no existence information |
| Job journey | Create, observe, cancel, retry, complete, and retrieve a cited report |
| Live probes | Explicit opt-in, bounded cost/time, no user content, reachability separated from authorization |

Query decomposition, additional providers, UI work, broad autonomous loops, and
schema migrations are non-goals for Slice 1.

# ADR: explicit Deep Research egress

- Status: Proposed
- Date: 2026-07-19
- Baseline: `e5602ef4aa88f3fffc13c3e97d11a42bf3df98b5`
- Scope: design only; Slice 0 changes no runtime behavior

## Context

Deep Research crosses three independent egress boundaries:

1. A search backend receives the research query.
2. The application fetches URLs returned by search.
3. An LLM provider receives source text and synthesis prompts.

The current search router rejects non-HTTP schemes, but the subsequent fetch
uses an HTTP client with redirects enabled and no request-level policy for
private or special-use addresses, redirect targets, DNS rebinding, inherited
proxy configuration, or streamed response size. The optional container egress
firewall is defense in depth; it is disabled by default and cannot substitute
for per-request URL authorization.

Deep Research must therefore remain outside the supported runtime path until a
safe fetch boundary exists.

## Decision

### 1. External source fetch is denied until policy is ready

The supported Deep Research path will keep external source fetching disabled by
default until a shared `SafeExternalFetcher` and its adversarial tests are
accepted. Configuration presence alone must not change this state.

### 2. Every egress class is explicit

Operator-facing documentation and preflight diagnostics will identify:

- the selected search backend and what query content leaves the system;
- the selected LLM provider and what source or prompt content leaves the
  system;
- whether external source fetching is disabled or protected by the accepted
  fetch policy;
- whether the optional container firewall is enabled as defense in depth.

Sensitive values, request headers, raw queries, source URLs, internal addresses,
resolved addresses, and provider exception text must never appear in a public
diagnostic response.

### 3. Request-level fetch authorization

The accepted fetcher must:

- allow only `http` and `https`;
- reject URL user information;
- reject unapproved ports;
- resolve both A and AAAA records;
- reject loopback, private, link-local, multicast, unspecified, reserved, and
  other special-use destinations;
- validate the initial destination and every redirect target;
- bound redirect hops;
- prevent DNS rebinding by connecting only through a validated and pinned
  resolution, or by using an equivalently reviewed network proxy;
- preserve correct TLS SNI and HTTP `Host` behavior;
- disable inherited proxy configuration unless an explicitly reviewed proxy is
  part of the deployment contract;
- stream response bytes and stop at a hard limit before buffering or decoding;
- bound connect, read, write, pool, and total operation time;
- return stable error codes without reflecting unsafe input.

The same policy applies to retries. A previously accepted URL is not an
authorization for a later resolution or redirect.

### 4. Container firewall remains defense in depth

`hermes/security/egress.py` may restrict known provider destinations, but it
does not authorize arbitrary source URLs. Its status may be included in
privileged diagnostics as a boolean and policy-validity result. Diagnostics
must not expose configured domains, addresses, or resolved rules.

### 5. Offline and live checks are distinct

Default and CI preflight is offline. It validates configuration shape and local
policy without making network requests.

Live checks require an explicit operator action. Before execution they state
which provider class will be contacted. A reachability result is not
automatically an authorization result: for example, a health check that treats
an HTTP authentication error as proof of service reachability cannot prove that
the configured credential works.

### 6. Limits are classified by enforcement

Diagnostics and documentation distinguish:

- admission-cost estimates;
- daily admission control;
- per-job soft warning or hard stop;
- hard LLM model-output cap;
- hard fetched-byte cap;
- source-count and per-source text caps;
- redirect, concurrency, phase-timeout, retry, total wall-clock, and persisted
  output limits.

At the pinned baseline, the configured per-source and final model-output values
are used for cost estimation but are not passed to the Deep Research LLM calls.
They must not be described as hard runtime caps. The existing per-job budget
check is also not a hard cancellation boundary.

## Consequences

- Slice 1 begins with safety and diagnostics rather than UI or broad feature
  enablement.
- A configured search credential is insufficient for `ready` status while safe
  source fetching is unavailable.
- Live tests remain opt-in and cannot be default CI evidence.
- Query decomposition is deferred. The implemented phase 1 currently sends a
  single query despite older architecture wording.

## Verification

Acceptance requires unit and integration tests covering the initial URL, each
redirect, IPv4 and IPv6 special-use ranges, rebinding or proxy enforcement,
timeouts, streamed byte limits, redaction, and stable diagnostic codes. A live
test is supplementary evidence, not a substitute for these deterministic
checks.

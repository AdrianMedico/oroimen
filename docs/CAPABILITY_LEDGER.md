# Public capability ledger

This ledger describes the public repository at commit
`b95afb4943a855eb0cc4fdd911218bbf0d6087b6` (merge of Slice 1C3
deterministic vertical E2E). It separates code presence from runtime
availability, runtime availability from deterministic integration, and
deterministic integration from live-provider behavior and from
research-quality measurement. A module is not considered a supported
feature merely because its implementation and component tests exist.

## Status vocabulary

- **Supported**: wired into the documented startup path, covered by its
  deterministic integration test, and the owner has approved it for
  the slice's evaluable behavior.
- **Implemented, runtime unavailable**: components exist, but production
  startup does not construct or register the required service. The
  component is exercised by deterministic tests only.
- **Implemented, runtime available, quality unmeasured**: the
  deterministic vertical integration is proven, but the slice has
  not yet been calibrated against a frozen benchmark. Absence of a
  measurement is not equivalent to a measurement of zero.
- **Optional**: wired only when an operator supplies explicit configuration.
- **Deferred**: intentionally outside the supported evaluator path.
- **Absent**: no implementation, no test, no documentation claim.
- **Design only**: documented for a later slice; no runtime behavior exists.

The vocabulary above replaces the Slice 0 status vocabulary, which
collapsed "runtime" and "live provider" into a single
"runtime unavailable" state. Slice 1C2 and Slice 1C3 split those
states apart.

## Critical truth distinctions (do not collapse)

1. **Code exists.** The component file is present in the repository.
2. **Runtime composition exists.** Production startup constructs the
   component, registers it as a singleton or service, and reaches a
   state where the documented HTTP contract returns the documented
   response shape.
3. **Deterministic vertical integration is proven.** An end-to-end
   golden-journey test exercises the real production code path through
   authenticated HTTP, real DB, real phase pipeline, real atomic
   writer, real reader, and real DTO mapping. External seams (search,
   fetch, LLM, notifier, scheduler trigger) are replaced by
   deterministic fakes. No network calls. No provider spending.
4. **Live provider behavior is proven.** The real external
   dependencies (Tavily, cloud LLM, Telegram transport, etc.) have
   been exercised end-to-end with provider credentials in an
   authorized, owner-bounded live run.
5. **Research quality is measured.** A frozen benchmark has been
   executed against the production pipeline, and structured human
   review has been applied to the resulting reports. Quality has not
   been benchmarked unless a published owner-approved protocol has
   been run.
6. **Product support is documented.** Public README, ADRs, and
   evaluator paths describe this slice as a supported product path.

The current Oroimen state at `b95afb4` is at level 3
(Deterministic vertical integration is proven) for the deep-research
report-retrieval vertical. It is at level 1 (Code exists) and level 2
(Runtime composition exists) for the deep-research runtime pipeline
behind opt-in. It is at level 0 for live-provider behavior and for
research-quality measurement. Slice 1C3 does not raise the level
beyond 3; it proves the deterministic vertical only.

## Ledger

| Capability | Public evidence | Runtime and support status | Disposition | Main risks or dependencies | Acceptance or demo criterion | Priority |
| --- | --- | --- | --- | --- | --- | --- |
| Local chat and health path | `README.md`; `hermes/receivers/http_api.py`; `tests/unit/test_http_api.py`; `tests/e2e/test_phase1_smoke.py` | Supported | Retain and regression-test | Local model and container readiness | Follow the README quickstart and complete health plus first-chat checks | P0 |
| File ingestion and grounded retrieval | `hermes/memory/drop_watcher.py`; `hermes/memory/ingest_router.py`; `hermes/memory/vault.py`; `tests/integration/test_files_e2e.py`; `tests/e2e/test_sprint_19_pipeline.py` | Supported | Retain and regression-test | Extractor and embedding availability; untrusted document content | Ingest a public fixture and retrieve grounded content through the documented path | P0 |
| Prompt-injection classification and tool bounds | `hermes/security/classifier.py`; `hermes/tools/security.py`; `hermes/agent/loop.py`; `tests/e2e/test_rag_injection.py`; `tests/unit/test_agent_loop.py` | Supported | Retain and regression-test | Model-dependent behavior remains separately evaluated | Deterministic security tests pass; live-model evidence is never inferred from skipped tests | P0 |
| Deep Research service components (orchestration, scheduler, recovery) | `hermes/jobs/service.py`; `hermes/jobs/scheduler.py`; `hermes/jobs/recovery.py`; `hermes/jobs/models.py`; `hermes/memory/db.py`; `hermes/__main__.py` (composition root) | Implemented, runtime available behind explicit opt-in (`HERMES_DEEP_RESEARCH_ENABLED=true`); the composition root constructs the real `LocalReportStore` inside the outer `try:` and publishes the singleton only after the report store, scheduler, and recovery succeed | Retain and regression-test | Disabled-by-default posture is preserved; runtime wiring is fail-closed (a failure anywhere in the outer `try:` runs `rollback()` and returns `(unavailable, None, None)`; the singleton is never published with a missing report store); a failed `LocalReportStore` construction propagates and the route returns 503 (defensive) or 500 (unavailable) rather than publishing a half-built service | The composition root's `DeepResearchCapabilities` is published with `service_wiring=True`, `recovery_wiring=True`, `search_backend_configured=True`, `llm_provider_configured=True`, `fetch_policy=True`, `external_fetch=True`, `model_output_enforced=True`, and `report_retrieval=True` only when the full wired path succeeds | P0 |
| Deep Research jobs API contract | `hermes/receivers/jobs_api.py`; `hermes/jobs/models.py`; `tests/integration/test_jobs_api.py` | Supported behind the same opt-in: with a wired service singleton the routes return the documented contract (200 detail, 200 report, 200 list, 200 budget, 200 preflight, 200/409 cancel, 201/409 retry); without a singleton the routes return 503 `service_unavailable` | Retain and regression-test | A defensive 503 branch protects against a service that was wired but lacks `_report_store`; this branch is a safety net, not a normal path | A real application instance serves create, detail, list, budget, preflight, cancel, and retry without test-only dependency injection | P0 |
| Deep Research preflight | `hermes/jobs/preflight.py`; `hermes/receivers/jobs_api.py` (route); `tests/unit/test_jobs_preflight.py`; `tests/integration/test_jobs_api.py` (`test_preflight_*`) | Supported (offline) | Retain and regression-test | The `evaluate_deep_research_preflight` evaluator is the single source of truth; the route wires the `DeepResearchCapabilities` from the composition root, NOT a patched response | `GET /v1/jobs/preflight` returns the contract from the real evaluator with `status` derived from the wired `capabilities` object; live checks remain `skip` in offline mode | P0 |
| Search routing | `hermes/services/search/`; `hermes/tools/web_search.py`; `hermes/__main__.py`; `tests/unit/test_search_router.py`; `tests/unit/test_search_tavily.py` | Optional; Deep Research intent routes to Tavily when configured | Retain as opt-in and expose redacted readiness | Cloud credentials, provider budget, network egress, and backend health semantics | Offline preflight reports configuration only; explicit live checks prove reachability without sending a user query | P0 |
| Safe external fetch for Deep Research | `hermes/jobs/safe_fetcher.py`; `hermes/jobs/service.py` (phase 2 uses `_fetcher.fetch(url)`); `tests/integration/test_safe_external_fetcher_transport.py`; `tests/integration/test_jobs_service_phases.py` | Implemented, runtime available; the production service uses `SafeExternalFetcher` exclusively for phase 2; no direct `httpx.AsyncClient` or fallback transport exists in the service | Retain and regression-test | Redirect SSRF, private and special-use addresses, IPv6, DNS rebinding, proxy inheritance, and fully buffered oversized responses are covered by adversarial unit tests | Adversarial tests cover the initial URL, every redirect hop, A and AAAA results, proxy behavior, and streamed byte limits | P0 |
| Deep Research model-output limits | `hermes/jobs/service.py` (per-source and final synthesis pass `max_tokens` to `llm.chat`); `hermes/jobs/cost.py`; `tests/integration/test_jobs_service_phases.py`; `tests/integration/test_jobs_cost_drift.py` | Implemented, runtime available; phase 3 (per-source) calls `llm.chat(..., max_tokens=settings.deep_research_per_source_max_tokens)` and phase 4 (final synthesis) calls `llm.chat(..., max_tokens=settings.deep_research_output_max_tokens)`; the composition root sets `model_output_enforced=True` only after this wiring is in place | Retain and regression-test | The two settings must reach both LLM call sites; cost drift between checkpoint, token-usage sum, and DB aggregate is reconciled via `reconcile_cost` | Per-source `max_tokens` matches `settings.deep_research_per_source_max_tokens`; final `max_tokens` matches `settings.deep_research_output_max_tokens` | P0 |
| Daily budget admission control | `hermes/jobs/service.py`; `hermes/config.py`; `tests/integration/test_jobs_cost_drift.py` | Implemented, runtime available; pre-check on submit + atomic TOCTOU check inside the running transaction | Retain and regression-test | The daily cap is checked before enqueue and re-checked at job start (atomic) to prevent TOCTOU | A job submitted and run after the cap is reached transitions to `failed` with `error_taxonomy="budget_exceeded"`; it is NOT silently enqueued | P0 |
| Per-job monetary budget enforcement | `hermes/jobs/service.py` (per-job `cost` recorded via `_record_token_usage` + `reconcile_cost`); `hermes/jobs/cost.py` | Implemented, runtime available, but **soft warning only** | Retain as soft warning; a hard per-job cancellation boundary is NOT yet implemented | The service records the per-job cost and surfaces it through the notifier and the report; it does NOT cancel the job mid-flight when the per-job budget is exceeded | The per-job cost is exposed in `JobDetail.cost_usd` and in the notifier call; the per-job budget is documented as a soft warning, not a hard cancellation | P0 |
| Deep Research report persistence | `hermes/jobs/service.py` (`_phase_write` uses `tmp + flush + os.fsync + os.replace`); `hermes/jobs/models.py` (DTOs without `output_path` / `partial_output_path` / `checkpoint_path` / `report_available`); `hermes/jobs/report_paths.py`; `hermes/jobs/report_store.py`; `hermes/receivers/jobs_api.py` (route); `tests/integration/test_jobs_api.py`; `tests/unit/test_report_paths.py`; `tests/unit/test_report_store.py`; `tests/e2e/test_deep_research_vertical.py` | Supported (Slice 1C2 + 1C3): owner-scoped `GET /v1/jobs/{job_id}/report` returns the final Markdown through the real `LocalReportStore`; the public DTOs no longer carry filesystem paths; the notifier template no longer embeds a path; the deterministic vertical golden journey proves the full owner-scoped flow through real HTTP | Retain and regression-test | A failed report-store construction is fail-closed (the singleton is not published and the route returns 500 `report_unavailable` or 503 `service_unavailable` defensively); the read path uses `report_store.derive_path(job_id)` and never reads from the DB `output_path` column; the writer and the reader share the same resolved absolute Path through `service._data_root = report_store.root` (round 2 wiring) | The owner retrieves report content by job ID with a constant `text/markdown; charset=utf-8` body, the documented headers, and a deterministic body shape; missing and foreign-owned jobs return byte-identical 404 `job_not_found`; complete status with a missing/oversize/invalid-UTF-8 file returns 500 `report_unavailable` with no internal details leaked | P0 |
| Deep Research report settings: `deep_research_data_root` and `deep_research_max_report_bytes` | `hermes/config.py`; `hermes/jobs/report_store.py`; `hermes/jobs/preflight.py`; `hermes/jobs/service.py` (writer uses canonical store root); `hermes/__main__.py` (composition root) | Supported (Slice 1C2): the report-store root is resolved at startup and the max-bytes cap is enforced inside the bounded read of the single opened handle (round 2); the composition root resolves `data_root` and the service uses `report_store.root` as `service._data_root` (round 2) | Retain and regression-test | The two settings must reach both the composition root and the read path; the max-bytes floor (10 KiB) and ceiling (50 MiB) are validated at construction | An oversize report is rejected with 500 `report_unavailable` (logs may tag `report_size_limit_exceeded`); a missing or non-creatable data root fails closed at startup; the writer and the reader share the same canonical Path | P0 |
| Deterministic Deep Research vertical E2E | `tests/e2e/test_deep_research_vertical.py` (NEW, 1 file, 831 lines, merged in Slice 1C3) | Implemented, runtime available, quality unmeasured | Retain and regression-test; a frozen benchmark execution is a separate future slice (see `docs/DR_Q1A_BASELINE_CALIBRATION_PLAN.md`) | The golden journey drives the full 9-step product path through authenticated HTTP: preflight, create, real 5-phase pipeline, atomic write, owner-scoped detail, owner-scoped report, notifier privacy, owner isolation, and lifecycle cleanup. Only the external seams (search, fetch, LLM, notifier, scheduler trigger) are faked; everything else is real | The 4 tests pass on every run, in 3 consecutive runs, and in 3 in-process cycles; the test carries `@pytest.mark.e2e` and no `slow` / `network` markers; the LLM mock uses a modulo-wrap `side_effect` counter for determinism | P0 |
| Memory collections and embeddings | `hermes/memory/collections.py`; `hermes/services/embedding_router.py`; `hermes/services/embed_vault.py`; corresponding unit and integration tests | Present; supported behavior depends on the selected local or optional provider | Retain; document provider readiness separately | Model availability and resource limits | Public fixture ingestion and retrieval pass without cloud credentials on the default path | P1 |
| OCR routing and edge coordination | `hermes/memory/ocr_decision.py`; `hermes/memory/edge_coordinator.py`; `hermes/receivers/ocr_api.py`; OCR unit tests | Present; advanced edge paths are deployment-dependent | Retain public generic behavior; keep deployment assumptions out of public docs | External binaries, local vision model, and edge lifecycle | Public OCR decision tests pass; deployment-specific readiness is not claimed | P2 |
| LLM provider cascade and streaming | `hermes/llm/router.py`; `hermes/llm/ollama.py`; `hermes/llm/chatgpt5_6.py`; provider and streaming tests | Local provider is the supported default; cloud and frontier providers are opt-in | Retain explicit selection and fallback semantics | Credentials, cost, data egress, provider availability | Offline diagnostics list provider modes without values; live probes are explicit and bounded | P1 |
| Container egress firewall | `hermes/security/egress.py`; `tests/unit/security/test_egress.py` | Optional and disabled by default | Retain as defense in depth, not as request-level URL authorization | DNS is resolved when rules are applied; stale DNS and request-level SSRF remain separate concerns | Diagnostics report only enabled state and policy validity, never addresses or sensitive configuration | P1 |
| Deep Research iterative retrieval | absent | **Absent**: no multi-pass planning, no reflection step, no re-query with refined terms, no stopping decision | Deferred | An iterative loop would require a new boundary in the service, new preflight codes, and a benchmark to prove it improves quality; none of these is justified at the current baseline | Not supported as a product path | P2 |
| Deep Research query decomposition | absent | **Absent**: the service calls a single search query per job; no static or learned decomposition of complex queries into sub-questions | Deferred | A decomposition layer would require a new phase, new prompts, and a benchmark to prove it improves quality; none of these is justified at the current baseline | Not supported as a product path | P2 |
| Deep Research claim-level provenance and citation verification | absent | **Absent**: the service does not extract individual claims, does not verify citation support, and does not produce a claim ledger | Deferred | A claim parser + verifier would require a new phase, a new boundary, and a benchmark to prove it improves quality; none of these is justified at the current baseline | Not supported as a product path | P2 |
| Deep Research contradiction handling | absent | **Absent**: the service does not explicitly detect or surface contradictions between retrieved sources | Deferred | A contradiction-handling phase would require a new boundary, new prompts, and a benchmark; none of these is justified at the current baseline | Not supported as a product path | P2 |
| Deep Research quality benchmark | absent | **Absent (UNMEASURED)**: no frozen benchmark, no rubric, no run manifest, no human audit procedure has been published; the existing deterministic E2E is a smoke, not a quality measurement | Calibration slice planned (see `docs/DR_Q1A_BASELINE_CALIBRATION_PLAN.md`) | A real benchmark needs an owner-approved corpus, rubric, and reviewer workflow; the calibration plan exists but the benchmark has not been executed | A future measurement may show that the existing pipeline meets the bar; the existing pipeline MUST NOT be modified in response to LLM recommendations until a measurement is published | P0 |

## Sanitized disposition policy

1. Treat this public commit as the evidence source. Do not publish
   private deployment files, environment files, agent logs, personal
   data, credentials, or raw historical planning material.
2. If public implementation already exists, verify and productize it.
   Do not recopy code from another repository.
3. Distill still-relevant historical design into a current public
   decision record. Do not publish raw history.
4. Migrate generic tests or security tooling only after a fresh
   public-value and sanitization review.
5. Keep infrastructure topology, operator-specific telemetry, and
   deployment assumptions outside the public repository.
6. Require a public evidence path for every capability claim. Mark
   uncertainty instead of inferring runtime support from component
   presence.
7. Do not describe a deterministic integration test as live-provider
   proof. The deterministic integration test uses scripted fakes at
   the external seams and proves the local product path; it does not
   prove the real external dependencies.
8. Do not describe code presence as runtime availability. The Slice 0
   ledger collapsed these states; the current ledger separates them
   and adds a third state (runtime available, quality unmeasured)
   for components that are wired and exercised by deterministic tests
   but have not been calibrated against a frozen benchmark.
9. Do not use marketing language ("production ready", "fully
   sovereign", "frontier quality", "competitive with Perplexity") in
   the ledger. Each claim must be supportable from the public
   commit.
10. The Deep Research quality has not been benchmarked. Do not
    present the absence of a benchmark as the absence of a quality
    problem. The opposite is also false: do not present the absence
    of a benchmark as evidence of zero defects. Calibration is
    the next step, not a claim.

## Current deterministic proof (Slice 1C3, merged at b95afb4)

The deterministic Deep Research vertical journey exercises the
production code path through authenticated HTTP, in the same process
the public application would use, with the following steps proven
end-to-end:

1. Authenticated `GET /v1/jobs/preflight` returns 200 `ready` with
   the `dr.report.retrieval_available` gate `pass`. The response
   body is produced by the real
   `evaluate_deep_research_preflight` evaluator, not a patched
   constant.
2. Authenticated `POST /v1/jobs` returns 201 with a syntactically
   valid 12-hex lowercase UUID. The response body exposes no
   filesystem path. A DB row is created for the authenticated
   owner. The scheduler receives exactly one enqueue for the
   created job.
3. The real `DeepResearchService._run_research(job_id)` is driven
   in-process. The state machine transitions
   `pending -> running -> complete` through real SQL UPDATEs.
   Per-job deterministic token and cost accounting is asserted
   (2 per-source calls + 1 final synthesis call, each with fixed
   input and output token counts).
4. The atomic report writer (`tmp + fsync + os.replace`) writes
   the final Markdown to the canonical path
   `<root>/<job_id>.md` and leaves no `.md.tmp` residue. The
   writer and the reader share the same resolved absolute Path
   through the round-2 wiring `service._data_root =
   report_store.root`.
5. Authenticated `GET /v1/jobs/{job_id}` returns 200 with
   `status=complete` and omits the filesystem-path fields
   (`output_path`, `partial_output_path`, `checkpoint_path`,
   `report_available`).
6. Authenticated `GET /v1/jobs/{job_id}/report` returns 200
   with the exact documented headers
   (`Content-Type: text/markdown; charset=utf-8`,
   `Content-Disposition: inline; filename="research-{job_id}.md"`,
   `Cache-Control: private, no-store`,
   `X-Content-Type-Options: nosniff`) and the deterministic final
   Markdown body produced by the fake LLM through the real
   service pipeline.
7. The notifier receives exactly one call to
   `send_research_complete(job_id, cost_usd)`. No
   `output_path`, no `report_ref`, no filesystem path, and no
   `/v1/` URL is passed in any argument.
8. A foreign-owned job and a missing job return byte-identical
   404 `job_not_found` envelopes. The foreign 404 does not echo
   the requested job_id.
9. The fixture's teardown clears the singleton, calls
   `service.aclose()` to drain the scrape pool's
   `ThreadPoolExecutor`, and stops the fake scheduler. No
   background thread survives the test.

This proof is **deterministic**, **offline**, and **scope-bounded**
to the existing product path. It is not a quality measurement.
It is not a live-provider test. It is not a deployment test.

## Current unmeasured areas (do not collapse with absent)

The following areas are not in the deterministic proof and have not
been benchmarked. Each is honest about what is known and what is
not.

- **Live Tavily behavior.** The deterministic journey uses scripted
  HTTPS URLs. The real Tavily search has not been exercised
  end-to-end with provider credentials during Slice 1C3.
- **Live cloud LLM behavior.** The deterministic journey uses
  scripted `LLMResponse` objects. The real cloud LLM has not been
  exercised end-to-end with provider credentials during Slice 1C3.
- **Research quality.** No frozen benchmark has been executed
  against the production pipeline. The 4-test deterministic suite
  is a smoke, not a quality measurement. The absence of a
  benchmark is the absence of a measurement, not the absence of
  defects.
- **Telegram notifier transport.** The deterministic journey
  records the call but does not exercise the real Telegram
  transport.
- **Per-job monetary budget hard cancellation.** The per-job
  budget is recorded and surfaced but does NOT cancel the job
  mid-flight when exceeded. The hard cancellation boundary is
  not yet implemented. The daily budget IS enforced (it is
  checked before enqueue and re-checked at job start); the
  per-job budget is a soft warning.

## Slice 0 smoke baseline (superseded)

The Slice 0 smoke baseline was:

- Commit: `e5602ef4aa88f3fffc13c3e97d11a42bf3df98b5`
- Platform: Windows
- Python reported by pytest: 3.14.3
- Date: 2026-07-19

Deterministic unit gate:

```text
uv run pytest tests/unit -m "not slow and not network" -n 4 --tb=short -q
1766 passed, 2 skipped, 1 xfailed, 30 warnings in 94.11s
```

Focused Deep Research and search component gate:

```text
uv run pytest tests/integration/test_jobs_api.py \
  tests/integration/test_jobs_service_phases.py \
  tests/integration/test_jobs_recovery_real.py \
  tests/integration/test_jobs_cost_drift.py \
  tests/unit/test_jobs_cost_estimate.py \
  tests/unit/test_search_router.py \
  tests/unit/test_search_tavily.py \
  -m "not slow and not network" -n 4 --tb=short -q
96 passed in 10.03s
```

These results prove the Slice 0 deterministic unit and component
baselines. They do **not** prove a Deep Research product journey: the
production startup did not register the service, the component tests
use fakes or mocks where appropriate, and live search and LLM egress
were not exercised. The current ledger supersedes the Slice 0
status language; the Slice 0 numbers above are preserved only as a
historical reference for the unit and component baselines that
underpin the current slice work.

## Approved shortlist (status as of b95afb4)

Slice 1 has produced a deterministic vertical for the deep-research
report-retrieval path. The next authorized step is calibration, not
a new production feature. Calibration is a measurement, not a
modification.

1. **Calibration (DR-Q1A, planned).** Execute a frozen benchmark
   against the current production pipeline. Use the protocol
   described in `docs/DR_Q1A_BASELINE_CALIBRATION_PLAN.md`. Do not
   modify the pipeline during calibration. Identify the dominant
   observed failure (if any) and propose a follow-up experiment
   only after the measurement is published.
2. **Future experiments, not yet authorized.** Possible follow-up
   experiments after calibration include:

   - claim extraction and citation-verifier experiment (if
     citation support / completeness is the dominant failure);
   - static query-decomposition experiment (if multi-branch
     coverage is the dominant failure);
   - source deduplication and authority-policy experiment (if
     duplicated or weak sources is the dominant failure);
   - depth-curve and stopping experiment (if excessive retrieval
     without quality gain is the dominant failure);
   - contradiction-handling experiment (if silently merged
     contradictions is the dominant failure);
   - **no change** (if no dominant failure is observed).

   None of these experiments is approved in this slice. None of
   them is implemented in this slice. Each will require its own
   owner authorization when the calibration result is published.

Query decomposition, iterative retrieval, evidence ledger, claim
parser, NLI verifier, LLM-as-judge automation, modularization,
worker or process separation, and broad autonomous research
behavior remain deferred until the calibration result is published
and the dominant failure (if any) is identified.

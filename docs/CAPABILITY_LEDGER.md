# Public capability ledger

This ledger describes the public repository at commit
`0e29dd6f38372ed24e060e1649d8f8456c49adbc` (post-PR #8; the most
recent merged slice is Slice 1C3 deterministic vertical E2E, and
the current baseline reflects the DR-Q1A protocol merge). It
separates code presence from runtime availability, runtime
availability from deterministic integration, and deterministic
integration from live-provider behavior and from research-quality
measurement. A module is not considered a supported feature merely
because its implementation and component tests exist.

## Status vocabulary

Every ledger row uses exactly one of the eight statuses below. No
row uses a status not in this list. The Slice 0 vocabulary
collapsed "runtime" and "live provider" into a single state; the
post-1C3 vocabulary separates them.

- **Supported** — wired into the documented default startup path,
  covered by its deterministic integration test, and the owner has
  approved it for the slice's evaluable behavior. Enabled by
  default. No operator opt-in is required to use the documented
  behavior.
- **Implemented, runtime available behind opt-in** — components
  exist and the composition root will construct and register them
  ONLY when the operator supplies explicit configuration (for
  example `HERMES_DEEP_RESEARCH_ENABLED=true`). Without the
  opt-in, the service is not constructed and the route returns a
  defensive 503 or 500. The deterministic vertical integration
  test exercises the wired path; the live product surface is
  gated by the configuration flag.
- **Implemented, runtime available, quality unmeasured** — the
  deterministic vertical integration is proven (real HTTP, real
  DB, real phase pipeline, real atomic writer, real reader, real
  DTO mapping) and the wired path is exercised by the
  deterministic E2E, but the slice has not been calibrated
  against a frozen benchmark. No live Tavily, no live cloud LLM,
  no benchmark has been run. Absence of a measurement is not
  equivalent to a measurement of zero.
- **Implemented, runtime unavailable** — components exist, but
  production startup does not construct or register the required
  service. The component is exercised by deterministic unit or
  integration tests only and is NOT part of the live product
  path. The route is defensive (returns 503 or 404) and the
  service is never wired in production.
- **Optional** — wired only when an operator supplies explicit
  configuration; the default path does NOT include it. Distinct
  from "behind opt-in" in that the operator enables a feature
  flag (for example `use_cloud_provider=true`), not the entire
  deep-research runtime. Offline diagnostics report enabled
  state without values.
- **Deferred** — intentionally outside the supported evaluator
  path. No implementation, no test, no documentation claim in
  the current commit. A future experiment MAY propose
  implementation; the current slice does not approve it.
- **Absent** — no implementation, no test, no documentation
  claim. The capability is not part of the current product
  surface and is not deferred as a future experiment either.
- **Design only** — documented for a later slice; the design
  exists in an ADR or sketch but no runtime behavior exists in
  the current commit.

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

The current Oroimen state at `0e29dd6f` is summarized below using
the same six named states above. Numeric level numbers are not
used; the named state itself is the only label.

- **Deep Research report-retrieval vertical (HTTP → create → real
  5-phase pipeline → atomic write → owner-scoped detail → owner-
  scoped report → notifier → owner isolation → cleanup):**
  **Deterministic vertical integration is proven.** Real
  components exercise the full product path through authenticated
  HTTP; external seams (search, fetch, LLM, notifier, scheduler
  trigger) are scripted fakes. NOT **Live provider behavior is
  proven** and NOT **Research quality is measured.**
- **Deep Research runtime pipeline behind opt-in
  (`HERMES_DEEP_RESEARCH_ENABLED=true`):** **Code exists** AND
  **Runtime composition exists.** The composition root
  constructs the real `LocalReportStore`, scheduler, and recovery,
  publishes `model_output_enforced=True` only after full wiring,
  and is fail-closed on any failure. NOT **Deterministic vertical
  integration is proven** (no public deterministic vertical is
  wired behind opt-in outside the
  `tests/e2e/test_deep_research_vertical.py` test) and NOT **Live
  provider behavior is proven.**
- **Live Tavily behavior, live cloud LLM behavior, Telegram
  notifier transport:** **Code exists** only. No live provider
  run has been authorized and no live state is proven in this
  commit.
- **Research quality:** **Code exists** (the calibration plan
  exists in this slice) — but NOT **Deterministic vertical
  integration is proven** (no frozen benchmark has been executed
  against the production pipeline) and NOT **Research quality is
  measured** (no structured human review has been published).
- **Product support:** **Code exists** (the public repo describes
  the deterministic vertical and the calibration plan) — but NOT
  **Product support is documented** (no public README or ADR
  claims "supported product path" for deep-research quality,
  because the calibration has not been run).

Slice 1C3 raised the report-retrieval vertical to **Deterministic
vertical integration is proven**; it did not raise the
live-provider or research-quality states, and the ledger does not
conflate them.

## Ledger

| Capability | Public evidence | Runtime and support status | Disposition | Main risks or dependencies | Acceptance or demo criterion | Priority |
| --- | --- | --- | --- | --- | --- | --- |
| Local chat and health path | `README.md`; `hermes/receivers/http_api.py`; `tests/unit/test_http_api.py`; `tests/e2e/test_phase1_smoke.py` | Supported | Retain and regression-test | Local model and container readiness | Follow the README quickstart and complete health plus first-chat checks | P0 |
| File ingestion and grounded retrieval | `hermes/memory/drop_watcher.py`; `hermes/memory/ingest_router.py`; `hermes/memory/vault.py`; `tests/integration/test_files_e2e.py`; `tests/e2e/test_sprint_19_pipeline.py` | Supported | Retain and regression-test | Extractor and embedding availability; untrusted document content | Ingest a public fixture and retrieve grounded content through the documented path | P0 |
| Prompt-injection classification and tool bounds | `hermes/security/classifier.py`; `hermes/tools/security.py`; `hermes/agent/loop.py`; `tests/e2e/test_rag_injection.py`; `tests/unit/test_agent_loop.py` | Supported | Retain and regression-test | Model-dependent behavior remains separately evaluated | Deterministic security tests pass; live-model evidence is never inferred from skipped tests | P0 |
| Deep Research service components (orchestration, scheduler, recovery) | `hermes/jobs/service.py`; `hermes/jobs/scheduler.py`; `hermes/jobs/recovery.py`; `hermes/jobs/models.py`; `hermes/memory/db.py`; `hermes/__main__.py` (composition root) | Implemented, runtime available behind opt-in (`HERMES_DEEP_RESEARCH_ENABLED=true`); the composition root constructs the real `LocalReportStore` inside the outer `try:` and publishes the singleton only after the report store, scheduler, and recovery succeed | Retain and regression-test | Disabled-by-default posture is preserved; runtime wiring is fail-closed (a failure anywhere in the outer `try:` runs `rollback()` and returns `(unavailable, None, None)`; the singleton is never published with a missing report store); a failed `LocalReportStore` construction propagates and the route returns 503 (defensive) or 500 (unavailable) rather than publishing a half-built service | The composition root's `DeepResearchCapabilities` is published with `service_wiring=True`, `recovery_wiring=True`, `search_backend_configured=True`, `llm_provider_configured=True`, `fetch_policy=True`, `external_fetch=True`, `model_output_enforced=True`, and `report_retrieval=True` only when the full wired path succeeds | P0 |
| Deep Research jobs API contract | `hermes/receivers/jobs_api.py`; `hermes/jobs/models.py`; `tests/integration/test_jobs_api.py` | Implemented, runtime available behind opt-in: with a wired service singleton the routes return the documented contract (200 detail, 200 report, 200 list, 200 budget, 200 preflight, 200/409 cancel, 201/409 retry); without a singleton the routes return 503 `service_unavailable` | Retain and regression-test | A defensive 503 branch protects against a service that was wired but lacks `_report_store`; this branch is a safety net, not a normal path | A real application instance serves create, detail, list, budget, preflight, cancel, and retry without test-only dependency injection | P0 |
| Deep Research preflight | `hermes/jobs/preflight.py`; `hermes/receivers/jobs_api.py` (route); `tests/unit/test_jobs_preflight.py`; `tests/integration/test_jobs_api.py` (`test_preflight_*`) | Implemented, runtime available behind opt-in (offline evaluator) | Retain and regression-test | The `evaluate_deep_research_preflight` evaluator is the single source of truth; the route wires the `DeepResearchCapabilities` from the composition root, NOT a patched response | `GET /v1/jobs/preflight` returns the contract from the real evaluator with `status` derived from the wired `capabilities` object; live checks remain `skip` in offline mode | P0 |
| Search routing | `hermes/services/search/`; `hermes/tools/web_search.py`; `hermes/__main__.py`; `tests/unit/test_search_router.py`; `tests/unit/test_search_tavily.py` | Optional; Deep Research intent routes to Tavily when configured | Retain as opt-in and expose redacted readiness | Cloud credentials, provider budget, network egress, and backend health semantics | Offline preflight reports configuration only; explicit live checks prove reachability without sending a user query | P0 |
| Safe external fetch for Deep Research | `hermes/jobs/safe_fetcher.py`; `hermes/jobs/service.py` (phase 2 uses `_fetcher.fetch(url)`); `tests/integration/test_safe_external_fetcher_transport.py`; `tests/integration/test_jobs_service_phases.py` | Implemented, runtime available behind opt-in; the production service uses `SafeExternalFetcher` exclusively for phase 2; no direct `httpx.AsyncClient` or fallback transport exists in the service | Retain and regression-test | Redirect SSRF, private and special-use addresses, IPv6, DNS rebinding, proxy inheritance, and fully buffered oversized responses are covered by adversarial unit tests | Adversarial tests cover the initial URL, every redirect hop, A and AAAA results, proxy behavior, and streamed byte limits | P0 |
| Deep Research model-output limits | `hermes/jobs/service.py` (per-source and final synthesis pass `max_tokens` to `llm.chat`); `hermes/jobs/cost.py`; `tests/integration/test_jobs_service_phases.py`; `tests/integration/test_jobs_cost_drift.py` | Implemented, runtime available behind opt-in; phase 3 (per-source) calls `llm.chat(..., max_tokens=settings.deep_research_per_source_max_tokens)` and phase 4 (final synthesis) calls `llm.chat(..., max_tokens=settings.deep_research_output_max_tokens)`; the composition root sets `model_output_enforced=True` only after this wiring is in place | Retain and regression-test | The two settings must reach both LLM call sites; cost drift between checkpoint, token-usage sum, and DB aggregate is reconciled via `reconcile_cost` | Per-source `max_tokens` matches `settings.deep_research_per_source_max_tokens`; final `max_tokens` matches `settings.deep_research_output_max_tokens` | P0 |
| Daily budget admission control | `hermes/jobs/service.py`; `hermes/config.py`; `tests/integration/test_jobs_cost_drift.py` | Implemented, runtime available behind opt-in; pre-check on submit + atomic TOCTOU check inside the running transaction | Retain and regression-test | The daily cap is checked before enqueue and re-checked at job start (atomic) to prevent TOCTOU; ``reconcile_cost`` persists the reconciled maximum via ``_db.set_research_job_cost_monotonic`` (atomic ``MAX(cost_usd, ?)``) so subsequent reads of the aggregate — including the completion notifier's ``cost_usd`` argument and ``JobDetail.cost_usd`` after completion — expose the same persisted reconciled value; the persistence is monotonic (the aggregate is never decreased) and idempotent (re-running ``reconcile_cost`` on the same state returns the same value with no double counting) | A job submitted and run after the cap is reached transitions to `failed` with `error_taxonomy="budget_exceeded"`; it is NOT silently enqueued; the notifier receives the persisted reconciled maximum across checkpoint cost, recorded token-usage sum, and existing job aggregate (NOT the pre-reconciliation stale aggregate and NOT solely the ``TokenUsageEntry`` aggregate, because the checkpoint may legitimately win after a token-usage DB write failure) | P0 |
| Per-job monetary budget enforcement | `hermes/jobs/service.py` (per-job `cost` recorded via `_record_token_usage` + `reconcile_cost`); `hermes/jobs/cost.py`; `hermes/memory/db.py` (`set_research_job_cost_monotonic`) | Implemented, runtime available behind opt-in; soft warning only (the service records the per-job cost and emits a log warning when the soft budget is exceeded, but does NOT cancel the job mid-flight) | Retain as soft warning; a hard per-job cancellation boundary is NOT yet implemented | The service records the per-job cost and surfaces it through `JobDetail` / token-usage telemetry and the notifier call; the notifier receives the **persisted reconciled maximum** across checkpoint cost, recorded token-usage sum, and existing job aggregate (NOT the pre-reconciliation stale aggregate and NOT solely the `TokenUsageEntry` aggregate, because the checkpoint may legitimately win after a token-usage DB write failure); the per-job cost is NOT embedded in the final Markdown report; the service does NOT cancel the job mid-flight when the per-job budget is exceeded | The per-job cost is exposed in `JobDetail.cost_usd` and `TokenUsageEntry.cost_usd` (DTO and embedded in `JobDetail.token_usage`) and in the notifier call; the notifier's `cost_usd` argument equals the post-completion `JobDetail.cost_usd` (both come from the same persisted reconciled maximum); the per-job budget is documented as a soft warning, not a hard cancellation; the report body is the LLM-generated content only | P0 |
| Cost truth and runtime documentation (DR-Q1A-PRE1A) | `hermes/jobs/cost.py` (`PRICING_TABLE`, `PRICING_BASIS`, `PRICING_AS_OF`, `PRICING_SOURCE`); `hermes/jobs/service.py` (`cancel_job`, `_run_phase_with_retry` docstrings); `hermes/jobs/models.py` (Field descriptions on `JobResponse.estimated_cost_usd`, `JobSummary.cost_usd`, `TokenUsageEntry.cost_usd`, `DailyBudgetStatus.{today_cost_usd, daily_cap_usd, remaining_usd}`, `CancelResponse.{status, graceful}`); `tests/unit/test_jobs_cost.py`; `tests/unit/test_jobs_cost_truth.py`; `tests/unit/test_jobs_lifecycle_docstring.py` | Implemented, runtime available; the cost truth is enforced at the source-of-truth module (`hermes.jobs.cost`) and propagated through every call site of `calculate_cost` and `estimate_research_cost`; the cancel and retry documentation is corrected to match the actual current behavior; the public DTO Field descriptions make the paygo-equivalent semantics explicit at the API surface | Retain and regression-test | `cost_usd` is an **estimated pay-as-you-go-equivalent amount** at the official standard rates, NOT actual provider billing. Operators using a subscription or quota-backed plan must treat `cost_usd` as a relative cost proxy, not as a spend figure. The `PRICING_TABLE` reflects the official standard tier, "Permanent 50% off" promotional pricing, `<=512k` input tokens tier (verified 2026-07-21 from `https://platform.minimax.io/docs/guides/pricing-paygo`); the `>512k` tier is a future-slice dispatch concern. `cost_usd` is exposed through the Deep Research job API DTOs (`JobResponse.estimated_cost_usd`, `JobSummary.cost_usd`, `JobDetail.cost_usd`, `TokenUsageEntry.cost_usd`), the daily-budget admission-control DTO (`DailyBudgetStatus.{today_cost_usd, daily_cap_usd, remaining_usd}`), InfluxDB / metrics writes, and the notifier call. `cost_usd` is NOT automatically embedded in the final Markdown report returned by `GET /v1/jobs/{job_id}/report`; the report body is the LLM-generated content only. The `cancel_job` and `_run_phase_with_retry` docstrings are accurate to the current behavior; no implementation changed in this slice. Calls that time out without returning are not represented in the token-usage table. **PR #9 scope statement (truth-patch 3):** Truth-patch 3 itself has no runtime behavior change. PR #9 intentionally changes paygo-equivalent cost-estimation and budget-admission values by correcting `PRICING_TABLE` (the verified official paygo rates for `MiniMax-M3` and `MiniMax-M2.7-highspeed` per `PRICING_AS_OF = "2026-07-21"`). PR #9 does NOT change cancellation behavior, retry behavior, provider execution, API response shape, DTO field names/types, OpenAPI contract, or database-schema behavior. The pre-submit `estimate_research_cost` arithmetic and the `_ESTIMATION_SAFETY_MARGIN_PCT = Decimal("1.30")` heuristic value are unchanged; only their inline comments and docstrings were clarified in this slice. | All call sites of `calculate_cost` and `estimate_research_cost` use the verified rates; `PRICING_BASIS == "official_paygo_equivalent"`; `PRICING_AS_OF == "2026-07-21"`; `cancel_job` documentation describes DB-only behavior and the absence of in-flight task or provider-request cancellation; `_run_phase_with_retry` documentation describes 3 total attempts with effective waits [1, 4] and the residue value 16 in `_RETRY_BACKOFF_SCHEDULE` that is not consumed by the current loop; the public DTO Field descriptions contain the paygo-equivalent semantics and the report-exposure truth | P0 |
| Real Deep Research cancellation contract (DR-Q1A-PRE1B) | `hermes/jobs/service.py` (`_active_tasks`, `_user_cancel_intent`, `_terminal_locks`, `_get_terminal_lock`, `_register_active_task`, `_unregister_active_task`, `_peek_active_task`, `_mark_user_cancel_intent`, `_user_cancel_intended`, `_clear_user_cancel_intent`, `_handle_cancellation`, `_run_research` outer/inner split, `_phase_write` terminal-seam split, `_update_phase` conditional update, `cancel_job` state-machine refactor); `hermes/jobs/scheduler.py` (`enqueue` non-enqueueable list now includes `cancelling`, `cancel_scheduled` best-effort); `hermes/memory/db.py` (`transition_research_job_status` CAS, `update_research_job_phase` conditional update, `get_today_research_cost` now includes cancelled rows); `hermes/jobs/models.py` (`CancelResponse` and `cancel` route docstrings re-defined for real cancellation); `hermes/receivers/jobs_api.py` (`graceful` Query description re-defined as a wait mode); `hermes/config.py` (`deep_research_cancel_wait_s` validated setting, default 5.0s, ge=0.1, le=30.0, env alias `HERMES_DEEP_RESEARCH_CANCEL_WAIT_S`); `.env.example`; `tests/integration/test_jobs_cancellation.py` (NEW, 22 tests covering A–T); `tests/unit/test_jobs_lifecycle_docstring.py` (PRE1A-era "does not prove task cancellation" caveat removed and replaced with a pending-job `cancelled` finalization test); `tests/unit/test_jobs_dto_truth.py` (PRE1A-era "not prove" caveat assertion removed) | Implemented, runtime available behind opt-in. The previous PRE1A behavior was DB-only: the cancel endpoint wrote `status='cancelling'` to the DB and the running asyncio task was not signalled. PRE1B makes cancellation real for every state: the service registers the active task in `_active_tasks` before the pending→running CAS, marks a per-job user-cancel intent, signals the task via `task.cancel()`, and finalizes the row through a single conditional CAS. `graceful` is a WAIT MODE, not a strength mode; both values request real local cancellation. The cancel endpoint never claims a provider-side guarantee: an already-received provider request may still be processed or counted; cancellation does NOT claim quota reversal, refund, or reversal of billed tokens. | Retain and regression-test | The product contract: when the owner cancels a Deep Research job, Oroimen stops executing that job locally. The state machine handles pending (synchronous finalize as `cancelled`), running (transition to `cancelling`, signal task, bounded wait for `graceful=True` bounded by `deep_research_cancel_wait_s`), already-cancelling (idempotent re-signal), already-cancelled (200 idempotent), and complete/failed (409 `JobAlreadyTerminalError`). Cancelled cost is reconciled before cleanup and remains part of daily-budget accounting. No `partial_output_path` is exposed; transient artifacts (`.md.tmp`, checkpoint, job dir) are cleaned; DB job and token-usage remain inspectable. The per-job monetary budget remains a soft warning; automatic hard monetary cancellation is NOT implemented (deferred to a measured-data-driven slice). | The 22 cancellation tests pass deterministically offline; no `partial_output_path` in `CancelResponse`; `graceful` Query and `CancelResponse.graceful` Field descriptions document the real-cancellation contract; `transition_research_job_status` returns the conditional-update outcome and is the only writer of `cancelling`/`cancelled` outside the in-process finalizer; `_ESTIMATION_SAFETY_MARGIN_PCT` and `calculate_cost` are unchanged from PRE1A | P0 |
| Deep Research report persistence | `hermes/jobs/service.py` (`_phase_write` uses `tmp + flush + os.fsync + os.replace`); `hermes/jobs/models.py` (DTOs without `output_path` / `partial_output_path` / `checkpoint_path` / `report_available`); `hermes/jobs/report_paths.py`; `hermes/jobs/report_store.py`; `hermes/receivers/jobs_api.py` (route); `tests/integration/test_jobs_api.py`; `tests/unit/test_report_paths.py`; `tests/unit/test_report_store.py`; `tests/e2e/test_deep_research_vertical.py` | Implemented, runtime available behind opt-in (Slice 1C2 + 1C3): owner-scoped `GET /v1/jobs/{job_id}/report` returns the final Markdown through the real `LocalReportStore`; the public DTOs no longer carry filesystem paths; the notifier template no longer embeds a path; the deterministic vertical golden journey proves the full owner-scoped flow through real HTTP | Retain and regression-test | A failed report-store construction is fail-closed (the singleton is not published and the route returns 500 `report_unavailable` or 503 `service_unavailable` defensively); the read path uses `report_store.derive_path(job_id)` and never reads from the DB `output_path` column; the writer and the reader share the same resolved absolute Path through `service._data_root = report_store.root` (round 2 wiring) | The owner retrieves report content by job ID with a constant `text/markdown; charset=utf-8` body, the documented headers, and a deterministic body shape; missing and foreign-owned jobs return byte-identical 404 `job_not_found`; complete status with a missing/oversize/invalid-UTF-8 file returns 500 `report_unavailable` with no internal details leaked | P0 |
| Deep Research report settings: `deep_research_data_root` and `deep_research_max_report_bytes` | `hermes/config.py`; `hermes/jobs/report_store.py`; `hermes/jobs/preflight.py`; `hermes/jobs/service.py` (writer uses canonical store root); `hermes/__main__.py` (composition root) | Implemented, runtime available behind opt-in (Slice 1C2): the report-store root is resolved at startup and the max-bytes cap is enforced inside the bounded read of the single opened handle (round 2); the composition root resolves `data_root` and the service uses `report_store.root` as `service._data_root` (round 2) | Retain and regression-test | The two settings must reach both the composition root and the read path; the max-bytes floor (10 KiB) and ceiling (50 MiB) are validated at construction | An oversize report is rejected with 500 `report_unavailable` (logs may tag `report_size_limit_exceeded`); a missing or non-creatable data root fails closed at startup; the writer and the reader share the same canonical Path | P0 |
| Deterministic Deep Research vertical E2E | `tests/e2e/test_deep_research_vertical.py` (NEW, 1 file, 831 lines, merged in Slice 1C3) | Implemented, runtime available, quality unmeasured | Retain and regression-test; a frozen benchmark execution is a separate future slice (see `docs/DR_Q1A_BASELINE_CALIBRATION_PLAN.md`) | The golden journey drives the full 9-step product path through authenticated HTTP: preflight, create, real 5-phase pipeline, atomic write, owner-scoped detail, owner-scoped report, notifier privacy, owner isolation, and lifecycle cleanup. Only the external seams (search, fetch, LLM, notifier, scheduler trigger) are faked; everything else is real | The 4 tests pass on every run, in 3 consecutive runs, and in 3 in-process cycles; the test carries `@pytest.mark.e2e` and no `slow` / `network` markers; the LLM mock uses a modulo-wrap `side_effect` counter for determinism | P0 |
| Memory collections and embeddings | `hermes/memory/collections.py`; `hermes/services/embedding_router.py`; `hermes/services/embed_vault.py`; corresponding unit and integration tests | Optional: supported behavior depends on the operator-selected local or cloud provider; the default path runs without cloud credentials | Retain; document provider readiness separately | Model availability and resource limits | Public fixture ingestion and retrieval pass without cloud credentials on the default path | P1 |
| OCR routing and edge coordination | `hermes/memory/ocr_decision.py`; `hermes/memory/edge_coordinator.py`; `hermes/receivers/ocr_api.py`; OCR unit tests | Implemented, runtime unavailable: the public generic OCR decision and route are present and unit-tested offline, but the production startup does not wire the advanced edge-coordinator path; the edge path is deployment-dependent and not part of the public product surface | Retain public generic behavior; keep deployment assumptions out of public docs | External binaries, local vision model, and edge lifecycle | Public OCR decision tests pass; deployment-specific readiness is not claimed | P2 |
| LLM provider cascade and streaming | `hermes/llm/router.py`; `hermes/llm/ollama.py`; `hermes/llm/chatgpt5_6.py`; provider and streaming tests | Supported: the local provider is the default and is enabled without operator opt-in; the cloud and frontier providers are separately Optional and require operator opt-in (per the row above for cloud providers) | Retain explicit selection and fallback semantics | Credentials, cost, data egress, provider availability | Offline diagnostics list provider modes without values; live probes are explicit and bounded | P1 |
| Container egress firewall | `hermes/security/egress.py`; `tests/unit/security/test_egress.py` | Optional (disabled by default) | Retain as defense in depth, not as request-level URL authorization | DNS is resolved when rules are applied; stale DNS and request-level SSRF remain separate concerns | Diagnostics report only enabled state and policy validity, never addresses or sensitive configuration | P1 |
| Deep Research iterative retrieval | absent | Absent: no multi-pass planning, no reflection step, no re-query with refined terms, no stopping decision | Deferred | An iterative loop would require a new boundary in the service, new preflight codes, and a benchmark to prove it improves quality; none of these is justified at the current baseline | Not supported as a product path | P2 |
| Deep Research query decomposition | absent | Absent: the service calls a single search query per job; no static or learned decomposition of complex queries into sub-questions | Deferred | A decomposition layer would require a new phase, new prompts, and a benchmark to prove it improves quality; none of these is justified at the current baseline | Not supported as a product path | P2 |
| Deep Research claim-level provenance and citation verification | absent | Absent: the service does not extract individual claims, does not verify citation support, and does not produce a claim ledger | Deferred | A claim parser + verifier would require a new phase, a new boundary, and a benchmark to prove it improves quality; none of these is justified at the current baseline | Not supported as a product path | P2 |
| Deep Research contradiction handling | absent | Absent: the service does not explicitly detect or surface contradictions between retrieved sources | Deferred | A contradiction-handling phase would require a new boundary, new prompts, and a benchmark; none of these is justified at the current baseline | Not supported as a product path | P2 |
| Deep Research quality benchmark | absent | Absent: no frozen benchmark, no rubric, no run manifest, no human audit procedure has been published in the current commit; the existing deterministic E2E is a smoke, not a quality measurement; the calibration plan in `docs/DR_Q1A_BASELINE_CALIBRATION_PLAN.md` is a design-only artifact, not an executed benchmark | Calibration slice planned (see `docs/DR_Q1A_BASELINE_CALIBRATION_PLAN.md`) | A real benchmark needs an owner-approved corpus, rubric, and reviewer workflow; the calibration plan exists but the benchmark has not been executed | A future measurement may show that the existing pipeline meets the bar; the existing pipeline MUST NOT be modified in response to LLM recommendations until a measurement is published | P0 |

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

## Current deterministic proof (Slice 1C3, merged at b95afb4; current ledger at 0e29dd6f)

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
- **Live DR cancellation transport behavior.** PRE1B proves that
  the local asyncio cancellation reaches a blocking fake
  `MockTransport` awaitable and prevents later phase starts in
  deterministic offline tests. The behavior against a real
  in-flight provider request (Tavily, MiniMax, OpenAI) has
  NOT been measured end-to-end with provider credentials. The
  contract is honest: an already-received provider request may
  still be processed or counted by the provider; cancellation
  does NOT claim quota reversal, refund, or reversal of billed
  tokens.
- **Per-job monetary budget hard cancellation.** The per-job
  budget is recorded and surfaced but does NOT cancel the job
  mid-flight when exceeded. The hard cancellation boundary is
  not yet implemented. The daily budget IS enforced (it is
  checked before enqueue and re-checked at job start); the
  per-job budget is a soft warning. PRE1B only makes
  user-initiated cancellation real.

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

## Approved shortlist (status as of 0e29dd6f)

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

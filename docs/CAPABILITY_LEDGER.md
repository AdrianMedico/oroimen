# Public capability ledger

This ledger describes the public repository at commit
`e5602ef4aa88f3fffc13c3e97d11a42bf3df98b5`. It separates code presence from
runtime availability and product support. A module is not considered a
supported feature merely because its implementation and component tests exist.

## Status vocabulary

- **Supported**: wired into the documented quickstart and covered by its
  verification path.
- **Implemented, runtime unavailable**: components exist, but production
  startup does not construct or register the required service.
- **Optional**: wired only when an operator supplies explicit configuration.
- **Deferred**: intentionally outside the supported evaluator path.
- **Design only**: documented for a later slice; no runtime behavior exists.

## Ledger

| Capability | Public evidence | Runtime and support status | Disposition | Main risks or dependencies | Acceptance or demo criterion | Priority |
| --- | --- | --- | --- | --- | --- | --- |
| Local chat and health path | `README.md`; `hermes/receivers/http_api.py`; `tests/unit/test_http_api.py`; `tests/e2e/test_phase1_smoke.py` | Supported evaluator path | Retain and regression-test | Local model and container readiness | Follow the README quickstart and complete health plus first-chat checks | P0 |
| File ingestion and grounded retrieval | `hermes/memory/drop_watcher.py`; `hermes/memory/ingest_router.py`; `hermes/memory/vault.py`; `tests/integration/test_files_e2e.py`; `tests/e2e/test_sprint_19_pipeline.py` | Supported evaluator path | Retain and regression-test | Extractor and embedding availability; untrusted document content | Ingest a public fixture and retrieve grounded content through the documented path | P0 |
| Prompt-injection classification and tool bounds | `hermes/security/classifier.py`; `hermes/tools/security.py`; `hermes/agent/loop.py`; `tests/e2e/test_rag_injection.py`; `tests/unit/test_agent_loop.py` | Supported security path | Retain and regression-test | Model-dependent behavior remains separately evaluated | Deterministic security tests pass; live-model evidence is never inferred from skipped tests | P0 |
| Deep Research service components | `hermes/jobs/service.py`; `hermes/jobs/models.py`; `hermes/jobs/scheduler.py`; `hermes/jobs/recovery.py`; `hermes/memory/db.py`; `tests/integration/test_jobs_service_phases.py`; `tests/integration/test_jobs_recovery_real.py`; `tests/integration/test_jobs_cost_drift.py` | **Implemented, runtime unavailable, and deferred** | Productize existing public components; do not copy another implementation | Startup wiring absent; external fetch is not SSRF-safe; configured model-output settings are not hard caps; report content has no retrieval endpoint | A future supported smoke must create a job through the real app, observe all phases, retrieve report content, and exercise cancellation and retry | P0 |
| Deep Research jobs API contract | `hermes/receivers/jobs_api.py`; `hermes/jobs/models.py`; `tests/integration/test_jobs_api.py` | Routes are mounted, but the service dependency is not registered by production startup; requests therefore take the documented 503 path | Keep the contract, add safe runtime wiring only after preflight and fetch controls | Component API tests use a fake service and do not prove startup integration | A real application instance serves create, detail, list, budget, cancel, and retry without test-only dependency injection | P0 |
| Search routing | `hermes/services/search/`; `hermes/tools/web_search.py`; `hermes/__main__.py`; `tests/unit/test_search_router.py`; `tests/unit/test_search_tavily.py` | Optional; Deep Research intent routes to Tavily when configured | Retain as opt-in and expose redacted readiness | Cloud credentials, provider budget, network egress, and backend health semantics | Offline preflight reports configuration only; explicit live checks prove reachability without sending a user query | P0 |
| Source URL fetching for Deep Research | `hermes/jobs/service.py`; `hermes/services/search/router.py` | Implemented but not safe to enable as a supported external-fetch path | Replace direct fetch with a shared safe fetch policy before runtime enablement | Redirect SSRF, private and special-use addresses, IPv6, DNS rebinding, proxy inheritance, and fully buffered oversized responses | Adversarial tests cover the initial URL, every redirect hop, A and AAAA results, proxy behavior, and streamed byte limits | P0 |
| Deep Research report persistence | `hermes/jobs/service.py`; `hermes/jobs/models.py`; `hermes/jobs/report_paths.py`; `hermes/jobs/report_store.py`; `hermes/receivers/jobs_api.py`; `tests/integration/test_jobs_api.py`; `tests/unit/test_report_paths.py`; `tests/unit/test_report_store.py` | Supported (Slice 1C2): owner-scoped `GET /v1/jobs/{job_id}/report` returns the final Markdown via the new `LocalReportStore`; public DTOs no longer carry filesystem paths; the notifier template no longer embeds a path | Retain and regression-test | A failed report-store construction is fail-closed (the singleton is not published and the route returns 500 `report_unavailable`); file is canonical-derivation only and never read from the DB `output_path` column | The owner retrieves report content by job ID with a constant `text/markdown; charset=utf-8` body; missing and foreign-owned jobs return byte-identical 404 `job_not_found`; complete status with a missing/oversize/invalid-UTF-8 file returns 500 `report_unavailable` with no internal details leaked | P0 |
| Deep Research report settings: `deep_research_data_root` and `deep_research_max_report_bytes` | `hermes/config.py`; `hermes/jobs/report_store.py`; `hermes/jobs/preflight.py` | Supported (Slice 1C2): the report-store root is resolved at startup and the max-bytes cap is enforced before the body is read into memory | Retain and regression-test | The two settings must reach both the composition root and the read path; the max-bytes floor (10 KiB) and ceiling (50 MiB) are validated at construction | An oversize report is rejected with 500 `report_unavailable` (logs may tag `report_size_limit_exceeded`); a missing or non-creatable data root fails closed at startup | P0 |
| Memory collections and embeddings | `hermes/memory/collections.py`; `hermes/services/embedding_router.py`; `hermes/services/embed_vault.py`; corresponding unit and integration tests | Present; supported behavior depends on the selected local or optional provider | Retain; document provider readiness separately | Model availability and resource limits | Public fixture ingestion and retrieval pass without cloud credentials on the default path | P1 |
| OCR routing and edge coordination | `hermes/memory/ocr_decision.py`; `hermes/memory/edge_coordinator.py`; `hermes/receivers/ocr_api.py`; OCR unit tests | Present; advanced edge paths are deployment-dependent | Retain public generic behavior; keep deployment assumptions out of public docs | External binaries, local vision model, and edge lifecycle | Public OCR decision tests pass; deployment-specific readiness is not claimed | P2 |
| LLM provider cascade and streaming | `hermes/llm/router.py`; `hermes/llm/ollama.py`; `hermes/llm/chatgpt5_6.py`; provider and streaming tests | Local provider is the supported default; cloud and frontier providers are opt-in | Retain explicit selection and fallback semantics | Credentials, cost, data egress, provider availability | Offline diagnostics list provider modes without values; live probes are explicit and bounded | P1 |
| Container egress firewall | `hermes/security/egress.py`; `tests/unit/security/test_egress.py` | Optional and disabled by default | Retain as defense in depth, not as request-level URL authorization | DNS is resolved when rules are applied; stale DNS and request-level SSRF remain separate concerns | Diagnostics report only enabled state and policy validity, never addresses or sensitive configuration | P1 |

## Sanitized disposition policy

1. Treat this public commit as the evidence source. Do not publish private
   deployment files, environment files, agent logs, personal data, credentials,
   or raw historical planning material.
2. If public implementation already exists, verify and productize it. Do not
   recopy code from another repository.
3. Distill still-relevant historical design into a current public decision
   record. Do not publish raw history.
4. Migrate generic tests or security tooling only after a fresh public-value
   and sanitization review.
5. Keep infrastructure topology, operator-specific telemetry, and deployment
   assumptions outside the public repository.
6. Require a public evidence path for every capability claim. Mark uncertainty
   instead of inferring runtime support from component presence.

## Slice 0 smoke baseline

Environment used for this baseline:

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

These results prove the deterministic unit and component baselines. They do
**not** prove a Deep Research product journey: the production startup does not
register the service, the component tests use fakes or mocks where appropriate,
and live search and LLM egress were not exercised.

## Approved shortlist

Slice 1 should enable one safe Deep Research vertical slice in this order:

1. Offline preflight contract and stable diagnostics.
2. Request-level safe external fetcher, disabled by default until its
   adversarial tests pass.
3. Production startup wiring behind explicit opt-in configuration.
4. Enforced LLM output limits and accurately classified budget controls.
5. Authenticated report-content retrieval.
6. Explicit, bounded live verification and user documentation.

Query decomposition and broader autonomous research behavior remain deferred
until the supported vertical slice is measurable and stable.

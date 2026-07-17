# Oroimen Final Verification

**Status:** adversarial review passed; external release gates remain open
**Baseline HEAD:** `8003dc9`
**Candidate SHA:** PENDING
**Prepared:** 2026-07-17

This report separates verified results from pending human or external
work. A skipped, mocked, collected-only, or undocumented live test is
never counted as live evidence.

## Gate status

| Gate | Status | Evidence |
|---|---|---|
| A — offline worktree | PASS with documented warning debt | 1765 passed, 2 skipped, 1 expected failure; Ruff passes on `hermes` and `tests`; mypy passes 91 source files |
| B — GPT-5.6 live | PENDING | Router-level opt-in smoke exists and is bounded at 60 seconds; no real result recorded |
| C — container demo | PENDING | Compose syntax passes; Docker daemon was unavailable for image inspection and smoke |
| D — submission integrity | PENDING | Candidate manifest exists; final SHA and fresh clone do not |
| E — adversarial review | PASS | R6 integration: 0 BLOCKING / 0 MAJOR. R7 correctness: 0 BLOCKING / 0 MAJOR after producer-consumer VAULT_INGEST alignment. |
| F — submission | HUMAN PENDING | Public repository, video URL, and primary build-task `/feedback` ID are not recorded |

## Verified commands

| Command | Result |
|---|---|
| `uv run pytest tests/unit/test_chatgpt5_6_provider.py -q` | 28 passed |
| `.venv\\Scripts\\pytest.exe -p no:cacheprovider tests/unit/ -m "not slow" -n 4 -q` | 1765 passed, 2 skipped, 1 expected failure, 30 marker warnings |
| `python -m pytest tests/unit/test_f2_public_datasets.py -m "not slow" -q` | 9 passed, 3 skipped, 50 deselected |
| `python -m ruff check hermes tests` | PASS |
| `python -m mypy hermes` | PASS: 91 source files |
| Router import check | PASS |
| `docker compose config --quiet` | PASS |
| `gitleaks dir . --redact=100 --no-banner` | PASS: 0 findings across the 242-file candidate tree |
| Live F2 collection | 14 cases collected; no live result claimed |
| GPT-5.6 live smoke without external access | skipped as designed |

The repository-wide format check and lint over all local scripts are
not claimed clean. They expose pre-existing working-tree debt outside
the polished candidate. Only the manifest allowlist may enter the
submission commit.

## First dual R1 findings

The fresh correctness/security reviewer returned **4 BLOCKING + 2 MAJOR**.
The fresh integration/evidence reviewer returned **7 BLOCKING + 4 MAJOR**.
Overlapping findings were consolidated and personally verified. Confirmed
issues included: provider-returned model evidence, Compose model bootstrap,
drop-folder/API wiring, stale redaction and second-provider claims, nonexistent
manifest paths, runtime script completeness, platform/license metadata, and
demo/path drift.

Code and document remediation is in this worktree. The first R1 verdict remains
recorded below rather than overwritten by later fixes.

## Second dual R1 findings

The fresh correctness/security reviewer returned **2 BLOCKING + 1 MAJOR**.
The fresh integration/evidence reviewer returned **3 BLOCKING + 1 MAJOR**.
Confirmed issues were residual deployment identifiers, an uppercase OCI image
namespace, plural provider wording, a public RAG path that was not enabled by
Compose, unverified setup/resource claims, and unnecessary internal documents
in the public allowlist.

The worktree remediation now bootstraps both local models, configures the local
1024-dimensional embedding tier for chat RAG and vault ingest, enables the
file-search tools and drop watcher, mounts `./drop`, and polls embedding work
every five seconds. A focused dimension/factory regression test was added.
Public narrative and allowlist scans were clean at the end of R2, all 25 then-listed
manifest paths existed, and the unit gate passed. Runtime container smoke,
immutable image pinning, and a fresh-clone gate remained external/final-SHA work.

## Third dual R1 findings

The fresh correctness reviewer returned **4 BLOCKING + 4 MAJOR** and the fresh
integration reviewer returned **5 BLOCKING + 5 MAJOR**. Confirmed defects were:
lazy embedding initialization preventing `search_files` registration; a split
storage contract where ingestion wrote `vault_chunks` but search read legacy
`file_embeddings`; normal chat never reaching the enabled frontier; an
unwritable default inbox path; outbound tools being enabled by the local tools
switch; local vision advertised without public runtime wiring; one recursive
test import absent from the manifest; malformed F2 opening tags; and stale or
unsupported documentation claims.

The remediation now constructs one `VaultEmbedder` for ingestion and retrieval,
lazily initializes the embedding service, returns ranked chunk text through the
guarded tool-output path, exposes `oroimen-agent-frontier` as an explicit opt-in API model alias,
adds a writable inbox mount contract, and separates local tools from outbound
network tools. Search, weather, and Agent-Reach default off. The public Compose
path explicitly enables the HTTP API while keeping it loopback-bound. Vision is
identified as an unwired adapter, `scripts/pr_review.py` is allowlisted, the F2
tag is valid, and the affected budget tests derive wrapper overhead dynamically.

Post-remediation evidence: 1765 unit tests passed, 2 skipped, 1 expected failure;
Ruff passes; mypy passes 91 source files; 172 focused RAG/frontier tests pass.
A fourth fresh dual review is still required before Gate E can pass.

## Fifth dual adversarial review

The R5 correctness reviewer found **1 BLOCKING + 4 MAJOR**: exact F2
budgeting on the multimodal file path, authenticated CORS preflight,
embedding-client shutdown, forged client `tool` history, and per-policy
retrieval isolation. The integration reviewer found **0 BLOCKING + 3 MAJOR
+ 2 MINOR**: a frontier identity contradiction, missing one-shot
cross-component drop-to-RAG evidence, stale manifest evidence wording, and
residual deployment-specific terminology.

The current worktree counts escaped text, wrappers, separators, and notes in
one F2 budget; permits CORS preflight through auth and rate limiting; rejects
client-authored tool roles; closes all core resources robustly; filters the
embedding cache by policy; uses a provider-neutral frontier prompt; and
removes the residual deployment wording. Seven focused regressions were added.
The all-in-one drop-to-RAG test is explicitly pending rather than claimed as
closed. Gate A now passes 1765 tests, with the same two skips, one expected
failure, and 30 disclosed marker warnings. R6 integration and R7 correctness re-reviews passed with no remaining BLOCKING or MAJOR findings in scope.

## BLOCKING status

| Issue | Status | Remaining closure |
|---|---|---|
| B1 accessible repository | PENDING | project owner confirms public access; publish only the final sanitized SHA and verify logged out |
| B2 clean-clone completeness | WORKTREE COMPLETE, CLONE PENDING | Manifest paths now exist and Docker runtime scripts are included; final candidate SHA and fresh clone remain required |
| B3 real GPT-5.6 | CODE + UNIT TEST CLOSED, LIVE PENDING | Client propagates the provider-returned model; focused suite passes 28 tests; funded live result remains required |
| B4 video and primary task evidence | HUMAN PENDING | Record/upload the public sub-three-minute video and supply the primary build-task `/feedback` ID |
| B5 Build Week delta | CLOSED IN CANDIDATE | README contains dated, commit-linked pre-period versus Build Week ledger |
| B6 unsupported redaction claim | REMEDIATED, FINAL SCAN PENDING | README, architecture, demo, and provider summary now describe explicit opt-in without automatic outbound redaction |
| B7 public-set sanitization | PASS | The exact 242-file staged tree has 0 hard-privacy hits, 0 missing paths, 0 broken local links, and 0 Gitleaks findings. Generic security identifiers were manually adjudicated under the owner-authorized exception; synthetic credential fixtures use narrow inline scanner annotations. |

## MAJOR status

| Issue | Status | Evidence or defer |
|---|---|---|
| M1 public 50-case skeleton | CLOSED IN CANDIDATE | Test/module docs and README explicitly identify it as skipped, pending coverage |
| M2 second-provider claim | REMEDIATED, FINAL SCAN PENDING | Public narrative now attributes 7/7 only to MiniMax-M3 and marks the second provider pending |
| M3 stale SDD | DOCS RECONCILED, REVIEW PENDING | Header points to v0.7/`a35fecb`; nonexistent SHA removed; no final PASS claimed |
| M4 static gates | CLOSED | export order fixed; obsolete suppressions removed; Ruff and mypy pass |
| M5 hanging live suite | CODE CLOSED, LIVE PENDING | SDK retries disabled, connect/read/write/pool bounds added, transport-only skips, live modules marked, helper calls use one 20-second attempt |
| M6 Bandit mediums | DEFERRED WITH TRIAGE | 13 medium findings are enumerated below; no broad suppression added |
| M7 blocking disk I/O | DEFERRED | Cross-component async/durability refactor is unsafe in the submission window; no event-loop performance claim is made |
| M8 moving image tags | BLOCKED | Exact tested tags/digests unavailable while Docker daemon is stopped; values must not be invented |
| M9 hosted/no-rebuild path | DEFERRED | Apps for Your Life uses the public video as no-rebuild evaluation; URL and tested platform table remain human/external work |
| M10 governance | DEFERRED | Minimal governance files await a public-safe reporting route; code of conduct is post-submission |
| M11 marker warnings | DEFERRED | 30 known warnings are disclosed; full suite passes and no warning-free claim is made |
| M12 stale evaluation strategy | CLOSED IN CANDIDATE | Dataset, classifier target, dated 7/7 evidence, pending 50-case path, and pending second provider are separated |
| M13 public drop-to-RAG wiring | SEAMS VERIFIED; CROSS-COMPONENT + CONTAINER PENDING | Focused tests verify ingestion, `vault_chunks`, chunk-text retrieval, tool quarantine, and Compose independently. A single real cross-component test and live container proof remain pending; this is not claimed closed. |

## M6 medium-finding triage

Code changes are deferred; each finding requires focused review rather
than a broad scanner annotation.

| Path:line | Class | Current disposition |
|---|---|---|
| `hermes/config.py:484` | all-interface default | Review direct-run contract; container mapping is separate |
| `hermes/config.py:914` | all-interface default | Review direct-run contract; container mapping is separate |
| `hermes/memory/collections.py:363` | dynamic SQL fragment | Confirm fragment derives only from fixed internal choices |
| `hermes/memory/db.py:1405` | dynamic SQL fragment | Confirm fragment derives only from fixed internal choices |
| `hermes/memory/db.py:1459` | dynamic SQL fragment | Confirm fragment derives only from fixed internal choices |
| `hermes/memory/db.py:3047` | dynamic SQL fragment | Confirm fragment derives only from fixed internal choices |
| `hermes/memory/db.py:3111` | dynamic SQL fragment | Confirm fragment derives only from fixed internal choices |
| `hermes/memory/ocr_pending_repo.py:239` | dynamic SQL fragment | Confirm fragment derives only from fixed internal choices |
| `hermes/memory/ocr_pending_repo.py:291` | dynamic SQL fragment | Confirm fragment derives only from fixed internal choices |
| `hermes/tools/agent_reach.py:120` | deterministic temporary path | Move to per-invocation directory after submission |
| `hermes/tools/agent_reach.py:178` | deterministic temporary path | Move to per-invocation directory after submission |
| `hermes/tools/agent_reach.py:209` | deterministic temporary path | Move to per-invocation directory after submission |
| `hermes/tools/agent_reach.py:306` | deterministic temporary path | Move to per-invocation directory after submission |

## Human and external inputs required

1. Confirm the intended repository is public.
2. Provide funded external access for the text-only GPT-5.6 Sol smoke.
3. Start Docker and exercise exact image versions before pinning.
4. Record and upload the public video.
5. Supply `/feedback` from the primary build task.

No submission-ready claim is permitted until every PENDING gate above
is replaced by evidence tied to the final candidate SHA.
## Public alias and on-demand F2 closure (2026-07-17)

The public OpenAI-compatible contract now advertises `oroimen-agent`,
`oroimen-agent-fast`, and the opt-in `oroimen-agent-frontier`. Legacy
`hermes-agent*` inputs are accepted only as hidden transition aliases; unknown
models are rejected before any database mutation, and invalid/reserved/colliding
model overrides fail validation.

The provider-backed F2 workflow is strictly `workflow_dispatch`: no PR, nightly,
artifact upload, or issue-writing trigger. It fails on missing credentials,
skipped cases, or an incomplete 14-case collection; verdict majority is computed
across three independent generations. Public-corpus Ollama tests are marked
`network` + `slow` and require explicit manual flags. Raw prompts and model
responses are omitted from retained output.

Final evidence for this closure: 1765 unit tests passed, 2 skipped, 1 expected
failure; mypy passed 91 modules; both fresh adversarial re-reviews returned PASS
with no remaining BLOCKING, MAJOR, or MINOR findings in scope. Live provider calls
were deliberately not executed.
## Public CI follow-up (2026-07-17)

The public follow-up adds deterministic CI for pull requests and main without
changing the manual-only provider policy. Static checks, unit tests, local
integration tests, Compose validation, and Gitleaks run without repository
secrets beyond GitHub's read-only workflow token. External provider tests remain
workflow_dispatch only.

Local baseline before publication:

- Serial unit gate: 1766 passed, 2 skipped, 1 expected failure, 73 deselected (206.91 s).
- Serial dry/integration/e2e gate: 221 passed, 7 skipped, 1 expected failure, 18 deselected (31.03 s).
- Ruff and mypy: PASS.
- Compose configuration: PASS.
- CI lock dry-run: 96 Python 3.13/Linux packages resolved with required hashes.
- F2 workflow: restricted to `main` and the `f2-live` environment; environment
  protection and secrets remain a GitHub-side setup gate before any live run.
- Adversarial CI review: PASS after R3, with 0 BLOCKING and 0 MAJOR findings.
- Gitleaks and hard-privacy scans: PASS — 0 findings; generic CI security identifiers were manually adjudicated.

## Exact public tree expansion (246 files)

The following paths are the complete allowlist-expanded public tree used for the
clean candidate export:

```text
.dockerignore
.env.example
.gitattributes
.github/actions/setup-python-uv/action.yml
.github/workflows/ci.yml
.github/workflows/f2-tests.yml
.gitignore
BUILD_PROCESS.md
docker-compose.yml
Dockerfile
docs/ARCHITECTURE.md
docs/CI_TEST_PARALLELISM.md
docs/DEMO_SCRIPT.md
docs/EVAL_STRATEGY.md
docs/SECURITY_TESTING.md
drop/.gitkeep
FINAL_VERIFICATION.md
hermes/__init__.py
hermes/__main__.py
hermes/agent/__init__.py
hermes/agent/loop.py
hermes/backup.py
hermes/config.py
hermes/handlers/__init__.py
hermes/handlers/chunker.py
hermes/handlers/commands.py
hermes/handlers/messages.py
hermes/handlers/notifications.py
hermes/handlers/ocr_commands.py
hermes/health.py
hermes/jobs/__init__.py
hermes/jobs/cost.py
hermes/jobs/exceptions.py
hermes/jobs/models.py
hermes/jobs/prompts.py
hermes/jobs/recovery.py
hermes/jobs/scheduler.py
hermes/jobs/service.py
hermes/llm/__init__.py
hermes/llm/breaker.py
hermes/llm/chatgpt5_6.py
hermes/llm/ocr.py
hermes/llm/ollama.py
hermes/llm/router.py
hermes/logging_setup.py
hermes/memory/__init__.py
hermes/memory/chunker.py
hermes/memory/collections.py
hermes/memory/db.py
hermes/memory/drop_watcher.py
hermes/memory/edge_coordinator.py
hermes/memory/embedder.py
hermes/memory/extractors/__init__.py
hermes/memory/extractors/openpyxl_extractor.py
hermes/memory/extractors/plain.py
hermes/memory/extractors/pymupdf_extractor.py
hermes/memory/extractors/python_docx.py
hermes/memory/extractors/tesseract.py
hermes/memory/facts.py
hermes/memory/file_id.py
hermes/memory/ingest_router.py
hermes/memory/ocr_decision.py
hermes/memory/ocr_pending_repo.py
hermes/memory/seed.py
hermes/memory/sleep_cycle.py
hermes/memory/vault.py
hermes/observability/__init__.py
hermes/observability/influxdb.py
hermes/receivers/__init__.py
hermes/receivers/auth.py
hermes/receivers/base.py
hermes/receivers/http_api.py
hermes/receivers/jobs_api.py
hermes/receivers/ocr_api.py
hermes/receivers/polling.py
hermes/scheduler.py
hermes/security/__init__.py
hermes/security/classifier.py
hermes/security/egress.py
hermes/services/embed_vault.py
hermes/services/embedding_router.py
hermes/services/embeddings.py
hermes/services/search/__init__.py
hermes/services/search/budget.py
hermes/services/search/errors.py
hermes/services/search/exa.py
hermes/services/search/protocol.py
hermes/services/search/resilience.py
hermes/services/search/router.py
hermes/services/search/searxng.py
hermes/services/search/tavily.py
hermes/shutdown.py
hermes/stt/__init__.py
hermes/stt/gemini.py
hermes/stt/queue.py
hermes/telemetry.py
hermes/tools/__init__.py
hermes/tools/agent_reach.py
hermes/tools/builtin.py
hermes/tools/collections.py
hermes/tools/registry.py
hermes/tools/scripts/__init__.py
hermes/tools/scripts/rss_read.py
hermes/tools/search_files.py
hermes/tools/security.py
hermes/tools/web_search.py
hermes/util/__init__.py
hermes/util/paths.py
LICENSE
mypy.ini
pytest.ini
README.md
requirements.txt
requirements-dev.txt
requirements-ci.lock
ruff.toml
scripts/__init__.py
scripts/pr_review.py
scripts/setup_agent_reach.py
SUBMISSION_MANIFEST.md
tests/__init__.py
tests/conftest.py
tests/dry/test_hermes_imports.py
tests/e2e/__init__.py
tests/e2e/_helpers.py
tests/e2e/conftest.py
tests/e2e/test_cache_invariants.py
tests/e2e/test_chatgpt5_6_live.py
tests/e2e/test_multi_tier.py
tests/e2e/test_ollama_provider.py
tests/e2e/test_phase1_smoke.py
tests/e2e/test_rag_injection.py
tests/e2e/test_rag_injection_file_content.py
tests/e2e/test_real_llm_validation.py
tests/e2e/test_sprint_19_pipeline.py
tests/fixtures/datasets/deepset_prompt_injections.jsonl
tests/integration/__init__.py
tests/integration/conftest.py
tests/integration/test_concurrency.py
tests/integration/test_drop_watcher_m6_pipeline.py
tests/integration/test_files_e2e.py
tests/integration/test_jobs_api.py
tests/integration/test_jobs_cost_drift.py
tests/integration/test_jobs_recovery_real.py
tests/integration/test_jobs_service_phases.py
tests/integration/test_migration_timing.py
tests/integration/test_resilience.py
tests/integration/test_text_flow.py
tests/integration/test_voice_flow.py
tests/unit/__init__.py
tests/unit/security/__init__.py
tests/unit/security/test_egress.py
tests/unit/test_agent_loop.py
tests/unit/test_agent_loop_file_refs.py
tests/unit/test_agent_reach.py
tests/unit/test_backfill_logic.py
tests/unit/test_backup.py
tests/unit/test_breaker.py
tests/unit/test_builtin.py
tests/unit/test_chatgpt5_6_provider.py
tests/unit/test_ci_exit_status.py
tests/unit/test_chunker.py
tests/unit/test_classifier_meta.py
tests/unit/test_collections.py
tests/unit/test_collections_api.py
tests/unit/test_collections_bugs.py
tests/unit/test_collections_tools.py
tests/unit/test_collections_tools_schemas.py
tests/unit/test_commands.py
tests/unit/test_config.py
tests/unit/test_db.py
tests/unit/test_db_files.py
tests/unit/test_db_lifecycle.py
tests/unit/test_db_memory_facts.py
tests/unit/test_db_upsert_embedding.py
tests/unit/test_drop_watcher.py
tests/unit/test_drop_watcher_extraction.py
tests/unit/test_edge_coordinator.py
tests/unit/test_embed_vault.py
tests/unit/test_embedder.py
tests/unit/test_embedding_router.py
tests/unit/test_embeddings.py
tests/unit/test_embeddings_integration.py
tests/unit/test_extractors.py
tests/unit/test_f2_public_datasets.py
tests/unit/test_facts_retrieval.py
tests/unit/test_file_id.py
tests/unit/test_file_ref_budget.py
tests/unit/test_health.py
tests/unit/test_health_checker.py
tests/unit/test_health_notifier.py
tests/unit/test_http_api.py
tests/unit/test_http_api_dedup.py
tests/unit/test_http_api_refs.py
tests/unit/test_ingest_router.py
tests/unit/test_ingest_router_m6_extended.py
tests/unit/test_jobs_api_auth.py
tests/unit/test_jobs_cost.py
tests/unit/test_jobs_cost_estimate.py
tests/unit/test_jobs_models.py
tests/unit/test_jobs_notifier.py
tests/unit/test_jobs_prompts.py
tests/unit/test_jobs_recovery.py
tests/unit/test_jobs_threadpool.py
tests/unit/test_jobs_time_format.py
tests/unit/test_local_vision_ocr.py
tests/unit/test_m6_reconcile.py
tests/unit/test_main.py
tests/unit/test_memory_facts.py
tests/unit/test_messages.py
tests/unit/test_ocr_commands.py
tests/unit/test_ocr_decision.py
tests/unit/test_ocr_pending_repo_slice4c.py
tests/unit/test_ocr_provider.py
tests/unit/test_ollama_provider.py
tests/unit/test_para_seeding.py
tests/unit/test_path_posix.py
tests/unit/test_pr_review.py
tests/unit/test_rate_limit.py
tests/unit/test_registry.py
tests/unit/test_router.py
tests/unit/test_run_stream.py
tests/unit/test_scheduler.py
tests/unit/test_search_budget.py
tests/unit/test_search_errors.py
tests/unit/test_search_exa.py
tests/unit/test_search_files_tool.py
tests/unit/test_search_logging.py
tests/unit/test_search_protocol.py
tests/unit/test_search_resilience.py
tests/unit/test_search_router.py
tests/unit/test_search_searxng.py
tests/unit/test_search_settings.py
tests/unit/test_search_tavily.py
tests/unit/test_search_wireup.py
tests/unit/test_security.py
tests/unit/test_seed.py
tests/unit/test_sleep_cycle.py
tests/unit/test_sleep_cycle_atomicity.py
tests/unit/test_streaming.py
tests/unit/test_stt_gemini.py
tests/unit/test_stt_queue.py
tests/unit/test_telemetry.py
tests/unit/test_v25_migration.py
tests/unit/test_vault.py
tests/unit/test_web_search_tool.py
```

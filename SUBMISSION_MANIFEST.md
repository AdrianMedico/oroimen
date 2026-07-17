# Oroimen Submission Manifest

**Status:** candidate manifest after adversarial PASS; external release gates pending; not yet a release record
**Baseline:** `8003dc9` plus the reviewed Build Week fix bundle
**Rule:** only paths listed below may enter the public submission commit; directory rows are recursive candidate scope and must be expanded before the final SHA
**Final SHA:** PENDING
**Final public-manifest scan:** PASS — exact 246-file tree; 0 hard-privacy hits and 0 Gitleaks findings
**Fresh-clone Gate A:** PENDING

## Required public paths

| Path | Purpose | Tracking state | Final scan |
|---|---|---|---|
| `LICENSE` | AGPLv3 license text | tracked at `8003dc9` | PASS |
| `README.md` | judge path, evidence, setup, Build Week delta | candidate change | PASS |
| `BUILD_PROCESS.md` | dated AI-assisted build process | tracked at `8003dc9` | PASS |
| `Dockerfile` | application image build | tracked before baseline | PASS |
| `docker-compose.yml` | optional hands-on judge path | tracked before baseline | PASS |
| `.env.example` | documented optional configuration | tracked before baseline | PASS |
| `.gitattributes` | enforces portable LF line endings across platforms | tracked before baseline | PASS |
| `.gitignore` | prevents local secrets, environments, and generated artifacts from being committed | tracked before baseline | PASS |
| `.dockerignore` | excludes local secrets and development artifacts from the image build context | tracked before baseline | PASS |
| `.github/actions/setup-python-uv/action.yml` | reproducible Python/uv bootstrap shared by public workflows | candidate change | PASS |
| `.github/workflows/ci.yml` | public pull-request and main-branch quality, deterministic test, Compose, and secret-scan gates | candidate change | PASS |
| `.github/workflows/f2-tests.yml` | strictly manual provider-backed prompt-injection validation | candidate change | PASS |
| `drop/.gitkeep` | deterministic writable drop-directory bind source for the public Compose path | candidate new file | PASS |
| `requirements.txt` | bounded Python dependency ranges used by Docker; exact lock pending Gate C | tracked before baseline | PASS |
| `requirements-dev.txt` | developer and verification dependencies for the documented Gate A commands | candidate comments | PASS |
| `requirements-ci.lock` | Python 3.13/Linux CI dependency lock with distribution hashes | candidate new file | PASS |
| `mypy.ini` | strict type-check configuration used by Gate A | tracked before baseline | PASS |
| `ruff.toml` | lint and formatting configuration used by Gate A | tracked before baseline | PASS |
| `pytest.ini` | test markers and defaults | tracked before baseline | PASS |
| `scripts/__init__.py` | runtime package marker required by startup import | tracked before baseline | PASS |
| `scripts/setup_agent_reach.py` | runtime setup imported by `hermes.__main__` | tracked before baseline | PASS |
| `scripts/pr_review.py` | module imported by the recursive public unit suite | tracked before baseline | PASS |
| `hermes/` | application package | tracked; candidate provider/config fixes | PASS |
| `tests/` | public automated test suite | tracked; candidate live-bound and sanitization fixes | PASS |
| `tests/conftest.py` | shared deterministic test setup | candidate change | PASS |
| `tests/unit/test_chatgpt5_6_provider.py` | frontier provider contract | tracked at `8003dc9` | PASS |
| `tests/unit/test_ci_exit_status.py` | regression ensuring GitHub Actions preserves pytest failures | candidate new file | PASS |
| `tests/unit/test_f2_public_datasets.py` | public-dataset classifier evidence | tracked at `b9d39cf` | PASS |
| `tests/unit/test_local_vision_ocr.py` | local vision provider evidence | tracked at `a35fecb` | PASS |
| `tests/e2e/conftest.py` | bounded opt-in live-provider fixtures | candidate change | PASS |
| `tests/e2e/test_real_llm_validation.py` | seven self-crafted live F2 cases | candidate live-bound change | PASS |
| `tests/e2e/test_chatgpt5_6_live.py` | bounded router-level GPT-5.6 smoke | candidate new file | PASS |
| `tests/e2e/test_rag_injection_file_content.py` | file-content injection path | tracked before baseline | PASS |
| `docs/ARCHITECTURE.md` | judge-readable system architecture | tracked at `8003dc9` | PASS |
| `docs/CI_TEST_PARALLELISM.md` | public/local pytest worker policy and evidence-based promotion rule | candidate new file | PASS |
| `docs/DEMO_SCRIPT.md` | truthful sub-three-minute recording script | candidate change | PASS |
| `docs/EVAL_STRATEGY.md` | measured versus pending evaluation state | candidate change | PASS |
| `docs/SECURITY_TESTING.md` | manual-only red-team policy, privacy rules, and future corpus plan | candidate new file | PASS |



| `FINAL_VERIFICATION.md` | commands, results, defer decisions, and pending gates | candidate new file | PASS |
| `SUBMISSION_MANIFEST.md` | allowlist for the public artifact | candidate new file | PASS |

## Pending external evidence

The live GPT-5.6 report is not part of the required allowlist until a real
run exists. When created, add its exact dated path only after sanitization.

- `docs/TEST_REPORTS/gpt_5_6_smoke_<date>.md` — not yet created.

## Explicitly excluded

These remain local/private and must not be staged for the public artifact:

- Root audit, issue, scoring, and fix-planning reports.
- Migration, sanitization, and deep-research working papers.
- Deployment-specific compose files, dashboards, host configuration, and deploy scripts.
- Local demo audio, voice models, caches, reports not listed above, and temporary files.
- Any file that fails the final public-manifest scan.

## Closure procedure

1. Resolve or explicitly defer every pending external-evidence path; never use a placeholder filename as a required artifact.
2. Expand directory rows to the exact staged file list in `FINAL_VERIFICATION.md`.
3. Confirm every included path with `git ls-files --error-unmatch`.
4. Run the repository-prescribed scan over this allowlist and the staged diff.
5. Create the sanitized candidate commit.
6. Clone that exact SHA into an empty directory and run Gate A.
7. Replace the header fields and every remaining `PENDING` only from recorded evidence.

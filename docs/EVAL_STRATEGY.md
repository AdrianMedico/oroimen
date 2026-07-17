# Eval Strategy — Rounding the Evaluation

> **Concept source**: LangChain talk (2026-07-16), "rounding eval suite"
> + Zalando 4 eval types for agents.
>
> **Status**: reviewed evaluation record; final candidate SHA remains pending
> **Owner**: project owner (vision) + Mavis (scaffolding)
> **For**: post-Build-Week hardening (no immediate scope)

## 0. LangChain's 4 eval types (for AI assistants / agents)

The LangChain talk names **four distinct dimensions** of evaluation
coverage. A system with only one or two of these is brittle; a system
with all four is "rounded":

1. **OSS eval** — open-source eval datasets, public + curated.
   Examples: HackAPrompt, PromptBench, HotpotQA, MMLU, HumanEval.
   Use for: security regression tests, capability benchmarks.

2. **End-to-end scenario evals** — full user journeys, stress-tested
   for complex realistic flows. Use for: "does the whole thing
   work together?"

3. **Sub-piece optimization** — measure isolated pieces of the
   architecture in their own right. Use for: "where is the
   bottleneck?" "does the F2 fix layer-by-layer still work?"

4. **Diverse eval data** — variations of data size and domain that
   reflect what the engine processes in the real world. Use for:
   "does it generalize across file types, languages, sizes?"

---

## 1. The concept (from the talk)

A "rounding eval suite" covers **four orthogonal dimensions** so
that issues caught by one lens are not invisible to the others:

1. **OSS eval** — public + curated datasets
2. **End-to-end scenario evals** — full user journeys, stress-tested
3. **Sub-piece optimization** — measure isolated pieces of the
   architecture
4. **Diverse eval data** — variations of size and domain

A system with only one of these dimensions is brittle. A system with
all four is "rounded".

---

## 2. How it applies to Oroimen

### 2.1 OSS eval (curated datasets)

**Goal**: leverage public, peer-reviewed eval datasets to validate
that known attacks / benchmarks are handled correctly.

| Dataset | What it tests | How to integrate |
|---|---|---|
| **HackAPrompt** (AIcrowd) | 600K+ prompt injection attacks | `tests/e2e/test_hackaprompt_f2.py` — run 50-100 known attacks against F2 fix |
| **Prompt-Injection-Bench** (Princeton) | Academic coverage of injection variants | Same pattern as HackAPrompt |
| **HotpotQA** | Multi-hop reasoning | `tests/e2e/test_hotpotqa_retrieval.py` — multi-hop questions across vault |
| **MMLU** | Multi-domain knowledge | `tests/e2e/test_mmlu_baseline.py` — verify LLM doesn't regress |
| **HumanEval** | Code | `tests/e2e/test_humaneval_code.py` — code-related queries |
| **FEVER** | Fact verification | Future — claim + evidence retrieval |

### 2.2 End-to-end scenario evals

**Goal**: a real user journey, top to bottom, with realistic friction.

Candidate scenarios for Oroimen:

| # | Scenario | What it tests |
|---|---|---|
| 1 | "User drops a 50-page legal PDF, asks about a specific clause" | ingest + chunk + embed + retrieve + generate |
| 2 | "User drops a CV with prompt injection in the footer" | F2 fix (security) |
| 3 | "User drops a supported document into the public drop folder" | extraction + local embedding + `vault_chunks` retrieval |
| 4 | "User explicitly selects the frontier model for a task" | explicit opt-in route; only the selected conversation is sent to the provider |
| 5 | "User's vault is 5000 files, asks a cross-doc question" | scale, multi-collection retrieval |
| 6 | "User drops a file in a non-Latin language" | multilingual + OCR |
| 7 | "Two users share the same deployment, query at the same time" | concurrency |
| 8 | "User accidentally drops a 1 GB file" | size limits + DoS defense |

These would be **integration-level e2e tests** in `tests/e2e/`,
each scenario = 1-3 tests, total 10-20 scenarios.

### 2.3 Sub-piece optimization

**Goal**: each component in the architecture has its own micro-bench
and unit-style eval, independent of the full system.

Candidate sub-pieces:

| Component | Eval questions |
|---|---|
| **F2 fix (3 layers)** | Per-layer test: XML escape catches X% of payloads, wrap format valid, system rule triggers on file content. |
| **LLM router** | Per-provider: latency, cost, fallback behavior. Circuit breaker: opens at N fails, closes at reset. |
| **Embeddings** | Per-tier: latency (p50/p95/p99), quality (top-3 jaccard), cost. Per-policy: chat_rag vs vault_ingest vs facts routing. |
| **OCR adapters** | Unit-level accuracy and size-limit checks; local-vision runtime wiring is outside the current public Compose path. |
| **Drop watcher** | Polling vs inotify, file-type detection, idempotency, error recovery. |
| **Vault** | SQLite WAL throughput, concurrent reads, backup/restore. |
| **Memory** | F1 facts recall, fact-consolidation, sleep cycle (S10, deferred). |
| **Tools (search_files, agent_reach, etc.)** | Per-tool: success rate, latency, edge cases. |

These would be a mix of **unit tests** (already partially covered)
and **micro-benchmarks** (e.g., `tests/bench/` or `benchmarks/`).

### 2.4 Diverse eval data

**Goal**: vary the inputs across size, domain, and language so the
evals reflect real-world conditions, not just our test fixtures.

Candidate dimensions:

| Dimension | Variants we should test |
|---|---|
| **File type** | PDF, DOCX, XLSX, PPTX, TXT, MD, PNG, JPG, WEBP, HTML, EML |
| **File size** | 1 KB, 100 KB, 1 MB, 10 MB, 50 MB (at the image cap) |
| **Language** | EN, ES, DE, FR, ZH, JA, AR (we already have multilingual patterns) |
| **Document category** | Legal, medical, financial, technical, casual, creative |
| **Query type** | Factual, analytical, creative, multi-doc, code, math |
| **Image quality** | Clean scan, low DPI, handwritten, photo-of-screen, rotated |
| **User profile** | Casual user, expert, multi-lingual, accessibility needs |
| **Network conditions** | Slow, lossy, intermittent, fast |

**Public datasets to leverage** (from the LangChain talk + community):

| Dataset | Source | Use case |
|---|---|---|
| **HackAPrompt** | AIcrowd | 600K prompt injection attacks → F2 fix regression |
| **Garak** | NVIDIA | LLM vulnerability scanner (CLI tool) |
| **PyRIT** | Microsoft | Risk Identification Tool, automated red-teaming |
| **Natural Questions** | Google | Open-domain QA |
| **FEVER** | fact verification | Claims + evidence |
| **GSM8K** | math reasoning | Numerical queries |

(Public datasets are covered in §2.1 — repeated here for visibility.)

---

## 3. Current state vs target

| Dimension | Current evidence | Next target |
|---|---|---|
| **Public dataset loaded** | 662 deepset examples (263 injection, 399 benign) in `tests/unit/test_f2_public_datasets.py` | Add further public corpora only after the submission baseline is stable |
| **Local classifier benchmark** | Test implemented over all 662 examples in `tests/unit/test_f2_public_datasets.py`; target ≥95% recall per class; no dated final-candidate metric recorded | Record the final candidate result |
| **Self-crafted live chat** | 7/7 passed against MiniMax-M3 on 2026-07-16; executable evidence: `tests/e2e/test_real_llm_validation.py` | Keep as the bounded live regression set |
| **Public live chat** | 50-case stratified skeleton is present but not wired | Implement the opt-in production chat path after submission |
| **Second-provider baseline** | Pending; the audited fixture did not return within its bound | Re-run only with bounded network fixtures |
| **Offline suite** | Audited post-R5 remediation Gate A: 1753 passed, 5 skipped, 1 expected failure | Preserve the deterministic command and disclose 30 marker warnings |

**Gap**: public classifier evaluation is implemented; public live
chat-path validation remains the largest evaluation gap.

---

## 4. Action plan (post-hackathon, ranked)

### 4.1 Quick wins (1-2 days each)

1. **HackAPrompt regression test** (§2.1 OSS eval): extend
   `tests/e2e/` with a `test_hackaprompt_dataset.py` that runs
   50-100 known attacks from HackAPrompt against the F2 fix.
   Update the README claim from "7/7" to "100+ HackAPrompt
   attacks + 7/7 baseline".

2. **Garak integration** (§2.1 OSS eval): add `garak` as a dev
   dependency, run it in CI against the local Ollama chain,
   report findings.

3. **Bench suite for the F2 layers** (§2.3 Sub-piece): micro-bench
   that times each layer individually (XML escape, wrap, system
   rule injection) to find the slowest component.

4. **File-type coverage table test** (§2.4 Diverse eval): parametrized
   test that runs the full RAG flow on PDF, DOCX, XLSX, MD, TXT.
   Confirms the extractor chain handles all types.

### 4.2 Medium-effort (1 week each)

5. **Scenario eval suite** (§2.2 E2E): implement the 8 scenarios
   as `tests/e2e/scenarios/`. Each scenario = 1 pytest file with
   1-3 tests, 10-20 total.

6. **HotpotQA retrieval eval** (§2.1 OSS eval): integrate the
   HotpotQA dataset to test multi-hop reasoning across the
   vault. Measures retrieval quality, not just generation.

7. **Multilingual regression** (§2.4 Diverse eval): extend the
   e2e suite to include EN, ES, DE, FR queries. Verify the F2 fix
   works in non-English.

### 4.3 Long-term (1+ month)

8. **PyRIT integration**: Microsoft's automated red-teaming. Set up
   as a scheduled CI job that reports new attacks weekly.

9. **Live eval tracking**: dashboard that runs the eval suite daily
   against the dev deployment, tracks regressions over time.

10. **User-driven evals**: collect real queries (anonymized) from
    the private deployment, use them as regression tests.

---

## 5. How this fits the Build Week pitch

For the demo on 2026-07-21, we can leverage the "rounding eval
suite" concept in the README and the demo video:

> "The deterministic post-R5 remediation unit gate passes 1753 tests. Seven self-crafted
> F2 cases passed against MiniMax-M3 on the dated live run. The public
> classifier target, second-provider baseline, and GPT-5.6 live smoke
> are reported separately so pending work is not presented as measured
> evidence."

This is the "AI-assisted + security-first + tested" story.

---

## 6. Update discipline

When the eval strategy evolves:
- Add a new "action item" to §4 with status (planned, in-progress, done)
- Cross-link from `FINAL_VERIFICATION.md` when a result becomes submission evidence
- Update §3 "current state" when new evals are added
- The eval suite lives in `tests/`; this doc describes the strategy,
  not the implementation

This doc freezes at the Build Week end. Post-hackathon, a new
"EVAL_STRATEGY.md v0.2" with the new public-dataset integrations.

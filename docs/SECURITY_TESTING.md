# Security Testing — Prompt Injection & F2 Fix

> **Goal**: validate the F2 prompt-injection fix against public,
> peer-reviewed attack datasets + automated red-teaming tools.
> Make the "X/X blocked" claim credible and track regressions.
>
> **Status**: DRAFT v0.1 (2026-07-16)
> **Owner**: Adrian (vision) + Mavis (scaffolding)
> **For**: post-Build Week hardening of the F2 fix
> **Context**: discovered at LangChain talk 2026-07-16 (recommend
> OSS eval datasets for security)

---

## Execution policy

The prompt-injection/red-team suite is **manual-only**:

- no `pull_request` trigger;
- no nightly or scheduled trigger;
- no automatic execution against a personal vault or production database;
- use public/synthetic fixtures and a disposable database;
- do not print or upload raw prompts, model responses, or private document content;
- retain only case IDs, aggregate counts, classifier verdicts, model/version, and run metadata.

Cheap deterministic unit tests for escaping, budgets, and message contracts remain
part of normal CI. Corpus evaluation and all live-model probes run only through
`.github/workflows/f2-tests.yml` via `workflow_dispatch` or an explicit local command.

## 1. The 4 main tools / datasets

### 1.1 HackAPrompt (AIcrowd)

- **What**: 600K+ prompt injection attacks, many document-based
- **Source**: AIcrowd competition, sponsored by
- **Coverage**: instruction-hijack, role-confusion, payload-injection
- **Why for us**: has **document-based attacks** specifically — exactly
  what the F2 fix protects against. This is the highest-priority
  dataset for us.
- **Integration**: `tests/e2e/test_hackaprompt_f2.py` — pick 50-100
  known attacks, run against the F2 fix, assert blocked
- **Cost**: free, public dataset
- **URL**: https://www.aicrowd.com/challenges/hackaprompt

### 1.2 Prompt-Injection-Bench (Princeton)

- **What**: academic, comprehensive benchmark for prompt injection
- **Source**: Princeton University researchers
- **Coverage**: ~30 attack categories, both naive and adaptive
- **Why for us**: academic rigor, peer-reviewed, baseline comparison
- **Integration**: `tests/e2e/test_princeton_pib.py` — same pattern
  as HackAPrompt
- **Cost**: free, public dataset
- **URL**: https://github.com/Princeton-SysML/Jailbreak_LLM

### 1.3 Garak (NVIDIA)

- **What**: LLM vulnerability scanner, runs many probes automatically
- **Source**: NVIDIA
- **Coverage**: 100+ probes, prompt injection + jailbreaks + hallucination +
  data leakage + many more
- **Why for us**: easy to integrate, runs against any OpenAI-compat
  endpoint (including our local Ollama), comprehensive coverage
- **Integration**: `pip install garak`, then `garak --model ollama
  --model_name qwen2.5:7b` — runs all probes against our chain
- **Cost**: free, open-source tool
- **URL**: https://github.com/NVIDIA/garak

### 1.4 PyRIT (Microsoft)

- **What**: Python Risk Identification Tool, automated red-teaming
- **Source**: Microsoft
- **Coverage**: multi-turn attack scenarios, converters (encoding,
  obfuscation), scoring
- **Why for us**: more enterprise-grade, supports complex multi-step
  attacks that simple dataset tests miss
- **Integration**: `pip install pyrit`, then write attack scenarios
  programmatically
- **Cost**: free, open-source
- **URL**: https://github.com/Azure/PyRIT

### 1.5 Optional / nice-to-have

| Tool | Type | Use case |
|---|---|---|
| **Tensor Trust** (Lakera) | Interactive web-based CTF | Learning, training the team |
| **DeepTeam** | Python library | Custom red-teaming, less mature |
| **Gandalf** (Lakera) | Web-based CTF | Beginner-friendly intro |

---

## 2. Current state vs target

| Tool | Today | Target |
|---|---|---|
| HackAPrompt | 0 attacks integrated | 50-100 curated attacks in the manual regression suite |
| Prompt-Injection-Bench | 0 attacks integrated | 30 categories in `test_princeton_pib.py` |
| Garak | Not installed | Optional tool, run manually against a disposable local target |
| PyRIT | Not installed | Optional tool, run manually before selected releases |

**Current baseline**: deterministic wrapping checks over the public dataset live in
`tests/unit/test_f2_public_datasets.py`; classifier and provider-backed coverage remain
manual-only. The live-model suite is deliberately
small, manually triggered F2 suite. Broader end-to-end corpus coverage is pending.

---

## 3. Action plan (post-Build Week)

### 3.1 Quick wins (1-2 days each)

1. **HackAPrompt regression test** (1-2 days): download dataset,
   pick 50-100 document-based attacks, integrate as
   `tests/e2e/test_hackaprompt_f2.py`. Update README claim from
   "5/5 against M3" to "100+ HackAPrompt + Princeton baseline + 5/5
   against M3".

2. **Prompt-Injection-Bench baseline** (1 day): same pattern,
   fewer attacks but academic rigor.

3. **Garak integration** (1 day): add as dev dependency, run
   against a disposable local endpoint on demand. Report sanitized findings in
   `docs/SECURITY_AUDIT_GARAK.md`.

### 3.2 Medium-effort (1 week each)

4. **PyRIT custom scenarios** (1 week): write multi-turn attack
   scenarios that mimic real-world abuse patterns. Harder to
   bypass than single-prompt attacks.

5. **F2 fix regression suite** (1 week): consolidate all the above
   into a single `tests/e2e/test_f2_security_suite.py` that runs
   ALL the public datasets + custom scenarios in one pytest run.

### 3.3 Long-term (1+ month)

6. **Repeatable red-teaming** (1+ month): provide a manual PyRIT runbook
   for selected release candidates. Track new attack patterns and update the
   F2 fix as new bypasses are found.

7. **Bug bounty** (1+ month, if serious): open a public bug bounty
   for F2 bypasses. Crowdsourced security testing.

---

## 4. Integration pattern (for any of the above)

For each tool/dataset, the integration follows the same pattern:

```python
# tests/e2e/test_hackaprompt_f2.py (example)

import pytest
from pathlib import Path

# Load 50-100 HackAPrompt attacks (curated, document-based)
ATTACKS = load_hackaprompt_attacks(
    count=100,
    filter="document_based",
    path="datasets/hackaprompt_subset.json",
)

@pytest.mark.network
@pytest.mark.parametrize("attack", ATTACKS)
async def test_f2_blocks_hackaprompt_attack(attack, disposable_f2_harness):
    """Exercise the production wrapper + Layer 3 rule, never raw concatenation."""
    verdict = await disposable_f2_harness.evaluate(
        file_content=attack.file_content,
        user_question=attack.user_query,
    )
    assert verdict.blocked, (
        f"F2 bypassed by public case {attack.id}; raw content intentionally omitted"
    )
```

This pattern is reusable across all four tools. Each must use the disposable
production-path harness and remain explicitly marked as a manual network test.

---

## 5. What this is NOT

- **Not a one-time audit** — security is a continuous process.
- **Not a replacement for the F2 fix** — the F2 fix is the defense;
  these tests are how we measure the defense.
- **Not a substitute for human review** — automated tools catch
  known patterns. Novel bypasses need human review.
- **Not a hackathon deliverable** — this is post-Build Week work.

---

## 6. Open questions for when we start

1. **Cost**: Garak + PyRIT run against our local Ollama = $0.
   But if we want to test against MiniMax M3 too (more realistic),
   it's ~$0.001-0.01 per test run. Acceptable.

2. **Execution scope**: resolved — Garak/PyRIT and live corpus tests are
   manual-only. Deterministic security invariants remain in normal CI.

3. **Bypass handling**: when a tool finds a new bypass, what's the
   process? Open issue → fix F2 → add regression test → re-run.
   This needs a runbook.

4. **Tool combination**: do we run all 4 together? Some attacks
   overlap. We could run them as a coverage grid (tool × attack category)
   and dedupe results.

5. **Reporting**: how do we present the security claim to judges
   / users? "5/5 hand-crafted attacks" vs "100/100 HackAPrompt
   + 30/30 Princeton" vs "Garak ran X probes, Y failures"?

---

## 7. References (to read later)

- HackAPrompt: https://www.aicrowd.com/challenges/hackaprompt
- Prompt-Injection-Bench: https://github.com/Princeton-SysML/Jailbreak_LLM
- Garak: https://github.com/NVIDIA/garak
- PyRIT: https://github.com/Azure/PyRIT
- Tensor Trust: https://tensortrust.ai/
- DeepTeam: https://github.com/confident-ai/deepteam
- Gandalf: https://gandalf.lakera.ai/

---

## 8. Update discipline

When the security testing evolves:
- Add new tools to §1
- Update §2 with current state
- Update §3 with progress on each action item
- When a bypass is found, add to a "Known bypasses (and how we fixed)" section

This doc is post-Build Week. Status: DRAFT v0.1, pending execution.

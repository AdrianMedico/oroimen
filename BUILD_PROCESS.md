# BUILD_PROCESS.md — How Oroimen was built

> **Why this file exists**: the OpenAI Build Week asked us to be
> transparent about how AI helped. So we are. This is a factual
> breakdown of which decisions were AI-assisted, which were human,
> and which were a collaboration.
>
> **Status**: candidate factual record (2026-07-17); final human
> attribution review remains a submission gate.
> **Maintainer**: project owner (decisions) + AI assistants (drafting/review)

---

## The AI assistants involved

### Mavis (M3) — the daily companion

Mavis is the orchestrator agent in the Mavis Code workspace.
Powered by MiniMax-M3. Mavis has been involved in every sprint
since the project started, doing:

- **Code generation** — module-level scaffolding, test fixtures,
  helper functions
- **Test authoring** — unit tests, integration tests, F2
  injection test suite
- **TDD drafting** — most `docs/TDD_*.md` files are Mavis drafts
  with project owner's design input
- **Sprint retrospectives** — historical project retrospectives
  is Mavis's first pass
- **R1 reviews** — sub-agent verifier reviews, used to harden
  TDDs and implementation
- **Documentation** — module docstrings, README drafts, this file
- **Codebase archaeology** — the recent `docs/ARCHITECTURE.md`
  was a Mavis synthesis from walking the actual modules

Mavis is *fast*, *consistent*, and *cheap* (compared to a
frontier model). It's the daily workhorse.

### ChatGPT 5.6 — the strategic advisor

ChatGPT 5.6 (the new OpenAI frontier model) was consulted
for the **harder architectural and design decisions** — the
ones that shape the project's identity, not just the
implementation. The repository history attributes the following
design discussions to ChatGPT 5.6. These are process notes, not runtime-evaluation evidence:

- **F2 RAG injection hardening review** — the 3-layer defense
  (XML escape + tag + system rule) was stress-tested with
  ChatGPT 5.6's critique. Several refinements came from
  that review.
- **Multi-tier embeddings architecture** — the per-policy
  tier selection (chat_rag → NAS, vault_ingest → edge,
  facts → edge) was a ChatGPT 5.6 design call.
- **AGPLv3 license selection** — the trade-off (Apache 2.0
  vs MIT vs AGPLv3) was discussed with ChatGPT 5.6. The
  "no closed-source wrappers" guarantee was the deciding
  factor.
- **Hackathon scope cuts** — the shipment and defer list was shaped
  through repeated AI critique and human approval.
- **Cleanup review** — rename history and irreversible steps were
  cross-reviewed before implementation.

ChatGPT 5.6 is *expensive* and *slow*, but it catches
things Mavis misses. Used sparingly, on the hard calls.

---

## The collaboration model

The pattern that worked:

1. **Mavis drafts** — TDDs, code, tests, retros
2. **project owner reviews** — catches the "this doesn't match my
   mental model" issues
3. **Mavis iterates** — revises based on project owner's input
4. **Mavis runs R1 (verifier sub-agent)** — catches the
   "this has a security hole" issues
5. **Mavis iterates again** — closes the R1 findings
6. **project owner ships** — final approval, manual verification,
   deploy

For the strategic calls (cleanup plan, license, scope
cuts), the pattern was:

1. **project owner frames the question** — "should we cut STT?"
2. **Mavis drafts a position** — based on what the code
   shows
3. **project owner asks ChatGPT 5.6** — for the harder framing
4. **project owner synthesizes** — final decision is human

---

## Build Week delta ledger

| Date | Addition | Evidence commit |
|---|---|---|
| 2026-07-17 | GPT-5.6 frontier provider and router integration | `8003dc9` |
| 2026-07-16 | Local Ollama chat and WebUI judge path | `7d8a0cd`, `1566eb2` |
| 2026-07-17 | Public-dataset F2 classifier benchmark | `b9d39cf` |
| 2026-07-16 | Local vision OCR provider | `a35fecb` |

The pre-period baseline was the private assistant with messaging, memory,
and deployment infrastructure. Final attribution remains tied to the
candidate SHA and the operator's review.

---

## Specific deliverables and their origin

| Deliverable | Origin | AI contribution |
|---|---|---|
| `docs/ARCHITECTURE.md` | Mavis synthesis | 100% Mavis (from module walk) |
| `hermes/agent/loop.py` F2 fix | project owner + Mavis + R1 | 60% Mavis, 30% project owner, 10% R1 catches |
| `hermes/llm/ocr.py` (LocalVisionOcrProvider) | Mavis with R1 dual review | 70% Mavis, 30% R1 |
| `README.md` | Mavis first pass + adversarial review | AI draft with human product direction |
| `BUILD_PROCESS.md` (this file) | Mavis structure, project owner fills in | 50% Mavis, 50% project owner |

---

## What AI was NOT used for

- **Hardware deployment decisions** — the operator chose the
  self-hosted hardware profile and local model size. AI was
  consulted for model selection trade-offs but the final
  call is project owner's.
- **License final decision** — AGPLv3 was project owner's call
  (with AI providing the trade-off analysis).
- **Hackathon deadline management** — project owner's call.
- **Build Week submission narrative** — the "private, secure,
  self-hosted on low resources" pitch is project owner's framing,
  with Mavis helping articulate it.

---

## Why we're explicit

Three reasons:

1. **OpenAI Build Week asks for it** — they want to see
   how AI helped, not just that it did.
2. **Honesty** — pretending AI didn't help would be lying.
   Pretending AI did it all would be lying too. The truth
   is in the table above: most of the code is Mavis drafts
   with human review and verification.
3. **Future developers** — when you read this codebase in
   6 months, knowing which parts are AI-generated and which
   are human decisions helps you decide where to trust
   and where to verify.

---

## Update discipline

When the build process changes (e.g., a new AI tool is
added to the stack, a major design decision is made with
a new model), update this file.

- **New AI tool** → §1, add a subsection
- **New specific decision** → §3, add a row to the table
- **New category of "not used"** → §4

This file is the audit trail of how the project was built.
When the project ships, freeze this doc and reference it
from the README's "Built with AI" section.

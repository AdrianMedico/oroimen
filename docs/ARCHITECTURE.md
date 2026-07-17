# Architecture — Oroimen (formerly Hermes)

> **What this is**: a whole-system map of Oroimen. Who talks to whom,
> where data lives, where the trust boundaries are, and where to look
> when something breaks.
>
> **Audience**: future-You in 6 months, future contributors, and anyone
> who needs to understand the system in one sitting. Optimized for
> skim-then-dive.
>
> **Last review**: 2026-07-16 10:55 CET (against `main` at `a35fecb`)
> **Status**: reviewed candidate architecture; final container smoke remains pending
> **Maintainer**: Mavis with project owner's design input

---

## 0. TL;DR

Oroimen is a personal AI assistant that runs on your own hardware. A
user message enters through a client and is parsed at the HTTP API;
bearer authentication is conditional on `HERMES_API_API_KEY`. It then
lands in the **agent loop**, which gathers context (memory facts, file
content, tool outputs), calls the **LLM** through a circuit breaker,
and returns a response. **Knowledge is local-first**: the vault lives
in SQLite and the public Compose path computes embeddings with
qwen3-embedding:0.6b through local Ollama. Cloud providers are opt-in
and disabled in the default evaluator path. A local-vision adapter
exists in code but is not wired into that public runtime.

```
┌────────────┐      ┌──────────────┐      ┌─────────────────┐
│  Clients   │─────▶│  HTTP API    │─────▶│  Agent Loop     │
│ (3 fronts) │      │  (FastAPI)   │      │  (loop.py)      │
└────────────┘      └──────────────┘      └────────┬────────┘
                                                  │
                       ┌──────────────────────────┼──────────────────┐
                       ▼                          ▼                  ▼
                ┌─────────────┐           ┌─────────────┐    ┌─────────────┐
                │   Memory    │           │     LLM     │    │   Tools     │
                │  (RAG/F1)   │           │   Router    │    │  (search,   │
                │  Vault      │           │   + OCR     │    │   reach)    │
                └──────┬──────┘           └──────┬──────┘    └──────┬──────┘
                       │                        │                  │
                       ▼                        ▼                  ▼
                ┌──────────────────────────────────────────────────────────┐
                │        Providers & Tiers (Local / optional Edge / optional Cloud)     │
                └──────────────────────────────────────────────────────────┘
```

If that sketch is enough, you can stop here. The rest is detail.

---

## 1. The big picture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CLIENTS (3 frontends)                           │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐                    │
│  │   WebUI     │  │   Telegram   │  │   Mobile     │  (Rikkahub fork)   │
│  │  (hero)     │  │  (secondary) │  │   (partial)  │  Own containers,   │
│  │  own cont.  │  │  long-poll   │  │   Android    │  HTTP API only     │
│  └──────┬──────┘  └──────┬───────┘  └──────┬───────┘                    │
└─────────┼─────────────────┼──────────────────┼──────────────────────────┘
          │ HTTPS           │ Telegram         │ HTTPS
          ▼                 ▼                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                  HTTP API  (hermes/receivers/)                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ http_api.py        (96KB) — main FastAPI app, /v1/chat, /v1/files│   │
│  │ jobs_api.py        (17KB) — /v1/jobs (deep research)             │   │
│  │ ocr_api.py         (12KB) — /v1/ocr (user-only external OCR)     │   │
│  │ polling.py         ( 4KB) — Telegram long-polling adapter        │   │
│  │ auth.py            ( 3KB) — bearer auth dependency                │   │
│  │ base.py            ( 0KB) — UpdateReceiver ABC                   │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│  create_app() @ http_api.py:419                                          │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                 AGENT LOOP  (hermes/agent/loop.py, 62KB)                 │
│                                                                          │
│  The heart. User message + history → response.                            │
│                                                                          │
│  Responsibilities:                                                        │
│   1. Resolve file refs from user message  (_resolve_file_refs)            │
│   2. Inject file content with F2 wrap   (wrap_file_content)               │
│   3. Recall F1 facts (long-term memory)                                   │
│   4. Optionally call tools (search, reach, etc.)                         │
│   5. Build messages, call LLM                                              │
│   6. Stream response back                                                 │
│                                                                          │
│  Key files:                                                               │
│   - wrap_file_content()        — F2 RAG injection fix (Sprint 19.6)       │
│   - FILE_CONTENT_SYSTEM_RULE   — F2 Layer 3 wording                       │
│   - _xml_escape()              — F1/F2 Layer 1                            │
│   - _inject_file_content_system_rule() — Layer 3 injection                │
│                                                                          │
│  Executable security evidence lives under tests/unit and tests/e2e.                  │
└──────────────────┬──────────────────────┬─────────────────────┬──────────┘
                   │                      │                     │
                   ▼                      ▼                     ▼
        ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
        │   LLM + OCR      │   │   Memory (RAG)   │   │   Tools          │
        │ (hermes/llm/)    │   │ (hermes/memory/)  │   │ (hermes/tools/)  │
        │  69KB + 27KB     │   │  ~400KB total     │   │   ~70KB total    │
        └────────┬─────────┘   └────────┬─────────┘   └────────┬─────────┘
                 │                      │                      │
                 ▼                      ▼                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              PROVIDERS, STORAGE, INFRASTRUCTURE                          │
│                                                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│  │ Local tier  │  │  Edge tier  │  │  Cloud tier │  │  Local      │      │
│  │ qwen3 embed │  │ configurable│  │  OpenAI     │  │  vision     │      │
│  │ via Ollama  │  │  optional   │  │  opt-in     │  │  via Ollama │      │
│  │ public path │  │             │  │  explicit   │  │  optional   │      │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘      │
│                                                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│  │  SQLite     │  │  Vault      │  │  InfluxDB   │  │  File       │      │
│  │ db.py 149KB │  │ vault.py    │  │ observab.   │  │  system     │      │
│  │ WAL+backup  │  │  47KB       │  │  6.6KB      │  │  (drop dir) │      │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. The four layers

| Layer | Lives in | Job | Failure mode |
|---|---|---|---|
| **Input** | `hermes/receivers/`, `hermes/handlers/` | Accept and parse user input; enforce auth when configured | Conditional 401, malformed input |
| **Domain** | `hermes/agent/loop.py`, `hermes/memory/`, `hermes/llm/`, `hermes/tools/`, `hermes/services/` | Decide what to do, recall context, call models, format output | Logic bugs, OOM on huge contexts |
| **Storage** | `hermes/memory/db.py`, `hermes/memory/vault.py`, file system | Persist conversations, files, facts, embeddings | DB locks, disk full, vault path missing |
| **Infrastructure** | `Dockerfile`, `docker-compose.yml`, `.github/workflows/`, `hermes/observability/`, `hermes/util/`, `hermes/security/`, `hermes/config.py` | Run the box, ship logs, validate env | Image build, missing env vars, OOM at container level |

The **agent loop** is the only layer that knows about all of memory,
LLM, and tools in the same call. Everything else is a service that
the loop composes.

---

## 3. The clients

| Client | Status | Container | Protocol | Why |
|---|---|---|---|---|
| **WebUI** | Hero, in own container | Yes (separate from backend) | HTTP API | Primary interface. Public Compose binds the API to loopback; bearer auth is enforced when a key is configured. Keeps the container boundary explicit. |
| **Telegram** | Secondary, north star | Backend long-polls Telegram | Telegram Bot API | Bot token in `.env`, allowed_user_ids ACL, /externalOCR command. Sanitization burden: high (secrets). De-prioritized but still maintained. |
| **Mobile (Rikkahub fork)** | Partial, Android | n/a (native app) | HTTP API | Forked upstream Rikkahub to add Oroimen chat. Build requires Android Studio, 3+ days to ship. Citing upstream in public repo, not shipping the fork. |
| **CLI (`hermes` command)** | Yes, via `__main__.py` | Backend | n/a | `hermes/__main__.py` (36KB) is the operator CLI: start server, run migrations, ingest a file, trigger OCR, etc. Not a user client. |

**Trust boundary note**: WebUI and Mobile talk to the backend via
HTTP API only. They never import Python from the backend, so they
stay independent works under AGPLv3. Telegram is more entangled
because handlers live in the same Python package — but Telegram
support is being deprecated per the cleanup plan, not a public-repo
concern.

---

## 4. The agent loop (`hermes/agent/loop.py`, 62KB)

This is the heart. The whole point of the system is to take a user
message and return a response, and this is where it happens.

### 4.1 What happens on each user message

```
  User message
       │
       ▼
  ┌────────────────────────────────────────────────────────────┐
  │ 1. Parse + validate                                        │
  │    - resolve file refs from @file / $vault notation        │
  │    - check budget: len(wrap_file_content(...)) ≤ cap       │
  └────────────┬───────────────────────────────────────────────┘
               │
               ▼
  ┌────────────────────────────────────────────────────────────┐
  │ 2. Recall F1 facts (long-term memory)                      │
  │    - query hermes/memory/facts.py                          │
  │    - wrap with USER_MEMORY_WRAPPER_PROSE                   │
  └────────────┬───────────────────────────────────────────────┘
               │
               ▼
  ┌────────────────────────────────────────────────────────────┐
  │ 3. Resolve file content (F2 path)                          │
  │    - read bytes from vault/source_path                     │
  │    - extract text via extractors/ (tesseract, pymupdf, …)  │
  │    - if extraction confidence low → queue to OCR           │
  │    - wrap with wrap_file_content() (XML-escape + tag)      │
  └────────────┬───────────────────────────────────────────────┘
               │
               ▼
  ┌────────────────────────────────────────────────────────────┐
  │ 4. (Optional) Call tools                                   │
  │    - search_files (vault), web_search, agent_reach, etc.   │
  │    - tool outputs appended to messages                     │
  └────────────┬───────────────────────────────────────────────┘
               │
               ▼
  ┌────────────────────────────────────────────────────────────┐
  │ 5. Build messages, inject FILE_CONTENT_SYSTEM_RULE         │
  │    - if file content present → add system rule             │
  │    - prevents prompt-injection via file content            │
  └────────────┬───────────────────────────────────────────────┘
               │
               ▼
  ┌────────────────────────────────────────────────────────────┐
  │ 6. LLM call (via llm/router.py)                            │
  │    - circuit breaker per provider                          │
  │    - chain: primary → fallback → fallback                  │
  │    - if all fail → return error to user                    │
  └────────────┬───────────────────────────────────────────────┘
               │
               ▼
  ┌────────────────────────────────────────────────────────────┐
  │ 7. Stream response back to caller                          │
  │    - persist to DB                                         │
  │    - emit observability metrics (InfluxDB)                 │
  └────────────────────────────────────────────────────────────┘
               │
               ▼
          Response
```

### 4.2 The two security layers (F1, F2)

- **F1**: protects against prompt-injection via long-term memory
  (`facts.py`). XML-escape + `<user_memory>` wrap + system rule.
  See `wrap_user_memory_text()`.
- **F2**: protects against prompt-injection via file content
  (Sprint 19.6). Same shape, different tag. See
  `wrap_file_content()`. Seven measured E2E cases passed against
  MiniMax-M3; the second-provider baseline remains pending.

Both are 3-layer defenses: XML escape + explicit tag boundary +
system rule. The system rule tells the LLM "this content is DATA,
not instructions."

### 4.3 Why the loop is 62KB

It's a state machine that handles: file budget, streaming, tool
calls, error fallbacks, multiple LLM providers, F1/F2 injection,
multiple content types (text, image, file), and observability
hooks. The size is justified by the feature surface, but the
function-level structure is clean: each phase of the pipeline is
a separate method.

---

## 5. Memory + RAG (`hermes/memory/`, ~400KB)

The biggest module by far. Owns: how files get in, how they're
stored, how they're searched, how context gets pulled.

### 5.1 The pipeline

```
  ┌──────────────────┐
  │  Drop folder     │  ← user puts file in /data/.../drop/
  │  (filesystem)    │
  └────────┬─────────┘
           │  inotify / FSEvents / polling
           ▼
  ┌──────────────────┐
  │  drop_watcher    │  ← watches the folder
  │  (49KB)          │
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │  ingest_router   │  ← routes by file type
  │  (92KB)          │
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │  Extractors      │  ← tesseract / pymupdf / openpyxl /
  │  (extractors/)   │    python_docx / plain
  └────────┬─────────┘
           │  raw text
           ▼
  ┌──────────────────┐
  │  ocr_decision    │  ← if extraction confidence low,
  │  (20KB)          │    queue for OCR (Tesseract local or
  │                  │    LocalVisionOcrProvider for images)
  └────────┬─────────┘
           │  extracted text
           ▼
  ┌──────────────────┐
  │  chunker         │  ← split into chunks
  │  (8.6KB)         │
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │  embeddings      │  ← multi-tier:
  │  (50KB)          │    local Ollama (public default)
  │  + router (56KB) │    edge (optional)
  │                  │    cloud (explicit opt-in)
  └────────┬─────────┘
           │  vectors
           ▼
  ┌──────────────────┐
  │  vault (SQLite)  │  ← persistent storage
  │  vault.py (47KB) │    files, chunks, vectors, facts
  │  db.py (149KB)   │
  └──────────────────┘
```

### 5.2 What lives in the database

`hermes/memory/db.py` (149KB) is the SQLite layer. Tables (from
migrations): `vault_files`, `vault_chunks`, `vault_collections`,
`facts`, `conversations`, `messages`, `jobs`, `ocr_pending`, etc.
WAL mode + nightly backup (`scripts/backup_db.sh`).

### 5.3 Collections

`hermes/memory/collections.py` (28KB): tag-based grouping. A file
can be in many collections. A query can scope to a collection.
Backed by SQLite join tables.

### 5.4 Edge coordinator (`hermes/memory/edge_coordinator.py`, 23KB)

An optional higher-capacity tier between local Compose and cloud.
It can route work to an operator-configured inference endpoint.
This tier is disabled by default and is not required by the public
evaluator path.

### 5.5 Background sleep cycle (`hermes/memory/sleep_cycle.py`, 15KB)

Sprint S10's autonomy: when idle, the agent runs maintenance
(deep re-embed stale chunks, fact consolidation, etc.). Hard to
demo in 3 minutes, defer from hackathon.

---

## 6. LLM + OCR providers (`hermes/llm/`)

### 6.1 The router

`hermes/llm/router.py` is the multi-provider LLM router. It reads
`Settings.text_chain_full`: the normal local/fallback chain plus the
frontier provider only when frontier is explicitly enabled. Providers
are tried in order; the circuit breaker in `hermes/llm/breaker.py`
opens on repeated failures and advances to the next tier.

### 6.2 OCR adapters

`hermes/llm/ocr.py` contains hosted and local-vision adapters. The
public Compose evaluator path does not wire the local-vision adapter or
pull its model, so this document makes no runtime or latency claim for
it. The hosted adapter can make network requests when an operator wires
and enables it outside the default path. Local extractors such as
Tesseract remain part of document ingestion.

### 6.3 Provider config convention

Every provider has a Settings class with env var bindings. The
convention is `<DOMAIN>__<FIELD>` double-underscore. Examples:
`LLM_TEXT_PRIMARY__MODEL`, `EMBEDDING_TIER_NAS__BASE_URL`. This
lets you configure tiers without code changes.

---

## 7. Embeddings multi-tier (Sprint 19.5)

The public evaluator path uses the same embedding router for query and
ingestion vectors. Optional tiers remain configuration-driven.

```
  Query embedding request
           │
           ▼
  ┌──────────────────────────────────────────────────┐
  │  embedding_router.py (56KB)                      │
  │  Tries tiers in order, based on policy:          │
  │   - chat_rag      → configured policy order      │
  │   - vault_ingest  → configured policy order      │
  │   - only declared policies are selectable       │
  └────────┬─────────────────────────────────────────┘
           │
           ├─→ Local tier  (qwen3-embedding:0.6b in public Compose)
           ├─→ Edge tier   (optional)
           └─→ Cloud tier  (explicit opt-in)
```

Per-policy tier selection is implemented by the embedding router and
configured with `EMBEDDING_POLICY_*` environment variables.

---

## 8. Storage

| Store | Module | What | Why this store |
|---|---|---|---|
| **SQLite (WAL)** | `hermes/memory/db.py` | conversations, files, chunks, vectors, facts, jobs | Single-box, no separate DB process, WAL backup is trivial |
| **File system (vault)** | `hermes/memory/vault.py` | original file content, OCR results, extractors' raw output | Files > DB rows. Path-based, content-addressed by SHA-256 file_id |
| **Pydantic Settings** | `hermes/config.py` | env-driven config | Type-safe, validates at boot, no surprises |
| **InfluxDB** | `hermes/observability/influxdb.py` | metrics (latency, token usage, errors) | Optional. If InfluxDB unreachable, observability is no-op, doesn't break the agent. |

### 8.1 Why SQLite, not Postgres

- Single-user system. No concurrency contention that needs Postgres.
- WAL + nightly backup (24h RPO) is the documented rollback.
- Migrations are simple SQL files. No Alembic ceremony.
- The cleanup plan v2.1 (Q6) confirmed: rename tables to `oroimen_*`
  via `v24_rename_hermes_tables` migration. (In the new public repo
  from day 1, tables are `oroimen_*`.)

---

## 9. Trust boundaries — what stays where

This is the privacy story. The diagram below shows what data goes
where during normal operation.

```
                    ┌────────────────────────────────────┐
                    │      YOUR MACHINE (the box)         │
                    │                                    │
   User input ──▶   │  ┌──────────────────────────────┐  │
                    │  │   WebUI / Telegram / Mobile  │  │
                    │  └──────────────┬───────────────┘  │
                    │                 ▼                  │
                    │  ┌──────────────────────────────┐  │
                    │  │      Backend (loop.py)       │  │
                    │  └──────────────┬───────────────┘  │
                    │                 │                  │
                    │     ┌───────────┼───────────┐      │
                    │     ▼           ▼           ▼      │
                    │  ┌──────┐   ┌──────┐   ┌──────┐    │
                    │  │ SQLite│   │Vault │   │Ollama│    │
                    │  │       │   │(disk)│   │local │    │
                    │  └──────┘   └──────┘   └──────┘    │
                    │                                    │
                    │  All user data, conversations,    │
                    │  files and embeddings: HERE.    │
                    └────────┬───────────────────┬───────┘
                             │                   │
              ┌──────────────┘                   │
              │  LAN only (<internal-network>)          │
              │  Optional, opt-in                │
              ▼                                  │
       ┌──────────────┐                          │
       │ Local Ollama │  Public default          │
       │ embeddings   │  No cloud required       │
       └──────────────┘                          │
                                                 │
       ┌──────────────┐                          │
       │  Edge tier   │  Optional, opt-in        │
       │  endpoint    │                          │
       └──────────────┘                          │
                                                 │
              ┌──────────────────────────────────┘
              │  Internet (only if user opts in)
              ▼
       ┌──────────────┐
       │  OpenAI      │  Selected conversation is sent.
       │  cloud tier  │  Explicit opt-in only.
       └──────────────┘
```

**What leaves the box by default**: nothing. The default
configuration runs everything local. Frontier mode is explicit
opt-in and sends the selected conversation without automatic redaction.

---

## 10. End-to-end flows

### 10.1 Chat with file (the most common path)

```
1. User drops a supported document in `./drop/` (mounted at `/app/drop`)
2. `drop_watcher` detects it on its polling interval
3. `ingest_router` selects the extractor and obtains text
4. the chunker splits the extracted text
5. `VaultEmbedder` calls the configured local embedding policy
6. vectors and chunk text are stored in `vault_chunks`
7. the file is marked indexed
8. User asks "what's the contract termination clause?"
9. WebUI POSTs `/v1/chat/completions` with the question
10. the API conditionally checks bearer auth and calls the agent loop
11. the agent invokes `search_files` when retrieval is needed
12. `search_files` embeds the query and searches `vault_chunks`
13. ranked fragment text returns through the escaped `<tool_output>` boundary
14. the local chat model generates a grounded answer
15. the response returns to the WebUI and the conversation is persisted
```

### 10.2 Drop folder, no user interaction

Steps 1-9 above. Then idle. Until user asks a question, the
system is just persisting and indexing.

### 10.3 OCR adapter boundary (outside the public evaluator path)

Legacy handlers and OCR adapters remain in the package, but the public
Compose file does not pull a vision model or construct the
`LocalVisionOcrProvider`. Exercising that adapter requires explicit
operator wiring and separate verification; it is not part of the
submission runtime claim.

### 10.4 Deep research job (Sprint S14)

```
1. WebUI POST /v1/jobs {query, budget}
2. jobs_api validates, persists job
3. Background scheduler picks the job
4. service.py: decompose → web_search → analyze → synthesize
5. Multi-step: each step uses LLM + tools
6. Persist results, mark job complete
7. WebUI polls /v1/jobs/{id} for status
```

Heavy. Defer from hackathon.

---

## 11. Where to look when...

| Symptom | First place to look | Likely cause |
|---|---|---|
| LLM returns garbage | `hermes/llm/router.py` + circuit breaker status | Breaker open on a provider, all tiers failing |
| Slow responses | InfluxDB metrics → `llm_latency` | Probably NAS/edge tier down, falling through to cloud |
| File not ingested | `hermes/memory/drop_watcher.py` logs | inotify limit hit, or drop path not mounted in container |
| Search returns nothing | `hermes/services/embeddings.py` + tier logs | Embedding tier misconfigured, vectors empty |
| Prompt injection succeeds | `hermes/agent/loop.py` + F2 fix | Did you bypass `wrap_file_content`? Don't. |
| 401 from API | `hermes/receivers/auth.py` | Bearer auth was configured and the token is missing or wrong |
| Container won't start | `Dockerfile`, `.env`, `docker-compose.yml` | Missing env var, image build failure |
| Tests fail on CI only | `.github/workflows/`, `tests/e2e/_helpers.py` | Missing API keys (env vars in CI secrets) |
| Vault path errors | `hermes/memory/vault.py:VaultConfig` | Path not mounted, permissions wrong |
| OCR adapter fails in a custom deployment | `hermes/llm/ocr.py:LocalVisionOcrProvider` | Adapter was wired without a reachable, separately configured model |

---

## 12. Submission caveats

- The Python package is still named `hermes`; public docs and imports use that
  real package name consistently.
- Live-provider tests require operator-supplied access and are never counted
  when skipped.
- The final image, resource envelope, and fresh-clone results remain tied to
  the pending candidate SHA. See `FINAL_VERIFICATION.md`.

---

## 13. Public references

- `README.md` — setup, hero features, and truthful current status.
- `docs/DEMO_SCRIPT.md` — recording plan gated by the container smoke.
- `docs/EVAL_STRATEGY.md` — measured versus pending evaluation evidence.
- `FINAL_VERIFICATION.md` — authoritative gate status.

---

## 14. Update discipline

When the architecture changes meaningfully, update this doc.
- **New major component** → section 1 diagram + new section
- **New flow / scenario** → add to section 10
- **New rough edge discovered** → section 12
- **New "where to look when"** → section 11
- **Trust boundary change** → section 9 (this is sensitive)

The doc is intentionally skim-friendly. Long prose goes in TDDs;
here, the priority is "can I find the thing in 30 seconds".

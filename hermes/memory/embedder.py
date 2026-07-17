"""Slice 2.5 — VaultEmbedder + EmbedWatcher + search.

Reference: docs/TDD_VAULT_EMBEDDINGS.md (V1.1)

Responsibilities:
1. `VaultEmbedder.embed_file(file_id)` — chunk + embed + atomic DELETE+INSERT
   on `vault_chunks` for that file_id. Idempotent: re-call on same file
   (same text_version) leaves only the new chunks.
2. `VaultEmbedder.find_files_needing_embed()` — stateless LEFT JOIN query:
   - Files with no chunks (never embedded).
   - Files where vault_files.text_version != any chunk's text_version
     (tier upgrade v0 → v15: stale chunks).
3. `VaultEmbedder.search(query, top_k)` — in-memory NumPy Cosine over
   all chunks with current text_version. FAISS deferred >10K chunks.
4. `EmbedWatcher.run_once()` / `run()` — poll loop that calls
   `find_files_needing_embed()` and `embed_file()` for each. Process-local
   `_in_flight: set[str]` prevents re-pick if `poll_interval_s <
   embed_duration_s`.

M6 reminder (load-bearing per Gemini round 3, 2026-07-08):
`_reconcile_db_from_filesystem()` is the FIRST thing this module
checks at startup. The Watcher MUST NOT start processing until the
filesystem mirror is in sync with `ingest_jobs`. See
`ingest_router.py:_reconcile_db_from_filesystem()`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np

from hermes.memory.chunker import Chunker
from hermes.observability.influxdb import write_research_metric

if TYPE_CHECKING:
    from hermes.config import Settings
    from hermes.memory.db import Database
    from hermes.memory.vault import Vault
    from hermes.services.embeddings import EmbeddingsService
# PR #113d round 4 NIT: use `_safely_rollback` (vault.py:177) instead of
# bare `await conn.rollback()` in except BaseException blocks. The bare
# version can itself be cancelled by Python's task-cancellation machinery,
# poisoning the connection. The shielded variant drains cancel state and
# surfaces the original error to the caller.
from hermes.memory.vault import _safely_rollback
from hermes.services.embedding_router import EmbeddingPolicy

logger = logging.getLogger(__name__)


# PR #113c (B7 fix + Round 2 MAJOR 1): bounded InfluxDB tag
# cardinality. InfluxDB creates a new series per unique (measurement,
# tag-set) combination; an unbounded `error_type` tag (raw Python
# exception class name) easily produces 50+ unique series over a
# deployment's lifetime. The fix: classify the exception into a small
# fixed taxonomy (`transient` / `ratelimit` / `permanent` / `config` /
# `unknown`) for the `error_taxonomy` tag, and put the raw class name
# in a field (which is queryable but doesn't create new series).
#
# The `jobs/service.py:678-682` pattern uses `error_taxonomy` as the
# tag name (bounded enum) — PR #113c used `error_type` (raw class
# name). Round 2 NIT renames for consistency.
#
# PR #113c round 3 MAJOR-2 fix: extend coverage to common embedding-
# backend exceptions (httpx, openai, sqlite3) that the round 2 version
# left in "unknown". The previous version only checked explicit class
# names in frozensets; the new version adds isinstance checks for
# exception hierarchies that aren't named exactly (e.g., httpx uses
# RemoteProtocolError but also ConnectError, ReadTimeout,
# ConnectTimeout — all OSError subclasses via httpx's hierarchy).
_ERROR_TAXONOMY_TRANSIENT: frozenset[str] = frozenset(
    {
        "asyncio.TimeoutError",
        "TimeoutError",
        "ConnectionError",
        "ConnectionResetError",
        "ConnectionAbortedError",
        "BrokenPipeError",
        "RemoteProtocolError",
        "IncompleteReadError",
        "APIConnectionError",
        "APITimeoutError",
        "httpx.ConnectError",
        "httpx.ReadTimeout",
        "httpx.ConnectTimeout",
        "httpx.PoolTimeout",
        "aiohttp.ClientError",
    }
)
_ERROR_TAXONOMY_RATELIMIT: frozenset[str] = frozenset(
    {
        "RateLimitError",
        "TooManyRequests",
    }
)
_ERROR_TAXONOMY_CONFIG: frozenset[str] = frozenset(
    {
        "ValueError",
        "TypeError",
        "KeyError",
        "AttributeError",
        "ConfigError",
    }
)


def _classify_exception(exc: BaseException) -> str:
    """Map an exception to a bounded taxonomy tag value.

    Returns one of: `transient`, `ratelimit`, `permanent`, `config`,
    `unknown`. Stable across Python exception hierarchies (uses
    `type(exc).__module__ + "." + type(exc).__name__` for the lookup
    so subclassed exceptions still match their parent's bucket).
    """
    cls = type(exc)
    qualname = f"{cls.__module__}.{cls.__name__}"
    if qualname in _ERROR_TAXONOMY_RATELIMIT or cls.__name__ in _ERROR_TAXONOMY_RATELIMIT:
        return "ratelimit"
    if qualname in _ERROR_TAXONOMY_TRANSIENT or cls.__name__ in _ERROR_TAXONOMY_TRANSIENT:
        return "transient"
    if qualname in _ERROR_TAXONOMY_CONFIG or cls.__name__ in _ERROR_TAXONOMY_CONFIG:
        return "config"
    # PR #113d round 4 MAJOR: json.JSONDecodeError IS-A ValueError, so
    # this check MUST come BEFORE the ValueError catchall below,
    # otherwise it would be dead code (round 3 bug). JSON decode errors
    # are `permanent` (corrupted LLM response, not a misconfig).
    if isinstance(exc, json.JSONDecodeError):
        return "permanent"
    if isinstance(exc, (KeyError | ValueError | TypeError | AttributeError)):
        return "config"
    # PR #113c round 3 MAJOR-2: network error subclasses (httpx,
    # aiohttp, asyncio) need to bucket as `transient`, not fall through
    # to `permanent` via OSError catchall. Check BEFORE OSError so the
    # subclasses match the network-error bucket.
    if isinstance(
        exc,
        ConnectionError
        | TimeoutError
        | ConnectionAbortedError
        | ConnectionResetError
        | BrokenPipeError,
    ):
        return "transient"
    # SQLite OperationalError "database is locked" is transient
    # (retry succeeds once the writer releases the lock).
    if isinstance(exc, sqlite3.OperationalError):
        return "transient"
    # SQLite DatabaseError parent (corruption, disk full) is permanent.
    if isinstance(exc, sqlite3.DatabaseError):
        return "permanent"
    # PR #113d round 4 MAJOR: openai.BadRequestError, AuthenticationError,
    # PermissionDeniedError, NotFoundError, UnprocessableEntityError all
    # subclass openai.APIStatusError. They represent HTTP 4xx (client
    # fault — bad request, wrong key, etc.) — NEVER transient. Without
    # this check they fall through to `unknown`. Wrapped in try/except
    # because the project may run without the openai SDK (the embedding
    # service abstraction is backend-agnostic).
    try:
        import openai as _openai_mod

        if isinstance(exc, _openai_mod.APIStatusError):
            return "permanent"
    except ImportError:
        pass
    # OS errors — narrow remaining OSError (e.g., disk I/O, permission).
    if isinstance(exc, OSError | IOError):
        return "permanent"
    return "unknown"


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One result from VaultEmbedder.search().

    Sorted by `score` DESC at the caller. `score` is cosine similarity
    in [-1, 1] (typically [0, 1] for normalized embeddings).

    PR #113c: `source_path` and `added_at` are now part of the shape
    so RAG callers (e.g. the HTTP API) can show the user WHERE the
    hit came from without a second `Vault.get_file()` round-trip.
    Both are fetched via a JOIN with `vault_files` in the search
    query. The two extra columns are nullable on the join (if a
    chunk's file_id is somehow missing from vault_files, which
    shouldn't happen but we don't want to crash); defaults = None.
    """

    file_id: str
    chunk_id: str
    chunk_index: int
    text: str
    score: float
    text_version: str
    embedding_model: str
    # PR #113c (SUGGESTION): forward-compat for caller UX.
    # None = vault_files row was missing at query time (defensive).
    source_path: str | None = None
    added_at: str | None = None


class VaultEmbedder:
    """Chunk + embed + persist for vault_files.

    Usage:
        embedder = VaultEmbedder(vault, db, embeddings, chunker, settings)
        chunk_count = await embedder.embed_file(file_id)
        candidates = await embedder.find_files_needing_embed()
        hits = await embedder.search("how does the LAN worker pull jobs?", top_k=5)
    """

    def __init__(
        self,
        *,
        vault: Vault,
        db: Database,
        embeddings: EmbeddingsService,
        chunker: Chunker,
        settings: Settings,
    ) -> None:
        self._vault = vault
        self._db = db
        self._embeddings = embeddings
        self._chunker = chunker
        self._settings = settings
        # Re-entrancy guard: Watcher's process-local "in-flight" set.
        # Prevents re-pick if poll_interval_s < embed_duration_s.
        self._in_flight: set[str] = set()
        self._in_flight_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # find_files_needing_embed — Watcher query (stateless)
    # ------------------------------------------------------------------
    async def find_files_needing_embed(self, *, model_version: str | None = None) -> list[str]:
        """Return file_ids that need (re-)embedding.

        Two conditions match:
        1. `vault_files` has no rows in `vault_chunks` (never embedded).
        2. `vault_files.text_version` != any row in `vault_chunks` for
           that file (tier upgrade: v0 → v15; chunks are stale).

        The Watcher calls this and `embed_file()` on each result.

        SQL: LEFT JOIN. Stateless (no cursor in memory) — restart-safe.
        Q2 (chunk visibility during re-embed) is resolved by SQLite WAL
        (Gemini round 3 architect, 2026-07-08).

        PR #113c (B9 fix): filter out files with NULL text.
        `vault_files.text` is nullable — a row can exist (e.g. added
        via `Vault.add()` while a Tier 0 extraction is in flight, or
        the extraction failed entirely) without any text to embed.
        `embed_file()` already short-circuits with a "skip_empty_text"
        log in that case, but returning the file_id every cycle
        flooded the log with no-ops. Filtering at the query level
        keeps the log signal clean: only files that COULD be embedded
        this cycle are returned.

        Args:
            model_version: filter to chunks with this embedding_model.
                None = no filter (return all file_ids needing embed).

        Returns:
            list[str] of file_ids. Empty list = nothing to do.
        """
        # Slice 2.5 GREEN: query matches TDD §"Watcher query".
        # Note: this is the LEFT JOIN statelessly — no in-memory state.
        # PR #113c: added `f.text IS NOT NULL` to skip files with no
        # extractable text (Tier 0 not yet arrived, or failed).
        sql = """
            SELECT DISTINCT f.file_id
            FROM vault_files f
            LEFT JOIN vault_chunks c ON f.file_id = c.file_id
            WHERE f.text IS NOT NULL
              AND (c.chunk_id IS NULL
                   OR f.text_version != c.text_version)
        """
        params: tuple = ()
        if model_version is not None:
            sql = """
                SELECT DISTINCT f.file_id
                FROM vault_files f
                LEFT JOIN vault_chunks c
                    ON f.file_id = c.file_id AND c.embedding_model = ?
                WHERE f.text IS NOT NULL
                  AND (c.chunk_id IS NULL
                       OR f.text_version != c.text_version)
            """
            params = (model_version,)

        async with self._db.conn.execute(sql, params) as cur:
            rows = await cur.fetchall()
        return [row["file_id"] for row in rows]

    # ------------------------------------------------------------------
    # embed_file — chunk + embed + atomic DELETE+INSERT
    # ------------------------------------------------------------------
    async def embed_file(self, file_id: str, *, model_version: str | None = None) -> int:
        """Embed `vault_files.text` for file_id. Returns chunk count.

        Algorithm (TDD §"embed_file"):
        1. Acquire `_db._write_lock` (canonical serialization point).
        2. Read text + text_version from vault (raises KeyError if missing).
        3. If text is empty: log INFO, return 0 (no chunks).
        4. Chunker → list[Chunk].
        5. embed_batch(texts) → list[np.ndarray] (one call, batched).
        6. BEGIN IMMEDIATE transaction:
           a. DELETE existing chunks for file_id (purge stale).
           b. INSERT new chunks with (text, embedding, embedding_model,
              text_version) for each.
           c. COMMIT.
        7. Release the lock.

        Atomicity: if any step fails (chunking OOM, embed API error,
        DB error), the entire transaction rolls back. vault_chunks
        for file_id is unchanged from before the call.

        PR #113c round 2 (MAJOR-1 concurrency fix): the read of
        `text` + `text_version` and the BEGIN IMMEDIATE+DELETE+INSERT
        are now in the SAME `_write_lock` critical section. The
        previous B6 fix only locked around the DELETE+INSERT, leaving
        a window where a concurrent `Vault.update_text` (Tier 1.5
        worker delivering v15_lan_worker between Watcher tick and
        embed BEGIN) could update `vault_files.text + text_version`
        while the embed was reading the OLD text. The chunk then
        got the OLD text + NEW text_version, producing stale RAG
        content with a version tag the user trusted. Cost: the lock
        is now held during the LLM embed call (O(seconds) for
        text-embedding-3-small). Acceptable for NAS host single-process
        Hermes (the same lock is held by `process_inbox` + `update_text`
        anyway; no new contention).

        Raises:
            KeyError: file_id not in vault_files.
        """
        if model_version is None:
            model_version = self._settings.vault_embedding_model

        # 1) Acquire write lock FIRST. Both the read of
        # text/text_version and the BEGIN+DELETE+INSERT must be
        # inside the same critical section. (PR #113c round 2
        # MAJOR-1 fix; the B6 fix only locked the write half.)
        async with self._db._write_lock:
            # 2) Read text + text_version via Vault (raises KeyError).
            text = await self._vault.get_text(file_id)
            text_version = await self._vault.get_text_version(file_id)

            # 3) Empty text → no chunks.
            if not text or not text.strip():
                logger.info(
                    "vault_embedder_skip_empty_text",
                    extra={"file_id": file_id, "text_version": text_version},
                )
                return 0

            # 4) Chunker.
            chunks = self._chunker.chunk(text)
            if not chunks:
                logger.info(
                    "vault_embedder_skip_no_chunks",
                    extra={
                        "file_id": file_id,
                        "text_version": text_version,
                        "text_length": len(text),
                    },
                )
                return 0

            # 5) Batched embed. Held under the lock (intentional):
            # the read+write must be atomic vs concurrent update_text.
            chunk_texts = [c.text for c in chunks]
            try:
                embeddings = await self._embeddings.embed_batch(
                    chunk_texts,
                    use_case=EmbeddingPolicy.VAULT_INGEST,
                    model=model_version,
                )
            except Exception as exc:
                # Per Slice 2.5 TDD: embed failure must not break the
                # caller (Watcher catches + continues). Re-raise so the
                # Watcher can log + skip this file_id.
                logger.exception(
                    "vault_embedder_embed_failed",
                    extra={
                        "file_id": file_id,
                        "chunk_count": len(chunks),
                        "model": model_version,
                        "error": str(exc),
                    },
                )
                # PR #113c (B7 fix): emit failure metric so the
                # operator dashboard can alert on a spike of
                # `vault_embedder_embed_failed` (e.g. rate(1m) > 5
                # → page). Fire-and-forget; no-op without InfluxDB.
                # PR #113c round 2 (MAJOR 1): tag is bounded
                # `error_taxonomy` (one of transient/ratelimit/
                # permanent/config/unknown) instead of raw class
                # name, to avoid InfluxDB series cardinality
                # explosion. Raw class name lives in a field.
                write_research_metric(
                    "vault_embedder_embed_failed",
                    tags={
                        "model": model_version or "unknown",
                        "error_taxonomy": _classify_exception(exc),
                    },
                    fields={
                        "files": 1,
                        "chunk_count": len(chunks),
                        "exception_class": type(exc).__name__,
                    },
                )
                raise

            if len(embeddings) != len(chunks):
                # Defensive: backend returned a different count. Should not
                # happen with OpenAI / OpenRouter (batched), but log + raise.
                raise RuntimeError(
                    f"embed_batch returned {len(embeddings)} embeddings for "
                    f"{len(chunks)} chunks (file_id={file_id!r})"
                )

            expected_dim = int(getattr(self._settings, "vault_embedding_dim", 0))
            for index, vector in enumerate(embeddings):
                if not isinstance(vector, np.ndarray) or vector.ndim != 1 or vector.size == 0:
                    raise RuntimeError(
                        f"invalid embedding vector at index {index}: expected non-empty 1-D ndarray"
                    )
                if expected_dim > 0 and vector.size != expected_dim:
                    raise RuntimeError(
                        f"embedding dimension mismatch at index {index}: "
                        f"got {vector.size}, expected {expected_dim}"
                    )

            # 6) Atomic DELETE+INSERT in the same critical section.
            #
            # PR #113c (B2 fix): the `Database` class owns a process-wide
            # `asyncio.Lock` (`_write_lock`) that serializes all writers.
            # We MUST take it before any `BEGIN IMMEDIATE` because SQLite
            # busy-timeout only retries for a finite window; under
            # concurrent embed_file() calls the SECOND writer would fail
            # with `database is locked`. The lock is the canonical entry
            # gate; the BEGIN IMMEDIATE inside the lock is belt+suspenders
            # (defense in depth — even if the lock semantics drift later,
            # the BEGIN prevents partial-write visibility).

            await self._db.conn.execute("BEGIN IMMEDIATE")
            try:
                # 6a) Purge old chunks for file_id (idempotent re-embed).
                await self._db.conn.execute(
                    "DELETE FROM vault_chunks WHERE file_id = ?",
                    (file_id,),
                )
                # 6b) Insert new chunks.
                for chunk, embedding in zip(chunks, embeddings, strict=True):
                    chunk_id = str(uuid.uuid4())
                    # np.ndarray → bytes via float32 .tobytes() (Slice 2.5
                    # contract: BLOB stores packed float32 little-endian).
                    emb_bytes = embedding.astype(np.float32).tobytes()
                    await self._db.conn.execute(
                        "INSERT INTO vault_chunks "
                        "(chunk_id, file_id, chunk_index, text, embedding, "
                        " embedding_model, text_version) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            chunk_id,
                            file_id,
                            chunk.chunk_index,
                            chunk.text,
                            emb_bytes,
                            model_version,
                            text_version,
                        ),
                    )
                await self._db.conn.commit()
            except BaseException:
                # PR #113d round 4 NIT: use `_safely_rollback` (vault.py:177)
                # to shield the rollback from re-cancellation + drain cancel
                # state. Bare `await conn.rollback()` can itself be cancelled
                # by Python's task-cancellation machinery when the task is
                # already in the "cancelling" state, leaving the SQLite
                # connection poisoned for subsequent BEGIN IMMEDIATE.
                await _safely_rollback(self._db.conn)
                raise

        chunk_count = len(chunks)
        logger.info(
            "vault_embedder_embedded",
            extra={
                "file_id": file_id,
                "chunk_count": chunk_count,
                "model": model_version,
                "text_version": text_version,
            },
        )
        # PR #113c (B7 fix): emit InfluxDB metric for Slice 2.5
        # observability. Fire-and-forget; if InfluxDB is not
        # initialized (dev/CI without env vars), write_research_metric
        # is a no-op. Tags use the embedding model + tier (text_version
        # encodes the tier via TEXT_VERSION_ORDER).
        write_research_metric(
            "vault_embedder_embedded",
            tags={
                "model": model_version,
                "text_version": text_version,
            },
            fields={
                "chunk_count": chunk_count,
                "files": 1,
            },
        )
        return chunk_count

    # ------------------------------------------------------------------
    # search — in-memory NumPy Cosine
    # ------------------------------------------------------------------
    async def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        model_version: str | None = None,
    ) -> list[SearchHit]:
        """Top-k chunks by cosine similarity to `query`.

        Strategy: load ALL current chunks into memory as a single
        np.ndarray, then cosine. Acceptable for <10K chunks; switch
        to FAISS / sqlite-vss at >10K (TDD I-5).

        Filters:
        - `text_version` matches the current `vault_files.text_version`
          for the file (excludes stale chunks from tier upgrade).
        - `embedding_model` matches `model_version` (default: current
          settings.vault_embedding_model).

        Args:
            query: text to embed and search.
            top_k: max results (default 5).
            model_version: filter to chunks embedded with this model.

        Returns:
            list[SearchHit] sorted by score DESC. Empty list if no
            chunks match, or if RAG is disabled.
        """
        if model_version is None:
            model_version = self._settings.vault_embedding_model

        if not query or not query.strip():
            return []

        # 1) Embed the query.
        try:
            query_emb = await self._embeddings.embed(
                query, use_case=EmbeddingPolicy.VAULT_INGEST
            )
        except Exception as exc:
            logger.exception(
                "vault_embedder_search_embed_failed",
                extra={"error": str(exc)},
            )
            # PR #113c round 3 MAJOR fix: emit bounded metric so
            # operators see "embedding backend down" instead of
            # silent empty results. Without this, users get
            # "no matches" when in reality the embed service is
            # down — invisible to ops dashboards.
            write_research_metric(
                "vault_embedder_search_embed_failed",
                tags={"error_taxonomy": _classify_exception(exc)},
                fields={"exception_class": type(exc).__name__},
            )
            return []
        q_vec = query_emb.astype(np.float32).reshape(1, -1)

        # 2) Load all matching chunks. JOIN with vault_files to filter
        # by current text_version.
        sql = """
            SELECT c.chunk_id, c.file_id, c.chunk_index, c.text,
                   c.embedding, c.embedding_model, c.text_version,
                   f.source_path, f.added_at
            FROM vault_chunks c
            JOIN vault_files f ON f.file_id = c.file_id
            WHERE c.embedding_model = ?
              AND c.text_version = f.text_version
        """
        async with self._db.conn.execute(sql, (model_version,)) as cur:
            rows = await cur.fetchall()

        if not rows:
            return []

        # 3) Stack into a (N, dim) array, compute cosine, top-k.
        # Filter out any non-float32-sized rows (defensive: if a chunk
        # was written with a different model, its dim may differ).
        dim = q_vec.shape[1]
        expected_bytes = dim * 4  # float32 = 4 bytes
        valid_rows: list[dict] = []
        valid_vecs: list[np.ndarray] = []
        for r in rows:
            emb = r["embedding"]
            # PR #113c (B8 fix): defensive None guard.
            #
            # `vault_chunks.embedding` is declared `BLOB NOT NULL` in
            # the schema, so under normal writes this branch is
            # unreachable. However, a corrupted row (e.g. partial
            # commit that left a NULL after SQLite recovery, or
            # manual SQL injection during ops) would make `len(emb)`
            # raise `TypeError: object of type 'NoneType' has no len()`,
            # killing the entire search call. The cost of the guard
            # is one `is None` check per row; the cost of the bug is
            # a silent crash of all RAG queries until manual cleanup.
            if emb is None:
                logger.warning(
                    "vault_embedder_search_null_embedding",
                    extra={
                        "chunk_id": r["chunk_id"],
                        "file_id": r["file_id"],
                    },
                )
                continue
            if len(emb) != expected_bytes:
                logger.warning(
                    "vault_embedder_search_dim_mismatch",
                    extra={
                        "chunk_id": r["chunk_id"],
                        "got_bytes": len(emb),
                        "expected_bytes": expected_bytes,
                    },
                )
                continue
            valid_rows.append(dict(r))
            valid_vecs.append(np.frombuffer(emb, dtype=np.float32).copy())

        if not valid_vecs:
            return []

        mat = np.stack(valid_vecs, axis=0)
        # Cosine similarity: dot / (||q|| * ||mat_i||). Use eps to avoid /0.
        q_norm = float(np.linalg.norm(q_vec))
        m_norms = np.linalg.norm(mat, axis=1)
        # Avoid divide-by-zero: replace 0 with 1 (zero vector → similarity 0).
        safe_m_norms = np.where(m_norms == 0, 1.0, m_norms)
        safe_q_norm = q_norm if q_norm > 0 else 1.0
        sims = (mat @ q_vec.T).reshape(-1) / (safe_m_norms * safe_q_norm)

        # Top-k by score DESC.
        order = np.argsort(-sims)[:top_k]
        hits: list[SearchHit] = []
        for idx in order:
            r = valid_rows[idx]
            hits.append(
                SearchHit(
                    file_id=r["file_id"],
                    chunk_id=r["chunk_id"],
                    chunk_index=r["chunk_index"],
                    text=r["text"],
                    score=float(sims[idx]),
                    text_version=r["text_version"],
                    embedding_model=r["embedding_model"],
                    # PR #113c (SUGGESTION): from JOIN, may be None
                    # if vault_files row is missing for this chunk.
                    source_path=r["source_path"],
                    added_at=r["added_at"],
                )
            )
        return hits


class EmbedWatcher:
    """Poll loop that calls `find_files_needing_embed()` and embeds each.

    Usage:
        watcher = EmbedWatcher(embedder, settings, db)
        await watcher.start()  # spawns run() as a background task
        # On shutdown:
        await watcher.stop()
    """

    #: Local in-flight re-entrancy guard: max time between "picked" and
    #: "embedded" before the next cycle may re-pick.
    _PICK_TIMEOUT_S: Final[int] = 600  # 10 min

    def __init__(
        self,
        *,
        embedder: VaultEmbedder,
        settings: Settings,
        interval_s: int | None = None,
        stop_event: asyncio.Event | None = None,
    ) -> None:
        self._embedder = embedder
        self._settings = settings
        self._interval_s = (
            interval_s if interval_s is not None else settings.vault_watcher_poll_interval_s
        )
        # PR #118 (Sprint 18 hardening, Gemini P0 #2 "Peligro de Apagado"):
        # accept an EXTERNAL stop_event (from __main__.run()) so SIGTERM
        # propagates to the watcher and Hermes shutdown can complete within
        # the Docker grace period (10s default). The previous version
        # created an internal stop_event that was NEVER observed by
        # Hermes shutdown — only by manual `watcher.stop()` (which no
        # production code called). Now: if stop_event is provided, the
        # watcher observes the SAME event as the rest of Hermes, so
        # docker stop → SIGTERM → stop_event.set() → watcher sees it
        # between files in run_once().
        self._stop_event = stop_event if stop_event is not None else asyncio.Event()

    async def run_once(self) -> int:
        """Single poll cycle. Returns number of files embedded this cycle.

                Algorithm:
                1. candidates = find_files_needing_embed()
        2. For each file_id (in deterministic order):
                   - **PR #118: Check stop_event BEFORE starting the next file.**
                     If SIGTERM landed during the previous embed_file's LLM call,
                     we still finish that one (in-flight, atomic, can't cancel
                     mid-embed safely), but we DON'T start the next one. This
                     bounds shutdown latency to "1 in-flight embed's LLM timeout"
                     instead of "all remaining files x LLM timeout".
                   - Skip if already in-flight (re-pick protection).
                   - Mark in-flight, call embed_file(), unmark.
                   - On error: log + continue (loop stays alive).
                3. Return count.
        """
        candidates = await self._embedder.find_files_needing_embed()
        if not candidates:
            return 0

        embedded_count = 0
        for file_id in sorted(candidates):
            # PR #118 (Sprint 18 hardening, Gemini P0 #2): bail out between
            # files if shutdown was signaled. The current embed_file (if
            # any) is in the previous iteration and will finish; we just
            # don't START a new one. This bounds Hermes shutdown latency
            # to "1 in-flight embed" instead of "N embeds x LLM timeout".
            if self._stop_event.is_set():
                logger.info(
                    "vault_watcher_bailing_out_stop_signal",
                    extra={
                        "embedded_so_far": embedded_count,
                        "remaining_candidates": len(candidates) - embedded_count,
                    },
                )
                break
            async with self._embedder._in_flight_lock:
                if file_id in self._embedder._in_flight:
                    logger.debug(
                        "vault_watcher_skip_in_flight",
                        extra={"file_id": file_id},
                    )
                    continue
                self._embedder._in_flight.add(file_id)
            try:
                chunk_count = await self._embedder.embed_file(file_id)
                embedded_count += 1
                logger.info(
                    "vault_watcher_embedded",
                    extra={
                        "file_id": file_id,
                        "chunk_count": chunk_count,
                    },
                )
            except Exception as exc:
                # Per TDD: error must not kill the loop. Log + continue.
                logger.exception(
                    "vault_watcher_embed_file_failed",
                    extra={"file_id": file_id, "error": str(exc)},
                )
                # PR #113c (B7 fix): per-file failure metric. The
                # embedder-level `vault_embedder_embed_failed` is
                # already emitted inside embed_file(); this one
                # measures Watcher-side errors (KeyError, asyncio.
                # CancelledError, RuntimeError on chunk-count-mismatch,
                # OSError on transient DB hiccups) which are a
                # different signal.
                # PR #113c round 3 BLOCKING fix: previous version
                # used `tags={"error_type": type(exc).__name__}` —
                # the SAME unbounded-tag bug round 2 corrected for
                # the embedder-side metric. Fixed here with the
                # bounded `_classify_exception` taxonomy + raw
                # class name as a field (queryable but doesn't
                # create new series).
                write_research_metric(
                    "vault_watcher_embed_file_failed",
                    tags={"error_taxonomy": _classify_exception(exc)},
                    fields={
                        "files": 1,
                        "exception_class": type(exc).__name__,
                    },
                )
            finally:
                async with self._embedder._in_flight_lock:
                    self._embedder._in_flight.discard(file_id)

        if embedded_count > 0:
            logger.info(
                "vault_watcher_cycle_done",
                extra={"embedded": embedded_count},
            )
        # PR #113c (B7 fix): per-cycle metric so the operator
        # can chart "files embedded per cycle" trend.
        # PR #113c round 2 (MAJOR 2): emit on every non-empty
        # cycle (not just successful ones). The previous `if
        # embedded_count > 0` gate SILENTLY DROPPED the
        # observability signal precisely when the operator needs
        # it most: a Watcher tick that found candidates but ALL
        # of them failed (e.g. embedder backend down). Result
        # tag is now an enum {ok, partial, all_failed} so the
        # Grafana dashboard can alert on `result="all_failed"`
        # rate spikes.
        if candidates:
            if embedded_count == 0:
                result = "all_failed"
            elif embedded_count < len(candidates):
                result = "partial"
            else:
                result = "ok"
            write_research_metric(
                "vault_watcher_cycle",
                tags={"result": result},
                fields={"embedded": embedded_count, "candidates": len(candidates)},
            )
        return embedded_count

    async def run(self) -> None:
        """Long-running poll loop. Stop with `stop()`."""
        logger.info(
            "vault_watcher_started",
            extra={"interval_s": self._interval_s},
        )
        while not self._stop_event.is_set():
            try:
                await self.run_once()
            except Exception as exc:
                # Outer guard: any unhandled error in run_once (e.g.
                # DB down). Log + continue.
                logger.exception(
                    "vault_watcher_iteration_error",
                    extra={"error": str(exc)},
                )
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self._interval_s,
                )
                # wait() returned → stop was called.
                break
            except TimeoutError:
                # Normal: interval elapsed, next cycle.
                pass
        logger.info("vault_watcher_stopped")

    def stop(self) -> None:
        """Signal the run() loop to exit."""
        self._stop_event.set()

    async def shutdown(self, *, timeout_s: float = 10.0) -> bool:
        """Sprint 18 hardening (Gemini P0 #2): bounded shutdown.

        Signals stop_event, then waits up to `timeout_s` for the
        in-flight embed to complete. Returns True if the watcher
        drained cleanly within the timeout, False if it had to be
        abandoned.

        This is the ÉPICA 1 contract: Hermes must shut down within
        the Docker grace period (default 10s). Without this method,
        EmbedWatcherScheduler.shutdown() blocks on APScheduler's
        `wait=True`, which waits for the currently-running LLM call
        to complete. With OpenRouter free tier, that can be 60+ seconds,
        and Docker SIGKILLs the container mid-LLM — losing tokens,
        leaving the DB lock held, risking connection poisoning.

        Behavior:
        1. `self._stop_event.set()` — watched between files in run_once().
        2. Wait up to `timeout_s` for the in-flight set to drain.
        3. If drain completes within timeout: return True.
        4. If timeout elapses: log warning, return False (caller decides
           whether to force-shutdown the scheduler).

        Args:
            timeout_s: max seconds to wait for in-flight embed to drain.
                Default 10.0 matches Docker's default SIGTERM grace.

        Returns:
            True if watcher drained within timeout, False otherwise.
        """
        # Signal the run() loop + run_once() between-files bail.
        self._stop_event.set()

        if not self._embedder._in_flight:
            logger.debug("vault_watcher_shutdown_no_inflight")
            return True

        # Wait for the in-flight set to drain. Bounded by timeout_s.
        # Poll with short sleeps (avoids asyncio.wait_for on a sleep
        # which can itself be cancelled, and avoids task leaks).
        #
        # PR #119 LLM review (MiMo v2.5, BLOCKING #2): the `_in_flight`
        # set is read here without the lock. Safe because (a) asyncio is
        # single-threaded so the set is only mutated at await points,
        # (b) `discard()` is atomic in CPython, and (c) once stop_event
        # is set, the only mutation possible is REMOVAL (run_once bails
        # out between files, so the set can only shrink — never grow).
        # Therefore the loop always terminates: either the set drains
        # within timeout, or the deadline is hit and we return False.
        # PR #119 LLM review (SUGGESTION #4): use get_running_loop()
        # instead of deprecated get_event_loop().
        deadline = asyncio.get_running_loop().time() + timeout_s
        try:
            while self._embedder._in_flight:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    break
                await asyncio.sleep(min(remaining, 0.05))
        except Exception:
            # Cancellation or unexpected error: still try to drain.
            logger.exception("vault_watcher_shutdown_wait_error")

        drained = not self._embedder._in_flight
        if drained:
            logger.info(
                "vault_watcher_shutdown_drained",
                extra={"timeout_s": timeout_s},
            )
        else:
            logger.warning(
                "vault_watcher_shutdown_timeout",
                extra={
                    "timeout_s": timeout_s,
                    "still_in_flight": list(self._embedder._in_flight),
                },
            )
        return drained


__all__ = [
    "EmbedWatcher",
    "SearchHit",
    "VaultEmbedder",
]

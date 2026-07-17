"""OcrPendingRepo — Sprint 19 Slice 4 (§4.3).

CRUD para la tabla `ocr_pending`: cola para archivos cuya extracción
de texto local (Tesseract) dio confianza < 0.85 (Sprint 19 §4.1). Las
filas esperan:
- Auto-queue al edge PC (Sprint 19 §4.4.1) si el PC está online, o
- Decisión manual del usuario (`/acceptNull`, `/externalOCR`, etc.).

NO es el watcher (eso es `drop_watcher.py`); NO es la lógica de
auto-queue al edge (eso es `edge_coordinator.py`, Slice 4c). Este
módulo es solo data access.

Schema (migration v22):
- file_id TEXT PRIMARY KEY (FK -> vault_files.file_id ON DELETE CASCADE)
- local_confidence REAL NOT NULL  -- 0.0 a 1.0 (Tesseract)
- local_text TEXT                  -- NULL si Tesseract no extrajo nada
- local_model TEXT NOT NULL        -- 'tesseract-5' (futuro: 'paddleocr-v4')
- status TEXT NOT NULL DEFAULT 'pending_review'
    -- 'pending_review' | 'edge_queued' | 'edge_processed' | 'edge_failed'
    -- | 'accepted_null' | 'manually_edited' | 'external_processed' | 'user_skipped'
- external_model TEXT              -- hosted VLM si project owner escaló
- external_confidence REAL
- edge_model TEXT                  -- 'pc-tesseract-5' | 'pc-llava-7b-q4'
- edge_queued_at TEXT              -- ISO 8601 UTC
- edge_processed_at TEXT           -- ISO 8601 UTC
- created_at TEXT NOT NULL
- updated_at TEXT NOT NULL
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import aiosqlite

    from hermes.memory.db import Database

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    """ISO 8601 UTC timestamp, second precision. Format: YYYY-MM-DDTHH:MM:SSZ."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# Status enum (Sprint 19 §4.3). String literal type para no cerrar la
# puerta a status nuevos en el futuro (Sprint 20+).
OcrPendingStatus = str  # Literal["pending_review", "edge_queued", ...]


@dataclass(frozen=True, slots=True)
class OcrPendingRow:
    """Row de `ocr_pending`. Inmutable."""

    file_id: str
    local_confidence: float
    local_text: str | None
    local_model: str
    status: str
    external_model: str | None
    external_confidence: float | None
    edge_model: str | None
    edge_queued_at: str | None
    edge_processed_at: str | None
    created_at: str
    updated_at: str

    @classmethod
    def from_row(cls, row: aiosqlite.Row) -> OcrPendingRow:
        return cls(
            file_id=row["file_id"],
            local_confidence=row["local_confidence"],
            local_text=row["local_text"],
            local_model=row["local_model"],
            status=row["status"],
            external_model=row["external_model"],
            external_confidence=row["external_confidence"],
            edge_model=row["edge_model"],
            edge_queued_at=row["edge_queued_at"],
            edge_processed_at=row["edge_processed_at"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


class OcrPendingRepo:
    """Repositorio para la tabla `ocr_pending`.

    Diseño:
    - Una instancia por `Database` (cheap; solo guarda referencia).
    - Reads no toman el write_lock (WAL permite concurrent reads).
    - Writes serializan vía `Database._write_lock` (F-CONC-3).
    - Cada write es una transacción atómica con `BEGIN IMMEDIATE` / COMMIT
      / ROLLBACK.
    - Rowcount via SQLite `changes()` (aiosqlite cursor.rowcount unreliable).
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    @asynccontextmanager
    async def _write_section(self, op: str):
        """Acquire el write-lock del Database. Telemetría básica."""
        import asyncio

        t0 = asyncio.get_event_loop().time()
        await self._db._write_lock.acquire()
        try:
            yield
        finally:
            held_ms = (asyncio.get_event_loop().time() - t0) * 1000
            if held_ms > 1000:
                logger.warning(
                    "ocr_pending_lock_long_held",
                    extra={"op": op, "held_ms": round(held_ms, 1)},
                )
            self._db._write_lock.release()

    async def get(self, file_id: str) -> OcrPendingRow | None:
        """Lee una fila por file_id. None si no existe."""
        async with self._db.conn.execute(
            "SELECT * FROM ocr_pending WHERE file_id = ?",
            (file_id,),
        ) as cur:
            row = await cur.fetchone()
        return OcrPendingRow.from_row(row) if row else None

    async def list_by_status(
        self,
        status: str,
        limit: int = 100,
    ) -> list[OcrPendingRow]:
        """Lista filas con un status dado, FIFO por created_at."""
        async with self._db.conn.execute(
            "SELECT * FROM ocr_pending WHERE status = ? ORDER BY created_at ASC LIMIT ?",
            (status, limit),
        ) as cur:
            rows = await cur.fetchall()
        return [OcrPendingRow.from_row(r) for r in rows]

    async def create(
        self,
        file_id: str,
        local_confidence: float,
        local_text: str | None,
        local_model: str,
        status: str = "pending_review",
    ) -> OcrPendingRow:
        """Inserta una fila nueva. Idempotente: si ya existe, no hace nada
        (devuelve la fila existente).

        FK: file_id debe existir en vault_files. El caller (drop_watcher)
        es responsable de insertar en vault_files ANTES de llamar aquí.

        Raises:
            sqlite3.IntegrityError: si file_id no existe en vault_files.
        """
        now = _utc_now_iso()
        async with self._write_section("create"):
            await self._db.conn.execute("BEGIN IMMEDIATE")
            try:
                try:
                    await self._db.conn.execute(
                        """
                        INSERT INTO ocr_pending
                            (file_id, local_confidence, local_text, local_model,
                             status, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (file_id, local_confidence, local_text, local_model, status, now, now),
                    )
                    await self._db.conn.execute("COMMIT")
                except sqlite3.IntegrityError as e:
                    msg = str(e)
                    if "UNIQUE constraint failed" in msg and "ocr_pending.file_id" in msg:
                        await self._db.conn.execute("ROLLBACK")
                        existing = await self.get(file_id)
                        assert existing is not None
                        return existing
                    raise
            except BaseException:
                with suppress(Exception):
                    await self._db.conn.execute("ROLLBACK")
                raise
        result = await self.get(file_id)
        assert result is not None
        return result

    async def update_status(
        self,
        file_id: str,
        new_status: str,
        **fields: str | float | None,
    ) -> bool:
        """Actualiza status (y opcionalmente otros campos) de una fila.

        Fields opcionales: external_model, external_confidence, edge_model,
        edge_queued_at, edge_processed_at, local_text, local_confidence.

        Returns: True si actualizado, False si la fila no existe.

        Uso típico:
            await repo.update_status(
                file_id,
                "edge_queued",
                edge_model="pc-tesseract-5",
                edge_queued_at=utc_now_iso(),
            )
        """
        if not fields:
            fields = {}
        # Whitelist de campos updatable (defense in depth: evita typos)
        allowed = {
            "status",
            "external_model",
            "external_confidence",
            "edge_model",
            "edge_queued_at",
            "edge_processed_at",
            "local_text",
            "local_confidence",
        }
        sets: list[str] = ["updated_at = ?"]
        params: list = [_utc_now_iso()]
        for k, v in fields.items():
            if k not in allowed:
                raise ValueError(f"field not in whitelist: {k!r}")
            sets.append(f"{k} = ?")
            params.append(v)
        sets.append("status = ?")
        params.append(new_status)
        params.append(file_id)
        sql = f"UPDATE ocr_pending SET {', '.join(sets)} WHERE file_id = ?"

        async with self._write_section("update_status"):
            await self._db.conn.execute("BEGIN IMMEDIATE")
            try:
                cur = await self._db.conn.execute(sql, params)
                changes = cur.rowcount
                await cur.close()
                await self._db.conn.execute("COMMIT")
                return changes > 0
            except BaseException:
                with suppress(Exception):
                    await self._db.conn.execute("ROLLBACK")
                raise

    # ------------------------------------------------------------------
    # Sprint 19 Slice 4c (§4.4.4 + §4.4.5): catch-up pass + zombie recovery
    # ------------------------------------------------------------------

    async def fetch_pending_for_catchup(
        self,
        path_extensions: list[str],
        limit: int,
    ) -> list[tuple[str, str, float]]:
        """Fetch (file_id, path, local_confidence) tuples for files in
        `pending_review` whose vault path ends with one of the given
        extensions (e.g. ['.jpg', '.jpeg', '.png']). FIFO by created_at.

        Used by the edge coordinator's catch-up pass on offline→online
        transition (TDD §4.4.4). Only image files qualify for auto-queue;
        PDFs/DOCX/XLSX skip the edge (they have local text extractors).

        Note: the vault_files column is `source_path` (not `path` as the
        TDD prose says — schema is the source of truth). The path is
        POSIX-formatted (forward slashes) per hermes/util/paths.py:to_posix.

        Returns: list of (file_id, posix_path, local_confidence) tuples.

        Edge case: if `path_extensions` is empty, returns [].

        Perf note (Sprint 19 LLM review 2026-07-11): local_confidence is
        included in the SELECT to avoid an N+1 query in the catch-up
        loop (was previously fetched per-row via ocr_repo.get()).
        """
        if not path_extensions:
            return []
        # Build "LIKE ?" placeholders for each extension.
        # LIKE '%.jpg' matches both '/path/to/foo.jpg' and '/path/.jpg_hidden'
        # — we accept the latter as a tradeoff for SQL simplicity.
        like_clauses = " OR ".join(["vf.source_path LIKE ?"] * len(path_extensions))
        like_args = [f"%{ext}" for ext in path_extensions]
        async with self._db.conn.execute(
            f"""
            SELECT op.file_id, vf.source_path AS path, op.local_confidence
            FROM ocr_pending op
            JOIN vault_files vf ON op.file_id = vf.file_id
            WHERE op.status = 'pending_review'
              AND ({like_clauses})
            ORDER BY op.created_at ASC
            LIMIT ?
            """,
            (*like_args, limit),
        ) as cur:
            rows = await cur.fetchall()
        return [(r["file_id"], r["path"], r["local_confidence"]) for r in rows]

    async def fetch_zombie_candidates(
        self,
        cutoff_iso: str,
    ) -> list[OcrPendingRow]:
        """Fetch rows in `edge_queued` whose `edge_queued_at` is older than
        the cutoff (ISO 8601 string).

        Used by M6 Phase 5 zombie recovery (TDD §4.4.5). Partial index
        `idx_ocr_pending_edge_queued_at` makes this O(rows_to_recover)
        even when `ocr_pending` has millions of rows.

        Returns: list of OcrPendingRow.
        """
        async with self._db.conn.execute(
            """
            SELECT * FROM ocr_pending
            WHERE status = 'edge_queued'
              AND edge_queued_at IS NOT NULL
              AND edge_queued_at < ?
            ORDER BY edge_queued_at ASC
            """,
            (cutoff_iso,),
        ) as cur:
            rows = await cur.fetchall()
        return [OcrPendingRow.from_row(r) for r in rows]

    async def revert_to_pending(self, file_id: str) -> bool:
        """M6 Phase 5: revert a row from `edge_queued` back to
        `pending_review`, clearing `edge_queued_at` (and `edge_model`).

        Idempotent: if row is already in `pending_review` (race with
        manual action), the UPDATE is a no-op and we return False.

        Returns: True if the row was reverted, False otherwise.
        """
        async with self._write_section("revert_to_pending"):
            await self._db.conn.execute("BEGIN IMMEDIATE")
            try:
                cur = await self._db.conn.execute(
                    """
                    UPDATE ocr_pending
                    SET status = 'pending_review',
                        edge_queued_at = NULL,
                        edge_model = NULL,
                        updated_at = ?
                    WHERE file_id = ? AND status = 'edge_queued'
                    """,
                    (_utc_now_iso(), file_id),
                )
                changes = cur.rowcount
                await cur.close()
                await self._db.conn.execute("COMMIT")
                return changes > 0
            except BaseException:
                with suppress(Exception):
                    await self._db.conn.execute("ROLLBACK")
                raise

"""VaultCollectionsRepo — Sprint 19 Slice 1.

Repositorio para las tablas `vault_collections` y `vault_file_collections`,
más gestión del campo `vault_files.orphaned_at`.

CRITICAL contracts (no negociar):
- `vault_collections`: jerárquica, soft-delete via flag `archived` (0/1).
- `vault_file_collections`: APPEND-ONLY bridge (Gemini Sprint 19 review).
  NUNCA borrar rows en archive. Las queries de lectura JOIN-filtran
  `WHERE archived = 0`. El único `DELETE FROM vault_file_collections`
  legítimo es la acción explícita del usuario ("unlink") o recovery de error.
- `vault_files.orphaned_at`: se setea cuando el archivo físico desaparece
  vía SMB. Text + embeddings persisten (audit trail / second-brain value).
  Search filtra `WHERE orphaned_at IS NULL`.
- Cascade archive via recursive CTE con depth ≤ 20 (Gemini Option A, EC-5).
  NO cascade unarchive. NO auto-reparent.
- All writes serializados por `Database._write_lock` (aiosqlite single-thread
  + asyncio.Lock per-Database — F-CONC-3).
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import uuid
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import aiosqlite

    from hermes.memory.db import Database

logger = logging.getLogger(__name__)


# Maximum depth for cascade archive (Gemini EC-5: protect against self-referencing
# cycles or pathological chains). 20 covers realistic PARA nesting (PARA → Project
# → Sub-project → Topic → Sub-topic = 5 levels; 20 is comfortable headroom).
_CASCADE_DEPTH_CAP = 20


def _utc_now_iso() -> str:
    """ISO 8601 UTC timestamp, second precision. Format: YYYY-MM-DDTHH:MM:SSZ."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _new_collection_id() -> str:
    """UUID4 hex without dashes. 32 lowercase hex chars."""
    return uuid.uuid4().hex


async def _rowcount(conn: aiosqlite.Connection) -> int:
    """Return the number of rows changed by the last INSERT/UPDATE/DELETE.

    Workaround for aiosqlite: `cursor.rowcount` is unreliable in some
    contexts (returns -1 for DML after BEGIN IMMEDIATE). Using SQLite's
    built-in `changes()` function is the canonical reliable way.

    Must be called BEFORE COMMIT — `changes()` resets on commit.
    """
    async with conn.execute("SELECT changes()") as cur:
        row = await cur.fetchone()
    return int(row[0]) if row else 0


@dataclass(frozen=True, slots=True)
class Collection:
    """Row from `vault_collections`. Inmutable."""

    collection_id: str
    name: str
    parent_collection_id: str | None
    description: str | None
    sort_order: int
    archived: bool
    archived_at: str | None
    created_at: str

    @classmethod
    def from_row(cls, row: aiosqlite.Row) -> Collection:
        return cls(
            collection_id=row["collection_id"],
            name=row["name"],
            parent_collection_id=row["parent_collection_id"],
            description=row["description"],
            sort_order=row["sort_order"],
            archived=bool(row["archived"]),
            archived_at=row["archived_at"],
            created_at=row["created_at"],
        )


class CollectionNotFoundError(LookupError):
    """Raised when a collection_id is not in vault_collections."""


class DuplicateCollectionError(ValueError):
    """Raised when the UNIQUE constraint on vault_collections.name fires."""


class VaultCollectionsRepo:
    """Repositorio para vault_collections + vault_file_collections + vault_files.orphaned_at.

    Diseño:
    - Una instancia por `Database` (cheap; solo guarda referencia).
    - Reads no toman el write_lock (WAL permite concurrent reads).
    - Writes serializan vía `Database._write_lock` (F-CONC-3).
    - Cada write es una transacción atómica con `BEGIN IMMEDIATE` /
      `COMMIT` / `ROLLBACK`.
    - Rowcount via SQLite `changes()` (aiosqlite cursor.rowcount unreliable).
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    # ------------------------------------------------------------------
    # Write lock helper (mirrors hermes/memory/vault.py:_write_section
    # pero con telemetría mínima — collections son writes de baja
    # frecuencia comparados con vault_files ingestion).
    # ------------------------------------------------------------------
    @asynccontextmanager
    async def _write_section(self, op: str):
        """Acquire el write-lock del Database. Telemetría básica."""
        t0 = asyncio.get_event_loop().time()
        await self._db._write_lock.acquire()
        try:
            yield
        finally:
            held_ms = (asyncio.get_event_loop().time() - t0) * 1000
            if held_ms > 1000:
                logger.warning(
                    "collections_lock_long_held",
                    extra={"op": op, "held_ms": round(held_ms, 1)},
                )
            self._db._write_lock.release()

    # ------------------------------------------------------------------
    # CRUD: collections
    # ------------------------------------------------------------------
    async def create_collection(
        self,
        name: str,
        parent_collection_id: str | None = None,
        description: str | None = None,
        sort_order: int = 0,
    ) -> Collection:
        """Crea una collection nueva.

        Raises:
            ValueError: si `name` está vacío o solo whitespace.
            CollectionNotFoundError: si `parent_collection_id` no existe.
            DuplicateCollectionError: si `name` ya está tomado (UNIQUE).
        """
        if not name or not name.strip():
            raise ValueError("collection name cannot be empty")
        name = name.strip()
        cid = _new_collection_id()
        now = _utc_now_iso()

        async with self._write_section("create_collection"):
            await self._db.conn.execute("BEGIN IMMEDIATE")
            try:
                if parent_collection_id is not None:
                    cur = await self._db.conn.execute(
                        "SELECT 1 FROM vault_collections WHERE collection_id = ?",
                        (parent_collection_id,),
                    )
                    row = await cur.fetchone()
                    if row is None:
                        raise CollectionNotFoundError(
                            f"parent_collection_id={parent_collection_id} does not exist"
                        )
                try:
                    await self._db.conn.execute(
                        """
                        INSERT INTO vault_collections
                            (collection_id, name, parent_collection_id, description,
                             sort_order, archived, archived_at, created_at)
                        VALUES (?, ?, ?, ?, ?, 0, NULL, ?)
                        """,
                        (cid, name, parent_collection_id, description, sort_order, now),
                    )
                except sqlite3.IntegrityError as e:
                    msg = str(e)
                    # Sprint 19 Slice 4d v2: v23 migration changed UNIQUE on
                    # vault_collections.name (GLOBAL) to UNIQUE on
                    # (name COLLATE NOCASE, parent_collection_id) via
                    # idx_vault_collections_name_parent. The error message
                    # may name the constraint as either the column or the
                    # index, so check for either form.
                    if "UNIQUE constraint failed" in msg and (
                        "vault_collections" in msg or "idx_vault_collections_name_parent" in msg
                    ):
                        raise DuplicateCollectionError(
                            f"collection name already exists: {name!r}"
                        ) from e
                    raise
                await self._db.conn.execute("COMMIT")
            except BaseException:
                # ROLLBACK is best-effort: aiosqlite may auto-commit if the
                # BEGIN failed, in which case ROLLBACK raises.
                with suppress(Exception):
                    await self._db.conn.execute("ROLLBACK")
                raise

        result = await self.get_collection(cid)
        assert result is not None  # We just inserted it.
        return result

    async def get_collection(self, collection_id: str) -> Collection | None:
        async with self._db.conn.execute(
            "SELECT * FROM vault_collections WHERE collection_id = ?",
            (collection_id,),
        ) as cur:
            row = await cur.fetchone()
        return Collection.from_row(row) if row else None

    async def get_collection_by_name(self, name: str) -> Collection | None:
        async with self._db.conn.execute(
            "SELECT * FROM vault_collections WHERE name = ?",
            (name,),
        ) as cur:
            row = await cur.fetchone()
        return Collection.from_row(row) if row else None

    async def get_active_collection_by_name(self, name: str) -> Collection | None:
        """Como `get_collection_by_name` pero filtra `archived=0`.

        Usado por el DropWatcher (Sprint 19 Slice 4 §4): si el subdir del
        drop folder matchea una colección ARCHIVADA, NO se linkean archivos
        nuevos a ella (el usuario probablemente no quiere eso). En su lugar,
        el watcher skipea el archivo con warning `collection_archived` y el
        usuario decide: o unarchiva la colección o usa otro subdir.

        Returns: la Collection activa (archived=0) con ese name, o None si
        no existe o está archivada. El caller debe distinguir ambos casos
        si quiere reportar "archivada" vs "no existe".
        """
        async with self._db.conn.execute(
            "SELECT * FROM vault_collections WHERE name = ? AND archived = 0",
            (name,),
        ) as cur:
            row = await cur.fetchone()
        return Collection.from_row(row) if row else None

    async def find_by_name_and_parent(
        self,
        name: str,
        parent_collection_id: str | None,
        *,
        case_insensitive: bool = True,
    ) -> Collection | None:
        """Case-insensitive lookup at a specific parent level.

        Sprint 19 Slice 4d v2 (TDD_VAULT_COLLECTIONS_v0.5 §11 MAJOR-4):
        - Uses Python-side `.casefold()` for matching, NOT SQL COLLATE NOCASE.
          Reason: COLLATE NOCASE is ASCII-only; PARA has accents
          (02_Áreas_de_Responsabilidad) that COLLATE NOCASE doesn't match.
          `.casefold()` is unicode-correct (handles German ß→ss, Greek
          final sigma, etc.).
        - Default case_insensitive=True. Pass False for exact match (rare).
        - The composite UNIQUE index idx_vault_collections_name_parent
          (v23) uses COLLATE NOCASE + COALESCE(NULL, '<<ROOT>>') for the
          SQL-level enforcement. The Python-side match is a SECOND line of
          defense (and handles unicode correctly).

        Returns: the matching Collection (active or archived), or None.
        """
        target_cf = name.casefold() if case_insensitive else name
        # Query the small set of rows at the given parent level and filter
        # in Python. The collections table is small (4 default + a few
        # user-created); SQL-side COLLATE NOCASE filtering is not worth a
        # functional index for this use case.
        sql = """
        SELECT collection_id, name, parent_collection_id, description,
               sort_order, archived, archived_at, created_at
        FROM vault_collections
        WHERE (parent_collection_id IS ? OR parent_collection_id = ?)
        """
        params: tuple[Any, ...] = (parent_collection_id, parent_collection_id)
        async with self._db.conn.execute(sql, params) as cur:
            async for row in cur:
                row_name = row[1]
                row_cf = row_name.casefold() if case_insensitive else row_name
                if row_cf == target_cf:
                    return Collection(
                        collection_id=row[0],
                        name=row_name,  # preserve original case
                        parent_collection_id=row[2],
                        description=row[3],
                        sort_order=row[4],
                        archived=bool(row[5]),
                        archived_at=row[6],
                        created_at=row[7],
                    )
        return None

    async def list_collections(
        self,
        include_archived: bool = False,
        parent_collection_id: str | None = None,
    ) -> list[Collection]:
        sql = "SELECT * FROM vault_collections"
        params: list = []
        clauses: list[str] = []
        if not include_archived:
            clauses.append("archived = 0")
        if parent_collection_id is not None:
            clauses.append("parent_collection_id = ?")
            params.append(parent_collection_id)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        # Roots first (parent_collection_id IS NULL = 1, children = 0).
        # Critical: agent tools cap the flat list at _MAX_TREE_NODES; if
        # children sort alphabetically before parents, the cap would
        # retain ONLY children and 0 roots visible to LLM (regression
        # caught in adversarial review R2 MAJOR-1, fixed 2026-07-10).
        sql += " ORDER BY (parent_collection_id IS NULL) DESC, sort_order, name"

        async with self._db.conn.execute(sql, params) as cur:
            rows = await cur.fetchall()
        return [Collection.from_row(r) for r in rows]

    async def archive_collection(
        self,
        collection_id: str,
        cascade: bool = True,
    ) -> int:
        """Soft-archive a collection. Si `cascade=True`, archive todos los
        descendientes hasta depth=20 (Gemini EC-5 Option A).

        NUNCA borra rows. NUNCA borra vault_file_collections rows.
        El bridge se queda; las queries de lectura JOIN-filtran archived=0.

        Returns: número de rows archivadas (0 si ya estaba archivada).
        Raises: CollectionNotFoundError si el id no existe (pre-check).
        """
        now = _utc_now_iso()
        async with self._write_section("archive_collection"):
            await self._db.conn.execute("BEGIN IMMEDIATE")
            try:
                # Existence check pre-update (so missing → raise, not silent no-op)
                cur = await self._db.conn.execute(
                    "SELECT 1 FROM vault_collections WHERE collection_id = ?",
                    (collection_id,),
                )
                row = await cur.fetchone()
                if row is None:
                    raise CollectionNotFoundError(collection_id)

                if cascade:
                    # Recursive CTE: archive the parent + descendants up to depth=20.
                    # Semantics: _CASCADE_DEPTH_CAP=20 means 20 TOTAL levels
                    # (head at depth 0 + 19 descendants at depths 1..19). The
                    # recursive step's WHERE stops when current depth is at
                    # the second-to-last level (19), so we never expand
                    # beyond depth 19. This protects against self-referencing
                    # cycles.
                    sql = f"""
                    WITH RECURSIVE descendants(id, depth) AS (
                        SELECT collection_id, 0 FROM vault_collections
                        WHERE collection_id = ?
                        UNION ALL
                        SELECT vc.collection_id, d.depth + 1
                        FROM vault_collections vc
                        JOIN descendants d ON vc.parent_collection_id = d.id
                        WHERE d.depth < {_CASCADE_DEPTH_CAP - 1}
                    )
                    UPDATE vault_collections
                    SET archived = 1, archived_at = ?
                    WHERE collection_id IN (SELECT id FROM descendants)
                      AND archived = 0
                    """
                    await self._db.conn.execute(sql, (collection_id, now))
                else:
                    await self._db.conn.execute(
                        """
                        UPDATE vault_collections
                        SET archived = 1, archived_at = ?
                        WHERE collection_id = ? AND archived = 0
                        """,
                        (now, collection_id),
                    )
                rowcount = await _rowcount(self._db.conn)
                await self._db.conn.execute("COMMIT")
                return rowcount
            except BaseException:
                with suppress(Exception):
                    await self._db.conn.execute("ROLLBACK")
                raise

    async def restore_collection(self, collection_id: str) -> int:
        """Restaura una collection archivada. NO cascade unarchive (cada nivel
        se restaura independientemente — Gemini EC-5).

        Returns: 1 si restaurada, 0 si ya estaba activa o no existe.
        """
        async with self._write_section("restore_collection"):
            await self._db.conn.execute("BEGIN IMMEDIATE")
            try:
                await self._db.conn.execute(
                    """
                    UPDATE vault_collections
                    SET archived = 0, archived_at = NULL
                    WHERE collection_id = ? AND archived = 1
                    """,
                    (collection_id,),
                )
                rowcount = await _rowcount(self._db.conn)
                await self._db.conn.execute("COMMIT")
                return rowcount
            except BaseException:
                with suppress(Exception):
                    await self._db.conn.execute("ROLLBACK")
                raise

    # ------------------------------------------------------------------
    # Bridge: vault_file_collections
    # ------------------------------------------------------------------
    async def add_file_to_collection(self, file_id: str, collection_id: str) -> bool:
        """Inserta una row en el bridge. Idempotente (UNIQUE PK).

        Returns: True si insertada, False si ya existía (UNIQUE).
        Raises:
            CollectionNotFoundError: si `collection_id` no existe.
            sqlite3.IntegrityError: si `file_id` no existe en vault_files
                (FK ON DELETE CASCADE desde vault_files → aquí es REJECT).
                Pasamos el error al caller; contractualmente debería
                pre-crearse el vault_files row antes de añadir al bridge.
        """
        now = _utc_now_iso()
        async with self._write_section("add_file_to_collection"):
            await self._db.conn.execute("BEGIN IMMEDIATE")
            try:
                cur = await self._db.conn.execute(
                    "SELECT 1 FROM vault_collections WHERE collection_id = ?",
                    (collection_id,),
                )
                row = await cur.fetchone()
                if row is None:
                    raise CollectionNotFoundError(collection_id)

                try:
                    await self._db.conn.execute(
                        """
                        INSERT INTO vault_file_collections
                            (file_id, collection_id, added_at)
                        VALUES (?, ?, ?)
                        """,
                        (file_id, collection_id, now),
                    )
                    await self._db.conn.execute("COMMIT")
                    return True
                except sqlite3.IntegrityError as e:
                    msg = str(e)
                    if "UNIQUE constraint failed" in msg or "PRIMARY KEY constraint failed" in msg:
                        await self._db.conn.execute("ROLLBACK")
                        return False
                    raise
            except BaseException:
                with suppress(Exception):
                    await self._db.conn.execute("ROLLBACK")
                raise

    async def remove_file_from_collection(self, file_id: str, collection_id: str) -> bool:
        """Borra la row del bridge. Idempotente.

        ⚠️ USO RESTRINGIDO: el bridge es contractualmente append-only.
        Esta función solo debe llamarse desde acciones explícitas del
        usuario ("unlink") o desde recovery de errores. Archive NO
        borra del bridge — JOIN-filter se encarga.

        Returns: True si removida, False si no existía.
        """
        async with self._write_section("remove_file_from_collection"):
            await self._db.conn.execute("BEGIN IMMEDIATE")
            try:
                await self._db.conn.execute(
                    """
                    DELETE FROM vault_file_collections
                    WHERE file_id = ? AND collection_id = ?
                    """,
                    (file_id, collection_id),
                )
                rowcount = await _rowcount(self._db.conn)
                await self._db.conn.execute("COMMIT")
                return rowcount > 0
            except BaseException:
                with suppress(Exception):
                    await self._db.conn.execute("ROLLBACK")
                raise

    async def _move_bridge(
        self,
        file_id: str,
        old_collection_id: str | None,
        new_collection_id: str | None,
    ) -> dict[str, bool]:
        """Sprint 19 Slice 4d v2: soft-delete old bridge + insert new bridge.

        This is the watcher's move detection (R1 #3 B-2 fix). It does NOT
        use the existing `remove_file_from_collection` (which does a
        hard DELETE — that's reserved for user-initiated unlinks, per
        the append-only contract on vault_file_collections).

        Behavior:
        - old_collection_id == new_collection_id: no-op, return both False
        - old_collection_id=None: skip the soft-delete, just insert new
        - new_collection_id=None: skip the insert, just soft-delete old
        - both: UPDATE old bridge superseded_at = NOW(); INSERT new
          bridge (superseded_at=NULL), idempotent on PK

        Must be called inside a BEGIN IMMEDIATE transaction (caller
        responsibility). The methods that orchestrate this (DropWatcher,
        M6ReconcileScheduler) hold the write_lock + BEGIN IMMEDIATE.

        Returns: {"old_superseded": bool, "new_inserted": bool}
        """
        result: dict[str, bool] = {"old_superseded": False, "new_inserted": False}

        if old_collection_id == new_collection_id:
            return result

        now = _utc_now_iso()

        # Soft-delete old bridge
        if old_collection_id is not None:
            cursor = await self._db.conn.execute(
                "UPDATE vault_file_collections "
                "SET superseded_at = ? "
                "WHERE file_id = ? AND collection_id = ? "
                "AND superseded_at IS NULL",
                (now, file_id, old_collection_id),
            )
            result["old_superseded"] = cursor.rowcount > 0

        # Insert new bridge (idempotent on PK: file_id + collection_id)
        if new_collection_id is not None:
            try:
                await self._db.conn.execute(
                    "INSERT INTO vault_file_collections "
                    "(file_id, collection_id, added_at, superseded_at) "
                    "VALUES (?, ?, ?, NULL)",
                    (file_id, new_collection_id, now),
                )
                result["new_inserted"] = True
            except sqlite3.IntegrityError:
                # Already linked (file already in target collection) — no-op.
                # Sprint 19 Slice 4d v2 v0.6 §1 spec: "Idempotent on PK".
                pass

        return result

    async def list_files_in_collection(
        self,
        collection_id: str,
        *,
        include_orphans: bool = False,
    ) -> list[str]:
        """Lista file_ids en una collection, ordenados por added_at DESC.

        Filtros:
        - Excluye archivos cuya collection está archivada (siempre, no param).
        - Si include_orphans=False (default): excluye vault_files.orphaned_at
          no nulos.
        - Sprint 19 Slice 4d v2: filtra `vfc.superseded_at IS NULL` para no
          mostrar "phantom" files (mismo file_id listado 2 veces — la versión
          superseded + la nueva). v23 migration v0.4 §0 B4.
        """
        sql = """
        SELECT vf.file_id
        FROM vault_file_collections vfc
        JOIN vault_collections vc ON vc.collection_id = vfc.collection_id
        JOIN vault_files vf ON vf.file_id = vfc.file_id
        WHERE vfc.collection_id = ?
          AND vc.archived = 0
          AND vfc.superseded_at IS NULL
        """
        params: list = [collection_id]
        if not include_orphans:
            sql += " AND vf.orphaned_at IS NULL"
        sql += " ORDER BY vfc.added_at DESC"

        async with self._db.conn.execute(sql, params) as cur:
            rows = await cur.fetchall()
        return [r["file_id"] for r in rows]

    async def list_collections_for_file(self, file_id: str) -> list[Collection]:
        """Lista collections ACTIVAS (no archivadas) que contienen este file.

        Sprint 19 Slice 4d v2: filtra `vfc.superseded_at IS NULL` para no
        listar collections con bridges superseded (Gemini EC-5 phantom
        files warning). Si el file se movió de collection A a B, A
        aparece como superseded (no listada) y B como activa (listada).
        """
        async with self._db.conn.execute(
            """
            SELECT vc.*
            FROM vault_collections vc
            JOIN vault_file_collections vfc ON vfc.collection_id = vc.collection_id
            WHERE vfc.file_id = ? AND vc.archived = 0
              AND vfc.superseded_at IS NULL
            ORDER BY vc.sort_order, vc.name
            """,
            (file_id,),
        ) as cur:
            rows = await cur.fetchall()
        return [Collection.from_row(r) for r in rows]

    # ------------------------------------------------------------------
    # Orphan tracking: vault_files.orphaned_at
    # ------------------------------------------------------------------
    async def set_file_orphaned(
        self,
        file_id: str,
        orphaned_at: str | None = None,
    ) -> bool:
        """Marca un archivo como orphaned (el archivo físico desapareció).

        Text + embeddings persisten. Search filtra `WHERE orphaned_at IS NULL`.
        Returns: True si actualizado, False si el file_id no existe.
        """
        ts = orphaned_at if orphaned_at is not None else _utc_now_iso()
        async with self._write_section("set_file_orphaned"):
            await self._db.conn.execute("BEGIN IMMEDIATE")
            try:
                await self._db.conn.execute(
                    "UPDATE vault_files SET orphaned_at = ? WHERE file_id = ?",
                    (ts, file_id),
                )
                rowcount = await _rowcount(self._db.conn)
                await self._db.conn.execute("COMMIT")
                return rowcount > 0
            except BaseException:
                with suppress(Exception):
                    await self._db.conn.execute("ROLLBACK")
                raise

    async def clear_file_orphaned(self, file_id: str) -> bool:
        """Revierte set_file_orphaned (el archivo reapareció en disco).

        Returns: True si actualizado, False si el file_id no existe.
        """
        async with self._write_section("clear_file_orphaned"):
            await self._db.conn.execute("BEGIN IMMEDIATE")
            try:
                await self._db.conn.execute(
                    "UPDATE vault_files SET orphaned_at = NULL WHERE file_id = ?",
                    (file_id,),
                )
                rowcount = await _rowcount(self._db.conn)
                await self._db.conn.execute("COMMIT")
                return rowcount > 0
            except BaseException:
                with suppress(Exception):
                    await self._db.conn.execute("ROLLBACK")
                raise

    async def list_orphaned_files(self, limit: int = 100) -> list[str]:
        """Lista file_ids que están orphaned, oldest first (purgeable)."""
        async with self._db.conn.execute(
            """
            SELECT file_id FROM vault_files
            WHERE orphaned_at IS NOT NULL
            ORDER BY orphaned_at ASC
            LIMIT ?
            """,
            (limit,),
        ) as cur:
            rows = await cur.fetchall()
        return [r["file_id"] for r in rows]

"""Mnemosyne Vault — Sprint 17 Slice 1 (storage + dedup).

Atomic content-addressable blob store with logical file rows. Backed by
SQLite via the project `Database` (aiosqlite + asyncio). Designed as the
foundation for Slice 1.5 (ingest_router 4-tier) and beyond.

Diseño (ver docs/TDD_VAULT_CORE.md V1.4 final para contrato completo):

Tablas (migration v=16):

  vault_blobs (content_sha256 PK)
      Almacena bytes deduplicados por SHA-256. `ref_count` cuenta cuántos
      vault_files apuntan al blob; un blob con ref_count=0 puede ser
      purgado por el operador (Slice 2). `size_bytes` con CHECK >= 0
      pineado por V1.2 review B-2 db-schema.

  vault_files (file_id PK = UUID4 hex)
      Filas lógicas con UNIQUE(source_path, content_sha256, mtime) que
      pinea la idempotencia a SQL (V1.2 review Vulnerabilidad #1).
      El `added_at` usa strftime('%H:%M:%f') con precisión subsegundo
      (V1.2 review B-3). El tiebreaker file_id DESC sobre added_at=empatado
      evita orden no-determinista bajo asyncio.gather concurrente.

Garantías del contrato:

  - `add()` es idempotente: dos llamadas con el mismo source_path+mtime
    producen el mismo file_id. Patrón: INSERT OR IGNORE + SELECT, NO el
    patrón racy SELECT-then-INSERT-or-UPDATE (V1.2 review MAJOR-4).
  - Lectura de bytes va por `asyncio.to_thread` (Vulnerabilidad #2: leer
    50MB del HDD dentro del event loop bloquearía el bot).
  - `remove_file()` es atómico en un solo `BEGIN IMMEDIATE` con UPDATE
    ref_count + DELETE vault_files en la misma tx.
  - `list_files()` ordena por added_at DESC, file_id DESC (determinístico
    sin depender de insertion order).

No cubre (deferido a Slice 1.5+):

  - text_version / text_tier (los crea ingest_router en Slice 1.5).
  - vacuum / orphan GC (Sprint 18+ cuando haya métricas reales).
  - backups (SQLite WAL online backup via BackupManager ya existe).
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import aiosqlite

    from hermes.memory.db import Database
    from hermes.memory.ingest_router import IngestRouter

#: Módulo-level logger. Slice 1 Vault emits structured log events with
#: `extra={...}` matching `hermes/agent/loop.py` style.
#:
#: F-OBS-1 (observability review 2026-07-07): la ausencia de un logger
#: en vault.py hacía al módulo operacionalmente ciego. Slice 1.5
#: ingest_router y S19 HTTP API dependen de estos eventos para
#: diagnosticar "por qué el archivo no apareció en el vault".
logger = logging.getLogger(__name__)

#: Lock-acquire wait threshold (ms) para emitir `vault_lock_contention`.
#: 100 ms es suficiente para detectar starvation en NAS host reales sin
#: emitir warnings en condiciones normales (probe D del reviewer: 20
#: add()s secuenciales avg=3.1 ms).
_LOCK_CONTENTION_THRESHOLD_MS: float = 100.0

#: Lock-held threshold (ms) para emitir `vault_lock_long_held`.
#: 1 s detecta BEGIN IMMEDIATE en cola por WAL flush sin spammear.
_LOCK_LONG_HELD_THRESHOLD_MS: float = 1_000.0

#: Límite máximo de bytes para `add()`. Constante de módulo para que tests
#: puedan hacer `monkeypatch.setattr` sin tocar el call site.
#: 50 MiB por defecto: cubre PDFs típicos de <100 páginas sin penalizar
#: la DB con blobs gigantes. Subirlo en Slice 2+ si surveillance lo pide.
MAX_FILE_SIZE: int = 50 * 1024 * 1024

#: Longitud esperada (hex chars) de un SHA-256.
_SHA256_HEX_LEN: int = 64


class VaultError(Exception):
    """Base para errores del Vault."""


class VaultSizeError(ValueError):
    """add() rechazó un archivo por exceder MAX_FILE_SIZE.

    Es ValueError (no VaultError genérico) porque el contrato público lo
    expone así: test_add_rejects_oversized_file pinea `pytest.raises(
    ValueError, match="too large")` y los callers pueden atrapar el
    ValueError genérico sin acoplarse al módulo.
    """

    def __init__(self, size: int, limit: int) -> None:
        super().__init__(f"file too large: {size} bytes exceeds MAX_FILE_SIZE={limit}")
        self.size = size
        self.limit = limit


@dataclass(frozen=True, slots=True)
class VaultEntry:
    """Fila lógica de un archivo en el vault.

    `file_id` es UUID4 hex (V1.4 pin a `uuid.UUID(version=4)` para
    rechazar IDs mal-formados al import).

    PR #113b Slice 2.5: añade `text` y `text_version` opcionales
    (defaults `None`/`None`) para que `Vault.get_file()` retorne
    el shape completo de la fila. Antes los callers tenían que
    hacer una query separada via `get_text()` + `get_text_version()`
    para reconstruir el state. Slice 2.5 embedder-style callers
    esperan el shape unificado.

    PR #113c round 2 (MAJOR-2 fix): the previous docstring claimed
    defaults `None` / `'v0_pymupdf'`, but the actual code defaults
    are `None` / `None`. The asymmetry was confusing — `Vault.add()`
    returns `VaultEntry(text=None, text_version=None)` (text never
    set), `Vault.get_file()` returns the real values, and
    `Vault.get_text_version()` falls back to `'v0_pymupdf'` if the
    column is NULL. Three names for the same concept, two values.
    Aligned: the dataclass defaults are `None` / `None` (idiomatic
    "unset" sentinel, matches "has update_text() been called yet?"
    semantic). The migration default `'v0_pymupdf'` is a SEPARATE
    concept (the column DEFAULT for never-updated rows) and is
    surfaced via `get_text_version()`'s explicit fallback.
    """

    file_id: str
    source_path: str
    content_sha256: str
    size_bytes: int
    mtime: float
    added_at: str  # formato 'YYYY-MM-DD HH:MM:SS.fff'
    # PR #113b: forward-compat fields (Slice 2.5+). Optional con
    # defaults para no romper callers existentes (frozen dataclass
    # no permite kwargs en posición final; usamos defaults).
    # PR #113c round 2: both default to None (NOT 'v0_pymupdf').
    # The migration default for `vault_files.text_version` is
    # 'v0_pymupdf' (set by the SQL DEFAULT clause); the dataclass
    # default is None to represent "this field was not populated
    # from the DB". After get_file(), the field is populated.
    text: str | None = None
    text_version: str | None = None
    # Sprint 19 Slice 4 (TDD_VAULT_COLLECTIONS.md §4.4.6): provenance of
    # the current text. 'tesseract_local' | 'edge_pc' | 'vlm_hosted' |
    # 'manual' | None (text not yet extracted).
    text_source: str | None = None


@dataclass(frozen=True, slots=True)
class VaultStats:
    """Snapshot agregado del vault.

    `dedup_ratio`: file_count / blob_count (1.0 si 0 blobs — convención
    definida en TDD §"VaultStats semantics"; evita división por cero sin
    cambiar el comportamiento de tests).
    """

    file_count: int
    blob_count: int
    total_bytes: int
    dedup_ratio: float


def _sha256_bytes(data: bytes) -> str:
    """SHA-256 hex (lower-case, 64 chars) — pineado por la test suite."""
    return hashlib.sha256(data).hexdigest()


async def _safely_rollback(conn: aiosqlite.Connection) -> None:
    """Shield the rollback from re-cancellation + drain cancel state.

    F-CONC-1 Slice 1 review (plan_e5219d01/concurrency, 2026-07-07):
    In Python 3.11+, when a task is cancelled mid-`add()`/`remove_file()`,
    the CancelledError propagates to the `except BaseException` block.
    The follow-up `await conn.rollback()` can itself be cancelled by
    Python's task-cancellation machinery (since the task is already in the
    "cancelling" state). If the rollback is interrupted before execution,
    the SQLite connection remains in pending-tx state.

    Subsequent BEGIN IMMEDIATE on the same connection raises
    `OperationalError("cannot start a transaction within a transaction")`,
    bricking the Vault for the rest of the process. Probe 4c reproduced
    this deterministically: cancel @ 0ms (before BEGIN) is clean;
    cancel @ >= 1ms (after BEGIN) leaves the connection poisoned.

    `asyncio.shield()` prevents re-cancellation of the rollback. On
    Python 3.11+, `asyncio.current_task().uncancel()` drains any pending
    cancel count so a follow-up op in the same call site isn't itself
    immediately re-cancelled.

    If the shielded rollback still fails (e.g. connection is gone), we
    swallow the secondary exception — the original CancelledError (or
    whatever triggered the except) is what the caller needs to see.
    """
    with contextlib.suppress(BaseException):
        # Last-resort: surface the original error. The connection MAY
        # remain poisoned; we cannot recover here. Logging happens via
        # F-OBS-13 once that lands.
        await asyncio.shield(conn.rollback())
    if hasattr(asyncio, "uncancel"):
        # asyncio.current_task() is None if not in a task (shouldn't
        # happen here — caller is async). Drain cancel state defensively.
        current = asyncio.current_task()
        while current is not None and current.cancelling() > 0:
            current.uncancel()


def _is_uuid_v4(s: object) -> bool:
    """V1.4 Nemotron BLO-1: rechaza IDs mal-formados al import.

    F-OBS-3 V3 (observability review 2026-07-07): además de malformados,
    rechaza no-strings (None, int, bytes). Sin este guard, `uuid.UUID(None)`
    raise TypeError no documentado, y Slice 4 HTTP API pasando un path
    segment faltante → None crash-ea con TypeError (debería ser KeyError
    o False silencioso, según el call site).
    """
    if not isinstance(s, str):
        return False
    try:
        uuid.UUID(s, version=4)
        return True
    except (ValueError, AttributeError):
        return False


class Vault:
    """Mnemosyne Vault — Slice 1.

    Uso:
        db = Database(Path("/var/lib/hermes/hermes.db"))
        await db.initialize()
        vault = Vault(db)
        entry = await vault.add(Path("doc.md"))
        blob = await vault.get_blob(entry.content_sha256)
    """

    def __init__(
        self,
        db: Database,
        root: Path | None = None,
        *,
        ingest_router: IngestRouter | None = None,
    ) -> None:
        """Construct a Vault around an initialized `Database`.

        Args:
            db: An already-initialized `Database` (its connection is
                reused — Vault does not open its own .db file).
            root: Optional trusted-root directory. When set, every
                `add()` MUST resolve to a path inside this root
                (verified via `Path.is_relative_to(root.resolve())`).
                Input paths that resolve outside the root raise
                `ValueError("path escapes vault root")`. This is the
                trust boundary for S19+ HTTP-API callers; today's
                operator-only deploys can leave it None.

                F-SEC-1 (adversary review 2026-07-07): without this
                guard, `path.resolve()` follows symlinks and an
                attacker who can write a symlink in the input dir
                can ingest any file the process can read. The
                resolved absolute path also becomes a
                filesystem-layout disclosure via the resulting
                `source_path`.

                Performance: root is resolved ONCE here
                (root.resolve()), not per add() call.
        """
        self._db = db
        self._root: Path | None = root.resolve() if root is not None else None
        # PR #113b (B1 fix): optional IngestRouter wiring. When set
        # AND `settings.vault_auto_ingest_on_add=True` (default),
        # every successful `add()` automatically kicks off Tier 0
        # extract via `router.ingest(file_id)`. This closes the
        # Slice 1.5 GREEN → prod gap: previously, /v1/files wrote
        # to the legacy `files` table, NEVER to vault, and Tier 0
        # never ran for new uploads. The wiring is opt-out for
        # unit tests that don't want ingest side-effects.
        # Import local to avoid circular: ingest_router imports
        # VaultProtocol (defined in ingest_router.py) → would loop.
        self._router = ingest_router
        # Pin V1.2 MAJOR-4: aiosqlite connection es single-thread; bajo
        # asyncio.gather() las awaits intercalan. Sin lock, dos add()
        # concurrentes hacen cada una `BEGIN IMMEDIATE` y la segunda
        # explota con "cannot start a transaction within a transaction".
        # El lock serializa writes (add/remove_file); reads no lo
        # necesitan (WAL permite concurrent reads con single writer).
        # F-CONC-3 (V3): el lock vive en `Database._write_lock`,
        # no aquí. El constructor de Vault NO crea su propio lock
        # porque múltiples Vault contra el mismo DB compartirían
        # la conexión single-thread de aiosqlite y necesitan
        # serialización común; sin lock per-Database, dos Vault
        # podrían llegar a BEGIN concurrente y romper.
        # Ver `_write_section` abajo para telemetría.

    @asynccontextmanager
    async def _write_section(self, op: str):
        """Acquire el write-lock del Database y emite telemetry de contention
        + held time.

        F-CONC-3 (V3 concurrency review): el lock vive en Database, no en
        Vault, para que múltiples Vault instances contra el mismo
        Database (S4+ multi-tenant) compartan la serialización. La
        aiosqlite connection es single-thread, por lo que el lock per-
        Database es la única granularidad correcta.

        F-OBS-4 (observability review): telemetría de contention + held.

        Logs:
        - `vault_lock_contention` (warning): wait_ms supera
          `_LOCK_CONTENTION_THRESHOLD_MS` (100 ms por default).
        - `vault_lock_long_held` (warning): held_ms supera
          `_LOCK_LONG_HELD_THRESHOLD_MS` (1 s por default).
        """
        t0 = time.perf_counter()
        await self._db._write_lock.acquire()
        wait_ms = (time.perf_counter() - t0) * 1000
        if wait_ms > _LOCK_CONTENTION_THRESHOLD_MS:
            logger.warning(
                "vault_lock_contention",
                extra={"op": op, "wait_ms": round(wait_ms, 1)},
            )
        try:
            yield
        finally:
            held_ms = (time.perf_counter() - t0) * 1000
            if held_ms > _LOCK_LONG_HELD_THRESHOLD_MS:
                logger.warning(
                    "vault_lock_long_held",
                    extra={"op": op, "held_ms": round(held_ms, 1)},
                )
            self._db._write_lock.release()

    # ---------------------------------------------------------------------
    # add()
    # ---------------------------------------------------------------------
    async def add(self, file: Path | str) -> VaultEntry:
        """Indexa `file` en el vault. Idempotente: misma tupla (path,
        content_sha256, mtime) → mismo file_id.

        Raises:
            ValueError: si `file` es string vacío, path vacío, o archivo
                de 0 bytes.
            FileNotFoundError: si el archivo no existe.
            PermissionError: si el archivo no es legible (POSIX).
            VaultSizeError: si excede MAX_FILE_SIZE.
        """
        # F-OBS-1: timestamp de inicio del add() (input validation → return).
        # Reportado en `vault_add_succeeded` para performance tracking.
        _add_t0 = time.perf_counter()

        # --- Validación de input ------------------------------------------------
        if isinstance(file, str):
            if not file:
                raise ValueError("source_path cannot be empty")
            path = Path(file)
        else:
            if not str(file):
                raise ValueError("source_path cannot be empty")
            path = file

        # Q2: source_path se almacena como str(Path.resolve()) para que
        # "./foo.md" y "/abs/foo.md" al mismo archivo sean el mismo file_id.
        # No resolvemos todavía — solo validamos path vacío.
        if str(path) == "":
            raise ValueError("source_path cannot be empty")

        # F-SEC-1 V3 (adversary review): defense-in-depth contra symlinks
        # en el input. Aunque `path.resolve()` ya sigue el symlink y el
        # root check (más abajo) lo detecta si termina fuera del root,
        # rechazar el input-path-es-symlink ANTES elimina ambigüedad:
        # "tú me pediste un symlink" = REJECT, sin tocar el FS. Esto
        # también cierra un escenario donde el symlink resuelve dentro
        # del root pero apunta a un archivo que el operador prefirió
        # no exponer vía add() directo.
        if path.is_symlink():
            logger.warning(
                "vault_add_rejected_security",
                extra={"reason": "symlink_input", "input_path": str(path)},
            )
            raise ValueError(f"refusing to ingest symlink (use the resolved path): {path}")

        # BLO-3 Nemotron PR-review round-1: resolver ANTES de stat/read
        # para cerrar el TOCTOU window entre operaciones.
        resolved = path.resolve()
        source_path = str(resolved)

        # F-SEC-1 V3 (root boundary): si el caller pasó root, el path
        # resuelto DEBE caer dentro. Sin esta verificación, `path.resolve()`
        # sigue symlinks y un symlink malicioso en un input dir puede
        # leer /etc/passwd y persistir el path absoluto como `source_path`.
        # `is_relative_to` es Py 3.9+; estamos en 3.13.
        if self._root is not None and not resolved.is_relative_to(self._root):
            logger.warning(
                "vault_add_rejected_security",
                extra={
                    "reason": "outside_root",
                    "resolved": source_path,
                    "root": str(self._root),
                },
            )
            raise ValueError(f"path escapes vault root: {resolved} not under {self._root}")

        if not resolved.exists():
            raise FileNotFoundError(f"file not found: {resolved}")

        # --- Lectura de bytes en thread pool (V2 pin) --------------------------
        # Lee stat primero (sync, barato): mtime + size_bytes antes de gastar
        # bandwidth leyendo bytes. Si size > MAX_FILE_SIZE, no leemos nada.
        stat_result = await asyncio.to_thread(resolved.stat)
        mtime: float = stat_result.st_mtime
        if stat_result.st_size > MAX_FILE_SIZE:
            logger.warning(
                "vault_add_rejected_size",
                extra={
                    "size_bytes": stat_result.st_size,
                    "limit_bytes": MAX_FILE_SIZE,
                },
            )
            raise VaultSizeError(stat_result.st_size, MAX_FILE_SIZE)

        data = await asyncio.to_thread(resolved.read_bytes)

        # Re-check post-read (POSIX truth = bytes leídos, no stat).
        if len(data) > MAX_FILE_SIZE:
            logger.warning(
                "vault_add_rejected_size",
                extra={
                    "size_bytes": len(data),
                    "limit_bytes": MAX_FILE_SIZE,
                    "phase": "post_read",
                },
            )
            raise VaultSizeError(len(data), MAX_FILE_SIZE)
        if not data:
            raise ValueError(f"file is empty: {resolved}")

        sha = _sha256_bytes(data)

        # --- Inserción atómica (UPSERT pattern, no SELECT-then-INSERT) ----------
        # Patrón V1.2 review MAJOR-4 + Nemotron BLO-1 round-1:
        #   1) INSERT OR IGNORE vault_files con rowcount check: si la tupla
        #      (path, sha, mtime) ya existe, rowcount=0 → NO bumpamos
        #      ref_count (idempotente). Si rowcount=1 → bumpamos.
        #      Esto evita el +2 race que Nemotron detectó: dos add()
        #      del MISMO archivo bumpan 2 veces el ref_count sin
        #      añadir nuevas file rows.
        #   2) Para el blob, UPSERT (ON CONFLICT DO UPDATE) — solo si
        #      el file row era nuevo (ref_count refleja "número de
        #      archivos apuntando al blob", no "número de add() calls").
        #   3) SELECT file_id WHERE (path, sha, mtime) — devuelve el id
        #      definitivo, sea la fila nueva o la preexistente.
        #
        # Toda la secuencia corre en una sola tx para evitar que un
        # concurrent remove_file() nos deje con file row sin blob.
        file_id = str(uuid.uuid4())
        conn = self._db.conn
        # PR #113b (B1 fix): hoist `file_rows_inserted` to outer scope
        # so the post-add auto-ingest branch can read it. Initialized
        # to 0 here, set inside the lock-protected tx, consumed after.
        file_rows_inserted: int = 0

        async with self._write_section(op="vault.add"):
            # BEGIN IMMEDIATE: bloquea inmediatamente cualquier otra escritura
            # en la misma DB hasta COMMIT/ROLLBACK. Evita la clásica race
            # INSERT-then-SELECT en SQLite WAL.
            await conn.execute("BEGIN IMMEDIATE")
            try:
                # --- vault_files: nueva fila o noop si ya existe ----------------
                cur_files = await conn.execute(
                    "INSERT OR IGNORE INTO vault_files "
                    "(file_id, source_path, size_bytes, content_sha256, mtime) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (file_id, source_path, len(data), sha, mtime),
                )
                # BLO-1 Nemotron PR-review round-1: rowcount==1 ↔ fila
                # realmente nueva. Si idempotent re-add del mismo archivo,
                # rowcount==0 → NO bumpar ref_count (idempotencia
                # simétrica con vault_files). El bug original
                # (INSERT OR IGNORE + UPSERT siempre) sumaba ref_count
                # aunque la file row no cambiara.
                # PR #113b (B1): hoisted to outer scope (no `:=` here)
                # so the post-add auto-ingest branch can read it.
                file_rows_inserted = cur_files.rowcount
                await cur_files.close()

                if file_rows_inserted == 0:
                    # Idempotent re-add — el caller pidió re-ingerir el
                    # mismo archivo. Logging debug porque es esperado en
                    # patrones de re-encoding u operación retry.
                    logger.debug(
                        "vault_add_idempotent_hit",
                        extra={
                            "source_path": source_path,
                            "sha256": sha,
                            "size_bytes": len(data),
                        },
                    )

                if file_rows_inserted:
                    # --- vault_blobs: UPSERT conditional on new file row -----
                    # Inserta con ref_count=1 si no existe; si existe,
                    # incrementa. Patrón SQLite >= 3.24 ON CONFLICT.
                    await conn.execute(
                        "INSERT INTO vault_blobs "
                        "(content_sha256, data, size_bytes, ref_count) "
                        "VALUES (?, ?, ?, 1) "
                        "ON CONFLICT (content_sha256) DO UPDATE "
                        "SET ref_count = ref_count + 1",
                        (sha, data, len(data)),
                    )
                # --- Recoger el file_id canónico ---------------------------------
                async with conn.execute(
                    "SELECT file_id, added_at FROM vault_files "
                    "WHERE source_path = ? AND content_sha256 = ? AND mtime = ?",
                    (source_path, sha, mtime),
                ) as cur:
                    row = await cur.fetchone()

                if row is None:
                    # No debería pasar — INSERT OR IGNORE asegura que o existe
                    # o lo creamos — pero por seguridad:
                    raise RuntimeError(
                        f"vault_files row missing after INSERT: ({source_path!r}, {sha!r})"
                    )

                canonical_file_id: str = row["file_id"]
                added_at: str = row["added_at"]
                await conn.commit()
            except BaseException:
                # F-CONC-1: shielded rollback that drains cancel state.
                # Without this, a task cancelled between BEGIN and COMMIT
                # leaves the SQLite connection in pending-tx state and
                # the next add() raises "cannot start a transaction
                # within a transaction" until process restart.
                await _safely_rollback(conn)
                raise

        # F-OBS-1: log final exitoso con file_id, sha256, tamaño,
        # duración. Operadores ven este evento por cada add() exitoso y
        # pueden alertar sobre duraciones excesivas o patrones de error.
        _add_duration_ms = (time.perf_counter() - _add_t0) * 1000
        logger.info(
            "vault_add_succeeded",
            extra={
                "file_id": canonical_file_id,
                "sha256": sha,
                "source_path": source_path,
                "size_bytes": len(data),
                "duration_ms": round(_add_duration_ms, 1),
            },
        )

        # PR #113b (B1 fix): auto-kick Tier 0 extract after a successful
        # add(). Only when the file is NEW (file_rows_inserted == 1),
        # not on idempotent re-add (where ingest already ran on first
        # call). The router itself is idempotent so duplicate calls
        # are safe, but we skip the round-trip when the row preexisted.
        # `vault_auto_ingest_on_add` default True; tests opt-out.
        # Tier 0 always runs in GREEN; only the post-add
        # threshold-based queueing depends on Settings.
        if file_rows_inserted and self._router is not None:
            # Read settings flag via the router's own settings (the
            # router already holds a Settings ref). We avoid a
            # separate Vault settings field to keep Vault focused
            # on storage.
            try:
                if self._router._settings.vault_auto_ingest_on_add:
                    await self._router.ingest(canonical_file_id)
            except Exception as exc:
                # PR #113b: ingest() failure must NOT break add().
                # The file is durably in the vault; Tier 0 is a
                # best-effort enhancement. If it fails (e.g. Docling
                # import error or worker-side race), log warning and
                # continue. Operator can re-trigger via apply_worker_result
                # or a manual ingest() call.
                logger.warning(
                    "vault_add_auto_ingest_failed",
                    extra={
                        "file_id": canonical_file_id,
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    },
                )

        return VaultEntry(
            file_id=canonical_file_id,
            source_path=source_path,
            content_sha256=sha,
            size_bytes=len(data),
            mtime=mtime,
            added_at=added_at,
            # PR #113b: text/text_version are populated AFTER
            # vault.add() via update_text() (called by IngestRouter
            # on auto-ingest, or manually). For the entry returned
            # here, both are None — caller should call
            # vault.get_text(file_id) for the current text if needed.
            text=None,
            text_version=None,
        )

    # ---------------------------------------------------------------------
    # get_blob()
    # ---------------------------------------------------------------------
    async def get_blob(self, content_sha256: str) -> bytes:
        """Recupera bytes de un blob vía su SHA-256.

        Raises:
            KeyError: si `content_sha256` está vacío o no existe.
        """
        if not content_sha256:
            # F-OBS-7: log miss con shape para distinguir caller-error vs data-missing.
            logger.debug(
                "vault_get_blob_miss",
                extra={"reason": "empty_input", "sha256_provided": ""},
            )
            raise KeyError("content_sha256 cannot be empty")
        if len(content_sha256) != _SHA256_HEX_LEN:
            logger.debug(
                "vault_get_blob_miss",
                extra={"reason": "bad_length", "sha256_provided": content_sha256[:64]},
            )
            raise KeyError(f"not a sha256 hex (len={len(content_sha256)})")

        async with self._db.conn.execute(
            "SELECT data FROM vault_blobs WHERE content_sha256 = ?",
            (content_sha256,),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            logger.debug(
                "vault_get_blob_miss",
                extra={"reason": "not_in_db", "sha256_provided": content_sha256[:64]},
            )
            raise KeyError(f"no blob with content_sha256={content_sha256!r}")
        # SUG-6 Nemotron PR-review round-1: aiosqlite ya devuelve bytes
        # para BLOB, no hace falta `bytes(...)` que copiaba.
        return row["data"]  # BLOB → bytes (sqlite.Row ya viene tipado)

    # ---------------------------------------------------------------------
    # get_blob_for_file() — convenience for IngestRouter
    # ---------------------------------------------------------------------
    async def get_blob_for_file(self, file_id: str) -> bytes:
        """Recupera bytes del archivo indexado por `file_id` (UUID4).

        PR #113b (B1 fix): IngestRouter.ingest() llama este método
        con `file_id` (no `sha`). Resuelve `file_id → sha` via una
        sola query a `vault_files`, luego delega a `get_blob(sha)`.
        Convenience method — equivalente a:
            file_entry = await self.get_file(file_id)
            return await self.get_blob(file_entry.content_sha256)
        pero en un solo round-trip optimizado.

        Raises:
            KeyError: si `file_id` no existe en vault_files.
        """
        if not _is_uuid_v4(file_id):
            raise KeyError(f"not a uuid v4: {file_id!r}")
        async with self._db.conn.execute(
            "SELECT content_sha256 FROM vault_files WHERE file_id = ?",
            (file_id,),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            raise KeyError(f"file_id not found in vault: {file_id!r}")
        return await self.get_blob(row["content_sha256"])

    # ---------------------------------------------------------------------
    # get_text() / get_text_version() — Slice 2.5 (VaultEmbedder-facing)
    # ---------------------------------------------------------------------
    async def get_text(self, file_id: str) -> str:
        """Recupera el texto canónico persistido para `file_id`.

        PR #113b Slice 2.5: VaultEmbedder.embed_file() necesita el
        texto de vault_files.text (no el blob binario). Este método
        es la versión "texto" de get_blob/get_blob_for_file.

        Args:
            file_id: UUID4.

        Returns:
            El texto canónico actual (puede ser "" si update_text
            nunca se llamó). NO None — string vacío es válido.

        Raises:
            KeyError: si `file_id` no existe en vault_files.
        """
        if not _is_uuid_v4(file_id):
            raise KeyError(f"not a uuid v4: {file_id!r}")
        async with self._db.conn.execute(
            "SELECT text FROM vault_files WHERE file_id = ?",
            (file_id,),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            raise KeyError(f"file_id not found in vault: {file_id!r}")
        return row["text"] or ""

    async def get_text_version(self, file_id: str) -> str:
        """Recupera la text_version actual (e.g. "v0_pymupdf", "v15_lan_worker").

        PR #113b Slice 2.5: VaultEmbedder lo usa para etiquetar los
        chunks con la versión correcta. Si el operator cambia el tier
        de extracción (v0 → v15 via Slice 1.5), el text_version bump
        y el Watcher re-embebe (text_version mismatch en chunks).

        Returns:
            La versión del texto. Default "v0_pymupdf" si nunca se
            llamó update_text (coherente con v=17 migration default).

        Raises:
            KeyError: si `file_id` no existe en vault_files.
        """
        if not _is_uuid_v4(file_id):
            raise KeyError(f"not a uuid v4: {file_id!r}")
        async with self._db.conn.execute(
            "SELECT text_version FROM vault_files WHERE file_id = ?",
            (file_id,),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            raise KeyError(f"file_id not found in vault: {file_id!r}")
        return row["text_version"] or "v0_pymupdf"

    # ---------------------------------------------------------------------
    # update_text() — Slice 1.5 GREEN (ingest_router usa esto)
    # ---------------------------------------------------------------------
    async def update_text(
        self,
        file_id: str,
        *,
        text: str,
        text_version: str,
        tier: str,
    ) -> VaultEntry | None:
        """Escribe texto canónico para file_id (ingest_router-facing).

        V1.2 BLOCKING-5 + MAJOR-11 invariants (Slice 1.5 GREEN):
        1. NO-DOWNGRADE: si `vault_files.text_version` rank (TEXT_VERSION_ORDER)
           >= incoming rank, no-op (sin excepción, sin text_at bump).
        2. text_at BUMP: en update exitoso, text_at = NOW() (UTC ISO8601).
        3. tier ↔ text_version CONSISTENCY: tier debe matchear family
           (TIER_VERSION_FAMILIES[tier]). Mismatch raises ValueError.

        PR #113c round 2 (MAJOR-1 fix): now returns `VaultEntry` (the
        post-update row) on success, or `None` if the no-downgrade
        rule fired (caller knows the update was a no-op). Previously
        the function returned `None` always; callers that wanted the
        post-update entry had to do a separate `vault.get_file()` call
        (1 extra round-trip + 1 race window). The return is read
        INSIDE the same `_write_lock` so it's atomic with the UPDATE.

        Referencia: docs/TDD_VAULT_INGEST_WORKER.md §"API contract".
        """
        # Import aquí (no top-level) para evitar ciclo al cargar
        # ingest_router → vault.
        from hermes.memory.ingest_router import (
            TEXT_VERSION_ORDER,
            TIER_VERSION_FAMILIES,
        )

        # Invariant 3: tier ↔ text_version consistency check.
        # PR #113b (M-INV-3 fix): use `is not None` not truthy check.
        # Old: `if expected_prefix and not text_version.startswith(...)`.
        # An UNKNOWN tier (not in TIER_VERSION_FAMILIES) made
        # `expected_prefix = None`, which is falsy → condition is
        # False → invariant silently skipped. Now: unknown tier
        # raises explicitly.
        expected_prefix = TIER_VERSION_FAMILIES.get(tier)
        if expected_prefix is None:
            raise ValueError(
                f"unknown tier '{tier}'. Known: {sorted(TIER_VERSION_FAMILIES.keys())}"
            )
        if not text_version.startswith(expected_prefix):
            raise ValueError(
                f"text_version '{text_version}' does not match tier "
                f"'{tier}' family (expected prefix '{expected_prefix}')"
            )

        if not _is_uuid_v4(file_id):
            raise KeyError(f"not a uuid v4: {file_id!r}")

        async with self._db._write_lock:
            conn = self._db.conn
            # Read current text_version rank (if any).
            async with conn.execute(
                "SELECT text_version FROM vault_files WHERE file_id = ?",
                (file_id,),
            ) as cur:
                row = await cur.fetchone()
            if row is None:
                raise KeyError(f"file_id not found in vault: {file_id!r}")

            # Invariant 1: no-downgrade check.
            current_version = row["text_version"]
            current_rank = TEXT_VERSION_ORDER.get(current_version, 0)
            incoming_rank = TEXT_VERSION_ORDER.get(text_version, 0)
            if incoming_rank < current_rank:
                logger.debug(
                    "vault_update_text_downgrade_refused",
                    extra={
                        "file_id": file_id,
                        "current": current_version,
                        "incoming": text_version,
                    },
                )
                return None  # no-op: caller knows the update was skipped

            # Invariant 2: text_at BUMP + version/tier write.
            # PR #113b (M-INV-2 fix): write BOTH `text_at` (ISO8601 TEXT,
            # human-readable for debug/observability) AND
            # `text_at_epoch` (epoch seconds INTEGER, queryable via
            # `WHERE text_at_epoch > ?`). Old code: only `text_at`
            # TEXT, which broke lexicographic comparison when
            # `datetime.now(UTC).isoformat()` returned a shorter
            # string (microsecond=0 means the format omits the
            # `.000000` part → 26 chars instead of 32 chars).
            # Slice 2 embed_vault uses `text_at_epoch` for
            # cache invalidation queries — both columns written
            # atomically here.
            now = datetime.now(UTC)
            text_at = now.isoformat()
            text_at_epoch = int(now.timestamp())
            await conn.execute(
                "UPDATE vault_files "
                "SET text = ?, text_version = ?, text_tier = ?, text_at = ?, "
                "    text_at_epoch = ? "
                "WHERE file_id = ?",
                (text, text_version, tier, text_at, text_at_epoch, file_id),
            )
            await conn.commit()
            logger.info(
                "vault_text_updated",
                extra={
                    "file_id": file_id,
                    "text_version": text_version,
                    "tier": tier,
                    "text_length": len(text),
                    "text_at": text_at,
                    "text_at_epoch": text_at_epoch,
                },
            )
            # PR #113c round 2 (MAJOR-1 fix): return the post-update
            # VaultEntry. Read inside the same _write_lock so the
            # returned entry is atomic with the UPDATE (no race with
            # a concurrent writer). Use SELECT fields that match
            # get_file() so the shape is consistent.
            async with conn.execute(
                "SELECT file_id, source_path, content_sha256, size_bytes, mtime, "
                "       added_at, text, text_version, text_source "
                "FROM vault_files WHERE file_id = ?",
                (file_id,),
            ) as cur:
                updated_row = await cur.fetchone()
            if updated_row is None:
                # Shouldn't happen — we just UPDATEd it — but be
                # defensive: return None rather than crash.
                return None
            return VaultEntry(
                file_id=updated_row["file_id"],
                source_path=updated_row["source_path"],
                content_sha256=updated_row["content_sha256"],
                size_bytes=updated_row["size_bytes"],
                mtime=updated_row["mtime"],
                added_at=updated_row["added_at"],
                text=updated_row["text"],
                text_version=updated_row["text_version"],
                text_source=updated_row["text_source"],
            )

    # ---------------------------------------------------------------------
    # list_files()
    # ---------------------------------------------------------------------
    async def list_files(self, limit: int = 20, offset: int = 0) -> list[VaultEntry]:
        """Lista entradas, más recientes primero.

        Orden: `added_at DESC, file_id DESC` (V1.2 review B-3/BLOCKING-2).
        Dos adds en el mismo seg würden empatan en added_at; el tiebreaker
        file_id garantiza determinismo.
        """
        if limit <= 0:
            raise ValueError(f"limit must be > 0, got {limit}")
        if offset < 0:
            raise ValueError(f"offset must be >= 0, got {offset}")

        async with self._db.conn.execute(
            "SELECT file_id, source_path, content_sha256, size_bytes, mtime, "
            "       added_at, text, text_version, text_source "
            "FROM vault_files ORDER BY added_at DESC, file_id DESC "
            "LIMIT ? OFFSET ?",
            (limit, offset),
        ) as cur:
            rows = await cur.fetchall()
        return [
            VaultEntry(
                file_id=r["file_id"],
                source_path=r["source_path"],
                content_sha256=r["content_sha256"],
                size_bytes=r["size_bytes"],
                mtime=r["mtime"],
                added_at=r["added_at"],
                text=r["text"],
                text_version=r["text_version"],
                text_source=r["text_source"],
            )
            for r in rows
        ]

    # ---------------------------------------------------------------------
    # get_file()
    # ---------------------------------------------------------------------
    async def get_file(self, file_id: str) -> VaultEntry:
        """Recupera una entrada por su file_id (UUID4 hex).

        Raises:
            KeyError: si el id no existe (o es un UUID4 mal-formado).
        """
        if not _is_uuid_v4(file_id):
            raise KeyError(f"not a uuid v4: {file_id!r}")
        async with self._db.conn.execute(
            "SELECT file_id, source_path, content_sha256, size_bytes, mtime, "
            "       added_at, text, text_version, text_source "
            "FROM vault_files WHERE file_id = ?",
            (file_id,),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            raise KeyError(f"no file with file_id={file_id!r}")
        return VaultEntry(
            file_id=row["file_id"],
            source_path=row["source_path"],
            content_sha256=row["content_sha256"],
            size_bytes=row["size_bytes"],
            mtime=row["mtime"],
            added_at=row["added_at"],
            text=row["text"],
            text_version=row["text_version"],
            text_source=row["text_source"],
        )

    # ---------------------------------------------------------------------
    # remove_file()
    # ---------------------------------------------------------------------
    async def remove_file(self, file_id: str) -> bool:
        """Borra una entrada. Decrementa ref_count del blob asociado y lo
        purga si llega a 0. Idempotente: si el id no existe, devuelve False
        sin excepción.

        Returns:
            True si se borró, False si no había nada con ese file_id.
        """
        if not _is_uuid_v4(file_id):
            # Treat malformed id as "no such row" (documented contract:
            # remove_file on bad id returns False, doesn't raise).
            # F-OBS-7 (V3 observability review): log al menos debug para
            # distinguir "malformed caller" de "id never existed" en
            # auditoría S19 HTTP API.
            logger.debug(
                "vault_remove_no_op",
                extra={"reason": "malformed_uuid", "file_id_provided": str(file_id)[:128]},
            )
            return False

        conn = self._db.conn
        async with self._write_section(op="vault.remove_file"):
            await conn.execute("BEGIN IMMEDIATE")
            try:
                # 1) Encontrar el content_sha256 del row a borrar.
                async with conn.execute(
                    "SELECT content_sha256 FROM vault_files WHERE file_id = ?",
                    (file_id,),
                ) as cur:
                    row = await cur.fetchone()
                if row is None:
                    await conn.commit()
                    logger.debug(
                        "vault_remove_no_op",
                        extra={"reason": "id_not_found", "file_id": file_id},
                    )
                    return False
                sha: str = row["content_sha256"]

                # 2) DELETE row.
                await conn.execute("DELETE FROM vault_files WHERE file_id = ?", (file_id,))

                # 3) Decrementar ref_count y purgar si llega a 0.
                # BLO-2 Nemotron PR-review round-1: WHERE ref_count > 0
                # para no violar CHECK (ref_count >= 0) si un manual DB
                # edit dejó el contador en 0 con file rows aún presentes.
                await conn.execute(
                    "UPDATE vault_blobs SET ref_count = ref_count - 1 "
                    "WHERE content_sha256 = ? AND ref_count > 0",
                    (sha,),
                )
                await conn.execute(
                    "DELETE FROM vault_blobs WHERE content_sha256 = ? AND ref_count <= 0",
                    (sha,),
                )
                await conn.commit()
                # F-OBS-1: log exitoso
                logger.info(
                    "vault_remove_succeeded",
                    extra={"file_id": file_id, "sha256": sha},
                )
                return True
            except BaseException:
                # F-CONC-1: shielded rollback that drains cancel state.
                # Same pattern as add() — see _safely_rollback docstring.
                await _safely_rollback(conn)
                raise

    # ---------------------------------------------------------------------
    # stats()
    # ---------------------------------------------------------------------
    async def stats(self) -> VaultStats:
        """Snapshot del estado del vault."""
        async with self._db.conn.execute(
            "SELECT "
            "(SELECT COUNT(*) FROM vault_files) AS file_count, "
            "(SELECT COUNT(*) FROM vault_blobs) AS blob_count, "
            "(SELECT COALESCE(SUM(size_bytes), 0) FROM vault_blobs) AS total_bytes"
        ) as cur:
            row = await cur.fetchone()

        # NIT-13 Nemotron PR-review round-1: en `SELECT COUNT(*) ... FROM x`
        # SQLite SIEMPRE devuelve exactamente 1 fila (incluso si x está
        # vacía), con `0` y `NULL`/`COALESCE` como defaults. El check
        # `if row is None` era dead code. Confiamos en el contrato SQL.
        assert row is not None, "aggregate query must return a row"

        file_count = int(row["file_count"])
        blob_count = int(row["blob_count"])
        total_bytes = int(row["total_bytes"])

        # Convención TDD: dedup_ratio = file_count / blob_count, 1.0 si 0 blobs.
        # Garantiza que "no hay dedup" (cada archivo = blob único) == 1.0.
        dedup_ratio: float = float(file_count) / float(blob_count) if blob_count > 0 else 1.0

        return VaultStats(
            file_count=file_count,
            blob_count=blob_count,
            total_bytes=total_bytes,
            dedup_ratio=dedup_ratio,
        )


__all__ = [
    "MAX_FILE_SIZE",
    "Vault",
    "VaultEntry",
    "VaultError",
    "VaultSizeError",
    "VaultStats",
]

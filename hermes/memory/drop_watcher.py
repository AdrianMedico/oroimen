"""DropWatcher — Sprint 19 Slice 4 (§4).

Watcher del directorio `<vault_root>/pending/drop/<subdir>/`. Detecta
archivos nuevos vía `watchfiles` (inotify/FSEvents/ReadDirectoryChangesW),
valida la extensión, deriva el `file_id` del contenido, lo inserta en
`vault_files`, lo enlaza a la `vault_collections` correspondiente, y
escribe el manifest que `process_inbox()` (Sprint 17) recoge.

NORTH STAR: Cero datos salen del NAS. Cero hosted APIs automáticas. La
OCR para JPG/PNG la hace Tesseract local (Slice 4b). El edge PC es
opt-in (Slice 4c). Hosted VLM es opt-in por comando explícito (4d).

CONTRACTS (Sprint 19 §4 + §4.2):
- `file_id` = SHA-256(content)[:32], NO random UUID (§4.2 — load-bearing).
- Extension whitelist enforced here AND in M6 Phase 2 (defense-in-depth).
- Collection auto-create: si el subdir no existe como collection, se crea
  con archived=0, sin parent, sin descripción. Idempotente (UNIQUE name).
- vault_files insert: idempotente (UNIQUE path). Si el file_id ya existe
  con otro path, NO se duplica (content-based dedup).
- Manifest: JSON al lado del archivo con el schema_version + id (= file_id)
  + path + created_at.

SCOPE (Slice 4a — este PR):
- DropWatcher con process_path() (testable) + run() (watchfiles loop)
- Extension whitelist (8 extensiones: pdf, docx, xlsx, txt, md, jpg, jpeg, png)
- File ID contract (SHA-256)
- Collection auto-create + link
- Manifest write
- Wire en __main__.py

OUT OF SCOPE (futuros slices):
- Slice 4b: extractors reales (Tesseract para JPG/PNG, pymupdf para PDF, etc.)
- Slice 4b: ocr_pending row creation cuando confidence < 0.60
- Slice 4c: edge coordinator (auto-queue, catch-up pass)
- Slice 4c: M6 Phase 5 (zombie recovery)
- Slice 4d: 6 user commands (/pendingOCR, /acceptNull, etc.)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sqlite3
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hermes.memory.collections import VaultCollectionsRepo
    from hermes.memory.db import Database
    from hermes.memory.edge_coordinator import EdgeCoordinator
    from hermes.memory.ocr_pending_repo import OcrPendingRepo

logger = logging.getLogger(__name__)

#: Extension whitelist (Sprint 19 §4.1). Single source of truth para
#: el watcher Y M6 Phase 2 (defense-in-depth). Cualquier extensión fuera
#: de este set se skipea con `drop_watcher_skipped_unknown_ext`.
#:
#: Formato: lowercase con punto. `.jpg` y `.jpeg` ambos aceptados
#: (Tesseract necesita ambos — algunos cameras emiten uno, otros el otro).
ALLOWED_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".pdf",
        ".docx",
        ".xlsx",
        ".txt",
        ".md",
        ".jpg",
        ".jpeg",
        ".png",
    }
)

#: Manifest schema version (Sprint 19 §4.2). Bump si cambia el shape
#: (e.g., añadir fields). El `process_inbox()` (Sprint 17) valida
#: schema_version antes de procesar.
MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ProcessingResult:
    """Outcome de `process_path`. Usado por tests y métricas."""

    file_id: str
    file_path: str
    collection_id: str
    collection_name: str
    action: str  # 'inserted' | 'linked_existing' | 'replaced'
    # | 'skipped_unknown_ext' | 'skipped_root' | 'skipped_manifest'
    # | 'skipped_dir' | 'skipped_outside_drop' | 'skipped_disappeared'
    # | 'skipped_archived_collection' | 'skipped_invalid_path'


def _utc_now_iso() -> str:
    """ISO 8601 UTC timestamp, second precision."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _manifest_path(file_path: Path) -> Path:
    """Devuelve la ruta al manifest del archivo.

    Formato: `<file>.md.json` (Sprint 17 manifest convention + 19 §4.2).
    """
    return file_path.with_suffix(file_path.suffix + ".json")


def _find_relevant_root(path: Path, drop_root: Path, monitor_roots: list[Path]) -> Path | None:
    """Devuelve el root al que pertenece `path` (drop_root o monitor_root),
    o None si está fuera de todos.

    Sprint 19 Slice 4d v2 (R1 v0.6 M2 fix): un archivo puede estar bajo
    drop_root O bajo uno de los monitor_roots. Antes, solo se aceptaban
    archivos bajo drop_root.
    """
    resolved_path = path.resolve(strict=False)
    for root in [drop_root, *monitor_roots]:
        try:
            resolved_path.relative_to(root.resolve(strict=False))
            return root
        except ValueError:
            continue
    return None


def _is_within_drop_root(path: Path, drop_root: Path) -> bool:
    """True si `path` está directamente bajo `drop_root/` (no en un subdir).

    El watcher REQUIERE que los archivos estén en `drop/<subdir>/`. Un
    archivo en `drop/foo.pdf` (sin subdir) se rechaza — el usuario
    debe organizarlos en colecciones.
    """
    rel = path.relative_to(drop_root)
    return len(rel.parts) == 1  # solo el filename, sin subdir


def _subdir_name(path: Path, drop_root: Path) -> str | None:
    """Devuelve el nombre del subdir inmediato bajo `drop_root`, o None
    si el path está en el root."""
    rel = path.relative_to(drop_root)
    if len(rel.parts) < 2:
        return None
    return rel.parts[0]


class DropWatcher:
    """Watcher del directorio `drop/` y reconciliador de archivos nuevos.

    Diseño:
    - Una instancia por proceso Hermes.
    - `process_path()` es la entry-point testeable: toma un Path,
      devuelve un `ProcessingResult` con el outcome. No toca el event
      loop de watchfiles; es coroutine-friendly para tests async.
    - `run()` es el loop principal que usa `watchfiles.awatch()` para
      detectar nuevos archivos y llamar a `process_path()` por cada uno.
      Idempotente: si el container se reinicia, scan-on-startup
      procesa los archivos que ya estén en `drop/`.
    - La collection lookup + auto-create usa el `VaultCollectionsRepo`
      existente (Sprint 19 Slice 1). NO duplica la lógica de collections.
    - El `vault_files` insert respeta la UNIQUE(path) constraint. Si el
      archivo ya existe (caller reintentó), no se duplica — el
      `file_id` del row existente se devuelve.
    - El manifest es opcional pero escrito siempre que el watcher
      inserta/procesa. process_inbox() (Sprint 17) lo lee para
      evitar re-derivar el file_id.

    Concurrencia:
    - El watchfiles loop corre en su propio task. Cada evento genera
      un task de process_path. Usamos un semáforo para limitar la
      concurrencia (default 4) y no saturar la DB.
    - Los writes a vault_files/vault_file_collections van por
      `Database._write_lock` (F-CONC-3).
    """

    def __init__(
        self,
        db: Database,
        collections_repo: VaultCollectionsRepo,
        drop_root: Path,
        ocr_pending_repo: OcrPendingRepo | None = None,
        edge_coordinator: EdgeCoordinator | None = None,
        *,
        autoqueue_threshold: float = 0.85,
        debounce_ms: int = 100,
        max_concurrent: int = 4,
        max_pending: int = 1000,
        monitor_roots: list[Path] | None = None,
        default_collection: str | None = None,
        ocr_fallback_dpi: int = 200,
        ocr_fallback_grayscale: bool = True,
        ocr_fallback_lang: str = "deu+eng+spa",
    ) -> None:
        self._db = db
        self._collections_repo = collections_repo
        self._ocr_pending_repo = ocr_pending_repo
        self._edge_coordinator = edge_coordinator
        self._drop_root = drop_root
        # Sprint 19 v0.5 §3: monitor_roots are additional recursive
        # roots (besides drop_root). v0.6 M6 partition rule: monitor_roots
        # MUST NOT be under VAULT_INBOX_ROOT (validated by M6ReconcileScheduler).
        # None = legacy behavior (only drop_root).
        self._monitor_roots: list[Path] = list(monitor_roots) if monitor_roots else []
        # Sprint 19 followup (user feedback 2026-07-12): if set, root-level
        # files in drop_root (no subdir) get routed to this collection
        # instead of being skipped. True opt-in: None = skip (legacy).
        self._default_collection = default_collection
        # Sprint 19.5 (PR-A): OCR fallback settings for scanned PDFs.
        # Default values match the Settings defaults but can be overridden
        # per DropWatcher instance (e.g., in tests).
        self._ocr_fallback_dpi = ocr_fallback_dpi
        self._ocr_fallback_grayscale = ocr_fallback_grayscale
        self._ocr_fallback_lang = ocr_fallback_lang
        self._autoqueue_threshold = autoqueue_threshold
        self._debounce_s = debounce_ms / 1000.0
        self._sem = asyncio.Semaphore(max_concurrent)
        # v0.5: Bounded queue for backpressure. If full, events are written
        # to the dropped_events table for M6ReconcileScheduler to re-queue.
        self._max_pending = max_pending
        self._queue: asyncio.Queue[Path] = asyncio.Queue(maxsize=max_pending)
        self._workers: list[asyncio.Task[None]] = []

    # ------------------------------------------------------------------
    # Extension whitelist (exposed for tests + M6 Phase 2 reuse)
    # ------------------------------------------------------------------
    @staticmethod
    def is_extension_allowed(ext: str) -> bool:
        """True si `ext` está en la whitelist (case-insensitive)."""
        return ext.lower() in ALLOWED_EXTENSIONS

    # ------------------------------------------------------------------
    # Dropped event persistence (v0.5 §3 backpressure safety net)
    # ------------------------------------------------------------------
    async def _record_dropped_event(
        self,
        file_path: Path,
        reason: str,
    ) -> None:
        """Inserta un evento en `dropped_events` para que M6 lo reintente.

        Se llama cuando la queue está llena (backpressure) o el worker
        falla definitivamente. M6ReconcileScheduler lee de esta tabla
        cada 5 min y re-encola los eventos pendientes.

        Args:
            file_path: ruta al archivo que se dropeó.
            reason: 'queue_full' o 'worker_failed'.
        """
        try:
            await self._db.conn.execute(
                """
                INSERT INTO dropped_events (source_path, detected_at)
                VALUES (?, ?)
                """,
                (
                    file_path.as_posix(),
                    datetime.now(UTC).isoformat(),
                ),
            )
            await self._db.conn.commit()
            logger.warning(
                "drop_watcher_event_dropped",
                extra={
                    "path": file_path.as_posix(),
                    "reason": reason,
                },
            )
        except Exception:
            logger.exception(
                "drop_watcher_dropped_event_write_failed",
                extra={"path": file_path.as_posix(), "reason": reason},
            )

    # ------------------------------------------------------------------
    # process_path — core testable method
    # ------------------------------------------------------------------
    async def process_path(self, file_path: Path) -> ProcessingResult:
        """Procesa un archivo nuevo: extension check, file_id, vault_files
        insert, collection link, manifest write.

        Idempotente: si el path ya está en vault_files con el mismo
        contenido, no duplica; solo asegura el collection link.

        Si el path está en vault_files con contenido diferente (usuario
        reemplazó el archivo), elimina la row vieja (cascadea el bridge)
        y crea la nueva. Esto preserva la idempotencia del (path, content,
        mtime) sin dejar rows huérfanas.

        Returns:
            ProcessingResult con file_id, collection_id, action. La action
            describe qué hizo el watcher:
            - 'inserted': archivo nuevo, vault_files row creada
            - 'linked_existing': archivo ya estaba con mismo content
            - 'replaced': archivo reemplazado (distinto content, mismo path)
            - 'skipped_unknown_ext': extensión no en whitelist
            - 'skipped_root': archivo en drop root (no subdir)
            - 'skipped_manifest': archivo es un manifest (<file>.md.json)
            - 'skipped_dir': path es un directorio
            - 'skipped_outside_drop': path resuelto fuera de drop_root
            - 'skipped_invalid_path': path no resolvable (symlink roto, etc.)
            - 'skipped_disappeared': file movido/borrado entre detección y hash
            - 'skipped_archived_collection': subdir matchea colección archivada
        """
        posix_path = file_path.as_posix()

        # 0. Path safety: el archivo DEBE estar bajo drop_root O bajo
        # uno de los monitor_roots (defense-in-depth contra path
        # traversal). watchfiles normalmente solo emite eventos para
        # paths bajo el watch root, pero un symlink malicioso o un
        # filesystem race podría colar un path fuera. Resolvemos
        # symlinks y validamos. Si falla el resolve, skip (symlink
        # roto, permisos).
        try:
            resolved_path = file_path.resolve(strict=False)
            resolved_drop = self._drop_root.resolve(strict=False)
            resolved_monitors = [m.resolve(strict=False) for m in self._monitor_roots]
            allowed_roots = [resolved_drop, *resolved_monitors]
            if not any(resolved_path.is_relative_to(root) for root in allowed_roots):
                logger.warning(
                    "drop_watcher_skipped_outside_drop",
                    extra={
                        "path": posix_path,
                        "resolved": resolved_path.as_posix(),
                        "drop_root": resolved_drop.as_posix(),
                    },
                )
                return ProcessingResult(
                    file_id="",
                    file_path=posix_path,
                    collection_id="",
                    collection_name="",
                    action="skipped_outside_drop",
                )
        except (OSError, RuntimeError) as exc:
            # symlink roto, permission denied en resolve, etc.
            logger.warning(
                "drop_watcher_skipped_invalid_path",
                extra={"path": posix_path, "error": str(exc)},
            )
            return ProcessingResult(
                file_id="",
                file_path=posix_path,
                collection_id="",
                collection_name="",
                action="skipped_invalid_path",
            )

        # 1. Skip directories (watchfiles can emit events for both)
        if file_path.is_dir():
            logger.debug(
                "drop_watcher_skipped_dir",
                extra={"path": posix_path},
            )
            return ProcessingResult(
                file_id="",
                file_path=posix_path,
                collection_id="",
                collection_name="",
                action="skipped_dir",
            )

        # 2. Skip manifest files (they live alongside the file, e.g. foo.pdf.json)
        if file_path.name.endswith(".md.json"):
            return ProcessingResult(
                file_id="",
                file_path=posix_path,
                collection_id="",
                collection_name="",
                action="skipped_manifest",
            )

        # 3. Skip if file is at drop root (no subdir)
        relevant_root = _find_relevant_root(file_path, self._drop_root, self._monitor_roots)
        if relevant_root is None:
            return ProcessingResult(
                file_id="",
                file_path=posix_path,
                collection_id="",
                collection_name="",
                action="skipped_outside_drop",
            )
        if _is_within_drop_root(file_path, relevant_root):
            # File is at the root of drop_root / monitor_root (no subdir).
            # Sprint 19 followup (user feedback 2026-07-12): if the
            # default_collection is set, route the file there instead of
            # skipping. True opt-in: None = skip (legacy).
            if not self._default_collection:
                logger.info(
                    "drop_watcher_skipped_root",
                    extra={"path": posix_path, "reason": "no subdir"},
                )
                return ProcessingResult(
                    file_id="",
                    file_path=posix_path,
                    collection_id="",
                    collection_name="",
                    action="skipped_root",
                )
            # Route to the default collection (set below)
            logger.info(
                "drop_watcher_root_default_collection",
                extra={
                    "path": posix_path,
                    "default_collection": self._default_collection,
                },
            )

        # 4. Extension whitelist
        ext = file_path.suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            logger.info(
                "drop_watcher_skipped_unknown_ext",
                extra={"path": posix_path, "extension": ext},
            )
            return ProcessingResult(
                file_id="",
                file_path=posix_path,
                collection_id="",
                collection_name="",
                action="skipped_unknown_ext",
            )

        # 5. Compute content_sha256 + file_id (single hash pass) + mtime + size.
        # FIX: race-safe — if the file is moved/removed between fs event
        # detection and this point, FileNotFoundError is raised. We catch
        # and return skipped_disappeared (the next fs event will retry).
        from hermes.memory.file_id import is_valid_file_id

        try:
            h = hashlib.sha256()
            with file_path.open("rb") as f:
                for chunk in iter(lambda: f.read(65_536), b""):
                    h.update(chunk)
            full_hex = h.hexdigest()
            file_id = full_hex[:32]
            content_sha256 = full_hex
            assert is_valid_file_id(file_id)
            stat = file_path.stat()
            mtime = stat.st_mtime
            size_bytes = stat.st_size
        except FileNotFoundError:
            logger.info(
                "drop_watcher_skipped_disappeared",
                extra={"path": posix_path},
            )
            return ProcessingResult(
                file_id="",
                file_path=posix_path,
                collection_id="",
                collection_name="",
                action="skipped_disappeared",
            )

        # Resolve subdir: default_collection (if root-level + configured)
        # or extract from the file path. subdir is always set at this
        # point because we either skipped (returned) or routed to
        # default_collection.
        subdir: str | None
        if _is_within_drop_root(file_path, relevant_root) and self._default_collection:
            subdir = self._default_collection
        else:
            subdir = _subdir_name(file_path, relevant_root)
        assert subdir is not None  # we just checked it's not at root

        # 6. Resolve collection (auto-create if missing, skip if archived).
        # FIX: filter out archived collections. If the subdir matches an
        # archived collection, the user probably forgot they archived it
        # (or is reusing the subdir name). Skip with a clear warning so
        # the file isn't invisibly linked to a hidden collection.
        collection = await self._collections_repo.get_active_collection_by_name(subdir)
        if collection is None:
            # Could be: (a) no collection at all, or (b) archived. Check (b).
            archived = await self._collections_repo.get_collection_by_name(subdir)
            if archived is not None and archived.archived:
                logger.warning(
                    "drop_watcher_collection_archived",
                    extra={
                        "path": posix_path,
                        "subdir": subdir,
                        "collection_id": archived.collection_id,
                    },
                )
                return ProcessingResult(
                    file_id=file_id,
                    file_path=posix_path,
                    collection_id=archived.collection_id,
                    collection_name=archived.name,
                    action="skipped_archived_collection",
                )
            # (a) No collection: create
            collection = await self._collections_repo.create_collection(
                name=subdir,
                description=None,
                sort_order=0,
            )
            logger.info(
                "drop_watcher_collection_created",
                extra={
                    "collection_id": collection.collection_id,
                    "collection_name": subdir,
                },
            )

        # 7. INSERT or REPLACE vault_files.
        # FIX: if the path is already in vault_files with DIFFERENT content
        # (user replaced the file), the old row must be deleted (cascade
        # removes vault_file_collections links to the old file_id) and a
        # new row created. Otherwise we end up with two rows for the same
        # path (the UNIQUE is on (source_path, content_sha256, mtime),
        # not on source_path alone).
        await self._db._write_lock.acquire()
        try:
            await self._db.conn.execute("BEGIN IMMEDIATE")
            # Check if path already exists
            async with self._db.conn.execute(
                "SELECT file_id, content_sha256, mtime, size_bytes "
                "FROM vault_files WHERE source_path = ?",
                (posix_path,),
            ) as cur:
                existing = await cur.fetchone()

            if existing is None:
                # Fresh insert. But first, check if this file_id already
                # exists at a different source_path (move detection per
                # TDD v0.3 §4.4 / R1 M1 fix). If yes, the file content
                # has been moved to a new location; UPDATE source_path
                # + rebuild the bridge via _move_bridge (soft-delete
                # old bridge, insert new bridge).
                async with self._db.conn.execute(
                    "SELECT file_id, source_path FROM vault_files WHERE file_id = ?",
                    (file_id,),
                ) as cur:
                    existing_by_id = await cur.fetchone()

                if existing_by_id is not None:
                    # MOVE: same content, different path. Update
                    # source_path + mtime + size_bytes; rebuild bridge.
                    old_path = existing_by_id["source_path"]
                    await self._db.conn.execute(
                        "UPDATE vault_files "
                        "SET source_path = ?, mtime = ?, size_bytes = ? "
                        "WHERE file_id = ?",
                        (posix_path, mtime, size_bytes, file_id),
                    )
                    # Find the previous active bridge (the old
                    # collection) so _move_bridge can soft-delete it.
                    async with self._db.conn.execute(
                        "SELECT collection_id FROM vault_file_collections "
                        "WHERE file_id = ? AND superseded_at IS NULL "
                        "LIMIT 1",
                        (file_id,),
                    ) as cur:
                        prev_bridge = await cur.fetchone()
                    old_collection_id = prev_bridge["collection_id"] if prev_bridge else None
                    # Rebuild the bridge.
                    await self._collections_repo._move_bridge(
                        file_id=file_id,
                        old_collection_id=old_collection_id,
                        new_collection_id=collection.collection_id,
                    )
                    action = "moved"
                    logger.info(
                        "drop_watcher_file_moved",
                        extra={
                            "file_id": file_id,
                            "old_path": old_path,
                            "new_path": posix_path,
                            "old_collection_id": old_collection_id,
                            "new_collection_id": collection.collection_id,
                        },
                    )
                else:
                    # Truly fresh insert (new file_id, new path).
                    await self._db.conn.execute(
                        """
                        INSERT INTO vault_files
                            (file_id, source_path, content_sha256, mtime, size_bytes,
                             text, text_source, orphaned_at)
                        VALUES (?, ?, ?, ?, ?, NULL, NULL, NULL)
                        """,
                        (file_id, posix_path, content_sha256, mtime, size_bytes),
                    )
                    action = "inserted"
            elif existing["content_sha256"] == content_sha256:
                # Same content at same path. Just update mtime/size (the
                # file was re-touched — e.g., edited metadata, git pull).
                file_id = existing["file_id"]
                await self._db.conn.execute(
                    "UPDATE vault_files SET mtime = ?, size_bytes = ? WHERE file_id = ?",
                    (mtime, size_bytes, file_id),
                )
                action = "linked_existing"
            else:
                # Different content at same path. Delete old row (cascades
                # to vault_file_collections), insert new. The bridge
                # history is lost for the old file_id; that's OK because
                # the file is gone.
                old_file_id = existing["file_id"]
                await self._db.conn.execute(
                    "DELETE FROM vault_files WHERE file_id = ?",
                    (old_file_id,),
                )
                await self._db.conn.execute(
                    """
                    INSERT INTO vault_files
                        (file_id, source_path, content_sha256, mtime, size_bytes,
                         text, text_source, orphaned_at)
                    VALUES (?, ?, ?, ?, ?, NULL, NULL, NULL)
                    """,
                    (file_id, posix_path, content_sha256, mtime, size_bytes),
                )
                action = "replaced"
                logger.info(
                    "drop_watcher_file_replaced",
                    extra={
                        "old_file_id": old_file_id,
                        "new_file_id": file_id,
                        "path": posix_path,
                    },
                )

            # 8. INLINE bridge insert (Sprint 19 LLM review, 2026-07-10):
            # call add_file_to_collection inside the SAME transaction as the
            # vault_files insert. If this fails after vault_files commit,
            # the file exists in vault_files but is not linked to the
            # collection (orphan). The fix: inline the INSERT INTO
            # vault_file_collections here, before the COMMIT. The
            # collection is guaranteed to exist (we just got/created it).
            try:
                await self._db.conn.execute(
                    """
                    INSERT INTO vault_file_collections
                        (file_id, collection_id, added_at)
                    VALUES (?, ?, ?)
                    """,
                    (file_id, collection.collection_id, _utc_now_iso()),
                )
            except sqlite3.IntegrityError as e:
                # UNIQUE PK violation = bridge already exists (idempotent
                # restart recovery). Ignore.
                msg = str(e)
                if "UNIQUE constraint failed" in msg and "vault_file_collections" in msg:
                    pass
                else:
                    raise

            await self._db.conn.commit()
            if action != "replaced":
                logger.info(
                    "drop_watcher_file_inserted",
                    extra={
                        "file_id": file_id,
                        "path": posix_path,
                        "collection_id": collection.collection_id,
                        "size_bytes": size_bytes,
                        "content_sha256": content_sha256,
                    },
                )
        except sqlite3.IntegrityError as e:
            await self._db.conn.rollback()
            msg = str(e)
            if "UNIQUE constraint failed" in msg and "vault_files" in msg:
                # Race: another watcher beat us to the INSERT in a parallel
                # process_path call. Re-fetch the existing row and link it.
                async with self._db.conn.execute(
                    "SELECT file_id FROM vault_files WHERE source_path = ?",
                    (posix_path,),
                ) as cur:
                    row = await cur.fetchone()
                if row is not None:
                    file_id = row["file_id"]
                    action = "linked_existing"
                    # Also link the bridge in a separate transaction (since
                    # the outer one was rolled back due to the race).
                    # Note (Sprint 19 LLM review 2026-07-10): rely on
                    # aiosqlite's implicit transaction management rather
                    # than a manual `BEGIN IMMEDIATE`. aiosqlite
                    # auto-begins on the first execute() and tracks state
                    # itself; mixing explicit BEGIN with the implicit
                    # transaction can leave the connection in an
                    # inconsistent state. The UNIQUE constraint on
                    # vault_file_collections is the safety net for the
                    # race between the SELECT above and the INSERT here.
                    try:
                        await self._db.conn.execute(
                            """
                            INSERT INTO vault_file_collections
                                (file_id, collection_id, added_at)
                            VALUES (?, ?, ?)
                            """,
                            (file_id, collection.collection_id, _utc_now_iso()),
                        )
                        await self._db.conn.commit()
                    except sqlite3.IntegrityError as bridge_err:
                        await self._db.conn.rollback()
                        if "UNIQUE constraint failed" in str(
                            bridge_err
                        ) and "vault_file_collections" in str(bridge_err):
                            pass  # already linked (race lost)
                        else:
                            raise
                    except BaseException:
                        with suppress(Exception):
                            await self._db.conn.rollback()
                        raise
                else:
                    # Vanishingly rare: row was deleted between constraint
                    # fail and re-fetch. Re-raise so the caller logs it.
                    raise
            else:
                raise
        except BaseException:
            with suppress(Exception):
                await self._db.conn.rollback()
            raise
        finally:
            self._db._write_lock.release()

        # 9. Write manifest (overwrites any previous manifest at this path).
        # CRITICAL contract (Sprint 19 R1 integration check): the key MUST
        # be `vault_file_id`, NOT `id`. process_inbox._read_manifest_for()
        # (ingest_router.py:1559) reads `manifest.get("vault_file_id")` —
        # if we write `id` instead, the manifest is parsed as having no
        # file_id, the row never lands in ingest_jobs, and the file is
        # silently dropped from the pipeline.
        #
        # We write BOTH `vault_file_id` (canonical, used by process_inbox)
        # and `id` (alias, for human readability of the manifest file).
        # process_inbox only reads `vault_file_id`, so the alias is harmless.
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "vault_file_id": file_id,
            "id": file_id,  # alias of vault_file_id, for readability
            "path": posix_path,
            "created_at": _utc_now_iso(),
            "collection_id": collection.collection_id,
        }
        manifest_file = _manifest_path(file_path)
        with suppress(OSError):
            # Best-effort: if we can't write the manifest (read-only FS,
            # permission), the DB row is still authoritative. process_inbox()
            # will fall back to deriving file_id from content if the
            # manifest is missing.
            manifest_file.write_text(
                json.dumps(manifest, indent=2, sort_keys=True),
                encoding="utf-8",
            )

        # 10. Extract text via the appropriate extractor (Sprint 19 §4.1).
        # This is POST-PROCESSING: it happens AFTER the critical
        # vault_files + bridge transaction has committed. If extraction
        # fails or queues to ocr_pending, the file is still registered
        # in vault_files (just with text=NULL or low confidence).
        # Rationale: we don't want a missing Tesseract binary to block
        # the entire ingestion pipeline.
        if self._ocr_pending_repo is not None:
            try:
                # _process_extraction is async, but it internally runs the
                # sync Tesseract/OCR extractors via asyncio.to_thread to
                # avoid blocking the event loop (Sprint 19 LLM review
                # finding 2026-07-10: event loop was being blocked for
                # seconds per image).
                await self._process_extraction(
                    file_path=file_path,
                    ext=ext,
                    file_id=file_id,
                )
            except Exception:
                # Belt-and-suspenders: any unexpected error in extraction
                # must not affect the main return.
                logger.exception(
                    "drop_watcher_extraction_failed",
                    extra={"file_id": file_id, "path": posix_path},
                )

        return ProcessingResult(
            file_id=file_id,
            file_path=posix_path,
            collection_id=collection.collection_id,
            collection_name=collection.name,
            action=action,
        )

    async def _process_extraction(
        self,
        file_path: Path,
        ext: str,
        file_id: str,
    ) -> None:
        """Extract text from the file and update vault_files + ocr_pending.

        Called by process_path() after the critical vault_files + bridge
        transaction has committed. Failure here is non-fatal — the file
        is still registered, just with text=NULL.

        Logic per Sprint 19 §4.1:
        - confidence >= LOW_CONFIDENCE_THRESHOLD: UPDATE vault_files
          with text + text_source.
        - confidence < LOW_CONFIDENCE_THRESHOLD: INSERT into ocr_pending
          with status='pending_review' for manual review or auto-queue
          to edge PC.
        - error is set (e.g. tesseract_not_installed): INSERT into
          ocr_pending with the error in local_text; status='pending_review'.
          The future user can fix the env (install Tesseract) and the
          daemon retries via /pendingOCR or auto-catch-up.
        """
        # Lazy import to keep drop_watcher importable without all
        # extract deps installed.
        from hermes.memory.extractors import extract_for_extension

        # Extractors are synchronous (Tesseract / pymupdf / python-docx
        # all run blocking I/O). Run in a worker thread so we don't block
        # the event loop (Sprint 19 LLM review 2026-07-10).
        # Sprint 19.5 (PR-A): pass OCR settings (DPI, grayscale, lang) so
        # pymupdf_extractor can do OCR fallback on scanned PDFs.
        result = await asyncio.to_thread(
            extract_for_extension,
            file_path,
            ext=ext,
            ocr_dpi=self._ocr_fallback_dpi,
            ocr_grayscale=self._ocr_fallback_grayscale,
            ocr_lang=self._ocr_fallback_lang,
        )
        assert self._ocr_pending_repo is not None  # for type checker

        # Hard failure checks. Two error shapes we treat as "deps missing":
        # - "tesseract_not_installed": pytesseract import OK but binary missing
        # - "tesseract_deps_missing:<exc>": pytesseract package itself not installed
        # Both need a manual fix (apt install tesseract-ocr / pip install
        # pytesseract) before re-running. The user can trigger retry via
        # /pendingOCR or the daemon's catch-up pass (Slice 4c).
        if result.error is not None and (
            "not_installed" in result.error or result.error.startswith("tesseract_deps_missing")
        ):
            # Hard failure (e.g. Tesseract binary or pytesseract package
            # missing). Queue to ocr_pending with the error so the user
            # can see it and fix.
            await self._ocr_pending_repo.create(
                file_id=file_id,
                local_confidence=0.0,
                local_text=None,
                local_model=result.model,
                status="pending_review",
            )
            logger.warning(
                "drop_watcher_ocr_pending_extraction_failed",
                extra={
                    "file_id": file_id,
                    "model": result.model,
                    "error": result.error,
                },
            )
            return

        if result.error == "empty_extraction":
            # Tesseract found no text. Queue with confidence=0.0.
            await self._ocr_pending_repo.create(
                file_id=file_id,
                local_confidence=0.0,
                local_text=None,
                local_model=result.model,
                status="pending_review",
            )
            return

        if result.error is not None:
            # Other error (parse error, read error, etc.). Log + queue
            # for manual review.
            await self._ocr_pending_repo.create(
                file_id=file_id,
                local_confidence=0.0,
                local_text=result.text or None,
                local_model=result.model,
                status="pending_review",
            )
            logger.warning(
                "drop_watcher_extraction_partial_failure",
                extra={
                    "file_id": file_id,
                    "model": result.model,
                    "error": result.error,
                },
            )
            return

        # Success path. Branch on confidence per TDD §4.4.1:
        # - confidence >= self._autoqueue_threshold (default 0.85):
        #   text is good enough. Write to vault_files, NO ocr_pending
        #   row, no edge queue. User can search immediately.
        # - LOW_CONFIDENCE_THRESHOLD <= confidence < autoqueue_threshold
        #   (0.60-0.85): text written to vault_files (provisional,
        #   ranked low in search). ocr_pending row created for
        #   potential edge upgrade.
        # - confidence < LOW_CONFIDENCE_THRESHOLD (< 0.60): text=NULL
        #   in vault_files, ocr_pending row created.
        # In BOTH middle + low cases, try to auto-queue to edge if a
        # PC is online (handled by edge_coordinator.enqueue which
        # checks its own autoqueue_threshold + is_online()).
        from hermes.memory.extractors.tesseract import LOW_CONFIDENCE_THRESHOLD

        if result.confidence >= self._autoqueue_threshold:
            # Good enough. Write text to vault_files. No ocr_pending.
            await self._db.conn.execute(
                "UPDATE vault_files SET text = ?, text_source = ? WHERE file_id = ?",
                (result.text, result.model, file_id),
            )
            await self._db.conn.commit()
        elif result.confidence >= LOW_CONFIDENCE_THRESHOLD:
            # Provisional text. Write to vault_files (low-rank in
            # search) AND create ocr_pending row for edge upgrade.
            await self._db.conn.execute(
                "UPDATE vault_files SET text = ?, text_source = ? WHERE file_id = ?",
                (result.text, result.model, file_id),
            )
            await self._db.conn.commit()
            await self._ocr_pending_repo.create(
                file_id=file_id,
                local_confidence=result.confidence,
                local_text=result.text or None,
                local_model=result.model,
                status="pending_review",
            )
            await self._try_edge_enqueue(
                file_id=file_id,
                file_path=file_path,
                local_confidence=result.confidence,
            )
        else:
            # Low confidence. text=NULL, ocr_pending row created.
            await self._db.conn.execute(
                "UPDATE vault_files SET text = NULL, text_source = ? WHERE file_id = ?",
                (result.model, file_id),
            )
            await self._db.conn.commit()
            await self._ocr_pending_repo.create(
                file_id=file_id,
                local_confidence=result.confidence,
                local_text=result.text or None,
                local_model=result.model,
                status="pending_review",
            )
            await self._try_edge_enqueue(
                file_id=file_id,
                file_path=file_path,
                local_confidence=result.confidence,
            )

    async def _try_edge_enqueue(
        self,
        *,
        file_id: str,
        file_path: Path,
        local_confidence: float,
    ) -> None:
        """Best-effort: try to enqueue the file to the edge PC. Silent on
        failure (PC offline, threshold skipped, filesystem error). All
        errors are logged inside the coordinator — we just return.

        Called from `_process_extraction` after the ocr_pending row is
        created. Non-blocking for the caller: if the coordinator's
        internal state isn't ready yet, enqueue() returns False
        immediately and the catch-up pass will pick it up later.
        """
        if self._edge_coordinator is None:
            return
        try:
            await self._edge_coordinator.enqueue(
                file_id=file_id,
                path=file_path.as_posix(),
                local_confidence=local_confidence,
                requested_by="auto-queue",
            )
        except Exception:
            # Belt-and-suspenders: the coordinator's enqueue() already
            # catches OSError + DB errors and returns False. Anything
            # else (e.g. a bug in our code) should not affect the
            # watcher's main return.
            logger.exception(
                "drop_watcher_edge_enqueue_unexpected_error",
                extra={"file_id": file_id},
            )

    # ------------------------------------------------------------------
    # run — watchfiles-based loop
    # ------------------------------------------------------------------
    async def _worker(self) -> None:
        """Worker que consume de la cola y procesa cada path.

        Sale cuando recibe `None` (sentinel del shutdown).
        """
        while True:
            file_path = await self._queue.get()
            if file_path is None:
                self._queue.task_done()
                return
            try:
                await self.process_path(file_path)
            except Exception:
                logger.exception(
                    "drop_watcher_process_failed",
                    extra={"path": file_path.as_posix()},
                )
                # No relanzamos — el watcher sigue. Si el error es
                # permanente, M6 lo detectará como orphan.
            finally:
                self._queue.task_done()

    def _enqueue(self, file_path: Path) -> bool:
        """Encola `file_path` para procesamiento.

        Returns True si se encoló, False si la cola está llena (en cuyo
        caso el caller debe llamar a _record_dropped_event).

        Sprint 19 v0.5 §3: backpressure vía asyncio.Queue(maxsize).
        Si la cola está llena, el evento va a `dropped_events` para
        que M6 lo reintente más tarde.
        """
        try:
            self._queue.put_nowait(file_path)
            return True
        except asyncio.QueueFull:
            return False

    async def run(self, stop_event: asyncio.Event | None = None) -> None:
        """Loop principal: detecta archivos nuevos y los encola.

        Sprint 19 v0.5 §3 refactor: usa asyncio.Queue + N workers
        en vez de procesar inline. Esto desacopla watchfiles (rápido)
        de process_path (puede ser lento: hash + extractors) y
        provee backpressure: si la queue se llena, los eventos van
        a `dropped_events` y M6 los reintenta cada 5 min.

        Args:
            stop_event: si se setea, el loop sale limpiamente.
        """
        from watchfiles import awatch

        self._drop_root.mkdir(parents=True, exist_ok=True)
        # monitor_roots: create if missing so watchfiles has a target
        for root in self._monitor_roots:
            root.mkdir(parents=True, exist_ok=True)

        # v0.5: launch N workers (max_concurrent) that consume the queue
        num_workers = self._sem._value
        self._workers = [
            asyncio.create_task(self._worker(), name=f"drop_watcher_worker_{i}")
            for i in range(num_workers)
        ]

        logger.info(
            "drop_watcher_started",
            extra={
                "drop_root": self._drop_root.as_posix(),
                "monitor_roots": [r.as_posix() for r in self._monitor_roots],
                "queue_maxsize": self._max_pending,
                "num_workers": num_workers,
            },
        )
        stop_event = stop_event or asyncio.Event()
        try:
            # Watch drop_root + all monitor_roots together
            watch_paths = [self._drop_root, *self._monitor_roots]
            async for changes in awatch(
                *watch_paths,
                stop_event=stop_event,
                debounce=int(self._debounce_s * 1000),
                recursive=True,
            ):
                for _change_type, path_str in changes:
                    if _change_type.value != 1:  # Change.added == 1
                        continue
                    file_path = Path(path_str)
                    # Skip manifest files (process_path will too, but
                    # saves a queue slot)
                    if file_path.name == "manifest.json":
                        continue
                    if not self._enqueue(file_path):
                        # Backpressure: queue full, persist for M6
                        await self._record_dropped_event(file_path, "queue_full")
        finally:
            # Drain workers: send sentinel per worker. Use blocking
            # `put()` (not `put_nowait`) so the sentinel ALWAYS lands
            # in the queue. If the queue is full of real work,
            # `put()` waits until a worker pulls an item, then writes
            # the sentinel. R1+LLM fix: `put_nowait` + suppress was
            # lossy — if the queue was full at shutdown, the sentinel
            # was silently dropped and workers would hang forever on
            # `await self._queue.get()`. LLM PR review (Nemotron) caught
            # this; R1 missed it.
            for _ in self._workers:
                await self._queue.put(None)  # type: ignore[arg-type]
            # Wait for workers to finish current items
            await self._queue.join()
            # Cancel any workers still alive (defensive)
            for w in self._workers:
                if not w.done():
                    w.cancel()
            # Suppress cancellation noise
            for w in self._workers:
                with suppress(asyncio.CancelledError, Exception):
                    await w
            self._workers = []
            logger.info("drop_watcher_stopped")

    # ------------------------------------------------------------------
    # Startup scan (for restart recovery)
    # ------------------------------------------------------------------
    async def scan_existing(self) -> list[ProcessingResult]:
        """Procesa todos los archivos que ya están en `drop/` y
        `monitor_roots` (Sprint 19 v0.6 M2 fix).

        Se llama al startup del proceso (después de un restart o cold
        deploy). Idempotente: archivos ya en vault_files se re-procesan
        sin duplicar (action='linked_existing').

        R1 v0.6 M2: previously only scanned _drop_root, missing
        monitor_roots files added during downtime. watchfiles awatch
        only emits events for CHANGES, not for pre-existing files, so
        restart recovery must include monitor_roots explicitly.

        Returns: lista de ProcessingResult, una por archivo procesado.
        """
        results: list[ProcessingResult] = []
        # scan_existing covers drop_root + all monitor_roots
        roots = [self._drop_root, *self._monitor_roots]
        for root in roots:
            if not root.exists():
                continue
            for file_path in root.rglob("*"):
                if not file_path.is_file():
                    continue
                result = await self.process_path(file_path)
                if result.action not in {"skipped_dir", "skipped_manifest"}:
                    results.append(result)
        return results

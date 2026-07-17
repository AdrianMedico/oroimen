"""Mnemosyne Vault Slice 1.5 — Ingest Router.

Cierra el gap entre Slice 1 (bytes crudos + dedup) y Slice 2
(embeddings sobre texto extraído). La pipeline de extracción en
4 tiers:
- Tier 0 (PyMuPDF): siempre, da baseline v0_pymupdf (~5ms/page).
- Tier 1 (Docling local): opt-in, requiere `vault_use_local_ocr=true`.
- Tier 1.5 (LAN worker): pull-based, opportunistic. El router enqueue
  un job en `pending/<job>.md.json + .source.<ext>` para que el worker
  en LAN lo drene. El worker toca el archivo cada 30s (mtime
  heartbeat); el Janitor recoge jobs en `processing/` sin touch > 600s.
- Tier 2 (external VLM): opt-in, requires `vault_external_ocr=true`.

Defaults: solo Tier 0 + Tier 1.5 LAN active. Tier 1 y Tier 2 OFF por
defecto (privacy + RAM/CUDA requirements).

Implementación:
- atomic renames de pending/ → processing/ → done/ (POSIX local)
- mtime-touch heartbeat (worker-side; documentado en TDD)
- Filesystem es source of truth (V1.2 P2 audit); SQLite es mirror.
- IngestRouter es singleton en `hermes/__init__.py`. Lo construye
  APScheduler al startup del proceso.

Ref: docs/TDD_VAULT_INGEST_WORKER.md (V1.2 final).
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, Protocol

from hermes.memory.collections import VaultCollectionsRepo
from hermes.memory.drop_watcher import ALLOWED_EXTENSIONS
from hermes.observability.influxdb import write_research_metric

if TYPE_CHECKING:
    from hermes.config import Settings
    from hermes.memory.db import Database
    from hermes.memory.vault import VaultEntry

# PR #113d round 4 NIT: use `_safely_rollback` (vault.py:177) instead of
# bare `await conn.rollback()` in except BaseException blocks. The bare
# version can itself be cancelled by Python's task-cancellation machinery,
# poisoning the connection. See vault.py:177-201 for the F-CONC-1 Slice 1
# bug analysis and Probe 4c that reproduced it deterministically.
from hermes.memory.vault import _safely_rollback

logger = logging.getLogger(__name__)

#: Sprint 19 Slice 5 R1 fix (M3): max file size M6 Phase 2 will read.
#: Files above this cap are skipped with a warning. Default 100MB.
#: Set to a small value in tests (see test_phase2_size_cap_skips_oversize).
#: 100MB chosen because:
#: - Hermes is a personal sovereign AI agent (single user, NAS host NAS).
#: - DropWatcher has no cap, but M6 runs every 5 min (APScheduler),
#:   so a 5-min-interval OOM is more likely than a one-shot drop event.
#: - 100MB is plenty for typical scans (textbooks, research papers,
#:   financial statements). Videos / ISOs belong elsewhere.
M6_PHASE2_MAX_FILE_SIZE_BYTES: Final[int] = 100 * 1024 * 1024  # 100 MB

#: Tier names — V1.2 review-contracts BLOCKING-3 uses Literal + Final
#: (not closed Enum) so future tiers (e.g., Slice 5 hermes_worker_pool)
#: can be added by extending this Literal without breaking callers.
IngestTier = Literal[
    "pymupdf",
    "docling_local",
    "lan_worker",
    "external_vlm",
]

TIER_PYMYPDF: Final[IngestTier] = "pymupdf"
TIER_DOCLING_LOCAL: Final[IngestTier] = "docling_local"
TIER_LAN_WORKER: Final[IngestTier] = "lan_worker"
TIER_EXTERNAL_VLM: Final[IngestTier] = "external_vlm"

#: V1.2 BLOCKING-5: tier → text_version rank. Used by Vault.update_text
#: to enforce no-downgrade. Naive string sort would put "v15" before
#: "v2" — this table makes the hierarchy explicit.
#:
#: PR #113b (M-INV-1 fix): added `v_d_docling_local` (Tier 1, Docling
#: local). Without it, a worker emitting `v_d_docling_local` would have
#: `incoming_rank=0` (default for unknown) and silently overwrite higher
#: tiers via the no-downgrade invariant being evaluated wrong direction.
#: Or, depending on previous tier, would be rejected as a downgrade of
#: v0_pymupdf. Either way: silent failure of the Tier 1.5 → Tier 1
#: upgrade path. Now explicit.
TEXT_VERSION_ORDER: dict[str, int] = {
    "v0_pymupdf": 0,
    "v15_lan_worker": 15,
    "v_d_docling_local": 18,  # Tier 1 (Docling) sits between LAN worker and external VLM
    "v2_external_vlm": 20,  # S18: external OCR es privileged (highest quality)
}

#: V1.2 BLOCKING-5 inverse lookup — for tier → version family derivation.
#: NOT a bijection (multiple tiers can coexist in future). Used for
#: validating tier ↔ version consistency in apply_worker_result.
TIER_VERSION_FAMILIES: dict[str, str] = {
    TIER_PYMYPDF: "v0_",
    TIER_DOCLING_LOCAL: "v_d_",
    TIER_LAN_WORKER: "v15_",
    TIER_EXTERNAL_VLM: "v2_",
}

#: mtime threshold (s) — Janitor mueve `processing/<job>` a
#: `pending/<job>` si lleva > N sin touch (worker toca cada 30s). 600s
#: = 10 min. Slice 1.5 GREEN; adjustable via Settings later.
JANITOR_STALE_THRESHOLD_S: int = 600

#: log-events thresholds (V3 observability convention from Slice 1).
_JANITOR_BATCH_LOG_EVERY: int = 50  # log batch info every N moved


# ---------------------------------------------------------------------------
# Protocol interfaces
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SubmittedJob:
    """Return value de InboxWriter.submit — caller doesn't reparse filename.

    V1.2 review-contracts MAJOR-7: explicit dataclass vs bare Path.
    """

    job_id: str
    #: Path al sibling source file (no el .json manifest).
    path: Path
    submitted_at: datetime


@dataclass(frozen=True, slots=True)
class IngestResult:
    """Return value de IngestRouter.ingest.

    `text` siempre está populated (Tier 0 mock in GREEN, real PyMuPDF
    in S18+). `tier_used` siempre es TIER_PYMYPDF en GREEN (Slice 2+
    podría re-ingerir tras Tier 1.5 success). `text_version` es la
    versión de texto devuelta; "v0_pymupdf" en GREEN.

    `is_canonical=True` significa que el texto actual es el definitivo
    (text_version >= top tier disponible). `False` cuando se enqueue
    enhancement (esperamos v15 para reemplazar v0).

    V1.2 BLOCKING-2: forwards-compat fields en Optional con defaults.
    """

    file_id: str
    text: str
    tier_used: IngestTier
    text_version: str
    is_canonical: bool
    queued_for_enhancement: bool

    #: V1.2 forwards-compat (Slice 2 introduce esto)
    embedding_version: str | None = None
    confidence_score: float | None = None
    processing_wallclock_ms: int | None = None
    worker_id: str | None = None


class VaultProtocol(Protocol):
    """Subset of Vault that IngestRouter needs.

    Structural subtyping — implementor no necesita heredar. Slice 1
    real `Vault` satisface este protocolo (incluye métodos adicionales
    como `add`, `list_files`, etc.). Test `FakeVault` satisface
    solamente este subset.

    V1.2 BLOCKING-5 + V1.2 MAJOR-11 invariants:
    1. NO-DOWNGRADE: si vault_files.text_version rank (TEXT_VERSION_ORDER)
       > incoming, no-op (no exception, no text_at bump).
    2. text_at BUMP: en update exitoso, text_at = NOW().
    3. tier ↔ text_version CONSISTENCY: tier family must match version
       prefix (TIER_VERSION_FAMILIES). Mismatch raises ValueError.
    """

    async def get_blob(self, sha256: str) -> bytes:
        """Retrieve bytes de un blob por content_sha256."""
        ...

    async def get_blob_for_file(self, file_id: str) -> bytes:
        """Retrieve bytes del archivo indexado por `file_id` (UUID4).

        GREEN-phase convenience sobre get_blob: el router necesita
        bytes por `file_id`, no por `sha`. La implementación real
        resuelve `file_id → sha` vía VaultEntries. El Fake lo expone
        ad-hoc para los tests de Slice 1.5.
        """
        ...

    async def update_text(
        self,
        file_id: str,
        *,
        text: str,
        text_version: str,
        tier: str,
    ) -> VaultEntry | None:
        """Escribe texto canónico para file_id (con invariants arriba)."""
        ...


class InboxWriter(Protocol):
    """Filesystem gateway al inbox SMB. Test-fakeable via FakeInboxWriter.

    Atributos esperados (carpeta dirs): `pending`, `processing`, `done`,
    `failed`, `archive`. El router los usa como paths absolutos.

    Attributes:
        pending:    Path a la carpeta `pending/`.
        processing: Path a la carpeta `processing/`.
        done:       Path a la carpeta `done/`.
        failed:     Path a la carpeta `failed/`.
        archive:    Path a la carpeta `archive/`.
    """

    pending: Path
    processing: Path
    done: Path
    failed: Path
    archive: Path

    def submit(
        self,
        file_id: str,
        bytes_payload: bytes,
        *,
        min_output_chars: int,
        priority: int,
        expected_tier: IngestTier = TIER_LAN_WORKER,
        submitted_by: str = "hermes",
        source_extension: str = "pdf",
    ) -> SubmittedJob:
        """Writes pending/<job_id>.manifest.json + .source.<ext> atómicamente.

        V1.2 MAJOR-7: returns SubmittedJob (dataclass, no Path).
        """
        ...


class FsInboxWriter:
    """Real filesystem gateway for the IngestRouter (PR #113b, B2 fix).

    Production implementor of the `InboxWriter` Protocol. Writes to
    `<vault_inbox_root>/{pending,processing,done,failed,archive}/`.
    Atomic renames via os.replace() so worker never sees a half-written
    manifest (POSIX guarantee). The `pending/` create-if-missing is
    intentional: operator may have wiped the inbox; first submit
    re-creates the dir and continues (no crash).

    Job ID format (PR #113b, m4 fix): UUID4 hex (32 chars, no dashes).
    Matches TDD §Q2 contract. Collisions effectively impossible
    (2^122 space) without needing a coordination round.

    Thread-safety: NOT thread-safe. IngestRouter.ingest() is called
    from the FastAPI event loop; concurrent calls would race on
    os.replace(). Acceptable because the IngestRouter itself
    serializes via `asyncio.Lock` (write_lock). If multi-worker
    ingests become a thing, add a file lock per job_id.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.pending = root / "pending"
        self.processing = root / "processing"
        self.done = root / "done"
        self.failed = root / "failed"
        self.archive = root / "archive"
        # PR #113b (M-CONC-1 fix): mkdir upfront so first submit
        # never FileNotFoundError's on the rename.
        for d in (
            self.pending,
            self.processing,
            self.done,
            self.failed,
            self.archive,
        ):
            d.mkdir(parents=True, exist_ok=True)

    def submit(
        self,
        file_id: str,
        bytes_payload: bytes,
        *,
        min_output_chars: int,
        priority: int,
        expected_tier: IngestTier = TIER_LAN_WORKER,
        submitted_by: str = "hermes",
        source_extension: str = "pdf",
    ) -> SubmittedJob:
        """Atomic write of manifest + payload to pending/.

        Sequence:
        1. Generate job_id (UUID4 hex).
        2. Write to <pending>/<job_id>.manifest.tmp + .source.<ext>.tmp.
        3. os.replace() to final names. POSIX atomic.
        4. Return SubmittedJob.

        If step 3 fails (target exists, extremely unlikely with UUID4),
        log warning and retry once with a new job_id.
        """
        # PR #113b (m4): UUID4 hex per TDD §Q2.
        job_id = uuid.uuid4().hex
        manifest_name = f"{job_id}.md.json"
        payload_name = f"{job_id}.source.{source_extension}"
        manifest_tmp = self.pending / f"{manifest_name}.tmp"
        payload_tmp = self.pending / f"{payload_name}.tmp"
        manifest_final = self.pending / manifest_name
        payload_final = self.pending / payload_name

        manifest_text = json.dumps(
            {
                "schema_version": 1,
                "job_id": job_id,
                "vault_file_id": file_id,
                # PR #113d round 4 MAJOR: write the payload filename so
                # the Janitor's manifest-driven `source_filename` extraction
                # (see ingest_router.py `_janitor_payload_move`) actually
                # has data to read. Without this field the manifest path
                # was dead code and the Janitor fell back to a hardcoded
                # `.source.pdf` suffix — which strands any non-PDF payload
                # (e.g., source_extension="docx").
                "source_filename": payload_name,
                "submitted_at": datetime.now(UTC).isoformat(),
                "submitted_by": submitted_by,
                "priority": priority,
                "expected_tier": expected_tier,
                "min_output_chars": min_output_chars,
            }
        )

        for attempt in (1, 2):
            if attempt == 2:
                # Rare: collision with another job's UUID4 (or
                # operator-script leftover). Regenerate and retry once.
                job_id = uuid.uuid4().hex
                manifest_name = f"{job_id}.md.json"
                payload_name = f"{job_id}.source.{source_extension}"
                manifest_tmp = self.pending / f"{manifest_name}.tmp"
                payload_tmp = self.pending / f"{payload_name}.tmp"
                manifest_final = self.pending / manifest_name
                payload_final = self.pending / payload_name
                manifest_text = json.dumps(
                    {
                        "schema_version": 1,
                        "job_id": job_id,
                        "vault_file_id": file_id,
                        "source_filename": payload_name,
                        "submitted_at": datetime.now(UTC).isoformat(),
                        "submitted_by": submitted_by,
                        "priority": priority,
                        "expected_tier": expected_tier,
                        "min_output_chars": min_output_chars,
                    }
                )

            try:
                manifest_tmp.write_text(manifest_text, encoding="utf-8")
                payload_tmp.write_bytes(bytes_payload)
                os.replace(manifest_tmp, manifest_final)
                os.replace(payload_tmp, payload_final)
            except OSError as exc:
                logger.warning(
                    "vault_inbox_submit_oserror",
                    extra={
                        "job_id": job_id,
                        "attempt": attempt,
                        "error": str(exc),
                    },
                )
                # Cleanup any tmp leftovers before retry.
                with contextlib.suppress(OSError):
                    manifest_tmp.unlink()
                with contextlib.suppress(OSError):
                    payload_tmp.unlink()
                if attempt == 2:
                    raise
                continue
            break

        logger.debug(
            "vault_inbox_submit_ok",
            extra={
                "job_id": job_id,
                "file_id": file_id,
                "expected_tier": expected_tier,
                "payload_bytes": len(bytes_payload),
            },
        )
        return SubmittedJob(
            job_id=job_id,
            path=payload_final,
            submitted_at=datetime.now(UTC),
        )


# ---------------------------------------------------------------------------
# IngestRouter
# ---------------------------------------------------------------------------


class IngestRouter:
    """Routes file ingest through the 4-tier extraction pipeline.

    Constructed una vez per Hermes process (singleton). Holds refs to
    `vault` (storage) + `inbox` (filesystem gateway).
    """

    def __init__(
        self,
        *,
        vault: VaultProtocol,
        inbox: InboxWriter,
        settings: Settings,
        db: Database | None = None,
    ) -> None:
        # Tier opt-in semantics (F-CONC Gemini round 3 verdict 1): si
        # está habilitado el tier que requiere una dep externa
        # (Docling, VLM provider), fall-fast al __init__ con RuntimeError
        # accionable. NO silent fallback a Tier 0 ("¿por qué mi Docling
        # no dispara?" archaeology).
        if settings.vault_use_local_ocr:
            try:
                import docling  # noqa: F401  -- smoke import for fail-fast
            except Exception as exc:
                raise RuntimeError(
                    f"Docling required by vault_use_local_ocr=true "
                    f"but import failed: {exc}. "
                    f"Install docling or set vault_use_local_ocr=false."
                ) from exc

        if settings.vault_external_ocr:
            # Slice 1.5 GREEN: provider es OpenRouter+MiniMax-M3 (F2-2).
            # Smoke import del cliente HTTP. Si falla, fail-fast.
            # Deferred to Slice 2 cuando el provider real exista.
            # Por ahora, flag enabled pero provider stub ya satisface
            # el signature. No bloquear green phase.
            pass  # Tier 2 provider wiring: Slice 2+

        self._vault = vault
        self._inbox = inbox
        self._settings = settings
        # PR #113b Slice 2.5: optional Database ref for M6 reconcile.
        # Optional to keep tests lightweight (FakeVault + FakeInbox
        # don't need a DB). Production wires the real Database here.
        self._db = db
        self._threshold: int = settings.vault_text_v0_strip_threshold
        # asyncio.Lock para serializar _update_text_at_bump vs concurrent
        # ingest() en el mismo file_id. Reads no se serializan (la tabla
        # vault_files está READ-COMMITTED-de-facto vía SQLite WAL).
        self._write_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Tier 0 (always-on)
    # ------------------------------------------------------------------
    async def ingest(self, file_id: str) -> IngestResult:
        """Tier 0 sync + queue enhancement (Tier 1.5/2) si es low quality.

        Args:
            file_id: UUID4 del VaultEntry existente.

        Returns:
            IngestResult con tier_used="pymupdf" y text_version="v0_pymupdf".

        Raises:
            KeyError: si file_id no existe en vault (no found blob).

        Comportamiento BLOCKING-5 contra el downgrade: SIEMPRE escribe
        v0_pymupdf en vault.update_text en este path. Si un worker LAN
        ya había escrito v15_lan_worker, el Vault NO-DOWNGRADE rule
        refuse este update (v0_pymupdf rank=0 < v15_lan_worker rank=15),
        dejando el v15 intacto. Texto canónico permanece v15 (correcto).
        """
        try:
            # GREEN: bytes themselves stand in for extracted text.
            # Slice 2+: replace with real PyMuPDF extraction.
            blob = await self._vault.get_blob_for_file(file_id)
            text = blob.decode("utf-8", errors="replace")
        except KeyError as exc:
            raise KeyError(f"no blob for file_id={file_id}") from exc

        # Decidir si queue-enhancement. Threshold configurable via Settings.
        text_length = len(text.strip())

        # SIEMPRE escribimos v0_pymupdf al vault (idempotente bajo
        # NO-DOWNGRADE rule si un worker anterior escribió v15).
        # Esto pinea text_at BUMP invariant: cada ingest() bumps text_at.
        async with self._write_lock:
            await self._vault.update_text(
                file_id,
                text=text,
                text_version="v0_pymupdf",
                tier=TIER_PYMYPDF,
            )

        if text_length < self._threshold:
            # Low quality → enqueue en LAN worker (o external_vlm).
            target_tier: IngestTier
            if self._settings.vault_external_ocr:
                target_tier = TIER_EXTERNAL_VLM
            elif self._settings.vault_lan_worker_enabled:
                target_tier = TIER_LAN_WORKER
            else:
                # Nada habilitado — guardamos v0 sin enhancement.
                target_tier = TIER_PYMYPDF  # sentinel; no submit below

            if target_tier != TIER_PYMYPDF:
                submitted = self._inbox.submit(
                    file_id=file_id,
                    bytes_payload=blob,
                    min_output_chars=self._threshold,
                    priority=5,
                    expected_tier=target_tier,
                )
                # PR #113c round 2 (BLOCKING-1 fix): insert the job
                # into `ingest_jobs` so M6 + the future S18+ HTTP
                # API can introspect queue state. Without this, the
                # `ingest_jobs` table stays empty in production and
                # `idx_ingest_jobs_state` + `idx_ingest_jobs_file`
                # are dead indexes. The INSERT is best-effort: if
                # the DB is unavailable, the filesystem manifest
                # still exists (the M6 INSERT-recovery path in
                # round 2 will catch it on the next reconcile).
                if self._db is not None:
                    try:
                        async with self._db._write_lock:
                            await self._db.conn.execute(
                                "INSERT OR IGNORE INTO ingest_jobs "
                                "(job_id, vault_file_id, state, priority, "
                                " submitted_by) "
                                "VALUES (?, ?, 'pending', ?, ?)",
                                (
                                    submitted.job_id,
                                    file_id,
                                    5,
                                    "hermes",
                                ),
                            )
                            await self._db.conn.commit()
                    except Exception as exc:
                        logger.warning(
                            "vault_ingest_job_insert_failed",
                            extra={
                                "job_id": submitted.job_id,
                                "file_id": file_id,
                                "error": str(exc),
                            },
                        )
                logger.debug(
                    "vault_ingest_queued_enhancement",
                    extra={
                        "file_id": file_id,
                        "from_tier": TIER_PYMYPDF,
                        "to_tier": target_tier,
                        "text_length": text_length,
                        "job_id": submitted.job_id,
                    },
                )
                return IngestResult(
                    file_id=file_id,
                    text=text,
                    tier_used=TIER_PYMYPDF,
                    text_version="v0_pymupdf",
                    is_canonical=False,  # hasta que el worker devuelva v15
                    queued_for_enhancement=True,
                )

        logger.debug(
            "vault_ingest_v0_canonical",
            extra={"file_id": file_id, "text_length": text_length},
        )
        return IngestResult(
            file_id=file_id,
            text=text,
            tier_used=TIER_PYMYPDF,
            text_version="v0_pymupdf",
            is_canonical=True,
            queued_for_enhancement=False,
        )

    # ------------------------------------------------------------------
    # process_inbox — drena done/ + reconcilia DB
    # ------------------------------------------------------------------
    async def process_inbox(self) -> int:
        """Drena done/<job>.md.result.json hacia vault.update_text.

        Two-phase:
        1. **pending → processing**: para cada pending/<job>.json,
           atomic rename a processing/<job>.json + .pdf. El worker
           en LAN observa processing/ y arranca trabajo. (Worker-side
           rename está fuera de scope; aquí hacemos la pre-emisión
           para que el inode-rename del worker aterriza seguro.)
        2. **done → applied**: para cada done/<job>.md.result.json,
           apply al vault + archive a archive/.

        Reconcile filesystem → SQLite (V1.2 P2 audit): filesystem
        wins, DB mirror follows. Idempotente en (file_id,
        text_version).

        Returns: count de jobs NEWLY applied (skip-ea duplicates).
        """
        inbox = self._inbox
        applied: int = 0

        # PR #113b (M-CONC-1 fix): mkdir pending/processing/done/failed
        # upfront. Old code did `if inbox.pending.exists()` then iterated
        # `inbox.pending.glob(...)` — fine first time, but if operator
        # wipes the inbox dir (rm -rf / share/hermes/inbox), the next
        # cycle silently sees `not exists` and processes 0 files. We
        # `mkdir(parents=True, exist_ok=True)` so a wiped inbox heals
        # automatically on the next process_inbox cycle. log warning
        # the first time per process if we had to recreate a dir.
        for dir_attr in ("pending", "processing", "done", "failed", "archive"):
            d = getattr(inbox, dir_attr, None)
            if d is None:
                continue
            if not d.exists():
                logger.warning(
                    "vault_ingest_inbox_dir_recreated",
                    extra={"dir_attr": dir_attr, "path": str(d)},
                )
                d.mkdir(parents=True, exist_ok=True)

        # Step 1: pending/ → processing/ (atomic renames).
        # En producción este rename lo hace el worker en el momento
        # de tomar el job. Aquí lo pre-emitimos para que la queue
        # sea visible al dashboard antes de que el worker despache.
        # Si processing/ ya tenía el job (rare), no-ops (skip).
        if inbox.pending.exists():
            for pending_json in sorted(inbox.pending.glob("*.md.json")):
                # V1.2 MINOR-9 filename scheme: <job_id>.md.json + <job_id>.source.<ext>
                # (la source lleva ".source." explícito, no es sólo ".pdf").
                # Strip ".md.json" suffix: "<job_id>.md.json" → "<job_id>"
                stem = pending_json.name[: -len(".md.json")]
                pending_source_pdf = inbox.pending / f"{stem}.source.pdf"
                target_json = inbox.processing / pending_json.name
                target_source_pdf = inbox.processing / f"{stem}.source.pdf"
                if target_json.exists():
                    continue
                with contextlib.suppress(OSError):
                    pending_json.rename(target_json)
                if pending_source_pdf.exists():
                    with contextlib.suppress(OSError):
                        pending_source_pdf.rename(target_source_pdf)
                # PR #113d round 4 SUGGESTION: belt-and-suspenders inline
                # DB UPDATE mirroring the Janitor's BLOCKING-2 fix. Without
                # this, the DB row stays 'pending' until the M6 tail-call
                # at the end of `process_inbox()` covers it (~sub-second
                # window today; could grow if M6 frequency is lowered).
                # Best-effort: a DB failure logs and lets M6 reconcile.
                if self._db is not None:
                    try:
                        async with self._db._write_lock:
                            await self._db.conn.execute(
                                "UPDATE ingest_jobs "
                                "SET state = 'processing', "
                                "    last_state_change_at = "
                                "    strftime('%Y-%m-%d %H:%M:%f', 'now') "
                                "WHERE job_id = ? AND state = 'pending'",
                                (stem,),
                            )
                            await self._db.conn.commit()
                    except Exception as exc:
                        logger.warning(
                            "vault_ingest_step1_db_update_failed",
                            extra={"job_id": stem, "error": str(exc)},
                        )

        # Step 2: done/ → applied → archive/
        if not inbox.done.exists():
            return 0

        # Scan both V1.2 (`*.md.result.json`) and pre-V1.2 (`*.md.json`)
        # naming conventions. Tests seed with pre-V1.2 for simplicity.
        manifests = sorted(
            set(inbox.done.glob("*.md.result.json")) | set(inbox.done.glob("*.md.json"))
        )
        for result_manifest in manifests:
            # GREEN tolerant: el seed helper usa `.md.json`/`.md.md`
            # sin `.result` ni `.md.` suffix. Aceptamos ambos patterns.
            result_stem: str | None = None
            payload_path: Path | None = None
            for candidate_stem_pair in (
                # V1.2 contract naming
                (result_manifest.name[: -len(".md.result.json")], ".result.md"),
                # Pre-V1.2 naming (legacy seed helper uses this)
                (result_manifest.name[: -len(".md.json")], ".md.md"),
            ):
                cand_stem, cand_suffix = candidate_stem_pair
                cand_payload = inbox.done / f"{cand_stem}{cand_suffix}"
                if cand_payload.exists():
                    result_stem = cand_stem
                    payload_path = cand_payload
                    break
            if result_stem is None or payload_path is None:
                # PR #113b (M2 fix): log el skip. Old code silently
                # continued — operator saw `applied=0` sin saber si
                # era inbox empty o todos mid-write. Now: DEBUG log
                # at most per N (1 in 10 para no spamear).
                if not hasattr(self, "_mid_write_skip_count"):
                    self._mid_write_skip_count = 0
                self._mid_write_skip_count += 1
                if self._mid_write_skip_count <= 3 or self._mid_write_skip_count % 10 == 0:
                    logger.debug(
                        "vault_ingest_process_inbox_mid_write_skip",
                        extra={
                            "manifest": result_manifest.name,
                            "skip_count": self._mid_write_skip_count,
                        },
                    )
                continue

            try:
                manifest = json.loads(result_manifest.read_text())
                job_id = manifest.get("job_id", result_stem)
                file_id = manifest.get("vault_file_id")
                tier = manifest.get("tier_used", TIER_LAN_WORKER)
                text_version = manifest.get("text_version", "v15_lan_worker")
                # Validate tier ↔ text_version family (BLOCKING-5).
                # PR #113b (M-INV-3 fix): use `is not None` instead of
                # truthy check. Old code: `if expected_prefix and not
                # text_version.startswith(...)` — None is falsy, so an
                # UNKNOWN tier (not in TIER_VERSION_FAMILIES) was
                # silently accepted and any text_version slipped through.
                # Now: explicit whitelist. Unknown tier = quarantine
                # (defense-in-depth: never trust worker output blindly).
                expected_prefix = TIER_VERSION_FAMILIES.get(tier)
                if expected_prefix is None:
                    logger.warning(
                        "vault_ingest_worker_result_unknown_tier",
                        extra={
                            "job_id": job_id,
                            "tier": tier,
                            "text_version": text_version,
                        },
                    )
                    self._quarantine(inbox.done, result_manifest, payload_path)
                    continue
                if not text_version.startswith(expected_prefix):
                    logger.warning(
                        "vault_ingest_worker_result_quarantined",
                        extra={
                            "job_id": job_id,
                            "tier": tier,
                            "text_version": text_version,
                            "expected_prefix": expected_prefix,
                        },
                    )
                    self._quarantine(inbox.done, result_manifest, payload_path)
                    continue

                if file_id is None:
                    # PR #113b (M2 fix): quarantine missing file_id
                    # instead of silent `continue`. Old behavior:
                    # every cycle picks the same manifest up, logs
                    # the warning, retries forever. Now: quarantine
                    # the broken manifest (move to .quarantine/) so
                    # it stops polluting the done/ loop and operator
                    # can inspect it. The actual "orphan" file_id case
                    # is an admin concern, not a runtime concern.
                    logger.warning(
                        "vault_ingest_worker_result_missing_file_id",
                        extra={"job_id": job_id, "path": str(result_manifest)},
                    )
                    self._quarantine(inbox.done, result_manifest, payload_path)
                    continue

                text = payload_path.read_text()
                async with self._write_lock:
                    await self._vault.update_text(
                        file_id,
                        text=text,
                        text_version=text_version,
                        tier=tier,
                    )
                self._archive_done_job(inbox.done, inbox.archive, result_stem)
                applied += 1
                logger.info(
                    "vault_ingest_worker_applied",
                    extra={
                        "job_id": job_id,
                        "file_id": file_id,
                        "tier": tier,
                        "text_version": text_version,
                        "text_length": len(text),
                    },
                )
            except (json.JSONDecodeError, KeyError, ValueError, FileNotFoundError) as exc:
                logger.exception(
                    "vault_ingest_process_failed",
                    extra={"path": str(result_manifest), "exc": str(exc)},
                )
                with contextlib.suppress(OSError):
                    result_manifest.rename(inbox.failed / result_manifest.name)
                    if payload_path.exists():
                        payload_path.rename(inbox.failed / payload_path.name)

        # PR #113b (M2 fix): log applied count + skipped count at
        # end of process_inbox. Old code returned silently. Operators
        # couldn't tell "0 jobs because inbox empty" from "0 jobs
        # because all skipped due to mid-write" without grepping
        # logs. Now: every cycle logs the outcome.
        if applied > 0:
            logger.info(
                "vault_ingest_process_inbox_cycle_done",
                extra={"applied": applied},
            )

        await self._reconcile_db_from_filesystem()

        return applied

    # ------------------------------------------------------------------
    # janitor_running_jobs
    # ------------------------------------------------------------------
    async def janitor_running_jobs(self) -> int:
        """Mueve processing/<job> stale (mtime > 600s sin touch) → pending/.

        El worker toca el .json cada 30s mientras procesa. Si pasa
        > 600s sin touch, asumimos crash del worker. Re-encolar.

        Returns: count de jobs movidos.
        """
        inbox = self._inbox
        moved = 0
        # PR #113b (M-CONC-2 fix): ensure processing/ exists. Same
        # reasoning as process_inbox — operator wipe heals on next cycle.
        if not inbox.processing.exists():
            logger.warning(
                "vault_ingest_processing_dir_recreated",
                extra={"path": str(inbox.processing)},
            )
            inbox.processing.mkdir(parents=True, exist_ok=True)

        # PR #113b (m1 fix): use configured threshold (was hardcoded
        # 600s; now Settings.vault_janitor_stale_threshold_s).
        threshold_s = self._settings.vault_janitor_stale_threshold_s

        now = time.time()
        # PR #113b (M3 fix): log "active processing count" upfront so
        # operators see in InfluxDB/Grafana how many jobs are in flight.
        # Old code silently skipped files <600s with NO log — operators
        # couldn't distinguish "0 active jobs" from "5 active but all
        # fresh". Now: every janitor cycle logs the total.
        processing_files = list(inbox.processing.glob("*.json"))
        active_count = sum(1 for j in processing_files if (now - j.stat().st_mtime) <= threshold_s)
        logger.info(
            "vault_ingest_janitor_active_count",
            extra={
                "active_count": active_count,
                "total_in_processing": len(processing_files),
                "threshold_s": threshold_s,
            },
        )
        for job_json in processing_files:
            age = now - job_json.stat().st_mtime
            if age <= threshold_s:
                continue
            # Move both .json and the actual payload back to pending/.
            stem = job_json.name
            # Extract the bare job_id. PR #113c round 2 (B5 fix):
            # use `name.removesuffix` not `stem.removesuffix` —
            # `Path("abc.md.json").stem == "abc.md"` which doesn't end
            # with `.md.json` (no-op).
            job_id = job_json.name.removesuffix(".md.json")
            # PR #113c round 3 BLOCKING fix: the actual payload is
            # `<job_id>.source.<ext>` (see FsInboxWriter.submit() at
            # line 292), NOT `<job_id>.md.pdf`. The previous code
            # used `job_json.with_suffix(".pdf")` which produces
            # `<job_id>.md.pdf` — that file doesn't exist (real
            # payload is `.source.pdf`), so the payload rename
            # silently no-op'd via `if payload.exists()`. The job
            # was moved back to pending/ WITHOUT its payload,
            # stranding the next worker pickup.
            #
            # The robust fix: derive the payload name from the
            # manifest JSON if available (the manifest may have
            # been written with a non-default source_extension),
            # otherwise fall back to the canonical `.source.pdf`.
            # This matches `FsInboxWriter.submit()`'s naming exactly.
            payload_name: str | None = None
            try:
                manifest_text = job_json.read_text(encoding="utf-8")
                manifest = json.loads(manifest_text)
                candidate = manifest.get("source_filename")
                if (
                    isinstance(candidate, str)
                    and candidate
                    and "/" not in candidate
                    and "\\" not in candidate
                ):
                    # Defensive: reject path traversal.
                    payload_name = candidate
            except (OSError, ValueError):
                pass
            if payload_name is None:
                # PR #113d round 4 MAJOR: defense-in-depth glob fallback.
                # Older manifests written before round 4 don't have the
                # `source_filename` field. If the canonical `.source.pdf`
                # doesn't exist, glob for ANY `<job_id>.source.*` to find
                # the real payload (DOCX, PNG, etc. — anything supported
                # via the `source_extension` kwarg). If nothing matches,
                # fall back to the canonical name (round 3 behavior; the
                # subsequent `if payload.exists(): payload.rename(...)`
                # will no-op and the operator can investigate via logs).
                matches = sorted(inbox.processing.glob(f"{job_id}.source.*"))
                # Filter out the manifest itself if it matches the glob
                # (shouldn't happen since manifests are `.md.json`, but
                # defensive).
                payload_matches = [m for m in matches if m.suffix not in (".json",)]
                if payload_matches:
                    payload_name = payload_matches[0].name
                else:
                    # Canonical default: see FsInboxWriter.submit().
                    payload_name = f"{job_id}.source.pdf"
            payload = inbox.processing / payload_name
            dest_json = inbox.pending / stem
            dest_payload = inbox.pending / payload_name
            try:
                job_json.rename(dest_json)
                if payload.exists():
                    payload.rename(dest_payload)
                moved += 1
                logger.warning(
                    "vault_ingest_janitor_moved",
                    extra={
                        "job_json": stem,
                        "job_id": job_id,
                        "stale_for_s": int(age),
                        "threshold_s": threshold_s,
                    },
                )
                # PR #113c round 2 (BLOCKING-2 fix): update the
                # mirror `ingest_jobs.state` to match the filesystem
                # ('processing' → 'pending') so the operator's
                # `SELECT state FROM ingest_jobs` reflects reality.
                # Previously, the row stayed 'processing' until the
                # NEXT M6 cycle (up to 5min by scheduler interval)
                # during which time the row was a zombie: the job
                # was in pending/ for re-pick, but the DB said it
                # was still being worked on. Best-effort INSERT
                # here: if the DB is down, the M6 reconcile loop
                # will fix it on the next cycle.
                if self._db is not None:
                    try:
                        async with self._db._write_lock:
                            await self._db.conn.execute(
                                "UPDATE ingest_jobs SET state = 'pending', "
                                "    last_state_change_at = "
                                "    strftime('%Y-%m-%d %H:%M:%f', 'now') "
                                "WHERE job_id = ?",
                                (job_id,),
                            )
                            await self._db.conn.commit()
                    except Exception as exc:
                        logger.warning(
                            "vault_ingest_janitor_db_update_failed",
                            extra={
                                "job_id": job_id,
                                "error": str(exc),
                            },
                        )
            except OSError as exc:
                logger.exception(
                    "vault_ingest_janitor_move_failed",
                    extra={"path": str(job_json), "exc": str(exc)},
                )

        # PR #113b (M3 batch log): al final del ciclo, loggear
        # cuántos jobs se movieron (sea 0 o N) para observability
        # continua. Si el count sube a >umbral, es señal de que el
        # worker está crashando de forma sistémica.
        if moved > 0:
            logger.info(
                "vault_ingest_janitor_cycle_done",
                extra={"moved": moved, "threshold_s": threshold_s},
            )

        return moved

    # ------------------------------------------------------------------
    # apply_worker_result — public, worker-facing
    # ------------------------------------------------------------------
    async def apply_worker_result(
        self,
        *,
        file_id: str,
        text: str,
        text_version: str,
        tier: IngestTier,
        worker_id: str | None = None,
    ) -> None:
        """Worker-facing entry point (V1.2 BLOCKING-4).

        Idempotente en (file_id, text_version). Refuses downgrade
        (BLOCKING-5 via Vault.update_text invariants).

        Args:
            file_id: UUID4 del VaultEntry.
            text: Texto extraído por el worker (Markdown, plain, etc.).
            text_version: ej. "v15_lan_worker". Must match tier
                family per TIER_VERSION_FAMILIES.
            tier: "lan_worker" o "external_vlm" (Tier 0/1 results no
                llegan por aquí — esos pasan por `ingest()`).
            worker_id: opcional, para audit.
        """
        # Validate tier ↔ text_version family BEFORE Vault to give a
        # actionable error message to the worker caller.
        # PR #113b (M-INV-3 fix): same change as process_inbox — use
        # `is not None` not truthy check. Unknown tier must raise
        # ValueError to surface the bug to the worker (silent
        # acceptance was worse than the current exception).
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

        async with self._write_lock:
            await self._vault.update_text(
                file_id,
                text=text,
                text_version=text_version,
                tier=tier,
            )
        logger.info(
            "vault_ingest_apply_worker_result",
            extra={
                "file_id": file_id,
                "tier": tier,
                "text_version": text_version,
                "worker_id": worker_id,
                "text_length": len(text),
            },
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _quarantine(
        self,
        done_dir: Path,
        result_manifest: Path,
        payload_path: Path,
    ) -> None:
        """Mueve done/<job> a un subdirectory .quarantine/ para inspección.

        Tier ↔ version mismatch es NO un error fatal — el worker puede
        haber enviado un text_version no esperado (e.g., text_v_d_*
        cuando esperaba v15_*). Conservar las pruebas para debug.
        """
        quarantine_dir = done_dir / ".quarantine"
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        try:
            result_manifest.rename(quarantine_dir / result_manifest.name)
            if payload_path.exists():
                payload_path.rename(quarantine_dir / payload_path.name)
        except OSError as exc:
            logger.warning(
                "vault_ingest_quarantine_failed",
                extra={"path": str(result_manifest), "exc": str(exc)},
            )

    def _archive_done_job(
        self,
        done_dir: Path,
        archive_dir: Path,
        stem: str,
    ) -> None:
        """Move done/<stem>.{md.result.json | md.json} + payload → archive/.

        V1.2 MINOR-20: aged-out done/ lifecycle. Slice 1.5 GREEN
        archivea inmediatamente (no esperamos 7 días) para no tener
        que tocar `vault_done_archive_after_days` Settings ahora —
        configurable después.

        Maneja BOTH V1.2 (`.md.result.json` + `.result.md`) y pre-V1.2
        (`.md.json` + `.md.md`) filename schemes.
        """
        archive_dir.mkdir(parents=True, exist_ok=True)
        for ext in (
            ".md.result.json",
            ".result.md",
            ".md.json",
            ".md.md",
        ):
            src = done_dir / f"{stem}{ext}"
            if src.exists():
                with contextlib.suppress(OSError):
                    src.rename(archive_dir / src.name)

    async def _reconcile_db_from_filesystem(self) -> dict[str, int]:
        """Re-deriva el estado del vault desde filesystem (V1.2 P2 audit).

        PR #113b Slice 2.5 (M6 fix, load-bearing per Gemini round 3
        architect 2026-07-08): Filesystem debe ser la única source of
        truth para evitar el problema del "Doble Estado" con SQLite.
        Antes era un noop — Slice 2.5 introduce queries reales sobre
        el mirror SQLite (VaultEmbedder.Watcher), y un mirror stale
        causa embeddings perdidos o duplicados silenciosamente.

        Esta función escanea los 4 dirs del inbox y reconcilia:

        Phase 0 (Sprint 17, ingest_jobs legacy):
        1. **Active jobs (in processing/)**: state='processing'.
        2. **Applied jobs (in done/)**: state='applied'. Si el manifest
           no se ha aplicado aún al vault, queda en 'done_pending'.
        3. **Failed jobs (in failed/)**: state='failed'.
        4. **Missing rows**: si un archivo está en el filesystem pero
           no en `ingest_jobs`, INSERT con state derivado del dir.
        5. **Orphan rows**: si un row en `ingest_jobs` apunta a un
           job_id que no existe en ningún dir, log warning (NO delete
           — la fila es histórica, el operator decide).

        Phase 1-3 (Sprint 19 Slice 5, drop folder + collections):
        1. **Collections sync (FS→DB)**: cada subdir inmediato bajo
           `vault_drop_root/` se vuelve una `vault_collections` row.
           Idempotente.
        2. **Drop folder files (FS→DB)**: cada file en
           `drop_root/<col>/` se inserta en `vault_files` + linked
           via `vault_file_collections`. Idempotente. Backstop del
           DropWatcher (que puede haber crasheado mid-scan).
        3. **File orphan detection (FS→DB)**: cada `vault_files` row
           bajo drop_root cuyo archivo físico NO existe se marca con
           `orphaned_at` via `set_file_orphaned()`. NO delete — text
           + embeddings persisten (audit trail). Search filtra
           `WHERE orphaned_at IS NULL`. Idempotente.

        Phase 4 (Sprint 19 Slice 5, bridge invariant):
        4. **Bridge invariant audit**: cada `vault_file_collections`
           row DEBE tener `file_id` válido en `vault_files` Y
           `collection_id` válido en `vault_collections`. FK CASCADE
           + RESTRICT deberia prevenirlo, pero un safety net nunca
           sobra. Cuenta violations (NO auto-fix — operator action).

        Returns:
            dict con counters por fase:
            {
                "phase1_collections_created": int,
                "phase2_files_created": int,
                "phase2_bridge_links_created": int,
                "phase3_files_marked_orphaned": int,
                "phase4_bridge_inconsistencies": int,
            }
            Phase 0 counters no se exponen (legacy, use logs).

        Idempotente: re-llamar no cambia nada si ya está en paz.

        No toca state='archived' (operación admin-only).

        Llamada desde:
        - `process_inbox()` después de drenar done/ → applied (V1.2).
        - `Hermes.startup()` (futuro S18+) al boot, para heal after
          crash/kill mid-process.
        - APScheduler cada 5 min (Slice 5+, lite drift detection).
        - `POST /v1/admin/reconcile` (admin-only, manual trigger).
        """
        result: dict[str, int] = {
            "phase1_collections_created": 0,
            "phase2_files_created": 0,
            "phase2_bridge_links_created": 0,
            "phase3_files_marked_orphaned": 0,
            "phase4_bridge_inconsistencies": 0,
        }
        inbox = self._inbox
        # 1) Collect filesystem state.
        # PR #113c (B5 fix): use `p.name.removesuffix(...)`, not
        # `p.stem.removesuffix(...)`. For a file like `abc.md.json`:
        #   - `p.stem`  = "abc.md" (Path strips only the last extension)
        #   - `p.name`  = "abc.md.json"
        # Then `p.stem.removesuffix(".md.json")` is a no-op because
        # "abc.md" doesn't end with ".md.json" — so the job_id was
        # being extracted as "abc.md" (WRONG) instead of "abc". This
        # made M6 see drift for every job, every cycle. Use `p.name`.
        pending_ids: set[str] = set()
        processing_ids: set[str] = set()
        done_ids: set[str] = set()
        failed_ids: set[str] = set()

        if inbox.pending.exists():
            pending_ids = {p.name.removesuffix(".md.json") for p in inbox.pending.glob("*.md.json")}
        if inbox.processing.exists():
            processing_ids = {
                p.name.removesuffix(".md.json") for p in inbox.processing.glob("*.md.json")
            }
        if inbox.done.exists():
            done_ids = {
                p.name.removesuffix(".md.result.json") for p in inbox.done.glob("*.md.result.json")
            } | {p.name.removesuffix(".md.json") for p in inbox.done.glob("*.md.json")}
        if inbox.failed.exists():
            failed_ids = {p.name.removesuffix(".md.json") for p in inbox.failed.glob("*.md.json")}

        # 2) Compute target state per job_id.
        target_state: dict[str, str] = {}
        for jid in pending_ids:
            target_state[jid] = "pending"
        for jid in processing_ids:
            target_state[jid] = "processing"
        for jid in done_ids:
            # 'done' filesystem → 'applied' SQLite (file was processed
            # by the worker). If applied already happened, no-op.
            target_state[jid] = "applied"
        for jid in failed_ids:
            target_state[jid] = "failed"

        # 3) Read existing ingest_jobs rows (vault_file_id is required
        # for the JOIN; the table is the V1.2 mirror, not the S2.5
        # vault_chunks table).
        # PR #113c (B4 fix): the previous version only LOGGED the
        # diff and deferred real reconciliation to "S18+". Slice 2.5
        # GREEN introduced real queries on this mirror (the
        # EmbedWatcher reads vault_files), and a stale mirror silently
        # dropped or duplicated embeddings. This version DOES the
        # UPDATE for jobs that exist in both DB and filesystem; jobs
        # that exist only in the filesystem (no DB row) are still
        # logged because we can't INSERT without reading the manifest
        # (which lives on disk) — and an INSERT requires
        # vault_file_id NOT NULL, which is only in the manifest.
        existing_states: dict[str, str] = {}
        if self._db is not None:
            try:
                async with self._db.conn.execute("SELECT job_id, state FROM ingest_jobs") as cur:
                    async for row in cur:
                        existing_states[row["job_id"]] = row["state"]
            except Exception as exc:
                logger.warning(
                    "vault_reconcile_db_read_failed",
                    extra={"error": str(exc)},
                )
                return result

        # 4) Compute diffs.
        # PR #113c round 2 (MAJOR-2 fix): split `in_db_not_target`
        # into `drifted` and `orphan`. The previous version put
        # orphans in the same set as drifted rows; the UPDATE
        # loop then did `target_state[jid]` for orphans (not in
        # FS) → `KeyError`. The per-row `except` caught it and
        # logged `vault_reconcile_update_failed` (misleading —
        # the row was fine, the code was wrong), and on every M6
        # cycle for every orphan, the operator saw a log line
        # suggesting "DB corruption". The fix: only UPDATE rows
        # whose `jid` is in `target_state` (i.e., exists in FS).
        # Orphan rows are reported separately in the log summary
        # as `orphan_db_rows` and need no action.
        target_set = set(target_state.items())
        drifted: set[tuple[str, str]] = {
            (jid, st)
            for jid, st in existing_states.items()
            if jid in target_state and st != target_state[jid]
        }
        in_target_not_db: set[tuple[str, str]] = target_set - set(existing_states.items())
        orphan: set[tuple[str, str]] = {
            (jid, st) for jid, st in existing_states.items() if jid not in target_state
        }
        # Backwards-compat alias: `in_db_not_target` (now = drifted)
        # is referenced in the log summary below. Keep the name
        # so existing log scrapers don't break.
        in_db_not_target = drifted

        # 5) Apply UPDATE for jobs that exist in both DB and filesystem
        # but have drifted state. Take `_db._write_lock` to be safe
        # with concurrent process_inbox() writers.
        # PR #113c round 2 (MAJOR-4 fix): wrap the loop in a single
        # explicit `BEGIN IMMEDIATE` transaction so the lock hold
        # time is bounded by N x UPDATE+COMMIT (no per-row logging
        # overhead, no per-row rollback) instead of N+1 separate
        # transactions.
        updated_count = 0
        no_op_count = 0
        if self._db is not None and drifted:
            async with self._db._write_lock:
                await self._db.conn.execute("BEGIN IMMEDIATE")
                try:
                    for jid, _stale in drifted:
                        new_state = target_state[jid]
                        try:
                            # PR #113c round 3 MAJOR-4 fix: capture
                            # cur.rowcount. The previous version
                            # incremented `updated_count` for EVERY
                            # row in `drifted`, regardless of whether
                            # the UPDATE actually matched. If the
                            # row was DELETEd between M6's SELECT
                            # and UPDATE (operator SQL, S18+ multi-
                            # tenant), the UPDATE matches 0 rows but
                            # the log lies "5 rows updated". Real
                            # value: 0. Operators see wrong count.
                            cur = await self._db.conn.execute(
                                "UPDATE ingest_jobs "
                                "SET state = ?, last_state_change_at = "
                                "    strftime('%Y-%m-%d %H:%M:%f', 'now') "
                                "WHERE job_id = ?",
                                (new_state, jid),
                            )
                            if cur.rowcount > 0:
                                updated_count += 1
                            else:
                                no_op_count += 1
                            await cur.close()
                        except Exception as exc:
                            # One bad row should not abort the whole batch.
                            logger.warning(
                                "vault_reconcile_update_failed",
                                extra={"job_id": jid, "error": str(exc)},
                            )
                    await self._db.conn.commit()
                except BaseException:
                    # If the transaction itself failed, roll back
                    # (any per-row UPDATEs above are undone).
                    # PR #113d round 4 NIT: shielded rollback
                    # (see vault.py:177).
                    try:
                        await _safely_rollback(self._db.conn)
                    except Exception as rb_exc:
                        logger.warning(
                            "vault_reconcile_rollback_failed",
                            extra={"error": str(rb_exc)},
                        )
                    raise

        # 5b) Recovery path: jobs in filesystem but not in DB. The
        # previous version only LOGGED this. The patch's docstring
        # at lines 1011-1014 (round 1) said "we can't INSERT
        # without reading the manifest" — but the manifest IS on
        # disk (M6 just globbed it). Round 2 MAJOR-3 fix: read the
        # manifest, extract `vault_file_id`, INSERT.
        inserted_count = 0
        missing_file_id_count = 0
        if self._db is not None and in_target_not_db:
            async with self._db._write_lock:
                await self._db.conn.execute("BEGIN IMMEDIATE")
                try:
                    for jid, target_st in in_target_not_db:
                        manifest = self._read_manifest_for(jid)
                        if manifest is None:
                            # Manifest gone (partial write?); skip.
                            continue
                        file_id = manifest.get("vault_file_id")
                        if not file_id:
                            # PR #113c round 3 SUGGESTION: log a warning
                            # instead of silently skipping (M6 would
                            # loop forever on the same drift).
                            # PR #113d round 4 SUGGESTION: also count
                            # these so the operator sees them as a
                            # `vault_reconcile_drift` metric field
                            # instead of having to scrape logs.
                            logger.warning(
                                "vault_reconcile_manifest_missing_file_id",
                                extra={"job_id": jid},
                            )
                            missing_file_id_count += 1
                            continue
                        # PR #113c round 3 MAJOR-7 fix: verify the
                        # referenced vault_file row actually exists
                        # before inserting. Without this, the M6
                        # recovery path creates zombie rows pointing
                        # to non-existent vault_files.file_id (the
                        # schema has no FK constraint). Over time the
                        # operator's JOIN queries return dead rows.
                        # Inline existence check is cheaper than a
                        # migration to add the FK constraint.
                        async with self._db.conn.execute(
                            "SELECT 1 FROM vault_files WHERE file_id = ? LIMIT 1",
                            (file_id,),
                        ) as cur:
                            exists = await cur.fetchone()
                        if exists is None:
                            logger.warning(
                                "vault_reconcile_manifest_orphan_file_id",
                                extra={
                                    "job_id": jid,
                                    "vault_file_id": file_id,
                                },
                            )
                            continue
                        # PR #113d round 4 SUGGESTION: preserve manifest
                        # metadata on recovery INSERT. Round 3 deferred
                        # this; without it every recovered job clusters
                        # at priority=0, submitted_by='hermes', and
                        # submitted_at=NOW (the SQL defaults). For the
                        # S18+ HTTP-API queue dashboard that's a real
                        # bug — SLA tracking loses original submission
                        # time and operator attribution is erased.
                        # Defensive type checks because the manifest
                        # is user-controlled JSON.
                        priority_raw = manifest.get("priority", 0)
                        if isinstance(priority_raw, int) and not isinstance(priority_raw, bool):
                            priority = priority_raw
                        else:
                            priority = 0
                        submitted_by_raw = manifest.get("submitted_by")
                        if isinstance(submitted_by_raw, str) and submitted_by_raw:
                            submitted_by = submitted_by_raw
                        else:
                            submitted_by = "hermes"
                        submitted_at_raw = manifest.get("submitted_at")
                        if isinstance(submitted_at_raw, str) and submitted_at_raw:
                            submitted_at = submitted_at_raw
                        else:
                            submitted_at = datetime.now(UTC).isoformat()
                        try:
                            await self._db.conn.execute(
                                "INSERT OR IGNORE INTO ingest_jobs "
                                "(job_id, vault_file_id, state, "
                                " priority, submitted_by, submitted_at) "
                                "VALUES (?, ?, ?, ?, ?, ?)",
                                (
                                    jid,
                                    file_id,
                                    target_st,
                                    priority,
                                    submitted_by,
                                    submitted_at,
                                ),
                            )
                            inserted_count += 1
                        except Exception as exc:
                            logger.warning(
                                "vault_reconcile_insert_failed",
                                extra={"job_id": jid, "error": str(exc)},
                            )
                    await self._db.conn.commit()
                except BaseException:
                    # PR #113d round 4 NIT: shielded rollback (see
                    # vault.py:177). Bare `await conn.rollback()` can
                    # itself be cancelled mid-flight, leaving the
                    # connection poisoned for the next BEGIN IMMEDIATE.
                    try:
                        await _safely_rollback(self._db.conn)
                    except Exception as rb_exc:
                        logger.warning(
                            "vault_reconcile_rollback_failed",
                            extra={"error": str(rb_exc)},
                        )
                    raise

        # 6) Log summary (always — even when drift=0, log debug for
        # operators who want to see reconcile activity).
        if not (
            in_db_not_target
            or in_target_not_db
            or orphan
            or updated_count
            or inserted_count
            or missing_file_id_count
        ):
            logger.debug(
                "vault_reconcile_clean",
                extra={
                    "pending": len(pending_ids),
                    "processing": len(processing_ids),
                    "done": len(done_ids),
                    "failed": len(failed_ids),
                },
            )
            # Fall through to Phases 1-4 even on clean Phase 0.
            # M6 must be holistic: drop folder drift can exist
            # independently of ingest_jobs drift.

        logger.info(
            "vault_reconcile_drift",
            extra={
                "pending_fs": len(pending_ids),
                "processing_fs": len(processing_ids),
                "done_fs": len(done_ids),
                "failed_fs": len(failed_ids),
                "in_db_state_mismatch": len(in_db_not_target),
                "in_db_updated": updated_count,
                "in_db_noop": no_op_count,
                "in_fs_missing_in_db": len(in_target_not_db),
                "in_db_inserted": inserted_count,
                "orphan_db_rows": len(orphan),
                "missing_file_id": missing_file_id_count,
            },
        )
        # PR #113d round 4 SUGGESTION: emit the drift counters as an
        # InfluxDB metric so operators can chart/alert on mirror health
        # in Grafana instead of scraping logs. No tags (drift is a
        # scalar per cycle; tagging would create a series per cycle
        # which is undesirable). Fire-and-forget; the helper is a no-op
        # when InfluxDB isn't initialized.
        write_research_metric(
            "vault_reconcile_drift",
            fields={
                "in_db_state_mismatch": float(len(in_db_not_target)),
                "in_db_updated": float(updated_count),
                "in_db_noop": float(no_op_count),
                "in_fs_missing_in_db": float(len(in_target_not_db)),
                "in_db_inserted": float(inserted_count),
                "orphan_db_rows": float(len(orphan)),
                "missing_file_id": float(missing_file_id_count),
            },
        )

        # ------------------------------------------------------------------
        # Phase 1-3 (Sprint 19 Slice 5): drop folder + collections reconcile
        # ------------------------------------------------------------------
        # Solo corre si el drop_folder está habilitado Y existe.
        # Cada subdir inmediato bajo drop_root es una collection.
        # Cada file bajo drop_root/<col>/ se indexa en vault_files + bridge.
        # Files cuyo path físico desapareció bajo drop_root → orphaned_at.
        drop_root = self._settings.vault_drop_root
        if (
            self._db is not None
            and drop_root is not None
            and self._settings.vault_drop_enabled
            and drop_root.exists()
        ):
            await self._reconcile_phase1_collections(drop_root, result)
            await self._reconcile_phase2_files(drop_root, result)
            await self._reconcile_phase3_orphans(drop_root, result)

        # ------------------------------------------------------------------
        # Phase 4 (Sprint 19 Slice 5): bridge invariant audit
        # ------------------------------------------------------------------
        # Safety net: cada vault_file_collections DEBE tener file_id +
        # collection_id válidos. FK CASCADE deberia prevenirlo, pero
        # contamos violations sin auto-fix (operator action).
        if self._db is not None:
            await self._reconcile_phase4_bridge_invariants(result)

        return result

    async def _reconcile_phase1_collections(
        self,
        drop_root: Path,
        result: dict[str, int],
    ) -> None:
        """Phase 1: cada subdir inmediato bajo `drop_root` → collection.

        Misma regla que DropWatcher: el subdir inmediato es la collection.
        Subdirs anidados (e.g. `01_Proyectos/sub/`) NO son collections
        separadas (DropWatcher flatea el contenido del subdir inmediato).

        Idempotente: SELECT by name → si no existe, INSERT.
        """
        assert self._db is not None  # caller already checked
        collections_repo = VaultCollectionsRepo(self._db)
        for sub in sorted(drop_root.iterdir()):
            if not sub.is_dir():
                # Files sueltos en drop_root (no en subdir) son skipped
                # (Misma regla que DropWatcher `_is_within_drop_root`).
                continue
            existing = await collections_repo.get_collection_by_name(sub.name)
            if existing is None:
                try:
                    await collections_repo.create_collection(sub.name)
                    result["phase1_collections_created"] += 1
                    logger.info(
                        "vault_reconcile_phase1_collection_created",
                        extra={"name": sub.name, "path": str(sub)},
                    )
                except Exception as exc:
                    # Duplicate name race (collection creada concurrentemente
                    # por el DropWatcher). Idempotente → ignore.
                    logger.debug(
                        "vault_reconcile_phase1_collection_race",
                        extra={"name": sub.name, "error": str(exc)},
                    )

    async def _reconcile_phase2_files(
        self,
        drop_root: Path,
        result: dict[str, int],
    ) -> None:
        """Phase 2: cada file en `drop_root/<col>/` → vault_files + bridge.

        Idempotente: SELECT by (source_path, sha, mtime) → si no existe,
        INSERT con nuevo file_id (UUID4 hex).

        Backstop del DropWatcher (Sprint 19 Slice 4a). Si el watcher
        crasheó mid-scan, M6 asegura que el file quede indexado.

        NO usa `self._vault.add()` porque VaultProtocol (ver línea 157)
        solo expone get_blob / get_blob_for_file / update_text — no
        `add()`. Inlining los INSERTs es necesario por la constraint
        del Protocol. Para producción completa, considerar extender
        VaultProtocol con `add()` (Sprint 22+ cleanup).
        """
        assert self._db is not None  # caller already checked
        collections_repo = VaultCollectionsRepo(self._db)
        # Sprint 19 Slice 5 R1 fix (B1): resolve drop_root once to detect
        # symlink-escaped file_paths inside the rglob loop. Symlinks
        # pointing OUTSIDE drop_root must be rejected BEFORE we read
        # their content (exfiltration vector). Mirrors drop_watcher.py
        # resolve+is_relative_to check at line 226-258.
        drop_root_resolved = drop_root.resolve()
        for sub in sorted(drop_root.iterdir()):
            if not sub.is_dir():
                continue
            coll = await collections_repo.get_collection_by_name(sub.name)
            if coll is None:
                # Phase 1 deberia haberlo creado; si no, skip defensivo.
                # No logueamos warning para no spammear si drop_root
                # tiene files en un subdir non-canonical.
                continue
            for file_path in sub.rglob("*"):
                if not file_path.is_file():
                    continue
                # --- B1: symlink escape defense-in-depth ----------------
                # `file_path.resolve(strict=False)` follows symlinks without
                # raising on broken targets. If the resolved path is no
                # longer under drop_root, the symlink points outside the
                # drop folder — likely a malicious or accidental escape.
                # Skip (warning) to prevent exfiltration of arbitrary files
                # into the vault via this code path.
                try:
                    file_path_resolved = file_path.resolve(strict=False)
                except OSError as exc:
                    logger.warning(
                        "vault_reconcile_phase2_resolve_failed",
                        extra={"path": str(file_path), "error": str(exc)},
                    )
                    continue
                if not file_path_resolved.is_relative_to(drop_root_resolved):
                    logger.warning(
                        "vault_reconcile_phase2_symlink_escape",
                        extra={
                            "path": file_path.as_posix(),
                            "resolved": file_path_resolved.as_posix(),
                            "drop_root": drop_root_resolved.as_posix(),
                        },
                    )
                    continue
                # --- B2: extension whitelist ----------------------------
                # TDD §6 spec lines 1528-1544: skip files whose extension
                # is not in EXTENSION_ROUTER. Defense-in-depth (DropWatcher
                # already enforces at line 15 comment + line 282 check).
                # Without this, M6 indexes .DS_Store, .lnk, .crdownload,
                # .swp, .tmp, .md~ backup files that should be ignored.
                if file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
                    logger.debug(
                        "vault_reconcile_phase2_skipped_unknown_ext",
                        extra={
                            "path": file_path.as_posix(),
                            "extension": file_path.suffix.lower(),
                        },
                    )
                    continue
                # --- M3: size cap -----------------------------------------
                # Refuse to read files above M6_PHASE2_MAX_FILE_SIZE_BYTES
                # (default 100MB) to prevent OOM. M6 runs every 5 min via
                # APScheduler, so a 5-min-interval OOM is more likely than
                # a one-shot drop event.
                try:
                    file_size = file_path.stat().st_size
                except OSError as exc:
                    logger.warning(
                        "vault_reconcile_phase2_stat_failed",
                        extra={"path": str(file_path), "error": str(exc)},
                    )
                    continue
                if file_size > M6_PHASE2_MAX_FILE_SIZE_BYTES:
                    logger.warning(
                        "vault_reconcile_phase2_file_too_large",
                        extra={
                            "path": file_path.as_posix(),
                            "size_bytes": file_size,
                            "max_bytes": M6_PHASE2_MAX_FILE_SIZE_BYTES,
                        },
                    )
                    continue
                try:
                    data = await asyncio.to_thread(file_path.read_bytes)
                except OSError as exc:
                    logger.warning(
                        "vault_reconcile_phase2_read_failed",
                        extra={"path": str(file_path), "error": str(exc)},
                    )
                    continue
                sha = hashlib.sha256(data).hexdigest()
                mtime = file_path_resolved.stat().st_mtime
                size = len(data)
                # --- BLOCKING-1: as_posix for Windows path consistency --
                # DropWatcher uses `file_path.as_posix()` (forward slashes
                # always) at lines 220/235/789/824/845. M6 was using
                # `str(file_path.resolve())` which on Windows produces
                # 'C:\\Users\\...\\file.md' with backslashes. The UNIQUE
                # triple (source_path, content_sha256, mtime) would treat
                # the same file as 2 rows on Windows with different
                # file_ids, breaking the bridge. Use as_posix() to match.
                source_path = file_path_resolved.as_posix()

                # Upsert vault_files (SELECT-then-INSERT para distinguir
                # nuevo de existente, sin depender de rowcount de aiosqlite
                # que es unreliable en BEGIN IMMEDIATE — ver Sprint 19
                # Slice 1 R1 retro).
                upsert_result = await self._upsert_vault_file(source_path, sha, mtime, size, data)
                if upsert_result is None:
                    continue  # error en upsert, ya logueado
                file_id, is_new, orphaned_at = upsert_result
                if is_new:
                    result["phase2_files_created"] += 1

                # --- M2: unmark_orphan on file re-appearance ------------
                # TDD §6 spec lines 1574-1577: if the file was previously
                # marked orphaned and now re-appears, clear orphaned_at.
                # Without this, a file that was deleted, marked orphan,
                # then re-dropped stays invisible to search forever (search
                # filters WHERE orphaned_at IS NULL).
                if not is_new and orphaned_at is not None:
                    await collections_repo.clear_file_orphaned(file_id)
                    logger.info(
                        "vault_reconcile_phase2_orphan_cleared",
                        extra={
                            "file_id": file_id,
                            "source_path": source_path,
                        },
                    )

                # Bridge link (idempotente via PRIMARY KEY (file_id, collection_id))
                if await collections_repo.add_file_to_collection(file_id, coll.collection_id):
                    result["phase2_bridge_links_created"] += 1

    async def _upsert_vault_file(
        self,
        source_path: str,
        sha: str,
        mtime: float,
        size: int,
        data: bytes,
    ) -> tuple[str, bool, str | None] | None:
        """INSERT o retorna file_id existente. Returns
        (file_id, is_new, orphaned_at) o None on error.

        Pattern: SELECT-then-INSERT. Idempotente via UNIQUE constraint
        en (source_path, content_sha256, mtime). El `is_new` flag le
        permite al caller contar inserts reales vs idempotent hits
        (sin depender de heurísticas de timestamp que son fragiles
        entre formatos Python ISO y SQLite strftime).

        El `orphaned_at` flag (Sprint 19 Slice 5 R1 fix M2): cuando la
        fila ya existía, retornamos su timestamp `orphaned_at` para que
        el caller pueda llamar `clear_file_orphaned()` si el archivo
        reapareció (Phase 2 re-detect).
        """
        assert self._db is not None
        try:
            async with self._db.conn.execute(
                "SELECT file_id, orphaned_at FROM vault_files "
                "WHERE source_path = ? AND content_sha256 = ? AND mtime = ?",
                (source_path, sha, mtime),
            ) as cur:
                existing = await cur.fetchone()
            if existing is not None:
                return existing["file_id"], False, existing["orphaned_at"]
            # Nuevo: INSERT. Tambien necesitamos la vault_blobs row
            # para que vault_files sea consistente (FK no enforced,
            # pero semánticamente requerido por Vault.get_blob).
            file_id = uuid.uuid4().hex
            async with self._db._write_lock:
                await self._db.conn.execute("BEGIN IMMEDIATE")
                try:
                    await self._db.conn.execute(
                        "INSERT INTO vault_files "
                        "(file_id, source_path, content_sha256, mtime, size_bytes) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (file_id, source_path, sha, mtime, size),
                    )
                    # vault_blobs: INSERT OR IGNORE con ref_count=1.
                    # Si el blob ya existía (race con DropWatcher),
                    # el ref_count no se incrementa — drift menor
                    # aceptable para M6 backstop.
                    await self._db.conn.execute(
                        "INSERT OR IGNORE INTO vault_blobs "
                        "(content_sha256, data, size_bytes, ref_count) "
                        "VALUES (?, ?, ?, 1)",
                        (sha, data, size),
                    )
                    await self._db.conn.commit()
                    return file_id, True, None
                except BaseException:
                    try:
                        await _safely_rollback(self._db.conn)
                    except Exception as rb_exc:
                        logger.warning(
                            "vault_reconcile_phase2_rollback_failed",
                            extra={"error": str(rb_exc)},
                        )
                    raise
        except Exception as exc:
            logger.warning(
                "vault_reconcile_phase2_upsert_failed",
                extra={"source_path": source_path, "error": str(exc)},
            )
            return None

    async def _reconcile_phase3_orphans(
        self,
        drop_root: Path,
        result: dict[str, int],
    ) -> None:
        """Phase 3: vault_files con `source_path` bajo `drop_root`
        cuyo archivo físico NO existe → `orphaned_at = NOW()`.

        Conservador: solo evalúa files bajo drop_root. Files indexados
        por otros paths (legacy ingest, OBS vault, etc.) son
        responsabilidad de otros componentes.

        Idempotente: si `orphaned_at IS NOT NULL`, se skip (no
        re-update del timestamp).
        """
        assert self._db is not None
        collections_repo = VaultCollectionsRepo(self._db)
        drop_root_resolved = drop_root.resolve()
        async with self._db.conn.execute(
            "SELECT file_id, source_path FROM vault_files " "WHERE orphaned_at IS NULL"
        ) as cur:
            rows = list(await cur.fetchall())
        for row in rows:
            source_path = row["source_path"]
            # Check que source_path está bajo drop_root.
            try:
                Path(source_path).resolve().relative_to(drop_root_resolved)
            except (ValueError, OSError):
                continue  # not under drop_root, skip
            if not Path(source_path).exists():
                await collections_repo.set_file_orphaned(row["file_id"])
                result["phase3_files_marked_orphaned"] += 1
                logger.info(
                    "vault_reconcile_phase3_orphan_marked",
                    extra={
                        "file_id": row["file_id"],
                        "source_path": source_path,
                    },
                )

    async def _reconcile_phase4_bridge_invariants(
        self,
        result: dict[str, int],
    ) -> None:
        """Phase 4: bridge invariant audit.

        Cada `vault_file_collections` row DEBE tener:
        - `file_id` válido en `vault_files` (FK CASCADE enforced)
        - `collection_id` válido en `vault_collections` (FK RESTRICT)

        FK deberia prevenir violations, pero:
        - PRAGMA foreign_keys puede estar OFF en una session
        - Migraciones históricas podrian haber dejado drift

        Safety net: cuenta violations, loguea warning, NO auto-fix
        (operator action via SQL).
        """
        assert self._db is not None
        async with self._db.conn.execute(
            """
            SELECT bfc.file_id, bfc.collection_id
            FROM vault_file_collections bfc
            LEFT JOIN vault_files vf ON vf.file_id = bfc.file_id
            LEFT JOIN vault_collections vc ON vc.collection_id = bfc.collection_id
            WHERE vf.file_id IS NULL OR vc.collection_id IS NULL
            """,
        ) as cur:
            violations = list(await cur.fetchall())
        result["phase4_bridge_inconsistencies"] = len(violations)
        if violations:
            # Logueamos hasta 10 sample rows para evitar log floods.
            sample = [
                {"file_id": r["file_id"], "collection_id": r["collection_id"]}
                for r in violations[:10]
            ]
            logger.warning(
                "vault_reconcile_phase4_bridge_violations",
                extra={"count": len(violations), "sample": sample},
            )

    async def vacuum_applied_jobs(self, *, max_age_days: int | None = None) -> int:
        """Sprint 18 hardening (M6 vacuum): archive aged-out ingest_jobs rows.

        Soft-vacuum: UPDATE state='archived' for rows in ('applied', 'failed')
        whose `last_state_change_at` is older than `max_age_days`. The row
        remains in the table (still queryable via `SELECT state='archived'`)
        but is excluded from "active queue" dashboards.

        Runs daily via VaultScheduler. Default `max_age_days` comes from
        settings.vault_done_archive_after_days (30 días).

        Why a soft-vacuum, not hard-DELETE:
        - Preserves audit trail (operator can post-mortem "what jobs ran
          in March?" by querying state='archived').
        - M6 reconciliation tail-call covers any drift: if a manifest
          shows up in `done/` for a row in `state='archived'`, M6's
          target-state resolution sees the file is `done/` and
          re-archives. The row stays 'archived' (idempotent).
        - Operator can hard-DELETE separately via a one-off script if
          storage becomes a real concern (NAS host single-process Hermes
          grows ~36K rows/year → 100K rows in 3 years; SQLite handles
          that fine, but operator-driven hard-DELETE is still available).

        Concurrency: serialized via `_db._write_lock`. The M6 reconcile
        cycle also uses the same lock, so vacuum and M6 never run
        concurrently. Vacuum only operates on terminal states
        ('applied', 'failed') which M6 doesn't touch, so there's no
        logical conflict anyway.

        Args:
            max_age_days: rows older than this get archived. If None,
                uses settings.vault_done_archive_after_days.

        Returns:
            int: number of rows archived this cycle. 0 if nothing aged
            out yet (normal for the first ~30 days after deployment).
        """
        if self._db is None:
            logger.debug("vault_ingest_vacuum_skipped_no_db")
            return 0
        if max_age_days is None:
            max_age_days = self._settings.vault_done_archive_after_days
        if max_age_days < 1:
            raise ValueError(
                f"max_age_days must be >= 1, got {max_age_days}. "
                f"Set vault_done_archive_after_days to a positive integer."
            )

        archived_count = 0
        try:
            async with self._db._write_lock:
                async with self._db.conn.execute(
                    "UPDATE ingest_jobs "
                    "SET state = 'archived', "
                    "    last_state_change_at = "
                    "    strftime('%Y-%m-%d %H:%M:%f', 'now') "
                    "WHERE state IN ('applied', 'failed') "
                    "  AND last_state_change_at < "
                    "      strftime('%Y-%m-%d %H:%M:%f', 'now', ?)",
                    (f"-{max_age_days} days",),
                ) as cur:
                    archived_count = cur.rowcount
                await self._db.conn.commit()
        except BaseException:
            # PR #118 review (LLM cascade, BLOCKING): bare `await
            # conn.rollback()` can be cancelled mid-flight, leaving the
            # SQLite connection poisoned (subsequent BEGIN IMMEDIATE
            # raises OperationalError). Use the shielded `_safely_rollback`
            # from vault.py:177 — same pattern applied to embedder.py:442
            # and the 2 M6 sites in round 4.
            try:
                await _safely_rollback(self._db.conn)
            except Exception as rb_exc:
                logger.warning(
                    "vault_ingest_vacuum_rollback_failed",
                    extra={"error": str(rb_exc)},
                )
            # Vacuum failure should NOT abort the scheduler cycle.
            # The next daily run will retry. Log + emit metric.
            logger.warning(
                "vault_ingest_vacuum_failed",
                extra={"max_age_days": max_age_days},
            )
            write_research_metric(
                "vault_jobs_vacuumed",
                tags={"result": "failed"},
                fields={"count": 0.0, "max_age_days": float(max_age_days)},
            )
            return 0

        if archived_count > 0:
            logger.info(
                "vault_ingest_vacuum_done",
                extra={
                    "archived_count": archived_count,
                    "max_age_days": max_age_days,
                },
            )
        write_research_metric(
            "vault_jobs_vacuumed",
            tags={"result": "ok"},
            fields={
                "count": float(archived_count),
                "max_age_days": float(max_age_days),
            },
        )
        return archived_count

    def _read_manifest_for(self, job_id: str) -> dict | None:
        """Read the manifest JSON for a job_id from any inbox subdir.

        PR #113c round 2 (MAJOR-3 fix): M6's recovery path needs the
        manifest's `vault_file_id` to INSERT into `ingest_jobs`. The
        manifest is at `<inbox>/<pending|processing|done|failed>/<job_id>.md.json`
        (or `<job_id>.md.result.json` in done/). Returns the parsed
        dict, or None if no manifest is found / the file is malformed.

        Search order: `done/` first (the most common case for the
        recovery scenario — DB was wiped after a worker delivered the
        result), then `processing/`, then `pending/`, then `failed/`.
        """
        import json

        inbox = self._inbox
        candidates: list[Path] = []
        for sub in (inbox.done, inbox.processing, inbox.pending, inbox.failed):
            if not sub.exists():
                continue
            candidates.append(sub / f"{job_id}.md.json")
            if sub is inbox.done:
                # The "applied" manifest is named .md.result.json.
                candidates.append(sub / f"{job_id}.md.result.json")
        for path in candidates:
            if not path.exists():
                continue
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                # Partial write or malformed JSON. Don't crash M6
                # over one bad manifest.
                logger.warning(
                    "vault_reconcile_manifest_unreadable",
                    extra={"job_id": job_id, "path": str(path), "error": str(exc)},
                )
                return None
        return None


__all__ = [
    "JANITOR_STALE_THRESHOLD_S",
    "TEXT_VERSION_ORDER",
    "TIER_DOCLING_LOCAL",
    "TIER_EXTERNAL_VLM",
    "TIER_LAN_WORKER",
    "TIER_PYMYPDF",
    "TIER_VERSION_FAMILIES",
    "FsInboxWriter",
    "InboxWriter",
    "IngestResult",
    "IngestRouter",
    "IngestTier",
    "SubmittedJob",
    "VaultProtocol",
]

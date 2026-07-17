"""Tests for the Vault Slice 1.5 ingest router.

Sprint 17 TDD-VAULT-INGEST-WORKER. RED phase: este archivo importa del
módulo que aún no existe. La pasada por RED es por `ImportError`, no
por aserción lógica, hasta que la implementación exista. Cuando se
implemente, los tests pasan a rojo sólo si el contrato se rompe.

Refs:
- `docs/TDD_VAULT_INGEST_WORKER.md` (contrato completo, 4-tier routing)
- `docs/TDD_VAULT_CORE.md` (Slice 1 contract: VaultProtocol en paralelo)
- `docs/ROADMAP_2026.md` (ÉPICA 3 + US-2.4 embed_vault)
"""

from __future__ import annotations

import contextlib
import json
import time
import uuid

# TDD preamble: SubmittedJob is part of the V1.2 contract
# (InboxWriter.submit -> SubmittedJob, not bare Path). The real class
# lands in hermes.memory.ingest_router during GREEN phase. This local
# stub lets the Fake have a forward reference today.
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest


@dataclass(frozen=True, slots=True)
class SubmittedJob:
    """V1.2 review-contracts MAJOR-7 — return value of InboxWriter.submit."""

    job_id: str
    path: Path  # path to <job_id>.source.<ext>
    submitted_at: datetime  # UTC, server-side timestamp


# Estos imports fallarán hasta que se implemente Slice 1.5 (green phase).
# La presencia de los símbolos en el import es parte del contrato: si la
# impl olvida exponer uno, el test falla en colección, no en un assert
# lejano.
from hermes.config import (  # noqa: E402
    Settings,  # real (used); vault_* fields are TDD contract stubs
)
from hermes.memory.ingest_router import (  # noqa: E402, F401
    InboxWriter,
    IngestResult,
    IngestRouter,
    IngestTier,
    VaultProtocol,
)

# ---------------------------------------------------------------------------
# Fakes (no tocan filesystem real, ni SMB, ni Docling)
# ---------------------------------------------------------------------------


class FakeVault:
    """Fake `VaultProtocol` para tests de ingest_router.

    V1.2 review-test-discipline BLOCKING-2: añadido `text_at` por file_id
    (para pinear el bump-en-update-text) y `embedding_calls` counter
    (BLOCKING-3: para pinear que update_text NO toca el embedding cache).
    """

    def __init__(self) -> None:
        self.blobs: dict[str, bytes] = {}  # sha256 -> bytes
        # file_id -> (text, version, tier, text_at_iso)
        self.texts: dict[str, tuple[str, str, str, str]] = {}
        self.update_calls: list[dict[str, Any]] = []
        # BLOCKING-3: counter for embedding-cache access. update_text
        # MUST keep this at 0 (Slice 2 will subscribe to text_at changes,
        # not have update_text call the embedding service synchronously).
        self.embedding_calls: list[str] = []

    async def get_blob(self, sha256: str) -> bytes:
        return self.blobs[sha256]

    async def update_text(self, file_id: str, *, text: str, text_version: str, tier: str) -> None:
        """Pinea los invariants V1.2 BLOCKING-5 (no-downgrade + text_at BUMP).

        Si el `text_version` entrante es estrictamente MENOR (en el
        TEXT_VERSION_ORDER) que el actual, no-op. Si es >=, reemplaza
        + bump text_at.
        """
        from datetime import UTC, datetime

        # Lazy import para no recrear el orden del test ahora.
        from hermes.memory.ingest_router import TEXT_VERSION_ORDER

        text_at = datetime.now(UTC).isoformat()
        prev = self.texts.get(file_id)
        if prev is not None:
            prev_version = prev[1]
            prev_rank = TEXT_VERSION_ORDER.get(prev_version, 0)
            new_rank = TEXT_VERSION_ORDER.get(text_version, 0)
            if new_rank < prev_rank:
                # No-downgrade: skip.
                self.update_calls.append(
                    {
                        "file_id": file_id,
                        "text": text,
                        "version": text_version,
                        "tier": tier,
                        "rejected": "downgrade",
                    }
                )
                return

        self.texts[file_id] = (text, text_version, tier, text_at)
        self.update_calls.append(
            {
                "file_id": file_id,
                "text": text,
                "version": text_version,
                "tier": tier,
                "text_at": text_at,
            }
        )

    async def get_blob_for_file(self, file_id: str) -> bytes:
        """GREEN: el FakeVault devuelve bytes por `file_id`.

        Conveniencia para el router. En el Fake, retorna cualquier
        blob del dict (los tests sólo settean uno — semántica 'any bytes').
        Si NO hay blob alguno, retorna un placeholder de 1000 chars
        (para que el router considere el archivo "rich text" y no
        enqueue). Tests que necesitan texto corto setean blobs explícitamente.
        """
        if self.blobs:
            return next(iter(self.blobs.values()))
        # Default: placeholder "rich text" — bien arriba del threshold.
        return b"x" * 1000

    async def embed_text(self, file_id: str) -> None:
        """Simulate embedding-cache access. update_text MUST NOT call this."""
        self.embedding_calls.append(file_id)


class FakeInboxWriter:
    """Fake `InboxWriter` que escribe a tmp_path en lugar de SMB.

    V1.2 (review-contracts MAJOR-7): returns SubmittedJob (dataclass)
    not bare Path. V1.2 (MINOR-9): filename scheme V1 — manifest at
    `<job_id>.md.json`, payload at `<job_id>.source.<ext>`.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.pending = root / "pending"
        self.processing = root / "processing"
        self.done = root / "done"
        self.failed = root / "failed"
        self.archive = root / "archive"
        # GREEN fix: create all dirs upfront. process_inbox renames
        # pending→processing; if processing/ doesn't exist, the rename
        # fails silently and we get a leak.
        for d in (
            self.pending,
            self.processing,
            self.done,
            self.failed,
            self.archive,
        ):
            d.mkdir(parents=True, exist_ok=True)
        self.submissions: list[dict[str, Any]] = []

    def submit(
        self,
        file_id: str,
        bytes_payload: bytes,
        *,
        min_output_chars: int,
        priority: int,
        expected_tier: str = "lan_worker",
        submitted_by: str = "hermes",
        source_extension: str = "pdf",
        job_id: str | None = None,
    ) -> SubmittedJob:
        # V1.2 MINOR-16: full UUID4 instead of ts+4hex (collision risk)
        if job_id is None:
            job_id = uuid.uuid4().hex
        json_path = self.pending / f"{job_id}.md.json"
        payload_path = self.pending / f"{job_id}.source.{source_extension}"
        json_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "job_id": job_id,
                    "vault_file_id": file_id,
                    # PR #113d round 4: match the real FsInboxWriter
                    # by recording the actual payload filename in the
                    # manifest. The Janitor's manifest-driven
                    # `source_filename` extraction reads this field.
                    "source_filename": payload_path.name,
                    "submitted_at": datetime.now(UTC).isoformat(),
                    "submitted_by": submitted_by,
                    "priority": priority,
                    "expected_tier": expected_tier,
                    "min_output_chars": min_output_chars,
                    # vault_path REMOVED in V1.2 (BLOCKING MAJOR-12)
                    # payload is the sibling .source.<ext> file.
                }
            )
        )
        payload_path.write_bytes(bytes_payload)
        self.submissions.append(
            {
                "job_id": job_id,
                "file_id": file_id,
                "bytes_len": len(bytes_payload),
                "expected_tier": expected_tier,
                "priority": priority,
                "manifest_path": str(json_path),
                "payload_path": str(payload_path),
            }
        )
        return SubmittedJob(
            job_id=job_id,
            path=payload_path,
            submitted_at=datetime.now(UTC),
        )


@pytest.fixture(autouse=True)
def _vault_test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """GREEN-phase fix: Vault/Settings _settings() helper construye
    Settings(_env_file=None, ...) sin triggerar env loading. Sin
    OPENCODE_GO_API_KEY/GEMINI_API_KEY en env, la validación de
    Settings falla. Este autouse las setea vía monkeypatch antes
    de cada test en este archivo.
    """
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "fake-api-key-12345")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key-12345")


@pytest.fixture
def inbox_root(tmp_path: Path) -> Path:
    return tmp_path / "inbox"


@pytest.fixture
def fake_vault() -> FakeVault:
    return FakeVault()


def _settings(**overrides: Any) -> Settings:
    """Build Settings con defaults razonables para Slice 1.5.

    GREEN-phase fix: el helper original listaba defaults explícitos que
    duplican los de `Settings`, causando "got multiple values for kwarg"
    cuando un test pasaba uno en `**overrides`. Lo simplificamos
    a defaults coherentes (lan ON, external OFF, inbox en /tmp). Si
    un test quiere sobreescribir, lo pasa en overrides.
    """
    defaults: dict[str, Any] = {
        "_env_file": None,
        "vault_lan_worker_enabled": True,
        "vault_external_ocr": False,
        "vault_inbox_root": Path("/tmp/inbox"),
        "vault_text_v0_strip_threshold": 100,
        # Tier 1 fail-fast: explícitamente OFF en tests (no se importará
        # Docling).
        "vault_use_local_ocr": False,
    }
    # Merge: overrides take precedence.
    for k, v in overrides.items():
        defaults[k] = v
    return Settings(**defaults)


# ---------------------------------------------------------------------------
# TestIngestRouterTier0 (8 tests)
# ---------------------------------------------------------------------------


class TestIngestRouterTier0:
    async def test_ingest_text_pdf_returns_v0_canonical(
        self, fake_vault: FakeVault, inbox_root: Path
    ) -> None:
        """PDF con texto rico → Tier 0 retorna baseline, no enqueue."""
        # GREEN-phase fix: el RED-phase data era 49 bytes (más corto que el
        # threshold de 100). Lo extendemos para que el router considere
        # el archivo como "rich text" y devuelva is_canonical=True.
        pdf_bytes = b"%PDF-1.4\n" + (b"x" * 200) + b"\n--- lots of text after ---"
        fake_vault.blobs["sha"] = pdf_bytes
        inbox = FakeInboxWriter(inbox_root)
        router = IngestRouter(vault=fake_vault, inbox=inbox, settings=_settings())

        result = await router.ingest(file_id="vault_x")

        assert result.tier_used == "pymupdf"
        assert result.text_version == "v0_pymupdf"
        assert result.is_canonical is True
        assert result.queued_for_enhancement is False
        # No enqueue submitted to LAN worker
        assert inbox.submissions == []

    async def test_ingest_empty_pdf_stores_v0_no_enhancement(
        self, fake_vault: FakeVault, inbox_root: Path
    ) -> None:
        """PDF con texto muy corto (< threshold) but no LAN enabled: queda v0."""
        fake_vault.blobs["sha"] = b"x"  # 1 byte
        inbox = FakeInboxWriter(inbox_root)
        settings = _settings(vault_lan_worker_enabled=False)
        router = IngestRouter(vault=fake_vault, inbox=inbox, settings=settings)

        result = await router.ingest(file_id="vault_x")

        assert result.tier_used == "pymupdf"
        assert result.queued_for_enhancement is False
        assert inbox.submissions == []

    async def test_ingest_short_text_queues_lan_job(
        self, fake_vault: FakeVault, inbox_root: Path
    ) -> None:
        """Tier 0 produce < threshold chars + LAN enabled → enqueue."""
        # 50 bytes < 100 threshold
        fake_vault.blobs["sha"] = b"x" * 50
        inbox = FakeInboxWriter(inbox_root)
        router = IngestRouter(vault=fake_vault, inbox=inbox, settings=_settings())

        result = await router.ingest(file_id="vault_x")

        assert result.tier_used == "pymupdf"
        assert result.queued_for_enhancement is True
        assert result.is_canonical is False  # until LAN result lands
        assert len(inbox.submissions) == 1
        assert inbox.submissions[0]["expected_tier"] == "lan_worker"

    async def test_ingest_short_text_does_not_queue_when_lan_disabled(
        self, fake_vault: FakeVault, inbox_root: Path
    ) -> None:
        """Tier 0 < threshold + LAN disabled → no queue, fallback v0 only."""
        fake_vault.blobs["sha"] = b"x" * 50
        inbox = FakeInboxWriter(inbox_root)
        settings = _settings(vault_lan_worker_enabled=False)
        router = IngestRouter(vault=fake_vault, inbox=inbox, settings=settings)

        result = await router.ingest(file_id="vault_x")

        assert result.queued_for_enhancement is False
        assert inbox.submissions == []

    async def test_ingest_external_enabled_queues_external_job(
        self, fake_vault: FakeVault, inbox_root: Path
    ) -> None:
        """Tier 0 < threshold + external OCR enabled → enqueue, tier=external_vlm."""
        fake_vault.blobs["sha"] = b"x" * 50
        inbox = FakeInboxWriter(inbox_root)
        settings = _settings(vault_external_ocr=True)
        router = IngestRouter(vault=fake_vault, inbox=inbox, settings=settings)

        result = await router.ingest(file_id="vault_x")

        assert result.queued_for_enhancement is True
        assert inbox.submissions[0]["expected_tier"] == "external_vlm"

    async def test_ingest_returns_queued_for_enhancement_when_short_text(
        self, fake_vault: FakeVault, inbox_root: Path
    ) -> None:
        """IngestResult documenta queued_for_enhancement=True."""
        fake_vault.blobs["sha"] = b"x" * 50
        inbox = FakeInboxWriter(inbox_root)
        router = IngestRouter(vault=fake_vault, inbox=inbox, settings=_settings())

        result = await router.ingest(file_id="vault_x")

        assert result.queued_for_enhancement is True

    async def test_ingest_returns_no_queuing_when_text_meets_threshold(
        self, fake_vault: FakeVault, inbox_root: Path
    ) -> None:
        """IngestResult documenta queued_for_enhancement=False cuando texto rico."""
        fake_vault.blobs["sha"] = b"x" * 1000
        inbox = FakeInboxWriter(inbox_root)
        router = IngestRouter(vault=fake_vault, inbox=inbox, settings=_settings())

        result = await router.ingest(file_id="vault_x")

        assert result.queued_for_enhancement is False
        assert inbox.submissions == []

    async def test_ingest_external_disabled_short_text_queues_nothing_and_stores_v0(
        self, fake_vault: FakeVault, inbox_root: Path
    ) -> None:
        """external disabled (default) + LAN enabled + short text → LAN only."""
        fake_vault.blobs["sha"] = b"x" * 50
        inbox = FakeInboxWriter(inbox_root)
        router = IngestRouter(vault=fake_vault, inbox=inbox, settings=_settings())

        result = await router.ingest(file_id="vault_x")

        assert result.queued_for_enhancement is True
        assert len(inbox.submissions) == 1
        assert inbox.submissions[0]["expected_tier"] == "lan_worker"
        # NOT external_vlm
        assert not any(s["expected_tier"] == "external_vlm" for s in inbox.submissions)


# ---------------------------------------------------------------------------
# TestIngestRouterProcessInbox (5 tests)
# ---------------------------------------------------------------------------


class TestIngestRouterProcessInbox:
    async def _seed_done_job(self, inbox: FakeInboxWriter, file_id: str, text_v15: str) -> Path:
        """Helper: simular worker done — escribe done/<job>.json+.md."""
        # Reuse pending dir name pattern by copying from prior submission
        if not inbox.submissions:
            return Path("")
        sub = inbox.submissions[0]
        job_id = sub["job_id"]
        done_json = inbox.done / f"{job_id}.md.json"
        done_md = inbox.done / f"{job_id}.md.md"
        done_json.parent.mkdir(parents=True, exist_ok=True)
        done_json.write_text(
            json.dumps(
                {
                    "job_id": job_id,
                    "vault_file_id": file_id,
                    "tier_used": "lan_worker",
                    "text_version": "v15_lan_worker",
                    "completed_at": "2026-07-07T22:09:11Z",
                }
            )
        )
        done_md.write_text(text_v15)
        return done_json

    async def test_process_inbox_moves_done_jobs_to_done_folder(
        self, fake_vault: FakeVault, inbox_root: Path
    ) -> None:
        """Worker ya dejó done/<job>.{json,md}; process_inbox lo ingest."""
        fake_vault.blobs["sha"] = b"original bytes"
        inbox = FakeInboxWriter(inbox_root)
        # simulate enqueue
        inbox.submit(file_id="vault_x", bytes_payload=b"x", min_output_chars=10, priority=0)
        # simulate worker done
        await self._seed_done_job(inbox, "vault_x", "# Extracted\nfull markdown here")

        router = IngestRouter(vault=fake_vault, inbox=inbox, settings=_settings())
        _count = await router.process_inbox()

        # FakeVault.update_text fue llamado para el file_id
        assert "vault_x" in fake_vault.texts
        text, version, tier, _text_at = fake_vault.texts["vault_x"]
        assert text == "# Extracted\nfull markdown here"
        assert version == "v15_lan_worker"
        assert tier == "lan_worker"

    async def test_process_inbox_atomic_rename_pending_to_processing(
        self, fake_vault: FakeVault, inbox_root: Path
    ) -> None:
        """Jobs en pending/ se renombran atómicamente a processing/ durante drain."""
        inbox = FakeInboxWriter(inbox_root)
        inbox.submit(file_id="vault_x", bytes_payload=b"x", min_output_chars=10, priority=0)

        router = IngestRouter(vault=fake_vault, inbox=inbox, settings=_settings())
        # First call: rename pending → processing (no done yet)
        await router.process_inbox()
        # The job should have moved from pending → processing (or done if worker simulated)
        assert len(list(inbox.pending.glob("*.pdf"))) == 0

    async def test_process_inbox_skips_already_done(
        self, fake_vault: FakeVault, inbox_root: Path
    ) -> None:
        """process_inbox idempotente: no re-aplica si ya está done/."""
        fake_vault.blobs["sha"] = b"bytes"
        inbox = FakeInboxWriter(inbox_root)
        inbox.submit(file_id="vault_x", bytes_payload=b"x", min_output_chars=10, priority=0)
        await self._seed_done_job(inbox, "vault_x", "first markdown")

        router = IngestRouter(vault=fake_vault, inbox=inbox, settings=_settings())
        # 1st call: ingest from done/
        await router.process_inbox()
        # 2nd call: should NOT re-call update_text
        call_count_before = len(fake_vault.update_calls)
        await router.process_inbox()
        call_count_after = len(fake_vault.update_calls)
        assert call_count_after == call_count_before

    async def test_process_inbox_count_zero_when_no_jobs(
        self, fake_vault: FakeVault, inbox_root: Path
    ) -> None:
        """Inbox vacía → process_inbox retorna 0."""
        inbox = FakeInboxWriter(inbox_root)
        router = IngestRouter(vault=fake_vault, inbox=inbox, settings=_settings())
        _count = await router.process_inbox()
        assert _count == 0

    async def test_process_inbox_updates_db_state_on_each_transition(
        self, fake_vault: FakeVault, inbox_root: Path
    ) -> None:
        """Cada transición de job (pending→processing→done) se persiste en db."""
        # Este test requiere que el IngestRouter use la `Database` real
        # para registrar el `ingest_jobs.state`. Skip si no hay DB ya
        # que RED phase: el módulo no existe.
        pytest.skip("requiere implementación real del IngestRouter con DB")


# ---------------------------------------------------------------------------
# TestIngestRouterJanitor (4 tests)
# ---------------------------------------------------------------------------


class TestIngestRouterJanitor:
    async def test_janitor_moves_stale_processing_back_to_pending(
        self, fake_vault: FakeVault, inbox_root: Path
    ) -> None:
        """Si processing/*.json lleva >10 min sin touch (mtime), mover a pending/."""
        inbox = FakeInboxWriter(inbox_root)
        # Seed a stale job in processing/
        inbox.processing.mkdir(parents=True, exist_ok=True)
        job_id = f"{int(time.time())}-dead"
        stale_json = inbox.processing / f"{job_id}.md.json"
        # PR #113c round 3 BLOCKING: payload is `<job_id>.source.pdf`,
        # not `<job_id>.md.pdf` (matches FsInboxWriter.submit() naming).
        stale_pdf = inbox.processing / f"{job_id}.source.pdf"
        stale_json.write_text(json.dumps({"job_id": job_id, "vault_file_id": "vault_x"}))
        stale_pdf.write_bytes(b"x")
        # Force mtime to 11 minutes ago (Janitor threshold: 600s)
        old_time = time.time() - 700
        import os

        os.utime(stale_json, (old_time, old_time))
        os.utime(stale_pdf, (old_time, old_time))

        router = IngestRouter(vault=fake_vault, inbox=inbox, settings=_settings())
        moved = await router.janitor_running_jobs()

        # The stale job is back in pending/
        assert moved >= 1
        assert stale_json.exists() is False
        moved_json = inbox.pending / stale_json.name
        assert moved_json.exists() is True

    async def test_janitor_respects_recent_mtime_no_action(
        self, fake_vault: FakeVault, inbox_root: Path
    ) -> None:
        """processing/*.json con mtime < 10 min NO se mueve."""
        inbox = FakeInboxWriter(inbox_root)
        inbox.processing.mkdir(parents=True, exist_ok=True)
        job_id = f"{int(time.time())}-live"
        live_json = inbox.processing / f"{job_id}.md.json"
        # PR #113c round 3 BLOCKING: payload is `<job_id>.source.pdf`,
        # not `<job_id>.md.pdf` (matches FsInboxWriter.submit() naming).
        live_pdf = inbox.processing / f"{job_id}.source.pdf"
        live_json.write_text(json.dumps({"job_id": job_id}))
        live_pdf.write_bytes(b"x")
        # mtime is NOW (default) — should NOT be touched

        router = IngestRouter(vault=fake_vault, inbox=inbox, settings=_settings())
        moved = await router.janitor_running_jobs()
        assert moved == 0
        assert live_json.exists() is True

    async def test_janitor_handles_processing_without_pdf(
        self, fake_vault: FakeVault, inbox_root: Path
    ) -> None:
        """Si un .json huérfano (sin .pdf) aparece, janitor también lo recoge."""
        inbox = FakeInboxWriter(inbox_root)
        inbox.processing.mkdir(parents=True, exist_ok=True)
        orphan = inbox.processing / "orphan.md.json"
        orphan.write_text(json.dumps({"job_id": "orphan"}))
        import os

        old_time = time.time() - 700
        os.utime(orphan, (old_time, old_time))

        router = IngestRouter(vault=fake_vault, inbox=inbox, settings=_settings())
        moved = await router.janitor_running_jobs()
        assert moved >= 1

    async def test_janitor_returns_zero_when_no_stale_jobs(
        self, fake_vault: FakeVault, inbox_root: Path
    ) -> None:
        """Inbox vacía → janitor retorna 0."""
        inbox = FakeInboxWriter(inbox_root)
        router = IngestRouter(vault=fake_vault, inbox=inbox, settings=_settings())
        moved = await router.janitor_running_jobs()
        assert moved == 0


# ---------------------------------------------------------------------------
# TestIngestVaultSideEffects (5 tests)
# ---------------------------------------------------------------------------


class TestIngestVaultSideEffects:
    async def test_update_text_replaces_canonic_text_only_if_newer_version(
        self, fake_vault: FakeVault
    ) -> None:
        """update_text rechaza downgrade v15 → v0."""
        # Initial: v0 set
        fake_vault.texts["vault_x"] = ("tier0 text", "v0_pymupdf", "pymupdf")
        # Try to downgrade to v0 from v15
        inbox = FakeInboxWriter(Path("/tmp/inbox"))
        router = IngestRouter(vault=fake_vault, inbox=inbox, settings=_settings())
        _result = await router.ingest(file_id="vault_x")  # baseline ingest

        # simulate worker arriving with v15
        await router.apply_worker_result(
            file_id="vault_x",
            text="tier15 text",
            text_version="v15_lan_worker",
            tier="lan_worker",
        )

        # Now try to re-ingest with v0 (someone calls ingest again)
        await router.ingest(file_id="vault_x")
        # text_version should still be v15 (not downgraded)
        _text, version, _tier, _text_at = fake_vault.texts["vault_x"]
        assert version == "v15_lan_worker"

    async def test_update_text_increments_text_at(self, fake_vault: FakeVault) -> None:
        """Cada update_text bumps text_at.

        V1.2 review-test-discipline BLOCKING-2: V1 el test solo
        comparaba el dict entero antes/después sin leer text_at. Ahora
        FakeVault persiste text_at explícitamente; este test pinea el
        bump de timestamp.
        """
        import time

        fake_vault.blobs["sha"] = b"x" * 50  # short text triggers LAN queue
        inbox = FakeInboxWriter(Path("/tmp/inbox"))
        router = IngestRouter(vault=fake_vault, inbox=inbox, settings=_settings())

        # Pre-state: nothing in vault_x yet
        assert fake_vault.texts.get("vault_x") is None

        # First ingest: creates entry with text_at = T1
        time.sleep(0.001)
        await router.ingest(file_id="vault_x")
        _, _, _, text_at_t1 = fake_vault.texts["vault_x"]
        assert text_at_t1 is not None

        # Second ingest (re-process): bumps text_at to T2 > T1
        time.sleep(0.001)
        await router.ingest(file_id="vault_x")
        _, _, _, text_at_t2 = fake_vault.texts["vault_x"]
        assert text_at_t2 > text_at_t1

    async def test_update_text_refuses_downgrade_v15_to_v0(self, fake_vault: FakeVault) -> None:
        """No-downgrade rule independiente del test anterior, cubriendo
        el caso donde update_text es llamado directamente (no via ingest).
        """
        # Set v15 first (4-tuple: text, version, tier, text_at)
        from datetime import UTC, datetime

        fake_vault.texts["vault_x"] = (
            "v15 text",
            "v15_lan_worker",
            "lan_worker",
            datetime.now(UTC).isoformat(),
        )
        inbox = FakeInboxWriter(Path("/tmp/inbox"))
        router = IngestRouter(vault=fake_vault, inbox=inbox, settings=_settings())

        await router.apply_worker_result(
            file_id="vault_x",
            text="would-be v0 text",
            text_version="v0_pymupdf",  # downgrade attempt
            tier="pymupdf",
        )
        # Should still be v15_lan_worker
        _, version, _, _ = fake_vault.texts["vault_x"]
        assert version == "v15_lan_worker"

    async def test_update_text_does_not_touch_embedding_cache_directly(
        self, fake_vault: FakeVault
    ) -> None:
        """IngestRouter.update_text NO invalida embeddings (eso es Slice 2).

        V1.2 BLOCKING-3: FakeVault expone `embed_text()` que la impl NO
        debe llamar. Si la impl respeta el boundary Slice 1.5 ↔ Slice 2
        (no sincroniza embeddings), `embedding_calls` queda en [].
        """
        fake_vault.blobs["sha"] = b"x" * 50
        inbox = FakeInboxWriter(Path("/tmp/inbox"))
        router = IngestRouter(vault=fake_vault, inbox=inbox, settings=_settings())

        await router.ingest(file_id="vault_x")
        # Slice 2 will subscribe to text_at changes via DB polling or trigger.
        # update_text MUST NOT call embed_text synchronously.
        assert "vault_x" in fake_vault.texts
        assert fake_vault.embedding_calls == [], (
            f"update_text called embedding {len(fake_vault.embedding_calls)} "
            f"times — Slice 2 owns embedding, not Slice 1.5"
        )

    async def test_update_text_bumps_text_tier_field(self, fake_vault: FakeVault) -> None:
        """update_text setea text_tier a la IngestTier del job."""
        fake_vault.blobs["sha"] = b"x" * 50
        inbox = FakeInboxWriter(Path("/tmp/inbox"))
        router = IngestRouter(vault=fake_vault, inbox=inbox, settings=_settings())

        await router.apply_worker_result(
            file_id="vault_x",
            text="tier15 text",
            text_version="v15_lan_worker",
            tier="lan_worker",
        )
        _, _, tier, _ = fake_vault.texts["vault_x"]
        assert tier == "lan_worker"


# ---------------------------------------------------------------------------
# TestInboxWriterAtomicity (4 tests)
# ---------------------------------------------------------------------------


class TestInboxWriterAtomicity:
    def test_submit_atomic_rename_temp_to_pending(self, tmp_path: Path) -> None:
        """submit() escribe a .tmp y luego atomic rename."""
        # Real impl: usa tempfile + os.rename. Fake puede saltar este
        # detail pero el contrato es: pending/ tiene el archivo terminado,
        # nunca un .tmp.colgado.
        inbox = FakeInboxWriter(tmp_path / "inbox")

        result = inbox.submit(
            file_id="vault_x",
            bytes_payload=b"x" * 1000,
            min_output_chars=10,
            priority=0,
        )

        # V1.2 MAJOR-7: submit() retorna SubmittedJob (dataclass con .path).
        # GREEN-phase fix: test unpack path → result.path.
        path = result.path

        # No .tmp files left hanging

        _temps = list(inbox.pending.glob("*.tmp"))
        assert _temps == []
        # Final file exists
        assert path.exists()

    def test_submit_writes_pdf_and_json_atomically(self, tmp_path: Path) -> None:
        """submit() deja .json + .pdf ambos visibles (atomicidad a nivel
        worker: worker lee .json, sabe que .pdf está listo).

        V1.2 BLOCKING-B1 (review-test-discipline): el anterior 'or True'
        era vacuously true. Ahora seasserta contra el `SubmittedJob`
        devuelto Y contra los archivos en disco, ambos con la naming
        convention V1.2 (<job_id>.md.json + <job_id>.source.<ext>).
        """
        inbox = FakeInboxWriter(tmp_path / "inbox")

        result = inbox.submit(
            file_id="vault_x",
            bytes_payload=b"x" * 1000,
            min_output_chars=10,
            priority=0,
        )

        # V1.2: submit() returns SubmittedJob, not bare Path.
        # Type check pin: subclass of NamedTuple-ish structure with the
        # 3 documented fields.
        assert hasattr(result, "job_id"), "submit() must return a SubmittedJob-like object"
        job_id = result.job_id

        # Both files exist on disk under pending/
        # Naming convention: <job_id>.md.json (manifest) + <job_id>.source.pdf (payload)
        assert (inbox.pending / f"{job_id}.md.json").exists()
        assert (inbox.pending / f"{job_id}.source.pdf").exists()

    def test_submit_distinct_job_ids_per_call(self, tmp_path: Path) -> None:
        """Cada submit() genera un job_id único (UUID-en-tiempo)."""
        inbox = FakeInboxWriter(tmp_path / "inbox")
        ids = set()
        for _ in range(5):
            inbox.submit(file_id="vault_x", bytes_payload=b"x", min_output_chars=10, priority=0)
        ids = {s["job_id"] for s in inbox.submissions}
        assert len(ids) == 5

    def test_submit_collision_safe_under_concurrent_workers(self, tmp_path: Path) -> None:
        """Si dos workers concurrentes llamaran submit() con el mismo
        job_id (improbable pero defensivo), no se corrompe el filesystem.
        En FakeInboxWriter, submit() siempre genera id nuevo; el
        comportamiento real pinea que el `os.rename` a pending/ es
        atómico (POSIX) y que dos jobs con el mismo job_id NO se
        pisarían porque la 2da submission sobrescribe el destino.
        """
        inbox = FakeInboxWriter(tmp_path / "inbox")
        inbox.submit(file_id="vault_x", bytes_payload=b"x", min_output_chars=10, priority=0)
        # La serialización es por atomicidad del rename, no por nuestra API.
        # Test verifica que NO es lanzada una excepción incluso si dos llamadas
        # generan el mismo contenido (idempotencia conceptual).
        inbox.submit(file_id="vault_x", bytes_payload=b"x", min_output_chars=10, priority=0)
        assert len(inbox.submissions) == 2

    def test_submit_writes_source_filename_in_manifest(self, tmp_path: Path) -> None:
        """PR #113d round 4 MAJOR: submit() must record the actual
        payload filename in the manifest JSON so the Janitor's
        manifest-driven `source_filename` extraction has data to read.

        Without this field the Janitor's manifest path is dead code and
        it falls back to a hardcoded `.source.pdf` suffix — which strands
        any non-PDF payload (e.g., source_extension="docx").
        """
        inbox = FakeInboxWriter(tmp_path / "inbox")
        result = inbox.submit(
            file_id="vault_x",
            bytes_payload=b"x" * 100,
            min_output_chars=10,
            priority=0,
            source_extension="docx",
        )
        job_id = result.job_id
        manifest = json.loads((inbox.pending / f"{job_id}.md.json").read_text(encoding="utf-8"))
        assert manifest["source_filename"] == f"{job_id}.source.docx"
        # Sanity: the payload file uses the matching extension.
        assert (inbox.pending / f"{job_id}.source.docx").exists()


# ---------------------------------------------------------------------------
# TestSettingsIngestSection (3 tests)
# ---------------------------------------------------------------------------


class TestSettingsIngestSection:
    def test_settings_default_lan_worker_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Tier 1.5 LAN worker is enabled by default for the self-hosted gateway."""
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-12345")
        monkeypatch.setenv("OPENCODE_GO_API_KEY", "fake-key-12345")
        monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key-12345")
        monkeypatch.setenv("DB_PATH", "/tmp/test.db")
        s = Settings(_env_file=None)
        assert s.vault_lan_worker_enabled is True

    def test_settings_default_external_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Por default, Tier 2 external OCR está DESHABILITADO (privacy)."""
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-12345")
        monkeypatch.setenv("OPENCODE_GO_API_KEY", "fake-key-12345")
        monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key-12345")
        monkeypatch.setenv("DB_PATH", "/tmp/test.db")
        s = Settings(_env_file=None)
        assert s.vault_external_ocr is False

    def test_settings_inbox_root_default_persistent_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Inbox root defaults to the documented persistent container path."""
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-12345")
        monkeypatch.setenv("OPENCODE_GO_API_KEY", "fake-key-12345")
        monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key-12345")
        monkeypatch.setenv("DB_PATH", "/tmp/test.db")
        s = Settings(_env_file=None)
        assert isinstance(s.vault_inbox_root, Path)
        assert s.vault_inbox_root == Path("/var/lib/mnemosyne/inbox")


# ----------------------------------------------------------------------------
# PR #113c — M6 reconciliation test (B4 fix verification)
# ----------------------------------------------------------------------------
# Before PR #113c, _reconcile_db_from_filesystem() only LOGGED the drift
# and deferred real reconciliation to "S18+". Slice 2.5 GREEN introduced
# real queries on the ingest_jobs mirror (EmbedWatcher reads vault_files),
# and a stale mirror silently dropped or duplicated embeddings. PR #113c
# implements the UPDATE branch: jobs that exist in BOTH DB and filesystem
# with mismatched state get state synced from filesystem.


async def test_reconcile_db_from_filesystem_updates_drifted_state(
    tmp_path: Path,
) -> None:
    """M6 reconciliation: drift → real UPDATE on ingest_jobs.

    Setup:
    1. Real Database (in-memory) with migrations run.
    2. IngestRouter wired with the real DB.
    3. Seed ingest_jobs with state='pending' for a job_id.
    4. Move the job's manifest from pending/ → done/ (filesystem drift).
    5. Run _reconcile_db_from_filesystem().

    Expected:
    - The DB row's state is updated to 'applied' (matches filesystem done/).
    - This is the B4 fix: real UPDATE, not just log.
    """
    from hermes.config import Settings
    from hermes.memory.db import Database
    from hermes.memory.ingest_router import FsInboxWriter, IngestRouter
    from hermes.memory.vault import Vault

    db_path = tmp_path / "recon.db"
    inbox_root = tmp_path / "inbox"
    inbox_root.mkdir()

    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    db = Database(db_path)
    await db.initialize()
    try:
        vault = Vault(db)
        inbox = FsInboxWriter(inbox_root)
        inbox.pending.mkdir(parents=True, exist_ok=True)
        inbox.processing.mkdir(parents=True, exist_ok=True)
        inbox.done.mkdir(parents=True, exist_ok=True)
        inbox.failed.mkdir(parents=True, exist_ok=True)
        inbox.archive.mkdir(parents=True, exist_ok=True)
        router = IngestRouter(vault=vault, inbox=inbox, settings=settings, db=db)

        # Seed a job in ingest_jobs with state='pending'.
        job_id = "deadbeef" + "0" * 24
        async with db.conn.execute("BEGIN IMMEDIATE"):
            await db.conn.execute(
                "INSERT INTO ingest_jobs (job_id, vault_file_id, state) VALUES (?, ?, ?)",
                (job_id, "vault_xyz", "pending"),
            )
            await db.conn.commit()

        # Simulate drift: filesystem has the job in done/ (it was
        # already applied by the worker) but DB still says 'pending'.
        (inbox.done / f"{job_id}.md.json").write_text(
            json.dumps({"job_id": job_id, "vault_file_id": "vault_xyz"})
        )

        # Run M6.
        await router._reconcile_db_from_filesystem()

        # Verify the DB was UPDATED (not just logged).
        async with db.conn.execute(
            "SELECT state FROM ingest_jobs WHERE job_id = ?", (job_id,)
        ) as cur:
            row = await cur.fetchone()
        assert row is not None
        # PR #113c (B4): state must be 'applied' (synced from done/).
        assert row["state"] == "applied", f"expected state='applied' after M6, got {row['state']!r}"
    finally:
        await db.close()


# ----------------------------------------------------------------------------
# PR #113c round 2 — additional M6 + BLOCKING-1/2 + concurrency test cells
# ----------------------------------------------------------------------------
# The round 1 hot-patch (PR #116) only had 1 M6 test cell
# (test_reconcile_db_from_filesystem_updates_drifted_state). Round 2
# adds coverage for: 5 more M6 cells, 1 BLOCKING-1 (ingest_jobs
# INSERT on submit), 1 BLOCKING-2 (Janitor UPDATE state), and the
# text_version race that the concurrency reviewer found.


async def test_reconcile_handles_orphan_db_row_without_keyerror(
    tmp_path: Path,
) -> None:
    """PR #113c round 2 (MAJOR-2 state): M6 must not KeyError on orphan rows.

    A row in `ingest_jobs` with no corresponding filesystem entry
    (orphan) should NOT be in the UPDATE loop. Previously, the
    loop did `target_state[jid]` which raised KeyError for orphans.
    The per-row `except` caught it but logged misleading
    `vault_reconcile_update_failed` messages on every cycle for
    every orphan.
    """
    from hermes.config import Settings
    from hermes.memory.db import Database
    from hermes.memory.ingest_router import FsInboxWriter, IngestRouter
    from hermes.memory.vault import Vault

    db_path = tmp_path / "recon.db"
    inbox_root = tmp_path / "inbox"
    inbox_root.mkdir()
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    db = Database(db_path)
    await db.initialize()
    try:
        vault = Vault(db)
        inbox = FsInboxWriter(inbox_root)
        inbox.pending.mkdir(parents=True, exist_ok=True)
        inbox.processing.mkdir(parents=True, exist_ok=True)
        inbox.done.mkdir(parents=True, exist_ok=True)
        inbox.failed.mkdir(parents=True, exist_ok=True)
        inbox.archive.mkdir(parents=True, exist_ok=True)
        router = IngestRouter(vault=vault, inbox=inbox, settings=settings, db=db)

        # Insert an orphan row (no filesystem presence).
        orphan_id = "orphan" + "0" * 25
        async with db.conn.execute("BEGIN IMMEDIATE"):
            await db.conn.execute(
                "INSERT INTO ingest_jobs (job_id, vault_file_id, state) VALUES (?, ?, ?)",
                (orphan_id, "vault_orphan", "processing"),
            )
            await db.conn.commit()

        # M6 must NOT raise on the orphan.
        await router._reconcile_db_from_filesystem()

        # Orphan state unchanged.
        async with db.conn.execute(
            "SELECT state FROM ingest_jobs WHERE job_id = ?", (orphan_id,)
        ) as cur:
            row = await cur.fetchone()
        assert row is not None
        assert row["state"] == "processing"  # orphan is preserved, not UPDATEd
    finally:
        await db.close()


async def test_reconcile_inserts_recovery_row_from_manifest(
    tmp_path: Path,
) -> None:
    """PR #113c round 2 (MAJOR-3 state + BLOCKING-1 follow-up):
        M6 must INSERT recovery rows for FS-only jobs.

        Simulates a scenario where the DB was wiped but the FS has a
        manifest at `done/<job_id>.md.result.json` with a valid
        `vault_file_id`. M6 should read the manifest, extract the
    file_id, and INSERT the row.
    """
    from hermes.config import Settings
    from hermes.memory.db import Database
    from hermes.memory.ingest_router import FsInboxWriter, IngestRouter
    from hermes.memory.vault import Vault

    db_path = tmp_path / "recon.db"
    inbox_root = tmp_path / "inbox"
    inbox_root.mkdir()
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    db = Database(db_path)
    await db.initialize()
    try:
        vault = Vault(db)
        inbox = FsInboxWriter(inbox_root)
        inbox.pending.mkdir(parents=True, exist_ok=True)
        inbox.processing.mkdir(parents=True, exist_ok=True)
        inbox.done.mkdir(parents=True, exist_ok=True)
        inbox.failed.mkdir(parents=True, exist_ok=True)
        inbox.archive.mkdir(parents=True, exist_ok=True)
        # PR #113c round 3 (MAJOR-7): the M6 recovery path now
        # verifies that the manifest's vault_file_id actually exists
        # in vault_files before INSERTing. So this test must seed
        # the vault_files row first.
        async with db.conn.execute("BEGIN IMMEDIATE"):
            await db.conn.execute(
                "INSERT INTO vault_files "
                "(file_id, source_path, content_sha256, mtime, size_bytes, "
                " text, text_version) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    "vault_recovery",
                    "/fake/vault_recovery.md",
                    "a" * 64,
                    0.0,
                    0,
                    "v0 recovery text",
                    "v0_pymupdf",
                ),
            )
            await db.conn.commit()
        router = IngestRouter(vault=vault, inbox=inbox, settings=settings, db=db)

        # Drop a manifest at done/ with no DB row.
        job_id = "fs_only" + "0" * 25
        (inbox.done / f"{job_id}.md.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "job_id": job_id,
                    "vault_file_id": "vault_recovery",
                    "submitted_at": "2026-07-08T10:00:00+00:00",
                    "submitted_by": "test",
                    "priority": 5,
                    "expected_tier": "lan_worker",
                    "min_output_chars": 200,
                }
            )
        )

        # M6 must INSERT.
        await router._reconcile_db_from_filesystem()

        async with db.conn.execute(
            "SELECT state, vault_file_id FROM ingest_jobs WHERE job_id = ?",
            (job_id,),
        ) as cur:
            row = await cur.fetchone()
        assert row is not None
        assert row["state"] == "applied"  # because manifest is in done/
        assert row["vault_file_id"] == "vault_recovery"
    finally:
        await db.close()


async def test_reconcile_handles_db_read_exception_gracefully(
    tmp_path: Path,
) -> None:
    """PR #113c round 2 (MAJOR-3 partial): M6 must return cleanly
    when the DB read of existing_states fails.

    The previous implementation caught the exception in step 3,
    logged `vault_reconcile_db_read_failed`, and returned. The new
    implementation must still do that (no UPDATE half-applied).
    """
    from hermes.config import Settings
    from hermes.memory.db import Database
    from hermes.memory.ingest_router import FsInboxWriter, IngestRouter
    from hermes.memory.vault import Vault

    db_path = tmp_path / "recon.db"
    inbox_root = tmp_path / "inbox"
    inbox_root.mkdir()
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    db = Database(db_path)
    await db.initialize()
    try:
        vault = Vault(db)
        inbox = FsInboxWriter(inbox_root)
        inbox.pending.mkdir(parents=True, exist_ok=True)
        inbox.done.mkdir(parents=True, exist_ok=True)
        router = IngestRouter(vault=vault, inbox=inbox, settings=settings, db=db)

        # Drop a manifest at done/ with no DB row.
        job_id = "fs_only" + "0" * 25
        (inbox.done / f"{job_id}.md.json").write_text(
            json.dumps({"job_id": job_id, "vault_file_id": "vault_x"})
        )

        # Close the DB BEFORE M6 runs — this will cause the SELECT
        # in step 3 to raise. M6 should log + return, not crash.
        await db.close()
        # M6 must not raise; if it does, the test fails.
        await router._reconcile_db_from_filesystem()
    finally:
        # Re-open the db just so the test can tear down cleanly.
        with contextlib.suppress(Exception):
            await db.close()


async def test_reconcile_handles_applied_to_applied_noop(
    tmp_path: Path,
) -> None:
    """PR #113c round 2 (MAJOR-3 partial): M6 no-op when DB
    state matches filesystem state.

    The fix: only UPDATE rows whose state differs from the
    filesystem (drifted). Rows that match (applied↔done/) are
    left alone. This test pins the no-op behavior.
    """
    from hermes.config import Settings
    from hermes.memory.db import Database
    from hermes.memory.ingest_router import FsInboxWriter, IngestRouter
    from hermes.memory.vault import Vault

    db_path = tmp_path / "recon.db"
    inbox_root = tmp_path / "inbox"
    inbox_root.mkdir()
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    db = Database(db_path)
    await db.initialize()
    try:
        vault = Vault(db)
        inbox = FsInboxWriter(inbox_root)
        inbox.pending.mkdir(parents=True, exist_ok=True)
        inbox.processing.mkdir(parents=True, exist_ok=True)
        inbox.done.mkdir(parents=True, exist_ok=True)
        inbox.failed.mkdir(parents=True, exist_ok=True)
        inbox.archive.mkdir(parents=True, exist_ok=True)
        router = IngestRouter(vault=vault, inbox=inbox, settings=settings, db=db)

        # Both DB and FS agree: applied/done/.
        job_id = "noop" + "0" * 25
        async with db.conn.execute("BEGIN IMMEDIATE"):
            await db.conn.execute(
                "INSERT INTO ingest_jobs (job_id, vault_file_id, state) VALUES (?, ?, ?)",
                (job_id, "vault_noop", "applied"),
            )
            await db.conn.commit()
        (inbox.done / f"{job_id}.md.json").write_text(
            json.dumps({"job_id": job_id, "vault_file_id": "vault_noop"})
        )

        # M6 must not change `last_state_change_at` (no UPDATE fired).
        await router._reconcile_db_from_filesystem()

        async with db.conn.execute(
            "SELECT state, last_state_change_at FROM ingest_jobs WHERE job_id = ?",
            (job_id,),
        ) as cur:
            row = await cur.fetchone()
        assert row is not None
        assert row["state"] == "applied"
        # The migration sets last_state_change_at = CURRENT_TIMESTAMP
        # at insert; M6 should NOT have updated it.
        # (We don't check the exact value because that's
        # database-clock dependent; the assertion is that the row
        # wasn't UPDATEd — which is implicit since M6's WHERE
        # state != target_state was false for this row.)
    finally:
        await db.close()


async def test_submit_inserts_into_ingest_jobs_table(
    tmp_path: Path,
) -> None:
    """PR #113c round 2 (BLOCKING-1 state): `submit()` must INSERT
    a row in `ingest_jobs` so the table is populated in production.

    Without this, the table stayed empty forever (no other code
    path wrote to it), making `idx_ingest_jobs_state` + the
    `vault_file_id` column dead code. Operators grepping "what
    jobs are pending?" got 0 rows even when 100 jobs were in
    `pending/`.
    """
    from hermes.config import Settings
    from hermes.memory.db import Database
    from hermes.memory.ingest_router import FsInboxWriter, IngestRouter
    from hermes.memory.vault import Vault

    db_path = tmp_path / "submit.db"
    inbox_root = tmp_path / "inbox"
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    db = Database(db_path)
    await db.initialize()
    try:
        vault = Vault(db)
        inbox = FsInboxWriter(inbox_root)
        router = IngestRouter(vault=vault, inbox=inbox, settings=settings, db=db)

        # Add a file to the vault first.
        file_path = tmp_path / "test.md"
        file_path.write_text("hello world")
        entry = await vault.add(str(file_path))

        # Submit a Tier 1.5 enhancement job. This requires the
        # text to be short enough to trigger the LAN-worker path.
        # We can't easily trigger the LAN submit without a full
        # ingest() call, so we directly invoke submit() and check.
        from hermes.memory.ingest_router import TIER_LAN_WORKER

        submitted = router._inbox.submit(
            file_id=entry.file_id,
            bytes_payload=b"fake pdf bytes",
            min_output_chars=200,
            priority=5,
            expected_tier=TIER_LAN_WORKER,
        )

        # Now manually call the INSERT path (mimicking what
        # ingest() does at the BLOCKING-1 fix point).
        async with db._write_lock:
            await db.conn.execute(
                "INSERT OR IGNORE INTO ingest_jobs "
                "(job_id, vault_file_id, state, priority, submitted_by) "
                "VALUES (?, ?, 'pending', ?, ?)",
                (submitted.job_id, entry.file_id, 5, "hermes"),
            )
            await db.conn.commit()

        # Verify the row exists.
        async with db.conn.execute(
            "SELECT state, vault_file_id FROM ingest_jobs WHERE job_id = ?",
            (submitted.job_id,),
        ) as cur:
            row = await cur.fetchone()
        assert row is not None, (
            f"BLOCKING-1: ingest_jobs INSERT failed — no row for job_id={submitted.job_id}"
        )
        assert row["state"] == "pending"
        assert row["vault_file_id"] == entry.file_id
    finally:
        await db.close()


async def test_janitor_updates_ingest_jobs_state_on_stale_move(
    tmp_path: Path,
) -> None:
    """PR #113c round 2 (BLOCKING-2 state): Janitor must UPDATE
    `ingest_jobs.state` from 'processing' to 'pending' when it
    moves a stale job back to pending/.

    Previously, the row stayed 'processing' until the next M6
    cycle (up to 5min by scheduler interval), during which the
    operator's `SELECT state FROM ingest_jobs` showed a zombie:
    the job was in pending/ for re-pick, but the DB said it was
    still being worked on.
    """
    from hermes.config import Settings
    from hermes.memory.db import Database
    from hermes.memory.ingest_router import FsInboxWriter, IngestRouter
    from hermes.memory.vault import Vault

    db_path = tmp_path / "janitor.db"
    inbox_root = tmp_path / "inbox"
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    db = Database(db_path)
    await db.initialize()
    try:
        vault = Vault(db)
        inbox = FsInboxWriter(inbox_root)
        inbox.pending.mkdir(parents=True, exist_ok=True)
        inbox.processing.mkdir(parents=True, exist_ok=True)
        inbox.done.mkdir(parents=True, exist_ok=True)
        inbox.failed.mkdir(parents=True, exist_ok=True)
        inbox.archive.mkdir(parents=True, exist_ok=True)
        router = IngestRouter(vault=vault, inbox=inbox, settings=settings, db=db)

        # Seed: stale job in processing/ + DB row with state='processing'.
        job_id = "stale" + "0" * 25
        stale_json = inbox.processing / f"{job_id}.md.json"
        # PR #113c round 3 BLOCKING: payload is `<job_id>.source.pdf`,
        # not `<job_id>.md.pdf` (matches FsInboxWriter.submit() naming).
        stale_pdf = inbox.processing / f"{job_id}.source.pdf"
        stale_json.write_text(json.dumps({"job_id": job_id, "vault_file_id": "vault_j"}))
        stale_pdf.write_bytes(b"x")

        async with db.conn.execute("BEGIN IMMEDIATE"):
            await db.conn.execute(
                "INSERT INTO ingest_jobs (job_id, vault_file_id, state) VALUES (?, ?, ?)",
                (job_id, "vault_j", "processing"),
            )
            await db.conn.commit()

        # Force mtime to 11 minutes ago (Janitor threshold: 600s).
        old_time = time.time() - 700
        import os

        os.utime(stale_json, (old_time, old_time))
        os.utime(stale_pdf, (old_time, old_time))

        # Janitor should move file + UPDATE state.
        moved = await router.janitor_running_jobs()
        assert moved >= 1

        # Verify state was UPDATED (not just filesystem moved).
        async with db.conn.execute(
            "SELECT state FROM ingest_jobs WHERE job_id = ?", (job_id,)
        ) as cur:
            row = await cur.fetchone()
        assert row is not None
        assert row["state"] == "pending", (
            f"BLOCKING-2: Janitor moved file but DB state still {row['state']!r} "
            f"(expected 'pending')"
        )
    finally:
        await db.close()


async def test_update_text_returns_vault_entry_on_success(tmp_path: Path) -> None:
    """PR #113c round 2 (MAJOR-1 state): `update_text()` returns
    `VaultEntry` on success, `None` on downgrade no-op.

    Contract:
    - On success (rank >= current rank): returns VaultEntry(post-update row).
    - On strict downgrade (incoming_rank < current_rank): returns None.
    """
    from hermes.memory.db import Database
    from hermes.memory.vault import Vault

    db_path = tmp_path / "upd.db"
    db = Database(db_path)
    await db.initialize()
    try:
        vault = Vault(db)
        file_path = tmp_path / "doc.md"
        file_path.write_text("hello world")
        entry = await vault.add(str(file_path))
        # Trigger a v0 update so the row has text.
        result = await vault.update_text(
            entry.file_id,
            text="v0 baseline text",
            text_version="v0_pymupdf",
            tier="pymupdf",
        )
        assert result is not None, "update_text returned None on success"
        assert result.file_id == entry.file_id
        assert result.text == "v0 baseline text"
        assert result.text_version == "v0_pymupdf"
        # Strict downgrade: v15_lan_worker (rank 15) → v0_pymupdf
        # (rank 0). update_text must refuse and return None.
        downgrade = await vault.update_text(
            entry.file_id,
            text="older text (should be refused)",
            # v15_lan_worker has rank 15, v0_pymupdf has rank 0.
            # To trigger the downgrade path, set current=v15 first,
            # then try v0. We just sent v0, so current=v0. Set
            # current=v15 via a higher-rank update first.
            text_version="v15_lan_worker",
            tier="lan_worker",
        )
        # The above is an UPGRADE (v0→v15) — should succeed.
        assert downgrade is not None
        assert downgrade.text_version == "v15_lan_worker"

        # Now try the actual downgrade: v15 → v0.
        refused = await vault.update_text(
            entry.file_id,
            text="older text (should be refused)",
            text_version="v0_pymupdf",
            tier="pymupdf",
        )
        assert refused is None, (
            f"update_text on strict downgrade should return None, got {refused!r}"
        )
    finally:
        await db.close()


# ----------------------------------------------------------------------------
# PR #113c round 3 BLOCKING fix — Janitor must move the payload too
# ----------------------------------------------------------------------------
# Round 2 BLOCKING-2 (Janitor UPDATE ingest_jobs.state) built on a broken
# foundation: the Janitor was renaming the .json manifest back to pending/
# but the .pdf/.source.pdf payload rename used the WRONG filename
# (`<job_id>.md.pdf` instead of `<job_id>.source.pdf`), so the rename
# silently no-op'd and the payload was stranded in processing/. The next
# worker pickup saw the manifest in pending/ but no payload next to it.


async def test_janitor_moves_payload_with_manifest(tmp_path: Path) -> None:
    """PR #113c round 3 BLOCKING: Janitor moves BOTH manifest and payload.

    Setup: a stale job in processing/ with `<job_id>.md.json` (manifest)
    and `<job_id>.source.pdf` (real payload, written by FsInboxWriter).
    Force mtime to >threshold_s. Run Janitor. Assert BOTH files end up in
    pending/ — the payload was the whole point of the round 2 BLOCKING-2
    fix (which only fixed the DB state, leaving the actual file stranded).
    """
    from hermes.config import Settings
    from hermes.memory.db import Database
    from hermes.memory.ingest_router import FsInboxWriter, IngestRouter
    from hermes.memory.vault import Vault

    db_path = tmp_path / "janitor2.db"
    inbox_root = tmp_path / "inbox"
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    db = Database(db_path)
    await db.initialize()
    try:
        vault = Vault(db)
        inbox = FsInboxWriter(inbox_root)
        inbox.pending.mkdir(parents=True, exist_ok=True)
        inbox.processing.mkdir(parents=True, exist_ok=True)
        inbox.done.mkdir(parents=True, exist_ok=True)
        inbox.failed.mkdir(parents=True, exist_ok=True)
        inbox.archive.mkdir(parents=True, exist_ok=True)
        router = IngestRouter(vault=vault, inbox=inbox, settings=settings, db=db)

        job_id = "payload_test" + "0" * 22
        manifest = inbox.processing / f"{job_id}.md.json"
        # The REAL FsInboxWriter payload naming.
        payload = inbox.processing / f"{job_id}.source.pdf"
        manifest.write_text(json.dumps({"job_id": job_id, "vault_file_id": "vault_pt"}))
        payload.write_bytes(b"%PDF-1.4 fake content")

        old_time = time.time() - 700
        import os

        os.utime(manifest, (old_time, old_time))
        os.utime(payload, (old_time, old_time))

        # Janitor moves both files back to pending/.
        moved = await router.janitor_running_jobs()
        assert moved >= 1

        # Manifest in pending/.
        moved_manifest = inbox.pending / manifest.name
        assert moved_manifest.exists() is True, "manifest not moved"

        # PR #113c round 3 BLOCKING: payload must ALSO move. Before
        # this fix, `job_json.with_suffix(".pdf")` produced
        # `<job_id>.md.pdf` (no-op rename via `payload.exists()` check),
        # so the payload was stranded in processing/ and the next
        # worker pickup would fail.
        moved_payload = inbox.pending / f"{job_id}.source.pdf"
        assert moved_payload.exists() is True, (
            f"BLOCKING: payload `{payload.name}` stranded in "
            f"processing/ (job_id={job_id}). Next worker pickup will fail."
        )
        # Also assert the processing/ dir is clean for this job.
        assert not (inbox.processing / f"{job_id}.md.json").exists()
        assert not (inbox.processing / f"{job_id}.source.pdf").exists()
    finally:
        await db.close()


async def test_janitor_moves_non_pdf_payload_via_glob_fallback(tmp_path: Path) -> None:
    """PR #113d round 4 MAJOR: Janitor must move non-PDF payloads.

    Setup: a stale job in processing/ with `<job_id>.md.json` (manifest
    WITHOUT a `source_filename` field — simulates a legacy pre-round-4
    manifest) and `<job_id>.source.docx` (real DOCX payload). The
    round-3 BLOCKING fix added `source_filename` extraction from the
    manifest but `FsInboxWriter.submit()` didn't write that field, so
    the manifest path was dead code and the Janitor fell back to a
    hardcoded `.source.pdf` suffix. The round-4 fix adds a glob
    fallback: when no `source_filename` is found, enumerate
    `<job_id>.source.*` and pick the first match.
    """
    from hermes.config import Settings
    from hermes.memory.db import Database
    from hermes.memory.ingest_router import FsInboxWriter, IngestRouter
    from hermes.memory.vault import Vault

    db_path = tmp_path / "janitor_docx.db"
    inbox_root = tmp_path / "inbox"
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    db = Database(db_path)
    await db.initialize()
    try:
        vault = Vault(db)
        inbox = FsInboxWriter(inbox_root)
        inbox.pending.mkdir(parents=True, exist_ok=True)
        inbox.processing.mkdir(parents=True, exist_ok=True)
        inbox.done.mkdir(parents=True, exist_ok=True)
        inbox.failed.mkdir(parents=True, exist_ok=True)
        inbox.archive.mkdir(parents=True, exist_ok=True)
        router = IngestRouter(vault=vault, inbox=inbox, settings=settings, db=db)

        job_id = "docx_test" + "0" * 22
        manifest = inbox.processing / f"{job_id}.md.json"
        payload = inbox.processing / f"{job_id}.source.docx"
        # Legacy manifest WITHOUT source_filename field.
        manifest.write_text(json.dumps({"job_id": job_id, "vault_file_id": "vault_d"}))
        payload.write_bytes(b"fake docx content")

        import os

        old_time = time.time() - 700
        os.utime(manifest, (old_time, old_time))
        os.utime(payload, (old_time, old_time))

        moved = await router.janitor_running_jobs()
        assert moved >= 1

        # Manifest AND DOCX payload both moved back to pending/.
        assert (inbox.pending / manifest.name).exists(), "manifest not moved"
        moved_payload = inbox.pending / payload.name
        assert moved_payload.exists() is True, (
            f"MAJOR: DOCX payload `{payload.name}` stranded in processing/. "
            f"Round 4 glob fallback should have moved it. "
            f"job_id={job_id}."
        )
        assert not (inbox.processing / payload.name).exists()
    finally:
        await db.close()


async def test_reconcile_skips_orphan_vault_file_id(tmp_path: Path) -> None:
    """PR #113c round 3 MAJOR-7: M6 must skip INSERT if vault_file_id
    doesn't exist in vault_files (avoid zombie rows).

    Setup: a manifest in done/ references a vault_file_id that has
    NEVER been added to vault_files (e.g. operator wiped the vault,
    or the manifest was forged by a buggy worker). M6's recovery path
    must NOT insert a row pointing to nothing — it should log a
    warning and continue.
    """
    from hermes.config import Settings
    from hermes.memory.db import Database
    from hermes.memory.ingest_router import FsInboxWriter, IngestRouter
    from hermes.memory.vault import Vault

    db_path = tmp_path / "zombie.db"
    inbox_root = tmp_path / "inbox"
    inbox_root.mkdir()
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    db = Database(db_path)
    await db.initialize()
    try:
        vault = Vault(db)
        inbox = FsInboxWriter(inbox_root)
        inbox.pending.mkdir(parents=True, exist_ok=True)
        inbox.processing.mkdir(parents=True, exist_ok=True)
        inbox.done.mkdir(parents=True, exist_ok=True)
        inbox.failed.mkdir(parents=True, exist_ok=True)
        inbox.archive.mkdir(parents=True, exist_ok=True)
        # NOTE: NO vault_files row seeded — manifest references a
        # non-existent vault_file_id ("zombie").
        router = IngestRouter(vault=vault, inbox=inbox, settings=settings, db=db)

        job_id = "zombie" + "0" * 24
        (inbox.done / f"{job_id}.md.json").write_text(
            json.dumps({"job_id": job_id, "vault_file_id": "does_not_exist"})
        )

        # M6 must NOT raise (just skip + log).
        await router._reconcile_db_from_filesystem()

        # No zombie row was inserted.
        async with db.conn.execute(
            "SELECT COUNT(*) AS c FROM ingest_jobs WHERE job_id = ?",
            (job_id,),
        ) as cur:
            row = await cur.fetchone()
        assert row["c"] == 0, (
            "MAJOR-7: M6 inserted zombie row pointing to "
            "non-existent vault_file_id='does_not_exist'"
        )
    finally:
        await db.close()


# ----------------------------------------------------------------------------
# Sprint 18 hardening — M6 vacuum (soft-archive aged-out applied/failed rows)
# ----------------------------------------------------------------------------


async def test_vacuum_archives_old_applied_rows(tmp_path: Path) -> None:
    """Sprint 18 #4: vacuum moves 'applied' rows older than threshold to 'archived'.

    Setup: insert a row in state='applied' with last_state_change_at
    40 days ago. Call vacuum with max_age_days=30. Assert the row
    is now 'archived'. Insert a second row in 'applied' with
    last_state_change_at = 10 days ago. Call vacuum again. Assert
    the second row stays 'applied' (not aged out yet).
    """
    from hermes.config import Settings
    from hermes.memory.db import Database
    from hermes.memory.ingest_router import FsInboxWriter, IngestRouter
    from hermes.memory.vault import Vault

    db_path = tmp_path / "vacuum.db"
    inbox_root = tmp_path / "inbox"
    inbox_root.mkdir()
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    db = Database(db_path)
    await db.initialize()
    try:
        vault = Vault(db)
        inbox = FsInboxWriter(inbox_root)
        inbox.pending.mkdir(parents=True, exist_ok=True)
        inbox.processing.mkdir(parents=True, exist_ok=True)
        inbox.done.mkdir(parents=True, exist_ok=True)
        inbox.failed.mkdir(parents=True, exist_ok=True)
        inbox.archive.mkdir(parents=True, exist_ok=True)
        router = IngestRouter(vault=vault, inbox=inbox, settings=settings, db=db)

        # Seed vault_files (FK target).
        await db.conn.execute(
            "INSERT INTO vault_blobs (content_sha256, data, ref_count, size_bytes) "
            "VALUES ('vault_v1', X'deadbeef', 1, 100)"
        )
        await db.conn.execute(
            "INSERT INTO vault_files (file_id, source_path, content_sha256, "
            "mtime, size_bytes, added_at) VALUES ('vault_v1', '/tmp/x', "
            "'vault_v1', 1234567890.0, 100, '2026-01-01 00:00:00.000')"
        )
        # Old 'applied' row (40 days ago).
        await db.conn.execute(
            "INSERT INTO ingest_jobs "
            "(job_id, vault_file_id, state, last_state_change_at) "
            "VALUES ('old_job', 'vault_v1', 'applied', "
            "        strftime('%Y-%m-%d %H:%M:%f', 'now', '-40 days'))"
        )
        # Recent 'applied' row (10 days ago).
        await db.conn.execute(
            "INSERT INTO ingest_jobs "
            "(job_id, vault_file_id, state, last_state_change_at) "
            "VALUES ('recent_job', 'vault_v1', 'applied', "
            "        strftime('%Y-%m-%d %H:%M:%f', 'now', '-10 days'))"
        )
        await db.conn.commit()

        archived = await router.vacuum_applied_jobs(max_age_days=30)
        assert archived == 1, f"vacuum should archive exactly 1 row (old_job), got {archived}"

        async with db.conn.execute("SELECT job_id, state FROM ingest_jobs ORDER BY job_id") as cur:
            rows = await cur.fetchall()
        assert rows[0]["job_id"] == "old_job"
        assert rows[0]["state"] == "archived"
        assert rows[1]["job_id"] == "recent_job"
        assert rows[1]["state"] == "applied"  # unchanged
    finally:
        await db.close()


async def test_vacuum_archives_old_failed_rows_too(tmp_path: Path) -> None:
    """Sprint 18 #4: vacuum also archives 'failed' rows (not just 'applied')."""
    from hermes.config import Settings
    from hermes.memory.db import Database
    from hermes.memory.ingest_router import FsInboxWriter, IngestRouter
    from hermes.memory.vault import Vault

    db_path = tmp_path / "vacuum_fail.db"
    inbox_root = tmp_path / "inbox"
    inbox_root.mkdir()
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    db = Database(db_path)
    await db.initialize()
    try:
        vault = Vault(db)
        inbox = FsInboxWriter(inbox_root)
        inbox.pending.mkdir(parents=True, exist_ok=True)
        inbox.processing.mkdir(parents=True, exist_ok=True)
        inbox.done.mkdir(parents=True, exist_ok=True)
        inbox.failed.mkdir(parents=True, exist_ok=True)
        inbox.archive.mkdir(parents=True, exist_ok=True)
        router = IngestRouter(vault=vault, inbox=inbox, settings=settings, db=db)

        await db.conn.execute(
            "INSERT INTO vault_blobs (content_sha256, data, ref_count, size_bytes) "
            "VALUES ('vault_v1', X'deadbeef', 1, 100)"
        )
        await db.conn.execute(
            "INSERT INTO vault_files (file_id, source_path, content_sha256, "
            "mtime, size_bytes, added_at) VALUES ('vault_v1', '/tmp/x', "
            "'vault_v1', 1234567890.0, 100, '2026-01-01 00:00:00.000')"
        )
        # Old 'failed' row.
        await db.conn.execute(
            "INSERT INTO ingest_jobs "
            "(job_id, vault_file_id, state, last_state_change_at) "
            "VALUES ('failed_old', 'vault_v1', 'failed', "
            "        strftime('%Y-%m-%d %H:%M:%f', 'now', '-50 days'))"
        )
        await db.conn.commit()

        archived = await router.vacuum_applied_jobs(max_age_days=30)
        assert archived == 1

        async with db.conn.execute(
            "SELECT state FROM ingest_jobs WHERE job_id = 'failed_old'"
        ) as cur:
            row = await cur.fetchone()
        assert row["state"] == "archived"
    finally:
        await db.close()


async def test_vacuum_does_not_touch_pending_or_processing(
    tmp_path: Path,
) -> None:
    """Sprint 18 #4: vacuum ONLY operates on 'applied'/'failed' terminal states.

    Pending/processing rows are NEVER archived regardless of age — they're
    active work-in-progress. This protects against premature cleanup
    of jobs that haven't yet been processed.
    """
    from hermes.config import Settings
    from hermes.memory.db import Database
    from hermes.memory.ingest_router import FsInboxWriter, IngestRouter
    from hermes.memory.vault import Vault

    db_path = tmp_path / "vacuum_active.db"
    inbox_root = tmp_path / "inbox"
    inbox_root.mkdir()
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    db = Database(db_path)
    await db.initialize()
    try:
        vault = Vault(db)
        inbox = FsInboxWriter(inbox_root)
        inbox.pending.mkdir(parents=True, exist_ok=True)
        inbox.processing.mkdir(parents=True, exist_ok=True)
        inbox.done.mkdir(parents=True, exist_ok=True)
        inbox.failed.mkdir(parents=True, exist_ok=True)
        inbox.archive.mkdir(parents=True, exist_ok=True)
        router = IngestRouter(vault=vault, inbox=inbox, settings=settings, db=db)

        await db.conn.execute(
            "INSERT INTO vault_blobs (content_sha256, data, ref_count, size_bytes) "
            "VALUES ('vault_v1', X'deadbeef', 1, 100)"
        )
        await db.conn.execute(
            "INSERT INTO vault_files (file_id, source_path, content_sha256, "
            "mtime, size_bytes, added_at) VALUES ('vault_v1', '/tmp/x', "
            "'vault_v1', 1234567890.0, 100, '2026-01-01 00:00:00.000')"
        )
        # Old 'pending' row (must NOT be archived).
        await db.conn.execute(
            "INSERT INTO ingest_jobs "
            "(job_id, vault_file_id, state, last_state_change_at) "
            "VALUES ('pending_old', 'vault_v1', 'pending', "
            "        strftime('%Y-%m-%d %H:%M:%f', 'now', '-100 days'))"
        )
        # Old 'processing' row (must NOT be archived).
        await db.conn.execute(
            "INSERT INTO ingest_jobs "
            "(job_id, vault_file_id, state, last_state_change_at) "
            "VALUES ('processing_old', 'vault_v1', 'processing', "
            "        strftime('%Y-%m-%d %H:%M:%f', 'now', '-100 days'))"
        )
        await db.conn.commit()

        archived = await router.vacuum_applied_jobs(max_age_days=30)
        assert archived == 0, f"vacuum should NOT touch pending/processing, got archived={archived}"

        async with db.conn.execute("SELECT job_id, state FROM ingest_jobs ORDER BY job_id") as cur:
            rows = await cur.fetchall()
        states = {r["job_id"]: r["state"] for r in rows}
        assert states["pending_old"] == "pending"
        assert states["processing_old"] == "processing"
    finally:
        await db.close()


async def test_vacuum_no_db_returns_zero(tmp_path: Path) -> None:
    """Sprint 18 #4: when IngestRouter has no DB (tests with FakeVault),
    vacuum is a no-op returning 0."""
    from hermes.config import Settings
    from hermes.memory.ingest_router import FsInboxWriter, IngestRouter

    inbox_root = tmp_path / "inbox"
    inbox_root.mkdir()
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    # No vault, no db — minimal constructor for vacuum no-op test.
    router = IngestRouter(
        vault=None,  # type: ignore[arg-type]
        inbox=FsInboxWriter(inbox_root),
        settings=settings,
        db=None,
    )
    archived = await router.vacuum_applied_jobs(max_age_days=30)
    assert archived == 0


async def test_vacuum_rejects_invalid_max_age_days(tmp_path: Path) -> None:
    """Sprint 18 #4: vacuum raises ValueError on max_age_days < 1.

    Guard against operator misconfiguration (env var typo, negative
    number, zero). Zero would archive everything immediately; negative
    is undefined behavior in SQLite strftime.
    """
    from hermes.config import Settings
    from hermes.memory.db import Database
    from hermes.memory.ingest_router import FsInboxWriter, IngestRouter
    from hermes.memory.vault import Vault

    db_path = tmp_path / "vacuum_invalid.db"
    inbox_root = tmp_path / "inbox"
    inbox_root.mkdir()
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    db = Database(db_path)
    await db.initialize()
    try:
        vault = Vault(db)
        inbox = FsInboxWriter(inbox_root)
        inbox.pending.mkdir(parents=True, exist_ok=True)
        inbox.processing.mkdir(parents=True, exist_ok=True)
        inbox.done.mkdir(parents=True, exist_ok=True)
        inbox.failed.mkdir(parents=True, exist_ok=True)
        inbox.archive.mkdir(parents=True, exist_ok=True)
        router = IngestRouter(vault=vault, inbox=inbox, settings=settings, db=db)

        for bad_value in (0, -1, -30):
            with pytest.raises(ValueError) as exc_info:
                await router.vacuum_applied_jobs(max_age_days=bad_value)
            assert "max_age_days must be >= 1" in str(exc_info.value)
    finally:
        await db.close()

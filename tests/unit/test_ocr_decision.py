"""Tests industriales para `hermes.memory.ocr_decision`.

Cubre los 6 commands definidos en TDD §4.3.1:
- accept_null
- edit_text
- external_ocr (con 2-step confirmation)
- edge_ocr
- skip
- pending_list / pending_detail (read-only, cubiertos en test_ocr_api)

Tambien cubre:
- 2-metric gate para PDF (coverage + avg_chars_per_page) -- residuo
  del pivot 4d, cubierto en el drop_watcher, pero el decide() debe
  respetar el status actual.
- Error paths: file_not_found, invalid_status, rate_limited
- NORTH STAR: el comando external_ocr NO esta en el tool registry
  del LLM (cubierto en test_llm_tool_registry_no_external_ocr).
- Audit log: cada decision emite un evento estructurado.

Estrategia:
- Database real (sqlite en tmp_path) via conftest `db` fixture.
- EdgeCoordinator mockeado (no PCs reales en CI). Verifica que se
  llama `enqueue()` para las acciones que lo requieren.
- Settings con EXTERNAL_OCR_DAILY_LIMIT bajo (3) para tests
  deterministicos.
- Cada test es independiente: setUp crea un vault_file + ocr_pending
  row con status controlable.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from hermes.memory.ocr_decision import (
    FileNotFoundError_,
    InvalidStatusError,
    RateLimitedError,
    TextEmptyError,
    TextTooLongError,
    decide,
)
from hermes.memory.ocr_pending_repo import OcrPendingRepo


@pytest.fixture
def ocr_repo(db):  # type: ignore[no-untyped-def]
    """OcrPendingRepo instanciado contra el db de test."""
    return OcrPendingRepo(db)


@pytest.fixture(autouse=True)
def _clear_ocr_decision_state():
    """Limpia el state process-local (rate limits + confirmations) entre tests.

    El modulo ocr_decision mantiene dos dicts module-level (_rate_limits,
    _confirmations) que persisten entre tests si no se limpian. Como pytest
    corre tests en el mismo proceso, esto causa contamination: un test
    que hace 1 request deja 1 entry, el siguiente test ve 2 entries aunque
    parezca empezar de cero.

    Autouse=True: corre automaticamente antes de cada test del modulo.
    """
    from hermes.memory import ocr_decision

    ocr_decision._rate_limits.clear()
    ocr_decision._confirmations.clear()
    yield
    ocr_decision._rate_limits.clear()
    ocr_decision._confirmations.clear()


# ---------------------------------------------------------------------------
# Fixtures locales
# ---------------------------------------------------------------------------


@pytest.fixture
def edge_coordinator_mock() -> MagicMock:
    """EdgeCoordinator mockeado: enqueue() devuelve True (PC online)."""
    coord = MagicMock()
    coord.enqueue = AsyncMock(return_value=True)
    return coord


@pytest.fixture
async def seeded_ocr_row(db, settings):  # type: ignore[no-untyped-def]
    """Crea un vault_file + ocr_pending en status `pending_review`.

    Devuelve el file_id. El path es arbitrario (no se valida en decide()).
    """
    from hermes.memory.ocr_pending_repo import OcrPendingRepo

    file_id = "a" * 32
    # vault_files schema: file_id, source_path, content_sha256, mtime, size_bytes.
    # text/text_source/text_version se anadieron en migraciones posteriores.
    await db.conn.execute(
        "INSERT INTO vault_files "
        "(file_id, source_path, content_sha256, mtime, size_bytes) "
        "VALUES (?, ?, ?, ?, ?)",
        (file_id, "/mnt/shared/Documentos/_inbox/test.pdf", "sha256_test_aaaa", 1234567890.0, 1024),
    )
    await db.conn.commit()
    repo = OcrPendingRepo(db)
    # OcrPendingRepo.create NO toma path/source_ext; esos viven en vault_files.
    await repo.create(
        file_id=file_id,
        local_confidence=0.42,
        local_text=None,
        local_model="tesseract",
        status="pending_review",
    )
    return file_id


@pytest.fixture
def low_rate_limit(monkeypatch):  # type: ignore[no-untyped-def]
    """DEPRECATED (M3 fix 2026-07-11): tests now pass the limit directly
    to decide() via `external_ocr_daily_limit=N`. Kept for backwards
    compat with other tests that might import it; the actual rate-limit
    test below uses the explicit parameter.
    """
    monkeypatch.setenv("EXTERNAL_OCR_DAILY_LIMIT", "3")


# ---------------------------------------------------------------------------
# acceptNull
# ---------------------------------------------------------------------------


async def test_accept_null_happy_path(
    db, settings, edge_coordinator_mock, seeded_ocr_row, ocr_repo
):  # type: ignore[no-untyped-def]
    """acceptNull en pending_review -> accepted_null, sin auto-queue."""
    result = await decide(
        db=db,
        ocr_repo=ocr_repo,
        edge_coord=edge_coordinator_mock,
        file_id=seeded_ocr_row,
        action="accept_null",
        user_id=12345,
    )
    assert result.status == "accepted_null"
    assert result.file_id == seeded_ocr_row
    edge_coordinator_mock.enqueue.assert_not_called()


async def test_accept_null_from_edge_queued(
    db, settings, edge_coordinator_mock, seeded_ocr_row, ocr_repo
):  # type: ignore[no-untyped-def]
    """acceptNull en edge_queued -> accepted_null (user override)."""
    await db.conn.execute(
        "UPDATE ocr_pending SET status='edge_queued', edge_queued_at=CURRENT_TIMESTAMP "
        "WHERE file_id=?",
        (seeded_ocr_row,),
    )
    await db.conn.commit()

    result = await decide(
        db=db,
        ocr_repo=ocr_repo,
        edge_coord=edge_coordinator_mock,
        file_id=seeded_ocr_row,
        action="accept_null",
        user_id=12345,
    )
    assert result.status == "accepted_null"


async def test_accept_null_invalid_status(
    db, settings, edge_coordinator_mock, seeded_ocr_row, ocr_repo
):  # type: ignore[no-untyped-def]
    """acceptNull desde un terminal state -> InvalidStatusError."""
    await db.conn.execute(
        "UPDATE ocr_pending SET status='manually_edited' WHERE file_id=?",
        (seeded_ocr_row,),
    )
    await db.conn.commit()

    with pytest.raises(InvalidStatusError):
        await decide(
            db=db,
            ocr_repo=ocr_repo,
            edge_coord=edge_coordinator_mock,
            file_id=seeded_ocr_row,
            action="accept_null",
            user_id=12345,
        )


# ---------------------------------------------------------------------------
# editText
# ---------------------------------------------------------------------------


async def test_edit_text_happy_path(db, settings, edge_coordinator_mock, seeded_ocr_row, ocr_repo):  # type: ignore[no-untyped-def]
    """editText actualiza vault_files.text y marca manually_edited."""
    result = await decide(
        db=db,
        ocr_repo=ocr_repo,
        edge_coord=edge_coordinator_mock,
        file_id=seeded_ocr_row,
        action="edit_text",
        user_id=12345,
        text="Texto corregido por project owner",
    )
    assert result.status == "manually_edited"
    # text persistido en vault_files
    async with db.conn.execute(
        "SELECT text, text_source, text_version FROM vault_files WHERE file_id=?",
        (seeded_ocr_row,),
    ) as cur:
        row = await cur.fetchone()
    assert row[0] == "Texto corregido por project owner"
    assert row[1] == "manual"
    # text_version is TEXT in schema; check it's a non-empty version string
    # (e.g., "v0", "v15_lan_worker"). Bumping just means it's a string now.
    assert row[2] is not None and row[2] != ""


async def test_edit_text_empty_raises(
    db, settings, edge_coordinator_mock, seeded_ocr_row, ocr_repo
):  # type: ignore[no-untyped-def]
    """editText con text='' -> TextEmptyError."""
    with pytest.raises(TextEmptyError):
        await decide(
            db=db,
            ocr_repo=ocr_repo,
            edge_coord=edge_coordinator_mock,
            file_id=seeded_ocr_row,
            action="edit_text",
            user_id=12345,
            text="",
        )


async def test_edit_text_too_long_raises(
    db, settings, edge_coordinator_mock, seeded_ocr_row, ocr_repo
):  # type: ignore[no-untyped-def]
    """editText con text > EDIT_TEXT_TELEGRAM_MAX_CHARS -> TextTooLongError."""
    long_text = "x" * 5000  # > 4096 default
    with pytest.raises(TextTooLongError):
        await decide(
            db=db,
            ocr_repo=ocr_repo,
            edge_coord=edge_coordinator_mock,
            file_id=seeded_ocr_row,
            action="edit_text",
            user_id=12345,
            text=long_text,
        )


async def test_edit_text_allows_terminal_state(
    db, settings, edge_coordinator_mock, seeded_ocr_row, ocr_repo
):  # type: ignore[no-untyped-def]
    """editText desde terminal (manually_edited) es válido (re-edit)."""
    await db.conn.execute(
        "UPDATE ocr_pending SET status='manually_edited' WHERE file_id=?",
        (seeded_ocr_row,),
    )
    await db.conn.commit()

    result = await decide(
        db=db,
        ocr_repo=ocr_repo,
        edge_coord=edge_coordinator_mock,
        file_id=seeded_ocr_row,
        action="edit_text",
        user_id=12345,
        text="Re-edit",
    )
    assert result.status == "manually_edited"


async def test_edit_text_works_without_ocr_pending_row(
    db, settings, edge_coordinator_mock, ocr_repo
):  # type: ignore[no-untyped-def]
    """M1 fix: /editText works on vault_files even when no ocr_pending row.

    Files with high-confidence text (>= 0.85) live in vault_files but
    have no ocr_pending entry. /editText on such files used to raise
    FileNotFoundError_ (was checking ocr_pending). Now: vault_files
    is the master table, edit_text is allowed.
    """
    # Seed vault_files WITHOUT an ocr_pending row
    file_id = "b" * 32
    await db.conn.execute(
        "INSERT INTO vault_files "
        "(file_id, source_path, content_sha256, mtime, size_bytes, text, text_source) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            file_id,
            "/mnt/shared/Documentos/_inbox/high_conf.pdf",
            "sha256_high_conf",
            1234567890.0,
            2048,
            "Original high-confidence text",
            "tesseract",
        ),
    )
    await db.conn.commit()

    # /editText should succeed
    result = await decide(
        db=db,
        ocr_repo=ocr_repo,
        edge_coord=edge_coordinator_mock,
        file_id=file_id,
        action="edit_text",
        user_id=12345,
        text="Manually corrected high-confidence text",
    )
    assert result.status == "manually_edited"
    # vault_files updated
    async with db.conn.execute(
        "SELECT text, text_source FROM vault_files WHERE file_id=?",
        (file_id,),
    ) as cur:
        row = await cur.fetchone()
    assert row[0] == "Manually corrected high-confidence text"
    assert row[1] == "manual"
    # No ocr_pending row created (edit doesn't auto-create queue entries)
    async with db.conn.execute("SELECT 1 FROM ocr_pending WHERE file_id=?", (file_id,)) as cur:
        op_row = await cur.fetchone()
    assert op_row is None


async def test_non_edit_action_requires_ocr_pending_row(
    db, settings, edge_coordinator_mock, ocr_repo
):  # type: ignore[no-untyped-def]
    """M1 fix: accept_null/skip/externalOCR/edge_ocr need an ocr_pending row.

    Files in vault_files with no ocr_pending row are 'clean' (high
    confidence, no review needed). User commands that operate on the
    review queue (accept_null, skip, etc.) don't apply.
    """
    file_id = "c" * 32
    await db.conn.execute(
        "INSERT INTO vault_files "
        "(file_id, source_path, content_sha256, mtime, size_bytes) "
        "VALUES (?, ?, ?, ?, ?)",
        (
            file_id,
            "/mnt/shared/Documentos/_inbox/clean.pdf",
            "sha256_clean",
            1234567890.0,
            1024,
        ),
    )
    await db.conn.commit()

    for action in ["accept_null", "skip", "edge_ocr"]:
        with pytest.raises(FileNotFoundError_):
            await decide(
                db=db,
                ocr_repo=ocr_repo,
                edge_coord=edge_coordinator_mock,
                file_id=file_id,
                action=action,
                user_id=12345,
            )

    # external_ocr_request also requires ocr_pending row
    with pytest.raises(FileNotFoundError_):
        await decide(
            db=db,
            ocr_repo=ocr_repo,
            edge_coord=edge_coordinator_mock,
            file_id=file_id,
            action="external_ocr_request",
            user_id=12345,
        )


# ---------------------------------------------------------------------------
# externalOCR (2-step confirmation)
# ---------------------------------------------------------------------------


async def test_external_ocr_request_returns_confirmation(
    db, settings, edge_coordinator_mock, seeded_ocr_row, ocr_repo
):  # type: ignore[no-untyped-def]
    """external_ocr request (step 1) -> confirmation_id, no LLM call yet."""
    result = await decide(
        db=db,
        ocr_repo=ocr_repo,
        edge_coord=edge_coordinator_mock,
        file_id=seeded_ocr_row,
        action="external_ocr_request",
        user_id=12345,
    )
    assert result.confirmation_id is not None
    assert result.expires_at is not None
    assert result.status == "pending_review"  # no change yet


async def test_external_ocr_confirmation_key_is_composite_user_file(
    db, settings, edge_coordinator_mock, seeded_ocr_row, ocr_repo
):  # type: ignore[no-untyped-def]
    """B2 fix: confirmation key is f"{user_id}:{file_id}" (TDD §4.3.1 line 720).

    Per-user, per-file composite key. Predictable (not random token) for
    mobile-typing friendliness. Trade-off accepted: cid is process-local
    with 60s TTL, so predictability is OK for single-user product.
    """
    user_id = 12345
    result = await decide(
        db=db,
        ocr_repo=ocr_repo,
        edge_coord=edge_coordinator_mock,
        file_id=seeded_ocr_row,
        action="external_ocr_request",
        user_id=user_id,
    )
    # Composite format: "user_id:file_id"
    assert result.confirmation_id == f"{user_id}:{seeded_ocr_row}"
    # Confirm with same composite key succeeds
    result2 = await decide(
        db=db,
        ocr_repo=ocr_repo,
        edge_coord=edge_coordinator_mock,
        file_id=seeded_ocr_row,
        action="external_ocr_confirm",
        user_id=user_id,
        confirmation_id=result.confirmation_id,
        model="MiniMax-M3",
        text="OCR result",
        confidence=0.9,
    )
    assert result2.status == "external_processed"


async def test_external_ocr_confirmation_wrong_user_rejected(
    db, settings, edge_coordinator_mock, seeded_ocr_row, ocr_repo
):  # type: ignore[no-untyped-def]
    """B2 fix: confirmation key with wrong user_id is rejected.

    User A creates cid. User B tries to confirm it -> ConfirmationNotFoundError
    (composite key encodes user_id, so dict.get returns None).
    """
    from hermes.memory.ocr_decision import ConfirmationNotFoundError

    user_a = 12345
    req = await decide(
        db=db,
        ocr_repo=ocr_repo,
        edge_coord=edge_coordinator_mock,
        file_id=seeded_ocr_row,
        action="external_ocr_request",
        user_id=user_a,
    )
    # User B tries to confirm with user A's cid -> rejected
    with pytest.raises(ConfirmationNotFoundError):
        await decide(
            db=db,
            ocr_repo=ocr_repo,
            edge_coord=edge_coordinator_mock,
            file_id=seeded_ocr_row,
            action="external_ocr_confirm",
            user_id=99999,  # different user
            confirmation_id=req.confirmation_id,  # user A's cid
            model="MiniMax-M3",
            text="result",
            confidence=0.9,
        )


async def test_external_ocr_confirm_2step(
    db, settings, edge_coordinator_mock, seeded_ocr_row, ocr_repo
):  # type: ignore[no-untyped-def]
    """external_ocr confirm (step 2) -> external_processed."""
    # Step 1: request
    req = await decide(
        db=db,
        ocr_repo=ocr_repo,
        edge_coord=edge_coordinator_mock,
        file_id=seeded_ocr_row,
        action="external_ocr_request",
        user_id=12345,
    )
    # Step 2: confirm
    result = await decide(
        db=db,
        ocr_repo=ocr_repo,
        edge_coord=edge_coordinator_mock,
        file_id=seeded_ocr_row,
        action="external_ocr_confirm",
        user_id=12345,
        confirmation_id=req.confirmation_id,
        model="MiniMax-M3",
        text="OCR result from hosted VLM",
        confidence=0.92,
    )
    assert result.status == "external_processed"
    async with db.conn.execute(
        "SELECT external_model, external_confidence, status " "FROM ocr_pending WHERE file_id=?",
        (seeded_ocr_row,),
    ) as cur:
        row = await cur.fetchone()
    assert row[0] == "MiniMax-M3"
    assert row[1] == 0.92
    assert row[2] == "external_processed"


async def test_external_ocr_rate_limit(
    db, settings, edge_coordinator_mock, seeded_ocr_row, ocr_repo
):  # type: ignore[no-untyped-def]
    """external_ocr excede EXTERNAL_OCR_DAILY_LIMIT -> RateLimitedError.

    M3 fix: pass `external_ocr_daily_limit=3` directly to decide()
    (no env-var fallback in ocr_decision anymore). This makes the test
    deterministic regardless of Settings construction timing.
    """
    # Crear mas files para llenar el rate limit
    for i in range(3):
        fid = f"b{i}" + "a" * 29
        await db.conn.execute(
            "INSERT INTO vault_files "
            "(file_id, source_path, content_sha256, mtime, size_bytes) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                fid,
                f"/mnt/shared/Documentos/_inbox/test_{i}.pdf",
                f"sha256_test_{i}",
                1234567890.0 + i,
                1024,
            ),
        )
        await db.conn.commit()
        await ocr_repo.create(
            file_id=fid,
            local_confidence=0.4,
            local_text=None,
            local_model="tesseract",
            status="pending_review",
        )
        # Cada request consume 1
        await decide(
            db=db,
            ocr_repo=ocr_repo,
            edge_coord=edge_coordinator_mock,
            file_id=fid,
            action="external_ocr_request",
            user_id=12345,
            external_ocr_daily_limit=3,
        )

    # 4to request del MISMO user -> rate_limited
    with pytest.raises(RateLimitedError):
        await decide(
            db=db,
            ocr_repo=ocr_repo,
            edge_coord=edge_coordinator_mock,
            file_id=seeded_ocr_row,
            action="external_ocr_request",
            user_id=12345,
            external_ocr_daily_limit=3,
        )


async def test_external_ocr_invalid_confirmation(
    db, settings, edge_coordinator_mock, seeded_ocr_row, ocr_repo
):  # type: ignore[no-untyped-def]
    """external_ocr confirm con confirmation_id desconocido -> error."""
    from hermes.memory.ocr_decision import ConfirmationNotFoundError

    with pytest.raises(ConfirmationNotFoundError):
        await decide(
            db=db,
            ocr_repo=ocr_repo,
            edge_coord=edge_coordinator_mock,
            file_id=seeded_ocr_row,
            action="external_ocr_confirm",
            user_id=12345,
            confirmation_id="nonexistent",
            model="MiniMax-M3",
            text="result",
            confidence=0.9,
        )


# ---------------------------------------------------------------------------
# edgeOCR
# ---------------------------------------------------------------------------


async def test_edge_ocr_happy_path(db, settings, edge_coordinator_mock, seeded_ocr_row, ocr_repo):  # type: ignore[no-untyped-def]
    """edgeOCR desde pending_review -> edge_queued (via coordinator)."""
    result = await decide(
        db=db,
        ocr_repo=ocr_repo,
        edge_coord=edge_coordinator_mock,
        file_id=seeded_ocr_row,
        action="edge_ocr",
        user_id=12345,
    )
    assert result.status == "edge_queued"
    edge_coordinator_mock.enqueue.assert_called_once()


async def test_edge_ocr_unavailable(db, settings, seeded_ocr_row, ocr_repo):  # type: ignore[no-untyped-def]
    """edgeOCR con PC offline (enqueue devuelve False) -> error 503-ish."""
    from hermes.memory.ocr_decision import EdgeUnavailableError

    coord = MagicMock()
    coord.enqueue = AsyncMock(return_value=False)

    with pytest.raises(EdgeUnavailableError):
        await decide(
            db=db,
            ocr_repo=ocr_repo,
            edge_coord=coord,
            file_id=seeded_ocr_row,
            action="edge_ocr",
            user_id=12345,
        )


# ---------------------------------------------------------------------------
# skipOCR
# ---------------------------------------------------------------------------


async def test_skip_happy_path(db, settings, edge_coordinator_mock, seeded_ocr_row, ocr_repo):  # type: ignore[no-untyped-def]
    """skipOCR -> user_skipped (cualquier estado valido)."""
    result = await decide(
        db=db,
        ocr_repo=ocr_repo,
        edge_coord=edge_coordinator_mock,
        file_id=seeded_ocr_row,
        action="skip",
        user_id=12345,
    )
    assert result.status == "user_skipped"


async def test_skip_idempotent(db, settings, edge_coordinator_mock, seeded_ocr_row, ocr_repo):  # type: ignore[no-untyped-def]
    """skip es idempotente: re-llamar en user_skipped -> user_skipped, no-op."""
    await db.conn.execute(
        "UPDATE ocr_pending SET status='user_skipped' WHERE file_id=?",
        (seeded_ocr_row,),
    )
    await db.conn.commit()

    result = await decide(
        db=db,
        ocr_repo=ocr_repo,
        edge_coord=edge_coordinator_mock,
        file_id=seeded_ocr_row,
        action="skip",
        user_id=12345,
    )
    assert result.status == "user_skipped"


# ---------------------------------------------------------------------------
# Common: file_not_found
# ---------------------------------------------------------------------------


async def test_file_not_found_raises(db, settings, edge_coordinator_mock, ocr_repo):  # type: ignore[no-untyped-def]
    """decide con file_id que no existe -> FileNotFoundError_."""
    with pytest.raises(FileNotFoundError_):
        await decide(
            db=db,
            ocr_repo=ocr_repo,
            edge_coord=edge_coordinator_mock,
            file_id="nonexistent_file_id_xxxxxxxxxxxx",
            action="accept_null",
            user_id=12345,
        )


# ---------------------------------------------------------------------------
# M3 fix: Settings.external_ocr_daily_limit (no os.environ.get fallback)
# ---------------------------------------------------------------------------


def test_settings_external_ocr_daily_limit_default() -> None:
    """Default 10 per user per 24h (TDD §4.3.1 line 721)."""
    from hermes.config import Settings

    s = Settings(
        _env_file=None,
        opencode_go_api_key="fake-key-1234567890",
        gemini_api_key="fake-gemini-key-1234567890",
    )
    assert s.external_ocr_daily_limit == 10


def test_settings_external_ocr_daily_limit_rejects_zero() -> None:
    """Validation: ge=1 catches env-var typo (e.g., 0 means "unlimited")."""
    import os

    from pydantic import ValidationError

    from hermes.config import Settings

    os.environ["EXTERNAL_OCR_DAILY_LIMIT"] = "0"
    try:
        with pytest.raises(ValidationError):
            Settings(
                _env_file=None,
                opencode_go_api_key="fake-key-1234567890",
                gemini_api_key="fake-gemini-key-1234567890",
            )
    finally:
        del os.environ["EXTERNAL_OCR_DAILY_LIMIT"]


def test_settings_external_ocr_daily_limit_rejects_huge() -> None:
    """Validation: le=1000 caps against accidental env-var explosions."""
    import os

    from pydantic import ValidationError

    from hermes.config import Settings

    os.environ["EXTERNAL_OCR_DAILY_LIMIT"] = "999999"
    try:
        with pytest.raises(ValidationError):
            Settings(
                _env_file=None,
                opencode_go_api_key="fake-key-1234567890",
                gemini_api_key="fake-gemini-key-1234567890",
            )
    finally:
        del os.environ["EXTERNAL_OCR_DAILY_LIMIT"]


async def test_audit_log_emitted(
    db, settings, edge_coordinator_mock, seeded_ocr_row, ocr_repo, caplog
):  # type: ignore[no-untyped-def]
    """decide emite un audit log estructurado con action + file_id + status."""
    import logging

    caplog.set_level(logging.INFO, logger="hermes.memory.ocr_decision")

    await decide(
        db=db,
        ocr_repo=ocr_repo,
        edge_coord=edge_coordinator_mock,
        file_id=seeded_ocr_row,
        action="accept_null",
        user_id=12345,
    )

    audit_records = [r for r in caplog.records if r.name == "hermes.memory.ocr_decision"]
    assert len(audit_records) >= 1
    rec = audit_records[0]
    # El record tiene un message o extra con la action
    assert hasattr(rec, "action")
    assert rec.action == "accept_null"
    assert rec.file_id == seeded_ocr_row


# ---------------------------------------------------------------------------
# NORTH STAR: external_ocr NO esta en el tool registry
# ---------------------------------------------------------------------------


def test_external_ocr_not_in_llm_tool_registry():
    """NORTH STAR: /externalOCR NO esta expuesto como tool al LLM.

    Es un comando de usuario EXPLICITO. El LLM NO puede invocarlo
    autonomamente. Esto blinda la soberania de datos: el LLM nunca
    puede escalar a hosted VLM sin que el usuario escriba el comando
    en Telegram o WebUI.

    M4 fix (2026-07-11): xfail instead of skip. Rationale: skip is
    silent in coverage reports (the test "doesn't run"), so a future
    implementation of `hermes.agent.tools` that VIOLATES NORTH STAR
    (e.g., exposes external_ocr as a tool) would go undetected. xfail
    is explicit: the test is "expected to fail" because the module
    doesn't exist; the next developer who creates `hermes.agent.tools`
    will see this test in the failure report and must either:
    (a) make the test pass by NOT exposing external_ocr (correct), or
    (b) xfail-no-trace if they intentionally want the gate deferred.
    """
    try:
        from hermes.agent import tools  # type: ignore[import-not-found]
    except ImportError:
        # M4 fix: xfail, not skip. Reason documented in pytest -rx output
        # so it's visible in CI. The test will pass automatically once
        # `hermes.agent.tools` is created AND external_ocr is absent.
        pytest.xfail(
            "hermes.agent.tools not yet implemented (4d code). "
            "When created, this test enforces NORTH STAR: external_ocr "
            "MUST NOT be in the LLM tool registry (TDD §4.3.1)."
        )

    tool_names = [name for name in dir(tools) if not name.startswith("_")]
    assert "external_ocr" not in tool_names, (
        "NORTH STAR VIOLATION: external_ocr must NOT be a tool callable "
        "by the LLM. It's a user-only command (TDD §4.3.1)."
    )
    assert "edit_text" not in tool_names
    assert "skip_ocr" not in tool_names
    assert "edge_ocr" not in tool_names
    assert "accept_null" not in tool_names

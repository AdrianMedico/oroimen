"""Tests Sprint 15 (US-3.1 §4 PR #69): embed_vault.scan().

Cubre:
- Vault vacio / inexistente -> resultado con scanned=0 (no falla).
- Vault con archivos no soportados (.jpg, .iso) -> skipped_unsupported.
- Vault con archivos > MAX_FILE_BYTES -> skipped_too_large.
- Happy path: vault con 3 .md -> 3 embeddings, content_hash dedupe.
- Idempotencia: scan dos veces -> segunda vez skipped_unchanged.
- Dedup cross-vault-upload: subir via /v1/files + escanear el mismo
  archivo en vault -> no duplica.
- RAG disabled: scan no falla, embedded=0 (warning loggeado).
- Concurrencia: 20 archivos procesados en paralelo limitado a 5
  simultaneos (asyncio.Semaphore, no se observa OOM).
- Cancelacion: cancel_event seteado entre archivos -> scan termina
  antes de procesar todos.

Patron: NO pegamos contra Qwen 3 / OpenRouter. Usamos el fixture
`embeddings_mock` del conftest (PR #69) que devuelve un vector fijo
de 4096 dims sin red. Esto:
- 0 ms por embed vs ~150 ms real.
- Determinismo: cosine similarity siempre = 1.0 (vector constante).
- Sin rate limit / sin coste.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from hermes.memory.db import Database


@pytest.fixture
async def db(tmp_path: Path) -> Database:
    d = Database(tmp_path / "test.db")
    await d.initialize()
    yield d
    await d.close()


def _make_vault(tmp_path: Path, files: dict[str, str]) -> Path:
    """Helper: crea un vault con archivos {filename: content}."""
    vault = tmp_path / "vault"
    vault.mkdir()
    for name, content in files.items():
        (vault / name).write_text(content, encoding="utf-8")
    return vault


# ---------------------------------------------------------------------------
# Tests: vault vacio / invalido
# ---------------------------------------------------------------------------


async def test_scan_empty_vault_returns_zero_scanned(
    db: Database, embeddings_mock: Any, tmp_path: Path
) -> None:
    """Vault que no existe (o vacio) -> scanned=0, no falla."""
    from hermes.services.embed_vault import scan

    result = await scan(
        vault_path=tmp_path / "no_existe",
        db=db,
        embeddings_service=embeddings_mock,
    )
    assert result.scanned == 0
    assert result.embedded == 0
    assert result.failed == 0


async def test_scan_vault_with_only_unsupported_extensions(
    db: Database, embeddings_mock: Any, tmp_path: Path
) -> None:
    """Archivos .jpg, .bin se cuentan como skipped_unsupported."""
    from hermes.services.embed_vault import scan

    vault = _make_vault(
        tmp_path,
        {"foto.jpg": "binary", "backup.bin": "more binary"},
    )
    result = await scan(vault_path=vault, db=db, embeddings_service=embeddings_mock)
    assert result.scanned == 0
    assert result.skipped_unsupported == 2


async def test_scan_skips_files_larger_than_max(
    db: Database, embeddings_mock: Any, tmp_path: Path, monkeypatch
) -> None:
    """Archivos > MAX_FILE_BYTES (10MB default) -> skipped_too_large.

    No creamos un archivo de 10MB en disco (lento). Reducimos
    MAX_FILE_BYTES a 1 byte via monkeypatch y luego creamos un
    archivo de 2 bytes.
    """
    from hermes.services import embed_vault

    monkeypatch.setattr(embed_vault, "MAX_FILE_BYTES", 1)
    from hermes.services.embed_vault import scan

    vault = _make_vault(tmp_path, {"tiny.md": "ab"})  # 2 bytes > 1
    result = await scan(vault_path=vault, db=db, embeddings_service=embeddings_mock)
    assert result.scanned == 0
    assert result.skipped_too_large == 1


# ---------------------------------------------------------------------------
# Tests: happy path
# ---------------------------------------------------------------------------


async def test_scan_embeds_three_markdown_files(
    db: Database, embeddings_mock: Any, tmp_path: Path
) -> None:
    """3 archivos .md -> 3 embeddings, todos con content_hash en DB."""
    from hermes.services.embed_vault import scan

    vault = _make_vault(
        tmp_path,
        {
            "a.md": "primera nota",
            "b.md": "segunda nota distinta",
            "c.txt": "tercer archivo texto plano",
        },
    )
    result = await scan(vault_path=vault, db=db, embeddings_service=embeddings_mock)
    assert result.scanned == 3
    assert result.embedded == 3
    assert result.skipped_unchanged == 0
    assert result.failed == 0
    # Verificamos que los embeddings estan en la DB.
    pending = await db.list_files_pending_embedding()
    assert pending == []  # todos tienen embedding


async def test_scan_is_idempotent(db: Database, embeddings_mock: Any, tmp_path: Path) -> None:
    """scan() dos veces seguidas -> 2da vez skipped_unchanged, no duplica.

    Caso de uso real: cron que corre embed_vault cada hora. No debe
    re-embeber 5000 archivos ya procesados.
    """
    from hermes.services.embed_vault import scan

    vault = _make_vault(tmp_path, {"a.md": "contenido estable", "b.md": "otro"})
    # 1er scan: embe 2
    r1 = await scan(vault_path=vault, db=db, embeddings_service=embeddings_mock)
    assert r1.embedded == 2
    # 2do scan: 0 nuevos, 2 skipped_unchanged
    r2 = await scan(vault_path=vault, db=db, embeddings_service=embeddings_mock)
    assert r2.embedded == 0
    assert r2.skipped_unchanged == 2
    # Solo 2 filas en files (no duplica).
    all_files = await db.list_files(limit=100)
    assert len(all_files) == 2


async def test_scan_dedups_against_existing_uploads(
    db: Database, embeddings_mock: Any, tmp_path: Path
) -> None:
    """Archivo subido via /v1/files con mismo texto -> no se duplica en vault.

    Escenario: user subio paper.pdf via Telegram (con embedding), luego
    el mismo paper aparece en vault via scp. Scan debe detectar
    content_hash Y que ya tiene embedding -> skipped_unchanged.

    Caso contrario (file pre-existente SIN embedding): se re-embed el
    file existente en vez de duplicar el row.
    """
    import hashlib

    from hermes.services.embed_vault import scan

    shared_text = "este es el contenido del paper que esta en ambos sitios"
    content_hash = hashlib.sha256(shared_text.encode("utf-8")).hexdigest()
    # 1) Upload via API path (simulado) CON embedding
    file_id = "file_uploaded_123"
    await db.add_file(
        file_id=file_id,
        filename="paper.pdf",
        mime_type="application/pdf",
        size_bytes=len(shared_text),
        extracted_text=shared_text,
        extraction_method="pypdf",
        source="upload",
        content_hash=content_hash,
    )
    # Embed directo (sin pasar por embeddings_mock para setup claro)
    import numpy as np

    await db.add_file_embedding(
        file_id, np.full(4096, 0.5, dtype=np.float32).tobytes(), model="setup"
    )
    # 2) Scan vault con el mismo texto
    vault = _make_vault(tmp_path, {"paper-copy.md": shared_text})
    result = await scan(vault_path=vault, db=db, embeddings_service=embeddings_mock)
    # El scan detecta content_hash hit + file YA tiene embedding ->
    # skipped_unchanged (no duplica, no re-embed).
    assert result.embedded == 0
    assert result.skipped_unchanged == 1
    # Solo 1 fila en files (la del upload).
    all_files = await db.list_files(limit=100)
    assert len(all_files) == 1
    assert all_files[0]["id"] == file_id


async def test_scan_uses_content_hash_prefix_as_file_id(
    db: Database, embeddings_mock: Any, tmp_path: Path
) -> None:
    """El file_id generado para archivos de vault es estable entre re-scans.

    Si el archivo se mueve / se re-scanea, el ID no cambia. Esto es la
    base del inode-style preview de PR #68.
    """
    from hermes.services.embed_vault import scan

    vault = _make_vault(tmp_path, {"my-note.md": "contenido unico"})
    await scan(vault_path=vault, db=db, embeddings_service=embeddings_mock)
    all_files = await db.list_files(limit=100)
    assert len(all_files) == 1
    fid = all_files[0]["id"]
    assert fid.startswith("file_")
    # 24 chars despues del prefix (mismo shape que /v1/files).
    assert len(fid) == 5 + 24


# ---------------------------------------------------------------------------
# Tests: cancellation y concurrency
# ---------------------------------------------------------------------------


async def test_scan_respects_cancel_event(
    db: Database, embeddings_mock: Any, tmp_path: Path
) -> None:
    """cancel_event pre-set() -> scan termina sin procesar nada.

    Cubrir cancelacion es importante porque embed_vault corre a veces
    en background (cron de sistema) y queremos poder pararlo sin kill -9.
    El check de cancel_event esta al inicio de _process_one, asi que
    si el caller setea cancel antes de scan(), NINGUN archivo se procesa.
    """
    from hermes.services.embed_vault import scan

    files = {f"f{i}.md": f"contenido {i}" for i in range(10)}
    vault = _make_vault(tmp_path, files)
    cancel = asyncio.Event()
    cancel.set()  # pre-cancelled

    result = await scan(
        vault_path=vault,
        db=db,
        embeddings_service=embeddings_mock,
        cancel_event=cancel,
    )
    # Ningun archivo procesado (todos skipped via cancel check).
    assert result.scanned == 10  # encontrados en disco
    assert result.embedded == 0  # ninguno llego a embeber


async def test_scan_uses_semaphore_concurrency_5(
    db: Database, embeddings_mock: Any, tmp_path: Path
) -> None:
    """20 archivos se procesan en paralelo limitado a 5 simultaneos.

    No podemos medir concurrencia directamente sin instrumentar el
    backend, pero verificamos que el scan termina sin OOM / deadlock
    cuando hay 20 archivos. Esto confirma que el asyncio.Semaphore(5)
    no es bloqueante y que asyncio.gather completa.
    """
    from hermes.services.embed_vault import scan

    files = {f"f{i:02d}.md": f"contenido {i}" for i in range(20)}
    vault = _make_vault(tmp_path, files)
    result = await scan(vault_path=vault, db=db, embeddings_service=embeddings_mock)
    assert result.scanned == 20
    assert result.embedded == 20
    assert result.failed == 0


async def test_scan_times_out_on_hanging_embedder(
    db: Database, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Chore 2026-07-05 (Nemotron 3 Ultra 550B review): embed timeout.

    Si `embed_and_store` cuelga (network stall, rate limit sin
    respuesta), el scan sin timeout atasca 5 slots del semaphore para
    siempre. Con `asyncio.wait_for(EMBED_TIMEOUT_S)`, el archivo cae
    en `failed` con un mensaje claro y el scan continua.

    Mockeamos el embeddings service con un sleep > timeout. Reducimos
    EMBED_TIMEOUT_S a 0.1s para que el test sea rapido.
    """
    import asyncio as _asyncio

    from hermes.services import embed_vault

    monkeypatch.setattr(embed_vault, "EMBED_TIMEOUT_S", 0.1)

    class _HangingEmbeddings:
        is_enabled = True

        async def embed_and_store(self, file_id: str, text: str) -> bool:
            await _asyncio.sleep(5.0)  # > timeout
            return True

        async def embed(self, text: str):
            return None

    vault = _make_vault(tmp_path, {"a.md": "contenido"})
    result = await embed_vault.scan(
        vault_path=vault,
        db=db,
        embeddings_service=_HangingEmbeddings(),
    )
    # El archivo NO se embe (timeout) -> failed += 1
    assert result.embedded == 0
    assert result.failed == 1
    assert any("embed timeout" in err for _, err in result.errors)


# ---------------------------------------------------------------------------
# Tests: RAG disabled
# ---------------------------------------------------------------------------


async def test_scan_rag_disabled_returns_immediately(db: Database, tmp_path: Path) -> None:
    """Si embeddings.is_enabled = False, scan no falla ni embebe nada.

    El scan debe ser no-op (con warning loggeado) cuando RAG está
    deshabilitado, NO fallar. Asi un user con key de OpenRouter
    vacía puede correr el scan sin crashes.
    """

    class _DisabledService:
        is_enabled = False

        async def embed_and_store(self, file_id: str, text: str) -> bool:
            raise AssertionError("should not be called when RAG disabled")

    vault = _make_vault(tmp_path, {"a.md": "contenido"})

    # Reimportamos dentro del test para que el path sea correcto.
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from hermes.services.embed_vault import scan

    result = await scan(vault_path=vault, db=db, embeddings_service=_DisabledService())
    assert result.scanned == 0  # ni siquiera intentamos
    assert result.embedded == 0


async def test_scan_counts_empty_text_as_skipped_empty(
    db: Database, embeddings_mock: Any, tmp_path: Path
) -> None:
    """Files con texto vacio (PDF sin OCR, .txt vacio) cuentan como skipped_empty.

    Chore 2026-07-05: Nemotron 3 Super review del PR #70 encontro que
    un archivo con extracted_text vacio se contaba en `scanned` pero
    en ningun otro counter (skipped_*, embedded, failed). Esto
    dejaba un gap en el accounting: scanned != suma del resto.

    Tras el fix, `skipped_empty` lleva la cuenta de estos casos para
    que el caller (CLI, dashboard) pueda reportar "10 archivos no
    tenian texto extraible (PDFs escaneados, .txt vacios)" en vez
    de mostrar un numero fantasma en scanned.
    """
    from hermes.services.embed_vault import scan

    # Mezcla: 1 archivo con contenido real + 1 vacio + 1 con solo whitespace
    vault = _make_vault(
        tmp_path,
        {
            "real.md": "contenido real",
            "empty.md": "",
            "whitespace.md": "   \n\t  ",
        },
    )
    result = await scan(vault_path=vault, db=db, embeddings_service=embeddings_mock)
    # scanned cuenta todos los que llegaron a candidates (los 3)
    assert result.scanned == 3
    # 1 se embe
    assert result.embedded == 1
    # 2 van a skipped_empty (empty.md y whitespace.md)
    assert result.skipped_empty == 2
    # Accounting cuadra: 3 == 1 + 2 + 0 (sin failed ni skipped_unchanged)
    assert result.scanned == (
        result.embedded
        + result.skipped_empty
        + result.failed
        + result.skipped_unchanged
        + result.skipped_too_large
        + result.skipped_unsupported
    )

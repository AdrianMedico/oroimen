"""Tests Sprint 15 (US-3.1): /v1/files upload dedup + DB-only reads.

Cubre:
- POST /v1/files dedup transparente por SHA256 del texto extraido:
  HTTP 200 + deduplicated=true si el content_hash ya existe.
  HTTP 201 + deduplicated=false si es nuevo.
- GET /v1/files/{id}: lee directamente de DB (sin cache in-memory).
- Files sin texto extraido (binarios/imagenes) tienen content_hash=None
  y NO se deduplican entre si (evita falsos positivos).
- content_hash es SHA256 hex de 64 chars.

Patron: usar el fixture `db` real + TestClient de FastAPI para crear
el `app` con `create_app`. Asi verificamos el camino end-to-end
(upload -> sha256 -> dedup -> DB -> GET).
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from hermes.config import Settings
from hermes.memory.db import Database
from hermes.receivers.http_api import create_app


@pytest.fixture
def http_settings(
    settings: Settings,
    tmp_path: pytest.TempPathFactory,  # type: ignore[valid-type]
    monkeypatch: pytest.MonkeyPatch,
) -> Settings:
    """Reusa el fixture `settings` del conftest + bearer token para auth."""
    monkeypatch.setenv("HERMES_FILES_DIR", str(tmp_path))
    monkeypatch.setenv("HERMES_HTTP_API_API_KEY", "test-key-1234")
    monkeypatch.setenv("HERMES_HTTP_API_AUTH_ENABLED", "true")
    monkeypatch.setenv("HERMES_RATE_LIMIT_PER_MINUTE", "0")
    return Settings(_env_file=None, **settings.model_dump())  # type: ignore[arg-type]


@pytest.fixture
def http_client(http_settings: Settings, db: Database) -> Any:
    """TestClient con app real + router/registry/embeddings mock."""
    from unittest.mock import MagicMock

    router = MagicMock()
    registry = None  # tools deshabilitadas para simplificar
    embeddings = None  # RAG deshabilitado para tests de dedup
    app = create_app(
        settings=http_settings,
        db=db,
        router=router,
        registry=registry,
        embeddings_service=embeddings,
    )
    return TestClient(app, headers={"Authorization": "Bearer test-key-1234"})


def _upload(client: TestClient, filename: str, content: bytes, mime: str = "text/plain"):
    """Helper: POST /v1/files como multipart/form-data."""
    return client.post(
        "/v1/files",
        files={"file": (filename, content, mime)},
        params={"purpose": "assistants"},
    )


# ---------------------------------------------------------------------------
# Tests Sprint 15 (US-3.1)
# ---------------------------------------------------------------------------


async def test_upload_file_new_returns_201_with_deduplicated_false(
    http_client: TestClient, db: Database
) -> None:
    """Upload de file nuevo -> HTTP 201 + deduplicated=false."""
    resp = _upload(http_client, "hello.txt", b"Hello world!")
    assert resp.status_code == 201
    body = resp.json()
    assert body["deduplicated"] is False
    assert body["filename"] == "hello.txt"
    assert body["extracted_text"] == "Hello world!"
    # Persistido en DB con content_hash = SHA256("Hello world!")
    import hashlib as _hl

    expected_hash = _hl.sha256(b"Hello world!").hexdigest()
    row = await db.get_file(body["id"])
    assert row is not None
    assert row["content_hash"] == expected_hash


async def test_upload_file_dedup_via_sha256_returns_existing(
    http_client: TestClient, db: Database
) -> None:
    """Re-upload del MISMO texto -> HTTP 200 + deduplicated=true, mismo file_id.

    Sprint 15 (US-3.1 §2.1): el user sube un PDF, lo borra (o reinicia),
    y lo vuelve a subir. En vez de crear un duplicado en la library,
    devolvemos el file_id existente. Esto cubre tambien el caso de
    Open WebUI que reintenta uploads tras un timeout.
    """
    # Primer upload: NUEVO
    r1 = _upload(http_client, "greeting.txt", b"Buenos dias")
    assert r1.status_code == 201
    first_id = r1.json()["id"]
    assert r1.json()["deduplicated"] is False
    # Segundo upload: mismo TEXTO pero filename DIFERENTE
    r2 = _upload(http_client, "saludo.txt", b"Buenos dias")
    assert r2.status_code == 200  # OK idempotente
    assert r2.json()["id"] == first_id  # MISM0 file_id
    assert r2.json()["deduplicated"] is True
    # Solo 1 row en la DB (no se duplico)
    rows = await db.list_files(limit=100)
    assert len([r for r in rows if r["id"] == first_id]) == 1


async def test_upload_file_dedup_ignores_filename_only(
    http_client: TestClient, db: Database
) -> None:
    """Dos filenames distintos con el mismo texto colisionan en dedup.

    Razon: la deduplicacion es POR CONTENIDO (texto extraido), no por
    nombre. Asi preparamos el terreno para Sprint 10 (Drive cross-source
    dedup): un file de Drive con el mismo texto colisionara con un
    upload local, manteniendo la library limpia.
    """
    r1 = _upload(http_client, "v1.txt", b"contenido identico")
    assert r1.status_code == 201
    r2 = _upload(http_client, "v2-renamed.txt", b"contenido identico")
    assert r2.status_code == 200
    assert r2.json()["id"] == r1.json()["id"]
    # El filename del primero se preserva (el segundo es solo un hit)
    assert r2.json()["filename"] == "v1.txt"


async def test_upload_file_no_text_uses_null_content_hash(
    http_client: TestClient, db: Database
) -> None:
    """Files sin texto extraido (imagenes) tienen content_hash=NULL.

    Sin esto, dos imagenes distintas darian SHA256(b"") = el mismo hash,
    lo cual provocaria falsos positivos: subir foto1.jpg + foto2.jpg
    devolveria 200 deduplicated=true para la segunda. Bug inaceptable.
    """
    # El helper _extract_file_text_async retorna "" para binarios.
    # Simulamos un PDF "vacio" o un binario subiendo bytes arbitrarios:
    resp = _upload(
        http_client,
        "image.png",
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR...",
        mime="image/png",
    )
    assert resp.status_code == 201
    body = resp.json()
    row = await db.get_file(body["id"])
    assert row is not None
    assert row["content_hash"] is None  # dedup disabled para binarios


async def test_get_file_reads_from_db_after_restart(http_client: TestClient, db: Database) -> None:
    """GET /v1/files/{id} lee de DB directo (simula restart).

    Sprint 15: sin cache in-memory, el file es recuperable tras un
    restart del container. Este test simula el restart creando una
    nueva instancia de TestClient con el MISMO `db` (que es el
    singleton de hermes en produccion).
    """
    # Subimos
    r1 = _upload(http_client, "persistent.txt", b"dato importante")
    assert r1.status_code == 201
    file_id = r1.json()["id"]
    # GET funciona
    r_get = http_client.get(f"/v1/files/{file_id}")
    assert r_get.status_code == 200
    assert r_get.json()["filename"] == "persistent.txt"
    # Simulamos "restart" creando un NUEVO TestClient con el mismo db
    from unittest.mock import MagicMock

    new_app = create_app(
        settings=http_client.app.state.settings,
        db=db,  # mismo DB singleton
        router=MagicMock(),
        registry=None,
    )
    new_client = TestClient(new_app, headers={"Authorization": "Bearer test-key-1234"})
    r_after_restart = new_client.get(f"/v1/files/{file_id}")
    assert r_after_restart.status_code == 200  # sigue accesible
    assert r_after_restart.json()["filename"] == "persistent.txt"


async def test_get_file_unknown_id_returns_404(http_client: TestClient, db: Database) -> None:
    """GET /v1/files/{id} con id inexistente -> 404."""
    resp = http_client.get("/v1/files/file_does_not_exist")
    assert resp.status_code == 404
    body = resp.json()
    assert body["detail"]["error"]["type"] == "not_found"


async def test_upload_file_dedup_hash_format_is_sha256_hex(
    http_client: TestClient, db: Database
) -> None:
    """content_hash se almacena como SHA256 hex de 64 chars.

    Esto valida el shape y tambien que la columna `content_hash TEXT`
    tiene espacio de sobra. Si sub-cambiamos a un hash de 32 chars (e.g.
    md5), rompera este test y el UNIQUE INDEX.
    """
    import re

    resp = _upload(http_client, "shape.txt", b"validar formato de hash")
    assert resp.status_code == 201
    file_id = resp.json()["id"]
    row = await db.get_file(file_id)
    assert row is not None
    assert row["content_hash"] is not None
    assert len(row["content_hash"]) == 64
    assert re.fullmatch(r"[0-9a-f]{64}", row["content_hash"])


# ---------------------------------------------------------------------------
# Tests Sprint 15 (US-3.1 §4 PR #69): GET /v1/files list endpoint
# ---------------------------------------------------------------------------


async def test_list_files_returns_empty_library(http_client: TestClient) -> None:
    """Library vacia -> data=[], count=0, has_more=false."""
    resp = http_client.get("/v1/files")
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    assert body["data"] == []
    assert body["count"] == 0
    assert body["has_more"] is False


async def test_list_files_returns_uploaded_files(
    http_client: TestClient,
) -> None:
    """3 archivos subidos -> list devuelve 3, ordenados DESC."""
    for i, content in enumerate([b"primero", b"segundo", b"tercero"]):
        r = _upload(http_client, f"file{i}.txt", content)
        assert r.status_code == 201
    resp = http_client.get("/v1/files")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 3
    assert len(body["data"]) == 3
    # DESC: tercero primero.
    assert body["data"][0]["filename"] == "file2.txt"


async def test_list_files_respects_limit(http_client: TestClient) -> None:
    """limit=2 con 5 archivos -> 2 entries, has_more=true."""
    for i in range(5):
        _upload(http_client, f"f{i}.txt", f"content{i}".encode())
    resp = http_client.get("/v1/files?limit=2")
    body = resp.json()
    assert body["count"] == 2
    assert body["limit"] == 2
    assert body["has_more"] is True


async def test_list_files_has_more_is_false_on_last_exact_page(
    http_client: TestClient,
) -> None:
    """Chore 2026-07-05 (Nemotron 3 Ultra 550B review): has_more fiable.

    Antes: `has_more = len(rows) == limit` daba falso positivo en la
    ultima pagina exacta. Ej: 4 archivos con limit=4 -> devolvia 4
    rows -> has_more=True incorrecto (no hay mas).
    Despues: pedimos limit+1, descartamos el extra si lo hay.

    Caso 1: 4 archivos + limit=4 -> has_more=False (correcto).
    Caso 2: 5 archivos + limit=4 -> has_more=True (correcto, hay 1 mas).
    """
    # Caso 1: exactamente limit archivos
    for i in range(4):
        _upload(http_client, f"a{i}.txt", f"a{i}".encode())
    body = http_client.get("/v1/files?limit=4").json()
    assert body["count"] == 4
    assert body["has_more"] is False  # antes: True (bug)
    # Caso 2: limit + 1 archivos
    _upload(http_client, "extra.txt", b"extra")
    body = http_client.get("/v1/files?limit=4").json()
    assert body["count"] == 4
    assert body["has_more"] is True


async def test_list_files_pagination_with_offset(
    http_client: TestClient,
) -> None:
    """offset=2 salta las 2 primeras entries (DESC order estable)."""
    for i in range(5):
        _upload(http_client, f"f{i}.txt", f"content{i}".encode())
    page1 = http_client.get("/v1/files?limit=2&offset=0").json()
    page2 = http_client.get("/v1/files?limit=2&offset=2").json()
    # Las pages no se solapan.
    page1_ids = {f["id"] for f in page1["data"]}
    page2_ids = {f["id"] for f in page2["data"]}
    assert page1_ids.isdisjoint(page2_ids)
    # Union cubre los 4 archivos esperados (queda 1 sin paginar).
    assert len(page1_ids | page2_ids) == 4


async def test_list_files_clamps_limit_to_500(
    http_client: TestClient,
) -> None:
    """limit fuera de [1, 500] se clampa. 10000 -> 500."""
    resp = http_client.get("/v1/files?limit=10000")
    body = resp.json()
    assert body["limit"] == 500  # clampeado al techo


async def test_list_files_filter_by_source(http_client: TestClient, db: Database) -> None:
    """source=upload filtra a uploads; source=vault filtra a vault."""
    _upload(http_client, "via_api.txt", b"subido via API")
    # Insertar un file con source distinto via DB directa.
    await db.add_file(
        "file_vault_001",
        "from_vault.txt",
        "text/plain",
        10,
        "vault content",
        "raw",
        source="vault",
    )
    # Solo uploads
    uploads = http_client.get("/v1/files?source=upload").json()
    assert all(f["source"] == "upload" for f in uploads["data"])
    # Solo vault
    vault = http_client.get("/v1/files?source=vault").json()
    assert len(vault["data"]) == 1


# Sprint 17 (F4-1): race condition entre find_file_by_content_hash + add_file.
# Documentado en hermes/receivers/http_api.py:1338-1394 (Sprint 15) y
# en test_files_e2e.py:391-399. UNIQUE INDEX en files.content_hash
# (partial WHERE content_hash IS NOT NULL) es la red de seguridad.
# El handler debe devolver HTTP 200 idempotente cuando un request
# concurrente gana el INSERT primero.


@pytest.mark.asyncio
async def test_upload_file_integrity_error_returns_idempotent_200(
    monkeypatch: pytest.MonkeyPatch, http_client: TestClient, db: Database
) -> None:
    """Sprint 17 (F4-1): cuando add_file() levanta IntegrityError (otro
    request concurrente gano el UNIQUE INDEX), el handler hace refetch
    del file ganador y devuelve 200 deduplicated=true en vez de 500.

    Simula la race: el primer request pasa dedup check (find_file=None),
    pero el UNIQUE INDEX detecta que el contenido ya existe (porque un
    request anterior lo inserto). El handler refetchea y responde
    idempotentemente.
    """
    import sqlite3 as _sqlite3

    from hermes.memory import db as _db_module

    # Pre-poblar la DB con un file que tiene el hash del texto que vamos a subir
    text = b"contenido del race winner"
    pre_id = "file_race_winner_0000000000"
    import hashlib as _hl

    expected_hash = _hl.sha256(text).hexdigest()
    await db.add_file(
        file_id=pre_id,
        filename="winner.txt",
        mime_type="text/plain",
        size_bytes=len(text),
        extracted_text=text.decode("utf-8"),
        extraction_method="",
        source="upload",
        content_hash=expected_hash,
    )
    # Pre-add OK. Ahora simular que un request concurrente ya paso el
    # dedup check (find_file_by_content_hash=None) Y que add_file se
    # llama despues de que el row pre-existente fue committeado. Forzamos
    # el IntegrityError monkey-patching add_file en el namespace del
    # modulo http_api (que es donde se hace la llamada).
    call_count = {"n": 0}

    async def add_file_raising_integrity(self: Any, **kwargs: Any) -> Any:
        call_count["n"] += 1
        raise _sqlite3.IntegrityError("UNIQUE constraint failed: files.content_hash")

    monkeypatch.setattr("hermes.receivers.http_api.Database.add_file", add_file_raising_integrity)

    # Subir el mismo texto — handler debe:
    # 1) hacer dedup check (find_file) -> encuentra el pre_id (porque
    #    ya esta en DB), asi que retornara 200 deduplicated=true via el
    #    path normal SIN tocar add_file. Eso es lo correcto, pero no es
    #    el test del race.
    # Para forzar el race, monkey-patch el path: que find_file devuelva
    # None aunque el row exista (simula el find hecho antes del INSERT
    # del winner).
    original_find = _db_module.Database.find_file_by_content_hash

    async def find_file_returning_none(self: Any, h: str) -> Any:
        return None  # simula: el winner todavia no era visible cuando hicimos find

    monkeypatch.setattr(
        "hermes.receivers.http_api.Database.find_file_by_content_hash",
        find_file_returning_none,
    )

    # Para que el refetch dentro del except funcione, restauramos find_file
    # al original justo DESPUES del primer call (que devuelve None):
    call_state = {"first": True}

    async def find_file_smart(self: Any, h: str) -> Any:
        if call_state["first"]:
            call_state["first"] = False
            return None
        return await original_find(self, h)

    monkeypatch.setattr(
        "hermes.receivers.http_api.Database.find_file_by_content_hash",
        find_file_smart,
    )

    # Subir el mismo texto: debe devolver 200 deduplicated=true con
    # el file_id del winner (no el nuevo generado por el loser).
    resp = _upload(http_client, "loser.txt", text)
    assert resp.status_code == 200, f"esperaba 200 idempotente, got {resp.status_code}"
    body = resp.json()
    assert body["id"] == pre_id, f"esperaba winner id {pre_id}, got {body['id']}"
    assert body["deduplicated"] is True
    # Y add_file SI fue llamado (y levanto IntegrityError)
    assert call_count["n"] == 1


@pytest.mark.asyncio
async def test_upload_file_integrity_error_with_no_existing_row_raises(
    monkeypatch: pytest.MonkeyPatch, http_client: TestClient, db: Database
) -> None:
    """Sprint 17 (F4-1) edge case: si add_file levanta IntegrityError PERO
    el refetch no encuentra el row ganador (caso muy raro: UNIQUE
    violation que se resuelve entre medio, o index corrupto), el handler
    debe propagar el error en vez de devolver respuesta falsa.
    """
    import sqlite3 as _sqlite3

    async def add_file_raising(self: Any, **kwargs: Any) -> Any:
        raise _sqlite3.IntegrityError("UNIQUE constraint failed: files.content_hash")

    async def find_file_always_none(self: Any, h: str) -> Any:
        return None

    monkeypatch.setattr("hermes.receivers.http_api.Database.add_file", add_file_raising)
    monkeypatch.setattr(
        "hermes.receivers.http_api.Database.find_file_by_content_hash",
        find_file_always_none,
    )

    # El handler debe propagar el IntegrityError (no lo swallowea).
    # En el endpoint HTTP real eso se traduce a 500 Internal Server Error.
    # Usamos TestClient que propagara el error via raise_server_exceptions.
    with pytest.raises(_sqlite3.IntegrityError):
        _upload(http_client, "edge.txt", b"contenido edge case")

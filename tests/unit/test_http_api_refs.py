"""Tests Sprint 15 (US-3.1 §10): /v1/files reference endpoints.

Cubre:
- GET /v1/files/{id} expone `reference_count` (ya estaba en la DB, ahora
  visible al cliente) — base para que el cliente decida si re-embed.
- GET /v1/files/{id}/refs devuelve lista de mensajes que referencian
  este file, con metadata minima (msg_id, conv_id, role, timestamp,
  snippet del content).
- 404 en /refs si el file no existe (vs 200 + [] si existe pero sin
  refs — son señales útiles distintas).

Patron: igual que test_http_api_dedup.py — fixtures `settings` del
conftest (credenciales fake) + DB real (schema v15 de PR #66).
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
    """Reusa fixture `settings` del conftest + bearer token."""
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
    app = create_app(
        settings=http_settings,
        db=db,
        router=router,
        registry=None,
        embeddings_service=None,
    )
    return TestClient(app, headers={"Authorization": "Bearer test-key-1234"})


def _upload(client: TestClient, filename: str, content: bytes, mime: str = "text/plain"):
    """Helper: POST /v1/files."""
    return client.post(
        "/v1/files",
        files={"file": (filename, content, mime)},
        params={"purpose": "assistants"},
    )


# ---------------------------------------------------------------------------
# reference_count en /v1/files/{id}
# ---------------------------------------------------------------------------


async def test_get_file_exposes_reference_count(http_client: TestClient) -> None:
    """GET /v1/files/{id} ahora incluye `reference_count`.

    Antes de PR #68: la columna existia en la DB pero no estaba
    visible en la respuesta (el endpoint devolvia la fila raw sin
    esa key — bug menor de S9.0 que el cliente no podia usar para
    detectar archivos 'huérfanos por uso').

    Ahora: cualquier cliente puede saber 'cuantas veces se ha usado
    este PDF' sin tener que llamar /refs. Caso de uso Open WebUI:
    ordenar la library por uso descendente.
    """
    # Subimos un archivo (sin refs todavia).
    resp = _upload(http_client, "unused.txt", b"contenido")
    assert resp.status_code == 201
    file_id = resp.json()["id"]
    # GET expone reference_count.
    get = http_client.get(f"/v1/files/{file_id}")
    assert get.status_code == 200
    body = get.json()
    assert "reference_count" in body
    assert body["reference_count"] == 0  # nadie lo ha referenciado


# ---------------------------------------------------------------------------
# /v1/files/{id}/refs — listar mensajes que referencian el file
# ---------------------------------------------------------------------------


async def test_get_file_refs_empty_when_no_references(
    http_client: TestClient,
) -> None:
    """File subido pero nunca referenciado -> refs=[] (no 404).

    Distincion importante: 404 = "el file no existe", 200+[] = "el
    file existe pero nadie lo ha usado todavia". Permite al cliente
    detectar archivos 'huérfanos por uso' (existen pero nadie los
    referencia → candidatos a delete).
    """
    resp = _upload(http_client, "lonely.txt", b"sin amigos")
    assert resp.status_code == 201
    file_id = resp.json()["id"]
    # /refs devuelve [] (no 404).
    refs = http_client.get(f"/v1/files/{file_id}/refs")
    assert refs.status_code == 200
    body = refs.json()
    assert body["file_id"] == file_id
    assert body["data"] == []
    assert body["count"] == 0
    assert body["object"] == "list.refs"


async def test_get_file_refs_returns_messages_with_snippet(
    http_client: TestClient, db: Database
) -> None:
    """File referenciado en 2 mensajes -> refs devuelve ambos con snippet.

    Sprint 15 §10 (inode-like preview): el endpoint expone los mensajes
    que contienen este file_id en su `file_refs` (JSON array), junto
    con metadata minima para que el LLM/cliente pueda entender el
    contexto de cada referencia sin cargar el message entero.
    """
    # Subimos y referenciamos en 2 conversaciones distintas.
    resp = _upload(http_client, "shared.txt", b"contenido compartido")
    assert resp.status_code == 201
    file_id = resp.json()["id"]

    conv_a = await db.new_conversation(chat_id=0, user_id=1, thread_id=1)
    conv_b = await db.new_conversation(chat_id=0, user_id=1, thread_id=2)
    msg_a = await db.add_message(
        conv_a, "user", "primera pregunta sobre el PDF", file_refs=[file_id]
    )
    msg_b = await db.add_message(conv_b, "user", "segunda pregunta distinta", file_refs=[file_id])
    # /refs devuelve los 2 mensajes ordenados DESC (más reciente primero).
    refs = http_client.get(f"/v1/files/{file_id}/refs")
    assert refs.status_code == 200
    body = refs.json()
    assert body["count"] == 2
    assert len(body["data"]) == 2
    # msg_b es más reciente → debe ir primero.
    assert body["data"][0]["message_id"] == msg_b
    assert body["data"][0]["conversation_id"] == conv_b
    assert body["data"][0]["role"] == "user"
    # Snippet presente (primeros 200 chars del content del msg).
    assert "segunda pregunta" in body["data"][0]["content_snippet"]
    # msg_a en segundo lugar.
    assert body["data"][1]["message_id"] == msg_a
    assert body["data"][1]["conversation_id"] == conv_a
    assert "primera pregunta" in body["data"][1]["content_snippet"]


async def test_get_file_refs_excludes_messages_without_the_ref(
    http_client: TestClient, db: Database
) -> None:
    """json_each() matchea exactamente el file_id; otros file_refs se ignoran.

    Defensa contra falsos positivos: si un msg tiene
    `file_refs=["file_other"]` no debe aparecer en /refs de `file_target`.
    json_each() sobre el JSON array garantiza esto (vs LIKE que podria
    matchear substring de un ID mas largo).
    """
    resp = _upload(http_client, "target.txt", b"target")
    target_id = resp.json()["id"]
    resp2 = _upload(http_client, "other.txt", b"otro")
    other_id = resp2.json()["id"]

    conv = await db.new_conversation(chat_id=0, user_id=1, thread_id=3)
    await db.add_message(conv, "user", "msg con other", file_refs=[other_id])
    await db.add_message(conv, "user", "msg con target", file_refs=[target_id])
    await db.add_message(conv, "user", "msg con ambos", file_refs=[target_id, other_id])
    # /refs del target debe devolver 2 mensajes (target solo + ambos).
    refs = http_client.get(f"/v1/files/{target_id}/refs")
    assert refs.status_code == 200
    assert refs.json()["count"] == 2
    # Ninguno debe tener "other" en su snippet.
    for row in refs.json()["data"]:
        assert "other" not in row["content_snippet"]


async def test_get_file_refs_404_for_unknown_file(
    http_client: TestClient,
) -> None:
    """404 si el file_id no existe (vs 200+[] cuando existe sin refs)."""
    refs = http_client.get("/v1/files/file_does_not_exist/refs")
    assert refs.status_code == 404
    assert refs.json()["detail"]["error"]["type"] == "not_found"


async def test_get_file_refs_respects_limit(http_client: TestClient, db: Database) -> None:
    """El param `limit` corta la respuesta a N rows (default 50).

    Caso: un user tiene un PDF muy referenciado (e.g. manual de su
    empresa mencionado en 100 conversaciones). Sin paginación, el
    endpoint devolveria 100 rows → JSON pesado. Con limit=10 el cliente
    puede pedir el primer batch.
    """
    resp = _upload(http_client, "popular.txt", b"manual")
    file_id = resp.json()["id"]
    conv = await db.new_conversation(chat_id=0, user_id=1, thread_id=4)
    for i in range(5):
        await db.add_message(conv, "user", f"referencia {i}", file_refs=[file_id])
    refs = http_client.get(f"/v1/files/{file_id}/refs?limit=3")
    assert refs.status_code == 200
    assert refs.json()["count"] == 3
    assert len(refs.json()["data"]) == 3


async def test_get_file_refs_clamps_extreme_limits(
    http_client: TestClient,
) -> None:
    """Defensa: limit fuera de [1, 200] se clampa, no se propaga al DB.

    Chore 2026-07-05: el free-LLM review de PR #68 marco que sin clamp
    un cliente podria pasar limit=999999 y forzar query O(N). Probamos
    los 4 cuadrantes:
    - 0 -> 1 (floor)
    - 1 -> 1 (no-op, dentro del rango)
    - 200 -> 200 (no-op, techo)
    - 1000000 -> 200 (clamp al techo)
    """
    resp = _upload(http_client, "extreme.txt", b"x")
    file_id = resp.json()["id"]
    # limit=0 → clamp a 1
    r = http_client.get(f"/v1/files/{file_id}/refs?limit=0")
    assert r.status_code == 200
    # limit=1 → no-op
    r = http_client.get(f"/v1/files/{file_id}/refs?limit=1")
    assert r.status_code == 200
    # limit=200 → no-op (techo)
    r = http_client.get(f"/v1/files/{file_id}/refs?limit=200")
    assert r.status_code == 200
    # limit=1000000 → clamp a 200 (no crashea, no 500)
    r = http_client.get(f"/v1/files/{file_id}/refs?limit=1000000")
    assert r.status_code == 200
    # limit negativo → clamp a 1
    r = http_client.get(f"/v1/files/{file_id}/refs?limit=-50")
    assert r.status_code == 200


async def test_get_file_refs_no_false_positive_for_substring_ids(
    http_client: TestClient, db: Database
) -> None:
    """json_each() distingue file_abc de file_abcdef (substring trap).

    Si el cliente crea dos files con IDs que uno sea prefijo del otro
    (improbable en UUIDs pero posible), LIKE '%file_abc%' devolveria
    ambos. json_each() parsea el JSON array y matchea exactamente,
    asi que esto se evita.
    """
    # Subimos dos files con IDs hex distintos.
    r1 = _upload(http_client, "a.txt", b"aaa")
    file_short = r1.json()["id"]  # file_<24 hex>
    _r2 = _upload(http_client, "b.txt", b"bbb")
    # r2 lo subimos solo para garantizar un segundo file_id distinto
    # en la DB; no lo usamos directamente (synthetic construye el caso).

    # Para forzar el caso "uno es substring del otro" usamos un par
    # construido a mano. file_short_prefix es un prefijo del ID real.
    file_short_prefix = file_short[:16]  # ej. "file_aabbccdd" si real es "file_aabbccdd1234..."
    # Creamos un file con un ID custom (no via POST, directo a DB).
    file_synthetic = f"file_{file_short_prefix[5:]}extraneous"  # mismo prefijo
    await db.add_file(file_synthetic, "synthetic.txt", "text/plain", 5, "synthetic", "pypdf")

    conv = await db.new_conversation(chat_id=0, user_id=1, thread_id=5)
    await db.add_message(conv, "user", "ref corto", file_refs=[file_short])
    await db.add_message(conv, "user", "ref synthetic", file_refs=[file_synthetic])

    # /refs del corto NO debe incluir el synthetic (json_each exacto).
    refs = http_client.get(f"/v1/files/{file_short}/refs")
    assert refs.status_code == 200
    snippets = [r["content_snippet"] for r in refs.json()["data"]]
    assert "ref corto" in snippets
    assert "ref synthetic" not in snippets

"""Tests para Slice 5 — M6 reconciliation EXTENDIDA (4 fases nuevas).

Sprint 19 TDD-VAULT-COLLECTIONS §6 (CRITICAL). La función
`_reconcile_db_from_filesystem()` ya existía en Sprint 17 (PR #113b
Slice 2.5) pero solo cubría ingest_jobs (5 sub-fases sobre los 4 dirs
del inbox legacy). Slice 5 AÑADE 4 fases más que cierran el gap entre
el drop folder watcher (DropWatcher) y la DB:

  Phase 1 — Collections sync (FS→DB):
      Cada subdir inmediato bajo `drop_root/` es una collection.
      Si no existe en `vault_collections` por `name`, crearla.
      Idempotente.

  Phase 2 — Drop folder files (FS→DB):
      Cada file bajo `drop_root/<col>/` debe tener fila en
      `vault_files` + bridge row en `vault_file_collections`.
      Si falta, insertar (idempotente por `source_path`).
      Backstop del DropWatcher (que puede haber crasheado mid-scan).

  Phase 3 — File orphan detection (FS→DB):
      Cada fila en `vault_files` con `source_path` bajo `drop_root/`
      cuyo archivo físico NO existe se marca con `orphaned_at` via
      `set_file_orphaned()`. NO delete — text + embeddings persisten
      (segundo cerebro / audit trail). Search filtra
      `WHERE orphaned_at IS NULL`. Idempotente (no re-update si ya
      tiene timestamp).

  Phase 4 — Bridge invariant audit:
      Cada fila en `vault_file_collections` DEBE tener `file_id`
      válido en `vault_files` Y `collection_id` válido en
      `vault_collections`. FK CASCADE + RESTRICT deberia prevenirlo,
      pero un safety net nunca sobra. Reporta count de violaciones
      (NO auto-fix — operator decide).

  Phase 5 (zombie edge jobs) — ya cubierta por EdgeZombieScheduler
  (Sprint 19 Slice 4c, PR #137). NO se re-implementa aquí.

TDD red phase. Tests asumen el contrato descrito en
`docs/TDD_VAULT_COLLECTIONS.md` §6. Cuando la implementación exista
en `ingest_router._reconcile_db_from_filesystem()`, los tests pasan
a verde.

Refs:
- docs/TDD_VAULT_COLLECTIONS.md §6 (líneas 1384-1624)
- docs/SPRINT_19_RETROSPECTIVE.md (4a R1 — manifest id mismatch lesson)
- Sprint 17 PR #113b (M6 base, solo ingest_jobs)
- Sprint 19 Slice 1 (PR #126 — collections API)
- Sprint 19 Slice 4a-4c (PR #135-#137 — drop folder + edge)
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from hermes.memory.collections import VaultCollectionsRepo
from hermes.memory.db import Database
from hermes.memory.ingest_router import IngestRouter

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def collections_repo(db: Database) -> VaultCollectionsRepo:
    return VaultCollectionsRepo(db)


@pytest.fixture
def drop_root(tmp_path: Path) -> Path:
    """Path al drop folder, creado."""
    drop = tmp_path / "drop"
    drop.mkdir(parents=True, exist_ok=True)
    return drop


@pytest.fixture
def inbox_root(tmp_path: Path) -> Path:
    """Path al inbox (para fase 0 — ingest_jobs), creado."""
    inbox = tmp_path / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    return inbox


@pytest.fixture
def router(
    settings,  # conftest fixture con credenciales fake
    db: Database,
    inbox_root: Path,
    drop_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> IngestRouter:
    """IngestRouter con vault + inbox mocks (M6 no los usa).

    Pasamos `db=db` para que las 4 fases nuevas tengan acceso a la DB.
    `vault` y `inbox` son MagicMocks — M6 extendido no toca el inbox
    legacy (eso es la fase 0 de M6, ya implementada en Sprint 17).
    """
    monkeypatch.setenv("VAULT_INBOX_ROOT", str(inbox_root))
    monkeypatch.setenv("VAULT_DROP_ROOT", str(drop_root))
    monkeypatch.setenv("VAULT_DROP_ENABLED", "true")
    # Re-instanciar settings con los env vars seteados
    from hermes.config import Settings

    s = Settings(_env_file=None)
    inbox = MagicMock()
    inbox.root = inbox_root

    return IngestRouter(
        vault=MagicMock(),  # M6 extendido no usa vault
        inbox=inbox,
        settings=s,
        db=db,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _file_id_for_source_path(source_path: str) -> str:
    """Genera un file_id determinista (32 hex chars) a partir del path.

    No necesita ser UUID4 real — solo 32 hex chars para satisfacer la
    PK constraint. El path es único por archivo en el drop folder.
    """
    return hashlib.sha256(source_path.encode("utf-8")).hexdigest()[:32]


async def _insert_vault_file(
    db: Database,
    source_path: str,
    *,
    content_sha256: str | None = None,
    mtime: float = 100.0,
    size_bytes: int = 100,
) -> str:
    """Inserta un vault_files row manualmente (sin pasar por Vault.add
    para evitar la lectura real del archivo — los tests de orphan
    necesitan filas con source_path que NO existe en disco).

    Returns: el file_id generado.
    """
    file_id = _file_id_for_source_path(source_path)
    if content_sha256 is None:
        content_sha256 = hashlib.sha256(source_path.encode("utf-8")).hexdigest()
    await db.conn.execute(
        "INSERT INTO vault_files "
        "(file_id, source_path, content_sha256, mtime, size_bytes) "
        "VALUES (?, ?, ?, ?, ?)",
        (file_id, source_path, content_sha256, mtime, size_bytes),
    )
    await db.conn.commit()
    return file_id


async def _vault_files_count(db: Database) -> int:
    cur = await db.conn.execute("SELECT COUNT(*) AS c FROM vault_files")
    row = await cur.fetchone()
    return int(row["c"]) if row else 0


async def _bridge_count(db: Database) -> int:
    cur = await db.conn.execute("SELECT COUNT(*) AS c FROM vault_file_collections")
    row = await cur.fetchone()
    return int(row["c"]) if row else 0


async def _orphaned_count(db: Database) -> int:
    cur = await db.conn.execute(
        "SELECT COUNT(*) AS c FROM vault_files WHERE orphaned_at IS NOT NULL"
    )
    row = await cur.fetchone()
    return int(row["c"]) if row else 0


async def _bridge_violations(db: Database) -> list[tuple[str, str]]:
    """Lista de (file_id, collection_id) en vault_file_collections
    que violan el invariant (file_id o collection_id no existen)."""
    cur = await db.conn.execute(
        """
        SELECT bfc.file_id, bfc.collection_id
        FROM vault_file_collections bfc
        LEFT JOIN vault_files vf ON vf.file_id = bfc.file_id
        LEFT JOIN vault_collections vc ON vc.collection_id = bfc.collection_id
        WHERE vf.file_id IS NULL OR vc.collection_id IS NULL
        """
    )
    rows = await cur.fetchall()
    return [(r["file_id"], r["collection_id"]) for r in rows]


# ---------------------------------------------------------------------------
# Phase 1 — Collections sync (FS→DB)
# ---------------------------------------------------------------------------


async def test_phase1_creates_collection_for_each_subdir(
    router: IngestRouter,
    collections_repo: VaultCollectionsRepo,
    tmp_path: Path,
) -> None:
    """Cada subdir inmediato bajo `drop_root/` se vuelve una collection."""
    drop = tmp_path / "drop"
    (drop / "01_Proyectos_Activos").mkdir()
    (drop / "02_Areas").mkdir()
    (drop / "03_Resources").mkdir()

    result = await router._reconcile_db_from_filesystem()

    assert result["phase1_collections_created"] == 3
    assert await collections_repo.get_collection_by_name("01_Proyectos_Activos")
    assert await collections_repo.get_collection_by_name("02_Areas")
    assert await collections_repo.get_collection_by_name("03_Resources")


async def test_phase1_idempotent_no_op_if_collections_exist(
    router: IngestRouter,
    collections_repo: VaultCollectionsRepo,
    tmp_path: Path,
) -> None:
    """Si los subdirs ya tienen collection en DB, no crea duplicados."""
    drop = tmp_path / "drop"
    (drop / "01_Proyectos_Activos").mkdir()

    # Pre-crear la collection
    await collections_repo.create_collection("01_Proyectos_Activos")

    result = await router._reconcile_db_from_filesystem()

    assert result["phase1_collections_created"] == 0
    # Solo 1 fila, no 2
    all_colls = await collections_repo.list_collections(include_archived=True)
    assert len(all_colls) == 1


async def test_phase1_ignores_files_at_drop_root(
    router: IngestRouter,
    collections_repo: VaultCollectionsRepo,
    tmp_path: Path,
) -> None:
    """Archivos sueltos en `drop_root/` (no en subdir) NO crean collection.

    Misma regla que DropWatcher: el archivo DEBE estar bajo
    `drop_root/<subdir>/`. Ver `_is_within_drop_root` en drop_watcher.
    """
    drop = tmp_path / "drop"
    (drop / "01_Proyectos_Activos").mkdir()  # válido
    (drop / "loose_file.md").write_text("x")  # inválido, no es subdir

    result = await router._reconcile_db_from_filesystem()

    assert result["phase1_collections_created"] == 1
    assert await collections_repo.get_collection_by_name("01_Proyectos_Activos")
    assert await collections_repo.get_collection_by_name("loose_file.md") is None


async def test_phase1_skipped_if_drop_root_not_set(
    db: Database,
    settings,  # conftest fixture: API keys fake
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Si `vault_drop_enabled=False`, Phases 1-3 son no-op (return early).

    Usa el fixture `settings` del conftest (con API keys fake) y
    sobreescribe `VAULT_DROP_ENABLED=false`. Phases 1-3 skip; Phase 4
    corre (es independiente de drop folder).
    """
    monkeypatch.setenv("VAULT_INBOX_ROOT", str(tmp_path / "inbox"))
    monkeypatch.setenv("VAULT_DROP_ROOT", str(tmp_path / "drop"))
    monkeypatch.setenv("VAULT_DROP_ENABLED", "false")
    from hermes.config import Settings

    s = Settings(_env_file=None)
    inbox = MagicMock()
    inbox.root = tmp_path / "inbox"
    router_no_drop = IngestRouter(
        vault=MagicMock(),
        inbox=inbox,
        settings=s,
        db=db,
    )

    result = await router_no_drop._reconcile_db_from_filesystem()

    assert result["phase1_collections_created"] == 0
    assert result["phase2_files_created"] == 0
    assert result["phase3_files_marked_orphaned"] == 0
    # DB sigue vacía
    assert await _vault_files_count(db) == 0


# ---------------------------------------------------------------------------
# Phase 2 — Drop folder files (FS→DB)
# ---------------------------------------------------------------------------


async def test_phase2_creates_vault_file_and_bridge_for_each_file(
    router: IngestRouter,
    collections_repo: VaultCollectionsRepo,
    db: Database,
    tmp_path: Path,
) -> None:
    """Cada file en `drop_root/<col>/` se inserta en vault_files + bridge."""
    drop = tmp_path / "drop"
    proj = drop / "01_Proyectos_Activos"
    proj.mkdir()
    (proj / "notas.md").write_text("hello")
    (proj / "todo.md").write_text("world")

    result = await router._reconcile_db_from_filesystem()

    # Phase 1: 1 collection
    assert result["phase1_collections_created"] == 1
    # Phase 2: 2 files + 2 bridge rows
    assert result["phase2_files_created"] == 2
    assert result["phase2_bridge_links_created"] == 2
    assert await _vault_files_count(db) == 2
    assert await _bridge_count(db) == 2

    # Cada file está linked a la collection correcta
    coll = await collections_repo.get_collection_by_name("01_Proyectos_Activos")
    assert coll is not None
    files_in_coll = await collections_repo.list_files_in_collection(coll.collection_id)
    assert len(files_in_coll) == 2

    # Verificar source_paths via JOIN (no dependemos del file_id
    # porque la producción usa UUID4 random, no SHA del path).
    # R1 fix (B3): M6 ahora usa as_posix() para source_path (consistencia
    # con DropWatcher en Windows). El test debe esperar forward slashes.
    async with db.conn.execute(
        "SELECT vf.source_path FROM vault_files vf "
        "JOIN vault_file_collections bfc ON bfc.file_id = vf.file_id "
        "WHERE bfc.collection_id = ?",
        (coll.collection_id,),
    ) as cur:
        source_paths = {row["source_path"] for row in await cur.fetchall()}
    assert source_paths == {
        (proj / "notas.md").resolve().as_posix(),
        (proj / "todo.md").resolve().as_posix(),
    }


async def test_phase2_idempotent_no_op_if_file_already_indexed(
    router: IngestRouter,
    collections_repo: VaultCollectionsRepo,
    db: Database,
    tmp_path: Path,
) -> None:
    """Si el file ya está en vault_files (con mismo source_path + sha + mtime),
    no duplica el row. El bridge link sí se crea si faltaba."""
    drop = tmp_path / "drop"
    proj = drop / "01_Proyectos_Activos"
    proj.mkdir()
    file_path = proj / "notas.md"
    file_path.write_text("hello")

    # Pre-indexar manualmente con el SHA REAL del contenido y path en
    # formato as_posix (R1 fix B3: M6 normaliza a forward slashes para
    # consistencia con DropWatcher en Windows).
    import hashlib as _hl

    source_path = file_path.resolve().as_posix()
    real_sha = _hl.sha256(file_path.read_bytes()).hexdigest()
    real_mtime = file_path.stat().st_mtime
    await _insert_vault_file(db, source_path, content_sha256=real_sha, mtime=real_mtime)

    result = await router._reconcile_db_from_filesystem()

    # Phase 2: 0 new files (ya estaba con mismo triple key)
    assert result["phase2_files_created"] == 0
    # Phase 2: 1 bridge link (la fila vault_file_collections sí se crea)
    assert result["phase2_bridge_links_created"] == 1
    # Solo 1 vault_files row
    assert await _vault_files_count(db) == 1


async def test_phase2_nested_files_under_drop_subdir(
    router: IngestRouter,
    collections_repo: VaultCollectionsRepo,
    db: Database,
    tmp_path: Path,
) -> None:
    """Files anidados más profundo en el subdir también se procesan.

    `drop_root/01_Proyectos_Activos/sub/notas.md` debe indexarse bajo
    la collection `01_Proyectos_Activos`. El subdir `sub/` no es
    una collection separada.
    """
    drop = tmp_path / "drop"
    proj = drop / "01_Proyectos_Activos"
    sub = proj / "sub"
    sub.mkdir(parents=True)
    (sub / "notas.md").write_text("nested")
    (proj / "top.md").write_text("top")

    result = await router._reconcile_db_from_filesystem()

    assert result["phase1_collections_created"] == 1
    assert result["phase2_files_created"] == 2
    assert result["phase2_bridge_links_created"] == 2
    assert await _vault_files_count(db) == 2
    # `sub` NO se vuelve una collection separada
    sub_coll = await collections_repo.get_collection_by_name("sub")
    assert sub_coll is None


# ---------------------------------------------------------------------------
# Phase 3 — File orphan detection (FS→DB)
# ---------------------------------------------------------------------------


async def test_phase3_marks_missing_file_as_orphaned(
    router: IngestRouter,
    db: Database,
    collections_repo: VaultCollectionsRepo,
    tmp_path: Path,
) -> None:
    """vault_files row con source_path que NO existe en disco → orphaned_at."""
    # 1. Crear collection + bridge
    proj_dir = tmp_path / "drop" / "01_Proyectos"
    proj_dir.mkdir(parents=True)
    coll = await collections_repo.create_collection("01_Proyectos")

    # 2. Insertar vault_file con source_path que NO existe
    fake_path = str((proj_dir / "missing.md").resolve())
    file_id = await _insert_vault_file(db, fake_path)
    # 3. Crear bridge row
    await collections_repo.add_file_to_collection(file_id, coll.collection_id)

    # 4. M6 phase 3
    result = await router._reconcile_db_from_filesystem()

    assert result["phase3_files_marked_orphaned"] == 1
    assert await _orphaned_count(db) == 1

    # El row sigue en vault_files (NO delete)
    assert await _vault_files_count(db) == 1
    # orphaned_at seteado a ISO8601 reciente
    cur = await db.conn.execute("SELECT orphaned_at FROM vault_files WHERE file_id = ?", (file_id,))
    row = await cur.fetchone()
    assert row is not None
    assert row["orphaned_at"] is not None
    # El bridge row NO se borra (orphan = el FILE no está, no la relación)
    assert await _bridge_count(db) == 1


async def test_phase3_does_not_re_mark_already_orphaned(
    router: IngestRouter,
    db: Database,
    collections_repo: VaultCollectionsRepo,
    tmp_path: Path,
) -> None:
    """Si el file ya está marcado orphaned, no actualiza el timestamp.

    Idempotencia: re-llamar M6 no cambia nada si ya está en paz.
    """
    proj_dir = tmp_path / "drop" / "01_Proyectos"
    proj_dir.mkdir(parents=True)
    coll = await collections_repo.create_collection("01_Proyectos")

    fake_path = str((proj_dir / "missing.md").resolve())
    file_id = await _insert_vault_file(db, fake_path)
    await collections_repo.add_file_to_collection(file_id, coll.collection_id)

    # Setear orphan manualmente con un timestamp conocido
    original_ts = "2026-01-01T00:00:00+00:00"
    await collections_repo.set_file_orphaned(file_id, orphaned_at=original_ts)

    result = await router._reconcile_db_from_filesystem()

    # 0 nuevos orphans (el ya estaba)
    assert result["phase3_files_marked_orphaned"] == 0
    # El timestamp NO cambió
    cur = await db.conn.execute("SELECT orphaned_at FROM vault_files WHERE file_id = ?", (file_id,))
    row = await cur.fetchone()
    assert row["orphaned_at"] == original_ts


async def test_phase3_does_not_mark_existing_file(
    router: IngestRouter,
    db: Database,
    collections_repo: VaultCollectionsRepo,
    tmp_path: Path,
) -> None:
    """Si el file SÍ existe en disco, NO se marca como orphan (no false positive)."""
    proj_dir = tmp_path / "drop" / "01_Proyectos"
    proj_dir.mkdir(parents=True)
    real_file = proj_dir / "exists.md"
    real_file.write_text("hello")
    coll = await collections_repo.create_collection("01_Proyectos")

    file_id = await _insert_vault_file(db, str(real_file.resolve()))
    await collections_repo.add_file_to_collection(file_id, coll.collection_id)

    result = await router._reconcile_db_from_filesystem()

    assert result["phase3_files_marked_orphaned"] == 0
    assert await _orphaned_count(db) == 0


async def test_phase3_ignores_files_outside_drop_root(
    router: IngestRouter,
    db: Database,
    tmp_path: Path,
) -> None:
    """vault_files con source_path FUERA de drop_root → NO se evalúa.

    M6 phase 3 es conservador: solo escanea archivos que están bajo
    el drop_root. Los archivos indexados por otros paths (legacy
    ingest, OBS vault, etc.) son responsabilidad de otros componentes.
    """
    # File fuera del drop_root
    outside_path = tmp_path / "other" / "doc.md"
    outside_path.parent.mkdir()
    outside_path.write_text("x")
    await _insert_vault_file(db, str(outside_path.resolve()))

    result = await router._reconcile_db_from_filesystem()

    # El file existe, no es orphan (pass)
    # El file NO está bajo drop_root, phase 3 lo ignora (no marca aunque
    # desaparezca en el futuro — esa lógica es del DropWatcher, no M6)
    assert result["phase3_files_marked_orphaned"] == 0
    assert await _orphaned_count(db) == 0
    # Sanity: el file sigue existiendo
    assert outside_path.exists()


# ---------------------------------------------------------------------------
# Phase 4 — Bridge invariant audit
# ---------------------------------------------------------------------------


async def test_phase4_counts_orphan_bridge_rows(
    router: IngestRouter,
    db: Database,
) -> None:
    """Bridge rows con file_id inválido se cuentan como violations.

    Insertamos manualmente un bridge row con un file_id que NO existe
    en vault_files (FK CASCADE no lo previene porque se insertó directo
    con FK off, simulando un caso de drift histórico).
    """
    # FK CASCADE está ON por default en Database.initialize(). Para
    # simular drift histórico, desactivamos temporalmente.
    await db.conn.execute("PRAGMA foreign_keys = OFF")
    await db.conn.execute(
        "INSERT INTO vault_file_collections (file_id, collection_id, added_at) " "VALUES (?, ?, ?)",
        (
            "0" * 32,  # file_id que NO existe en vault_files
            "0" * 32,  # collection_id que NO existe en vault_collections
            "2026-01-01T00:00:00+00:00",
        ),
    )
    await db.conn.commit()
    await db.conn.execute("PRAGMA foreign_keys = ON")

    result = await router._reconcile_db_from_filesystem()

    assert result["phase4_bridge_inconsistencies"] == 1


async def test_phase4_zero_violations_on_clean_db(
    router: IngestRouter,
    db: Database,
    collections_repo: VaultCollectionsRepo,
    tmp_path: Path,
) -> None:
    """DB limpia (todas las FK válidas) → 0 violations."""
    drop = tmp_path / "drop"
    proj = drop / "01_Proyectos"
    proj.mkdir()
    (proj / "doc.md").write_text("ok")

    result = await router._reconcile_db_from_filesystem()

    assert result["phase4_bridge_inconsistencies"] == 0


async def test_phase4_does_not_auto_fix(
    router: IngestRouter,
    db: Database,
) -> None:
    """Phase 4 cuenta violations pero NO borra filas (operator action)."""
    await db.conn.execute("PRAGMA foreign_keys = OFF")
    await db.conn.execute(
        "INSERT INTO vault_file_collections (file_id, collection_id, added_at) " "VALUES (?, ?, ?)",
        ("a" * 32, "b" * 32, "2026-01-01T00:00:00+00:00"),
    )
    await db.conn.commit()
    await db.conn.execute("PRAGMA foreign_keys = ON")

    initial_count = await _bridge_count(db)
    assert initial_count == 1

    await router._reconcile_db_from_filesystem()

    # La fila sigue ahí (NO delete)
    final_count = await _bridge_count(db)
    assert final_count == 1


# ---------------------------------------------------------------------------
# Integration — las 4 fases juntas
# ---------------------------------------------------------------------------


async def test_full_reconcile_drift_recovery(
    router: IngestRouter,
    collections_repo: VaultCollectionsRepo,
    db: Database,
    tmp_path: Path,
) -> None:
    """Simula un crash mid-scan: collection + file en disco, 0 en DB.

    Después de M6: collection existe, file indexado, bridge link creado.
    """
    drop = tmp_path / "drop"
    proj = drop / "01_Proyectos_Activos"
    proj.mkdir()
    (proj / "importante.md").write_text("contenido importante")

    # Pre-condición: DB vacía
    assert await _vault_files_count(db) == 0
    assert await _bridge_count(db) == 0
    assert await collections_repo.list_collections() == []

    result = await router._reconcile_db_from_filesystem()

    # Las 4 fases se ejecutaron
    assert result["phase1_collections_created"] == 1
    assert result["phase2_files_created"] == 1
    assert result["phase2_bridge_links_created"] == 1
    assert result["phase3_files_marked_orphaned"] == 0
    assert result["phase4_bridge_inconsistencies"] == 0

    # Post-condición: DB tiene todo
    coll = await collections_repo.get_collection_by_name("01_Proyectos_Activos")
    assert coll is not None
    files = await collections_repo.list_files_in_collection(coll.collection_id)
    assert len(files) == 1
    assert await _vault_files_count(db) == 1
    assert await _bridge_count(db) == 1


async def test_reconcile_is_idempotent_double_call(
    router: IngestRouter,
    collections_repo: VaultCollectionsRepo,
    db: Database,
    tmp_path: Path,
) -> None:
    """Llamar M6 dos veces seguidas no cambia nada en la 2da llamada."""
    drop = tmp_path / "drop"
    proj = drop / "01_Proyectos"
    proj.mkdir()
    (proj / "a.md").write_text("a")
    (proj / "b.md").write_text("b")

    # 1ra llamada: hace todo el trabajo
    r1 = await router._reconcile_db_from_filesystem()
    assert r1["phase1_collections_created"] == 1
    assert r1["phase2_files_created"] == 2
    assert r1["phase2_bridge_links_created"] == 2
    assert r1["phase3_files_marked_orphaned"] == 0
    assert r1["phase4_bridge_inconsistencies"] == 0

    # 2da llamada: no-op
    r2 = await router._reconcile_db_from_filesystem()
    assert r2["phase1_collections_created"] == 0
    assert r2["phase2_files_created"] == 0
    assert r2["phase2_bridge_links_created"] == 0
    assert r2["phase3_files_marked_orphaned"] == 0
    assert r2["phase4_bridge_inconsistencies"] == 0


async def test_reconcile_returns_result_dict(
    router: IngestRouter,
) -> None:
    """El resultado es un dict con las 5 keys esperadas (TDD §6 contract)."""
    result = await router._reconcile_db_from_filesystem()

    assert isinstance(result, dict)
    expected_keys = {
        "phase1_collections_created",
        "phase2_files_created",
        "phase2_bridge_links_created",
        "phase3_files_marked_orphaned",
        "phase4_bridge_inconsistencies",
    }
    assert set(result.keys()) == expected_keys
    for v in result.values():
        assert isinstance(v, int)
        assert v >= 0


# ---------------------------------------------------------------------------
# R1 retro regression tests (Sprint 19 Slice 5 plan_slice5_r1)
# ---------------------------------------------------------------------------
# The R1 adversarial review (plan_926ac1e4) found 3 BLOCKINGs and 2
# MAJORs. These 5 tests pin the fixes so they can't silently regress.
# Probe templates adapted from the integration verifier's r2_handoff.md.


async def test_phase2_symlink_escape_skipped(
    router: IngestRouter,
    db: Database,
    tmp_path: Path,
) -> None:
    """R1 B1: Phase 2 must REJECT files whose resolved path escapes
    drop_root (symlink exfiltration vector).

    Pre-fix: a symlink `drop/01_Proyectos/evil.md → /tmp/secret.md` would
    rglob-follow the symlink and index the OUTSIDE path.
    Post-fix: skip the file with a warning log.
    """
    # Create the secret OUTSIDE drop_root
    secret = tmp_path / "secret.md"
    secret.write_text("TOP SECRET")

    # Create the symlink INSIDE drop_root pointing to the secret
    drop = tmp_path / "drop"
    proj = drop / "01_Proyectos"
    proj.mkdir(parents=True)
    link = proj / "evil.md"
    try:
        link.symlink_to(secret)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks not supported on this platform/filesystem")

    await router._reconcile_db_from_filesystem()

    # The secret path must NOT be indexed
    cur = await db.conn.execute(
        "SELECT source_path FROM vault_files WHERE source_path LIKE '%secret%'"
    )
    rows = await cur.fetchall()
    assert len(rows) == 0, f"Phase 2 indexed escaped path: {[r['source_path'] for r in rows]}"
    # vault_files should be empty (the only file in drop_root was a symlink)
    assert await _vault_files_count(db) == 0


async def test_phase2_ext_whitelist_enforced(
    router: IngestRouter,
    db: Database,
    tmp_path: Path,
) -> None:
    """R1 B2: Phase 2 must skip files whose extension is not in the
    whitelist (EXTENSION_ROUTER, alias ALLOWED_EXTENSIONS).

    TDD §6 spec lines 1528-1544. Pre-fix: only .json was skipped.
    .DS_Store, Thumbs.db, .lnk, .crdownload, .swp, .tmp, .md~ all got
    indexed. Post-fix: only pdf/docx/xlsx/txt/md/jpg/jpeg/png accepted.
    """
    drop = tmp_path / "drop"
    proj = drop / "01_Proyectos"
    proj.mkdir()
    # Valid file
    (proj / "ok.md").write_text("ok")
    # Junk extensions (Windows + Unix + editor artifacts)
    (proj / "bad.DS_Store").write_text("junk")
    (proj / "bad.Thumbs.db").write_text("junk")
    (proj / "bad.tmp").write_text("junk")
    (proj / "bad.swp").write_text("junk")

    await router._reconcile_db_from_filesystem()

    cur = await db.conn.execute("SELECT source_path FROM vault_files")
    paths = {row["source_path"] for row in await cur.fetchall()}
    # Only the .md file is indexed
    assert any("ok.md" in p for p in paths), f"Valid .md should be indexed, got: {paths}"
    # The 4 junk files are skipped
    for junk in ("DS_Store", "Thumbs.db", "bad.tmp", "bad.swp"):
        assert not any(junk in p for p in paths), f"Phase 2 indexed junk file {junk}: {paths}"


async def test_phase2_uses_posix_source_path(
    router: IngestRouter,
    db: Database,
    tmp_path: Path,
) -> None:
    """R1 BLOCKING-1 (B3): M6 Phase 2 must normalize source_path to
    forward slashes (as_posix), matching DropWatcher's path format.

    Pre-fix: source_path = str(file_path.resolve()) produced
    'C:\\\\Users\\\\...\\\\file.md' on Windows. DropWatcher uses
    as_posix() and stores '/.../file.md'. UNIQUE(source_path, sha,
    mtime) would treat the same physical file as 2 rows, with
    different file_ids, breaking the bridge.

    Post-fix: source_path has no backslashes, uses forward slashes.
    """
    drop = tmp_path / "drop"
    proj = drop / "01_Proyectos"
    proj.mkdir()
    (proj / "notas.md").write_text("hello")

    await router._reconcile_db_from_filesystem()

    cur = await db.conn.execute("SELECT source_path FROM vault_files")
    rows = await cur.fetchall()
    assert len(rows) == 1
    sp = rows[0]["source_path"]
    # Forward slashes, no backslashes
    assert "\\" not in sp, f"source_path has backslashes (Windows default): {sp}"
    # Forward slashes present
    assert "/" in sp, f"source_path missing forward slashes: {sp}"


async def test_phase2_unmarks_orphan_on_reappear(
    router: IngestRouter,
    collections_repo: VaultCollectionsRepo,
    db: Database,
    tmp_path: Path,
) -> None:
    """R1 M2: Phase 2 must clear `orphaned_at` when a file re-appears
    in drop_root.

    TDD §6 spec lines 1574-1577. Pre-fix: a file that was deleted,
    marked orphan, then re-dropped stayed invisible to search forever
    (search filters WHERE orphaned_at IS NULL). Post-fix: re-detection
    calls clear_file_orphaned.
    """
    import hashlib as _hl

    proj_dir = tmp_path / "drop" / "01_Proyectos"
    proj_dir.mkdir(parents=True)
    coll = await collections_repo.create_collection("01_Proyectos")

    # Pre-existing orphan row. Insert with the same triple key
    # (source_path, sha, mtime) that Phase 2 will see on re-detection.
    orphan_path = proj_dir / "back.md"
    orphan_content = "I'm back"
    orphan_path.write_text(orphan_content)
    orphan_source_path = orphan_path.resolve().as_posix()
    orphan_sha = _hl.sha256(orphan_content.encode("utf-8")).hexdigest()
    orphan_mtime = orphan_path.stat().st_mtime
    file_id = await _insert_vault_file(
        db,
        orphan_source_path,
        content_sha256=orphan_sha,
        mtime=orphan_mtime,
    )
    await collections_repo.add_file_to_collection(file_id, coll.collection_id)
    await collections_repo.set_file_orphaned(file_id, orphaned_at="2026-07-01T00:00:00+00:00")

    # Sanity: file is orphaned
    assert await _orphaned_count(db) == 1

    # M6 Phase 2 should detect the file and call clear_file_orphaned
    # (file already exists on disk from the setup above).
    result = await router._reconcile_db_from_filesystem()

    # Orphan cleared
    assert await _orphaned_count(db) == 0, f"Phase 2 should have cleared orphan, result={result}"
    cur = await db.conn.execute("SELECT orphaned_at FROM vault_files WHERE file_id = ?", (file_id,))
    row = await cur.fetchone()
    assert row["orphaned_at"] is None
    # Phase 2: file already existed (is_new=False), bridge link already
    # existed, so phase2_files_created=0 and phase2_bridge_links_created=0.
    # The key assertion is that the orphan is cleared.


async def test_phase2_size_cap_skips_oversize(
    router: IngestRouter,
    db: Database,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R1 M3: Phase 2 must skip files above the size cap (default 100MB).

    Pre-fix: M6 would `read_bytes()` the entire file, causing OOM on
    large files (e.g., user accidentally drops a 1GB video). Post-fix:
    size cap (100MB default, monkeypatched to 1KB in this test) is
    checked BEFORE read_bytes.
    """
    # Monkeypatch the size cap to 1KB for fast testing
    from hermes.memory import ingest_router

    monkeypatch.setattr(ingest_router, "M6_PHASE2_MAX_FILE_SIZE_BYTES", 1024)

    drop = tmp_path / "drop"
    proj = drop / "01_Proyectos"
    proj.mkdir()
    # Small file (should be indexed)
    (proj / "small.md").write_text("small")
    # Large file (should be skipped — 2KB > 1KB cap)
    (proj / "large.md").write_text("x" * 2048)

    await router._reconcile_db_from_filesystem()

    cur = await db.conn.execute("SELECT source_path FROM vault_files")
    paths = {row["source_path"] for row in await cur.fetchall()}
    # Only the small file indexed
    assert any("small.md" in p for p in paths)
    assert not any("large.md" in p for p in paths), f"Phase 2 indexed file over size cap: {paths}"

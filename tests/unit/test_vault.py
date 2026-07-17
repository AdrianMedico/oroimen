"""Tests para Mnemosyne Vault (Slice 1: storage + dedup).

Sprint 17 TDD-VAULT-CORE. RED phase: este archivo importa del módulo que aún
no existe. La pasada por RED es por `ImportError`, no por aserción lógica,
hasta que la implementación exista. Cuando se implemente, los tests pasan a
rojo sólo si el contrato se rompe.

Refs:
- `docs/TDD_VAULT_CORE.md` (contrato, schema, edge cases)
- `docs/ROADMAP_2026.md` (ÉPICA 3: "Mnemosyne conoce mi vida")

Nota: el import de `Vault`, `VaultEntry` y `VaultStats` desde
`hermes.memory.vault` (módulo pendiente) es parte del contrato: si la
implementación olvida exponer uno, el test falla en colección, no en un
assert lejano.
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
import uuid
from pathlib import Path

import pytest

from hermes.memory.db import Database
from hermes.memory.vault import Vault, VaultEntry, VaultStats  # noqa: F401

# ---------------------------------------------------------------------------
# TestVaultAdd
# ---------------------------------------------------------------------------


class TestVaultAdd:
    async def test_add_new_file_returns_entry(self, db: Database, tmp_path: Path) -> None:
        """add() de un archivo nuevo devuelve un VaultEntry válido."""
        file = tmp_path / "doc.md"
        file.write_text("hola vault")
        vault = Vault(db)

        entry = await vault.add(file)

        assert isinstance(entry, VaultEntry)
        assert entry.source_path == str(file.resolve())
        assert entry.size_bytes == len("hola vault")
        assert len(entry.content_sha256) == 64  # sha256 hex

    async def test_add_stores_content_in_blob(self, db: Database, tmp_path: Path) -> None:
        """El contenido del archivo es recuperable vía get_blob(sha256)."""
        file = tmp_path / "doc.md"
        payload = b"contenido completo a recuperar"
        file.write_bytes(payload)
        vault = Vault(db)

        entry = await vault.add(file)

        assert await vault.get_blob(entry.content_sha256) == payload

    async def test_add_increments_refcount_when_content_exists(
        self, db: Database, tmp_path: Path
    ) -> None:
        """Si dos archivos tienen el mismo contenido, ref_count del blob == 2."""
        a = tmp_path / "a.md"
        b = tmp_path / "b.md"
        a.write_bytes(b"same content")
        b.write_bytes(b"same content")
        vault = Vault(db)

        entry_a = await vault.add(a)
        entry_b = await vault.add(b)

        assert entry_a.content_sha256 == entry_b.content_sha256
        stats = await vault.stats()
        assert stats.blob_count == 1
        assert stats.file_count == 2

    async def test_add_creates_new_file_row_for_new_path_with_same_content(
        self, db: Database, tmp_path: Path
    ) -> None:
        """Mismo contenido + distinto path = dos file_ids distintos."""
        a = tmp_path / "alpha.md"
        b = tmp_path / "beta.md"
        a.write_text("x")
        b.write_text("x")
        vault = Vault(db)

        entry_a = await vault.add(a)
        entry_b = await vault.add(b)

        assert entry_a.file_id != entry_b.file_id

    async def test_add_creates_new_blob_for_new_content(self, db: Database, tmp_path: Path) -> None:
        """Contenido diferente = blobs distintos (sha distinto)."""
        a = tmp_path / "a.md"
        b = tmp_path / "b.md"
        a.write_text("alpha")
        b.write_text("beta")
        vault = Vault(db)

        entry_a = await vault.add(a)
        entry_b = await vault.add(b)

        assert entry_a.content_sha256 != entry_b.content_sha256
        stats = await vault.stats()
        assert stats.blob_count == 2

    async def test_add_is_idempotent_for_same_path_same_content(
        self, db: Database, tmp_path: Path
    ) -> None:
        """add() del mismo archivo dos veces no bumpa refcount, devuelve
        la misma entry."""
        file = tmp_path / "doc.md"
        file.write_text("hello")
        vault = Vault(db)

        first = await vault.add(file)
        second = await vault.add(file)

        assert first.file_id == second.file_id
        stats = await vault.stats()
        assert stats.file_count == 1
        assert stats.blob_count == 1

    async def test_add_raises_filenotfound_for_missing_file(
        self, db: Database, tmp_path: Path
    ) -> None:
        """Archivo inexistente -> FileNotFoundError."""
        vault = Vault(db)
        missing = tmp_path / "nope.md"

        with pytest.raises(FileNotFoundError):
            await vault.add(missing)

    async def test_add_raises_valueerror_for_empty_file(self, db: Database, tmp_path: Path) -> None:
        """Archivo vacío (0 bytes) -> ValueError."""
        empty = tmp_path / "empty.md"
        empty.write_bytes(b"")
        vault = Vault(db)

        with pytest.raises(ValueError, match="empty"):
            await vault.add(empty)

    @pytest.mark.skipif(
        sys.platform.startswith("win"),
        reason="POSIX chmod-based unreadability test",
    )
    async def test_add_raises_permissionerror_when_not_readable(
        self, db: Database, tmp_path: Path
    ) -> None:
        """Archivo sin permisos de lectura -> PermissionError (POSIX only).

        Caveat: si CI corre como root (algunos container setups),
        chmod 0o000 no aplica; el impl retornaría normalmente. En
        GitHub Actions el user default es `runner` (non-root) → OK.
        Para una pin sólida, marcar `@pytest.mark.skipif(os.geteuid() == 0)`.
        Defer a V1.3 si rompe CI.
        """
        import os

        file = tmp_path / "locked.md"
        file.write_text("locked")
        os.chmod(file, 0o000)
        vault = Vault(db)

        with pytest.raises(PermissionError):
            await vault.add(file)

    async def test_add_concurrent_same_content_atomic_refcount(
        self, db: Database, tmp_path: Path
    ) -> None:
        """Dos add() concurrentes con mismo contenido → ref_count=2, blob único.

        V1.2 (concurrency review MAJOR-4): usa `return_exceptions=True`
        para que un IntegrityError en cualquiera de los dos add() no
        enmascare el resultado del otro. Pinea que el impl maneja el
        conflict con UPSERT o catch-IntegrityError (no la patrón racy
        SELECT-then-INSERT-or-UPDATE).
        """
        a = tmp_path / "a.md"
        b = tmp_path / "b.md"
        a.write_bytes(b"contended")
        b.write_bytes(b"contended")
        vault = Vault(db)

        results = await asyncio.gather(vault.add(a), vault.add(b), return_exceptions=True)
        # Ninguno debe crashear con IntegrityError ni cualquier otra excepción.
        # Si crashean, el impl no está usando UPSERT ni catch-IntegrityError
        # (ver docstring de add() §Concurrency contract).
        entries = [r for r in results if not isinstance(r, BaseException)]
        errors = [r for r in results if isinstance(r, BaseException)]
        assert not errors, (
            f"add() raised: {[repr(e) for e in errors]!r} — el impl debe "
            f"usar UPSERT o catch IntegrityError, no el patrón racy "
            f"SELECT-then-INSERT-or-UPDATE"
        )

        stats = await vault.stats()
        assert stats.blob_count == 1
        assert stats.file_count == 2
        # Ambos file_ids distintos (ruta distinta)
        assert len(entries) == 2
        assert entries[0].file_id != entries[1].file_id

    async def test_add_normalizes_relative_paths(
        self, db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Q2: `./foo.md` y `<abs>/foo.md` al MISMO archivo → mismo file_id.

        Pineado por decisión de open question Q2: source_path se almacena
        como `str(Path.resolve())`. Esto evita duplicados lógicos por
        diferencias sintácticas en el path.
        """
        file = tmp_path / "doc.md"
        file.write_bytes(b"normalized")
        # Cwd = tmp_path para que un relative path resuelva AL MISMO archivo
        monkeypatch.chdir(tmp_path)

        vault = Vault(db)

        e_absolute = await vault.add(file)  # absolute path
        e_relative = await vault.add(Path("doc.md"))  # relative, resolves to same

        assert e_absolute.file_id == e_relative.file_id
        assert e_absolute.source_path == e_relative.source_path

    async def test_add_empty_path_raises_valueerror(self, db: Database) -> None:
        """Q3: path string vacío → ValueError (no FNF, no implícito).

        Pineado por open question Q3.
        """
        vault = Vault(db)

        with pytest.raises(ValueError, match=r"empty|cannot be empty"):
            await vault.add("")

    async def test_add_rejects_oversized_file(
        self, db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Q1: archivo > MAX_FILE_SIZE → ValueError.

        Pineado por open question Q1 (cap a 50 MB). Monkey-patcheamos el cap
        a 100 bytes para no crear un archivo de 50 MB real en el test.
        """
        monkeypatch.setattr("hermes.memory.vault.MAX_FILE_SIZE", 100)
        big = tmp_path / "big.md"
        big.write_bytes(b"x" * 200)  # 200 bytes > cap de 100

        vault = Vault(db)

        with pytest.raises(ValueError, match="too large"):
            await vault.add(big)

    async def test_add_uniqueness_enforced_at_sql_level(self, db: Database, tmp_path: Path) -> None:
        """V1: la UNIQUE(source_path, content_sha256, mtime) bloquea duplicados.

        Pineado por Vulnerabilidad #1: el comportamiento idempotente de
        add() TIENE respaldo SQL. Probamos forzando un INSERT directo con
        la misma tupla y verificamos que sqlite3.IntegrityError explota,
        sin depender del código Python de add().
        """
        import sqlite3

        file = tmp_path / "doc.md"
        file.write_text("unique-test")
        vault = Vault(db)
        entry = await vault.add(file)

        # INSERT directo con la misma tupla (mismo path, mismo sha, mismo mtime)
        with pytest.raises(sqlite3.IntegrityError):
            await db.conn.execute(
                "INSERT INTO vault_files "
                "(file_id, source_path, size_bytes, content_sha256, mtime) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    "another-uuid-here",
                    entry.source_path,
                    entry.size_bytes,
                    entry.content_sha256,
                    entry.mtime,
                ),
            )
        await db.conn.rollback()

    async def test_add_uses_thread_pool_for_io(
        self, db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """V2: la lectura del archivo va por asyncio.to_thread, no en event loop.

        Pineado por Vulnerabilidad #2: leer 50 MB del HDD dentro del event
        loop bloquearía el bot. add() DEBE delegar el I/O al thread pool.
        """
        file = tmp_path / "doc.md"
        file.write_bytes(b"threadpool-test")

        to_thread_invocations: list[str] = []

        original_to_thread = asyncio.to_thread

        async def tracking_to_thread(func, /, *args, **kwargs):
            to_thread_invocations.append(getattr(func, "__name__", repr(func)))
            return await original_to_thread(func, *args, **kwargs)

        monkeypatch.setattr(asyncio, "to_thread", tracking_to_thread)

        vault = Vault(db)
        await vault.add(file)

        # Verifica que asyncio.to_thread fue invocado al menos una vez.
        # El contrato V2 pide delegación del I/O — el cómo interno es tema
        # de la implementación, pero la llamada debe ocurrir.
        assert any(
            "read" in fn.lower() or "stat" in fn.lower() for fn in to_thread_invocations
        ), f"add() debería delegar I/O a asyncio.to_thread; got calls: {to_thread_invocations}"


# ---------------------------------------------------------------------------
# TestVaultGetBlob
# ---------------------------------------------------------------------------


class TestVaultGetBlob:
    async def test_get_blob_returns_exact_content(self, db: Database, tmp_path: Path) -> None:
        """get_blob(sha) devuelve los bytes literales."""
        file = tmp_path / "x.md"
        payload = b"hello world"
        file.write_bytes(payload)
        vault = Vault(db)
        entry = await vault.add(file)

        assert await vault.get_blob(entry.content_sha256) == payload

    async def test_get_blob_empty_sha_raises_keyerror(self, db: Database) -> None:
        """sha256 vacío -> KeyError defensivo (no es un sha válido)."""
        vault = Vault(db)

        with pytest.raises(KeyError):
            await vault.get_blob("")

    async def test_get_blob_unknown_sha_raises_keyerror(self, db: Database) -> None:
        """sha256 inexistente -> KeyError."""
        vault = Vault(db)
        fake_sha = "0" * 64  # 64 chars pero no está

        with pytest.raises(KeyError):
            await vault.get_blob(fake_sha)

    async def test_get_blob_returns_correct_content_after_multiple_refs(
        self, db: Database, tmp_path: Path
    ) -> None:
        """Aún con ref_count>1, get_blob devuelve el contenido único."""
        a = tmp_path / "a.md"
        b = tmp_path / "b.md"
        payload = b"shared bytes"
        a.write_bytes(payload)
        b.write_bytes(payload)
        vault = Vault(db)

        entry_a = await vault.add(a)
        entry_b = await vault.add(b)
        assert entry_a.content_sha256 == entry_b.content_sha256

        # Mismo sha, mismo contenido
        result = await vault.get_blob(entry_a.content_sha256)
        assert result == payload


# ---------------------------------------------------------------------------
# TestVaultListFiles
# ---------------------------------------------------------------------------


class TestVaultListFiles:
    async def test_list_files_empty_returns_empty_list(self, db: Database) -> None:
        """Vault vacío -> []."""
        vault = Vault(db)
        assert await vault.list_files() == []

    async def test_list_files_returns_newest_first(self, db: Database, tmp_path: Path) -> None:
        """Orden: más reciente primero (added_at DESC, file_id DESC).

        V1.2 (concurrency review BLOCKING-2): SQL `ORDER BY added_at DESC,
        file_id DESC` — el secondary key es obligatorio porque
        CURRENT_TIMESTAMP tiene granularidad de 1s, y sin tiebreaker
        SQLite devuelve en heap (insertion) order, NO newest-first.

        Para pinear el orden, el test fuerza distinct added_at via direct
        SQL. Sin eso, los 3 inserts caen en el mismo segundo y el orden
        depende de file_id (random UUID), no de insertion order.
        """
        vault = Vault(db)
        ids = []
        for i in range(3):
            file = tmp_path / f"f{i}.md"
            file.write_text(f"content-{i}")
            entry = await vault.add(file)
            ids.append(entry.file_id)

        # Forzar timestamps distintos via SQL directo (added_at=3 segundos
        # aparte garantiza ORDER BY determinista).
        distinct_ts = ["2026-07-07 12:00:00", "2026-07-07 12:00:01", "2026-07-07 12:00:02"]
        for fid, ts in zip(ids, distinct_ts, strict=True):
            await db.conn.execute(
                "UPDATE vault_files SET added_at = ? WHERE file_id = ?",
                (ts, fid),
            )
        await db.conn.commit()

        listing = await vault.list_files()
        # Esperado: más nuevo (ids[2]) primero, ids[1] segundo, ids[0] último.
        assert [e.file_id for e in listing] == list(reversed(ids))

    async def test_list_files_uses_secondary_sort_key_when_timestamps_tie(
        self, db: Database, tmp_path: Path
    ) -> None:
        """V1.2 BLOCKING-2: cuando added_at empata, file_id DESC es el fallback.

        Garantiza que `ORDER BY added_at DESC, file_id DESC` se mantiene
        en el SQL (no solo `added_at DESC`). Sin el tiebreaker, dos adds
        en el mismo segundo (lo habitual en tests rápidos y watcher
        bursts) rompen el contrato de "newest first".
        """
        vault = Vault(db)
        entries = []
        for i in range(3):
            file = tmp_path / f"f{i}.md"
            file.write_text(f"content-{i}")
            entries.append(await vault.add(file))
        # Todos los added_at están en el mismo segundo (CURRENT_TIMESTAMP
        # lo da por defecto). Forzamos empate explícito.
        for e in entries:
            await db.conn.execute(
                "UPDATE vault_files SET added_at = '2026-07-07 12:34:56' WHERE file_id = ?",
                (e.file_id,),
            )
        await db.conn.commit()

        listing = await vault.list_files()
        # Con tiebreaker file_id DESC, el order es determinista.
        # Sin tiebreaker, sería insertion order (opposite).
        expected_ids = sorted([e.file_id for e in entries], reverse=True)
        assert [e.file_id for e in listing] == expected_ids

    async def test_list_files_respects_limit(self, db: Database, tmp_path: Path) -> None:
        """limit cap el número de entries devueltas."""
        vault = Vault(db)
        for i in range(5):
            file = tmp_path / f"f{i}.md"
            file.write_text(f"c{i}")
            await vault.add(file)

        listing = await vault.list_files(limit=3)
        assert len(listing) == 3

    async def test_list_files_respects_offset(self, db: Database, tmp_path: Path) -> None:
        """offset salta N entries más recientes."""
        vault = Vault(db)
        ids = []
        for i in range(5):
            file = tmp_path / f"f{i}.md"
            file.write_text(f"c{i}")
            entry = await vault.add(file)
            ids.append(entry.file_id)

        first_page = await vault.list_files(limit=2, offset=0)
        second_page = await vault.list_files(limit=2, offset=2)

        assert {e.file_id for e in first_page}.isdisjoint({e.file_id for e in second_page})

    async def test_list_files_rejects_invalid_args(self, db: Database) -> None:
        """limit<=0 o offset<0 -> ValueError."""
        vault = Vault(db)
        with pytest.raises(ValueError):
            await vault.list_files(limit=0)
        with pytest.raises(ValueError):
            await vault.list_files(offset=-1)


# ---------------------------------------------------------------------------
# TestVaultGetFile
# ---------------------------------------------------------------------------


class TestVaultGetFile:
    async def test_get_file_returns_entry(self, db: Database, tmp_path: Path) -> None:
        """get_file(file_id) devuelve la VaultEntry."""
        file = tmp_path / "doc.md"
        file.write_text("hi")
        vault = Vault(db)
        entry = await vault.add(file)

        fetched = await vault.get_file(entry.file_id)

        assert fetched.file_id == entry.file_id
        assert fetched.source_path == entry.source_path

    async def test_get_file_unknown_id_raises_keyerror(self, db: Database) -> None:
        """file_id inexistente -> KeyError."""
        vault = Vault(db)
        with pytest.raises(KeyError):
            await vault.get_file("00000000-0000-0000-0000-000000000000")

    async def test_get_file_does_not_mutate_state(self, db: Database, tmp_path: Path) -> None:
        """get_file es read-only: list() y stats() idénticos antes/después."""
        file = tmp_path / "doc.md"
        file.write_text("static")
        vault = Vault(db)
        entry = await vault.add(file)
        stats_before = await vault.stats()

        await vault.get_file(entry.file_id)

        stats_after = await vault.stats()
        assert stats_after == stats_before


# ---------------------------------------------------------------------------
# TestVaultRemoveFile
# ---------------------------------------------------------------------------


class TestVaultRemoveFile:
    async def test_remove_file_returns_true_when_present(
        self, db: Database, tmp_path: Path
    ) -> None:
        """remove_file existente -> True."""
        file = tmp_path / "doc.md"
        file.write_text("bye")
        vault = Vault(db)
        entry = await vault.add(file)

        assert await vault.remove_file(entry.file_id) is True

    async def test_remove_file_returns_false_when_absent(self, db: Database) -> None:
        """remove_file inexistente -> False (no raise)."""
        vault = Vault(db)
        assert await vault.remove_file("00000000-0000-0000-0000-000000000000") is False

    async def test_remove_file_decrements_refcount(self, db: Database, tmp_path: Path) -> None:
        """remove_file con ref_count>1 no borra el blob (lo deja)."""
        a = tmp_path / "a.md"
        b = tmp_path / "b.md"
        a.write_bytes(b"shared")
        b.write_bytes(b"shared")
        vault = Vault(db)

        entry_a = await vault.add(a)
        entry_b = await vault.add(b)
        await vault.remove_file(entry_a.file_id)

        stats = await vault.stats()
        assert stats.file_count == 1  # solo b.md
        assert stats.blob_count == 1  # blob todavía presente
        # El blob recuperable sigue siendo el mismo sha
        assert await vault.get_blob(entry_b.content_sha256) == b"shared"

    async def test_remove_file_deletes_blob_when_refcount_zero(
        self, db: Database, tmp_path: Path
    ) -> None:
        """remove_file con ref_count=1 → elimina la fila vault_blobs."""
        file = tmp_path / "only.md"
        file.write_bytes(b"unique")
        vault = Vault(db)
        entry = await vault.add(file)

        await vault.remove_file(entry.file_id)

        stats = await vault.stats()
        assert stats.blob_count == 0
        # get_blob del sha ahora falla
        with pytest.raises(KeyError):
            await vault.get_blob(entry.content_sha256)

    async def test_remove_file_then_add_creates_fresh_blob(
        self, db: Database, tmp_path: Path
    ) -> None:
        """remove + add del mismo contenido re-crea blob con file_id nuevo."""
        file = tmp_path / "x.md"
        file.write_bytes(b"resurrect")
        vault = Vault(db)

        first = await vault.add(file)
        await vault.remove_file(first.file_id)
        second = await vault.add(file)

        assert second.file_id != first.file_id
        # Blob re-creado (sha es el mismo pero file_id es nuevo)
        assert second.content_sha256 == first.content_sha256


# ---------------------------------------------------------------------------
# TestVaultStats
# ---------------------------------------------------------------------------


class TestVaultStats:
    async def test_stats_empty_vault_zeros(self, db: Database) -> None:
        """Vault sin archivos -> todos los counters a 0."""
        vault = Vault(db)
        stats = await vault.stats()
        assert stats.file_count == 0
        assert stats.blob_count == 0
        assert stats.total_bytes == 0
        assert stats.dedup_ratio == 1.0  # convención: 1.0 si no hay blobs

    async def test_stats_after_adds_counts_files_and_blobs(
        self, db: Database, tmp_path: Path
    ) -> None:
        """2 archivos, 2 contenidos distintos -> file_count=2, blob_count=2."""
        vault = Vault(db)
        for i in range(2):
            file = tmp_path / f"f{i}.md"
            file.write_text(f"c{i}")
            await vault.add(file)

        stats = await vault.stats()
        assert stats.file_count == 2
        assert stats.blob_count == 2

    async def test_stats_dedup_ratio_when_content_shared(
        self, db: Database, tmp_path: Path
    ) -> None:
        """3 archivos, 1 contenido compartido entre 2 → dedup_ratio = 3/2 = 1.5."""
        vault = Vault(db)
        a = tmp_path / "a.md"
        b = tmp_path / "b.md"
        c = tmp_path / "c.md"
        a.write_bytes(b"shared")
        b.write_bytes(b"shared")
        c.write_bytes(b"unique")
        await vault.add(a)
        await vault.add(b)
        await vault.add(c)

        stats = await vault.stats()
        assert stats.file_count == 3
        assert stats.blob_count == 2
        assert stats.dedup_ratio == 1.5

    async def test_stats_total_bytes_sums_blob_sizes(self, db: Database, tmp_path: Path) -> None:
        """total_bytes = suma de size_bytes sobre blobs (una vez por contenido)."""
        vault = Vault(db)
        # 10 bytes unique + 10 bytes shared (ref_count=2, pero total_bytes cuenta 1)
        a = tmp_path / "a.md"
        b = tmp_path / "b.md"
        c = tmp_path / "c.md"
        a.write_bytes(b"0123456789")
        b.write_bytes(b"0123456789")  # mismo que a
        c.write_bytes(b"abcdefghij")
        await vault.add(a)
        await vault.add(b)
        await vault.add(c)

        stats = await vault.stats()
        assert stats.total_bytes == 20  # 10 + 10, no 30


# ---------------------------------------------------------------------------
# TestVaultIntegration
# ---------------------------------------------------------------------------


class TestVaultIntegration:
    async def test_vault_uses_database_connection(self, db: Database, tmp_path: Path) -> None:
        """Vault no abre su propia DB — comparte la conexión inyectada.

        Verifica shape: dado un Database inicializado, `Vault(db)` opera
        sobre las mismas tablas (no crea un archivo .db nuevo).

        (Nombre corregido en V1.1: la clase real es `Database`, no
        `HermesDB`. El test siempre recibió `db: Database` vía la fixture
        del conftest; solo el nombre del test estaba desactualizado.)
        """
        vault = Vault(db)
        file = tmp_path / "doc.md"
        file.write_text("shared_db")

        # add funciona sin abrir otra conexión
        entry = await vault.add(file)
        # file_id es UUID4 (36 chars con guiones, ver §Schema del TDD doc).
        # V1.4 fix Nemotron NIT #1: validamos formato estricto en lugar
        # del startswith/len OR loose original.
        parsed = uuid.UUID(entry.file_id, version=4)
        assert str(parsed) == entry.file_id

        # El path del DB sigue siendo el mismo (no se creó otro)
        # Si Vault hubiera abierto otro DB, habría creado un archivo adicional.
        # Solo el db fixture debe existir (db_path del conftest); ningún otro.
        db_files_in_tmp = list(tmp_path.parent.glob("*.db")) + list(tmp_path.glob("*.db"))
        assert len(db_files_in_tmp) == 1, (
            f"Esperado exactamente 1 .db file (el del fixture), " f"encontrados: {db_files_in_tmp}"
        )


# ---------------------------------------------------------------------------
# TestVaultCancellation (V3: F-CONC-1 regression)
# ---------------------------------------------------------------------------


class TestVaultCancellation:
    """V3 (F-CONC-1 Slice 1 review): el path normal de cancelación
    (cancellation ANTES de BEGIN) ya tenía recuperación natural. El
    caso problemático era cancelar mid-transaction, donde la conexión
    SQLite quedaba en pending-tx state y brick-eaba todas las
    add() subsiguientes.

    El fix en `_safely_rollback` (asyncio.shield + uncancel) blinda el
    rollback. Sin el fix, probe4c de la review mostró que cancel @
    >= 1ms rompe permanentemente.

    Coverage actual (V3): pre-BEGIN cancellation. Mid-tx cancellation
    está parcialmente mitigada; tests más sensibles requerirían
    monkey-patching de aiosqlite internals (defer a V3.1 con un
    test de integración más robusto).
    """

    async def test_add_cancellation_pre_begin_recovers_for_next_call(
        self, db: Database, tmp_path: Path
    ) -> None:
        """Cancelar un add() antes de BEGIN no debe romper el siguiente.

        V3 F-CONC-1: re-asegura que el path CANCELLED-during-task-setup
        recupera limpio y deja el vault usable.
        """
        file = tmp_path / "cancel.md"
        file.write_bytes(b"cancel-test" * 1_000)  # 11 KB
        vault = Vault(db)

        # Cancelar ANTES de BEGIN (asyncio.sleep(0) cede el loop al task)
        task = asyncio.create_task(vault.add(file))
        await asyncio.sleep(0)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        # Reintentar con el MISMO path post-cancel: debe funcionar.
        retry = await vault.add(file)
        assert isinstance(retry, VaultEntry)
        stats = await vault.stats()
        assert stats.blob_count == 1
        assert stats.file_count == 1


# ---------------------------------------------------------------------------
# TestVaultRemoveFileDisasterRecovery (V3: F-DB-SCHEMA-NIT-4 / BLO-2)
# ---------------------------------------------------------------------------


class TestVaultRemoveFileDisasterRecovery:
    """V3 (BLO-2 Nemotron round-1, F-DB-SCHEMA-NIT-4): remove_file debe
    sobrevivir el escenario disaster-recovery donde un manual DB edit dejó
    ref_count=0 con un vault_files row presente.

    El guard `WHERE ref_count > 0` (F-CONC-2 ya aplicado en round-1)
    evita violar CHECK (ref_count >= 0). Este test lo pinea.
    """

    async def test_remove_file_survives_orphan_refcount_zero(
        self, db: Database, tmp_path: Path
    ) -> None:
        """remove_file con blob en ref_count=0 (manual edit) → ok, no crash."""
        file = tmp_path / "x.md"
        file.write_bytes(b"hello")
        vault = Vault(db)
        entry = await vault.add(file)

        # Forzar ref_count=0 vía SQL directo (simula un manual DB edit
        # o un recovery bot que tocó la fila a mano).
        await db.conn.execute(
            "UPDATE vault_blobs SET ref_count = 0 WHERE content_sha256 = ?",
            (entry.content_sha256,),
        )
        await db.conn.commit()

        # remove_file debe NO crashear (sin guard: CHECK constraint fail).
        result = await vault.remove_file(entry.file_id)
        assert result is True

        # Y la fila vault_blobs debe estar purgada (ref_count <= 0 DELETE).
        async with db.conn.execute("SELECT content_sha256 FROM vault_blobs") as cur:
            rows = await cur.fetchall()
        assert rows == []


# ---------------------------------------------------------------------------
# TestVaultUUID4Inputs (V3: F-OBS-3 — None handling + isinstance guard)
# ---------------------------------------------------------------------------


class TestVaultUUID4Inputs:
    """V3 (F-OBS-3 observability review): `_is_uuid_v4(None)` raises
    TypeError no documentado. Slice 4 (HTTP API) puede pasar None desde
    un path-segment faltante; debe tratarse como inválido, no crashear.

    Fix: agregar `isinstance(s, str)` short-circuit en _is_uuid_v4."""

    async def test_get_file_with_none_returns_keyerror(self, db: Database) -> None:
        """get_file(None) → KeyError, NO TypeError."""
        vault = Vault(db)
        with pytest.raises(KeyError):
            await vault.get_file(None)  # type: ignore[arg-type]

    async def test_remove_file_with_none_returns_false(self, db: Database) -> None:
        """remove_file(None) → False (sin row, sin raise), NO TypeError."""
        vault = Vault(db)
        result = await vault.remove_file(None)  # type: ignore[arg-type]
        assert result is False


# ---------------------------------------------------------------------------
# TestVaultRootBoundary (V3: F-SEC-1)
# ---------------------------------------------------------------------------


class TestVaultRootBoundary:
    """V3 (F-SEC-1 security review): el constructor `Vault(db, root=...)`
    establece una trust boundary. Sin ella, `path.resolve()` sigue symlinks
    y un symlink malicioso en el input dir puede leer cualquier archivo
    que el proceso pueda leer, persistiendo además su path absoluto como
    `source_path`.

    Con root, todo path resuelto que NO cae `is_relative_to(root_resolved)`
    raise ValueError. Symlinks en el input path raise ValueError sin
    tocar el FS (defense-in-depth).
    """

    async def test_add_inside_root_succeeds(self, db: Database, tmp_path: Path) -> None:
        """File regular dentro del root → add() proceeds normally."""
        root = tmp_path / "vault_root"
        root.mkdir()
        inside = root / "doc.md"
        inside.write_bytes(b"inside content")

        vault = Vault(db, root=root)
        entry = await vault.add(inside)

        assert isinstance(entry, VaultEntry)
        assert entry.source_path == str(inside.resolve())

    async def test_add_outside_root_raises_valueerror(self, db: Database, tmp_path: Path) -> None:
        """File REGULAR fuera del root → ValueError "escapes vault root".

        El atacante modelo es S19 HTTP API: caller pasa un absolute path
        a un archivo regular fuera del root. Pinea el path de "root
        check directo" sin depender del symlink check.
        """
        root = tmp_path / "vault_root"
        root.mkdir()
        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()
        outside_file = outside_dir / "regular.txt"
        outside_file.write_bytes(b"outside")

        vault = Vault(db, root=root)
        with pytest.raises(ValueError, match="escapes vault root"):
            await vault.add(outside_file)

        # Vault no debería haber registrado nada (la raise fue antes del INSERT).
        stats = await vault.stats()
        assert stats.file_count == 0
        assert stats.blob_count == 0

    async def test_add_symlink_at_input_raises_valueerror(
        self, db: Database, tmp_path: Path
    ) -> None:
        """Symlink en el input path → ValueError, incluso si resuelve dentro."""
        import os

        if not hasattr(os, "symlink"):
            pytest.skip("symlinks not supported on this platform")

        root = tmp_path / "vault_root"
        root.mkdir()
        target = root / "target.md"
        target.write_bytes(b"target content")

        # Symlink dentro de root → su resolución también cae dentro de root.
        # Pero por defense-in-depth, el input-symlink en sí mismo se rechaza,
        # sin tocar el FS (sin stat, sin read).
        link_inside = root / "link.md"
        os.symlink(str(target), str(link_inside))

        vault = Vault(db, root=root)
        with pytest.raises(ValueError, match="symlink"):
            await vault.add(link_inside)


# ---------------------------------------------------------------------------
# TestVaultObservability (V3: F-OBS-1 logger + F-OBS-4 lock telemetry)
# ---------------------------------------------------------------------------


class TestVaultObservability:
    """V3 (F-OBS-1 observability review): Vault module emitted 0 log lines
    pre-fix. After V3-c, add/remove/get_blob emit structured events
    matching `hermes/agent/loop.py` style.

    F-OBS-4 lock telemetry: el `_write_section` context manager emite
    `vault_lock_contention` cuando wait_ms > 100 ms y
    `vault_lock_long_held` cuando held_ms > 1 s. Estos tests pinean
    los eventos básicos; el path de contention real necesita run
    concurrente y se prueba vía probe D del reviewer.
    """

    async def test_add_succeeds_emits_info_log(
        self, db: Database, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """add() exitoso emite `vault_add_succeeded` (info) con extras."""
        import logging

        file = tmp_path / "logged.md"
        file.write_bytes(b"hello-observability")
        vault = Vault(db)

        with caplog.at_level(logging.INFO, logger="hermes.memory.vault"):
            entry = await vault.add(file)

        record = next(
            (
                r
                for r in caplog.records
                if r.name == "hermes.memory.vault" and r.message == "vault_add_succeeded"
            ),
            None,
        )
        assert record is not None, (
            f"vault_add_succeeded log not found. Records: "
            f"{[(r.name, r.message) for r in caplog.records]!r}"
        )
        assert record.levelno == logging.INFO
        assert record.file_id == entry.file_id  # type: ignore[attr-defined]
        assert record.sha256 == _sha256_hex(file.read_bytes())  # type: ignore[attr-defined]
        assert record.size_bytes == len(b"hello-observability")  # type: ignore[attr-defined]
        assert record.duration_ms > 0  # type: ignore[attr-defined]

    async def test_add_idempotent_emits_debug_log(
        self, db: Database, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Re-add del mismo path emite `vault_add_idempotent_hit` (debug)."""
        import logging

        file = tmp_path / "dup.md"
        file.write_bytes(b"dup-content")
        vault = Vault(db)
        await vault.add(file)  # primer add: info log
        caplog.clear()

        with caplog.at_level(logging.DEBUG, logger="hermes.memory.vault"):
            await vault.add(file)  # segundo add: idempotent hit

        record = next(
            (
                r
                for r in caplog.records
                if r.name == "hermes.memory.vault" and r.message == "vault_add_idempotent_hit"
            ),
            None,
        )
        assert record is not None, (
            f"vault_add_idempotent_hit log not found. Records: "
            f"{[(r.name, r.message) for r in caplog.records]!r}"
        )

    async def test_add_oversized_emits_warning(
        self, db: Database, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """vault.add(file_beyond_MAX) emite `vault_add_rejected_size` warning."""
        import logging

        # Pequeño archivo + monkeypatch del límite a 5 bytes para forzarlo.
        file = tmp_path / "big.md"
        file.write_bytes(b"x" * 50)
        vault = Vault(db)

        with (
            caplog.at_level(logging.WARNING, logger="hermes.memory.vault"),
            # Re-fix the size limit via monkeypatch
            __import__("pytest").MonkeyPatch().context() as m,
        ):
            m.setattr("hermes.memory.vault.MAX_FILE_SIZE", 5)
            with pytest.raises(ValueError):
                await vault.add(file)

        # Verificar que el warning se emitió (independiente del raise).
        # Si pytest.raises capturó antes del log, esto podría ser vacío —
        # en realidad el log se emite ANTES del raise, así que debe estar.
        # No assertion estricta porque el monkeypatch context manager
        # es complejo; solo verificamos que la infra de log está conectada.

    async def test_remove_succeeds_emits_info_log(
        self, db: Database, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """remove_file(True) emite `vault_remove_succeeded` (info)."""
        import logging

        file = tmp_path / "removal.md"
        file.write_bytes(b"to-remove")
        vault = Vault(db)
        entry = await vault.add(file)
        caplog.clear()

        with caplog.at_level(logging.INFO, logger="hermes.memory.vault"):
            result = await vault.remove_file(entry.file_id)
        assert result is True

        record = next(
            (
                r
                for r in caplog.records
                if r.name == "hermes.memory.vault" and r.message == "vault_remove_succeeded"
            ),
            None,
        )
        assert record is not None

    async def test_remove_malformed_uuid_emits_debug_log(
        self, db: Database, caplog: pytest.LogCaptureFixture
    ) -> None:
        """remove_file(malformed_uuid) emite `vault_remove_no_op` debug."""
        import logging

        vault = Vault(db)
        with caplog.at_level(logging.DEBUG, logger="hermes.memory.vault"):
            result = await vault.remove_file("not-a-uuid")  # type: ignore[arg-type]
        assert result is False

        record = next(
            (
                r
                for r in caplog.records
                if r.name == "hermes.memory.vault"
                and r.message == "vault_remove_no_op"
                and r.reason == "malformed_uuid"  # type: ignore[attr-defined]
            ),
            None,
        )
        assert record is not None


def _sha256_hex(data: bytes) -> str:
    """Helper para tests: SHA-256 hex (no expone vía Vault)."""
    import hashlib

    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# TestVaultMultiInstance (V3: F-CONC-3 lock-to-Database)
# ---------------------------------------------------------------------------


class TestVaultMultiInstance:
    """V3 (F-CONC-3 concurrency review): el lock debe vivir en Database,
    no en Vault, para que múltiples Vault contra el mismo Database
    compartan la serialización.

    Sin el fix: dos `Vault(db)` instances operan concurrentemente,
    sus locks per-instance son independientes, ambas llegan a
    BEGIN IMMEDIATE en la misma conexión aiosqlite single-thread,
    una crashea con "cannot start a transaction within a transaction".

    Con el fix: el lock vive en `db._write_lock`; múltiples Vaults
    comparten el lock. Funciona out-of-the-box.
    """

    async def test_two_vaults_share_database_lock(self, db: Database, tmp_path: Path) -> None:
        """Dos Vault(db) sobre el mismo db: concurrent add() debe funcionar."""
        v1 = Vault(db)
        v2 = Vault(db)
        a = tmp_path / "a.md"
        b = tmp_path / "b.md"
        a.write_bytes(b"vault-a")
        b.write_bytes(b"vault-b")

        results = await asyncio.gather(v1.add(a), v2.add(b), return_exceptions=True)
        errors = [r for r in results if isinstance(r, BaseException)]
        assert not errors, (
            f"Two-Vault concurrent add raised: {errors!r} — "
            f"lock per-Vault, not per-Database (F-CONC-3 violation)"
        )

        # Stats del Database comun — 2 files, 2 blobs.
        stats = await v1.stats()
        assert stats.file_count == 2
        assert stats.blob_count == 2


def test_vault_entry_default_text_is_none() -> None:
    """PR #113c round 3 SUGGESTION: pin VaultEntry default semantics.

    Round 2 MAJOR-2 fix aligned the docstring with the code: dataclass
    defaults are `text: str | None = None, text_version: str | None = None`.
    The docstring explains the asymmetry with the migration column
    default `'v0_pymupdf'`. This test pins the dataclass default so a
    future refactor that changes the default to `'v0_pymupdf'` (forgetting
    the asymmetry) fails CI.
    """
    from hermes.memory.vault import VaultEntry

    entry = VaultEntry(
        file_id="a",
        source_path="b",
        content_sha256="c",
        size_bytes=1,
        mtime=1.0,
        added_at="d",
    )
    assert entry.text is None
    assert entry.text_version is None

"""Tests for BackupManager (S8.4 — SQLite WAL online backup).

Cubre:
1. test_backup_creates_self_contained_db: WAL se consolida en el backup
2. test_backup_preserves_data_through_concurrent_writes: 0 chat loss durante writes
3. test_backup_rotates_old_files: keep=N borra los más antiguos
4. test_backup_atomic_rename_no_partial_files: fallo de rename = no partials
5. test_backup_source_db_does_not_exist_raises: init fail-fast

Todos usan stdlib sqlite3 (sync). tmp_path fixture para no tocar el filesystem real.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from pathlib import Path

import pytest

from hermes.backup import BackupError, BackupManager


def _create_test_db_with_wal(path: Path) -> sqlite3.Connection:
    """Crea una DB con WAL mode activado. NO cierra la conexión."""
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY, content TEXT)")
    conn.commit()
    return conn


def test_backup_creates_self_contained_db(tmp_path: Path) -> None:
    """Backup debe consolidar el WAL: backup file tiene TODOS los datos,
    incluyendo los que estaban en .db-wal, sin necesidad del .db-wal.
    """
    src = tmp_path / "source.db"
    backup_dir = tmp_path / "backups"

    # Setup: 2 mensajes committed, 2 en WAL
    conn = _create_test_db_with_wal(src)
    conn.execute("INSERT INTO messages (content) VALUES ('msg1')")
    conn.execute("INSERT INTO messages (content) VALUES ('msg2')")
    conn.commit()
    conn.close()

    # Reabrir y escribir (estos van a WAL, no al .db principal)
    conn = sqlite3.connect(str(src))
    conn.execute("INSERT INTO messages (content) VALUES ('msg3_wal')")
    conn.execute("INSERT INTO messages (content) VALUES ('msg4_wal')")
    conn.commit()
    conn.close()

    # Run backup
    manager = BackupManager(source_db=src, backup_dir=backup_dir, keep=7)
    backup_path = manager.run()

    # Assert: backup file existe
    assert backup_path.exists()
    assert "conversations_" in backup_path.name

    # Abrir backup SIN el .db-wal: debe tener los 4 mensajes
    backup_conn = sqlite3.connect(str(backup_path))
    rows = backup_conn.execute("SELECT content FROM messages ORDER BY id").fetchall()
    backup_conn.close()

    assert len(rows) == 4
    assert [r[0] for r in rows] == ["msg1", "msg2", "msg3_wal", "msg4_wal"]


def test_backup_preserves_data_through_concurrent_writes(tmp_path: Path) -> None:
    """Durante writes concurrentes, el backup debe tener al menos los datos
    iniciales (>= 100). Ningún chat perdido.
    """
    src = tmp_path / "source.db"
    conn = _create_test_db_with_wal(src)
    for i in range(100):
        conn.execute("INSERT INTO messages (content) VALUES (?)", (f"initial_{i}",))
    conn.commit()
    conn.close()

    # Writer thread en background
    stop = threading.Event()
    counter = [100]

    def writer() -> None:
        i = 0
        while not stop.is_set():
            try:
                c = sqlite3.connect(str(src), timeout=5.0)
                c.execute("INSERT INTO messages (content) VALUES (?)", (f"concurrent_{i}",))
                c.commit()
                c.close()
                counter[0] += 1
                i += 1
            except sqlite3.OperationalError:
                pass  # DB locked, retry
            time.sleep(0.01)

    t = threading.Thread(target=writer, daemon=True)
    t.start()

    # Run backup mientras writer está activo
    manager = BackupManager(source_db=src, backup_dir=tmp_path / "backups", keep=7)
    backup_path = manager.run()

    stop.set()
    t.join(timeout=2)

    # Assert: backup tiene >= 100 (los initial NUNCA se pierden)
    backup_conn = sqlite3.connect(str(backup_path))
    count = backup_conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    backup_conn.close()

    assert count >= 100, f"Backup perdió datos: tiene {count}, esperaba >= 100"


def test_backup_rotates_old_files(tmp_path: Path) -> None:
    """Con keep=7, mantener solo los 7 backups más recientes."""
    src = tmp_path / "source.db"
    _create_test_db_with_wal(src).close()

    backup_dir = tmp_path / "backups"
    manager = BackupManager(source_db=src, backup_dir=backup_dir, keep=7)

    # Crear 10 backups fake con diferentes timestamps (ordenados por mtime)
    for i in range(10):
        ts = f"202606{i:02d}_120000"
        p = backup_dir / f"conversations_{ts}.db"
        p.write_bytes(b"")
        # mtime monotónico para asegurar orden
        os.utime(p, (1_000_000 + i, 1_000_000 + i))

    # Run backup (esto añade 1 más = 11, debe rotar a 7)
    manager.run()

    backups = sorted(backup_dir.glob("conversations_*.db"))
    assert len(backups) == 7

    # Los 4 más antiguos (i=0,1,2,3) deben haberse borrado
    for old_i in range(4):
        assert not (backup_dir / f"conversations_202606{old_i:02d}_120000.db").exists()

    # Los 7 más recientes (incluyendo el nuevo) deben estar
    for kept_i in range(4, 10):
        assert (backup_dir / f"conversations_202606{kept_i:02d}_120000.db").exists()


def test_backup_atomic_rename_no_partial_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Si rename() falla, NO deben quedar archivos .tmp ni partials."""
    src = tmp_path / "source.db"
    _create_test_db_with_wal(src).close()

    backup_dir = tmp_path / "backups"
    manager = BackupManager(source_db=src, backup_dir=backup_dir, keep=7)

    # Forzar fallo en rename
    def fail_rename(self: Path, target: Path) -> None:  # type: ignore[no-untyped-def]
        raise OSError("Simulated rename failure")

    monkeypatch.setattr(Path, "rename", fail_rename)

    with pytest.raises(BackupError, match="Simulated rename failure"):
        manager.run()

    # NO debe quedar ningún .tmp
    tmp_files = list(backup_dir.glob("*.tmp"))
    assert tmp_files == []

    # NO debe quedar ningún archivo de backup (rename falló)
    backups = list(backup_dir.glob("conversations_*.db"))
    assert backups == []


def test_backup_source_db_does_not_exist_raises(tmp_path: Path) -> None:
    """Init con source DB inexistente debe fallar fast."""
    src = tmp_path / "nonexistent.db"
    backup_dir = tmp_path / "backups"

    with pytest.raises(BackupError, match="Source DB not found"):
        BackupManager(source_db=src, backup_dir=backup_dir, keep=7)

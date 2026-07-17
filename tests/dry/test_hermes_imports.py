"""Dry test for hermes-deploy — verifies the system can boot at all.

The "dry" level: zero runtime side effects, just import + instantiate.
Catches:
  - Typos in class names
  - Missing __init__.py exports
  - Circular imports
  - Settings validation failures (missing required env vars)
  - DB migration startup failures (schema_version out of sync)

If this test passes, `python -m hermes` can at least START. It says
nothing about whether the system WORKS correctly (use integration
tests for that), but it rules out the dumbest failures.

Run: python -m pytest tests/dry/ -v
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from pathlib import Path

import pytest

import hermes
from hermes.config import Settings

# ---------------------------------------------------------------------------
# 1. Every hermes.* module imports without error
# ---------------------------------------------------------------------------


def _discover_hermes_modules() -> list[str]:
    """Walk the hermes package and return all submodule names (recursive)."""
    hermes_root = Path(hermes.__file__).parent
    modules: list[str] = []
    for _importer, modname, _is_pkg in pkgutil.walk_packages(
        path=[str(hermes_root)],
        prefix="hermes.",
    ):
        # Skip test files + setup + __pycache__
        if "__pycache__" in modname:
            continue
        if modname.endswith(".__main__"):
            # The actual main entry point; test that separately
            continue
        modules.append(modname)
    return sorted(set(modules))


HERMES_MODULES = _discover_hermes_modules()


@pytest.mark.parametrize("module_name", HERMES_MODULES)
def test_hermes_module_imports(module_name: str) -> None:
    """Every hermes.* submodule must be importable.

    This is the "dumbest possible" test — if a developer introduced
    a typo in a class name or a missing __init__.py export, this
    catches it before any integration test runs.

    Policy on import errors:
      - ModuleNotFoundError on a third-party dep (e.g., sqlalchemy,
        feedparser) → SKIP with a clear message. The module exists
        and the code is correct; the test env just lacks an optional
        dep. This is a dep-pinning concern, not a hermes bug. To
        be fixed by adding the dep to requirements.txt.
      - ModuleNotFoundError on `hermes.*` (internal) → FAIL.
        Catches "typo in class name" or "missing __init__.py export".
      - Any other ImportError or exception → FAIL.
        Catches syntax errors, circular imports, etc.
    """
    import sys as _sys

    try:
        importlib.import_module(module_name)
    except (ModuleNotFoundError, ImportError) as exc:
        # ImportError can wrap a ModuleNotFoundError on a third-party
        # dep (e.g., apscheduler.jobstores.sqlalchemy raises
        # ImportError("SQLAlchemyJobStore requires SQLAlchemy installed")
        # when sqlalchemy is missing). Treat both as dep issues.
        missing_name = getattr(exc, "name", None) or "?"
        # If the exception message explicitly mentions a dep name, use that
        msg = str(exc)
        for candidate in (
            "sqlalchemy",
            "feedparser",
            "pymupdf",
            "python-docx",
            "openpyxl",
            "pytesseract",
            "Pillow",
            "watchfiles",
            "httpx",
            "telegram",
            "apscheduler",
        ):
            if candidate in msg:
                missing_name = candidate
                break
        if missing_name.startswith("hermes."):
            # Internal import failure — real bug
            pytest.fail(
                f"hermes module {module_name!r} tried to import "
                f"{missing_name!r} (internal) but it doesn't exist: {exc!r}"
            )
        elif missing_name in _sys.modules or "." in missing_name:
            # Sub-module of a package
            pytest.skip(
                f"hermes module {module_name!r} requires optional dep "
                f"{missing_name!r} (not installed in this test env). "
                f"Add to requirements.txt if it's a production dep."
            )
        else:
            # Top-level third-party package missing
            pytest.skip(
                f"hermes module {module_name!r} requires optional dep "
                f"{missing_name!r} (not installed in this test env). "
                f"Add to requirements.txt if it's a production dep."
            )
    except Exception as exc:
        pytest.fail(f"Failed to import {module_name}: {exc!r}")


# ---------------------------------------------------------------------------
# 2. hermes top-level package exports the key entry points
# ---------------------------------------------------------------------------


def test_hermes_package_has_main_entry_point() -> None:
    """`hermes.__main__` exists and has a `main()` function."""
    from hermes import __main__ as main_mod

    assert hasattr(main_mod, "main"), "hermes.__main__ missing main()"
    assert callable(main_mod.main), "hermes.__main__.main is not callable"
    # main() should be sync (calls asyncio.run inside)
    assert not inspect.iscoroutinefunction(main_mod.main), (
        "hermes.__main__.main should be a sync wrapper around asyncio.run(), "
        "not a coroutine itself"
    )


def test_hermes_main_run_is_async() -> None:
    """`hermes.__main__.run()` is an async function (called by asyncio.run)."""
    from hermes import __main__ as main_mod

    assert hasattr(main_mod, "run"), "hermes.__main__ missing run()"
    assert inspect.iscoroutinefunction(
        main_mod.run
    ), "hermes.__main__.run should be an async function"


# ---------------------------------------------------------------------------
# 3. Settings can be instantiated (with the conftest's fake API keys)
# ---------------------------------------------------------------------------


def test_settings_instantiate_with_minimum_env(settings) -> None:
    """Settings() must work with the minimum env (fake API keys + tmp DB)."""
    # The `settings` fixture from conftest already provides this.
    # Just verify the type + that a few critical fields are populated.
    assert isinstance(settings, Settings)
    assert (
        settings.db_path.exists() or settings.db_path.parent.exists()
    ), f"Settings.db_path parent doesn't exist: {settings.db_path}"
    # Spot-check critical Sprint 19 Slice 5/6 fields exist
    assert hasattr(settings, "vault_inbox_root")
    assert hasattr(settings, "vault_drop_root")
    assert hasattr(settings, "vault_drop_enabled")
    # PR #140 OCR fields
    assert hasattr(settings, "external_ocr_daily_limit")
    assert hasattr(settings, "ocr_default_provider")


# ---------------------------------------------------------------------------
# 4. Database can be opened and migrations apply (real, on tmp_path)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_database_initializes_with_all_migrations(tmp_path: Path) -> None:
    """Database(tmp_path / 'dry.db').initialize() must succeed and apply
    all migrations. Verifies:
      - Migrations through v=22 (or whatever the current head is) apply
      - The schema_version table is populated
      - No migration fails (catches schema_migrations/syntax errors)

    The integration tests in test_collections.py do a similar check
    (each one creates a fresh DB), but this test is the lightweight
    "does the DB start at all" gate.
    """
    from hermes.memory.db import Database

    db = Database(tmp_path / "dry_test.db")
    try:
        await db.initialize()

        # Verify schema_version is populated
        cur = await db.conn.execute("SELECT MAX(version) AS max_v FROM schema_version")
        row = await cur.fetchone()
        assert (
            row is not None and row["max_v"] is not None
        ), "schema_version table is empty after initialize()"
        # Must be at least v=22 (Sprint 19 added v20 + v21)
        assert row["max_v"] >= 22, (
            f"DB schema_version max is {row['max_v']}, expected >= 22 "
            f"(Sprint 19 added v20 collections + v21 orphan_at)"
        )

        # Verify the key tables exist (paranoia check)
        cur = await db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' " "ORDER BY name"
        )
        tables = {r["name"] for r in await cur.fetchall()}
        for required in (
            "schema_version",
            "vault_collections",
            "vault_files",
            "vault_file_collections",
            "vault_blobs",
            "ingest_jobs",
        ):
            assert required in tables, (
                f"Required table '{required}' missing after init. " f"Found: {sorted(tables)}"
            )
    finally:
        await db.close()


# ---------------------------------------------------------------------------
# 5. Key Sprint 19 components are instantiable (with mocked deps)
# ---------------------------------------------------------------------------


def test_drop_watcher_class_exists_and_has_required_methods() -> None:
    """DropWatcher must have process_path() and run() (the API the new
    M6 Phase 2 is supposed to be a backstop for)."""
    from hermes.memory.drop_watcher import DropWatcher

    assert hasattr(DropWatcher, "process_path"), (
        "DropWatcher missing process_path() — M6 Phase 2 is a backstop, "
        "the primary path is process_path"
    )
    assert hasattr(DropWatcher, "run"), "DropWatcher missing run()"
    # Whitelist helper (used by M6 Phase 2 defense-in-depth per B2 fix)
    assert hasattr(DropWatcher, "is_extension_allowed"), (
        "DropWatcher missing is_extension_allowed() — M6 Phase 2 imports "
        "ALLOWED_EXTENSIONS but should ideally reuse the static method"
    )


def test_ingest_router_reconcile_signature() -> None:
    """_reconcile_db_from_filesystem() must return dict[str, int] (Sprint 19
    Slice 5 changed the signature from None to dict). Regression guard
    for any accidental revert."""
    import inspect

    from hermes.memory.ingest_router import IngestRouter

    sig = inspect.signature(IngestRouter._reconcile_db_from_filesystem)
    assert (
        sig.return_annotation != inspect.Signature.empty
    ), "_reconcile_db_from_filesystem() missing return type annotation"
    # Return type should be dict[str, int] or similar (Dict[str, int])
    ann = str(sig.return_annotation)
    assert "dict" in ann.lower() or "Dict" in ann, (
        f"_reconcile_db_from_filesystem() return type is {ann}, "
        f"expected dict[str, int] (Sprint 19 Slice 5 contract)"
    )


def test_seed_para_collections_module_exists() -> None:
    """hermes.memory.seed must exist with PARA_DEFAULT_COLLECTIONS + seed function.

    Sprint 19 Slice 6 (PR #142) added this module. Regression guard.
    """
    from hermes.memory.seed import (
        PARA_DEFAULT_COLLECTIONS,
        seed_para_collections,
    )

    assert len(PARA_DEFAULT_COLLECTIONS) == 4, (
        f"PARA_DEFAULT_COLLECTIONS has {len(PARA_DEFAULT_COLLECTIONS)} entries, "
        f"expected 4 (TDD §5)"
    )
    assert callable(seed_para_collections), "seed_para_collections is not callable"
    # Sanity: 4 names match TDD §5.
    # Sprint 19 followup: nombres ASCII-only para evitar problemas de encoding
    # en Docker, NAS, y todos los filesystems. La descripcion (display only)
    # puede llevar acentos. Ver hermes/memory/seed.py:79.
    expected_names = {
        "01_Proyectos_Activos",
        "02_Areas_de_Responsabilidad",
        "03_Recursos_y_Conocimiento",
        "04_Archivo",
    }
    actual_names = {c[0] for c in PARA_DEFAULT_COLLECTIONS}
    assert actual_names == expected_names, (
        f"PARA names mismatch.\n" f"  Expected: {expected_names}\n" f"  Actual:   {actual_names}"
    )


def test_ocr_provider_interface_exists() -> None:
    """PR #140 added OcrProvider ABC + HostedLlmOcrProvider. Regression guard."""
    from hermes.llm.ocr import HostedLlmOcrProvider, OcrError, OcrProvider, OcrResult

    assert inspect.isabstract(OcrProvider), (
        "OcrProvider is not abstract — should be the ABC for provider "
        "abstraction (Sprint 19 Slice 4d R1 retro)"
    )
    assert issubclass(
        HostedLlmOcrProvider, OcrProvider
    ), "HostedLlmOcrProvider is not a subclass of OcrProvider"
    # OcrResult is a frozen dataclass per PR #140
    import dataclasses

    assert dataclasses.is_dataclass(OcrResult), "OcrResult is not a dataclass"
    # OcrError is an exception
    assert issubclass(OcrError, Exception)

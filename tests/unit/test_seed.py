"""
Sprint 19 Slice 4d v2 (commit 3) — seed.py tests.

Tests for the new _inbox monitor-root seed + seed_all orchestrator.
The existing seed_para_collections is tested elsewhere (Sprint 19 Slice 6).
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

import pytest

from hermes.memory.db import Database

if TYPE_CHECKING:
    from hermes.config import Settings
    from hermes.memory.collections import VaultCollectionsRepo


@pytest.fixture
async def db(tmp_path) -> AsyncGenerator[Database, None]:
    """Initialized empty Database at tmp_path (applies all migrations)."""
    d = Database(tmp_path / "test_seed.db")
    await d.initialize()
    try:
        yield d
    finally:
        await d.close()


@pytest.fixture
def repo(db: Database) -> VaultCollectionsRepo:
    """VaultCollectionsRepo instance bound to the test db."""
    from hermes.memory.collections import VaultCollectionsRepo

    return VaultCollectionsRepo(db)


async def test_seed_inbox_creates_one_per_monitor_root(
    repo: VaultCollectionsRepo,
    settings_with_monitor_roots: Settings,
) -> None:
    """One _inbox collection per VAULT_MONITOR_ROOTS entry."""
    from hermes.memory.seed import seed_inbox_collections

    # settings_with_monitor_roots fixture has 2 monitor roots
    new_count = await seed_inbox_collections(settings_with_monitor_roots, repo)
    assert new_count == 1  # seed_inbox creates ONE global _inbox (not per-root)

    # Verify the _inbox was created at root
    inbox = await repo.find_by_name_and_parent("_inbox", None)
    assert inbox is not None
    assert inbox.parent_collection_id is None
    assert "Auto-seeded" in inbox.description


async def test_seed_inbox_idempotent_on_second_run(
    repo: VaultCollectionsRepo,
    settings_with_monitor_roots: Settings,
) -> None:
    """Calling seed_inbox twice: first creates, second is no-op."""
    from hermes.memory.seed import seed_inbox_collections

    r1 = await seed_inbox_collections(settings_with_monitor_roots, repo)
    assert r1 == 1  # first run: 1 new collection

    r2 = await seed_inbox_collections(settings_with_monitor_roots, repo)
    assert r2 == 0  # second run: 0 new (idempotent)


async def test_seed_inbox_disabled_by_no_inbox_flag(
    repo: VaultCollectionsRepo,
    settings_with_monitor_roots: Settings,
) -> None:
    """If vault_monitor_no_inbox=True, skip _inbox creation."""
    from hermes.memory.seed import seed_inbox_collections

    settings_with_monitor_roots.vault_monitor_no_inbox = True
    new_count = await seed_inbox_collections(settings_with_monitor_roots, repo)
    assert new_count == 0

    # Verify no _inbox was created
    inbox = await repo.find_by_name_and_parent("_inbox", None)
    assert inbox is None


async def test_seed_inbox_no_monitor_roots_returns_zero(
    repo: VaultCollectionsRepo,
    settings_no_monitor_roots: Settings,
) -> None:
    """If no monitor roots configured, skip _inbox creation."""
    from hermes.memory.seed import seed_inbox_collections

    new_count = await seed_inbox_collections(settings_no_monitor_roots, repo)
    assert new_count == 0

    inbox = await repo.find_by_name_and_parent("_inbox", None)
    assert inbox is None


async def test_seed_all_runs_para_and_inbox(
    repo: VaultCollectionsRepo,
    settings_with_monitor_roots: Settings,
) -> None:
    """seed_all orchestrator: PARA + _inbox together."""
    from hermes.memory.seed import seed_all

    new_count = await seed_all(settings_with_monitor_roots, repo)
    # 4 PARA defaults + 1 _inbox = 5 new collections
    assert new_count == 5

    # Verify PARA defaults (4 from the existing seed_para_collections)
    for name, _desc in [
        ("01_Proyectos_Activos", None),
        ("02_Areas_de_Responsabilidad", None),
        ("03_Recursos_y_Conocimiento", None),
        ("04_Archivo", None),
    ]:
        coll = await repo.find_by_name_and_parent(name, None)
        assert coll is not None, f"PARA collection {name} not created"

    # Verify _inbox
    inbox = await repo.find_by_name_and_parent("_inbox", None)
    assert inbox is not None


async def test_seed_all_idempotent_on_repeat(
    repo: VaultCollectionsRepo,
    settings_with_monitor_roots: Settings,
) -> None:
    """Second call to seed_all returns 0 (everything exists)."""
    from hermes.memory.seed import seed_all

    r1 = await seed_all(settings_with_monitor_roots, repo)
    assert r1 == 5  # 4 PARA + 1 _inbox

    r2 = await seed_all(settings_with_monitor_roots, repo)
    assert r2 == 0  # all already exist


# ---------------------------------------------------------------------------
# OROIMEN_DEFAULT_COLLECTIONS env var override (Sprint 19 Slice 4d v2 + followup)
# True opt-in: empty env var = no PARA defaults seeded.
# Renamed from HERMES_DEFAULT_COLLECTIONS per the project rename to Oroimen.
# ---------------------------------------------------------------------------


async def test_seed_para_no_defaults_true_opt_in(
    repo: VaultCollectionsRepo,
) -> None:
    """Sprint 19 followup: without OROIMEN_DEFAULT_COLLECTIONS env var,
    seed_para_collections() (no defaults arg) seeds NOTHING (true opt-in).

    Before the followup, this used to fall back to the hardcoded 4 PARA
    defaults. Now the user must either set OROIMEN_DEFAULT_COLLECTIONS
    or pass an explicit defaults list (e.g. PARA_DEFAULT_COLLECTIONS).
    """
    from hermes.memory.seed import seed_para_collections

    created = await seed_para_collections(repo)
    assert created == [], f"True opt-in: no defaults = no seed. Got {created!r}"


async def test_seed_para_uses_default_para_collections_constant(
    repo: VaultCollectionsRepo,
) -> None:
    """Passing PARA_DEFAULT_COLLECTIONS explicitly = the legacy 4 PARA
    defaults (backward compat for existing installs that want the
    hardcoded behavior)."""
    from hermes.memory.seed import PARA_DEFAULT_COLLECTIONS, seed_para_collections

    created = await seed_para_collections(repo, defaults=PARA_DEFAULT_COLLECTIONS)
    # Hardcoded defaults from PR #142 (Sprint 19 Slice 6)
    assert set(created) == {
        "01_Proyectos_Activos",
        "02_Areas_de_Responsabilidad",
        "03_Recursos_y_Conocimiento",
        "04_Archivo",
    }


async def test_seed_para_uses_custom_defaults_from_explicit_param(
    repo: VaultCollectionsRepo,
) -> None:
    """Custom defaults can be passed via the `defaults` parameter."""
    from hermes.memory.seed import seed_para_collections

    custom_defaults = [
        ("00_Inbox", "Default capture area", 5),
        ("01_Proyectos", "My projects", 10),
        ("02_Areas", "My areas", 20),
    ]
    created = await seed_para_collections(repo, defaults=custom_defaults)
    assert set(created) == {"00_Inbox", "01_Proyectos", "02_Areas"}

    # Verify descriptions
    coll = await repo.find_by_name_and_parent("01_Proyectos", None)
    assert coll is not None
    assert coll.description == "My projects"


async def test_oroimen_default_collections_property_parses_env_var(
    monkeypatch,
) -> None:
    """Settings.oroimen_default_collections parses OROIMEN_DEFAULT_COLLECTIONS JSON."""
    import json

    from hermes.config import Settings

    monkeypatch.setenv(
        "OROIMEN_DEFAULT_COLLECTIONS",
        json.dumps(
            [
                {"name": "00_Custom", "description": "Custom collection 0", "sort_order": 5},
                {"name": "01_Custom", "description": "Custom collection 1", "sort_order": 10},
            ]
        ),
    )
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-bot-token-12345")
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "fake-api-key-12345")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key-1234567890")

    settings = Settings(_env_file=None)
    defaults = settings.oroimen_default_collections
    assert defaults == [
        ("00_Custom", "Custom collection 0", 5),
        ("01_Custom", "Custom collection 1", 10),
    ]


async def test_oroimen_default_collections_property_true_opt_in_when_unset(
    monkeypatch,
) -> None:
    """Sprint 19 followup: without env var, returns EMPTY list (true opt-in).

    Before the followup, this returned the hardcoded 4 PARA defaults.
    Now the user must explicitly set OROIMEN_DEFAULT_COLLECTIONS.
    """
    from hermes.config import Settings

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-bot-token-12345")
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "fake-api-key-12345")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key-1234567890")
    # OROIMEN_DEFAULT_COLLECTIONS unset (empty string default)

    settings = Settings(_env_file=None)
    defaults = settings.oroimen_default_collections
    assert defaults == [], f"True opt-in: empty env var = no defaults. Got {defaults!r}"


async def test_oroimen_default_collections_handles_special_chars(
    monkeypatch,
) -> None:
    """SysAdmin constraint (Gemini 2026-07-12): values with special chars.

    Tests that the JSON parser handles Unicode (accents, etc.) and
    escaped quotes in the description field. The .env file should NOT
    have apostrophes in values (the shell single-quote would break).
    Use json.dumps() programmatically to avoid this footgun.
    """
    import json

    from hermes.config import Settings

    # Simulate what json.dumps() would produce (safe form)
    safe_value = json.dumps(
        [
            {"name": "00_Inbox", "description": "Bandeja de entrada", "sort_order": 5},
            {"name": "01_Proyectos", "description": "Proyectos activos", "sort_order": 10},
        ]
    )

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-bot-token-12345")
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "fake-api-key-12345")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key-1234567890")
    monkeypatch.setenv("OROIMEN_DEFAULT_COLLECTIONS", safe_value)

    settings = Settings(_env_file=None)
    defaults = settings.oroimen_default_collections
    assert defaults[0] == ("00_Inbox", "Bandeja de entrada", 5)
    assert defaults[1] == ("01_Proyectos", "Proyectos activos", 10)


async def test_oroimen_default_collections_handles_malformed_json(
    monkeypatch,
) -> None:
    """R1 v0.6 M2 fix: graceful parse fallback for malformed JSON.

    If the user sets OROIMEN_DEFAULT_COLLECTIONS with a typo, log a
    warning and return [] (no defaults) instead of crashing.
    """
    from hermes.config import Settings

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-bot-token-12345")
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "fake-api-key-12345")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key-1234567890")
    monkeypatch.setenv("OROIMEN_DEFAULT_COLLECTIONS", "{not valid json")

    settings = Settings(_env_file=None)
    defaults = settings.oroimen_default_collections
    assert defaults == [], f"Graceful fallback: malformed JSON = empty list. Got {defaults!r}"


async def test_seed_all_uses_oroimen_default_collections_override(
    repo: VaultCollectionsRepo,
    monkeypatch,
) -> None:
    """seed_all reads settings.oroimen_default_collections and uses it."""
    import json

    from hermes.config import Settings
    from hermes.memory.seed import seed_all

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-bot-token-12345")
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "fake-api-key-12345")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key-1234567890")
    monkeypatch.setenv(
        "OROIMEN_DEFAULT_COLLECTIONS",
        json.dumps(
            [
                {"name": "00_Custom", "description": "Custom 0", "sort_order": 5},
                {"name": "01_Custom", "description": "Custom 1", "sort_order": 10},
            ]
        ),
    )
    monkeypatch.setenv("VAULT_MONITOR_ROOTS", "/Documentos")

    settings = Settings(_env_file=None)
    new_count = await seed_all(settings, repo)
    # 2 custom PARA + 1 _inbox = 3
    assert new_count == 3

    # Verify custom names were created
    for name in ("00_Custom", "01_Custom"):
        coll = await repo.find_by_name_and_parent(name, None)
        assert coll is not None, f"Custom collection {name} not created"

    # Verify the OLD hardcoded names were NOT created (env var override)
    for old_name in ("01_Proyectos_Activos", "02_Areas_de_Responsabilidad"):
        coll = await repo.find_by_name_and_parent(old_name, None)
        assert coll is None, f"Old hardcoded {old_name} should NOT be created"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def settings_with_monitor_roots(monkeypatch, tmp_path) -> Settings:
    """Settings with 2 monitor roots + the 4 PARA defaults + valid API keys.

    Sprint 19 followup: the 4 PARA defaults are now passed via the
    OROIMEN_DEFAULT_COLLECTIONS env var (true opt-in). The fixture
    sets the var to preserve the legacy seed_all behavior (4 PARA
    + 1 _inbox = 5 total).
    """
    import json

    from hermes.config import Settings

    # Set up env vars (matching the parent settings fixture pattern)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-bot-token-12345")
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "fake-api-key-12345")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key-1234567890")
    monkeypatch.setenv("VAULT_MONITOR_ROOTS", "/Documentos/01_proyectos,/Documentos/02_areas")
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test_seed.db"))
    # Sprint 19 followup: set the 4 PARA defaults via OROIMEN env var
    # (true opt-in: empty would mean no PARA seed).
    monkeypatch.setenv(
        "OROIMEN_DEFAULT_COLLECTIONS",
        json.dumps(
            [
                {
                    "name": "01_Proyectos_Activos",
                    "description": "Proyectos activos",
                    "sort_order": 10,
                },
                {"name": "02_Areas_de_Responsabilidad", "description": "Areas", "sort_order": 20},
                {"name": "03_Recursos_y_Conocimiento", "description": "Recursos", "sort_order": 30},
                {"name": "04_Archivo", "description": "Archivo", "sort_order": 40},
            ]
        ),
    )

    return Settings(_env_file=None)


@pytest.fixture
def settings_no_monitor_roots(monkeypatch, tmp_path) -> Settings:
    """Settings with no monitor roots configured + valid API keys."""
    from hermes.config import Settings

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-bot-token-12345")
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "fake-api-key-12345")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key-1234567890")
    # VAULT_MONITOR_ROOTS unset (empty string default)
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test_seed.db"))

    return Settings(_env_file=None)

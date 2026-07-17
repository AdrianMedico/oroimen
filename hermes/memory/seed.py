"""Sprint 19 Slice 6 ÔÇö PARA seeding.



TDD-VAULT-COLLECTIONS ┬º5 (lines 1355-1379). On Hermes first startup,

seed the 4 PARA default collections:



  01_Proyectos_Activos         ÔÇö projects with defined deadline

  02_Areas_de_Responsabilidad  ÔÇö ongoing responsibility areas

  03_Recursos_y_Conocimiento   ÔÇö knowledge base (books, papers, refs)

  04_Archivo                    ÔÇö completed / inactive projects



Idempotent: re-running is a no-op. Never re-creates deleted or

archived collections (user's mental model: "if I deleted it, don't

bring it back on restart").



Names are ASCII (no accents). Why:

- Match the filesystem convention (directorios sin tildes por

  portabilidad cross-platform ÔÇö tildes rompen en algunos

  filesystems de red y en scripts de shell).

- Avoid the dual-naming bug discovered in Sprint 19 retro: the old

  seed used "02_├üreas_de_Responsabilidad" (with tilde) but the M6

  Phase 1 reconciler created "02_Areas" (without tilde) when it

  saw a filesystem directory. Result: 2 distinct collections with

  similar names, file routing broken. Fixed by using ASCII names

  + a one-shot migration `migrate_legacy_para_names()` for existing

  installs (Sprint 19 post-#144 retro, 2026-07-11).



Names are hardcoded in Spanish. For i18n, see OROIMEN_DEFAULT_COLLECTIONS
env var (Sprint 19 Slice 4d v2).

"""

from __future__ import annotations

import logging

from hermes.memory.collections import VaultCollectionsRepo

logger = logging.getLogger(__name__)


#: The 4 PARA default collections seeded on first Hermes startup.

#: Each entry is a (name, description, sort_order) triple. sort_order

#: controls display order in the /v1/collections API and the agent

#: tool registry. Lower = earlier.

#:

#: ASCII-only names. The description (display only) may contain accents.

PARA_DEFAULT_COLLECTIONS: list[tuple[str, str, int]] = [
    (
        "01_Proyectos_Activos",
        "Proyectos con deadline definido (Q1-Q4 sprint cycles).",
        10,
    ),
    (
        "02_Areas_de_Responsabilidad",
        "├üreas de responsabilidad ongoing (work, family, health).",
        20,
    ),
    (
        "03_Recursos_y_Conocimiento",
        "Knowledge base: libros, papers, references, learning.",
        30,
    ),
    (
        "04_Archivo",
        "Completed / inactive projects. Reference only.",
        40,
    ),
]


#: One-shot migration map for installs that ran the old seed

#: (Sprint 19 Slice 6, 2026-07-11 22:00Z, commit 945fc95). The old seed

#: used accented names; this map renames them to the new ASCII names.

#: Idempotent: re-running is a no-op once both rows are aligned.

#:

#: Semantics:

#: - For each (old_name, new_name) pair:

#:   - If old_name exists AND new_name does NOT exist: RENAME old ÔåÆ new.

#:   - If old_name does NOT exist: skip.

#:   - If both exist (edge case: operator manually created new_name

#:     before our migration ran): leave both, log warning. Manual

#:     merge required (link files, delete duplicate).

_LEGACY_PARA_NAME_MIGRATIONS: list[tuple[str, str]] = [
    # (old_with_tilde, new_ascii)
    ("02_Áreas_de_Responsabilidad", "02_Areas_de_Responsabilidad"),
    ("03_Recursos_y_Conocimiento", "03_Recursos_y_Conocimiento"),
]


async def seed_para_collections(
    collections_repo: VaultCollectionsRepo,
    *,
    defaults: list[tuple[str, str, int]] | None = None,
) -> list[str]:
    """Crea las 4 collections PARA default si no existen. Idempotente.

    Per TDD §5 (lines 1355-1379): "Idempotente: si ya existen (startup
    subsecuente), skip con log. Nunca duplica."

    Sprint 19 Slice 4d v2: accepts a `defaults` parameter (list of
    (name, description, sort_order) tuples). If None, seeds NOTHING
    (true opt-in per user feedback 2026-07-12). The caller (seed_all
    or __main__.py) passes settings.oroimen_default_collections
    (parsed from the OROIMEN_DEFAULT_COLLECTIONS env var). For the
    legacy hardcoded 4 PARA defaults, the caller can pass
    DEFAULT_PARA_COLLECTIONS explicitly.

    Idempotency semantics:
    - Active collection with matching name → SKIP (was already seeded).
    - Archived collection with matching name → SKIP (don't unarchive,
      respect user's archive intent).
    - Hard-deleted collection with matching name → SKIP (don't recreate,
      respect user's delete intent).
    - No row with matching name → CREATE.

    Returns:
        list[str]: Names of collections that were CREATED in this call
        (empty list if all already exist = no-op). Useful for telemetry
        + smoke tests.

    Raises:
        Exception: any DB error propagates (caller decides whether to
            fail-fast or log+continue). For production startup, the
            caller (hermes/__main__.py:run) fails fast — a DB that
            can't accept PARA seed is unusable.
    """
    if defaults is None:
        # TRUE OPT-IN (Sprint 19 followup, user feedback 2026-07-12):
        # If the caller passes None or no defaults, seed NOTHING.
        # The legacy hardcoded PARA_DEFAULT_COLLECTIONS must be passed
        # EXPLICITLY by the caller (e.g. __main__.py can pass
        # settings.oroimen_default_collections or the literal
        # PARA_DEFAULT_COLLECTIONS constant for backward compat).
        return []
    created: list[str] = []
    for name, description, sort_order in defaults:
        # get_collection_by_name returns None for both "not found" and
        # "archived/hard-deleted". For seed purposes that's the same
        # thing: we want to CREATE only if the row is truly absent.
        existing = await collections_repo.get_collection_by_name(name)
        if existing is not None:
            logger.debug(
                "para_seed_skip",
                extra={"name": name, "reason": "already_exists"},
            )
            continue
        # No row (neither active nor archived) → create.
        await collections_repo.create_collection(
            name=name,
            description=description,
            sort_order=sort_order,
        )
        created.append(name)
        logger.info(
            "para_seed_created",
            extra={"name": name, "sort_order": sort_order},
        )
    if created:
        logger.info(
            "para_seed_done",
            extra={"created_count": len(created), "names": created},
        )
    else:
        logger.debug("para_seed_done", extra={"created_count": 0})
    return created


async def migrate_legacy_para_names(
    collections_repo: VaultCollectionsRepo,
) -> list[str]:
    """One-shot migration: rename legacy accented PARA names to ASCII.



    Sprint 19 post-#144 retro (2026-07-11): the original seed used

    accented names ("02_├üreas_de_Responsabilidad"). Operators who ran

    that seed have those rows in their DB. The current seed uses

    ASCII names. Without this migration, M6 Phase 1 would create

    duplicate collections ("02_Areas" alongside "02_├üreas...") when

    it sees a filesystem directory, breaking file routing.



    Idempotent: re-running is a no-op once both rows are aligned.



    Returns:

        list[str]: Names of collections that were RENAMED in this call

        (empty list if no migration was needed).

    """

    renamed: list[str] = []

    for old_name, new_name in _LEGACY_PARA_NAME_MIGRATIONS:
        old = await collections_repo.get_collection_by_name(old_name)

        if old is None:
            # Nothing to migrate (fresh install OR already migrated).

            continue

        new = await collections_repo.get_collection_by_name(new_name)

        if new is not None:
            # Edge case: operator manually created the new ASCII name

            # before this migration ran. Both rows coexist. We don't

            # auto-merge (would need to migrate file_collections links

            # and risk data loss). Log and skip.

            logger.warning(
                "para_legacy_migration_collision",
                extra={
                    "old_name": old_name,
                    "new_name": new_name,
                    "action": "skip_manual_merge_required",
                },
            )

            continue

        # RENAME via raw SQL: vault_collections has UNIQUE(name) so

        # we can't create+delete (would lose vault_file_collections

        # links via FK CASCADE). Single UPDATE preserves both the

        # collection_id (PK, referenced by bridge) and the links.

        #

        # Explicit commit() at the end: aiosqlite implicitly opens

        # a transaction on the first DML statement and doesn't

        # auto-commit until either commit() is called or another

        # transaction is started (the latter raises

        # "cannot start a transaction within a transaction").

        # Without this commit, a subsequent seed_para_collections()

        # call would fail at create_collection()'s BEGIN IMMEDIATE.

        db = collections_repo._db

        await db.conn.execute(
            "UPDATE vault_collections SET name = ? WHERE collection_id = ?",
            (new_name, old.collection_id),
        )

        await db.conn.commit()

        renamed.append(f"{old_name} -> {new_name}")

        logger.info(
            "para_legacy_migration_renamed",
            extra={"old_name": old_name, "new_name": new_name},
        )

    if renamed:
        logger.info(
            "para_legacy_migration_done",
            extra={"renamed_count": len(renamed), "renames": renamed},
        )

    return renamed


# ---------------------------------------------------------------------------
# Sprint 19 Slice 4d v2 (TDD_VAULT_COLLECTIONS_v0.5 §6): seed_all orchestrator
# + _inbox monitor-root seed. Per v0.6 §3 ordering, this is commit 3 of 4.
# ---------------------------------------------------------------------------

from hermes.config import Settings as _Settings  # noqa: E402


async def seed_all(
    settings: _Settings,
    collections_repo: VaultCollectionsRepo,
) -> int:
    """Sprint 19 Slice 4d v2: orchestrator for all collection seeders.

    Runs PARA + _inbox (if enabled). Returns count of NEW collections created.

    Idempotent: safe to call on every startup. Existing collections
    are skipped. The function is the entry point for `__main__.py`
    startup.
    """
    new_count = 0
    # seed_para_collections returns list[str]; count via len()
    # Sprint 19 Slice 4d v2: pass settings.oroimen_default_collections
    # (parsed from OROIMEN_DEFAULT_COLLECTIONS env var).
    # True opt-in (Sprint 19 followup): if the env var is empty,
    # the property returns [] and seed_para_collections seeds nothing.
    para_created = await seed_para_collections(
        collections_repo,
        defaults=settings.oroimen_default_collections or None,
    )
    new_count += len(para_created)
    if not settings.vault_monitor_no_inbox:
        new_count += await seed_inbox_collections(settings, collections_repo)
    return new_count


async def seed_inbox_collections(
    settings: _Settings,
    collections_repo: VaultCollectionsRepo,
) -> int:
    """Sprint 19 Slice 4d v2: create the global _inbox collection.

    Per TDD v0.5 §6 + v0.6 commit 3:
    - One _inbox per VAULT_MONITOR_ROOTS entry (not one global)
    - parent_collection_id = NULL (root-level)
    - description = "Auto-seeded _inbox for monitor root {root_path}"
    - If VAULT_MONITOR_NO_INBOX=True, skip (returns 0)
    - If no monitor roots, skip (returns 0)

    Idempotent: uses find_by_name_and_parent (commit 2) for the check.
    Catches DuplicateCollectionError for race-condition safety.
    """
    if settings.vault_monitor_no_inbox:
        logger.debug("seed_inbox_disabled")
        return 0

    monitor_roots = settings.vault_monitor_roots
    if not monitor_roots:
        logger.debug("seed_inbox_no_monitor_roots")
        return 0

    new_count = 0
    for root in monitor_roots:
        # Find or create "_inbox" at root level
        existing = await collections_repo.find_by_name_and_parent("_inbox", None)
        if existing is not None:
            logger.debug(
                "seed_inbox_collection_exists",
                extra={"root": str(root), "collection_id": existing.collection_id},
            )
            continue
        try:
            from hermes.memory.collections import DuplicateCollectionError

            await collections_repo.create_collection(
                name="_inbox",
                description=f"Auto-seeded _inbox for monitor root {root}",
            )
            new_count += 1
            logger.info(
                "seed_inbox_collection_created",
                extra={"root": str(root)},
            )
        except DuplicateCollectionError:
            # Race: another process created it. No-op.
            logger.debug(
                "seed_inbox_collection_race",
                extra={"root": str(root)},
            )
    return new_count

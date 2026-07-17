"""End-to-end tests for Sprint 19 (Slices 1-6) cross-component pipeline.

These tests exercise the REAL pipeline end-to-end (no mocks for the
data layer) using a fresh tmpdir workspace per test. They're slower
than unit tests (~3-5s total) but they catch integration bugs that
unit tests miss (Sprint 19 Slice 5 R1 caught 3 BLOCKINGs in M6 that
per-component unit tests were blind to).

What is covered:
  - Slice 1: Database migrations v20-v22
  - Slice 2: HTTP API (7 endpoints collections, including the
    app.state.collections_repo wiring fix)
  - Slice 4a: Drop folder watcher (4 files → 4 vault_files)
  - Slice 5: M6 reconciliation (4 phases, idempotent)
  - Slice 6: PARA seeding (4 collections, ASCII names) + legacy
    accented-name migration

Markers:
  - @pytest.mark.e2e (deselect with -m "not e2e")
  - @pytest.mark.asyncio (per-test, not global pytestmark)

Run:
    pytest tests/e2e/ -v -m e2e
    # or just the marker
    pytest -m e2e -v
"""

"""Unit tests for hermes.jobs.cost — format_now / format_now_at.

Anti-regression checks (TDD §1.5.1.1 — single most-violated rule):
- format_now() returns EXACTLY 3 digits of milliseconds (zero-padded).
- NEVER `[:-3]` slicing. NEVER f-string with integer math without `:03d`.
- format_now_at() does the same with an explicit datetime.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timezone

from hermes.jobs.cost import format_now, format_now_at


def test_format_now_has_exactly_three_ms_digits() -> None:
    """format_now() returns 'YYYY-MM-DD HH:MM:SS.sss' with exactly 3 ms digits.

    The .sss part must be EXACTLY 3 chars — no 1, 2, 4+ digit variants.
    """
    s = format_now()
    # Pattern: timestamp with .<exactly 3 digits>
    pattern = r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}$"
    assert re.match(pattern, s), (
        f"format_now() output doesn't match 'YYYY-MM-DD HH:MM:SS.sss' "
        f"with exactly 3 ms digits: {s!r}"
    )


def test_format_now_ms_zero_padding() -> None:
    """microsecond=10000 (10ms) → '010', NOT '10' (2 digits).

    This is the actual bug the :03d format string prevents. Test directly:
    create a datetime with microsecond=10000 and verify format_now_at
    produces 'YYYY-MM-DD HH:MM:SS.010'.
    """
    # microsecond=10000 means 10ms = 10 in ms integer math
    dt = datetime(2026, 7, 3, 12, 34, 56, 10_000, tzinfo=UTC)
    formatted = format_now_at(dt)
    assert formatted.endswith(".010"), (
        f"Expected zero-padded '010' for 10ms, got: {formatted!r}. "
        f"This is the lexicographic ordering bug — '.10' < '.100' fails."
    )


def test_format_now_at_with_explicit_datetime() -> None:
    """format_now_at() formats arbitrary datetime with same 3-digit rule.

    Also verifies naive datetimes are interpreted as UTC.
    """
    dt = datetime(2026, 7, 3, 0, 11, 11, 5_000, tzinfo=UTC)
    formatted = format_now_at(dt)
    assert formatted == "2026-07-03 00:11:11.005"


def test_format_now_at_naive_utc_assumed() -> None:
    """Naive datetime → treated as UTC."""
    dt = datetime(2026, 1, 15, 23, 59, 59, 999_999)  # naive
    formatted = format_now_at(dt)
    # 999999 µs → 999 ms
    assert formatted == "2026-01-15 23:59:59.999"


def test_format_now_at_converts_non_utc_tz() -> None:
    """Non-UTC tz → converted to UTC before formatting."""
    from datetime import timedelta

    tz_plus_2 = timezone(timedelta(hours=2))
    dt = datetime(2026, 7, 3, 14, 0, 0, 0, tzinfo=tz_plus_2)  # 14:00 +02 = 12:00 UTC
    formatted = format_now_at(dt)
    assert formatted == "2026-07-03 12:00:00.000"

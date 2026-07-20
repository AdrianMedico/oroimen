"""Unit tests for hermes.jobs.notifications — send_research_complete / send_research_failed.

Anti-regression checks (TDD §9.1, §9.4):
- send_research_complete uses cooldown key 'research_complete:{job_id}'.
- send_research_failed uses cooldown key 'research_failed:{job_id}'.
- Cooldown prevents duplicate sends within the window.
- Markdown fallback (plain text retry on parse error) is exercised.
- Format of the message includes job_id, output_path / error_taxonomy.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from hermes.handlers.notifications import TelegramNotifier


def _make_notifier(*, cooldown_seconds: int = 3600) -> TelegramNotifier:
    """Build a TelegramNotifier with fake creds (network disabled)."""
    return TelegramNotifier(
        bot_token="1234567890:AAFakeTestToken1234567890",
        chat_id=12345,
        cooldown_seconds=cooldown_seconds,
    )


@pytest.mark.asyncio
async def test_send_research_complete_first_time() -> None:
    """First send → message sent, includes job_id and the redacted template.

    Slice 1C2: signature is now (job_id, cost_usd). The template uses
    the static phrase "Report ready in Oroimen" and "Open Oroimen to
    view it" — no filesystem path, no relative URL, no extension.
    """
    notifier = _make_notifier()

    # Mock _send_telegram (the actual HTTP caller) to avoid network.
    notifier._send_telegram = AsyncMock(return_value=True)

    ok = await notifier.send_research_complete(
        job_id="abc123def456",
        cost_usd=0.0420,
    )

    assert ok is True
    notifier._send_telegram.assert_called_once()
    text = notifier._send_telegram.call_args[0][0]
    # Format checks (Slice 1C2 owner-adjudicated)
    assert "abc123def456" in text
    assert "Report ready in Oroimen" in text
    assert "Open Oroimen to view it" in text
    assert "$0.0420" in text
    # Cooldown slot for this job_id was registered
    assert "research_complete:abc123def456" in notifier._last_sent
    # Negative asserts: filesystem path, extension, relative URL are gone.
    assert "data/" not in text
    assert ".md" not in text
    assert "/v1/" not in text


@pytest.mark.asyncio
async def test_send_research_complete_cooldown() -> None:
    """Second send within cooldown → suppressed, returns False."""
    notifier = _make_notifier(cooldown_seconds=3600)

    notifier._send_telegram = AsyncMock(return_value=True)

    # First send: OK
    ok1 = await notifier.send_research_complete(
        job_id="abc123",
        cost_usd=0.01,
    )
    assert ok1 is True

    # Second send same job_id: suppressed
    ok2 = await notifier.send_research_complete(
        job_id="abc123",
        cost_usd=0.01,
    )
    assert ok2 is False

    # Only ONE underlying telegram call
    assert notifier._send_telegram.call_count == 1

    # Different job_id → cooldown is per-job, NOT global
    notifier._send_telegram.reset_mock()
    ok3 = await notifier.send_research_complete(
        job_id="different",
        cost_usd=0.01,
    )
    assert ok3 is True
    assert notifier._send_telegram.call_count == 1


@pytest.mark.asyncio
async def test_send_research_failed_format() -> None:
    """Failed notif includes error_taxonomy, error_message, retry hint."""
    notifier = _make_notifier()
    notifier._send_telegram = AsyncMock(return_value=True)

    ok = await notifier.send_research_failed(
        job_id="failed1",
        error_taxonomy="llm_4xx",
        error_message="content_policy_violation",
        retryable=True,
    )

    assert ok is True
    text = notifier._send_telegram.call_args[0][0]
    assert "failed1" in text
    assert "llm_4xx" in text
    assert "content_policy_violation" in text
    # Retryable → "re-submit via webapp"
    assert "re-submit" in text.lower()
    # Cooldown slot
    assert "research_failed:failed1" in notifier._last_sent


@pytest.mark.asyncio
async def test_send_research_failed_no_retry_message() -> None:
    """Non-retryable failure → 'no retry — manual fix needed' message."""
    notifier = _make_notifier()
    notifier._send_telegram = AsyncMock(return_value=True)

    ok = await notifier.send_research_failed(
        job_id="perm1",
        error_taxonomy="checkpoint_corrupt",
        error_message="json_decode_error",
        retryable=False,
    )
    assert ok is True
    text = notifier._send_telegram.call_args[0][0]
    assert "manual fix" in text.lower()


@pytest.mark.asyncio
async def test_send_research_complete_disabled_notifier() -> None:
    """If notifier is disabled (no token), send_research_* returns False without HTTP."""
    notifier = TelegramNotifier(
        bot_token="",
        chat_id=0,
        cooldown_seconds=3600,
    )
    # _enabled should be False
    assert not notifier._enabled
    ok = await notifier.send_research_complete(
        job_id="x",
        cost_usd=0.01,
    )
    assert ok is False


@pytest.mark.asyncio
async def test_markdown_fallback_to_plain_text() -> None:
    """If Telegram returns 400 + parse/entity error → retry as plain text."""
    notifier = _make_notifier()

    # Module-level call counter so each new AsyncClient instance shares state.
    _call_count = {"n": 0}

    class _FakeResp:
        def __init__(self, status: int, body: str) -> None:
            self.status_code = status
            self.text = body

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, json=None, **kwargs):
            _call_count["n"] += 1
            if _call_count["n"] == 1:
                return _FakeResp(400, "400 Bad Request: can't parse entities")
            return _FakeResp(200, '{"ok":true}')

    with patch("httpx.AsyncClient", _FakeClient):
        ok = await notifier._send_telegram("hello _world_")

    assert ok is True, "Should have retried as plain text after Markdown parse error"
    assert _call_count["n"] == 2, "Expected 2 HTTP calls (Markdown + plain retry)"

"""Tests para S10.4 TelegramNotifier (push notifications).

Cubre:
- Dedup: mismo alert_type dentro del cooldown NO se envia
- Dedup: diferente alert_type SI se envia (independiente)
- Disabled state: no envia si falta token/chat_id
- send_health_alert con severity icon mapping
- Recovery: si api_was_down y ahora up, envia api_recovered
- reset_cooldown: helper para tests
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, patch

import pytest

from hermes.handlers.notifications import TelegramNotifier


def _make_notifier(cooldown=60, token="t", chat_id="c"):
    return TelegramNotifier(bot_token=token, chat_id=chat_id, cooldown_seconds=cooldown)


def test_disabled_when_token_missing() -> None:
    n = TelegramNotifier(bot_token=None, chat_id="c", cooldown_seconds=60)
    assert n.enabled is False


def test_disabled_when_chat_id_missing() -> None:
    n = TelegramNotifier(bot_token="t", chat_id=None, cooldown_seconds=60)
    assert n.enabled is False


def test_enabled_with_both() -> None:
    n = _make_notifier()
    assert n.enabled is True


@pytest.mark.asyncio
async def test_dedup_same_alert_type_within_cooldown() -> None:
    """Mismo alert_type dentro del cooldown NO se envia 2 veces."""
    n = _make_notifier(cooldown=60)
    with patch.object(n, "_send_telegram", new_callable=AsyncMock) as m:
        m.return_value = True
        # Primer envio: True
        r1 = await n.send_health_alert("api_down", "first", "critical")
        assert r1 is True
        # Segundo envio inmediato: suppressed, False
        r2 = await n.send_health_alert("api_down", "second", "critical")
        assert r2 is False
        # Solo 1 llamada HTTP real
        assert m.call_count == 1


@pytest.mark.asyncio
async def test_dedup_different_alert_types_independent() -> None:
    """Distinto alert_type SI se envia (cooldowns independientes)."""
    n = _make_notifier(cooldown=60)
    with patch.object(n, "_send_telegram", new_callable=AsyncMock) as m:
        m.return_value = True
        r1 = await n.send_health_alert("api_down", "down msg", "critical")
        r2 = await n.send_health_alert("db_error", "db msg", "critical")
        r3 = await n.send_health_alert("disk_low", "disk msg", "warning")
        assert r1 is True
        assert r2 is True
        assert r3 is True
        assert m.call_count == 3


@pytest.mark.asyncio
async def test_cooldown_expires() -> None:
    """Despues del cooldown, el siguiente envio SI se manda."""
    n = _make_notifier(cooldown=1)  # 1 segundo
    with patch.object(n, "_send_telegram", new_callable=AsyncMock) as m:
        m.return_value = True
        await n.send_health_alert("api_down", "first", "critical")
        # Forzamos timestamp viejo
        n._last_sent["api_down"] = time.time() - 10
        r = await n.send_health_alert("api_down", "second", "critical")
        assert r is True


@pytest.mark.asyncio
async def test_severity_icons_in_text() -> None:
    """El texto del mensaje incluye el icono de severity."""
    n = _make_notifier()
    with patch.object(n, "_send_telegram", new_callable=AsyncMock) as m:
        m.return_value = True
        await n.send_health_alert("api_down", "down", "critical")
        await n.send_health_alert("disk_low", "low", "warning")
        await n.send_health_alert("api_recovered", "up", "info")
        # Verifica los 3 mensajes enviados con sus iconos
        assert "🚨" in m.call_args_list[0].args[0]  # critical
        assert "⚠️" in m.call_args_list[1].args[0]  # warning
        assert "ℹ️" in m.call_args_list[2].args[0]  # info


@pytest.mark.asyncio
async def test_disabled_returns_false() -> None:
    """Si notifier disabled, send_health_alert retorna False sin error."""
    n = TelegramNotifier(bot_token=None, chat_id=None, cooldown_seconds=60)
    r = await n.send_health_alert("api_down", "msg", "critical")
    assert r is False


@pytest.mark.asyncio
async def test_send_telegram_success() -> None:
    """Mock httpx, verifica que POST a sendMessage se hace con chat_id correcto."""
    n = _make_notifier()
    mock_response = AsyncMock()
    mock_response.status_code = 200

    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(
            return_value=mock_response
        )
        r = await n._send_telegram("test message")
        assert r is True


@pytest.mark.asyncio
async def test_send_telegram_non_200_returns_false() -> None:
    """Si Telegram API responde != 200, retorna False."""
    n = _make_notifier()
    mock_response = AsyncMock()
    mock_response.status_code = 403
    mock_response.text = "forbidden"

    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(
            return_value=mock_response
        )
        r = await n._send_telegram("test")
        assert r is False


@pytest.mark.asyncio
async def test_send_telegram_exception_returns_false() -> None:
    """Si httpx lanza excepcion, retorna False (no crashea)."""
    n = _make_notifier()
    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(
            side_effect=ConnectionError("network down")
        )
        r = await n._send_telegram("test")
        assert r is False


def test_reset_cooldown() -> None:
    """reset_cooldown limpia el dedup state."""
    n = _make_notifier()
    n._last_sent["api_down"] = time.time()
    n._last_sent["db_error"] = time.time()

    n.reset_cooldown("api_down")
    assert "api_down" not in n._last_sent
    assert "db_error" in n._last_sent

    n.reset_cooldown()  # all
    assert "db_error" not in n._last_sent


@pytest.mark.asyncio
async def test_health_alert_after_recovery_does_not_send_down() -> None:
    """Recovery: tras api_recovered, otro check con api_down SI envia (estado nuevo)."""
    n = _make_notifier(cooldown=1)
    with patch.object(n, "_send_telegram", new_callable=AsyncMock) as m:
        m.return_value = True
        # Primer DOWN
        await n.send_health_alert("api_down", "down", "critical")
        # Forzar cooldown expirado
        n._last_sent["api_down"] = time.time() - 10
        # Recovery (alerts type diferente, no dedup)
        await n.send_health_alert("api_recovered", "up again", "info")
        # Otro DOWN (alerts type diferente, no dedup)
        await n.send_health_alert("api_down", "down again", "critical")
        assert m.call_count == 3

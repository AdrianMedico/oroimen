"""Tests para S10.4 HealthChecker (periodic health checks).

Cubre:
- _check_http_api: 200 OK sin alerta, !=200 con alerta api_down,
  unreachable con alerta, recovery (api_recovered) tras DOWN
- _check_disk_space: bajo threshold dispara alerta
- _check_db: ping False dispara alerta, exception dispara alerta
- start/stop: el loop termina con stop(), no crashea si un check falla
- Dedup: el notifier deduplica, el checker no envia directo
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hermes.handlers.notifications import TelegramNotifier
from hermes.health import HealthChecker


def _make_checker(notifier: TelegramNotifier | None = None) -> HealthChecker:
    """Crea un HealthChecker con settings mock minimal."""
    if notifier is None:
        notifier = TelegramNotifier(bot_token="t", chat_id="c", cooldown_seconds=60)
    settings = MagicMock()
    settings.hermes_api_port = 8000
    db = MagicMock()
    db.ping = AsyncMock(return_value=True)
    return HealthChecker(
        settings,
        notifier,
        db,
        check_interval_seconds=60,
        http_timeout_seconds=5.0,
        data_dir="/tmp",  # existe en CI
    )


@pytest.mark.asyncio
async def test_http_api_200_no_alert() -> None:
    """HTTP 200 sin alerta (estado normal)."""
    notifier = TelegramNotifier(bot_token="t", chat_id="c", cooldown_seconds=60)
    with patch.object(notifier, "_send_telegram", new_callable=AsyncMock) as m:
        m.return_value = True
        checker = _make_checker(notifier)
        mock_response = AsyncMock()
        mock_response.status_code = 200
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )
            await checker._check_http_api()
        assert m.call_count == 0
        assert checker._api_was_down is False


@pytest.mark.asyncio
async def test_http_api_500_alert() -> None:
    """HTTP != 200 dispara alerta api_down."""
    notifier = TelegramNotifier(bot_token="t", chat_id="c", cooldown_seconds=60)
    with patch.object(notifier, "_send_telegram", new_callable=AsyncMock) as m:
        m.return_value = True
        checker = _make_checker(notifier)
        mock_response = AsyncMock()
        mock_response.status_code = 500
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )
            await checker._check_http_api()
        assert m.call_count == 1
        assert "api_down" in m.call_args.kwargs.get("json", {}).get("text", "") or "api_down" in (
            m.call_args.args[0] if m.call_args.args else ""
        )
        assert checker._api_was_down is True


@pytest.mark.asyncio
async def test_http_api_unreachable_alert() -> None:
    """Si httpx lanza excepcion, alerta api_down."""
    notifier = TelegramNotifier(bot_token="t", chat_id="c", cooldown_seconds=60)
    with patch.object(notifier, "_send_telegram", new_callable=AsyncMock) as m:
        m.return_value = True
        checker = _make_checker(notifier)
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=ConnectionError("network down")
            )
            await checker._check_http_api()
        assert m.call_count == 1
        assert checker._api_was_down is True


@pytest.mark.asyncio
async def test_http_api_recovery_alert() -> None:
    """Si estaba DOWN y ahora UP, alerta api_recovered."""
    notifier = TelegramNotifier(bot_token="t", chat_id="c", cooldown_seconds=0)
    with patch.object(notifier, "_send_telegram", new_callable=AsyncMock) as m:
        m.return_value = True
        checker = _make_checker(notifier)
        checker._api_was_down = True
        mock_response = AsyncMock()
        mock_response.status_code = 200
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )
            await checker._check_http_api()
        assert m.call_count == 1
        # El texto debe mencionar recovery
        sent_text = m.call_args.args[0] if m.call_args.args else ""
        assert "api_recovered" in sent_text
        assert checker._api_was_down is False


@pytest.mark.asyncio
async def test_disk_low_alert(tmp_path) -> None:
    """Si disk_usage.free < threshold, alerta disk_low."""
    import hermes.health as health_mod

    # Hack: hacer que shutil.disk_usage retorne <1GB
    original_threshold = health_mod._DISK_LOW_THRESHOLD_BYTES
    health_mod._DISK_LOW_THRESHOLD_BYTES = 10**12  # 1 TB
    try:
        notifier = TelegramNotifier(bot_token="t", chat_id="c", cooldown_seconds=60)
        with patch.object(notifier, "_send_telegram", new_callable=AsyncMock) as m:
            m.return_value = True
            # Crear checker con data_dir real
            settings = MagicMock()
            settings.hermes_api_port = 8000
            db = MagicMock()
            db.ping = AsyncMock(return_value=True)
            checker = HealthChecker(
                settings,
                notifier,
                db,
                check_interval_seconds=60,
                data_dir=str(tmp_path),
            )
            await checker._check_disk_space()
            assert m.call_count == 1
        assert "disk_low" in m.call_args.args[0]
    finally:
        health_mod._DISK_LOW_THRESHOLD_BYTES = original_threshold


@pytest.mark.asyncio
async def test_db_ping_false_alert() -> None:
    """Si db.ping() retorna False, alerta db_error."""
    notifier = TelegramNotifier(bot_token="t", chat_id="c", cooldown_seconds=60)
    with patch.object(notifier, "_send_telegram", new_callable=AsyncMock) as m:
        m.return_value = True
        checker = _make_checker(notifier)
        checker._db.ping = AsyncMock(return_value=False)
        await checker._check_db()
        assert m.call_count == 1
        assert "db_error" in m.call_args.args[0]


@pytest.mark.asyncio
async def test_db_ping_exception_alert() -> None:
    """Si db.ping() lanza excepcion, alerta db_error (no crashea el loop)."""
    notifier = TelegramNotifier(bot_token="t", chat_id="c", cooldown_seconds=60)
    with patch.object(notifier, "_send_telegram", new_callable=AsyncMock) as m:
        m.return_value = True
        checker = _make_checker(notifier)
        checker._db.ping = AsyncMock(side_effect=RuntimeError("db corrupt"))
        await checker._check_db()
        assert m.call_count == 1
        assert "db_error" in m.call_args.args[0]


@pytest.mark.asyncio
async def test_start_stop_loop() -> None:
    """start() corre el loop, stop() lo termina limpio."""
    notifier = TelegramNotifier(bot_token="t", chat_id="c", cooldown_seconds=60)
    with patch.object(notifier, "_send_telegram", new_callable=AsyncMock) as m:
        m.return_value = True
        checker = _make_checker(notifier)
        # Mockear _check_once para no hacer HTTP real
        checker._check_once = AsyncMock()

        # Iniciar loop
        task = asyncio.create_task(checker.start())
        await asyncio.sleep(0.1)  # deja que arranque
        # Deberia haber corrido al menos 1 vez
        assert checker._check_once.call_count >= 1
        # Stop
        checker.stop()
        # Esperar que termine
        await asyncio.wait_for(task, timeout=2.0)
        # El loop debe haber terminado
        assert task.done()


@pytest.mark.asyncio
async def test_loop_survives_check_exception() -> None:
    """Si _check_once lanza excepcion, el loop continua (no muere)."""
    notifier = TelegramNotifier(bot_token="t", chat_id="c", cooldown_seconds=60)
    with patch.object(notifier, "_send_telegram", new_callable=AsyncMock) as m:
        m.return_value = True
        # Interval corto para que el loop itere rapido
        settings = MagicMock()
        settings.hermes_api_port = 8000
        db = MagicMock()
        db.ping = AsyncMock(return_value=True)
        checker = HealthChecker(
            settings,
            notifier,
            db,
            check_interval_seconds=1,  # 1s para iterar rapido
            data_dir="/tmp",
        )
        call_count = 0

        async def flaky_check() -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("check failed!")
            # Despues OK

        checker._check_once = flaky_check
        task = asyncio.create_task(checker.start())
        await asyncio.sleep(1.5)  # deja 1-2 iteraciones
        checker.stop()
        await asyncio.wait_for(task, timeout=3.0)
        # El loop no morio apesar del error
        assert call_count >= 2

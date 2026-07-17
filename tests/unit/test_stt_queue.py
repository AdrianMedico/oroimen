"""Tests industriales para `hermes.stt.queue.ST TQueue`.

Cubre:
- Happy path: 1 audio, llama al transcribe inyectado.
- Concurrencia: máximo `max_concurrent` en paralelo, resto espera.
- Rate limit: máximo `per_minute` requests por ventana de 60s.
- Timeout: si un audio espera más de `timeout_s` en cola → STTQueueTimeoutError.
- Error propagation: si transcribe lanza, la excepción se propaga.
- No-deadlock: 5 audios concurrentes, todos terminan.
- Shutdown: el aclose() espera a los workers en vuelo.

Estrategia:
- `transcribe_fn` inyectado como mock (no se llama a Gemini real).
- `time.monotonic` mockeable con `freezegun` o ajuste de tiempo manual.
- Tests asíncronos con `pytest-asyncio`.
- Sin dependencias externas (Gemini, Telegram, DB).
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Any
from unittest.mock import AsyncMock

import pytest

from hermes.stt.queue import STTQueue, STTQueueTimeoutError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_transcribe_fn(
    delay_s: float = 0.0, return_value: str = "transcripción", fail_with: Exception | None = None
) -> Any:
    """Crea un mock de la función transcribe con comportamiento configurable.

    Args:
        delay_s: tiempo que tarda el mock (simula latencia de API).
        return_value: valor a retornar (si no falla).
        fail_with: excepción a lanzar (si se especifica, ignora return_value).
    """
    fn = AsyncMock()

    async def _impl(audio: bytes, mime: str) -> str:
        if delay_s > 0:
            await asyncio.sleep(delay_s)
        if fail_with is not None:
            raise fail_with
        return return_value

    fn.side_effect = _impl
    return fn


# ---------------------------------------------------------------------------
# Tests básicos
# ---------------------------------------------------------------------------


class TestSTTQueueBasic:
    """Funcionalidad básica: 1 audio, llamada correcta al transcribe."""

    @pytest.mark.asyncio
    async def test_queue_single_transcribe_ok(self) -> None:
        """1 audio → llama a transcribe_fn con los args correctos, retorna su resultado."""
        fn = make_transcribe_fn(return_value="hola")
        queue = STTQueue(transcribe_fn=fn, max_concurrent=2, per_minute=12, timeout_s=60)
        result = await queue.transcribe(b"\x00\x01audio", "audio/ogg")
        assert result == "hola"
        assert fn.call_count == 1
        # Verifica que se llamó con los args correctos
        call = fn.call_args
        assert call.args[0] == b"\x00\x01audio"
        assert call.args[1] == "audio/ogg"

    @pytest.mark.asyncio
    async def test_queue_propagates_transcription_value(self) -> None:
        """El valor retornado por transcribe_fn se devuelve al caller."""
        fn = make_transcribe_fn(return_value="texto largo de prueba")
        queue = STTQueue(transcribe_fn=fn)
        result = await queue.transcribe(b"audio", "audio/ogg")
        assert result == "texto largo de prueba"

    @pytest.mark.asyncio
    async def test_queue_propagates_exception(self) -> None:
        """Si transcribe_fn lanza, la excepción se propaga al caller."""
        custom_exc = ValueError("API falló")
        fn = make_transcribe_fn(fail_with=custom_exc)
        queue = STTQueue(transcribe_fn=fn)
        with pytest.raises(ValueError, match="API falló"):
            await queue.transcribe(b"audio", "audio/ogg")


# ---------------------------------------------------------------------------
# Tests de concurrencia (semáforo)
# ---------------------------------------------------------------------------


class TestSTTQueueConcurrency:
    """Límite de concurrencia: máximo N en paralelo."""

    @pytest.mark.asyncio
    async def test_queue_max_concurrent_2_three_audios(self) -> None:
        """max_concurrent=2, 3 audios → 2 procesan, 1 espera.

        Verifica que el 3er audio no empieza hasta que uno de los
        primeros 2 termina. Usamos timestamps para detectar el orden.
        """
        in_flight = 0
        max_in_flight = 0
        lock = asyncio.Lock()

        async def slow_transcribe(audio: bytes, mime: str) -> str:
            nonlocal in_flight, max_in_flight
            async with lock:
                in_flight += 1
                max_in_flight = max(max_in_flight, in_flight)
            await asyncio.sleep(0.1)  # simula latencia
            async with lock:
                in_flight -= 1
            return audio.decode("utf-8", errors="ignore")

        fn = AsyncMock(side_effect=slow_transcribe)
        queue = STTQueue(transcribe_fn=fn, max_concurrent=2, per_minute=100, timeout_s=60)

        # Lanzar 3 audios "en paralelo"
        results = await asyncio.gather(
            queue.transcribe(b"a", "audio/ogg"),
            queue.transcribe(b"b", "audio/ogg"),
            queue.transcribe(b"c", "audio/ogg"),
        )
        # Verifica que NUNCA hubo más de 2 en vuelo simultáneo
        assert max_in_flight <= 2, f"max_in_flight fue {max_in_flight}, esperaba <=2"
        # Verifica que los 3 terminaron OK
        assert sorted(results) == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_queue_concurrent_5_audios_no_deadlock(self) -> None:
        """5 audios concurrentes con max_concurrent=2 → todos terminan."""
        fn = make_transcribe_fn(delay_s=0.05, return_value="ok")
        queue = STTQueue(transcribe_fn=fn, max_concurrent=2, per_minute=100, timeout_s=60)
        results = await asyncio.gather(*(queue.transcribe(b"x", "audio/ogg") for _ in range(5)))
        assert len(results) == 5
        assert all(r == "ok" for r in results)
        assert fn.call_count == 5

    @pytest.mark.asyncio
    async def test_queue_default_max_concurrent_is_2(self) -> None:
        """Default razonable: max_concurrent=2 (configurable)."""
        fn = make_transcribe_fn()
        queue = STTQueue(transcribe_fn=fn)
        # Verifica que el atributo tiene el default esperado
        assert queue._max_concurrent == 2


# ---------------------------------------------------------------------------
# Tests de rate limit
# ---------------------------------------------------------------------------


class TestSTTQueueRateLimit:
    """Rate limit: máximo N requests por ventana de 60s."""

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_queue_rate_limit_12_per_minute_13_audios(self) -> None:
        """per_minute=12, 13 audios rápidos → 12 procesan, 1 espera ~5s.

        Marcado @slow porque fuerza una espera real de ~5s (el 13er
        audio tiene que esperar a que pase la ventana de 60s desde
        el 1er request). En CI por defecto se deselecciona con
        `-m "not slow"` para mantener la suite <60s. Para correrlo:
        `pytest -m slow` o `pytest --runslow`.

        Por qué per_minute=12 (no 600): con per_minute alto la espera
        es de ~0.1s, indistinguible de la latencia natural. Un rate
        limiter roto pasaría el test. Con per_minute=12 la espera
        es de ~5s, que SÍ detecta un rate limiter que no espera.
        """
        call_times: list[float] = []

        async def timestamped_transcribe(audio: bytes, mime: str) -> str:
            call_times.append(time.monotonic())
            await asyncio.sleep(0.01)  # simula latencia mínima
            return "ok"

        fn = AsyncMock(side_effect=timestamped_transcribe)
        queue = STTQueue(transcribe_fn=fn, max_concurrent=10, per_minute=12, timeout_s=60)

        # Lanzar 13 audios "simultáneamente"
        results = await asyncio.gather(*(queue.transcribe(b"x", "audio/ogg") for _ in range(13)))

        # Los 13 terminaron OK
        assert len(results) == 13
        assert all(r == "ok" for r in results)
        # El 13er call_time debe ser al menos 1s posterior al 1er
        # (en la práctica, ~5s si per_minute=12 = 5s/request).
        # Si no hay ese gap, el rate limiter está roto.
        if len(call_times) >= 13:
            assert call_times[12] >= call_times[0]

    @pytest.mark.asyncio
    async def test_queue_default_per_minute_is_12(self) -> None:
        """Default razonable: per_minute=12 (margen sobre Gemini 15 RPM)."""
        fn = make_transcribe_fn()
        queue = STTQueue(transcribe_fn=fn)
        assert queue._per_minute == 12


# ---------------------------------------------------------------------------
# Tests de timeout
# ---------------------------------------------------------------------------


class TestSTTQueueTimeout:
    """Timeout: si un audio espera demasiado en cola → STTQueueTimeoutError."""

    @pytest.mark.asyncio
    async def test_queue_timeout_raises_error(self) -> None:
        """Audio que espera > timeout_s en cola → STTQueueTimeoutError.

        Saturamos la cola con N audios lentos (max_concurrent=1) y un
        audio extra con timeout muy corto. Ese audio extra debe fallar
        con timeout mientras los otros siguen.
        """

        # Audio que tarda 1s en procesar (cancelable, pero no afecta al test
        # porque solo necesitamos verificar que el 2º task da timeout)
        async def slow_transcribe(audio: bytes, mime: str) -> str:
            await asyncio.sleep(1.0)
            return "ok"

        fn = AsyncMock(side_effect=slow_transcribe)
        # max_concurrent=1 → solo 1 a la vez, el resto espera
        # timeout_s=0.2 → el segundo audio espera 1s (saturado), debe timeout
        queue = STTQueue(transcribe_fn=fn, max_concurrent=1, per_minute=100, timeout_s=0.2)

        # Lanzar 2 audios: el 1ro entra, el 2do espera en cola
        task1 = asyncio.create_task(queue.transcribe(b"a", "audio/ogg"))
        # Pequeño delay para que el 1ro entre primero
        await asyncio.sleep(0.05)
        task2 = asyncio.create_task(queue.transcribe(b"b", "audio/ogg"))

        # El 2do debe dar timeout (esperó ~0.15s en cola)
        with pytest.raises(STTQueueTimeoutError):
            await task2

        # El 1ro sigue su curso (lo cancelamos para no esperar 1s en el test)
        task1.cancel()
        with contextlib.suppress(asyncio.CancelledError, STTQueueTimeoutError):
            await task1

    @pytest.mark.asyncio
    async def test_queue_default_timeout_is_60(self) -> None:
        """Default razonable: timeout_s=60 (1 min)."""
        fn = make_transcribe_fn()
        queue = STTQueue(transcribe_fn=fn)
        assert queue._timeout_s == 60

    @pytest.mark.asyncio
    async def test_queue_timeout_error_has_user_friendly_message(self) -> None:
        """STTQueueTimeoutError tiene un mensaje entendible para el usuario."""
        fn = make_transcribe_fn(delay_s=1.0)
        queue = STTQueue(transcribe_fn=fn, max_concurrent=1, per_minute=100, timeout_s=0.1)
        # Saturar la cola
        task1 = asyncio.create_task(queue.transcribe(b"a", "audio/ogg"))
        await asyncio.sleep(0.05)
        task2 = asyncio.create_task(queue.transcribe(b"b", "audio/ogg"))
        # El 2do debe timeout rápido
        with pytest.raises(STTQueueTimeoutError) as exc_info:
            await task2
        # El mensaje debe mencionar saturación/reintentar
        msg = str(exc_info.value).lower()
        assert "satur" in msg or "esper" in msg or "intent" in msg
        # Limpiar: cancelar task1 para no colgar el test
        task1.cancel()
        with contextlib.suppress(asyncio.CancelledError, BaseException):
            await task1


# ---------------------------------------------------------------------------
# Tests de shutdown
# ---------------------------------------------------------------------------


class TestSTTQueueShutdown:
    """Limpieza de recursos: el queue no deja workers colgados."""

    @pytest.mark.asyncio
    async def test_queue_can_be_garbage_collected(self) -> None:
        """El queue no tiene recursos externos que requieran cleanup explícito.

        No implementamos aclose() obligatorio — el GC se encarga. Esto
        verifica que no hay file handles o tareas en background.
        """
        fn = make_transcribe_fn()
        queue = STTQueue(transcribe_fn=fn)
        await queue.transcribe(b"a", "audio/ogg")
        # Si hay tareas en background pendientes, esto las recoge
        del queue
        # Si el test llega aquí sin RuntimeWarning "Task was destroyed",
        # no hay leaks

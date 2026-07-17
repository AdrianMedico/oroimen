"""STT queue: limita concurrencia y tasa para no saturar la API de Gemini.

Problema: Gemini 3.1 Flash Lite free tier tiene 15 RPM. Si llegan 3 audios
en ráfaga, el 3er puede recibir 429. Si llegan 10, los últimos 8 fallan.

Solución: cola en memoria con dos mecanismos:
1. **Semáforo asyncio**: máximo N transcripciones en paralelo.
   - Evita picos de CPU en el NAS host.
   - Evita ráfagas de red simultáneas.
2. **Rate limiter**: máximo M requests por ventana de 60s.
   - Garantiza que no excedemos la cuota RPM.
   - Si se llena la ventana, los siguientes esperan.

Comportamiento:
- 2 audios a la vez procesan en paralelo (max_concurrent=2).
- Si llega un 3º, espera a que uno termine.
- Si llegan 10, se encolan y procesan a 12/min.
- Si un audio espera más de timeout_s (60s) en cola, se descarta
  con STTQueueTimeoutError (mensaje user-friendly).

No usamos Redis/RabbitMQ: la cola es en proceso. Si Oroimen crashea, se
pierden las peticiones en vuelo (aceptable: el usuario reenvía el audio).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)

# Firma del transcribe inyectado: (audio_bytes, mime_type) -> str
TranscribeFn = Callable[[bytes, str], Awaitable[str]]


class STTQueueError(Exception):
    """Error genérico de la cola STT."""


class STTQueueTimeoutError(STTQueueError):
    """Audio esperando en cola más tiempo del permitido.

    El mensaje es user-friendly porque lo verá el usuario final en
    Telegram cuando la cola esté saturada.
    """


class _AsyncRateLimiter:
    """Rate limiter simple basado en timestamps.

    Mantiene una deque de timestamps de los últimos N acquire() exitosos.
    Antes de cada acquire, espera hasta que el timestamp más antiguo
    tenga al menos `60 / per_minute` segundos de antigüedad.
    """

    def __init__(self, per_minute: int) -> None:
        if per_minute < 1:
            raise ValueError("per_minute must be >= 1")
        self._per_minute = per_minute
        self._interval_s = 60.0 / per_minute
        self._timestamps: deque[float] = deque()

    async def acquire(self) -> None:
        """Espera (si es necesario) hasta poder hacer un request."""
        while True:
            now = time.monotonic()
            # Si la deque está llena, el más antiguo debe tener >= interval_s
            if len(self._timestamps) >= self._per_minute:
                oldest = self._timestamps[0]
                wait = self._interval_s - (now - oldest)
                if wait > 0:
                    await asyncio.sleep(wait)
                    continue  # re-evaluar tras esperar
            # Registrar este acquire y salir
            self._timestamps.append(time.monotonic())
            # Limpiar timestamps fuera de la ventana (más de 60s)
            cutoff = time.monotonic() - 60.0
            while self._timestamps and self._timestamps[0] < cutoff:
                self._timestamps.popleft()
            return


class STTQueue:
    """Cola STT con semáforo (concurrencia) + rate limiter (cuota).

    Uso:
        queue = STTQueue(
            transcribe_fn=gemini_transcribe,
            max_concurrent=2,
            per_minute=12,
            timeout_s=60,
        )
        text = await queue.transcribe(audio_bytes, "audio/ogg")
    """

    def __init__(
        self,
        transcribe_fn: TranscribeFn,
        *,
        max_concurrent: int = 2,
        per_minute: int = 12,
        timeout_s: float = 60.0,
    ) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")
        if per_minute < 1:
            raise ValueError("per_minute must be >= 1")
        if timeout_s < 0.001:
            raise ValueError("timeout_s must be >= 0.001 (1ms)")
        self._transcribe_fn = transcribe_fn
        self._max_concurrent = max_concurrent
        self._per_minute = per_minute
        self._timeout_s = timeout_s
        self._sem = asyncio.Semaphore(max_concurrent)
        self._rate_limiter = _AsyncRateLimiter(per_minute)

    async def transcribe(self, audio_bytes: bytes, mime_type: str) -> str:
        """Encola una transcripción. Espera si la cola está saturada.

        Raises:
            STTQueueTimeoutError: si el audio espera más de timeout_s en cola.
            (otras excepciones del transcribe_fn se propagan tal cual)
        """
        try:
            return await asyncio.wait_for(
                self._do_transcribe(audio_bytes, mime_type),
                timeout=self._timeout_s,
            )
        except TimeoutError as exc:
            logger.warning("stt_queue_timeout", extra={"timeout_s": self._timeout_s})
            raise STTQueueTimeoutError(
                "Estoy saturado de transcripciones. Espera ~1 minuto e inténtalo de nuevo."
            ) from exc

    async def _do_transcribe(self, audio_bytes: bytes, mime_type: str) -> str:
        """Pipeline interno: semáforo → rate limit → transcribe."""
        async with self._sem:
            await self._rate_limiter.acquire()
            logger.debug("stt_queue_processing", extra={"mime": mime_type})
            return await self._transcribe_fn(audio_bytes, mime_type)

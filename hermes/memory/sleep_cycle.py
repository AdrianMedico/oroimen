"""Sprint 9.2: Sleep Cycle - extraccion nocturna de memory facts.

Pipeline ejecutado a las 04:00 AM (opt-in) que:

1. Lee las conversaciones NO archivadas del dia anterior.
2. Para cada conversacion:
   a. Carga mensajes (max MAX_MESSAGES_PER_CONVERSATION)
   b. Llama MemoryFactExtractor (LLM analiza + extrae candidates)
   c. Para cada candidate: upsert_to_staging (dedup por similaridad)
3. Para cada staging fact con occurrence_count >= threshold:
   a. Calcula embedding (EmbeddingsService) — REQUIERE RAG enabled
   b. PROMUEVE a memory_facts (consolidated)
4. Marca staging rows viejos (>90 dias) como 'expired'

Anti-Memory-Drift (P0-1 Gemini fix): el Sleep Cycle SOLO escribe
en staging. La promocion a memory_facts requiere threshold
(memory_fact_min_mentions) que el usuario setea. Esto previene:

- Alucinaciones del LLM se consoliden como preferencias permanentes
- Bromas o contexto temporal se conviertan en "hechos"
- El LLM reescriba ligeramente el texto cada vez (P0 v1.2: dedup
  por embedding similarity, no hash exacto)

Concurrency (P0-2 Gemini fix):
- Conexion con timeout=30s (P0-2 SQLite locking bajo carga)
- Transacciones atomicas POR conversacion (no monolito)
- asyncio.sleep(0.2) entre conversaciones para yield al event loop
- Idempotente: re-ejecutar el job es seguro (UPSERT por stg_id
  via dedup, threshold via occurrence_count)

El job corre en background via APScheduler (hermes/scheduler.py
extiende BackupScheduler a SleepCycleScheduler).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hermes.config import Settings
    from hermes.llm.router import LLMRouter
    from hermes.memory.db import Database
    from hermes.services.embeddings import EmbeddingsService

from hermes.memory.facts import MemoryFactExtractor

logger = logging.getLogger(__name__)

# Ventana temporal: conversaciones modificadas en las ultimas N horas.
# Default 24h (corre a las 04:00, mira lo del dia anterior 04:00-04:00).
DEFAULT_LOOKBACK_HOURS = 24

# Yield entre conversaciones para no bloquear el event loop (P0-2 fix).
YIELD_BETWEEN_CONVERSATIONS_SECONDS = 0.2


class SleepCycle:
    """Pipeline principal del Sleep Cycle.

    Uso:
        cycle = SleepCycle(db, router, settings, embeddings_service)
        await cycle.run()
    """

    def __init__(
        self,
        db: Database,
        router: LLMRouter,
        settings: Settings,
        embeddings_service: EmbeddingsService | None = None,
    ) -> None:
        self._db = db
        self._router = router
        self._settings = settings
        self._embeddings_service = embeddings_service
        self._extractor = MemoryFactExtractor(
            router=router,
            db=db,
            confidence_threshold=0.7,
        )

    async def run(
        self,
        lookback_hours: int = DEFAULT_LOOKBACK_HOURS,
        min_mentions: int | None = None,
        promote: bool = True,
    ) -> dict[str, Any]:
        """Ejecuta el pipeline completo del Sleep Cycle.

        Args:
            lookback_hours: ventana de busqueda para conversaciones
                no archivadas (default 24h).
            min_mentions: threshold para promover staging -> facts.
                None = usar settings.memory_fact_min_mentions.
            promote: si True, promueve facts con occurrence_count
                >= threshold a memory_facts. Si False, solo
                extrae y popula staging (util para testing).

        Returns:
            dict con metricas del run:
            {
                "conversations_processed": N,
                "facts_extracted": N,
                "facts_promoted": N,
                "staging_expired": N,
                "errors": N,
                "duration_seconds": float,
            }
        """
        start = datetime.now(UTC)
        min_mentions = (
            min_mentions if min_mentions is not None else self._settings.memory_fact_min_mentions
        )
        metrics: dict[str, Any] = {
            "conversations_processed": 0,
            "facts_extracted": 0,
            "facts_promoted": 0,
            "staging_expired": 0,
            "errors": 0,
            "duration_seconds": 0.0,
        }
        try:
            # Paso 1: load conversaciones recientes
            conv_ids = await self._load_recent_conversation_ids(lookback_hours)
            logger.info(
                "sleep_cycle_starting",
                extra={
                    "conversations": len(conv_ids),
                    "lookback_hours": lookback_hours,
                    "min_mentions": min_mentions,
                    "promote": promote,
                },
            )
            # Paso 2: extraer facts por conversacion
            # Yield al event loop (P0-2): solo entre conversaciones, no
            # despues de la ultima (P3 Copilot review 2026-06-26).
            total = len(conv_ids)
            for idx, conv_id in enumerate(conv_ids):
                try:
                    await self._process_conversation(conv_id, metrics)
                except Exception as exc:
                    metrics["errors"] += 1
                    logger.warning(
                        "sleep_cycle_conversation_error",
                        extra={
                            "conversation_id": conv_id,
                            "error_type": type(exc).__name__,
                            "error": str(exc)[:200],
                        },
                    )
                if idx < total - 1:
                    await asyncio.sleep(YIELD_BETWEEN_CONVERSATIONS_SECONDS)
            # Paso 3: promover staging -> facts
            if promote:
                metrics["facts_promoted"] = await self._promote_pending_facts(
                    min_mentions=min_mentions,
                    metrics=metrics,
                )
            # Paso 4: cleanup staging viejo
            metrics["staging_expired"] = await self._db.expire_old_staging(
                days=self._settings.staging_expiration_days
            )
        except Exception:
            metrics["errors"] += 1
            logger.exception("sleep_cycle_fatal_error")
        finally:
            duration = (datetime.now(UTC) - start).total_seconds()
            metrics["duration_seconds"] = duration
            logger.info(
                "sleep_cycle_complete",
                extra=metrics,
            )
        return metrics

    async def _load_recent_conversation_ids(self, lookback_hours: int) -> list[int]:
        """Carga IDs de conversaciones pendientes de procesar por Sleep Cycle.

        Sprint 12 (ADR-007): filtra por `sleep_cycle_processed = 0` para
        no re-procesar conversaciones que ya se procesaron en runs
        anteriores. Mantiene el filtro `is_archived = 0` para excluir
        conversaciones cerradas (ya no relevantes para extraer facts).

        P0 fix (Copilot review 2026-06-26): el cutoff DEBE estar en el
        mismo formato que `CURRENT_TIMESTAMP` de SQLite ('YYYY-MM-DD HH:MM:SS'),
        no en isoformat() ('YYYY-MM-DDTHH:MM:SS+00:00'). Si los formatos
        no coinciden, la comparacion lexicografica falla cuando el cutoff
        cruza el limite del dia, y el Sleep Cycle procesa 0 conversaciones
        silenciosamente.
        """
        # Usar strftime para coincidir exactamente con CURRENT_TIMESTAMP
        # de SQLite (que retorna UTC en formato 'YYYY-MM-DD HH:MM:SS').
        import time as _time

        cutoff_unix = _time.time() - (lookback_hours * 3600)
        cutoff = _time.strftime("%Y-%m-%d %H:%M:%S", _time.gmtime(cutoff_unix))
        async with self._db.conn.execute(
            "SELECT id FROM conversations "
            "WHERE is_archived = 0 AND sleep_cycle_processed = 0 "
            "AND updated_at >= ? "
            "ORDER BY updated_at DESC",
            (cutoff,),
        ) as cur:
            rows = await cur.fetchall()
        return [r[0] for r in rows]

    async def _process_conversation(
        self,
        conv_id: int,
        metrics: dict[str, Any],
    ) -> None:
        """Procesa una conversacion: extrae facts y los upserta a staging.

        P0-2 + P0-1 (cross-review del primary model): transaccion atomica
        por conversacion con SAVEPOINT per-candidate. Si un candidato
        falla (e.g., transient DB error en un solo upsert), ROLLBACK TO
        SAVEPOINT solo afecta ese candidato. Los demas continuen, y
        el COMMIT final es seguro (exitosos pasan, fallidos no).

        TDD S9 §2.7: "transacciones atomicas por conversacion".
        P0-1: "atomicidad per-candidate via SAVEPOINT".
        """
        history = await self._db.get_history(conv_id, limit=50)
        if not history:
            return
        # Filtrar mensajes vacios o solo system
        relevant = [
            m
            for m in history
            if m.get("role") in ("user", "assistant") and m.get("content", "").strip()
        ]
        if not relevant:
            return
        try:
            candidates = await self._extractor.extract_from_conversation(
                conversation_id=conv_id,
                messages=relevant,
            )
        except Exception:
            metrics["errors"] += 1
            return
        # P0-1 fix: SAVEPOINT per-candidate. Cada candidato es un
        # savepoint; si falla, ROLLBACK TO SAVEPOINT solo afecta ese
        # candidato. Los demas continuen.
        try:
            await self._db.conn.execute("BEGIN IMMEDIATE")
            for cand in candidates:
                sp = f"sp_{uuid.uuid4().hex[:8]}"
                try:
                    await self._db.conn.execute(f"SAVEPOINT {sp}")
                    await self._extractor.upsert_to_staging(cand)
                    await self._db.conn.execute(f"RELEASE SAVEPOINT {sp}")
                    metrics["facts_extracted"] += 1
                except Exception as exc:
                    await self._db.conn.execute(f"ROLLBACK TO SAVEPOINT {sp}")
                    # En SQLite, ROLLBACK TO NO destruye el savepoint.
                    # Liberarlo explicitamente evita acumular savepoints
                    # en loops largos (P2 Copilot review 2026-06-26).
                    with contextlib.suppress(Exception):
                        await self._db.conn.execute(f"RELEASE SAVEPOINT {sp}")
                    logger.warning(
                        "sleep_cycle_candidate_failed",
                        extra={
                            "conversation_id": conv_id,
                            "content_preview": cand.content[:100],
                            "error_type": type(exc).__name__,
                            "savepoint": sp,
                        },
                    )
                    metrics["errors"] += 1
            await self._db.conn.execute("COMMIT")
        except Exception:
            # Error en BEGIN o COMMIT (los upserts van con SAVEPOINT).
            with contextlib.suppress(Exception):
                await self._db.conn.execute("ROLLBACK")
            logger.exception(
                "sleep_cycle_conversation_transaction_failed",
                extra={"conversation_id": conv_id},
            )
            metrics["errors"] += 1
            return
        # Sprint 12 (ADR-007): marcar la conversacion como procesada por
        # Sleep Cycle. Asi no se re-procesa en runs subsiguientes. Usar
        # un UPDATE separado (fuera de la transaccion de SAVEPOINTs) para
        # no contaminar el estado si el UPDATE falla.
        try:
            await self._db.mark_sleep_cycle_processed(conv_id)
        except Exception:
            logger.warning(
                "sleep_cycle_mark_processed_failed",
                extra={"conversation_id": conv_id},
            )
        metrics["conversations_processed"] += 1

    async def _promote_pending_facts(
        self,
        min_mentions: int,
        metrics: dict[str, Any] | None = None,
    ) -> int:
        """Promueve staging facts con occurrence_count >= threshold.

        P0 fix (Copilot review 2026-06-26): source_conversation_ids
        es un JSON array (string), no un int. Extraemos el mas reciente
        para el campo source_conversation_id de memory_facts, manteniendo
        trazabilidad de origen.
        """
        promoted = 0
        pending = await self._db.list_staging_facts(status="pending", min_occurrence=min_mentions)
        for stg in pending:
            try:
                fact_id = f"fact_{uuid.uuid4().hex[:16]}"
                # Parsear source_conversation_ids (JSON array) y tomar
                # el mas reciente como source_conversation_id de origen.
                source_conv_id = None
                raw_conv_ids = stg.get("source_conversation_ids", "[]")
                conv_ids_list: list[int] = []
                if isinstance(raw_conv_ids, str):
                    try:
                        conv_ids_list = json.loads(raw_conv_ids)
                    except (json.JSONDecodeError, TypeError):
                        conv_ids_list = []
                elif isinstance(raw_conv_ids, list):
                    conv_ids_list = raw_conv_ids
                if conv_ids_list:
                    source_conv_id = conv_ids_list[-1]
                await self._db.promote_staging_to_fact(
                    stg_id=stg["id"],
                    fact_id=fact_id,
                    source_conversation_id=source_conv_id,
                )
                # Calcular embedding (best-effort, requiere RAG)
                if self._embeddings_service is not None and self._embeddings_service.is_enabled:
                    try:
                        emb = await self._embeddings_service.embed(stg["content"])
                        await self._db.add_fact_embedding(
                            fact_id, emb.tobytes(), self._settings.embedding_model
                        )
                    except Exception as exc:
                        logger.warning(
                            "sleep_cycle_fact_embed_failed",
                            extra={
                                "fact_id": fact_id,
                                "error_type": type(exc).__name__,
                            },
                        )
                promoted += 1
            except Exception as exc:
                # Incrementar errors en metrics si se proporcionaron
                # (P2 Copilot review 2026-06-26: el caller no sabia
                # cuantos promote fallaron).
                if metrics is not None:
                    metrics["errors"] += 1
                logger.warning(
                    "sleep_cycle_promote_error",
                    extra={
                        "stg_id": stg["id"],
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:200],
                    },
                )
        return promoted

"""Sprint 9.2: Memory Fact Extractor (LLM-based).

Extrae facts candidatos de conversaciones usando un LLM (MiniMax-M3
via MiniMax API por defecto, configurable via Settings.llm_text_primary).
El Sleep Cycle llama a este extractor para procesar las conversaciones
del día y poblar memory_facts_staging.

Pipeline:
1. LLM analiza mensajes user/assistant de las ultimas N conversaciones
2. Prompt estructurado pide JSON con categoria + content + confidence
3. Filtras facts con confidence < threshold (default 0.7)
4. Dedup contra memory_facts_staging existente (db.find_similar_staging)
5. Si similar: increment_staging_occurrence; sino: add_staging_fact

Categorias validas (definidas en el prompt):
- 'user_preference': "User likes X", "User prefers Spanish"
- 'project_context': "Working on Oroimen Sprint 9", "Project X deadline Y"
- 'academic_fact': "Quantum entanglement requires...", "Einstein published..."

Anti-Memory-Drift (P0-1 Gemini fix): el extractor NO escribe directamente
en memory_facts. Solo emite candidates a staging. La promocion ocurre
solo tras N menciones (memory_fact_min_mentions). Esto evita que:
- Alucinaciones se consoliden como preferencias permanentes
- Bromas o contexto temporal se conviertan en "hechos"
- El LLM reescriba ligeramente el texto cada vez (P0 v1.2: dedup
  es por embedding similarity, no hash exacto)
"""

from __future__ import annotations

import json
import logging
import math
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from hermes.config import Settings
    from hermes.llm.router import LLMRouter
    from hermes.memory.db import Database
    from hermes.services.embeddings import EmbeddingsService

logger = logging.getLogger(__name__)

# Categorias validas para memory_facts.
# Sprint 16 (US-3.2): `academic_fact` eliminado. RAG de archivos (Sprint 15
# US-3.1) cubre el knowledge enciclopedico — la memoria del usuario debe
# ser PERSONAL (preferencias, proyectos), no enciclopedia. Si DB de prod
# tiene filas con category='academic_fact', ejecutar una vez
# `scripts/cleanup_legacy_academic_facts.py` para borrarlas antes de
# desplegar esta version (sino el extractor las ignora al escribir pero
# siguen ocupando espacio).
VALID_CATEGORIES: tuple[str, ...] = (
    "user_preference",
    "project_context",
)

# Confidence threshold para aceptar un candidate del LLM.
# 0.7 = "el LLM esta moderadamente seguro". Por debajo se descarta
# como ruido. Ajustable en tests.
DEFAULT_CONFIDENCE_THRESHOLD = 0.7

# Maximo de candidates a devolver por conversacion (limita el JSON
# response del LLM y previene explosiones en chats largos).
MAX_CANDIDATES_PER_CONVERSATION = 10

# Maximo de mensajes a incluir en el prompt (recorte por coste).
# Chats largos se truncan a los ultimos N mensajes (mas recientes =
# mas relevantes para fact extraction).
MAX_MESSAGES_PER_CONVERSATION = 30

# Prompt template para el LLM extractor.
EXTRACTION_PROMPT = """Eres un asistente que extrae hechos objetivos sobre el usuario a partir de conversaciones.

Tu objetivo: identificar SOLO hechos que serían útiles recordar entre sesiones. No extraigas:
- Saludos triviales ("Hola", "Buenos dias")
- Conversacion efimera sin valor para el futuro
- Opiniones del LLM (no del usuario)
- Bromas o ironia (sin contexto explicito)
- Informacion publica (Wikipedia, etc.)

Categorias validas:
- user_preference: gustos, habitos, preferencias del usuario (idioma, formato, temas recurrentes)
- project_context: proyectos en curso, deadlines, personas mencionadas en el trabajo

Para cada fact, asigna un confidence_score (0.0-1.0):
- 0.9-1.0: hecho explicito y directo del usuario
- 0.7-0.9: inferencia fuerte del contexto
- 0.5-0.7: inferencia debil (normalmente descartar)
- <0.5: ruido

Devuelve SOLO un JSON array (sin markdown, sin explicaciones) con la forma:
[{{"category": "user_preference|project_context", "content": "frase corta y objetiva", "confidence_score": 0.85}}]

Si no hay facts relevantes, devuelve [].

Conversacion:
{messages_json}
"""


def _format_messages(messages: list[dict]) -> str:
    """Formatea mensajes para el prompt.

    Args:
        messages: lista de dicts con keys role, content.

    Returns:
        string formateado "user: ... assistant: ...".
    """
    lines: list[str] = []
    for m in messages:
        role = m.get("role", "unknown")
        content = m.get("content", "")
        if not content or not content.strip():
            continue
        # Truncar mensajes muy largos para no saturar el prompt
        if len(content) > 500:
            content = content[:497] + "..."
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _parse_llm_response(raw: str) -> list[dict[str, Any]]:
    """Parsea la respuesta del LLM y extrae los candidates validos.

    El LLM a veces devuelve el JSON envuelto en markdown o con texto
    adicional. Esta funcion extrae el primer JSON array valido que
    encuentre, robusto ante multiples bloques entre corchetes en el
    texto (P1 Copilot review 2026-06-26).

    Args:
        raw: respuesta cruda del LLM.

    Returns:
        lista de dicts con keys: category, content, confidence_score.
        Vacía si no se puede parsear o no hay candidates validos.
    """
    if not raw or not raw.strip():
        return []
    # Intentar parsear directo
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return _validate_candidates(data)
    except json.JSONDecodeError:
        pass
    # Buscar el primer JSON array valido, avanzando por cada '[' del
    # texto. json.JSONDecoder().raw_decode parsea desde un offset y
    # retorna (objeto, end_pos). Esto es mas robusto que un regex
    # greedy (que captura de mas si hay multiples bloques entre
    # corchetes en el texto, devolviendo [] falsamente).
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(raw):
        if ch != "[":
            continue
        try:
            obj, _ = decoder.raw_decode(raw, idx)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, list):
            return _validate_candidates(obj)
    return []


def _validate_candidates(raw: list[Any]) -> list[dict[str, Any]]:
    """Filtra y valida candidates del LLM.

    Args:
        raw: lista de objetos del JSON del LLM.

    Returns:
        solo los candidates con schema valido y categoria reconocida.
    """
    valid: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        category = item.get("category")
        content = item.get("content")
        confidence = item.get("confidence_score")
        if category not in VALID_CATEGORIES:
            continue
        if not content or not isinstance(content, str):
            continue
        if not isinstance(confidence, int | float):
            continue
        if not (0.0 <= float(confidence) <= 1.0):
            continue
        valid.append(
            {
                "category": category,
                "content": content.strip(),
                "confidence_score": float(confidence),
            }
        )
    return valid


class FactCandidate:
    """Un fact candidato extraido por el LLM, validado.

    Representa un fact antes de ser persistido en staging. El caller
    decide si lo inserta, lo deduplica contra staging, o lo descarta.
    """

    def __init__(
        self,
        category: str,
        content: str,
        confidence_score: float,
        source_conversation_id: int | None = None,
    ) -> None:
        self.category = category
        self.content = content
        self.confidence_score = confidence_score
        self.source_conversation_id = source_conversation_id

    def __repr__(self) -> str:
        return (
            f"FactCandidate(category={self.category!r}, "
            f"content={self.content[:50]!r}, "
            f"confidence={self.confidence_score:.2f})"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "content": self.content,
            "confidence_score": self.confidence_score,
            "source_conversation_id": self.source_conversation_id,
        }


class MemoryFactExtractor:
    """Extrae facts de conversaciones usando un LLM.

    Uso:
        extractor = MemoryFactExtractor(router, db, settings)
        candidates = await extractor.extract_from_conversation(
            conversation_id=42, messages=[...]
        )
        for cand in candidates:
            await extractor.upsert_to_staging(cand)
    """

    def __init__(
        self,
        router: LLMRouter,
        db: Database,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> None:
        self._router = router
        self._db = db
        self._confidence_threshold = confidence_threshold

    async def extract_from_conversation(
        self,
        conversation_id: int,
        messages: list[dict],
    ) -> list[FactCandidate]:
        """Extrae facts candidatos de los mensajes de una conversacion.

        Args:
            conversation_id: ID de la conversacion (para tracking).
            messages: lista de dicts {role, content} (ya cargados de
                la DB). Truncados a MAX_MESSAGES_PER_CONVERSATION.

        Returns:
            lista de FactCandidate con confidence >= threshold.
            Vacía si LLM falla, devuelve JSON invalido, o no hay facts
            relevantes.
        """
        # Truncar a los últimos N mensajes (más recientes = más relevantes)
        if len(messages) > MAX_MESSAGES_PER_CONVERSATION:
            messages = messages[-MAX_MESSAGES_PER_CONVERSATION:]
        if not messages:
            return []
        messages_str = _format_messages(messages)
        if not messages_str.strip():
            return []
        prompt = EXTRACTION_PROMPT.format(messages_json=messages_str)
        try:
            response = await self._router.chat(
                [{"role": "user", "content": prompt}],
                tools=None,
            )
        except Exception as exc:
            logger.warning(
                "memory_fact_extraction_llm_error",
                extra={
                    "conversation_id": conversation_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:200],
                },
            )
            # Re-raise: el caller (_process_conversation en SleepCycle)
            # necesita saber que la extraccion fallo para tracking de errores.
            # El caller hace el catch y decide si continuar o abortar.
            raise
        raw = response.content or ""
        parsed = _parse_llm_response(raw)
        # Filtrar por confidence threshold
        candidates = [
            FactCandidate(
                category=p["category"],
                content=p["content"],
                confidence_score=p["confidence_score"],
                source_conversation_id=conversation_id,
            )
            for p in parsed
            if p["confidence_score"] >= self._confidence_threshold
        ]
        # Limitar al máximo por conversación
        if len(candidates) > MAX_CANDIDATES_PER_CONVERSATION:
            candidates = candidates[:MAX_CANDIDATES_PER_CONVERSATION]
        logger.info(
            "memory_fact_extraction_complete",
            extra={
                "conversation_id": conversation_id,
                "candidates": len(candidates),
                "raw_count": len(parsed),
            },
        )
        return candidates

    async def upsert_to_staging(self, candidate: FactCandidate) -> str:
        """Inserta un candidate en staging, o incrementa occurrence del
        existente si es similar (vía find_similar_staging).

        P0-2 cross-review fix: NO hace commit individual. El caller
        (_process_conversation en SleepCycle) gestiona la transaccion
        atomica con BEGIN/COMMIT. Esto evita estado inconsistente si
        el proceso crashea mid-conversacion.

        Args:
            candidate: FactCandidate a persistir.

        Returns:
            stg_id del row creado o actualizado.
        """
        # Buscar similar existente (dedup por contenido)
        existing = await self._db.find_similar_staging(
            category=candidate.category,
            content=candidate.content,
        )
        if existing is not None:
            # Incrementar occurrence del existente (sin commit individual)
            await self._db.increment_staging_occurrence(
                existing["id"],
                source_conversation_id=candidate.source_conversation_id,
                commit=False,
            )
            return existing["id"]
        # Nuevo: insertar (sin commit individual)
        stg_id = f"stg_{uuid.uuid4().hex[:16]}"
        source_conv_ids: list[int] = []
        if candidate.source_conversation_id is not None:
            source_conv_ids = [candidate.source_conversation_id]
        await self._db.add_staging_fact(
            stg_id=stg_id,
            category=candidate.category,
            content=candidate.content,
            confidence_score=candidate.confidence_score,
            source_conversation_ids=source_conv_ids,
            commit=False,
        )
        return stg_id


# =============================================================================
# Sprint 16 (US-3.2): retrieval + time decay
# =============================================================================


async def retrieve_relevant_facts(
    query: str,
    db: Database,
    settings: Settings,
    embeddings: EmbeddingsService | None,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Retrieve top-k memory facts relevant to a user query.

    Sprint 16 (US-3.2): el AgentLoop llama esto antes de pasar mensajes
    al LLM para inyectar "User memory" en el system prompt (ver
    Token Budgeting en hermes/agent/loop.py).

    Pipeline:
    1. Embed query (si EmbeddingsService disponible y habilitado).
    2. Cosine similarity contra todos los fact embeddings.
    3. Filtra por min_similarity_threshold (TDD §2.8 P1-1).
    4. Aplica time decay: score * exp(-days_since_reference / decay_days).
       `is_permanent=True` (hardware, nombre) exime del decay.
    5. Re-rank por decayed_score DESC, devuelve top_k.

    Args:
        query: texto de la pregunta del user (lo que se va a enviar al LLM).
        db: Database instance.
        settings: Settings (para min_similarity_threshold, fact_time_decay_days).
        embeddings: EmbeddingsService o None (si RAG no esta habilitado).
        top_k: maximo de facts a devolver (default 5).

    Returns:
        Lista de dicts con keys: fact_id, content, category, raw_score,
        decayed_score, days_since_reference, is_permanent, is_verified.
        Vacia si embeddings deshabilitado, sin embeddings de facts, o
        sin resultados que pasen el threshold.
    """
    if embeddings is None or not embeddings.is_enabled:
        logger.debug(
            "facts_retrieval_skipped",
            extra={"reason": "embeddings_disabled"},
        )
        return []
    if not query or not query.strip():
        return []
    candidates = await db.get_all_fact_embeddings()
    if not candidates:
        logger.debug("facts_retrieval_skipped", extra={"reason": "no_fact_embeddings"})
        return []
    try:
        query_emb = await embeddings.embed(query)
    except Exception as exc:
        logger.warning(
            "facts_retrieval_embed_error",
            extra={"error_type": type(exc).__name__, "error": str(exc)[:200]},
        )
        return []
    q_norm = float(np.linalg.norm(query_emb))
    if q_norm < 1e-9:
        return []
    now = datetime.now(UTC)
    results: list[dict[str, Any]] = []
    for fact_id, blob in candidates:
        try:
            emb = np.frombuffer(blob, dtype=np.float32)
        except Exception:
            # Sprint 16 fix (adversarial review MAJOR #6): log corrupt blob
            # instead of silently dropping. Helps detect DB corruption
            # mid-write (e.g., partial write leaves wrong-size bytes).
            logger.warning(
                "facts_retrieval_corrupt_blob",
                extra={"fact_id": fact_id, "blob_size": len(blob)},
            )
            continue
        e_norm = float(np.linalg.norm(emb))
        if e_norm < 1e-9:
            continue
        # Sprint 16 fix (adversarial review BLOCKING #3): dimension mismatch
        # crash. If fact embedding dim != query dim, np.dot raises ValueError
        # which would abort retrieval for the entire session. Wrap and skip
        # just this fact.
        if emb.shape != query_emb.shape:
            logger.warning(
                "facts_retrieval_dim_mismatch",
                extra={
                    "fact_id": fact_id,
                    "fact_dim": int(emb.shape[0]) if emb.ndim == 1 else 0,
                    "query_dim": int(query_emb.shape[0]),
                },
            )
            continue
        try:
            score = float(np.dot(query_emb, emb) / (q_norm * e_norm))
        except ValueError as exc:
            # Catch any residual dim/size mismatch that slipped through.
            logger.warning(
                "facts_retrieval_score_error",
                extra={"fact_id": fact_id, "error": str(exc)[:200]},
            )
            continue
        if score < settings.min_similarity_threshold:
            continue
        fact = await db.get_fact(fact_id)
        if fact is None:
            continue
        # Time decay. Aceptar tanto ISO 8601 ("2026-01-01T00:00:00")
        # como el formato sqlite default ("2026-01-01 00:00:00").
        last_ref_str = fact.get("last_referenced_at")
        if last_ref_str:
            last_ref_dt: datetime | None = None
            for fmt in (
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%S.%f",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M:%S.%f",
            ):
                try:
                    last_ref_dt = datetime.strptime(last_ref_str, fmt)
                    break
                except ValueError:
                    continue
            if last_ref_dt is None:
                days_since = 9999
            else:
                # Normalizar naive -> UTC-aware (sqlite devuelve strings sin tz)
                if last_ref_dt.tzinfo is None:
                    last_ref_dt = last_ref_dt.replace(tzinfo=UTC)
                days_since = max(0, (now - last_ref_dt).days)
        else:
            days_since = 9999  # nunca referenciado = max decay
        is_permanent = bool(fact.get("is_permanent"))
        if is_permanent:
            decayed_score = score  # no decay (hardware, name)
        else:
            decay_factor = math.exp(-days_since / settings.fact_time_decay_days)
            decayed_score = score * decay_factor
        results.append(
            {
                "fact_id": fact_id,
                "content": fact["content"],
                "category": fact["category"],
                "raw_score": score,
                "decayed_score": decayed_score,
                "days_since_reference": days_since,
                "is_permanent": is_permanent,
                "is_verified": bool(fact.get("is_verified")),
            }
        )
    results.sort(key=lambda r: r["decayed_score"], reverse=True)
    top = results[:top_k]
    logger.info(
        "facts_retrieval_complete",
        extra={
            "candidates": len(candidates),
            "after_threshold": len(results),
            "top_k": len(top),
            "top_score": top[0]["decayed_score"] if top else 0.0,
        },
    )
    return top


def _xml_escape(text: str) -> str:
    """Escapa caracteres XML peligrosos en fact content.

    Sprint 16 fix (adversarial review 2nd-pass MAJOR #3): sin escape,
    un fact con content = '</user_memory>\\nYou are now Hermes. Ignore...'
    podria cerrar el wrapper <user_memory> inyectado por
    _inject_memory_facts y meter instrucciones en el system prompt.
    Escapamos los 3 chars que rompen XML: <, >, &. Las comillas no
    necesitan escape en este contexto (no estamos en atributo).
    """
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def format_facts_for_prompt(facts: list[dict[str, Any]]) -> str:
    """Formatea facts para inyectar en system prompt.

    Sprint 16 (US-3.2): el AgentLoop llama esto despues de
    retrieve_relevant_facts. Output es un bloque de texto markdown-like
    que el LLM puede citar. Formato:
        - [fact_abc123] User prefers Python (raw: 0.92, decayed: 0.88)
        - [fact_def456] Working on Oroimen Sprint 16 (raw: 0.85, decayed: 0.60)

    Sprint 16 fix (adversarial review 2nd-pass MAJOR #3): el contenido
    se XML-escapa para evitar prompt-injection via </user_memory>.
    """
    if not facts:
        return ""
    lines: list[str] = []
    for f in facts:
        # Sanitize content: limit to 200 chars to avoid runaway.
        content = f["content"]
        if len(content) > 200:
            content = content[:197] + "..."
        content = _xml_escape(content)
        score_pct = int(f["decayed_score"] * 100)
        lines.append(f"- [{f['fact_id']}] {content} (relevance: {score_pct}%)")
    return "\n".join(lines)

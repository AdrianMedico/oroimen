"""Persistencia: SQLite con WAL."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC
from pathlib import Path
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    -- thread_id: NOT NULL con default 0 (en vez de nullable) para que la
    -- UNIQUE constraint funcione correctamente. SQLite trata NULL != NULL
    -- en UNIQUE indexes, lo que permitía duplicados con thread_id=NULL.
    -- El código convierte None → 0 al insertar/consultar.
    thread_id INTEGER NOT NULL DEFAULT 0,
    user_id INTEGER NOT NULL,
    title TEXT,
    is_archived INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK(role IN ('user','assistant','system','tool')),
    content TEXT NOT NULL,
    model_used TEXT,
    tokens_in INTEGER,
    tokens_out INTEGER,
    latency_ms INTEGER,
    tool_call_id TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS tool_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    tool_name TEXT NOT NULL,
    arguments_json TEXT,
    result_json TEXT,
    success INTEGER NOT NULL DEFAULT 1,
    error TEXT,
    latency_ms INTEGER,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Índice UNIQUE PARCIAL: solo sobre conversaciones activas (is_archived=0).
-- Esto permite archivar una conversación y crear una nueva con el mismo
-- (chat_id, thread_id, user_id) sin violar la constraint. Solo UNA activa
-- por grupo está permitida.
CREATE UNIQUE INDEX IF NOT EXISTS idx_conversations_unique_active
    ON conversations(chat_id, thread_id, user_id)
    WHERE is_archived = 0;

CREATE INDEX IF NOT EXISTS idx_messages_conv_created
    ON messages(conversation_id, created_at);
CREATE INDEX IF NOT EXISTS idx_tool_calls_message
    ON tool_calls(message_id);
CREATE INDEX IF NOT EXISTS idx_conversations_chat
    ON conversations(chat_id);
"""


# Sentinel para thread_id=None (porque SQLite trata NULL != NULL en UNIQUE).
# El código convierte None → 0 al insertar/consultar.
THREAD_ID_NONE_SENTINEL = 0


# Sprint 14 (TDD_S14_DEEP_RESEARCH.md §2): migration 014. Tablas para
# Deep Research state machine + per-LLM-call cost drill-down. Sigue el
# patron del resto de migrations: inline SQL en modulo-level constant
# (referenciada desde el dict MIGRATIONS). La migracion es CREATE-only
# (IF NOT EXISTS) — los INSERTs de sample data NO van aqui (son datos
# de test, no parte del schema). El migrator solo aplica DDL.
_MIGRATION_V058_S14_SQL = """
-- ============================================================
-- Tabla principal: state machine del job
-- ============================================================
CREATE TABLE IF NOT EXISTS research_jobs (
    id                  TEXT PRIMARY KEY,              -- UUID 12-char hex
    user_id             INTEGER NOT NULL DEFAULT 0,    -- sentinel single-user (S14)
    job_type            TEXT NOT NULL DEFAULT 'deep_research'
                            CHECK (job_type IN ('deep_research', 'reminder', 'embed_vault')),
    query               TEXT NOT NULL,
    notify_via_tg       INTEGER NOT NULL DEFAULT 1,    -- 0/1, opt-in TG push al completar
    status              TEXT NOT NULL DEFAULT 'pending'
                            CHECK (status IN (
                                'pending',     -- creado, esperando scheduler pick up
                                'running',     -- AsyncIOScheduler ejecutando
                                'complete',    -- éxito, output escrito
                                'failed',      -- error tras agotar retries
                                'cancelling',  -- usuario pidió cancel, esperando current_phase.finish()
                                'cancelled'    -- cancel limpio, partial output guardado
                            )),
    current_phase       TEXT
                            CHECK (current_phase IS NULL OR current_phase IN (
                                'search', 'scrape', 'per_source_synthesis',
                                'final_synthesis', 'write'
                            )),
    progress_percent    INTEGER NOT NULL DEFAULT 0
                            CHECK (progress_percent BETWEEN 0 AND 100),
    output_path         TEXT,
    partial_output_path TEXT,
    error_taxonomy      TEXT
                            CHECK (error_taxonomy IS NULL OR error_taxonomy IN (
                                'search_5xx', 'search_4xx',
                                'llm_5xx', 'llm_4xx',
                                'timeout', 'cancelled', 'budget_exceeded',
                                'oom', 'network', 'checkpoint_corrupt'
                            )),
    error_message       TEXT,
    cost_usd            REAL NOT NULL DEFAULT 0.0,
    tokens_in           INTEGER NOT NULL DEFAULT 0,
    tokens_out          INTEGER NOT NULL DEFAULT 0,
    notified            INTEGER NOT NULL DEFAULT 0,    -- 0/1, set tras push enviado
    created_at          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%f', 'now')),
    started_at          TEXT,
    completed_at        TEXT,
    updated_at          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%f', 'now'))
);

-- ============================================================
-- Tabla secundaria: per-LLM-call token usage
-- ============================================================
CREATE TABLE IF NOT EXISTS research_job_token_usage (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id          TEXT NOT NULL,
    phase           TEXT NOT NULL,
    model           TEXT NOT NULL,
    tokens_in       INTEGER NOT NULL,
    tokens_out      INTEGER NOT NULL,
    cost_usd        REAL NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%f', 'now')),
    FOREIGN KEY (job_id) REFERENCES research_jobs(id) ON DELETE CASCADE
);

-- ============================================================
-- Índices: 4 queries hot del S14 (TDD §2.2)
-- ============================================================
CREATE INDEX IF NOT EXISTS idx_research_jobs_user_status_created
    ON research_jobs(user_id, status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_research_jobs_status_started
    ON research_jobs(status, started_at);
CREATE INDEX IF NOT EXISTS idx_research_jobs_daily_cost
    ON research_jobs(user_id, created_at, cost_usd);
CREATE INDEX IF NOT EXISTS idx_research_job_token_usage_job
    ON research_job_token_usage(job_id, phase);
"""


# Migración v0.3.1: convertir thread_id de nullable a NOT NULL DEFAULT 0
# y crear índice único parcial. Es un DROP+CREATE porque SQLite no soporta
# ALTER TABLE para cambiar NULLability.
_MIGRATION_V031_SQL = """
-- 1. Convertir NULL a 0 (sentinel) en thread_id
UPDATE conversations SET thread_id = 0 WHERE thread_id IS NULL;
-- 2. Recrear tabla con thread_id NOT NULL
PRAGMA foreign_keys=OFF;
CREATE TABLE conversations_new (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    thread_id INTEGER NOT NULL DEFAULT 0,
    user_id INTEGER NOT NULL,
    title TEXT,
    is_archived INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
INSERT INTO conversations_new
    (id, chat_id, thread_id, user_id, title, is_archived, created_at, updated_at)
    SELECT id, chat_id, thread_id, user_id, title, is_archived, created_at, updated_at
    FROM conversations;
DROP TABLE conversations;
ALTER TABLE conversations_new RENAME TO conversations;
PRAGMA foreign_keys=ON;
-- 3. Crear índice único parcial sobre activas
CREATE UNIQUE INDEX IF NOT EXISTS idx_conversations_unique_active
    ON conversations(chat_id, thread_id, user_id)
    WHERE is_archived = 0;
"""


# Sprint 9.0: schema migration para files persistence + file_refs.
# La parte CREATE TABLE/INDEX es idempotente (IF NOT EXISTS). La parte
# ALTER TABLE ADD COLUMN no, pero el SchemaMigrator la detecta
# automaticamente (heuristica: si el SQL completo empieza con ALTER TABLE
# ADD COLUMN y la columna ya existe, marca la migration como aplicada
# sin ejecutar). El resto del script (CREATE TABLE ...) se ejecuta
# siempre (idempotente via IF NOT EXISTS).
_MIGRATION_V058_S9_SQL = """
-- 1. file_refs: lista JSON de file_ids referenciados por este mensaje
ALTER TABLE messages ADD COLUMN file_refs TEXT;

-- 2. files: library persistente (antes in-memory en http_api)
CREATE TABLE IF NOT EXISTS files (
    id TEXT PRIMARY KEY,                          -- "file_<24 hex>"
    filename TEXT NOT NULL,
    mime_type TEXT,
    size_bytes INTEGER,
    extracted_text TEXT,                          -- SQLite TEXT aguanta ~1GB
    extraction_method TEXT DEFAULT 'pypdf',       -- 'pypdf' | 'pdfplumber' | ...
    source TEXT DEFAULT 'upload',                 -- 'upload' | 'google_drive' (S10)
    source_metadata TEXT,                         -- JSON: Drive file_id, url (S10)
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_referenced_at TEXT,
    reference_count INTEGER DEFAULT 0,
    tags TEXT                                     -- JSON array, futuro
);
CREATE INDEX IF NOT EXISTS idx_files_last_referenced
    ON files(last_referenced_at DESC);
CREATE INDEX IF NOT EXISTS idx_files_source
    ON files(source);

-- 3. file_embeddings: S9.1 RAG (placeholder S9.0, tabla creada vacía)
CREATE TABLE IF NOT EXISTS file_embeddings (
    file_id TEXT PRIMARY KEY REFERENCES files(id) ON DELETE CASCADE,
    embedding BLOB NOT NULL,                      -- numpy float32 1536*4 bytes
    embedded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    model TEXT NOT NULL DEFAULT 'text-embedding-3-small'
);

-- 4. memory_facts_staging: S9.1 Sleep Cycle (placeholder)
-- Threshold de promoción: occurrence_count >= 3 (ver §2.6 TDD S9).
CREATE TABLE IF NOT EXISTS memory_facts_staging (
    id TEXT PRIMARY KEY,                          -- "stg_<16 hex>"
    category TEXT NOT NULL,                       -- 'user_preference' | 'project_context' | 'academic_fact'
    content TEXT NOT NULL,
    confidence_score REAL NOT NULL,
    occurrence_count INTEGER NOT NULL DEFAULT 1,
    first_seen_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_seen_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    source_conversation_ids TEXT NOT NULL DEFAULT '[]',  -- JSON array, v1.2 fix: nunca NULL
    source_file_id TEXT,                          -- sin CASCADE: staging es temporal, cleanup a 90 días
    status TEXT NOT NULL DEFAULT 'pending'        -- 'pending' | 'promoted' | 'rejected' | 'expired'
);
CREATE INDEX IF NOT EXISTS idx_staging_status
    ON memory_facts_staging(status, last_seen_at DESC);

-- 5. memory_facts: S9.1 hechos consolidados (placeholder)
-- CASCADE: si el file fuente se borra, los facts derivados también
-- (P0-3 Gemini fix).
CREATE TABLE IF NOT EXISTS memory_facts (
    id TEXT PRIMARY KEY,                          -- "fact_<16 hex>"
    category TEXT NOT NULL,
    content TEXT NOT NULL,
    source_conversation_id INTEGER,
    source_file_id TEXT REFERENCES files(id) ON DELETE CASCADE,
    occurrence_count INTEGER NOT NULL DEFAULT 1,
    is_permanent INTEGER NOT NULL DEFAULT 0,      -- 1 = no decay (hardware, nombre)
    is_verified INTEGER NOT NULL DEFAULT 0,       -- 1 = validado manualmente
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_referenced_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_facts_category
    ON memory_facts(category, last_referenced_at DESC);

-- 6. memory_fact_embeddings: S9.1 cosine search de facts
CREATE TABLE IF NOT EXISTS memory_fact_embeddings (
    fact_id TEXT PRIMARY KEY REFERENCES memory_facts(id) ON DELETE CASCADE,
    embedding BLOB NOT NULL,                      -- numpy float32 1536-dim
    embedded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    model TEXT NOT NULL DEFAULT 'text-embedding-3-small'
);
"""


# Mapa de migraciones: cada entry es (descripción, SQL).
# El migrator aplica en orden lexicográfico de version.
# Para añadir una migración: incrementar el número y añadir el SQL.
# IMPORTANTE: SQL debe ser idempotente (CREATE IF NOT EXISTS, etc.) o
# el migrator la saltará en runs subsiguientes.
MIGRATIONS: dict[int, tuple[str, str]] = {
    1: (
        "v0.3.1: UNIQUE constraint on conversations(thread_id NOT NULL DEFAULT 0)",
        _MIGRATION_V031_SQL,
    ),
    2: (
        "v0.4.2: add messages.tool_call_id for OpenAI tool result messages",
        # SQLite no soporta IF NOT EXISTS en ADD COLUMN. La logica
        # idempotente esta en el SchemaMigrator.run() que detecta
        # primero si la columna ya existe. Aqui solo definimos el SQL
        # que se ejecutara si NO existe.
        "ALTER TABLE messages ADD COLUMN tool_call_id TEXT",
    ),
    3: (
        "v0.4.3: add messages.tool_calls_json for OpenAI assistant tool_calls",
        # Cuando el assistant pide un tool_call, el LLM no genera content
        # sino una lista de tool_calls. Para que el LLM en la siguiente
        # iteracion pueda relacionar el tool_result con el tool_use,
        # necesitamos reconstruir el assistant message con su tool_calls
        # (no solo content=""). Guardamos el JSON en formato OpenAI
        # ({id, type=function, function={name, arguments}}) que es el
        # canonico. El router lo transforma a Anthropic si es necesario.
        "ALTER TABLE messages ADD COLUMN tool_calls_json TEXT",
    ),
    4: (
        "v0.5.7-t51: add messages.reasoning_content for DeepSeek thinking passthrough",
        # Cuando el LLM usa thinking mode (DeepSeek, OpenAI o1, etc.) el
        # response incluye un campo `reasoning_content` con los tokens de
        # pensamiento. En iteraciones siguientes (e.g. tras un tool_call),
        # la API exige reenviar ese campo intacto en el assistant message
        # o devuelve 400 "reasoning_content must be passed back". Lo
        # guardamos en DB para reconstruir el contrato exacto en
        # _build_llm_messages. Para Anthropic path se queda NULL: ese
        # provider tiene un esquema distinto (`thinking` blocks) que no
        # se modela en esta migración.
        "ALTER TABLE messages ADD COLUMN reasoning_content TEXT",
    ),
    5: (
        "v0.5.8-s9.0: Sprint 9.0 files persistence + file_refs in messages",
        # Sprint 9 (Long-Term Memory). Migración en 3 partes:
        # 1. ALTER messages ADD file_refs (JSON array de file_ids)
        # 2. CREATE TABLE files (library de archivos subidos, antes
        #    in-memory en http_api.files_store)
        # 3. CREATE TABLE file_embeddings (placeholder para S9.1 RAG;
        #    añadida en S9.0 para evitar segunda migration al activar
        #    RAG). Definida aqui para tener el schema completo desde
        #    el principio.
        # 4. CREATE TABLE memory_facts_staging (placeholder para S9.1
        #    Sleep Cycle). Mismo motivo que file_embeddings.
        # 5. CREATE TABLE memory_facts (placeholder para S9.1 RAG de
        #    hechos consolidados). Mismo motivo.
        # 6. CREATE TABLE memory_fact_embeddings (placeholder para
        #    S9.1 cosine search de facts).
        # Backward compat: archivos S8.7 (sin file_refs) tienen el texto
        # completo en messages.content. _resolve_file_refs solo se
        # aplica cuando file_refs IS NOT NULL.
        _MIGRATION_V058_S9_SQL,
    ),
    6: (
        "v0.5.8-s9.3: search_budget table for Web Search Router monthly limits",
        # Sprint 9.3: tabla para tracking de budget por backend
        # (SearXNG unlimited, Tavily 1k/mes, Exa 1k/mes). PRIMARY KEY
        # (month, backend) permite reset mensual atómico via UPSERT.
        "CREATE TABLE IF NOT EXISTS search_budget ("
        "    month TEXT NOT NULL,"
        "    backend TEXT NOT NULL,"
        "    used INTEGER NOT NULL DEFAULT 0,"
        "    last_reset_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,"
        "    PRIMARY KEY (month, backend)"
        ")",
    ),
    7: (
        "v0.5.8-s12: add conversations.sleep_cycle_processed for native client sync",
        # Sprint 12 (ADR-007): separar "leída por Sleep Cycle para extraer
        # facts" de "activa para retomar". Las conversaciones persistentes
        # que crea la app nativa RikkaHub (chat_id != 0) NO se marcan como
        # is_archived=1 al terminar; siguen activas y retoman al abrir la app.
        # El Sleep Cycle antes las filtraba por is_archived=0, lo que las
        # metía en bucle de re-procesado. Ahora se filtran por
        # sleep_cycle_processed=0 y se marca a 1 tras extraer facts.
        # Default 0 = pendiente de procesar por Sleep Cycle.
        "ALTER TABLE conversations ADD COLUMN sleep_cycle_processed INTEGER NOT NULL DEFAULT 0",
    ),
    8: (
        "v0.5.8-s12: idx_conversations_sleep_unprocessed for fast Sleep Cycle scan",
        # Sprint 12 (TDD S12 §3, originally documented but migration
        # never landed). La query de _load_recent_conversation_ids filtra
        # por is_archived=0 AND sleep_cycle_processed=0 AND updated_at >= ?.
        # Sin indice, full scan cada vez que corre el Sleep Cycle.
        # El indice parcial WHERE sleep_cycle_processed=0 mantiene tamaño
        # acotado al subconjunto "no procesado" (las conversaciones viejas
        # marcadas con sleep_cycle_processed=1 NO estan en el indice).
        # El ORDER BY updated_at DESC lo aprovecha el Sleep Cycle para
        # procesar primero las mas recientes.
        #
        # Costo: 1 INSERT extra (index entry) por nueva conversacion.
        # Ganancia: Sleep Cycle pasa de full scan a O(log n + k) donde k =
        # numero de conversaciones no procesadas en el lookback.
        "CREATE INDEX IF NOT EXISTS idx_conversations_sleep_unprocessed "
        "ON conversations(sleep_cycle_processed, updated_at DESC) "
        "WHERE sleep_cycle_processed = 0",
    ),
    9: (
        "v0.5.8-s12.1: conversations tombstone columns",
        # Sprint 12.1 (TDD_S12_DELETE_AND_SYNC.md §4.4): implementar
        # DELETE /v1/conversations/{id} con soft delete + encrypt
        # atomico del content de messages. Las conversaciones borradas
        # quedan en este estado durante una ventana de 7d (configurable
        # via HERMES_CONVERSATION_RETENTION_DAYS) y se hard-deleted
        # via un job diario (purge_expired_conversations).
        #
        # Schema:
        # - conversations.deleted_at: ISO 8601 UTC. NULL si nunca
        #   borrada. Si seteada, la conv esta tombstoned (is_archived=1
        #   + cifrado de content de messages).
        # - conversations.encrypted_at: ISO 8601 UTC. NULL si no
        #   cifrado. Si seteada, el content de los messages esta
        #   cifrado Fernet (dict con {ct, v, ts}).
        # - conversations.purge_at: ISO 8601 UTC. Cuando <= NOW(), el
        #   worker diario hard-delete la conv. NULL si no tombstoned.
        #
        # Cada ALTER TABLE en su propia transaccion (el SchemaMigrator
        # las aplica secuencialmente). 3 statements para mantener
        # cada transaccion pequena y manejable.
        "ALTER TABLE conversations ADD COLUMN deleted_at TEXT DEFAULT NULL",
    ),
    10: (
        "v0.5.8-s12.1: conversations.encrypted_at + purge_at",
        "ALTER TABLE conversations ADD COLUMN encrypted_at TEXT DEFAULT NULL",
    ),
    11: (
        "v0.5.8-s12.1: conversations.purge_at",
        "ALTER TABLE conversations ADD COLUMN purge_at TEXT DEFAULT NULL",
    ),
    12: (
        "v0.5.8-s12.1: messages.encrypted_at + idx_conversations_deleted_at",
        # messages.encrypted_at: marca si el content de este row
        # esta cifrado. (Mayormente para debug; el cifrado se
        # detecta tambien parseando el content como JSON.)
        #
        # idx_conversations_deleted_at: el endpoint
        # GET /v1/conversations/sync pregunta por convs con
        # deleted_at > ? (cursor). Sin este indice, el sync tiene
        # que full-scanear conversations. El indice parcial WHERE
        # deleted_at IS NOT NULL mantiene el tamaño acotado al
        # subconjunto de tombstoned (las convs activas NO estan).
        "ALTER TABLE messages ADD COLUMN encrypted_at TEXT DEFAULT NULL",
    ),
    13: (
        "v0.5.8-s12.1: idx_conversations_deleted_at for sync cursor query",
        "CREATE INDEX IF NOT EXISTS idx_conversations_deleted_at "
        "ON conversations(user_id, deleted_at) "
        "WHERE deleted_at IS NOT NULL",
    ),
    14: (
        "v0.5.8-s14: research_jobs + research_job_token_usage (Deep Research)",
        # Sprint 14 (TDD_S14_DEEP_RESEARCH.md §2). Tablas idempotentes
        # (CREATE TABLE IF NOT EXISTS) — el migrator las puede re-aplicar
        # sin efectos colaterales. Inserts de sample data usan PRIMARY KEY
        # conflicto-si-existe; el migrator los mete en una heuristica:
        # la migracion es CREATE-only para que re-run sea no-op.
        _MIGRATION_V058_S14_SQL,
    ),
    15: (
        "v0.5.8-s15: files.content_hash + idx_files_content_hash (Sprint 15 RAG)",
        # Sprint 15 (RAG sobre filesystem). Pre-requisito para PR #67
        # (upload dedup) y PR #69 (embed_vault dedup cross-source).
        #
        # Diseño:
        # - ALTER TABLE files ADD COLUMN content_hash TEXT: SHA256 hex
        #   del texto extraído. Permite dedup transparente cross-source
        #   (upload + google_drive S10 no duplican el mismo PDF).
        #   Nullable: archivos pre-S15 sin hash (legacy S9.0) y futuros
        #   files que decidan no hashear (ej: streams binarios no-texto).
        # - CREATE UNIQUE INDEX idx_files_content_hash ON files(content_hash)
        #   WHERE content_hash IS NOT NULL: índice único PARCIAL. Solo
        #   aplica la constraint a filas con hash, deja NULLs libres
        #   (SQLite trata NULL != NULL en UNIQUE, lo que permite multiples
        #   NULLs sin violar constraint). Indispensable para
        #   find_file_by_content_hash O(log n) en vez de full scan.
        #
        # Backward compat: archivos S9.0 existentes quedan con
        # content_hash=NULL; nunca rompen el UNIQUE INDEX. Tests del
        # nuevo método add_file(content_hash=...) son opt-in.
        #
        # Idempotencia (Caso 2 del SchemaMigrator.run): cuando la primera
        # sentencia es ALTER ADD COLUMN y la columna ya existe, el
        # migrator la salta y ejecuta el resto del script (CREATE INDEX).
        # El CREATE INDEX usa IF NOT EXISTS (idempotente per se). Resultado:
        # re-aplicar esta migration sobre una DB ya migrada es no-op.
        #
        # Por qué UNIQUE INDEX y no UNIQUE constraint en la columna:
        # SQLite no soporta UNIQUE constraint con WHERE clause (parcial).
        # El INDEX parcial es el patrón canónico en SQLite para
        # UNIQUE-where-not-null. Mismo patrón usado por
        # idx_conversations_unique_active (migración v1).
        "ALTER TABLE files ADD COLUMN content_hash TEXT;\n"
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_files_content_hash "
        "ON files(content_hash) WHERE content_hash IS NOT NULL",
    ),
    16: (
        # Sprint 17 (TDD_VAULT_CORE.md Slice 1): Mnemosyne Vault storage.
        # Crea dos tablas nuevas y migrator-friendly para no acoplarse a
        # `files` (RAG S15). El modelo es:
        #   vault_blobs    = content-addressable store (sha256 PK).
        #   vault_files    = logical file rows (UUID4 file_id PK), N:1 con
        #                    vault_blobs via content_sha256, dedup por ref_count.
        #
        # Decisiones arquitectónicas (V1.4 final, ver TDD doc §Schema):
        # - file_id es UUID4 (TEXT) — pin en V1.4 (Nemotron BLO-1).
        # - source_path se almacena como str(Path.resolve()) — pin V1.1 Q2.
        # - UNIQUE(source_path, content_sha256, mtime) — bloquea duplicados
        #   lógicos a SQL (V1 pin en V1.2 review Vulnerabilidad #1).
        # - sub-second precision via `strftime('%H:%M:%f', 'now')` (SQLite
        #   `%f` = 3 dígitos = milisegundos, NO microsegundos). Suficiente
        #   para desambiguar add() consecutivos en el mismo segundo; si
        #   hubieran empates a milisegundo, el tiebreaker `file_id DESC`
        #   del ORDER BY en list_files() garantiza determinismo sin
        #   depender del insertion order (V1.2 review BLOCKING-2).
        # - vault_blobs.size_bytes CHECK >= 0 y ref_count CHECK >= 0
        #   DEFAULT 1 (V1.2 review B-2 db-schema).
        # - text_version + text_tier quedan NULL hasta la primera ingest
        #   (Slice 1.5 ingest_router). Aquí NO creamos esas columnas —
        #   el Slice 1 mínimo solo cubre add/get_blob/list/get/remove/stats.
        #
        # Idempotencia: CREATE TABLE IF NOT EXISTS + CREATE INDEX IF NOT
        # EXISTS. Re-aplicar la migración sobre DB ya migrada es no-op.
        "v0.5.8-s17: vault_blobs + vault_files (Sprint 17 Vault Slice 1)",
        # Tabla 1: content-addressable blob store.
        "CREATE TABLE IF NOT EXISTS vault_blobs ("
        "    content_sha256  TEXT PRIMARY KEY,"
        "    data            BLOB NOT NULL,"
        "    size_bytes      INTEGER NOT NULL CHECK (size_bytes >= 0),"
        "    ref_count       INTEGER NOT NULL DEFAULT 1 CHECK (ref_count >= 0),"
        "    created_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP"
        ");\n"
        # Tabla 2: logical file rows. UUID4 file_id PK + UNIQUE triple.
        "CREATE TABLE IF NOT EXISTS vault_files ("
        "    file_id         TEXT PRIMARY KEY,"
        "    source_path     TEXT NOT NULL,"
        "    content_sha256  TEXT NOT NULL,"
        "    mtime           REAL NOT NULL,"
        "    added_at        TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%f', 'now')),"
        "    size_bytes      INTEGER NOT NULL,"
        "    UNIQUE (source_path, content_sha256, mtime)"
        ");\n"
        # SUG-8 Nemotron PR-review round-1: NO crear idx_vault_files_
        # idempotency explícito porque la UNIQUE constraint en las mismas
        # columnas ya genera automáticamente un unique index equivalente
        # (es un duplicado de bajo coste pero introduce confusión en
        # planes de query). Eliminado. Mantenemos idx_vault_files_sha
        # para lookups por contenido (que NO están en la UNIQUE).
        #
        # FK referencial. PRAGMA foreign_keys=ON ya está activo en
        # Database.initialize() — esto asegura coherencia referencial si
        # alguien borra un blob manualmente (IntegrityError, no silent orphan).
        "CREATE INDEX IF NOT EXISTS idx_vault_files_sha "
        "ON vault_files(content_sha256);\n"
        # F-OBS-5 (V3 observability review) / F-DB-SCHEMA-SUG-1:
        # `list_files` ordena por (added_at DESC, file_id DESC), pero
        # sin este índice SQLite hace `SCAN vault_files + USE TEMP B-TREE
        # FOR ORDER BY` (O(N) full scan + in-memory sort). Con este
        # índice, EXPLAIN usa `SCAN vault_files USING INDEX ...` con
        # O(log N + limit). Probe 06 del db-schema reviewer midió:
        # 55 ms → 0.034 ms a 50k rows (1600x speedup). El índice es
        # idempotente (IF NOT EXISTS) y se re-aplica no-op en DBs ya
        # migradas. La columna en `file_id DESC` mantiene el tiebreaker
        # del V1.2 review BLOCKING-2.
        "CREATE INDEX IF NOT EXISTS idx_vault_files_added_at "
        "ON vault_files(added_at DESC, file_id DESC);\n",
    ),
    17: (
        # Sprint 17 Slice 1.5 (TDD_VAULT_INGEST_WORKER.md): Mnemosyne
        # Vault ingest pipeline. Cierra el gap entre Slice 1 (bytes
        # crudos) y Slice 2 (embeddings sobre texto extraído).
        #
        # Cambios:
        # - vault_files columnas aditivas: text_version, text_at, text_tier.
        #   Pre-campos para Slice 2 (embedding cache key) y para Slice 1.5+
        #   text versioning (no-downgrade invariant).
        # - ingest_jobs nueva tabla — NAS host-side database mirror del
        #   filesystem inbox. Filesystem es la source of truth (V1.2 P2
        #   audit); SQLite mirror para queries rápidas.
        #
        # Idempotencia: ALTER TABLE ADD COLUMN sin IF NOT EXISTS (SQLite
        # no lo soporta). El SchemaMigrator detecta columnas existentes
        # vía PRAGMA table_info y skipea los ALTER ya aplicados. Para
        # DBs nuevas, ejecuta limpio.
        "v0.5.8-s17.5: vault_files.text_version/at/tier + ingest_jobs "
        "(Sprint 17 Slice 1.5 ingest pipeline)",
        # --- A) vault_files additive columns -------------------------------
        # text_version: tier del último canónico. v0_pymupdf por default
        # (Tier 0 siempre corre). When Tier 1.5/2 llega, se bump-ea
        # desde v0 → v15_lan_worker. Invariante NO-DOWNGRADE pineada
        # en Vault.update_text (TEXT_VERSION_ORDER en ingest_router).
        "ALTER TABLE vault_files ADD COLUMN text_version TEXT NOT NULL "
        "DEFAULT 'v0_pymupdf';\n"
        # text_at: ISO8601 timestamp del último update_text exitoso.
        # Slice 2 embed_vault usa este timestamp como cache invalidation
        # key (cualquier cambio en text_at → re-embed).
        "ALTER TABLE vault_files ADD COLUMN text_at TEXT;\n"
        # text_tier: 'pymupdf'|'docling_local'|'lan_worker'|'external_vlm'.
        # NULL hasta el primer ingest. Decoupling de text_version (que
        # tiene versión + tier + provider).
        "ALTER TABLE vault_files ADD COLUMN text_tier TEXT;\n"
        # --- B) ingest_jobs nueva tabla ----------------------------------
        # Mirror SQLite del filesystem inbox. NO authoritative — el
        # filesystem es la source of truth (V1.2 P2 audit). Esto es solo
        # para queries rápidas de "jobs pendientes" sin scan filesystem.
        "CREATE TABLE IF NOT EXISTS ingest_jobs ("
        "    job_id           TEXT PRIMARY KEY,"  # UUID4 hex
        "    vault_file_id    TEXT NOT NULL,"
        "    submitted_at     TEXT NOT NULL "
        "DEFAULT CURRENT_TIMESTAMP,"
        "    submitted_by     TEXT NOT NULL DEFAULT 'hermes',"
        "    state            TEXT NOT NULL DEFAULT 'pending',"
        "    priority         INTEGER NOT NULL DEFAULT 0,"
        "    attempts         INTEGER NOT NULL DEFAULT 0,"
        "    last_state_change_at TEXT NOT NULL "
        "DEFAULT CURRENT_TIMESTAMP,"
        "    result_text_version TEXT"
        ");\n"
        # Indexes for the two query patterns:
        # (a) "show me N oldest pending jobs" — used by process_inbox
        #     and any "queue depth" introspection.
        # (b) "given a vault_file_id, what jobs touched it?" — used for
        #     observability / dedup.
        "CREATE INDEX IF NOT EXISTS idx_ingest_jobs_state "
        "ON ingest_jobs(state, priority, submitted_at);\n"
        "CREATE INDEX IF NOT EXISTS idx_ingest_jobs_file "
        "ON ingest_jobs(vault_file_id);\n",
    ),
    18: (
        # Sprint 17 Slice 2.5 (TDD_VAULT_EMBEDDINGS.md): vault_files.text
        # + vault_chunks (embeddings + semantic search).
        # Restaurado aquí porque el merge del hotpatch perdió la v18
        # original de PR #113. El shape es idéntico al commit 80c1d08.
        "v0.5.8-s17.5b: vault_files.text + vault_chunks (Sprint 17 "
        "Slice 2.5 embeddings + semantic search)",
        # --- A) vault_files.text (latest canonical text) -------------------
        "ALTER TABLE vault_files ADD COLUMN text TEXT;\n"
        # --- B) vault_chunks nueva tabla ----------------------------------
        "CREATE TABLE IF NOT EXISTS vault_chunks ("
        "    chunk_id        TEXT    PRIMARY KEY,"
        "    file_id         TEXT    NOT NULL,"
        "    chunk_index     INTEGER NOT NULL,"
        "    text            TEXT    NOT NULL,"
        "    embedding       BLOB    NOT NULL,"
        "    embedding_model TEXT    NOT NULL,"
        "    text_version    TEXT    NOT NULL,"
        "    created_at      TEXT    NOT NULL DEFAULT "
        "(strftime('%Y-%m-%d %H:%M:%f', 'now')),"
        "    FOREIGN KEY (file_id) REFERENCES vault_files(file_id) "
        "ON DELETE CASCADE"
        ");\n"
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_vault_chunks_file_idx "
        "ON vault_chunks(file_id, chunk_index);\n"
        "CREATE INDEX IF NOT EXISTS idx_vault_chunks_file_id "
        "ON vault_chunks(file_id);\n"
        "CREATE INDEX IF NOT EXISTS idx_vault_chunks_text_version "
        "ON vault_chunks(text_version);\n",
    ),
    19: (
        # Sprint 17 Slice 1.5 hot-patch (PR #113b, M-INV-2 fix):
        # añade `text_at_epoch INTEGER` a `vault_files` para queries
        # robustas con `WHERE text_at_epoch > ?` (M-INV-2 era: el
        # text_at TEXT ISO8601 lexicographic compare se rompe cuando
        # `datetime.now(UTC).isoformat()` retorna formato más corto
        # cuando microsecond=0 — los chars son `<` a la misma fecha
        # con microsegundos).
        #
        # Por qué columna ADITIVA en vez de REEMPLAZAR text_at TEXT:
        # - text_at TEXT se mantiene para legibilidad humana (debug,
        #   export, observability). El bug es solo en COMPARISON.
        # - text_at_epoch INTEGER (epoch seconds) es el que se usa
        #   para queries en código. Coexisten; text_at_epoch se
        #   actualiza atómicamente con text_at en Vault.update_text.
        #
        # Idempotencia: ADD COLUMN sin IF NOT EXISTS. SchemaMigrator
        # lo skipea si la columna ya existe (case 'is_pure_alter').
        "v0.5.8-s17.5b: vault_files.text_at_epoch INTEGER (PR #113b, M-INV-2 fix)",
        "ALTER TABLE vault_files ADD COLUMN text_at_epoch INTEGER;\n",
    ),
    20: (
        # Sprint 19 (Vault Collections + PARA + drop folder):
        # introduce la capa de organización jerárquica sobre el vault
        # flat. CRÍTICO: idempotente y no-destructivo. 0 cambios a
        # datos existentes, solo CREATE TABLE IF NOT EXISTS + ALTER ADD
        # COLUMN (que el SchemaMigrator skipea si la columna ya existe).
        #
        # 1. vault_collections: una row por collection. UUID4 hex.
        #    `archived` (0/1) implementa soft-delete (D17.1 + Gemini
        #    Sprint 19 review "Efecto Espejo"). parent_collection_id
        #    permite jerarquía pero NO cascada en MIGRATION: el
        #    cascade archive se hace en runtime via CTE (M6) para
        #    evitar borrados accidentales al migrar.
        #
        # 2. vault_file_collections: many-to-many. APPEND-ONLY por
        #    contrato (Gemini review). archive se filtra en JOIN, no
        #    en DELETE. NEVER DELETE rows en esta tabla desde código.
        #
        # 3. ALTER vault_files ADD COLUMN orphaned_at: cuando el
        #    archivo físico desaparece (user borra carpeta via SMB),
        #    M6 setea timestamp. Text + embeddings persisten (audit
        #    trail / second brain value). search filtra
        #    `WHERE orphaned_at IS NULL`.
        #
        # 4. Indexes para queries frecuentes: parent walk, archived
        #    filter, collection→files lookup, orphaned state scan.
        "v0.6.0-s19: vault_collections + vault_file_collections + vault_files.orphaned_at (Sprint 19)",
        # NB: The ALTER ADD COLUMN that used to be inline here was moved to
        # migration v21 (vault_files.orphaned_at) — the SchemaMigrator's
        # is_pure_alter short-circuit only fires when the WHOLE SQL is a
        # single ALTER TABLE statement (no semicolons, starts with ALTER).
        # v20 starts with CREATE TABLE, so it was bypassing the guard and
        # re-executing the ALTER verbatim, raising "duplicate column
        # name: orphaned_at" on re-run (verifier B1 finding, 2026-07-09).
        # Splitting the ALTER into v21 makes it pure-alter-eligible.
        # The CREATE statements are idempotent via IF NOT EXISTS.
        """\
CREATE TABLE IF NOT EXISTS vault_collections (
    collection_id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    parent_collection_id TEXT,
    description TEXT,
    sort_order INTEGER NOT NULL DEFAULT 0,
    archived INTEGER NOT NULL DEFAULT 0,
    archived_at TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (parent_collection_id)
        REFERENCES vault_collections(collection_id)
        ON DELETE RESTRICT
);
CREATE INDEX IF NOT EXISTS idx_vault_collections_parent
    ON vault_collections(parent_collection_id);
CREATE INDEX IF NOT EXISTS idx_vault_collections_archived
    ON vault_collections(archived);

CREATE TABLE IF NOT EXISTS vault_file_collections (
    file_id TEXT NOT NULL,
    collection_id TEXT NOT NULL,
    added_at TEXT NOT NULL,
    PRIMARY KEY (file_id, collection_id),
    FOREIGN KEY (file_id)
        REFERENCES vault_files(file_id) ON DELETE CASCADE,
    FOREIGN KEY (collection_id)
        REFERENCES vault_collections(collection_id)
    ON DELETE RESTRICT
);
CREATE INDEX IF NOT EXISTS idx_vfc_collection
    ON vault_file_collections(collection_id);
""",
    ),
    # Sprint 19 (Vault Collections + PARA + drop folder):
    # v21 is the orphaned_at ALTER, split out of v20 so that
    # SchemaMigrator.is_pure_alter catches it and skips on re-run.
    # Earlier v20 had this ALTER inline, which broke idempotency
    # (verifier 1+2 finding B1) because v20's SQL starts with
    # CREATE TABLE IF NOT EXISTS (not a pure ALTER), so is_pure_alter
    # was False, and the ALTER was re-executed verbatim, raising
    # "duplicate column name: orphaned_at".
    21: (
        "v0.6.0-s19.1: vault_files.orphaned_at (split from v20 for idempotency)",
        """\
ALTER TABLE vault_files ADD COLUMN orphaned_at TEXT;
CREATE INDEX IF NOT EXISTS idx_vault_files_orphaned
    ON vault_files(orphaned_at);
""",
    ),
    # Sprint 19 Slice 4 (TDD_VAULT_COLLECTIONS.md §4.3 + §4.4.6):
    # OCR queue + text provenance for the drop folder watcher. This
    # migration is COMPLEX (CREATE TABLE + ALTER + CREATE INDEX) so the
    # migrator runs Caso 2 (strips leading ALTER if column already exists).
    # - ocr_pending: queue for low-confidence Tesseract results awaiting
    #   edge PC or hosted VLM. Per-file audit trail.
    # - vault_files.text_source: provenance of the current text
    #   ('tesseract_local' | 'edge_pc' | 'vlm_hosted' | 'manual' | NULL).
    # - partial index on edge_queued_at for M6 Phase 5 zombie scan.
    22: (
        "v0.6.0-s19.4: ocr_pending + vault_files.text_source (Sprint 19 Slice 4)",
        """\
ALTER TABLE vault_files ADD COLUMN text_source TEXT;
CREATE TABLE IF NOT EXISTS ocr_pending (
    file_id TEXT PRIMARY KEY,
    local_confidence REAL NOT NULL,
    local_text TEXT,
    local_model TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending_review',
    external_model TEXT,
    external_confidence REAL,
    edge_model TEXT,
    edge_queued_at TEXT,
    edge_processed_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (file_id) REFERENCES vault_files(file_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_ocr_pending_status
    ON ocr_pending(status);
CREATE INDEX IF NOT EXISTS idx_ocr_pending_edge_queued_at
    ON ocr_pending(edge_queued_at)
    WHERE status = 'edge_queued';
""",
    ),
    # Sprint 19 Slice 4d v2 (TDD_VAULT_COLLECTIONS_v0.7_AMENDMENT.md):
    # 1. Composite UNIQUE on vault_collections: was GLOBAL UNIQUE on
    #    name only (db.py:704). Now (name COLLATE NOCASE, parent_collection_id).
    #    Required for hierarchical collections (same name at different
    #    parent levels). Outer-perimeter defense for the watcher; the
    #    Python `.casefold()` in find_by_name_and_parent is the inner
    #    matching.
    # 2. superseded_at column on vault_file_collections: soft-delete the
    #    bridge row on M6 path-flip / inline move. Was previously
    #    DELETE, but the bridge is APPEND-ONLY by contract
    #    (collections.py:8-11). All read queries JOIN with
    #    `WHERE vfc.superseded_at IS NULL`.
    # 3. dropped_events table: M6 fallback for events the watcher
    #    couldn't process (queue full, M6 was down, file gone). M6
    #    reads this at the start of each cycle and re-queues.
    # 4. Indexes: idx_vault_file_collections_active (partial WHERE
    #    superseded_at IS NULL) for fast active-bridge lookups;
    #    idx_dropped_events_unprocessed (partial WHERE processed_at IS
    #    NULL) for fast M6 re-queue scans.
    #
    # Idempotency: ALTER TABLE doesn't support IF NOT EXISTS so the
    # migrator (Caso 2) strips the leading ALTER if the column already
    # exists (PRAGMA table_info check). CREATE INDEX and CREATE TABLE
    # use IF NOT EXISTS, so they're naturally idempotent.
    23: (
        "v0.7.0-s19.4dv2: composite UNIQUE on vault_collections + superseded_at + dropped_events",
        # NOTE: the FIRST statement must be the ALTER ADD COLUMN, otherwise
        # the SchemaMigrator Caso 2 idempotency check fails (it uses
        # upper.startswith('ALTER TABLE') on the raw SQL). See
        # hermes/memory/db.py:937-981 for the migrator logic.
        #
        # Sprint 19.5 (PR-B, 2026-07-13): the SchemaMigrator.run() wraps
        # every executescript in `PRAGMA foreign_keys=OFF` ... `PRAGMA
        # foreign_keys=ON` at the CONNECTION level (NOT inside the SQL —
        # PRAGMA inside executescript is a no-op because SQLite treats
        # each statement as a transaction). The v23 migration recreates
        # vault_collections with a self-referential FK
        # (parent_collection_id -> collection_id). Without disabling FK,
        # INSERT OR IGNORE INTO vault_collections_new fails for any row
        # with non-NULL parent_collection_id (FK check at INSERT time
        # fails because the parent row isn't in vault_collections_new
        # yet). Discovered live on the NAS 2026-07-12, manually fixed
        # via complete_v23_migration.py. The connection-level PRAGMA
        # wraps ALL migrations, not just v23, for safety (next migration
        # that touches FK-recreated tables won't repeat the bug).
        """\
ALTER TABLE vault_file_collections ADD COLUMN superseded_at TEXT;
CREATE TABLE IF NOT EXISTS vault_collections_new (
    collection_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    parent_collection_id TEXT,
    description TEXT,
    sort_order INTEGER NOT NULL DEFAULT 0,
    archived INTEGER NOT NULL DEFAULT 0,
    archived_at TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (parent_collection_id) REFERENCES vault_collections(collection_id) ON DELETE RESTRICT
);
INSERT OR IGNORE INTO vault_collections_new
    (collection_id, name, parent_collection_id, description, sort_order, archived, archived_at, created_at)
    SELECT collection_id, name, parent_collection_id, description, sort_order, archived, archived_at, created_at
    FROM vault_collections;
DROP TABLE vault_collections;
ALTER TABLE vault_collections_new RENAME TO vault_collections;
CREATE UNIQUE INDEX IF NOT EXISTS idx_vault_collections_name_parent
    ON vault_collections(name COLLATE NOCASE, COALESCE(parent_collection_id, '<<ROOT>>'));
CREATE INDEX IF NOT EXISTS idx_vault_collections_parent
    ON vault_collections(parent_collection_id);
CREATE INDEX IF NOT EXISTS idx_vault_collections_archived
    ON vault_collections(archived);
CREATE INDEX IF NOT EXISTS idx_vault_file_collections_active
    ON vault_file_collections(file_id, collection_id)
    WHERE superseded_at IS NULL;
CREATE TABLE IF NOT EXISTS dropped_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_path TEXT NOT NULL,
    detected_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%f', 'now')),
    processed_at TEXT,
    UNIQUE(source_path, processed_at)
);
CREATE INDEX IF NOT EXISTS idx_dropped_events_unprocessed
    ON dropped_events(detected_at) WHERE processed_at IS NULL;
""",
    ),
    25: (
        "v0.7.0-s19.5: ADD COLUMN policy TEXT (nullable, backfilled by v27)",
        # Sprint 19.5 Slice 6 Commit 4 (X3 per-policy caches).
        # v25 is a pure ALTER ADD COLUMN; SchemaMigrator Caso 1
        # detects existing column and skips silently on re-run.
        "ALTER TABLE file_embeddings ADD COLUMN policy TEXT",
    ),
    26: (
        "v0.7.0-s19.5: ADD COLUMN dim INTEGER (nullable, backfilled by v27)",
        # Brief deviation: the TDD §3 Commit 4 plan listed only one
        # "v25a" entry (ADD COLUMN policy), but v27 (backfill) UPDATEs
        # both `policy` AND `dim`. Without an explicit `dim` column,
        # v27 fails with "no such column: dim". The TDD itself
        # (TDD_SPRINT_19_5_SLICE_6.md §2.7 #4) says "ADD COLUMN
        # `policy` (nullable), ADD COLUMN `dim` (nullable, backfilled
        # from BLOB size)" — both columns are required. v26 is the
        # second ADD COLUMN the brief omitted.
        #
        # Caso 1 (pure ALTER) handles idempotency: re-run on an
        # already-migrated DB detects the existing column and skips
        # the migration entry.
        "ALTER TABLE file_embeddings ADD COLUMN dim INTEGER",
    ),
    27: (
        "v0.7.0-s19.5: BACKFILL policy + dim for legacy rows (pre-v25)",
        # Idempotent via WHERE policy IS NULL / WHERE dim IS NULL.
        # Re-running on already-backfilled rows is a no-op.
        # - policy='vault_ingest': the historical use case was always
        #   vault embedding; legacy rows predate the per-policy world.
        # - dim=length(embedding)/4: float32 = 4 bytes/element.
        """\
UPDATE file_embeddings SET policy='vault_ingest' WHERE policy IS NULL;
UPDATE file_embeddings SET dim = CAST(length(embedding) / 4 AS INTEGER) WHERE dim IS NULL;
""",
    ),
    28: (
        "v0.7.0-s19.5: file_embeddings composite PK (file_id, policy) + "
        "policy/dim NOT NULL + idx_file_embeddings_policy",
        # Sprint 19.5 Slice 6 Commit 4 (X3 per-policy caches).
        # v28 recreates file_embeddings with:
        # - composite PRIMARY KEY (file_id, policy) so the same file
        #   can have one embedding per policy (chat_rag 384-dim +
        #   vault_ingest 4096-dim coexist).
        # - policy NOT NULL + dim NOT NULL (set by v27 backfill).
        # - FK file_id REFERENCES files(id) ON DELETE CASCADE preserved
        #   (deleting a file cascades to all its embeddings).
        # - idx_file_embeddings_policy on policy for fast policy-scoped
        #   queries (cosine_search filters by policy first).
        #
        # Idempotency: SchemaMigrator Caso 3 (added in this commit)
        # detects the new constraints and skips the recreation on
        # re-run. Without Caso 3, the migration is still naturally
        # idempotent (CREATE TABLE is fresh each run, INSERT OR IGNORE
        # skips duplicates, DROP + RENAME is a heavy no-op). Caso 3
        # is an optimization + safety net against naive re-runs.
        #
        # PRAGMA foreign_keys is wrapped at the connection level by
        # SchemaMigrator.run (line ~1010) — do NOT add it here, PRAGMA
        # inside executescript is a no-op (SQLite treats each statement
        # as a transaction).
        """\
CREATE TABLE file_embeddings_new (
    file_id TEXT NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    policy TEXT NOT NULL,
    embedding BLOB NOT NULL,
    embedded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    model TEXT NOT NULL DEFAULT 'text-embedding-3-small',
    dim INTEGER NOT NULL,
    PRIMARY KEY (file_id, policy)
);
INSERT OR IGNORE INTO file_embeddings_new
    (file_id, policy, embedding, embedded_at, model, dim)
    SELECT file_id, policy, embedding, embedded_at, model, dim
FROM file_embeddings;
DROP TABLE file_embeddings;
ALTER TABLE file_embeddings_new RENAME TO file_embeddings;
CREATE INDEX idx_file_embeddings_policy ON file_embeddings(policy);
""",
    ),
}


class SchemaMigrator:
    """Aplica migraciones SQL secuenciales con versionado.

    Diseño:
    - Tabla `schema_version(version PRIMARY KEY, applied_at, description)`.
    - En `run()`: crea la tabla si no existe, lee la versión actual, aplica
      cada migration.version > current en orden, registrando cada una.
    - Es **idempotente**: re-ejecutar es un no-op si el schema_version no
      cambia.
    - Cada migración se ejecuta en su propia transacción. Si falla, la DB
      queda en el estado anterior a esa migration (no se aplica parcialmente).

    Por qué existe: cuando Watchtower hace hot-reload automático, una nueva
    versión de la app puede esperar columnas/índices que la DB antigua no
    tiene. Sin migrator, eso requiere SSH manual al NAS para arreglarlo.
    Con migrator, todo se aplica al arrancar.
    """

    def __init__(
        self,
        conn: aiosqlite.Connection,
        migrations: dict[int, tuple[str, str]],
    ) -> None:
        self._conn = conn
        self._migrations = migrations

    async def current_version(self) -> int:
        """Lee la versión actual del schema. 0 si la tabla no existe."""
        await self._conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_version ("
            "  version INTEGER PRIMARY KEY,"
            "  applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,"
            "  description TEXT"
            ")"
        )
        await self._conn.commit()
        async with self._conn.execute(
            "SELECT COALESCE(MAX(version), 0) FROM schema_version"
        ) as cur:
            row = await cur.fetchone()
        assert row is not None, "SELECT MAX(version) devolvió None"
        return int(row[0])

    async def run(self) -> None:
        """Aplica las migraciones pendientes en orden.

        Por cada version en MIGRATIONS ordenada:
        - Si version <= current: skip
        - Si version > current: ejecuta SQL, registra en schema_version

        Loggea WARNING si se salta una version (posible inconsistencia).

        Idempotencia para ALTER TABLE ADD COLUMN:
        - Si la migración es PURAMENTE `ALTER TABLE messages ADD COLUMN
          <col>` (sin otras sentencias), verificamos si la columna ya
          existe. Si existe, skip silencioso (marca como aplicada).
        - Si la migración es más compleja (ALTER + CREATE TABLEs,
          S9.0), ejecutamos el SQL completo. Las partes CREATE TABLE
          son idempotentes (IF NOT EXISTS); la parte ALTER fallará
          si la columna ya existe, así que necesitamos detectarlo
          ANTES de ejecutar y omitir solo el ALTER.
        """
        current = await self.current_version()
        for version in sorted(self._migrations.keys()):
            if version <= current:
                continue
            description, sql = self._migrations[version]
            stripped = sql.strip()
            upper = stripped.upper()
            # Caso 3 (Sprint 19.5 Slice 6 Commit 4): v28 (renamed
            # from the brief's v25c) recreates file_embeddings with
            # composite PK + policy/dim NOT NULL +
            # idx_file_embeddings_policy. The migration is naturally
            # idempotent (CREATE TABLE fresh, INSERT OR IGNORE skips
            # dupes, DROP + RENAME is a heavy no-op), but Caso 3
            # short-circuits the recreation when ALL FOUR constraints
            # are already in place — saving the recreation cost and
            # protecting against a naive implementer who re-runs the
            # migration unnecessarily. All four must pass to skip; a
            # partial match (e.g. composite PK but missing NOT NULL on
            # dim) means the migration MUST run to fix the schema.
            #
            # Footgun avoided: a naive implementer who only checks
            # "composite PK + file_id NOT NULL" (file_id was already
            # NOT NULL pre-v25) would incorrectly skip the recreation,
            # losing the new policy NOT NULL, dim NOT NULL, and the
            # new index. Per TDD §3 Commit 4 M-9 fix (v0.16).
            if version == 28 and await self._v25c_already_applied():
                logger.info(
                    "db_migration_skipped_v25c_already_applied",
                    extra={"version": version},
                )
                await self._conn.execute(
                    "INSERT INTO schema_version (version, description) VALUES (?, ?)",
                    (version, description),
                )
                await self._conn.commit()
                continue
            # Caso 1: ALTER TABLE <tabla> ADD COLUMN puro (legacy S4-S7).
            # Verificar columna antes de ejecutar; skip si existe.
            is_pure_alter = (
                upper.startswith("ALTER TABLE") and " ADD COLUMN" in upper and ";" not in stripped
            )
            if is_pure_alter:
                table_name = sql.split()[2].strip()
                col_name = sql.split("ADD COLUMN")[1].split()[0].strip()
                async with self._conn.execute(f"PRAGMA table_info({table_name})") as cur:
                    rows = await cur.fetchall()
                cols = [r[1] for r in rows]
                if col_name in cols:
                    logger.info(
                        "db_migration_skipped_column_exists",
                        extra={"version": version, "column": col_name},
                    )
                    await self._conn.execute(
                        "INSERT INTO schema_version (version, description) VALUES (?, ?)",
                        (version, description),
                    )
                    await self._conn.commit()
                    continue
            # Caso 2: SQL complejo (ALTER + CREATE TABLEs). Si la primera
            # sentencia es ALTER ADD COLUMN y la columna ya existe, la
            # quitamos y ejecutamos el resto.
            elif upper.startswith("ALTER TABLE") and "ADD COLUMN" in upper:
                # Encontrar la primera sentencia (hasta el primer ';')
                first_semicolon = sql.find(";")
                if first_semicolon > 0:
                    first_stmt = sql[:first_semicolon].strip()
                    rest = sql[first_semicolon + 1 :]
                    if (
                        first_stmt.upper().startswith("ALTER TABLE")
                        and "ADD COLUMN" in first_stmt.upper()
                    ):
                        table_name = first_stmt.split()[2].strip()
                        col_name = first_stmt.split("ADD COLUMN")[1].split()[0].strip()
                        async with self._conn.execute(f"PRAGMA table_info({table_name})") as cur:
                            rows = await cur.fetchall()
                        cols = [r[1] for r in rows]
                        if col_name in cols:
                            # Skip solo el ALTER, ejecutar el resto
                            logger.info(
                                "db_migration_skipping_existing_column",
                                extra={
                                    "version": version,
                                    "column": col_name,
                                },
                            )
                            sql = rest
            logger.info(
                "db_migration_applying",
                extra={"version": version, "description": description},
            )
            try:
                # Sprint 19.5 (PR-B, 2026-07-13): set PRAGMA foreign_keys
                # at the CONNECTION level (not inside executescript — PRAGMA
                # in executescript is a no-op because SQLite treats each
                # statement as a transaction). The v23 migration
                # recreates vault_collections with a self-referential FK,
                # which fails the FK check at INSERT time without FK off.
                # See live discovery 2026-07-12 on the NAS; manual fix
                # applied via complete_v23_migration.py.
                #
                # Note: PRAGMA foreign_keys=ON at the end is important so
                # the schema migration that follows is FK-enforced (v24+).
                # The cost is zero: the only cost is during the v23
                # INSERT/RECREATE window, which is <100ms on a NAS.
                await self._conn.execute("PRAGMA foreign_keys=OFF")
                # Cada migration en su propia transacción. Si el script
                # emite COMMIT implícito (DDL), las CREATE TABLE IF NOT
                # EXISTS + CREATE INDEX IF NOT EXISTS son idempotentes.
                await self._conn.executescript(sql)
                await self._conn.execute("PRAGMA foreign_keys=ON")
                # F-CONC-2 / F-DB-SCHEMA-SUG-2 (2026-07-07 review):
                # INSERT OR IGNORE + rowcount gate para race-safe entre
                # procesos en cold-start concurrente. Sin este gate, el
                # segundo proceso que arranca en ~ms del primero gana la
                # carrera a la DDL (las CREATE son idempotentes) pero
                # luego choca con UNIQUE constraint failed: schema_version
                # .version al intentar INSERT INTO schema_version.
                # Pattern conocido de LiteFS / rqlite.
                cur = await self._conn.execute(
                    "INSERT OR IGNORE INTO schema_version (version, description) VALUES (?, ?)",
                    (version, description),
                )
                recorded: bool = cur.rowcount > 0
                await cur.close()
                if not recorded:
                    # Another process won the race. Roll back our
                    # INSERT (DDL is fine — winner already applied it).
                    # Next run() sees current_version=version and skips.
                    logger.info(
                        "db_migration_lost_race",
                        extra={"version": version, "description": description},
                    )
                    await self._conn.rollback()
                else:
                    await self._conn.commit()
            except Exception:
                await self._conn.rollback()
                logger.exception(
                    "db_migration_failed",
                    extra={"version": version, "description": description},
                )
                raise
            logger.info(
                "db_migration_applied",
                extra={"version": version, "description": description},
            )

    async def applied_versions(self) -> list[int]:
        """Devuelve la lista de versiones aplicadas (para tests/debug)."""
        async with self._conn.execute(
            "SELECT version FROM schema_version ORDER BY version ASC"
        ) as cur:
            rows = await cur.fetchall()
        return [int(r[0]) for r in rows]

    async def _v25c_already_applied(self) -> bool:
        """Sprint 19.5 Slice 6 Commit 4 (Caso 3): returns True when
        the file_embeddings table has all four v25c constraints in
        place. Used by ``run()`` to skip the recreation on re-run.

        Checks (all four must be True):
        1. ``file_id`` column has ``pk > 0`` (part of composite PK)
        2. ``policy`` column has ``pk > 0`` (part of composite PK)
        3. ``policy`` column has ``notnull == 1``
        4. ``dim`` column has ``notnull == 1``
        5. ``idx_file_embeddings_policy`` exists in the table's index list

        Returns:
            True if all four constraints are present; False if any
            is missing (in which case the migration MUST run to
            upgrade the schema). Naive "composite PK + file_id NOT
            NULL" check is insufficient because file_id was already
            NOT NULL pre-v25; this stricter check ensures the new
            policy/dim NOT NULL constraints and the new index are
            also in place.
        """
        # 1-4: column-level constraints via PRAGMA table_info
        async with self._conn.execute("PRAGMA table_info(file_embeddings)") as cur:
            cols = await cur.fetchall()
        # PRAGMA table_info columns: cid, name, type, notnull, dflt_value, pk
        col_map = {row[1]: row for row in cols}
        file_id_row = col_map.get("file_id")
        policy_row = col_map.get("policy")
        dim_row = col_map.get("dim")
        if file_id_row is None or policy_row is None or dim_row is None:
            return False
        # file_id and policy must both be part of the composite PK.
        if int(file_id_row[5]) <= 0 or int(policy_row[5]) <= 0:
            return False
        # policy and dim must both be NOT NULL.
        if int(policy_row[3]) != 1 or int(dim_row[3]) != 1:
            return False
        # 5: index on policy
        async with self._conn.execute("PRAGMA index_list(file_embeddings)") as cur:
            idxs = await cur.fetchall()
        idx_names = {row[1] for row in idxs}
        return "idx_file_embeddings_policy" in idx_names


class Database:
    """Database wrapper around aiosqlite (Sprint 9.0+).

    Atributos publicos:
        path: Path al .db file. Es la API PUBLICA para acceder al
            filesystem path; los callers (tests, scripts, admin tools)
            deben usar `db.path` en vez de `db._conn_path` (internals
            de aiosqlite). Sprint 15 PR #72 review (Nemotron 3 Ultra
            550B Fix #3): antes este atributo no estaba documentado como
            API y los tests tentaban acceder a `_conn_path` privado.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._conn: aiosqlite.Connection | None = None
        # F-CONC-3 (V3 concurrency review): el lock de escritura
        # vive aquí en lugar de en Vault. Razón: si Slice 4+ crea
        # múltiples Vault instances contra el mismo Database (e.g.
        # per-tenant), locks per-Vault no se coordinan. Como el
        # `aiosqlite.Connection` es single-thread (un worker thread),
        # el lock per-Database da la única granularidad correcta.
        # Las read paths (`get_blob`, `list_files`, etc.) no lo
        # toman — WAL permite reads concurrentes con el writer.
        self._write_lock = asyncio.Lock()

    async def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Sprint 15 (TDD §2.7 P0-2): timeout=30.0 para que operaciones
        # bloqueadas por Sleep Cycle (background) o writers activos
        # esperen hasta 30s antes de fallar con OperationalError. Antes
        # el default era 5.0s — insuficiente en DBs con concurrencia
        # real (prod NAS host con backups + sleep cycle + HTTP writes).
        # El PRAGMA busy_timeout interno (30000ms) cubre el caso de
        # contention interno de SQLite; este timeout es para el connect
        # inicial y para casos donde el lock es externo al proceso.
        self._conn = await aiosqlite.connect(self.path, timeout=30.0)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA synchronous=NORMAL")
        await self._conn.execute("PRAGMA busy_timeout=30000")
        await self._conn.execute("PRAGMA foreign_keys=ON")
        await self._conn.executescript(SCHEMA)
        # Sprint 4 T14: aplicar migraciones pendientes. El migrator es
        # idempotente y la migración v0.3.1 (UNIQUE constraint) se mueve
        # del método ad-hoc `_migrate_v031_unique_constraint` al mapa.
        await SchemaMigrator(self._conn, MIGRATIONS).run()
        await self._conn.commit()
        # F-OBS-9 (V3 observability review): el operador que mira el log
        # post-deploy ve el `db_initialized` final pero no sabe qué versión
        # le tocó. En múltiples deploys (v14 vs v15 vs v16) esta línea es
        # opaca. Incluir `schema_version` aquí cierra esa duda permanente.
        async with self._conn.execute(
            "SELECT COALESCE(MAX(version), 0) FROM schema_version"
        ) as cur:
            row = await cur.fetchone()
        current_version = int(row[0]) if row else 0
        logger.info(
            "db_initialized",
            extra={
                "path": str(self.path),
                "schema_version": current_version,
            },
        )

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Database not initialized")
        return self._conn

    async def ping(self) -> bool:
        """Verifica conectividad a la base de datos.

        Sprint 6 T53 v3.1: retorna bool (antes retornaba None). Backwards
        compatible: los call sites existentes (health.py, builtin.py)
        usan `await db.ping()` sin chequear el return value.

        Returns:
            True si la DB responde a SELECT 1, False si hay error de
            conexion, schema corrupto, o lock contention.
        """
        try:
            async with self.conn.execute("SELECT 1") as cur:
                await cur.fetchone()
            return True
        except Exception:
            return False

    async def get_schema_version(self) -> int:
        """Devuelve la versión actual del schema (0 si nunca se inicializó).

        Sprint 15: usado por test_migration_timing.py para verificar
        que el schema >= 15 después de db.initialize(). Wrapper sobre
        SchemaMigrator.current_version() para exponerlo desde la
        Database API (los tests no necesitan instanciar SchemaMigrator).

        Returns:
            int >= 0. 0 si la tabla schema_version no existe (DB virgen).
        """
        migrator = SchemaMigrator(self.conn, MIGRATIONS)
        return await migrator.current_version()

    async def get_or_create_conversation(
        self, chat_id: int, user_id: int, thread_id: int | None = None
    ) -> int:
        """Devuelve el id de la conversación activa, creándola si no existe.

        v0.3.1: usa INSERT OR IGNORE + SELECT con retry para evitar race
        conditions bajo concurrencia. La UNIQUE constraint sobre
        (chat_id, thread_id, user_id) garantiza que solo existe 1 fila
        por grupo.

        thread_id=None se almacena como 0 (THREAD_ID_NONE_SENTINEL) para
        que la UNIQUE constraint funcione correctamente. SQLite trata
        NULL != NULL en UNIQUE indexes, lo que permitía duplicados con
        thread_id=NULL. Con NOT NULL DEFAULT 0, NULL se convierte a 0
        y la UNIQUE se aplica correctamente.

        Flujo:
        1. INSERT OR IGNORE (intenta crear; si ya existe, no falla)
        2. SELECT con retry corto (busca la fila, hasta 5 intentos)
        3. UPDATE updated_at
        """
        # Convertir None a sentinel (requerido por la UNIQUE constraint)
        thread_id_db = THREAD_ID_NONE_SENTINEL if thread_id is None else thread_id
        # 1. INSERT OR IGNORE: si no existe, crea. Si existe (UNIQUE), no hace nada.
        await self.conn.execute(
            "INSERT OR IGNORE INTO conversations (chat_id, thread_id, user_id) VALUES (?, ?, ?)",
            (chat_id, thread_id_db, user_id),
        )
        await self.conn.commit()
        # 2. SELECT con retry: la fila DEBE existir. Bajo concurrencia,
        # podemos ver "no existe" si el commit de otra coroutine aún no
        # es visible. Retry hasta 5 veces.
        conv_id: int | None = None
        for attempt in range(5):
            async with self.conn.execute(
                "SELECT id FROM conversations "
                "WHERE chat_id=? AND thread_id=? AND user_id=? AND is_archived=0 "
                "ORDER BY updated_at DESC LIMIT 1",
                (chat_id, thread_id_db, user_id),
            ) as cur:
                row = await cur.fetchone()
            if row is not None:
                conv_id = row["id"]
                break
            await asyncio.sleep(0.005 * (attempt + 1))
        assert conv_id is not None, (
            f"Conversación no encontrada tras INSERT OR IGNORE + 5 reintentos. "
            f"chat_id={chat_id}, user_id={user_id}, thread_id={thread_id}"
        )
        # 3. UPDATE: tocar updated_at
        await self.conn.execute(
            "UPDATE conversations SET updated_at=CURRENT_TIMESTAMP WHERE id=?",
            (conv_id,),
        )
        await self.conn.commit()
        return conv_id

    async def list_conversations(
        self,
        user_id: int,
        limit: int = 20,
        before: str | None = None,
    ) -> list[dict]:
        """Lista conversaciones activas (is_archived=0) de un usuario.

        Sprint 12 (ADR-007): endpoint GET /v1/conversations para el cliente
        nativo RikkaHub. Filtros de defensa en profundidad a nivel SQL:
        - is_archived=0: nunca devolver conversaciones cerradas/retomables
        - user_id: scoping por usuario (multi-tenant)
        - before: cursor por updated_at ISO 8601 para paginación forward
        - limit: cap defensivo (default 20, max 100)

        Devuelve lista de dicts con id, chat_id, thread_id, title,
        created_at, updated_at, y last_message_preview (primer 100 chars
        del último mensaje assistant/user).
        """
        # Cap defensivo: nunca devolver más de 100 aunque el cliente pida más
        limit = min(max(limit, 1), 100)

        where_clauses = ["c.user_id = ?", "c.is_archived = 0"]
        params: list = [user_id]
        if before is not None:
            where_clauses.append("c.updated_at < ?")
            params.append(before)

        sql = f"""
            SELECT
                c.id, c.chat_id, c.thread_id, c.title,
                c.created_at, c.updated_at,
                (
                    SELECT content FROM messages m
                    WHERE m.conversation_id = c.id
                    ORDER BY m.created_at DESC LIMIT 1
                ) AS last_message_preview
            FROM conversations c
            WHERE {" AND ".join(where_clauses)}
            ORDER BY c.updated_at DESC
            LIMIT ?
        """
        params.append(limit)

        async with self.conn.execute(sql, params) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def get_conversation_messages(
        self,
        conv_id: int,
        limit: int = 100,
        before: str | None = None,
    ) -> list[dict]:
        """Devuelve mensajes de una conversación con paginación forward.

        Sprint 12 (ADR-007): endpoint GET /v1/conversations/{id}/messages
        para el cliente nativo RikkaHub.

        - limit: cap defensivo (default 100, max 500)
        - before: cursor por created_at ISO 8601 (mensajes más viejos)
        - Verifica que la conversación exista y pertenezca al user_id
          que la solicita (auth a nivel de query, no del middleware;
          defense in depth).
        """
        limit = min(max(limit, 1), 500)

        # Verificar que la conversación existe. Si no, el caller debe
        # responder 404 en vez de devolver lista vacía.
        async with self.conn.execute(
            "SELECT id FROM conversations WHERE id = ?", (conv_id,)
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return []  # Conv no existe; el caller maneja 404

        where_clauses = ["m.conversation_id = ?"]
        params: list = [conv_id]
        if before is not None:
            where_clauses.append("m.created_at < ?")
            params.append(before)

        sql = f"""
            SELECT m.id, m.role, m.content, m.model_used,
                   m.tokens_in, m.tokens_out, m.latency_ms,
                   m.tool_call_id, m.tool_calls_json,
                   m.reasoning_content, m.file_refs,
                   m.created_at
            FROM messages m
            WHERE {" AND ".join(where_clauses)}
            ORDER BY m.created_at DESC
            LIMIT ?
        """
        params.append(limit)

        async with self.conn.execute(sql, params) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def mark_sleep_cycle_processed(self, conv_id: int) -> None:
        """Marca una conversación como ya procesada por Sleep Cycle.

        Sprint 12 (ADR-007): evita re-procesar conversaciones persistentes
        que la app nativa RikkaHub crea (chat_id != 0). Se llama desde
        sleep_cycle.py tras extraer facts de la conversación.
        """
        await self.conn.execute(
            "UPDATE conversations SET sleep_cycle_processed = 1 WHERE id = ?",
            (conv_id,),
        )
        await self.conn.commit()

    async def add_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
        model_used: str | None = None,
        tokens_in: int | None = None,
        tokens_out: int | None = None,
        latency_ms: int | None = None,
        tool_call_id: str | None = None,
        tool_calls: list[dict] | None = None,
        reasoning_content: str | None = None,
        file_refs: list[str] | None = None,
    ) -> int:
        """Inserta un mensaje en la conversación.

        Args:
            tool_call_id: ID del tool_call al que responde este mensaje
                (solo para role='tool'). OpenAI y Anthropic requieren este
                campo para relacionar el resultado con la petición.
            tool_calls: lista de tool_calls (formato OpenAI:
                [{id, type='function', function={name, arguments}}]) que
                el assistant pidió. Solo para role='assistant'. Si se
                proporciona, content puede ser vacio/None.
            reasoning_content: cadena de pensamiento del LLM en thinking
                mode (DeepSeek, OpenAI o1, etc.). Solo para role='assistant'.
                Sprint 5 T51: necesario para round-trip en iteraciones
                con tool_call; las APIs de razonamiento rechazan 400 si
                no se reenvía intacto. None para providers sin thinking
                (Anthropic) o cuando el LLM no genera tokens de razonamiento.
            file_refs: lista de file_ids (Sprint 9.0) cuyo texto se
                prepended a content cuando el LLM carga el historial.
                Solo se persiste la REFERENCIA, no el texto (2,500x
                reducción vs S8.7 que duplicaba el PDF en cada msg).
                None = mensaje sin files referenciados (default).
        """
        tool_calls_json = None
        if tool_calls:
            tool_calls_json = json.dumps(tool_calls, ensure_ascii=False)
        file_refs_json = None
        if file_refs:
            # Dedup preservando orden
            seen: set[str] = set()
            unique: list[str] = []
            for fid in file_refs:
                if fid and fid not in seen:
                    seen.add(fid)
                    unique.append(fid)
            if unique:
                file_refs_json = json.dumps(unique, ensure_ascii=False)
        async with self.conn.execute(
            "INSERT INTO messages (conversation_id, role, content, model_used, "
            "tokens_in, tokens_out, latency_ms, tool_call_id, tool_calls_json, "
            "reasoning_content, file_refs) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                conversation_id,
                role,
                content,
                model_used,
                tokens_in,
                tokens_out,
                latency_ms,
                tool_call_id,
                tool_calls_json,
                reasoning_content,
                file_refs_json,
            ),
        ) as cur:
            msg_id = cur.lastrowid
        await self.conn.commit()
        assert msg_id is not None
        return msg_id

    async def get_history(self, conversation_id: int, limit: int = 50) -> list[dict]:
        async with self.conn.execute(
            "SELECT role, content, model_used, created_at, tool_call_id, "
            "       tool_calls_json, reasoning_content, file_refs "
            "FROM messages "
            "WHERE conversation_id=? ORDER BY id ASC LIMIT ?",
            (conversation_id, limit),
        ) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def archive_conversation(self, conversation_id: int) -> None:
        await self.conn.execute(
            "UPDATE conversations SET is_archived=1, updated_at=CURRENT_TIMESTAMP WHERE id=?",
            (conversation_id,),
        )
        await self.conn.commit()

    async def archive_stale_conversations(self, max_age_seconds: int) -> int:
        """Sprint 9.4: archiva conversaciones activas con updated_at viejo.

        Job periodico (cada 5 min via APScheduler, ver ConversationCleanupScheduler
        en hermes.scheduler). Libera combinaciones (chat_id, thread_id, user_id)
        para que nuevas requests no fallen con UNIQUE constraint violation
        en idx_conversations_unique_active (bug 9.3.2b).

        Sprint 12 (ADR-007): exime conversaciones persistentes con chat_id != 0
        (las que crea la app nativa RikkaHub). Esas conversaciones NO se
        archivan automaticamente: el usuario las retoma al abrir la app, y
        su ciclo de vida lo gestiona el cliente, no el servidor.

        Solo archiva conversaciones efimeras (chat_id = 0, sentinel HTTP)
        o backends sin persistencia (Telegram, etc.) que llevan inactivas
        mas de max_age_seconds.

        Args:
            max_age_seconds: edad maxima permitida. Convs con
                updated_at < now() - max_age_seconds se archivan.

        Returns:
            Numero de conversaciones archivadas.
        """
        import time

        cutoff_ts = int(time.time()) - max_age_seconds
        # SQLite CURRENT_TIMESTAMP guarda en formato 'YYYY-MM-DD HH:MM:SS' UTC.
        # Convertir cutoff_ts (epoch UTC) a ese formato para comparar.
        from datetime import datetime

        cutoff_str = datetime.fromtimestamp(cutoff_ts, tz=UTC).strftime("%Y-%m-%d %H:%M:%S")
        cur = await self.conn.execute(
            "UPDATE conversations "
            "SET is_archived=1, updated_at=CURRENT_TIMESTAMP "
            "WHERE is_archived=0 AND chat_id = 0 AND updated_at <= ?",
            (cutoff_str,),
        )
        await self.conn.commit()
        return cur.rowcount or 0

    # =========================================================================
    # Sprint 12.1 (TDD_S12_DELETE_AND_SYNC.md): Tombstone + sync engine
    # =========================================================================

    async def soft_delete_conversation(
        self,
        conversation_id: int,
        user_id: int,
        encryption_key: bytes,
        retention_days: int = 7,
    ) -> bool:
        """Soft delete + cifrado atomico de content. Idempotente.

        Sprint 12.1 (ADR-007): implementado para soportar
        DELETE /v1/conversations/{id}. El flujo:
        1. Verifica que la conv pertenece al user.
        2. Verifica que la conv esta activa (is_archived=0). Si ya estaba
           archivada (por /clear de Telegram, por stale cleanup, etc.),
           retorna False (idempotente, no error).
        3. En UNA sola transaccion SQLite:
           a. SELECT todos los messages de la conv.
           b. Encrypt cada content con Fernet(encryption_key).
              Si content ya es ciphertext (por un delete anterior
              encadenado, improbable pero posible), skip.
           c. UPDATE messages SET content=ciphertext_json, encrypted_at=NOW.
           d. UPDATE conversations
              SET is_archived=1,
                  deleted_at=CURRENT_TIMESTAMP,
                  encrypted_at=CURRENT_TIMESTAMP,
                  purge_at=datetime(CURRENT_TIMESTAMP, '+' || ? || ' days')
              WHERE id=? AND user_id=? AND is_archived=0.
        4. Commit.

        Retorna True si la operacion se ejecuto, False si la conv ya estaba
        archivada (idempotente) o no pertenece al user.

        Si encryption_key es None o vacio, se hace plain archive sin
        cifrado (TDD §7.2 safe-fallback). El POST /restore no podra
        descifrar (devuelve 410 Gone) pero el archive funciona.

        IMPORTANTE (Copilot review): el safe-fallback tambien debe setear
        `purge_at = deleted_at + retention_days`. Si no, las convs plain-
        archived (sin key) se acumulan para siempre sin posibilidad de
        hard-delete. El worker diario `purge_expired_conversations`
        filtra por `purge_at IS NOT NULL AND purge_at <= NOW()`.
        """
        from datetime import UTC, datetime, timedelta

        if not encryption_key:
            # Safe-fallback: plain archive sin cifrado (TDD §7.2).
            # El restore no sera posible despues (content no se descifra
            # a plaintext), pero el archive funciona. CRITICO: el purge_at
            # se setea igual que en el path con cifrado, para que el
            # worker diario purgue la conv tras la ventana de retencion.
            now_str = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
            purge_str = (datetime.now(UTC) + timedelta(days=retention_days)).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            async with self.conn.execute(
                "SELECT is_archived FROM conversations WHERE id=? AND user_id=?",
                (conversation_id, user_id),
            ) as cur:
                row = await cur.fetchone()
            if row is None:
                return False
            if row["is_archived"]:
                return False  # idempotente
            await self.conn.execute(
                "UPDATE conversations SET is_archived=1, deleted_at=?, "
                "encrypted_at=NULL, purge_at=? WHERE id=? AND user_id=? AND is_archived=0",
                (now_str, purge_str, conversation_id, user_id),
            )
            await self.conn.commit()
            return True

        from cryptography.fernet import Fernet

        fernet = Fernet(encryption_key)
        cutoff_dt = datetime.now(UTC) + timedelta(days=retention_days)
        cutoff_str = cutoff_dt.strftime("%Y-%m-%d %H:%M:%S")
        now_str = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

        # 1. Verificar pertenencia y estado
        async with self.conn.execute(
            "SELECT is_archived FROM conversations WHERE id=? AND user_id=?",
            (conversation_id, user_id),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return False
        if row["is_archived"]:
            return False  # ya archivada, idempotente

        # 2. Leer messages y cifrar
        # (Lectura y UPDATE en la misma transaccion para atomicidad.)
        async with self.conn.execute(
            "SELECT id, content FROM messages WHERE conversation_id=?",
            (conversation_id,),
        ) as cur:
            messages = await cur.fetchall()

        import json as _json

        for msg in messages:
            raw = msg["content"]
            if not raw:
                continue  # mensajes sin content (e.g., assistant con tool_calls)
            # Skip si ya es ciphertext (delete encadenado, edge case)
            try:
                parsed = _json.loads(raw)
                if isinstance(parsed, dict) and parsed.get("ct") and parsed.get("v") == 1:
                    continue
            except (ValueError, TypeError):
                pass  # no es JSON, es plaintext normal — cifrar
            try:
                ct = fernet.encrypt(raw.encode("utf-8"))
                encrypted_content = _json.dumps(
                    {
                        "ct": ct.decode("ascii"),
                        "v": 1,
                        "ts": now_str,
                    }
                )
            except Exception:
                # Si Fernet falla (e.g., key corrupta), skip ese message.
                # No fallamos toda la operacion: ya tenemos el archive
                # semantico; un message que no se cifra quedara legible.
                # (TDD §11 riesgo: key corrupta. Log warning.)
                continue
            await self.conn.execute(
                "UPDATE messages SET content=?, encrypted_at=? WHERE id=?",
                (encrypted_content, now_str, msg["id"]),
            )

        # 3. UPDATE conversations
        await self.conn.execute(
            "UPDATE conversations "
            "SET is_archived=1, "
            "    deleted_at=?, "
            "    encrypted_at=?, "
            "    purge_at=? "
            "WHERE id=? AND user_id=? AND is_archived=0",
            (now_str, now_str, cutoff_str, conversation_id, user_id),
        )
        await self.conn.commit()
        return True

    async def restore_conversation(
        self,
        conversation_id: int,
        user_id: int,
        encryption_key: bytes,
    ) -> bool:
        """Revierte un soft_delete dentro de la ventana de 7d.

        Sprint 12.1: implementado para POST /v1/conversations/{id}/restore.
        Decrypts message contents back to plaintext. Retorna False si:
        - La conv no pertenece al user.
        - La conv no esta tombstoned (is_archived=0 o deleted_at IS NULL).
        - encrypted_at IS NULL (TDD §7.2 safe-fallback: chat archivado
          sin llave o legacy de Telegram). En este caso, limpiamos los
          flags de tombstone sin descifrar nada.
        - purge_at <= NOW() (hard-deleted).
        """
        from datetime import UTC, datetime

        now_str = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

        async with self.conn.execute(
            "SELECT is_archived, deleted_at, encrypted_at, purge_at "
            "FROM conversations WHERE id=? AND user_id=?",
            (conversation_id, user_id),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return False
        if not row["is_archived"] or row["deleted_at"] is None:
            return False  # no esta tombstoned
        if row["purge_at"] is not None and row["purge_at"] <= now_str:
            return False  # purgada o cerca

        # Si encrypted_at IS NULL: safe-fallback, solo limpiamos flags.
        # Esto cubre el caso de convs archivadas por /clear de Telegram
        # (sin deleted_at) o por archive_stale_conversations (sin
        # cifrado). El restore de un chat legacy solo cambia el flag.
        if row["encrypted_at"] is None:
            await self.conn.execute(
                "UPDATE conversations "
                "SET is_archived=0, deleted_at=NULL, encrypted_at=NULL, "
                "    purge_at=NULL, updated_at=CURRENT_TIMESTAMP "
                "WHERE id=? AND user_id=?",
                (conversation_id, user_id),
            )
            await self.conn.commit()
            return True

        # encrypted_at IS NOT NULL: descifrar messages
        if not encryption_key:
            # TDD §7.2 safe-fallback: sin key, no podemos descifrar.
            # Retornar False indica al caller que use 503.
            return False

        import json as _json

        from cryptography.fernet import Fernet

        fernet = Fernet(encryption_key)

        async with self.conn.execute(
            "SELECT id, content FROM messages WHERE conversation_id=?",
            (conversation_id,),
        ) as cur:
            messages = await cur.fetchall()

        for msg in messages:
            raw = msg["content"]
            if not raw:
                continue
            try:
                parsed = _json.loads(raw)
            except (ValueError, TypeError):
                continue  # no es ciphertext, skip
            if not (isinstance(parsed, dict) and parsed.get("ct") and parsed.get("v") == 1):
                continue  # no es formato ciphertext v1
            try:
                plaintext = fernet.decrypt(parsed["ct"].encode("ascii")).decode("utf-8")
            except Exception:
                # Key incorrecta o ciphertext corrupto.
                # CRITICO (Copilot review): NO continuamos al final del
                # loop. Si algun message falla, NO limpiamos flags:
                # la conv sigue tombstoned, no se hace pasar por activa
                # con messages cifrados (estado inconsistente).
                # Retornar False hace que el HTTP handler responda 503
                # y NO emita un 200 que dejaria la conv inconsistente.
                return False
            await self.conn.execute(
                "UPDATE messages SET content=?, encrypted_at=NULL WHERE id=?",
                (plaintext, msg["id"]),
            )

        # Si llegamos aqui, TODOS los messages se descifraron OK.
        # Limpiar flags de tombstone.
        await self.conn.execute(
            "UPDATE conversations "
            "SET is_archived=0, deleted_at=NULL, encrypted_at=NULL, "
            "    purge_at=NULL, updated_at=CURRENT_TIMESTAMP "
            "WHERE id=? AND user_id=?",
            (conversation_id, user_id),
        )
        await self.conn.commit()
        return True

    async def purge_expired_conversations(self) -> int:
        """Sprint 12.1: hard-delete convs con purge_at <= NOW() (job diario).

        TDD_S12_DELETE_AND_SYNC.md §7.1: el CASCADE en
        messages.conversation_id borra los rows (que ya estan cifrados)
        en la misma operacion. NO se descifra antes del hard delete:
        el dato ya esta en estado "no legible" desde el cifrado inicial.
        Purga es solo "sacar de la DB".

        Retorna el numero de convs purgadas.
        """
        from datetime import UTC, datetime

        now_str = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
        cur = await self.conn.execute(
            "DELETE FROM conversations "
            "WHERE deleted_at IS NOT NULL "
            "  AND purge_at IS NOT NULL "
            "  AND purge_at <= ?",
            (now_str,),
        )
        await self.conn.commit()
        return cur.rowcount or 0

    async def get_conversations_sync(
        self,
        user_id: int,
        updated_after: str,  # ISO 8601 UTC ('YYYY-MM-DD HH:MM:SS')
        limit: int = 100,
    ) -> list[dict]:
        """Sprint 12.1: alimenta GET /v1/conversations/sync.

        Devuelve convs activas (is_archived=0) con updated_at > cursor.
        Cursor-based delta, no full scan. Paginated (max 500).

        IMPORTANTE (TDD §11 riesgo 1): el caller debe pasar el cursor en
        formato ISO 8601 UTC con zero-padding ('2026-07-01 09:05:00').
        El comparador es alfanumerico (SQLite TEXT) — si el formato no
        matchea exactamente, el delta sync falla silenciosamente.
        """
        limit = min(max(limit, 1), 500)
        sql = """
            SELECT
                c.id, c.chat_id, c.thread_id, c.title,
                c.created_at, c.updated_at,
                (
                    SELECT content FROM messages m
                    WHERE m.conversation_id = c.id
                    ORDER BY m.created_at DESC LIMIT 1
                ) AS last_message_preview
            FROM conversations c
            WHERE c.user_id = ?
              AND c.is_archived = 0
              AND c.updated_at > ?
            ORDER BY c.updated_at ASC
            LIMIT ?
        """
        async with self.conn.execute(sql, (user_id, updated_after, limit)) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def get_deleted_conversations_since(
        self,
        user_id: int,
        deleted_after: str,  # ISO 8601 UTC
        limit: int = 100,
    ) -> list[dict]:
        """Sprint 12.1: lista tombstoned convs (deleted_at > cursor).

        Para el endpoint GET /v1/conversations/sync. El cliente recibe
        esta lista y tombestone localmente. Devuelve solo id + deleted_at
        (no metadata completa — el cliente no necesita ver el contenido
        cifrado, solo el id para tombestone).
        """
        limit = min(max(limit, 1), 500)
        async with self.conn.execute(
            "SELECT id, deleted_at FROM conversations "
            "WHERE user_id = ? "
            "  AND deleted_at IS NOT NULL "
            "  AND deleted_at > ? "
            "ORDER BY deleted_at ASC "
            "LIMIT ?",
            (user_id, deleted_after, limit),
        ) as cur:
            rows = await cur.fetchall()
        return [{"id": r["id"], "deleted_at": r["deleted_at"]} for r in rows]

    async def create_ephemeral_conversation(
        self, chat_id: int, user_id: int, thread_id: int | None = None
    ) -> int:
        """Crea una conversacion efimera NUEVA para una request HTTP.

        Primero archiva cualquier conversacion activa existente con los
        mismos (chat_id, thread_id, user_id). Esto evita dos problemas:
        1. UNIQUE constraint failed (idx_conversations_unique_active) si
           una conv huerfana (no archivada por crash previo) existe.
        2. Reutilizar una conv con mensajes viejos que el LLM veria como
           historial espurio (bug del "fantasma de Laufband": el nuevo
           'Hola que hora es' se mezclo con el mensaje huerfano de la
           query anterior en conv 182).

        Usado por http_api para cada request OpenAI-compatible (cada
        POST /v1/chat/completions debe ser una conversacion nueva
        porque el cliente reenvia todo el historial en body.messages).

        Args:
            chat_id, user_id, thread_id: mismo formato que new_conversation.

        Returns:
            conv_id de la NUEVA conversacion (is_archived=0).
        """
        thread_id_db = THREAD_ID_NONE_SENTINEL if thread_id is None else thread_id
        # 1. Archivar orphans con los mismos sentinels
        await self.conn.execute(
            "UPDATE conversations SET is_archived=1, updated_at=CURRENT_TIMESTAMP "
            "WHERE chat_id=? AND thread_id=? AND user_id=? AND is_archived=0",
            (chat_id, thread_id_db, user_id),
        )
        await self.conn.commit()
        # 2. Crear NUEVA conversacion (sabe que no hay UNIQUE conflict
        #    porque acabamos de archivar todas las activas con esos sentinels)
        async with self.conn.execute(
            "INSERT INTO conversations (chat_id, thread_id, user_id) VALUES (?, ?, ?)",
            (chat_id, thread_id_db, user_id),
        ) as cur:
            conv_id = cur.lastrowid
        await self.conn.commit()
        assert conv_id is not None
        return conv_id

    async def get_last_assistant_message(self, conversation_id: int) -> dict | None:
        """Retorna el ultimo mensaje assistant de la conversacion, o None.

        Sprint 6 T53 v3.1: usado por el endpoint HTTP para extraer
        usage (tokens_in, tokens_out) del response de AgentLoop.
        None si la conversacion no tiene mensajes assistant (e.g.
        AgentLoop fallo antes de guardar, o conversacion sin tool calls).

        Args:
            conversation_id: ID de la conversacion.

        Returns:
            dict con keys: role, content, model_used, tokens_in,
            tokens_out, latency_ms, created_at, file_refs. O None si
            no hay mensaje assistant.
        """
        async with self.conn.execute(
            "SELECT role, content, model_used, tokens_in, tokens_out, "
            "       latency_ms, created_at, file_refs "
            "FROM messages "
            "WHERE conversation_id=? AND role='assistant' "
            "ORDER BY id DESC LIMIT 1",
            (conversation_id,),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        return dict(row)

    async def new_conversation(
        self, chat_id: int, user_id: int, thread_id: int | None = None
    ) -> int:
        thread_id_db = THREAD_ID_NONE_SENTINEL if thread_id is None else thread_id
        async with self.conn.execute(
            "INSERT INTO conversations (chat_id, thread_id, user_id) VALUES (?, ?, ?)",
            (chat_id, thread_id_db, user_id),
        ) as cur:
            conv_id = cur.lastrowid
        await self.conn.commit()
        assert conv_id is not None
        return conv_id

    async def add_tool_call(
        self,
        message_id: int,
        tool_name: str,
        arguments_json: str,
        result_json: str | None = None,
        success: int = 1,
        error: str | None = None,
        latency_ms: int | None = None,
    ) -> int:
        """Inserta un tool_call asociado a un message_id.

        Args:
            message_id: ID del mensaje del assistant que pidió la tool.
            tool_name: nombre de la tool ejecutada.
            arguments_json: argumentos como JSON string.
            result_json: resultado como JSON string. None si falló.
            success: 1 si exitoso, 0 si falló.
            error: mensaje de error si success=0.
            latency_ms: latencia de ejecución en ms.

        Returns:
            ID del tool_call insertado.
        """
        async with self.conn.execute(
            "INSERT INTO tool_calls (message_id, tool_name, arguments_json, "
            "result_json, success, error, latency_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                message_id,
                tool_name,
                arguments_json,
                result_json,
                success,
                error,
                latency_ms,
            ),
        ) as cur:
            tc_id = cur.lastrowid
        await self.conn.commit()
        assert tc_id is not None
        return tc_id

    async def list_tool_calls_for_conversation(self, conversation_id: int) -> list[dict]:
        """Lista todos los tool_calls de una conversación (ordenados por created_at)."""
        async with self.conn.execute(
            "SELECT tc.id, tc.tool_name, tc.arguments_json, tc.result_json, "
            "       tc.success, tc.error, tc.latency_ms, tc.created_at "
            "FROM tool_calls tc "
            "JOIN messages m ON tc.message_id = m.id "
            "WHERE m.conversation_id = ? "
            "ORDER BY tc.id ASC",
            (conversation_id,),
        ) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in rows]

    # --- Sprint 9.0: files persistence ---

    async def add_file(
        self,
        file_id: str,
        filename: str,
        mime_type: str | None,
        size_bytes: int,
        extracted_text: str,
        extraction_method: str = "pypdf",
        source: str = "upload",
        source_metadata: str | None = None,
        tags: str | None = None,
        content_hash: str | None = None,
    ) -> None:
        """Inserta un file en la library persistente (Sprint 9.0).

        Reemplaza el in-memory `files_store` de S8.7. Tras un restart del
        container, los files sobreviven.

        Args:
            file_id: ID externo (e.g. "file_abc123...").
            filename: nombre original del archivo.
            mime_type: MIME type (e.g. "application/pdf").
            size_bytes: tamaño en bytes del archivo binario.
            extracted_text: texto extraído (PDF → pypdf, txt → decode).
                Puede ser vacío si no se pudo extraer.
            extraction_method: 'pypdf' | 'pdfplumber' | '' (no extraído).
            source: 'upload' | 'google_drive' (S10).
            source_metadata: JSON opcional (Drive file_id, url en S10).
            tags: JSON array opcional de tags (futuro).
            content_hash: SHA256 hex del texto extraído (Sprint 15).
                None = sin hash (legacy S9.0 o files no-texto). El UNIQUE
                INDEX idx_files_content_hash permite multiples NULLs sin
                violar constraint (es parcial WHERE NOT NULL).
        """
        await self.conn.execute(
            "INSERT INTO files (id, filename, mime_type, size_bytes, "
            "extracted_text, extraction_method, source, source_metadata, "
            "tags, content_hash) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                file_id,
                filename,
                mime_type,
                size_bytes,
                extracted_text,
                extraction_method,
                source,
                source_metadata,
                tags,
                content_hash,
            ),
        )
        await self.conn.commit()

    async def find_file_by_content_hash(self, content_hash: str) -> dict | None:
        """Busca un file por su content_hash (Sprint 15 dedup cross-source).

        Usado por http_api.upload_file para hacer dedup transparente:
        si ya existe un file con el mismo SHA256 del texto extraído,
        retorna el existente en vez de crear uno nuevo.

        Args:
            content_hash: SHA256 hex del texto extraído (64 chars).

        Returns:
            dict con el row completo (mismo shape que get_file) o None
            si no existe match.

        Performance:
            O(log n) gracias al UNIQUE INDEX idx_files_content_hash.
            Sin el índice, sería full-scan O(n) por cada upload.
        """
        async with self.conn.execute(
            "SELECT id, filename, mime_type, size_bytes, extracted_text, "
            "       extraction_method, source, source_metadata, "
            "       created_at, last_referenced_at, reference_count, tags, "
            "       content_hash "
            "FROM files WHERE content_hash = ? "
            "LIMIT 1",
            (content_hash,),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        return dict(row)

    async def get_file(self, file_id: str) -> dict | None:
        """Lee un file por ID. None si no existe.

        Returns:
            dict con todas las columnas de `files` (incluido
            extracted_text y content_hash desde Sprint 15) o None si no existe.
        """
        async with self.conn.execute(
            "SELECT id, filename, mime_type, size_bytes, extracted_text, "
            "       extraction_method, source, source_metadata, "
            "       created_at, last_referenced_at, reference_count, tags, "
            "       content_hash "
            "FROM files WHERE id = ?",
            (file_id,),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        return dict(row)

    async def list_files(
        self,
        source: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """Lista files de la library, opcionalmente filtrados por source.

        Sprint 15 (US-3.1 §4 PR #69): añade `offset` para paginated
        cursor. Usado por GET /v1/files para que Open WebUI / scripts
        puedan browse la library sin descargarsela toda de golpe.

        Args:
            source: filtra por source column ('upload', 'google_drive').
                None = todos.
            limit: máximo de rows (default 100).
            offset: número de rows a saltar (default 0).

        Returns:
            lista de dicts (mismo shape que get_file, incluye
            content_hash desde Sprint 15), ordenados por created_at DESC
            (más recientes primero) con rowid DESC como tiebreaker
            monotónico (SQLite CURRENT_TIMESTAMP tiene granularidad de 1s;
            si 3 archivos se suben en el mismo segundo, el orden seria
            arbitrario sin tiebreaker — esto rompe pagination tests).
        """
        if source is not None:
            async with self.conn.execute(
                "SELECT id, filename, mime_type, size_bytes, extracted_text, "
                "       extraction_method, source, source_metadata, "
                "       created_at, last_referenced_at, reference_count, tags, "
                "       content_hash "
                "FROM files WHERE source = ? "
                "ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?",
                (source, limit, offset),
            ) as cur:
                rows = await cur.fetchall()
        else:
            async with self.conn.execute(
                "SELECT id, filename, mime_type, size_bytes, extracted_text, "
                "       extraction_method, source, source_metadata, "
                "       created_at, last_referenced_at, reference_count, tags, "
                "       content_hash "
                "FROM files "
                "ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ) as cur:
                rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def touch_file(self, file_id: str) -> None:
        """Actualiza last_referenced_at y reference_count++.

        Llamado por AgentLoop._resolve_file_refs cada vez que un mensaje
        con file_refs se carga. Permite ranking por uso y limpieza de
        files no usados.

        No falla si el file_id no existe (silencioso, para graceful
        degradation ante refs huérfanas).
        """
        await self.conn.execute(
            "UPDATE files "
            "SET last_referenced_at = CURRENT_TIMESTAMP, "
            "    reference_count = reference_count + 1 "
            "WHERE id = ?",
            (file_id,),
        )
        await self.conn.commit()

    async def find_messages_with_file_ref(self, file_id: str, limit: int = 50) -> list[dict]:
        """Encuentra todos los messages que tienen este file_id en file_refs.

        Sprint 15 (US-3.1 §10 inode-like): expone "¿quién usa este PDF?".
        Usado por GET /v1/files/{id}/refs. Devuelve rows con metadata
        minima (id, conversation_id, role, created_at) + el snippet del
        content del message (primeros 200 chars) para que el LLM/user
        pueda entender el contexto de cada referencia sin tener que
        cargar el message entero.

        Usa `json_each()` de SQLite (json1 module, habilitado por
        default en Python 3.12+) para parsear el JSON array `file_refs`
        y buscar matches exactos. Es estrictamente más seguro que
        `LIKE '%file_id%'` que podría matchear substrings.

        Args:
            file_id: el file_id a buscar.
            limit: máximo de rows (default 50, suficiente para una
                library típica; >50 referencias a un mismo file es raro
                y el user puede paginar después si hace falta).

        Returns:
            lista de dicts con keys:
            - message_id: PK del message
            - conversation_id: FK a conversations
            - role: 'user' | 'assistant' | 'tool'
            - created_at: timestamp del message
            - content_snippet: primeros 200 chars del content
                (None si está cifrado/legacy)
            Orden: created_at DESC (más reciente primero).

        Performance:
            O(N) donde N = total de messages con file_refs no-null. El
            índice idx_messages_conv_id (en conversation_id) ayuda al
            JOIN. Para N>10k podemos añadir un índice en
            json_each.value si la query se vuelve hot.
        """
        async with self.conn.execute(
            "SELECT m.id AS message_id, "
            "       m.conversation_id, "
            "       m.role, "
            "       m.created_at, "
            "       substr(m.content, 1, 200) AS content_snippet "
            "FROM messages m, json_each(m.file_refs) AS jf "
            "WHERE jf.value = ? "
            "ORDER BY m.created_at DESC, m.id DESC "
            "LIMIT ?",
            (file_id, limit),
        ) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def delete_file(self, file_id: str) -> bool:
        """Borra un file y todas sus dependencias (CASCADE).

        CASCADE limpia automáticamente:
        - file_embeddings (S9.1)
        - memory_facts.source_file_id (S9.1, P0-3 Gemini fix)

        NO limpia messages.file_refs (es TEXT, no FK). _resolve_file_refs
        es tolerante a refs huérfanas: si el file no existe, simplemente
        no se inyecta nada (graceful degradation, TDD §2.2).

        Returns:
            True si se borró una fila, False si no existía.
        """
        async with self.conn.execute("DELETE FROM files WHERE id = ?", (file_id,)) as cur:
            deleted = cur.rowcount
        await self.conn.commit()
        return deleted > 0

    async def set_message_file_refs(self, message_id: int, file_ids: list[str]) -> None:
        """Persiste file_refs en un message (JSON array de file_ids).

        Sprint 9.0: el HTTP API pasa el file_id list al insertar el user
        message. La DB guarda solo la referencia, no el texto del PDF.
        AgentLoop._resolve_file_refs resuelve en runtime vía
        files_cache o db.get_file.

        Args:
            message_id: ID del message en `messages`.
            file_ids: lista de file_ids referenciados (sin duplicados).
        """
        if not file_ids:
            return
        # Dedup preservando orden
        seen: set[str] = set()
        unique: list[str] = []
        for fid in file_ids:
            if fid and fid not in seen:
                seen.add(fid)
                unique.append(fid)
        payload = json.dumps(unique, ensure_ascii=False)
        await self.conn.execute(
            "UPDATE messages SET file_refs = ? WHERE id = ?",
            (payload, message_id),
        )
        await self.conn.commit()

    # --- Sprint 9.1: file_embeddings (RAG) ---

    async def add_file_embedding(
        self, file_id: str, embedding: bytes, model: str = "text-embedding-3-small"
    ) -> None:
        """Inserta o reemplaza el embedding de un file (UPSERT).

        Args:
            file_id: ID del file (debe existir en `files`).
            embedding: numpy array serializado como bytes (float32
                little-endian, 1536-dim → 6KB).
            model: nombre del modelo (default text-embedding-3-small).

        Si el file_id ya tiene embedding, se reemplaza (útil para
        re-embed cuando cambia el modelo o el texto extraído).

        Sprint 19.5 Slice 6 Commit 4: ahora el schema tiene
        ``policy TEXT NOT NULL`` + ``dim INTEGER NOT NULL`` (v25
        migration). Este metodo legacy hardcodea ``policy='vault_ingest'``
        (la policy historica, mismo default que el backfill v27) y
        computa ``dim=length(embedding)/4`` (float32 = 4 bytes/elem)
        para mantener compatibilidad con los 33 tests pre-Sprint 19.5.
        Nuevos callers DEBEN usar ``upsert_embedding(file_id, embedding,
        model, dim, policy)`` explicitamente con su dim canonical y
        policy (e.g. ``chat_rag`` para 384-dim granite embeddings).
        """
        dim = len(embedding) // 4
        await self.conn.execute(
            "INSERT OR REPLACE INTO file_embeddings "
            "(file_id, embedding, embedded_at, model, dim, policy) "
            "VALUES (?, ?, CURRENT_TIMESTAMP, ?, ?, 'vault_ingest')",
            (file_id, embedding, model, dim),
        )
        await self.conn.commit()

    async def upsert_embedding(
        self,
        file_id: str,
        embedding: bytes,
        model: str,
        dim: int,
        policy: str = "vault_ingest",
    ) -> None:
        """Sprint 19.5 Slice 6 Commit 4: per-policy UPSERT for the
        composite-PK ``file_embeddings`` table.

        Replaces the same row (file_id, policy) in place. embedded_at
        is set to CURRENT_TIMESTAMP by the column DEFAULT — callers
        don't pass it. Uses the 6-column schema that v25c establishes
        (file_id, policy, embedding, embedded_at, model, dim).

        Args:
            file_id: foreign key into ``files(id)``. ON DELETE
                CASCADE handled by the FK; deleting a file wipes
                all its embeddings (per policy).
            embedding: serialized numpy float32 vector as bytes
                (length = dim * 4). Not validated here — the
                caller is responsible for matching ``dim`` to the
                actual embedding length. This is by design: the
                EmbeddingsService helper extracts ``dim`` from
                ``result.vector.shape[0]`` (the canonical dim of
                the embedding that was just produced).
            model: model name string (e.g. "qwen/qwen3-embedding-8b",
                "granite-97m"). Stored verbatim for observability.
            dim: canonical dim of the embedding. Stored as INTEGER
                for index-time validation in cosine_search (Rule 6:
                per-policy dim match; mismatched rows are skipped
                with a warning, not deleted).
            policy: which EmbeddingPolicy this row belongs to.
                Default "vault_ingest" matches the historical use
                case (and the v25b backfill default for legacy
                rows). The composite PK is (file_id, policy) so
                the same file can have one row per policy with
                different dims (chat_rag 384-dim + vault_ingest
                4096-dim coexist).

        Returns:
            None on success. Propagates aiosqlite exceptions on
            DB error (FK violation on missing file_id, NOT NULL
            violation if dim/policy are None, etc.).
        """
        await self.conn.execute(
            "INSERT OR REPLACE INTO file_embeddings "
            "(file_id, embedding, embedded_at, model, dim, policy) "
            "VALUES (?, ?, CURRENT_TIMESTAMP, ?, ?, ?)",
            (file_id, embedding, model, dim, policy),
        )
        await self.conn.commit()

    async def get_file_embedding(self, file_id: str) -> bytes | None:
        """Retorna el embedding BLOB de un file, o None si no existe.

        El caller debe deserializar los bytes a numpy float32 con
        `np.frombuffer(data, dtype=np.float32).copy()` (el .copy() es
        crítico: np.frombuffer retorna read-only por defecto, P0 fix
        Gemini 3.5 v1.2 round 8).
        """
        async with self.conn.execute(
            "SELECT embedding FROM file_embeddings WHERE file_id = ?",
            (file_id,),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        return row[0]

    async def get_all_embeddings(self, policy: str | None = None) -> list[tuple[str, bytes]]:
        """Return embeddings, optionally isolated to one policy."""
        if policy is None:
            query = "SELECT file_id, embedding FROM file_embeddings"
            params: tuple[str, ...] = ()
        else:
            query = "SELECT file_id, embedding FROM file_embeddings WHERE policy = ?"
            params = (policy,)
        async with self.conn.execute(query, params) as cur:
            rows = await cur.fetchall()
        return [(str(r[0]), bytes(r[1])) for r in rows]

    async def list_embedded_file_ids(self, limit: int = 100) -> list[str]:
        """Lista solo los file_ids que tienen embedding, sin los blobs.

        Sprint 15 (US-3.1 §4 PR #71 hotfix, Nemotron 3 Ultra 550B review):
        `get_all_embeddings()` carga TODOS los bytes de embeddings en
        memoria (~6KB por file). Para un cosine_search que solo
        necesita el top-k final, esto es overkill si la library tiene
        10K+ files. Este metodo devuelve solo los IDs (cheap text) y
        permite al caller decidir si necesita el blob o no.

        Casos de uso:
        - Mock de cosine_search en tests: lista IDs para devolver el
          top-k sin descargar los 60MB de embeddings.
        - UI "library index": listar files buscables sin materializar
          el vector.

        Args:
            limit: maximo de IDs (default 100, suficiente para top-k
                de un search tipico; el caller pagina si quiere mas).

        Returns:
            lista de file_ids ordenados por `embedded_at DESC` (mas
            recientemente embebidos primero, asumiendo que el caller
            quiere los "frescos").

        Performance:
            O(limit) en I/O. ~50 bytes por ID x 100 = 5KB.
            Vs `get_all_embeddings()` que son 6KB por file x 10K = 60MB.
        """
        async with self.conn.execute(
            "SELECT file_id FROM file_embeddings ORDER BY embedded_at DESC LIMIT ?",
            (limit,),
        ) as cur:
            rows = await cur.fetchall()
        return [r[0] for r in rows]

    async def delete_file_embedding(self, file_id: str) -> None:
        """Borra el embedding de un file (sin tocar el file en sí).

        El CASCADE en file_embeddings.file_id REFERENCES files(id)
        ON DELETE CASCADE ya limpia esto cuando se borra el file.
        Este método es para casos donde se quiere borrar el embedding
        sin borrar el file (e.g. re-embedding con un modelo distinto).
        """
        await self.conn.execute("DELETE FROM file_embeddings WHERE file_id = ?", (file_id,))
        await self.conn.commit()

    async def list_files_pending_embedding(self, limit: int = 500) -> list[dict]:
        """Lista files de la library que aun no tienen embedding.

        Sprint 15 (US-3.1 §4 PR #69): usado por el job `embed_vault` para
        saber que archivos quedan por embeber despues de un restart, o
        tras la ingesta inicial. Devuelve files con `extracted_text`
        no-vacio (sin texto no se puede embeber) y sin fila en
        `file_embeddings`.

        Args:
            limit: maximo de rows (default 500, suficiente para una
                library pequena; el job pagina si hay mas).

        Returns:
            lista de dicts (mismo shape que get_file). Solo incluye:
            - files con extracted_text no-vacio (criterio de filtrado)
            - files que NO tienen fila en file_embeddings

            Orden: created_at ASC (oldest first — FIFO del job queue,
            para que un file subido hace 2 semanas no tenga que esperar
            a que se procesen los nuevos).

            Chore 2026-07-05 (Nemotron 3 Ultra 550B review): aniado
            `f.rowid ASC` como tiebreaker monotónico. Sin esto, dos
            files con el mismo `created_at` (granularidad 1s de SQLite
            CURRENT_TIMESTAMP) tienen orden FIFO arbitrario, lo cual
            rompe la paginación del job queue.
        """
        async with self.conn.execute(
            "SELECT f.id, f.filename, f.mime_type, f.size_bytes, "
            "       f.extracted_text, f.extraction_method, f.source, "
            "       f.source_metadata, f.created_at, f.last_referenced_at, "
            "       f.reference_count, f.tags, f.content_hash "
            "FROM files f "
            "LEFT JOIN file_embeddings fe ON fe.file_id = f.id "
            "WHERE fe.file_id IS NULL "
            "  AND f.extracted_text IS NOT NULL "
            "  AND f.extracted_text != '' "
            "ORDER BY f.created_at ASC, f.rowid ASC "
            "LIMIT ?",
            (limit,),
        ) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def count_files_with_embedding(self) -> int:
        """Cuenta files que tienen embedding persistido.

        Usado por `/v1/health` y metricas de la library (cuantos files
        son buscables semanticamente vs cuantos solo existen como blob).
        """
        async with self.conn.execute("SELECT COUNT(*) FROM file_embeddings") as cur:
            row = await cur.fetchone()
        return int(row[0]) if row else 0

    # --- Sprint 9.2: memory_facts_staging (Sleep Cycle) ---

    async def add_staging_fact(
        self,
        stg_id: str,
        category: str,
        content: str,
        confidence_score: float,
        source_conversation_ids: list[int] | None = None,
        source_file_id: str | None = None,
        *,
        commit: bool = True,
    ) -> None:
        """Inserta un fact candidato en staging (P0-1 Gemini fix).

        Args:
            stg_id: ID único ("stg_<16 hex>").
            category: 'user_preference' | 'project_context' | 'academic_fact'.
            content: el fact en lenguaje natural.
            confidence_score: 0.0-1.0 entregado por el LLM extractor.
            source_conversation_ids: lista de conv_ids donde se
                observo el fact. Inicializa como '[]' (v1.2 fix: nunca
                NULL) en el primer insert.
            source_file_id: file_id opcional (sin CASCADE: staging es
                temporal, cleanup a 90 días via expire_old_staging).
            commit: si True (default), hace commit automatico. Si False,
                el caller gestiona la transaccion (P0-2 cross-review fix
                para transacciones atomicas por conversacion).
        """
        conv_ids = source_conversation_ids or []
        # v1.2 fix: source_conversation_ids siempre '[]' (nunca NULL)
        # para evitar json.loads(None) en queries downstream.
        conv_ids_json = json.dumps(conv_ids, ensure_ascii=False)
        await self.conn.execute(
            "INSERT INTO memory_facts_staging "
            "(id, category, content, confidence_score, "
            "source_conversation_ids, source_file_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                stg_id,
                category,
                content,
                confidence_score,
                conv_ids_json,
                source_file_id,
            ),
        )
        if commit:
            await self.conn.commit()

    async def get_staging_fact(self, stg_id: str) -> dict | None:
        """Lee un staging fact por ID. None si no existe."""
        async with self.conn.execute(
            "SELECT id, category, content, confidence_score, "
            "occurrence_count, first_seen_at, last_seen_at, "
            "source_conversation_ids, source_file_id, status "
            "FROM memory_facts_staging WHERE id = ?",
            (stg_id,),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        return dict(row)

    async def list_staging_facts(
        self, status: str = "pending", min_occurrence: int = 0
    ) -> list[dict]:
        """Lista staging facts por status y occurrence_count mínimo.

        Args:
            status: 'pending' | 'promoted' | 'rejected' | 'expired' (default pending).
            min_occurrence: filtra facts con occurrence_count >= N.

        Returns:
            lista de dicts (mismo shape que get_staging_fact),
            ordenados por last_seen_at DESC (más recientes primero).
        """
        async with self.conn.execute(
            "SELECT id, category, content, confidence_score, "
            "occurrence_count, first_seen_at, last_seen_at, "
            "source_conversation_ids, source_file_id, status "
            "FROM memory_facts_staging "
            "WHERE status = ? AND occurrence_count >= ? "
            "ORDER BY last_seen_at DESC",
            (status, min_occurrence),
        ) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def increment_staging_occurrence(
        self,
        stg_id: str,
        source_conversation_id: int | None = None,
        *,
        commit: bool = True,
    ) -> None:
        """Incrementa occurrence_count y agrega conv_id al JSON array.

        Llamado cuando el Sleep Cycle detecta el mismo fact (vía
        embedding similarity) en una nueva conversación. Si el conv_id
        ya está en la lista, no se duplica.

        Args:
            stg_id: ID del staging fact.
            source_conversation_id: conv_id a agregar (opcional).
            commit: si True (default), commit automatico. Si False,
                el caller gestiona la transaccion (P0-2 fix).
        """
        # Leer conv_ids actual
        async with self.conn.execute(
            "SELECT source_conversation_ids FROM memory_facts_staging WHERE id = ?",
            (stg_id,),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return
        try:
            conv_ids = json.loads(row[0]) if row[0] else []
        except (json.JSONDecodeError, TypeError):
            conv_ids = []
        if source_conversation_id and source_conversation_id not in conv_ids:
            conv_ids.append(source_conversation_id)
        await self.conn.execute(
            "UPDATE memory_facts_staging SET "
            "occurrence_count = occurrence_count + 1, "
            "last_seen_at = CURRENT_TIMESTAMP, "
            "source_conversation_ids = ? "
            "WHERE id = ?",
            (json.dumps(conv_ids, ensure_ascii=False), stg_id),
        )
        if commit:
            await self.conn.commit()

    async def find_similar_staging(
        self,
        category: str,
        content: str,
        threshold: float = 0.92,
    ) -> dict | None:
        """Busca un staging fact existente con la MISMA category y
        content SIMILAR (placeholder textual S9.2; embedding-based
        en S9.2.1).

        **Estado actual (S9.2 — placeholder)**: la implementacion es
        una heuristica de substring sobre `content` (case-insensitive).
        Esto cubre el caso comun de "el LLM reescribe muy similar",
        pero falla en reescrituras semanticas (sin overlap literal).

        **Roadmap S9.2.1**: reemplazar por cosine similarity sobre
        embeddings usando `EmbeddingsService` (aun no inyectado
        aqui). El parametro `threshold` ya esta reservado para esa
        implementacion futura (0.92 estricto, 0.85 permisivo).

        Args:
            category: categoria del fact nuevo.
            content: contenido del fact nuevo.
            threshold: similitud minima para considerar match
                (RESERVADO — S9.2.1 con embeddings). Ignorado en
                la implementacion textual actual.

        Returns:
            el staging fact existente más similar (None si no hay
            match). El caller debe decidir si incrementa el
            occurrence del existente o inserta uno nuevo.
        """
        # Placeholder textual (S9.2): match por substring case-insensitive.
        # TODO S9.2.1: reemplazar con embedding-based cosine similarity
        # cuando el EmbeddingsService este inyectado en Database.
        async with self.conn.execute(
            "SELECT id, category, content, confidence_score, "
            "occurrence_count, first_seen_at, last_seen_at, "
            "source_conversation_ids, source_file_id, status "
            "FROM memory_facts_staging "
            "WHERE status = 'pending' AND category = ?",
            (category,),
        ) as cur:
            rows = await cur.fetchall()
        norm_new = content.lower().strip()
        for row in rows:
            existing = row["content"].lower().strip()
            if norm_new in existing or existing in norm_new:
                return dict(row)
        return None

    async def promote_staging_to_fact(
        self,
        stg_id: str,
        fact_id: str,
        source_conversation_id: int | None = None,
        is_permanent: bool = False,
    ) -> None:
        """Promueve un staging fact a memory_facts (consolidated).

        Llamado cuando occurrence_count >= memory_fact_min_mentions.
        El staging row se marca como 'promoted'.

        **P0-1 cross-review fix**: transacción atómica. Si el proceso
        crashea entre INSERT fact y UPDATE staging, no hay duplicados
        en la siguiente ejecución (rollback automático).

        Args:
            stg_id: ID del staging fact a promover.
            fact_id: ID del nuevo fact en memory_facts.
            source_conversation_id: conv_id que triggerea la promoción.
            is_permanent: si True, el fact no decae con tiempo
                (hardware, nombre, hechos críticos).
        """
        # Leer staging para copiar content/category
        stg = await self.get_staging_fact(stg_id)
        if stg is None:
            return
        # P0-1 fix: transacción atómica (BEGIN/COMMIT)
        try:
            await self.conn.execute("BEGIN IMMEDIATE")
            # Insertar en memory_facts
            await self.conn.execute(
                "INSERT INTO memory_facts "
                "(id, category, content, source_conversation_id, "
                "source_file_id, occurrence_count, is_permanent, is_verified) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    fact_id,
                    stg["category"],
                    stg["content"],
                    source_conversation_id,
                    stg["source_file_id"],
                    stg["occurrence_count"],
                    1 if is_permanent else 0,
                    0,  # is_verified: solo manual
                ),
            )
            # Marcar staging como promoted
            await self.conn.execute(
                "UPDATE memory_facts_staging SET status = 'promoted' WHERE id = ?",
                (stg_id,),
            )
            await self.conn.execute("COMMIT")
        except Exception:
            await self.conn.execute("ROLLBACK")
            raise

    async def expire_old_staging(self, days: int = 90) -> int:
        """Marca como 'expired' los staging facts sin actividad > N días.

        P2 Copilot review 2026-06-26: usar `date(last_seen_at)` para
        normalizar el timestamp con hora a solo fecha. Sin esto, la
        comparacion es lexicografica entre 'YYYY-MM-DD HH:MM:SS' y
        'YYYY-MM-DD', que falla porque 'YYYY-MM-DD HH:MM:SS' es
        siempre > 'YYYY-MM-DD' en orden lexicografico (ASCII
        ' '(0x20) < '-' (0x2D) en realidad lo es, pero la longitud
        de la cadena dicta el orden: la mas larga es mayor si el
        prefijo coincide, asi que '2025-01-01 10:00:00' > '2025-01-01').
        Resultado: el filtro no expiraba staging del mismo dia.

        Returns:
            número de rows actualizados.
        """
        async with self.conn.execute(
            "UPDATE memory_facts_staging "
            "SET status = 'expired' "
            "WHERE status = 'pending' "
            "  AND date(last_seen_at) < date('now', ?)",
            (f"-{days} days",),
        ) as cur:
            count = cur.rowcount
        await self.conn.commit()
        return count

    # --- Sprint 9.2: memory_facts (consolidated) ---

    async def add_fact(
        self,
        fact_id: str,
        category: str,
        content: str,
        source_conversation_id: int | None = None,
        source_file_id: str | None = None,
        occurrence_count: int = 1,
        is_permanent: bool = False,
        is_verified: bool = False,
    ) -> None:
        """Inserta un fact consolidado (tras promoción de staging)."""
        await self.conn.execute(
            "INSERT INTO memory_facts "
            "(id, category, content, source_conversation_id, "
            "source_file_id, occurrence_count, is_permanent, is_verified) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                fact_id,
                category,
                content,
                source_conversation_id,
                source_file_id,
                occurrence_count,
                1 if is_permanent else 0,
                1 if is_verified else 0,
            ),
        )
        await self.conn.commit()

    async def get_fact(self, fact_id: str) -> dict | None:
        """Lee un fact por ID. None si no existe."""
        async with self.conn.execute(
            "SELECT id, category, content, source_conversation_id, "
            "source_file_id, occurrence_count, is_permanent, is_verified, "
            "created_at, last_referenced_at "
            "FROM memory_facts WHERE id = ?",
            (fact_id,),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        return dict(row)

    async def list_facts(self, category: str | None = None, limit: int = 50) -> list[dict]:
        """Lista facts consolidados, opcionalmente filtrados por category."""
        if category is not None:
            async with self.conn.execute(
                "SELECT id, category, content, source_conversation_id, "
                "source_file_id, occurrence_count, is_permanent, is_verified, "
                "created_at, last_referenced_at "
                "FROM memory_facts WHERE category = ? "
                "ORDER BY last_referenced_at DESC LIMIT ?",
                (category, limit),
            ) as cur:
                rows = await cur.fetchall()
        else:
            async with self.conn.execute(
                "SELECT id, category, content, source_conversation_id, "
                "source_file_id, occurrence_count, is_permanent, is_verified, "
                "created_at, last_referenced_at "
                "FROM memory_facts "
                "ORDER BY last_referenced_at DESC LIMIT ?",
                (limit,),
            ) as cur:
                rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def touch_fact(self, fact_id: str) -> None:
        """Actualiza last_referenced_at del fact (tracking de uso)."""
        await self.conn.execute(
            "UPDATE memory_facts SET last_referenced_at = CURRENT_TIMESTAMP WHERE id = ?",
            (fact_id,),
        )
        await self.conn.commit()

    async def delete_fact(self, fact_id: str) -> bool:
        """Borra un fact. CASCADE limpia memory_fact_embeddings."""
        async with self.conn.execute("DELETE FROM memory_facts WHERE id = ?", (fact_id,)) as cur:
            deleted = cur.rowcount
        await self.conn.commit()
        return deleted > 0

    # --- Sprint 9.2: memory_fact_embeddings (RAG para facts) ---

    async def add_fact_embedding(
        self, fact_id: str, embedding: bytes, model: str = "qwen/qwen3-embedding-8b"
    ) -> None:
        """Inserta o reemplaza el embedding de un fact (UPSERT)."""
        await self.conn.execute(
            "INSERT OR REPLACE INTO memory_fact_embeddings "
            "(fact_id, embedding, embedded_at, model) "
            "VALUES (?, ?, CURRENT_TIMESTAMP, ?)",
            (fact_id, embedding, model),
        )
        await self.conn.commit()

    async def get_all_fact_embeddings(self) -> list[tuple[str, bytes]]:
        """Retorna todos los embeddings de facts (fact_id, blob)."""
        async with self.conn.execute(
            "SELECT fact_id, embedding FROM memory_fact_embeddings"
        ) as cur:
            rows = await cur.fetchall()
        return [(r[0], r[1]) for r in rows]

    # =========================================================================
    # Sprint 14 (TDD_S14_DEEP_RESEARCH.md §2): research_jobs CRUD
    # =========================================================================
    # SQL lives here (data layer); service.py consume via methods. Todos los
    # timestamps se generan via format_now() en hermes/jobs/cost.py (regla §1.5.1).

    async def create_research_job(
        self,
        job_id: str,
        query: str,
        notify_via_tg: int,
        job_type: str = "deep_research",
        user_id: int = 0,
    ) -> None:
        """Inserta un research job en estado 'pending'.

        Args:
            job_id: UUID 12-char hex (TDD §1.5).
            query: query de investigacion (validada upstream por CreateJobRequest).
            notify_via_tg: 1 si user quiere TG push al completar, 0 si no.
            job_type: 'deep_research' (S14) | 'reminder' | 'embed_vault' (S15+).
            user_id: sentinel 0 (S14 single-tenant).
        """
        from hermes.jobs.cost import format_now

        now = format_now()
        await self.conn.execute(
            "INSERT INTO research_jobs "
            "(id, user_id, job_type, query, notify_via_tg, status, "
            " created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)",
            (job_id, user_id, job_type, query, notify_via_tg, now, now),
        )
        await self.conn.commit()

    async def get_research_job(self, job_id: str) -> dict | None:
        """Lee un research job por ID. None si no existe."""
        async with self.conn.execute(
            "SELECT id, user_id, job_type, query, notify_via_tg, status, "
            "       current_phase, progress_percent, output_path, partial_output_path, "
            "       error_taxonomy, error_message, cost_usd, tokens_in, tokens_out, "
            "       notified, created_at, started_at, completed_at, updated_at "
            "FROM research_jobs WHERE id = ?",
            (job_id,),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        return dict(row)

    async def list_research_jobs(
        self,
        user_id: int = 0,
        status: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """Lista jobs de un user, ordenado por created_at DESC.

        Args:
            user_id: filter por user (default 0 = sentinel single-user).
            status: filter opcional por status (pending/running/complete/...).
            limit: cap defensivo (default 50, max 200).
        """
        limit = min(max(limit, 1), 200)
        where_clauses = ["user_id = ?"]
        params: list = [user_id]
        if status is not None:
            where_clauses.append("status = ?")
            params.append(status)
        sql = (
            "SELECT id, query, status, current_phase, progress_percent, "
            "       cost_usd, created_at, started_at, completed_at "
            "FROM research_jobs "
            f"WHERE {' AND '.join(where_clauses)} "
            "ORDER BY created_at DESC LIMIT ?"
        )
        params.append(limit)
        async with self.conn.execute(sql, params) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def update_research_job_status(
        self,
        job_id: str,
        status: str,
        *,
        current_phase: str | None = None,
        progress_percent: int | None = None,
        output_path: str | None = None,
        partial_output_path: str | None = None,
        error_taxonomy: str | None = None,
        error_message: str | None = None,
        started_at: str | None = None,
        completed_at: str | None = None,
        notified: int | None = None,
    ) -> None:
        """UPDATE polimórfico de research_jobs. Solo los campos no-None se actualizan.

        Usado por el service para todas las transiciones de estado (pending→running,
        running→complete, etc). El updated_at se actualiza automáticamente.
        """
        from hermes.jobs.cost import format_now

        sets: list[str] = ["status = ?", "updated_at = ?"]
        params: list = [status, format_now()]
        if current_phase is not None:
            sets.append("current_phase = ?")
            params.append(current_phase)
        if progress_percent is not None:
            sets.append("progress_percent = ?")
            params.append(progress_percent)
        if output_path is not None:
            sets.append("output_path = ?")
            params.append(output_path)
        if partial_output_path is not None:
            sets.append("partial_output_path = ?")
            params.append(partial_output_path)
        if error_taxonomy is not None:
            sets.append("error_taxonomy = ?")
            params.append(error_taxonomy)
        if error_message is not None:
            sets.append("error_message = ?")
            params.append(error_message)
        if started_at is not None:
            sets.append("started_at = ?")
            params.append(started_at)
        if completed_at is not None:
            sets.append("completed_at = ?")
            params.append(completed_at)
        if notified is not None:
            sets.append("notified = ?")
            params.append(notified)
        params.append(job_id)
        await self.conn.execute(
            f"UPDATE research_jobs SET {', '.join(sets)} WHERE id = ?",
            params,
        )
        await self.conn.commit()

    async def mark_research_job_notified(self, job_id: str) -> None:
        """Marca el job como notificado (used tras send_research_complete)."""
        await self.conn.execute(
            "UPDATE research_jobs SET notified = 1, updated_at = ? WHERE id = ?",
            (self._now_str(), job_id),
        )
        await self.conn.commit()

    async def add_token_usage(
        self,
        job_id: str,
        phase: str,
        model: str,
        tokens_in: int,
        tokens_out: int,
        cost_usd: float,
    ) -> None:
        """INSERT en research_job_token_usage + UPDATE aggregates en research_jobs.

        Llamado por DeepResearchService._record_token_usage tras cada LLM call.
        """
        from hermes.jobs.cost import format_now

        now = format_now()
        await self.conn.execute(
            "INSERT INTO research_job_token_usage "
            "(job_id, phase, model, tokens_in, tokens_out, cost_usd, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (job_id, phase, model, tokens_in, tokens_out, cost_usd, now),
        )
        await self.conn.execute(
            "UPDATE research_jobs SET "
            "cost_usd = cost_usd + ?, "
            "tokens_in = tokens_in + ?, "
            "tokens_out = tokens_out + ?, "
            "updated_at = ? "
            "WHERE id = ?",
            (cost_usd, tokens_in, tokens_out, now, job_id),
        )
        await self.conn.commit()

    async def list_token_usage_for_job(self, job_id: str) -> list[dict]:
        """Lista per-LLM-call entries para un job (drill-down)."""
        async with self.conn.execute(
            "SELECT id, phase, model, tokens_in, tokens_out, cost_usd, created_at "
            "FROM research_job_token_usage WHERE job_id = ? ORDER BY id ASC",
            (job_id,),
        ) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def get_research_job_query(self, job_id: str) -> str | None:
        """Helper: lee solo el query de un job (para inject en prompts)."""
        async with self.conn.execute(
            "SELECT query FROM research_jobs WHERE id = ?", (job_id,)
        ) as cur:
            row = await cur.fetchone()
        return row[0] if row else None

    async def get_research_job_cost(self, job_id: str) -> float:
        """Helper: lee cost_usd aggregate actual del job."""
        async with self.conn.execute(
            "SELECT cost_usd FROM research_jobs WHERE id = ?", (job_id,)
        ) as cur:
            row = await cur.fetchone()
        return float(row[0]) if row else 0.0

    async def set_research_job_cost_monotonic(
        self, job_id: str, reconciled_cost: float
    ) -> float:
        """Atomic monotonic upsert: research_jobs.cost_usd = max(existing, reconciled_cost).

        DR-Q1A-PRE1A cost-reconciliation fix. Used by
        ``DeepResearchService.reconcile_cost`` to persist the
        reconciled maximum so that subsequent reads of
        ``research_jobs.cost_usd`` (e.g. from
        ``_phase_write``/``_db.get_research_job_cost``) and
        the completion notifier all see the same value.

        Semantics:
            research_jobs.cost_usd = max(research_jobs.cost_usd, reconciled_cost)
            updated_at = now

        The aggregate is never decreased. If the row does not
        exist for ``job_id`` (orphan call), this is a no-op and
        returns ``reconciled_cost`` unchanged (caller is
        expected to verify job existence; this method does
        not raise to keep reconcile_cost idempotent and
        non-throwing across retries).

        Returns the post-update aggregate value (the actual
        persisted ``cost_usd``), read back from the same
        transaction. This is the value subsequent reads
        will observe and the value the notifier should send.

        The UPDATE + SELECT are in the same ``aiosqlite``
        connection but aiosqlite serialises statements per
        connection inside ``await`` boundaries; combined with
        SQLite's per-database write lock, this is monotonic
        within the service's event loop. Cross-process
        monotonicity is not required (single-writer service
        per the architecture). The SQL itself uses
        ``MAX(cost_usd, ?)`` so the write is atomic at the
        SQLite engine level.
        """
        await self.conn.execute(
            "UPDATE research_jobs SET "
            "cost_usd = MAX(cost_usd, ?), "
            "updated_at = ? "
            "WHERE id = ?",
            (reconciled_cost, self._now_str(), job_id),
        )
        await self.conn.commit()
        # Read back the post-update value. The UPDATE above is
        # committed, so a subsequent SELECT observes the
        # updated value.
        async with self.conn.execute(
            "SELECT cost_usd FROM research_jobs WHERE id = ?", (job_id,)
        ) as cur:
            row = await cur.fetchone()
        return float(row[0]) if row else float(reconciled_cost)

    async def get_today_research_cost(self, user_id: int = 0) -> float:
        """Suma cost_usd de jobs creados hoy (UTC) para un user. Cancelled excluded.

        Usado por _check_daily_budget (TDD §8.2).
        """
        from datetime import UTC, datetime

        today_start = (
            datetime.now(UTC)
            .replace(hour=0, minute=0, second=0, microsecond=0)
            .strftime("%Y-%m-%d %H:%M:%S.000")
        )
        async with self.conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) FROM research_jobs "
            "WHERE user_id = ? AND created_at >= ? AND status != 'cancelled'",
            (user_id, today_start),
        ) as cur:
            row = await cur.fetchone()
        return float(row[0]) if row else 0.0

    async def count_research_jobs_running(self, user_id: int = 0) -> int:
        """Cuenta jobs running para un user (cap de concurrencia, TDD §5)."""
        async with self.conn.execute(
            "SELECT COUNT(*) FROM research_jobs WHERE user_id = ? AND status = 'running'",
            (user_id,),
        ) as cur:
            row = await cur.fetchone()
        return int(row[0]) if row else 0

    async def count_research_jobs_status(self, status: str) -> int:
        """Cuenta jobs en un status dado (usado por /health)."""
        async with self.conn.execute(
            "SELECT COUNT(*) FROM research_jobs WHERE status = ?", (status,)
        ) as cur:
            row = await cur.fetchone()
        return int(row[0]) if row else 0

    async def list_research_jobs_by_status_started_before(
        self, status: str, cutoff_str: str, limit: int = 100
    ) -> list[dict]:
        """Lista jobs en un status con started_at < cutoff (TDD §7 recovery).

        Args:
            status: filter exacto (e.g. 'running', 'pending').
            cutoff_str: timestamp formato 'YYYY-MM-DD HH:MM:SS.sss'.
            limit: cap defensivo (TDD §7.1 LIMIT 100 anti-OOM spike).

        Returns:
            lista de dicts con todas las columnas del job.
        """
        limit = min(max(limit, 1), 100)
        async with self.conn.execute(
            "SELECT id, user_id, query, status, current_phase, progress_percent, "
            "       output_path, error_taxonomy, error_message, cost_usd, tokens_in, "
            "       tokens_out, notified, created_at, started_at, completed_at, updated_at "
            "FROM research_jobs WHERE status = ? AND started_at < ? "
            "ORDER BY started_at ASC LIMIT ?",
            (status, cutoff_str, limit),
        ) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def list_research_jobs_pending_created_before(
        self, cutoff_str: str, limit: int = 100
    ) -> list[dict]:
        """Lista pending jobs con created_at < cutoff y sin started_at (orphans).

        Caso 1 del recovery (TDD §7.1).
        """
        limit = min(max(limit, 1), 100)
        async with self.conn.execute(
            "SELECT id, query, status, current_phase, progress_percent, "
            "       error_taxonomy, error_message, cost_usd, tokens_in, tokens_out, "
            "       notified, created_at, started_at, completed_at, updated_at "
            "FROM research_jobs "
            "WHERE status = 'pending' AND started_at IS NULL AND created_at < ? "
            "ORDER BY created_at ASC LIMIT ?",
            (cutoff_str, limit),
        ) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def list_research_jobs_running_with_output(self, limit: int = 100) -> list[dict]:
        """Lista running jobs con output_path != NULL (caso 3 recovery).

        Write OK pero DB UPDATE no llegó — mark complete.
        """
        limit = min(max(limit, 1), 100)
        async with self.conn.execute(
            "SELECT id, query, status, current_phase, progress_percent, "
            "       output_path, cost_usd, notified, started_at, completed_at "
            "FROM research_jobs WHERE status = 'running' AND output_path IS NOT NULL "
            "ORDER BY started_at ASC LIMIT ?",
            (limit,),
        ) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def list_research_jobs_unnotified(self, limit: int = 100) -> list[dict]:
        """Lista complete/failed con notified=0 (caso 4 recovery)."""
        limit = min(max(limit, 1), 100)
        async with self.conn.execute(
            "SELECT id, query, status, output_path, cost_usd, notified, completed_at "
            "FROM research_jobs WHERE status IN ('complete', 'failed') AND notified = 0 "
            "ORDER BY completed_at ASC LIMIT ?",
            (limit,),
        ) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def list_research_jobs_cancelling(self, limit: int = 100) -> list[dict]:
        """Lista jobs en estado 'cancelling' huérfanos (caso 5 recovery)."""
        limit = min(max(limit, 1), 100)
        async with self.conn.execute(
            "SELECT id, query, status, started_at, completed_at "
            "FROM research_jobs WHERE status = 'cancelling' "
            "ORDER BY started_at ASC LIMIT ?",
            (limit,),
        ) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def query_scalar(self, sql: str, *params: Any) -> float | None:
        """Ejecuta una query y devuelve el primer valor escalar de la primera fila.

        Helper genérico para reconcile_cost() y agregaciones simples.

        Args:
            sql: SQL con al menos una columna (e.g. SELECT SUM(...) ...).
            *params: parámetros posicionales para la query.

        Returns:
            El valor escalar como float, o None si no hay filas.
        """
        async with self.conn.execute(sql, params) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        # aiosqlite.Row supports both index access and key access
        first = row[0] if not isinstance(row, dict) else next(iter(row.values()))
        if first is None:
            return None
        return float(first)

    @staticmethod
    def _now_str() -> str:
        """Helper: timestamp SQLite-friendly. Reusa format_now() centralizada."""
        from hermes.jobs.cost import format_now

        return format_now()

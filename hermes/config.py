"""Configuración cargada desde variables de entorno."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlparse

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

logger = logging.getLogger(__name__)


# Sprint 19.6+ Phase 3 (TDD v0.7 §10.4.5): local_vision_base_url MUST
# point to a local-only Ollama. The host allowlist is the defense
# against data exfiltration if .env is tampered (Sprint 19 north star
# forbids data egress).
_LOCAL_VISION_ALLOWED_HOSTS: frozenset[str] = frozenset({"localhost", "127.0.0.1", "::1"})

# Allowlist estricto de modelos MiniMax API (Sprint 12+).
# Fuente: https://platform.minimax.io/docs/llms.txt
# Política de privacidad: zero data retention, no se usan para entrenar.
#
# El nombre del frozenset (_ALLOWED_OPENCODE_GO_MODELS) y el alias de env
# (OPENCODE_GO_BASE_URL) son LEGACY de cuando el routing iba via el
# provider opencode-go. Se mantienen por compat con deploy scripts y
# Credential Manager entries. La allowlist YA apunta a modelos MiniMax
# direct API.
#
# Sprint 19.6+ Phase 5 (OpenAI Build Week): se anaden los modelos
# Ollama locales a la allowlist para que el default chain (qwen2.5:7b)
# pase el validator. La allowlist es ahora multi-provider (MiniMax +
# Ollama local); el nombre legacy se mantiene por compat.
#
# Rollback al provider opencode-go: si en el futuro queremos volver al
# routing legacy (poco probable — la quota se agoto y MiniMax API direct
# es estrictamente mejor en coste/latencia/thinking), basta con setear
# OPENCODE_GO_BASE_URL=https://opencode.ai/zen/go/v1 y cambiar
# llm_text_primary/fallback a los modelos legacy equivalentes.
#
# Reglas de inclusion:
# - MiniMax: solo modelos cuya retention/privacy este documentada.
#   Excluir preview/experimental hasta validar.
# - Ollama local: nombres exactos que Ollama sirve (sin version pin).
#   Anadir modelos nuevos aca si se cambia `llm_text_primary` por
#   default a un modelo Ollama distinto.
_ALLOWED_OPENCODE_GO_MODELS: frozenset[str] = frozenset(
    {
        "MiniMax-M3",  # 1M context, multimodal, con thinking
        "MiniMax-M2.7",  # 204k context, texto + tool use
        "MiniMax-M2.7-highspeed",  # mismo rendimiento, ~100 tps
        "MiniMax-M2.5",  # legacy compatible
        "MiniMax-M2.5-highspeed",
        "MiniMax-M2.1",  # 230B params, 10B activos
        "MiniMax-M2.1-highspeed",
        "MiniMax-M2",  # 200k context, agentic, streaming
        # Sprint 19.6+ Phase 5: Ollama local models (OpenAI Build Week
        # "just works out of the box" pitch). qwen2.5:7b is the
        # default chain primary; llama3.1:8b is a documented
        # alternative; mistral:7b is a lighter fallback.
        "qwen2.5:7b",
        "llama3.1:8b",
        "mistral:7b",
    }
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    # Sprint 11 (ADR-004): telegram_bot_token es ahora opcional.
    # Antes era required (min_length=10) — bloqueaba cualquier
    # deployment que no usara Telegram. Ahora es `str | None` con
    # default None. Es requerido SOLO si enable_telegram=True.
    # Si no, el bot no arranca (warning, no crash).
    telegram_bot_token: str | None = Field(default=None, validation_alias="TELEGRAM_BOT_TOKEN")
    # CSV de IDs permitidos (Telegram). Se parsea en property.
    # Acepta ALLOWED_USER_IDS como alias para retrocompatibilidad.
    allowed_user_ids: str = Field(default="", validation_alias="ALLOWED_USER_IDS")

    # Sprint 12+ migración de opencode-go a MiniMax API direct: la quota de
    # OpenCode Go se agota rápido, el default ahora apunta directo a
    # MiniMax API, que es OpenAI-compatible y más barato. Setear
    # OPENCODE_GO_BASE_URL en el .env para apuntar a otro proveedor.
    # El código sigue siendo compatible con OpenCode Go (mismo formato
    # OpenAI-compatible): solo cambia el base_url y el nombre del modelo.
    # El nombre de la env var se mantiene legacy por compat.
    #
    # Modelos MiniMax (https://platform.minimax.io/docs/llms.txt):
    # - MiniMax-M3: 1M context, multimodal, con thinking (default on)
    # - MiniMax-M2.7-highspeed: ~100 tps, sin vision, más barato
    # - MiniMax-M2.5-highspeed: legacy, similar a M2.7
    opencode_go_base_url: str = Field(
        default="https://api.minimax.io/v1",
        validation_alias="OPENCODE_GO_BASE_URL",
    )
    # Sprint 19.6+ Phase 5 (OpenAI Build Week): opencode_go_api_key
    # es ahora opcional. Antes era `Field(..., min_length=10)` (required)
    # — bloqueaba el flow "judge clones repo, docker compose up, chat
    # works" porque hermes fallaba al cargar Settings sin .env o con
    # .env.example (que tiene `OPENCODE_GO_API_KEY=` vacio). El
    # runtime gate en `text_chain` (L1612) sigue comprobando
    # `if self.opencode_go_api_key and .strip() and self.llm_text_fallback`
    # para que el fallback cloud SOLO se aniada si hay key. Sin key
    # → chain = [primary] (Ollama local, sin cloud calls). Mirror
    # del patron Sprint 11 ADR-004 aplicado a `telegram_bot_token`.
    opencode_go_api_key: str | None = Field(
        default=None,
        validation_alias="OPENCODE_GO_API_KEY",
    )
    # Smart routing: MiniMax-M3 como primary (capaz, multimodal),
    # M2.7-highspeed como fallback (rápido, sin vision).
    # Equivalencia legacy (opencode-go provider): el routing primario→
    # fallback equivalente se hacia con minimax-m3 → deepseek-v4-flash
    # a traves de la capa opencode-go. Ahora (Sprint 12+) esa capa
    # intermedia fue removida.
    #
    # Sprint 19.6+ Phase 5 (OpenAI Build Week): el default chain es
    # ahora LOCAL-FIRST. `llm_text_primary` por defecto es
    # `qwen2.5:7b` (modelo Ollama local), servido por la `ollama`
    # service en `docker-compose.yml` (mismo host, sin API key).
    # `llm_text_primary_provider="ollama"` activa el dispatch al
    # OllamaClient en `hermes/llm/router.py`.
    #
    # Para volver al comportamiento cloud-only de Sprint 19.6+ Phase 4
    # (MiniMax como primary), el operador setea:
    #   LLM_TEXT_PRIMARY=MiniMax-M3
    #   LLM_TEXT_PRIMARY_PROVIDER=minimax
    #   OPENCODE_GO_API_KEY=sk-...
    # El fallback (M2.7-highspeed) se mantiene MiniMax — solo cambia
    # el primary. Si el operador quiere fallback Ollama tmbien
    # (chain totalmente local), setea LLM_TEXT_FALLBACK_PROVIDER=ollama
    # y un modelo Ollama como fallback.
    llm_text_primary: str = "qwen2.5:7b"
    llm_text_fallback: str = "MiniMax-M2.7-highspeed"
    llm_voice_primary: str = "MiniMax-M3"
    # Sprint 19.6+ Phase 5: provider hint per model slot. Used by
    # LLMRouter to dispatch to the right client (OllamaClient vs the
    # main OpenAI/Anthropic httpx client). Default for the primary
    # is "ollama" (local-first); the fallback default is "minimax"
    # (cloud — only used if the operator has set OPENCODE_GO_API_KEY
    # AND the primary Ollama model failed; the smart routing skips
    # the fallback entirely when no API key is set, see
    # `text_chain` property).
    llm_text_primary_provider: str = Field(
        default="ollama",
        validation_alias="LLM_TEXT_PRIMARY_PROVIDER",
        description="Provider hint for llm_text_primary. 'ollama' (default, "
        "local-first via the ollama service in docker-compose.yml) or "
        "'minimax' (cloud via the OPENCODE_GO_* settings). Used by "
        "LLMRouter to dispatch to the right client.",
    )
    llm_text_fallback_provider: str = Field(
        default="minimax",
        validation_alias="LLM_TEXT_FALLBACK_PROVIDER",
        description="Provider hint for llm_text_fallback. Default 'minimax' "
        "(cloud fallback). Set to 'ollama' for a fully-local chain "
        "(e.g., qwen2.5:7b primary + llama3.1:8b fallback, both via "
        "the same ollama service).",
    )
    # Sprint 19.6+ Phase 5: Ollama endpoint config. The default base
    # URL `http://localhost:11434/v1` works for non-docker usage
    # (Ollama running on the host machine). In the docker compose
    # stack the backend container sets
    # LLM_TEXT_PRIMARY__BASE_URL=http://ollama:11434/v1 via the
    # `backend` service environment.
    #
    # `LLM_TEXT_PRIMARY__API_KEY` is "ollama" by default (Ollama
    # ignores the Authorization header value, but the `openai` SDK
    # requires a non-empty string when constructing the client).
    llm_text_primary_base_url: str = Field(
        default="http://localhost:11434/v1",
        validation_alias="LLM_TEXT_PRIMARY__BASE_URL",
        description="Base URL for the Ollama OpenAI-compat API. Default "
        "http://localhost:11434/v1 (Ollama local default). In docker "
        "compose the backend uses http://ollama:11434/v1 (internal "
        "docker network hostname).",
    )
    llm_text_primary_api_key: str = Field(
        default="ollama",
        validation_alias="LLM_TEXT_PRIMARY__API_KEY",
        description="API key for the Ollama endpoint. Ignored by Ollama "
        "(no real auth), but the openai SDK requires a non-empty "
        "value. The literal 'ollama' is the canonical placeholder "
        "used by Ollama's own docs.",
    )
    # Allowlist override (Sprint 12+). Lista de modelos aceptados por
    # el validator. Default = la lista production-safe hardcoded abajo.
    # Override via env (LLM_ALLOWED_MODELS='["model-a","model-b"]') para
    # tests o para anadir un modelo custom sin tocar codigo.
    # Por que model_validator en vez de field_validator: Pydantic no da
    # acceso a otros campos desde field_validator; necesitamos leer
    # self.llm_allowed_models tras la construccion del modelo completo.
    llm_allowed_models: list[str] = Field(
        default_factory=lambda: sorted(_ALLOWED_OPENCODE_GO_MODELS),
        validation_alias="LLM_ALLOWED_MODELS",
    )
    # Temperatura: MiniMax recomienda 1.0 (default). El 0.3 que
    # teníamos antes estaba calibrado para los modelos legacy
    # de OpenCode Go (más "creativos"); MiniMax con 1.0 produce
    # respuestas más útiles para asistencia factual.
    # Sprint 12 (ADR-007): model aliases que mapean a chains dedicadas.
    # Cuando un cliente pide `model: "oroimen-agent-fast"`, el smart
    # router usa la chain del alias en lugar de text_chain. Configurado
    # via env (HERMES_LLM_MODEL_OVERRIDES='{"alias":["modelo"]}').
    # NoDecode: indica a pydantic-settings que NO intente parsear la env
    # var como JSON antes de pasarsela a nuestro @field_validator(mode=before),
    # que es quien decide si el JSON es valido o cae a dict vacio gracefully.
    # Sin NoDecode, pydantic-settings 2.x lanza SettingsError al ver JSON
    # malformado ANTES de invocar nuestro validator (rompe startup).
    llm_model_overrides: Annotated[dict[str, list[str]], NoDecode] = Field(
        default_factory=dict,
        validation_alias="HERMES_LLM_MODEL_OVERRIDES",
    )
    # Temperatura: MiniMax default 1.0 (recomendado). Subir a 1.2+ solo
    # si quieres respuestas mas creativas. Bajar a 0.5-0.7 para
    # respuestas mas deterministas.
    llm_temperature: float = 1.0
    llm_timeout_seconds: int = 30
    llm_max_retries: int = 2
    # SPRINT 18 HOTFIX (2026-07-08): nucleus sampling + repetition
    # penalty. Bug production observado: respuestas largas del modelo
    # MiMo v2.5 (OpenRouter free tier fallback) terminaban en
    # repetition loops con caracteres chinos random ("适合适配适配...").
    # Root cause: temp=1.0 + top_p=1.0 (default desactivado) + sin
    # repetition_penalty = el sampleador tomaba tokens exoticos del
    # pre-training multilingue y se quedaba atascado en loops porque
    # no tenia presion anti-repeticion.
    #
    # top_p=0.9 (standard OpenAI): descarta el bottom 10% de tokens
    # candidatos por probabilidad. Previene drift de idioma (los
    # caracteres chinos "适合" caen fuera del 90% de masa probable).
    #
    # repetition_penalty=1.04 (gentle): cada vez que un token ya
    # aparecio en la respuesta, su probabilidad se multiplica por 1/1.04
    # (~3.85% penalty por repeticion). Suficiente para empujar al
    # modelo hacia <eos> sin romper repeticiones legitimas (listas,
    # terminos tecnicos repetidos como "Fidelity" 5 veces en una
    # respuesta sobre Fidelity). Valor par por preferencia del owner
    # (project owner). Si en el futuro vemos loops sutiles, subir a 1.06/1.08.
    llm_top_p: float = Field(
        default=0.9,
        validation_alias="HERMES_LLM_TOP_P",
        gt=0.0,
        le=1.0,
    )
    llm_repetition_penalty: float = Field(
        default=1.04,
        validation_alias="HERMES_LLM_REPETITION_PENALTY",
        ge=1.0,
        le=2.0,
    )
    # Sprint 16+ context: memory facts injection + RAG + tool use naturally
    # produce responses > 1024 tokens. 1024 was the pre-Sprint 16 default
    # (chat corto). 8192 matches Anthropic / Gemini defaults. Validation
    # (ge=100, le=32768) catches malformed env values silently and falls
    # back to default. Override via HERMES_LLM_MAX_TOKENS env var.
    llm_max_tokens: int = Field(
        default=8192,
        validation_alias="HERMES_LLM_MAX_TOKENS",
        ge=100,
        le=32768,
    )
    # Base del backoff exponencial entre reintentos (segundos).
    # Default 0.5s produce: attempt1→0.5s, attempt2→1.0s, attempt3→2.0s
    # (cap a 4.0s en el router). En tests se baja a ~0.001s para que la
    # suite no gaste 3s por test en retries de error.
    llm_retry_backoff_base: float = 0.5

    # Sprint 9.3.3: max iteraciones del AgentLoop. Default 10 permite
    # deep research (multiples lotes de busquedas + sintesis final).
    # El antiguo default 3 cortaba deep research a media (12 busquedas
    # en 3 lotes, sin tiempo de sintetizar). Para Sprint 10 (autonomo),
    # subir a 20-30 o hacerlo per-task configurable.
    agent_max_iterations: int = 25

    circuit_breaker_fail_max: int = 5
    circuit_breaker_reset_timeout: int = 60

    # Tool output budget (Sprint 5 T49).
    # Por qué: el techo hardcoded de 2500 chars truncaba el transcript
    # de YouTube (~52KB para videos de 10min) a solo ~1-2 min de
    # contenido, produciendo resúmenes vagos. Parametrizamos el límite
    # por categoría de tool (declarada en ToolSpec.tool_category).
    #   - system tools (get_weather, get_system_status, etc): 2500 chars
    #     (info corta, no necesitan más)
    #   - read tools (agent_reach_*, web_scrape futuro, etc): 150000 chars
    #     (~30K palabras = vídeos de 1+ hora)
    # Defense in depth (regex + XML wrap) escala bien a este tamaño:
    # is_suspicious sobre 150KB < 3ms con el motor C-backed de re.
    read_tool_max_chars: int = Field(
        default=150_000,
        ge=2500,
        le=1_000_000,
        description="Máximo de chars en output de read tools (agent_reach_*). "
        "Subir con precaución: aumenta coste de tokens proporcionalmente.",
    )
    system_tool_max_chars: int = Field(
        default=2500,
        ge=500,
        le=50_000,
        description="Máximo de chars en output de system tools (weather, status).",
    )

    # STT externo (Gemini 3.1 Flash Lite) - v1.2
    # Por qué (Sprint 12+ contexto): originalmente bug opencode/opencode#30389
    # obligaba a externalizar el audio (mimo-v2.5 NO procesaba input_audio
    # via OpenCode Go). Ahora con MiniMax-M3 (multimodal nativo) eso ya
    # no es limitante, pero MANTENEMOS STT externo por aislamiento de
    # cuota (Gemini free tier 500 RPD / 15 RPM no acopla audio al rate
    # limit del chat) y por coste (Flash Lite es gratis vs $0.30/$0.60
    # por M tokens de MiniMax-M3 multimodal).
    # Sprint 19.6+ Phase 5 (OpenAI Build Week): gemini_api_key es ahora
    # opcional. Sin key, el servicio de STT queda inactivo (los handlers
    # de audio devuelven warning en lugar de error). Mismo patron que
    # `opencode_go_api_key` y `telegram_bot_token` (Sprint 11 ADR-004):
    # Settings carga sin key, el runtime decide si la feature esta activa.
    gemini_api_key: str | None = Field(
        default=None,
        validation_alias="GEMINI_API_KEY",
    )
    stt_model: str = "gemini-3.1-flash-lite"
    stt_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    # Cola STT: limita concurrencia y tasa para no saturar la cuota.
    # 2 audios en paralelo + 12 RPM deja margen sobre los 15 RPM de Gemini.
    stt_max_concurrent: int = Field(default=2, ge=1, le=10)
    stt_per_minute: int = Field(default=12, ge=1, le=60)
    stt_queue_timeout_s: int = Field(default=60, ge=5, le=300)

    # Sprint 9.1 (renamed in 9.2): Embeddings para RAG via OpenRouter.
    #
    # **PRIVACIDAD CRÍTICA**: para una knowledge base personal con
    # papers/documentos sensibles, los datos embebidos NO deben usarse
    # para entrenar modelos. Por eso el default es OpenRouter con ZDR
    # (Zero Data Retention) forzado via header `X-OpenRouter-ZDR: true`.
    #
    # Modelo default: `qwen/qwen3-embedding-8b` (Qwen 3 Embedding 8B).
    # Por qué: 2x más barato que openai/text-embedding-3-small ($0.01/M
    # vs $0.02/M), 4x más context (32K vs 8K), mejor multilingual
    # (excelente en español/inglés), 4096-dim. Ver
    # https://openrouter.ai/qwen/qwen3-embedding-8b
    #
    # Alternativas via EMBEDDING_MODEL:
    # - `openai/text-embedding-3-small` (1536-dim, $0.02/M, 8K context)
    # - `voyage-3` (1024-dim, alta calidad, ~$0.06/M)
    # - Cualquier otro en https://openrouter.ai/models?q=embedding
    #
    # Providers:
    # 1. **OpenRouter** (default, ZDR): header `X-OpenRouter-ZDR: true`.
    # 2. **Gemini** (NO recomendado): free tier entrena con tus datos.
    #    Solo via EMBEDDING_PROVIDER=gemini explicitamente.
    #
    # Si ningún backend está disponible, RAG se desactiva (chat sigue
    # funcionando, búsqueda semántica no).
    #
    # Renombrado en Sprint 9.2: openai_api_key/base → openrouter_api_key/base
    # (los nombres viejos siguen funcionando como alias por backward compat).
    openrouter_api_base: str = Field(
        default="https://openrouter.ai/api/v1",
        validation_alias="OPENROUTER_API_BASE",
    )
    openrouter_api_key: str = Field(
        default="",
        validation_alias="OPENROUTER_API_KEY",
    )
    # Backward compat aliases (deprecated, will be removed in v1.0)
    openai_api_base: str = Field(
        default="",
        validation_alias="OPENAI_API_BASE",
    )
    openai_api_key: str = Field(
        default="",
        validation_alias="OPENAI_API_KEY",
    )
    # Default: Qwen 3 Embedding 8B (mejor precio, context, multilingual).
    embedding_model: str = Field(
        default="qwen/qwen3-embedding-8b",
        validation_alias="EMBEDDING_MODEL",
    )
    # Modelo Gemini (solo si EMBEDDING_PROVIDER=gemini). Default:
    # gemini-embedding-001 (3072-dim, reemplaza al deprecado
    # text-embedding-004). Sprint 16.5 empirical test: 404 confirmo
    # que text-embedding-004 fue removido de v1beta API en 2026.
    gemini_embedding_model: str = Field(
        default="gemini-embedding-001",
        validation_alias="GEMINI_EMBEDDING_MODEL",
    )
    # Override del provider. "openrouter" (default, OpenRouter con ZDR).
    # "gemini" = usar Gemini (NO ZDR, free tier entrena con tus datos).
    embedding_provider: str = Field(
        default="openrouter",
        validation_alias="EMBEDDING_PROVIDER",
    )
    # Cosine hard threshold: scores debajo de este valor se descartan.
    # Sprint 16.5 empirical test (2026-07-06): calibrated for
    # qwen/qwen3-embedding-8b (4096-dim, default via OpenRouter).
    # Threshold sweep on 30 synthetic facts:
    #   0.30 P=0.26 R=0.86 F1=0.40
    #   0.50 P=0.39 R=0.81 F1=0.52
    #   0.55 P=0.41 R=0.71 F1=0.52  <- chosen: balance F1 + lower noise
    #   0.60 P=0.43 R=0.50 F1=0.46
    #   0.70 P=0.21 R=0.29 F1=0.24
    #   0.82 P=0.00 R=0.00 F1=0.00  <- original; tuned for
    #                                  text-embedding-3-small 1536-dim
    #                                  which produces higher cosine scores.
    # If you change embedding backend, re-run calibration (see SP16_FOLLOWUP
    # F5: threshold calibration script).
    min_similarity_threshold: float = Field(
        default=0.55,
        ge=0.0,
        le=1.0,
        validation_alias="MIN_SIMILARITY_THRESHOLD",
    )

    # Sprint 9.2: Sleep Cycle (memory facts extraction).
    # Opt-in: si True, scheduler corre el job diario que extrae facts
    # de las conversaciones del día anterior y los promueve tras
    # `memory_fact_min_mentions` menciones en conversaciones distintas.
    sleep_cycle_enabled: bool = Field(default=False, validation_alias="SLEEP_CYCLE_ENABLED")
    # Hora del cron (formato 24h). Default 4 (4 AM, baja actividad).
    sleep_cycle_hour: int = Field(default=4, ge=0, le=23, validation_alias="SLEEP_CYCLE_HOUR")
    # Minimo de menciones en conversaciones distintas para promover
    # un fact de staging a memory_facts consolidado.
    # Threshold 3 = "el user lo ha mencionado 3+ veces en chats distintos,
    # es probable que sea una preferencia real, no ruido".
    memory_fact_min_mentions: int = Field(
        default=3, ge=1, le=20, validation_alias="MEMORY_FACT_MIN_MENTIONS"
    )
    # Timeout SQLite (P0-2 Gemini fix). El Sleep Cycle corre en
    # background y puede coincidir con writers activos (Telegram,
    # FastAPI). 30s da margen para que el lock se libere.
    db_connection_timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=120.0,
        validation_alias="DB_CONNECTION_TIMEOUT_SECONDS",
    )
    # Staging cleanup: facts en staging sin actividad > N días se
    # marcan como 'expired' y se eliminan en el siguiente ciclo.
    staging_expiration_days: int = Field(
        default=90, ge=7, le=365, validation_alias="STAGING_EXPIRATION_DAYS"
    )
    # Sprint 16 (US-3.2 memory facts retrieval): divisor del time decay
    # factor. Aplicado como `score * exp(-days_since_last_reference /
    # fact_time_decay_days)`. Default 30 dias = balance entre
    # "recency matters" y "facts established hace meses siguen valiendo".
    fact_time_decay_days: int = Field(
        default=30, ge=1, le=365, validation_alias="FACT_TIME_DECAY_DAYS"
    )
    # Sprint 16 (US-3.2 AgentLoop integration): total de chars disponibles
    # para el contexto del LLM (RAG vault + memory facts + system prompt).
    # Default 20000 chars es heuristica robusta para Anthropic Sonnet 4.5
    # (200k token context window, ~4 chars/token, holgura para thinking).
    max_context_chars: int = Field(
        default=20000,
        ge=1000,
        le=200000,
        validation_alias="MAX_CONTEXT_CHARS",
    )
    # Sprint 16 (US-3.2 AgentLoop integration): porcentaje del
    # max_context_chars reservado para memory facts del usuario.
    # Memoria del usuario SIEMPRE prioridad sobre vault RAG — el LLM
    # no debe olvidar quien es project owner mientras lee un PDF grande.
    # 10% = ~2000 chars para facts, ~18000 chars para vault en un
    # context default de 20000 chars. Gemini 3.1 Pro 2nd-pass:
    # 10% / 90% es heuristica robusta.
    memory_facts_token_budget_pct: float = Field(
        default=0.10,
        ge=0.0,
        le=0.5,
        validation_alias="MEMORY_FACTS_TOKEN_BUDGET_PCT",
    )

    db_path: Path = Path("/app/data/conversations.db")

    health_host: str = "0.0.0.0"
    health_port: int = 8000

    influx_url: str = ""
    influx_token: str = ""
    influx_org: str = ""  # Required: set via INFLUX_ORG env var
    influx_bucket: str = "hermes"

    log_level: str = "INFO"

    # Sprint 4 MVP-1: tool configuration
    # Zona horaria para get_current_time (IANA tz database)
    tz: str = "Europe/Madrid"
    # Path al Obsidian Vault para search_vault. Default: docs del usuario
    # en macOS. En Docker se sobreescribe vía env VAULT_PATH.
    vault_path: Path = Path.home() / "Documents" / "Obsidian Vault"
    # Sprint 17 Slice 1.5 (TDD_VAULT_INGEST_WORKER.md §"Settings additions"):
    # Feature flags para el ingest router 4-tier. TDD preamble — los
    # valores reales (de dónde cuelga el inbox writer, qué tier usar)
    # se wirean en el GREEN phase. Defaults seguros: Tier 1.5 ON
    # (NAS host es gateway), Tier 1/Tier 2 OFF (privacy + RAM).
    vault_lan_worker_enabled: bool = True  # VAULT_LAN_WORKER_ENABLED
    vault_external_ocr: bool = False  # VAULT_EXTERNAL_OCR
    vault_use_local_ocr: bool = False  # VAULT_USE_LOCAL_OCR  (Tier 1 Docling fail-fast at startup)
    vault_inbox_root: Path = Path("/var/lib/mnemosyne/inbox")  # VAULT_INBOX_ROOT
    # Sprint 19 Slice 4: drop folder watcher root. Si está habilitado,
    # el watcher detecta archivos nuevos en `<vault_drop_root>/<subdir>/`
    # y los inserta en vault_files + vault_file_collections + escribe
    # manifest para que process_inbox() (Sprint 17) los procese.
    # Default: <vault_inbox_root>/drop (co-located con el inbox legacy).
    vault_drop_root: Path | None = None  # VAULT_DROP_ROOT (None = auto-derive)
    vault_drop_enabled: bool = True  # VAULT_DROP_ENABLED (default on; opt-out)
    # Sprint 19 Slice 4d v2 + R1 retro: env var override for PARA seed.
    # Per the existing docstring in seed.py: "For i18n, future Sprint 22+
    # will add a `OROIMEN_DEFAULT_COLLECTIONS` Settings override (JSON list)."
    # Format: JSON list of {name, description, sort_order}:
    #   OROIMEN_DEFAULT_COLLECTIONS='[{"name":"01_Proyectos","description":"Mi lista","sort_order":10}, ...]'
    # Empty string = NO default PARA collections are seeded (true opt-in).
    # The user MUST set this explicitly if they want any collections
    # pre-created on first run. This is the transferability knob:
    # anyone forking the project can set their own PARA names without
    # modifying code, or skip the seed entirely by leaving it empty.
    #
    # Renamed from HERMES_DEFAULT_COLLECTIONS (commit 4fa313e) to
    # OROIMEN_DEFAULT_COLLECTIONS (Sprint 19 followup) per the
    # project rename to Oroimen. Breaking change: existing installs
    # that set the old var must rename to the new one.
    oroimen_default_collections_json: str = Field(
        default="",
        validation_alias="OROIMEN_DEFAULT_COLLECTIONS",
        description="JSON list of PARA default collections to seed on startup. "
        "Empty = no defaults (true opt-in).",
    )
    # Sprint 19 Slice 4d v2 (followup): inbox default collection. When a
    # file is dropped at the root of VAULT_DROP_ROOT (no subdir), instead
    # of skipping it, route it to this collection. True opt-in: empty
    # string = current skip behavior. Use to implement the "drop in _inbox
    # and classify later" UX.
    vault_drop_default_collection: str = Field(
        default="",
        validation_alias="VAULT_DROP_DEFAULT_COLLECTION",
        description="If set, root-level files in VAULT_DROP_ROOT go to this "
        "collection instead of being skipped. Empty = skip (legacy).",
    )
    # Sprint 19 Slice 4d v2 (commit 1 + commit 3): monitor folder design.
    # v0.7 §1 N1 fix: vault_monitor_roots is a str (read from
    # VAULT_MONITOR_ROOTS env var, comma-separated) + property that
    # splits into list[Path]. Pydantic v2 does NOT auto-split env vars
    # into list[X] (NB1 from correctness R1 r3), so the manual split
    # in the `vault_monitor_roots` property is required. The validation_alias
    # maps the env var VAULT_MONITOR_ROOTS to this field; the property
    # is the canonical list accessor.
    vault_monitor_roots_str: str = Field(default="", validation_alias="VAULT_MONITOR_ROOTS")
    vault_monitor_no_inbox: bool = False  # VAULT_MONITOR_NO_INBOX (default False = seed _inbox)
    vault_monitor_max_pending: int = 1000  # VAULT_MONITOR_MAX_PENDING (queue backpressure cap)
    vault_monitor_max_depth: int = 20  # VAULT_MONITOR_MAX_DEPTH (skip files deeper than N levels)
    # Sprint 19 Slice 4c (§4.4.1-§4.4.5): edge PC auto-queue + zombie recovery.
    # OCR_AUTO_EDGE_OCR: master switch for auto-queueing low-confidence
    # images to the edge PC. If False, all edge work requires explicit
    # `/edgeOCR <file_id>` user command (no auto-queue on the watcher).
    # Default True per TDD §4.4.1: "eliminates a class of 'should I
    # upgrade this?' decisions for project owner".
    ocr_auto_edge_ocr: bool = True  # OCR_AUTO_EDGE_OCR
    # OCR_EDGE_MAX_QUEUE_SIZE: catch-up pass drains at most this many
    # files per reconnect event. Overflow stays in `pending_review`
    # for the next reconnect, or for manual `/edgeOCR` targeting.
    # Default 1000 = 10x worst-case "I was on travel for 2 weeks" load.
    ocr_edge_max_queue_size: int = 1000  # OCR_EDGE_MAX_QUEUE_SIZE
    # OCR_EDGE_BATCH_DELAY_MS: debounce between catch-up enqueue calls.
    # 500ms = 200 files drain in 100s. Avoids hammering the SMB share
    # with rapid metadata writes. See TDD §4.4.4.
    ocr_edge_batch_delay_ms: int = 500  # OCR_EDGE_BATCH_DELAY_MS
    # OCR_EDGE_TIMEOUT_HOURS: M6 Phase 5 zombie recovery threshold.
    # Rows in `ocr_pending.status='edge_queued'` older than this get
    # reset to `pending_review`. Default 2h = 10x worst-case LLaVA
    # batch time on a 50-image batch (~12min serial).
    ocr_edge_timeout_hours: int = 2  # OCR_EDGE_TIMEOUT_HOURS
    # OCR_EDGE_ZOMBIE_SCAN_INTERVAL: how often M6 Phase 5 runs. Default
    # 900s (15min). Cheap SELECT on partial index, no perf concern.
    ocr_edge_zombie_scan_interval: int = 900  # OCR_EDGE_ZOMBIE_SCAN_INTERVAL
    # EDGE_COMPUTERS: comma-separated hostnames/IPs of edge PCs to
    # probe. Empty = no auto-queue, only manual `/edgeOCR` works.
    # Example: "edge.local,edge.example.com". See TDD §4.4.4.
    edge_computers: str = ""  # EDGE_COMPUTERS
    edge_smb_root_prefix: str = Field(
        default="/mnt/shared/",
        validation_alias="EDGE_SMB_ROOT_PREFIX",
        description="Shared-root prefix stripped before sending paths to edge workers.",
    )
    # OCR_EDGE_AUTOQUEUE_THRESHOLD: confidence cutoff for auto-queue
    # to edge. Per TDD §4.4.1: "0.85, not 0.60, because edge is
    # free local compute; eliminates upgrade decisions". Default 0.85.
    ocr_edge_autoqueue_threshold: float = 0.85  # OCR_EDGE_AUTOQUEUE_THRESHOLD
    # EDGE_PROBE_TIMEOUT_S: HTTP probe timeout per PC. Default 1s =
    # 30s for 30 PCs in a single poll cycle.
    edge_probe_timeout_s: float = 1.0  # EDGE_PROBE_TIMEOUT_S
    # EDGE_PROBE_INTERVAL_S: how often to poll PC availability. Default
    # 30s. Cheap single HTTP probe per PC.
    edge_probe_interval_s: float = 30.0  # EDGE_PROBE_INTERVAL_S
    # Sprint 19 Slice 4d (TDD §4.3.1): per-user daily cap for the
    # /externalOCR 2-step request. Exceeding it raises RateLimitedError
    # (429 on WebUI). Default 10/day = sane cap for a personal assistant
    # that may be used intermittently. Override via env.
    # R1 fix (2026-07-11): was os.environ.get at runtime in
    # ocr_decision._check_rate_limit. Moved to Settings for consistency
    # with the rest of the config (validated at startup, no per-call env
    # lookup, pydantic validation).
    external_ocr_daily_limit: int = Field(
        default=10,
        ge=1,
        le=1000,
        validation_alias="EXTERNAL_OCR_DAILY_LIMIT",
        description="Per-user daily cap for /externalOCR requests.",
    )
    # Sprint 19 Slice 4d (B1 fix, 2026-07-11): which OcrProvider the
    # /externalOCR command uses. Default "hosted_llm" (wraps the LLM
    # client, model = Settings.llm_text_primary, e.g. MiniMax-M3).
    # Sprint 22+ can add "local_vision" (LLaVA via Ollama) or
    # "edge_ocr_v2" (edge PC vision) without changing the OCR
    # decision logic. Provider-agnostic at the OcrProvider interface
    # level — never hardcoded to a specific model or API.
    ocr_default_provider: str = Field(
        default="hosted_llm",
        validation_alias="OCR_DEFAULT_PROVIDER",
        description="Default OcrProvider for /externalOCR. 'hosted_llm' "
        "(Sprint 4d) or 'local_vision' / 'edge_ocr_v2' (Sprint 22+).",
    )
    # Sprint 19.6+ Phase 3 (LocalVisionOcrProvider, TDD v0.6 §10):
    # Settings for the local vision OCR backend (Ollama on the edge host).
    # Master switch is OFF by default (opt-in, Sprint 19 north star:
    # "no automatic hosted APIs"). The /externalOCR command can still
    # explicitly select 'local_vision' provider even if disabled here
    # (the disable only blocks automatic drop-folder routing, not the
    # explicit user command).
    local_vision_enabled: bool = Field(
        default=False,
        validation_alias="LOCAL_VISION_ENABLED",
        description="Master switch for LocalVisionOcrProvider. "
        "Default False (opt-in, per Sprint 19 north star).",
    )
    local_vision_model: str = Field(
        default="qwen3-vl:2b",
        validation_alias="LOCAL_VISION_MODEL",
        description="Ollama-served vision model for OCR. Default "
        "qwen3-vl:2b (fastest, ~2GB VRAM). Override to qwen3-vl:4b "
        "(balanced) or qwen3-vl:8b (best quality, ~6GB VRAM).",
    )
    local_vision_base_url: str = Field(
        default="http://localhost:11434",
        validation_alias="LOCAL_VISION_BASE_URL",
        description="Ollama API base URL. Default http://localhost:11434 "
        "(Ollama default). For local edge host use, localhost is the only "
        "safe setting; the data must NOT leave the machine.",
    )
    local_vision_timeout_s: float = Field(
        default=120.0,
        gt=0.0,
        validation_alias="LOCAL_VISION_TIMEOUT_S",
        description="HTTP timeout for the Ollama /api/generate call. "
        "Default 120s to accommodate heavy thinking-mode models like "
        "qwen3-vl:8b (~30-45s latency on subsequent calls).",
    )
    local_vision_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        validation_alias="LOCAL_VISION_TEMPERATURE",
        description="Ollama generation temperature. Default 0.0 "
        "(deterministic for OCR; variance is undesirable).",
    )
    local_vision_num_predict: int = Field(
        default=4096,
        ge=128,
        le=32768,
        validation_alias="LOCAL_VISION_NUM_PREDICT",
        description="Ollama num_predict. Default 4096 to accommodate "
        "qwen3-vl thinking mode (model emits ~800 reasoning steps "
        "before the final answer; smaller values truncate mid-thinking).",
    )
    # Sprint 19.6+ Phase 3 (TDD v0.7 §10.4.4 MINOR-1): gate the
    # verbose error message (which reveals local topology + ollama
    # pull hint). Default False for prod; True for local dev where
    # the actionable message is useful.
    ocr_verbose_errors: bool = Field(
        default=False,
        validation_alias="OCR_VERBOSE_ERRORS",
        description="If True, OCR error messages include local topology "
        "(model name, `ollama pull` hint, daemon URL). If False, "
        "errors are generic ('OCR failed; check logs').",
    )
    vault_text_v0_strip_threshold: int = 100  # VAULT_TEXT_V0_STRIP_THRESHOLD
    # Sprint 17 Slice 1.5 hot-patch (PR #113b, m1 fix): umbral del
    # janitor en segundos. El worker toca mtime cada 30s mientras
    # procesa; si pasa este umbral sin touch, asumimos crash y
    # movemos el job de processing/ → pending/ para re-encolar.
    # Antes hardcoded 600s; ahora configurable para ops que tengan
    # workers lentos (large PDFs, GPU warming) o quieran janitor
    # más agresivo (e.g. 300s en NAS host con 8GB RAM).
    vault_janitor_stale_threshold_s: int = 600  # VAULT_JANITOR_STALE_THRESHOLD_S
    # PR #113b (B1 fix): enable el wire-ing del IngestRouter en
    # vault.add() — tras un add() exitoso, kick off Tier 0 extract.
    # Default True en prod; tests unit pueden setear False.
    vault_auto_ingest_on_add: bool = True  # VAULT_AUTO_INGEST_ON_ADD
    # Sprint 18 hardening (M6 vacuum): archive ingest_jobs rows whose
    # state is 'applied' or 'failed' and older than this many days.
    # Soft-vacuum only — the row gets state='archived' (still queryable
    # via SELECT state='archived') but the operator can hard-DELETE
    # separately if needed. Default 30 días balancea "dashboard
    # history visible" vs "tabla no crece infinito".
    # Runs daily via VaultScheduler (see scheduler.py).
    # PR #118 review (LLM cascade, SUGGESTION): Pydantic validator below
    # rejects <= 0 at startup, not at vacuum runtime. Defense-in-depth
    # against env-var typos (VAULT_DONE_ARCHIVE_AFTER_DAYS=0 would archive
    # everything immediately on first cycle).
    vault_done_archive_after_days: int = 30  # VAULT_DONE_ARCHIVE_AFTER_DAYS
    # Sprint 17 Slice 2.5 (TDD_VAULT_EMBEDDINGS.md §"Settings additions"):
    # Embedder + Watcher knobs.
    # Sprint 19.5 (PR-B, 2026-07-13): defaults changed to align with
    # `embedding_model` (RAG/chat) — both now default to Qwen 3 Embedding
    # 8B. Before this, vault used text-embedding-3-small (1536-dim, $0.02/M)
    # while RAG used qwen/qwen3-embedding-8b (4096-dim, $0.01/M). The
    # inconsistency caused:
    # 1. Cost: vault was 2x more expensive than RAG
    # 2. Quality: 1536-dim is lower than 4096-dim
    # 3. Multilingual: Qwen 3 8B is better than OpenAI 3-small for es/de/en
    # 4. Context: 32K vs 8K (chunk_max_tokens=1000 ~ 4000 chars, both fine)
    # After PR-C-svc lands, the actual model used will be Qwen3-Embedding-0.6B
    # (in-process on NAS via fastembed) for NAS tier, and qwen3-embedding:8b
    # (Ollama on edge) for edge tier. But the env var name stays the same
    # for backward compat — operators can override per deployment.
    vault_embedding_model: str = Field(  # VAULT_EMBEDDING_MODEL
        default="qwen/qwen3-embedding-8b",
        description="Modelo de embeddings. Override via env VAULT_EMBEDDING_MODEL. "
        "Default: qwen/qwen3-embedding-8b (Sprint 19.5, was text-embedding-3-small).",
    )
    vault_embedding_dim: int = Field(  # VAULT_EMBEDDING_DIM
        default=4096,
        description="Dimensión del vector legacy. Default: 4096 (Qwen 3 Embedding 8B).",
    )
    vault_chunk_max_tokens: int = Field(  # VAULT_CHUNK_MAX_TOKENS
        default=1000,
        description="Tokens máximos por chunk. Default 1000 (~4000 chars).",
    )
    vault_chunk_overlap_tokens: int = Field(  # VAULT_CHUNK_OVERLAP_TOKENS
        default=100,
        description="Tokens de overlap entre chunks consecutivos. 10% del max.",
    )
    # Sprint 19.5 (PR-A, 2026-07-13): PDF OCR fallback for scanned PDFs.
    # When pymupdf returns empty text (scanned PDF without text layer),
    # the extractor rasterizes each page to a temp PNG and runs tesseract
    # on it. Default 200 DPI grayscale (~13MB/page A4), configurable up to
    # 600 DPI (anti-OOM hard cap). Gemini review 2026-07-13: 200 DPI is
    # sufficient for tesseract on printed text; 300 DPI is overkill for
    # most docs and triples memory; grayscale reduces ~3x vs RGB.
    # Streaming pattern (write PNG, tesseract, delete) keeps memory
    # bounded to 1 page at a time (~13MB peak vs 1GB+ for 40-page doc).
    vault_ocr_fallback_dpi: int = Field(  # VAULT_OCR_FALLBACK_DPI
        default=200,
        description="DPI para rasterizar PDF escaneados a imagen. 200 default. Max 600 (anti-OOM).",
    )
    vault_ocr_fallback_grayscale: bool = Field(  # VAULT_OCR_FALLBACK_GRAYSCALE
        default=True,
        description="Convertir a grayscale antes de tesseract. Reduce ~3x el tamaño. Tesseract no necesita color.",
    )
    vault_ocr_fallback_lang: str = Field(  # VAULT_OCR_FALLBACK_LANG
        default="deu+eng+spa",
        description="Idiomas para tesseract. Default deu (alemán) + eng + spa. Vacío = usar default de tesseract.",
    )
    vault_watcher_poll_interval_s: int = Field(  # VAULT_WATCHER_POLL_INTERVAL_S
        default=300,
        description="Segundos entre polls del EmbedWatcher. Default 5 min.",
    )
    # Sprint 19.5 Slice 6 Commit 3: multi-tier embeddings opt-in.
    # Cuando EMBEDDING_TIER_<NAME>__ENABLED=true, EmbeddingsService
    # usa el EmbeddingRouter (hermes/services/embedding_router.py) en
    # vez del legacy single-backend. Si TODOS los __ENABLED=false o
    # no se setean, cae al legacy (OpenRouter / Gemini).
    #
    # Convencion de nombres: 5 campos por tier (ENABLED, BASE_URL,
    # MODEL, API_KEY, TIMEOUT_S). El nombre de la env var usa doble
    # underscore como separador (estilo pydantic-settings, igual que
    # ``LLM_ALLOWED_MODELS``). El campo Python usa doble underscore
    # para mapear exacto (validation_alias). TDD §2.2 / §3 commit 3.
    #
    # Tiers: ``nas`` (sidecar en NAS host, granite-311m per Sprint 19.5
    # hotfix v0.17 2026-07-15), ``edge`` (Ollama en edge host,
    # qwen3-embedding:8b), ``cloud`` (OpenRouter u otro OpenAI-compat,
    # qwen-8b). El factory ``_build_router`` valida Rule 7 (enabled
    # tier sin base_url/model -> ConfigError).
    embedding_tier_nas__enabled: bool = Field(
        default=False,
        validation_alias="EMBEDDING_TIER_NAS__ENABLED",
        description="Activa el tier NAS (sidecar granite-311m en NAS host). "
        "Default false. Si True, EMBEDDING_TIER_NAS__BASE_URL y __MODEL "
        "son requeridos (Rule 7).",
    )
    embedding_tier_nas__base_url: str = Field(
        default="",
        validation_alias="EMBEDDING_TIER_NAS__BASE_URL",
        description="Base URL del sidecar NAS (e.g. http://<internal-host>:8082/v1).",
    )
    embedding_tier_nas__model: str = Field(
        default="",
        validation_alias="EMBEDDING_TIER_NAS__MODEL",
        description="Nombre del modelo en el sidecar (e.g. granite-311m).",
    )
    embedding_tier_nas__api_key: str = Field(
        default="",
        validation_alias="EMBEDDING_TIER_NAS__API_KEY",
        description="API key para el sidecar NAS. Vacio = sin auth (LAN).",
    )
    embedding_tier_nas__timeout_s: float = Field(
        default=30.0,
        validation_alias="EMBEDDING_TIER_NAS__TIMEOUT_S",
        description="Timeout HTTP en segundos. Default 30.",
        ge=0.1,
        le=300.0,
    )
    embedding_tier_edge__enabled: bool = Field(
        default=False,
        validation_alias="EMBEDDING_TIER_EDGE__ENABLED",
        description="Activa el tier edge (Ollama en edge host).",
    )
    embedding_tier_edge__base_url: str = Field(
        default="",
        validation_alias="EMBEDDING_TIER_EDGE__BASE_URL",
        description="Base URL de Ollama en el edge host (e.g. http://edge.local:8800/v1).",
    )
    embedding_tier_edge__model: str = Field(
        default="",
        validation_alias="EMBEDDING_TIER_EDGE__MODEL",
        description="Nombre del modelo Ollama (e.g. qwen3-embedding:8b).",
    )
    embedding_tier_edge__api_key: str = Field(
        default="",
        validation_alias="EMBEDDING_TIER_EDGE__API_KEY",
        description="API key del edge. Vacio = sin auth (LAN).",
    )
    embedding_tier_edge__timeout_s: float = Field(
        default=30.0,
        validation_alias="EMBEDDING_TIER_EDGE__TIMEOUT_S",
        description="Timeout HTTP en segundos. Default 30.",
        ge=0.1,
        le=300.0,
    )
    embedding_tier_cloud__enabled: bool = Field(
        default=False,
        validation_alias="EMBEDDING_TIER_CLOUD__ENABLED",
        description="Activa el tier cloud (OpenRouter u OpenAI-compat).",
    )
    embedding_tier_cloud__base_url: str = Field(
        default="",
        validation_alias="EMBEDDING_TIER_CLOUD__BASE_URL",
        description="Base URL del provider cloud (e.g. https://openrouter.ai/api/v1).",
    )
    embedding_tier_cloud__model: str = Field(
        default="",
        validation_alias="EMBEDDING_TIER_CLOUD__MODEL",
        description="Nombre del modelo cloud (e.g. qwen/qwen3-embedding-8b).",
    )
    embedding_tier_cloud__api_key: str = Field(
        default="",
        validation_alias="EMBEDDING_TIER_CLOUD__API_KEY",
        description="API key del provider cloud. Requerido si el provider usa auth.",
    )
    embedding_tier_cloud__timeout_s: float = Field(
        default=30.0,
        validation_alias="EMBEDDING_TIER_CLOUD__TIMEOUT_S",
        description="Timeout HTTP en segundos. Default 30.",
        ge=0.1,
        le=300.0,
    )
    # Policies: CSV de nombres de tier en orden de prioridad. El primer
    # nombre es el canonical (almacenamiento + retrieval). El resto son
    # fallbacks para ``embed()`` live. Se parsean en
    # ``_build_router`` (factory de EmbeddingRouter) leyendo el env var
    # raw, NO como lista tipada (evita acoplar pydantic al orden de
    # prioridad como list). TDD §2.1 "first-wins".
    embedding_policy_chat_rag: str = Field(
        default="",
        validation_alias="EMBEDDING_POLICY_CHAT_RAG",
        description="CSV de tiers para chat RAG (live, low-latency). "
        "Default v0.16: 'nas' (B-1 fix, single tier). El primer tier "
        "es el canonical; el resto son fallbacks (no cross-dim).",
    )
    embedding_policy_vault_ingest: str = Field(
        default="",
        validation_alias="EMBEDDING_POLICY_VAULT_INGEST",
        description="CSV de tiers para vault ingest (batch, quality). "
        "Default v0.16: 'cloud,edge' (cloud primary, edge privacy-fallback). "
        "Dentro de un policy, todos los tiers deben compartir dim (Rule 3).",
    )
    # Versión de Oroimen (para get_system_status). Hardcoded para no
    # añadir dependencia de __init__.py en runtime paths.
    version: str = "0.4.0"
    # Feature flag: si True y hay tools registradas, el handler usa
    # AgentLoop (con tool calls) en vez de router.chat() directo.
    # Default False hasta validar en producción.
    tools_enabled: bool = False
    outbound_tools_enabled: bool = Field(
        default=False,
        validation_alias="OUTBOUND_TOOLS_ENABLED",
        description="Enable tools that can make outbound network requests.",
    )

    # Sprint 11 (ADR-004): WebUI-primary. HTTP API es ahora la
    # interfaz principal, default True. Open WebUI (Sprint 7) consume
    # este endpoint. Permite también que la app Android nativa (Sprint 11)
    # use el mismo OpenAI-compatible API.
    #
    # Migración: en S11.0 (Phase 1) cambia default False -> True para
    # preparar el rollout de la app Android. Si enable_http_api=True
    # Y enable_telegram=True, ambos arrancan (dual mode durante
    # transición). Sprint 12+ la app Android será primary, Telegram
    # opt-in legacy.
    #
    # Si True, arranca FastAPI en hermes_api_port y desactiva
    # HealthServer (mismo puerto, conflicto). HealthServer solo se
    # inicia si enable_http_api=False.
    enable_http_api: bool = Field(default=True, validation_alias="ENABLE_HTTP_API")
    hermes_api_host: str = Field(default="0.0.0.0", validation_alias="HERMES_API_HOST")
    hermes_api_port: int = Field(default=8000, ge=1, le=65535, validation_alias="HERMES_API_PORT")
    http_api_cors_origins: str = Field(
        default="http://localhost:8080,http://127.0.0.1:8080",
        validation_alias="HERMES_API_CORS_ORIGINS",
        description="Comma-separated browser origins allowed to call the HTTP API.",
    )

    # Sprint 11 (ADR-004): Gate para PollingReceiver (Telegram bot).
    # Sprint 11.0 mantiene default True para no romper deployments
    # existentes con Telegram. Sprint 12+ migrara a default False
    # (la app Android sera primary, Telegram opt-in legacy).
    #
    # Si enable_telegram=True y telegram_bot_token no esta configurado,
    # hermes arranca con warning (no crash) — el bot no se conecta
    # pero el resto (HTTP API, HealthChecker, etc) sigue funcionando.
    enable_telegram: bool = Field(default=True, validation_alias="ENABLE_TELEGRAM")
    # Sprint 9.5: bearer token para /v1/chat/completions y /v1/files/*.
    # None = no auth (legacy). Con valor = requiere Authorization: Bearer <token>.
    # Open-webui envia el token via OPENAI_API_KEYS (mismo nombre que open-webui
    # espera de cualquier OpenAI-compatible endpoint). /health y /v1/models
    # quedan SIN auth (liveness checks + open-webui model discovery).
    # Default None mantiene retrocompat con deployments que no quieren auth
    # (ej: hermes expuesto solo en LAN privada sin open-webui).
    http_api_api_key: str | None = Field(default=None, validation_alias="HERMES_API_API_KEY")

    # Sprint 8 S8.4: SQLite WAL backup online (scheduled job a 03:30 AM).
    # backup_enabled: opt-out para desactivar (ej: en CI/tests).
    # backup_dir: directorio destino de los .db DENTRO del container.
    #   An operator-specific Compose override can mount any writable host
    #   backup directory at /app/backups.
    # backup_keep: rotacion, mantener los N mas recientes.
    # backup_hour/minute: configurable local cron schedule.
    backup_enabled: bool = Field(default=True, validation_alias="BACKUP_ENABLED")
    backup_dir: Path = Field(default=Path("/app/backups"), validation_alias="BACKUP_DIR")
    backup_keep: int = Field(default=7, ge=1, le=30, validation_alias="BACKUP_KEEP")
    backup_hour: int = Field(default=3, ge=0, le=23, validation_alias="BACKUP_HOUR")
    backup_minute: int = Field(default=30, ge=0, le=59, validation_alias="BACKUP_MINUTE")

    # === Fase 0 Hardening: Rate limiting en HTTP API ===
    # Defensa contra DDoS accidental y cost-attack (cada request = 1 LLM
    # call = $$ en opencode-go). Por IP, ventana movil 60s.
    # /health y /v1/models exentos (liveness + open-webui discovery
    # hace multiples requests seguidas).
    # 0 = deshabilitado. Default 60 req/min (1 por segundo sostenido).
    http_api_rate_limit_per_minute: int = Field(
        default=60, ge=0, le=10000, validation_alias="HTTP_API_RATE_LIMIT_PER_MINUTE"
    )
    # Habilita el HealthChecker periodico (loop cada `check_interval_seconds`)
    # que envia push notifications via Telegram Bot API cuando detecta
    # problemas: HTTP API down, disk low, DB error, backup failed.
    # Dedup 1h por alert_type. Opt-out para tests/CI.
    push_notifications_enabled: bool = Field(
        default=True, validation_alias="PUSH_NOTIFICATIONS_ENABLED"
    )
    push_notification_cooldown_seconds: int = Field(
        default=3600, ge=0, le=86400, validation_alias="PUSH_NOTIFICATION_COOLDOWN_SECONDS"
    )
    # Intervalo del health check loop (default 60s). Mas bajo = mas
    # frecuente, pero mas overhead.
    health_check_interval_seconds: int = Field(
        default=60, ge=10, le=3600, validation_alias="HEALTH_CHECK_INTERVAL_SECONDS"
    )
    # Chat ID del user en Telegram (donde enviar alerts). Si falta,
    # el notifier es no-op. Setup: user inicia conversacion con el bot
    # y anota su chat_id (Telegram lo expone via getUpdates).
    telegram_chat_id: str | None = Field(default=None, validation_alias="TELEGRAM_CHAT_ID")

    # Sprint 12.1 (TDD_S12_DELETE_AND_SYNC.md §7.3): cifrado de tombstoned
    # conversations. Fernet key (bytes) generada con:
    #   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
    # Almacenar en .env (NUNCA en repo ni en git). Si falta, DELETE hace
    # plain archive sin cifrado (TDD §7.2 safe-fallback). POST /restore
    # falla con 503 si no hay key (no podemos descifrar sin ella).
    conversation_encryption_key: str | None = Field(
        default=None,
        validation_alias="HERMES_CONVERSATION_ENCRYPTION_KEY",
        description="Fernet key para cifrar content de tombstoned conversations",
    )

    # Sprint 12.1: dias de retencion de tombstoned conversations antes
    # del hard delete. Default 7d, ajustable via env. El job diario
    # `purge_tombstoned_conversations` corre cada 24h y hard-deletea
    # convs con purge_at <= NOW().
    conversation_retention_days: int = Field(
        default=7,
        validation_alias="HERMES_CONVERSATION_RETENTION_DAYS",
        ge=1,
        le=90,
        description="Dias de retencion de tombstoned convs antes de hard delete",
    )
    # Job periodico que archiva convs activas con updated_at viejo para
    # liberar combinaciones (chat_id, thread_id, user_id) y evitar el
    # bug 9.3.2b (UNIQUE constraint violation en idx_conversations_unique_active).
    # Opt-out: CLEANUP_ENABLED=false desactiva el job (ej: tests, CI).
    # Interval: cada N minutos (default 5, agresivo para evitar acumular huérfanas).
    # max_age: edad maxima en minutos (default 60). Convs mas viejas se archivan.
    cleanup_enabled: bool = Field(default=True, validation_alias="CLEANUP_ENABLED")
    cleanup_interval_minutes: int = Field(
        default=5, ge=1, le=60, validation_alias="CLEANUP_INTERVAL_MINUTES"
    )
    cleanup_max_age_minutes: int = Field(
        default=60, ge=5, le=1440, validation_alias="CLEANUP_MAX_AGE_MINUTES"
    )

    # === Sprint 9.3: Web Search Router ===
    # Habilita el tool hermes_search (default True; opt-out para tests/dev).
    search_enabled: bool = Field(default=False, validation_alias="SEARCH_ENABLED")
    # URL del SearXNG container. Default asume docker internal network.
    search_searxng_url: str = Field(
        default="http://searxng:8888", validation_alias="SEARCH_SEARXNG_URL"
    )
    # Default de resultados (1-50). Hard cap en router.
    search_default_num_results: int = Field(
        default=10, ge=1, le=50, validation_alias="SEARCH_DEFAULT_NUM_RESULTS"
    )
    search_max_num_results: int = Field(
        default=50, ge=1, le=50, validation_alias="SEARCH_MAX_NUM_RESULTS"
    )
    # S9.3.1 punto 4: size guard 50K → 200K. A $0.30/$0.60 per M tokens
    # (MiniMax-M3 input/output), 200K chars = ~50K tokens ≈ $0.015 input
    # / $0.03 output por search. Aceptable para uso personal; bajar si
    # se observa deriva. Permite que el LLM tenga más contexto
    # (cross-reference) en lugar de truncar agresivo. Métrica hit-rate
    # en logs para saber si el límite es relevante.
    search_size_guard_chars: int = Field(
        default=200000, ge=1000, le=500000, validation_alias="SEARCH_SIZE_GUARD_CHARS"
    )
    search_timeout_searxng: float = Field(
        default=10.0, ge=1.0, le=60.0, validation_alias="SEARCH_TIMEOUT_SEARXNG"
    )
    search_timeout_tavily: float = Field(
        default=15.0, ge=1.0, le=60.0, validation_alias="SEARCH_TIMEOUT_TAVILY"
    )
    search_timeout_exa: float = Field(
        default=15.0, ge=1.0, le=60.0, validation_alias="SEARCH_TIMEOUT_EXA"
    )
    # S9.3.1 punto 2: concurrency per-backend (Tavily/Exa mas permisivos que SearXNG).
    # DEPRECATED: search_max_concurrent (legacy S9.3.0). No se usa en el wireup
    # (__main__.py inicializa ConcurrencyLimiter con limits per-backend).
    # Se mantiene para backward compat con .env existentes pero se eliminara en S9.4.
    # Si los 3 nuevos per-backend no estan set, se usan estos defaults.
    search_max_concurrent: int = Field(
        default=3, ge=1, le=10, validation_alias="SEARCH_MAX_CONCURRENT"
    )
    # SearXNG: engines upstream (DuckDuckGo ~5 req/s). Default conservador.
    search_max_concurrent_searxng: int = Field(
        default=6, ge=1, le=20, validation_alias="SEARCH_MAX_CONCURRENT_SEARXNG"
    )
    # Tavily: 100 req/min documented. Default conservador (10 = 600 req/hr).
    search_max_concurrent_tavily: int = Field(
        default=10, ge=1, le=30, validation_alias="SEARCH_MAX_CONCURRENT_TAVILY"
    )
    # Exa: 50 req/min documented. Default conservador.
    search_max_concurrent_exa: int = Field(
        default=5, ge=1, le=20, validation_alias="SEARCH_MAX_CONCURRENT_EXA"
    )
    # Circuit breaker: N fails -> open por TTL segundos.
    search_circuit_breaker_threshold: int = Field(
        default=3, ge=1, le=20, validation_alias="SEARCH_CIRCUIT_BREAKER_THRESHOLD"
    )
    search_circuit_breaker_ttl_seconds: int = Field(
        default=300, ge=10, le=3600, validation_alias="SEARCH_CIRCUIT_BREAKER_TTL"
    )

    # Tavily (deep_research intent). Opcional: si vacio, el backend no se crea.
    tavily_api_key: str = Field(default="", validation_alias="TAVILY_API_KEY")
    tavily_monthly_limit: int = Field(
        default=1000, ge=0, le=100000, validation_alias="TAVILY_MONTHLY_LIMIT"
    )

    # Exa (semantic intent). Opcional: si vacio, el backend no se crea.
    exa_api_key: str = Field(default="", validation_alias="EXA_API_KEY")
    exa_monthly_limit: int = Field(
        default=1000, ge=0, le=100000, validation_alias="EXA_MONTHLY_LIMIT"
    )

    # === Sprint 14 (TDD_S14_DEEP_RESEARCH.md §5): Deep Research settings ===
    deep_research_enabled: bool = Field(
        default=False,
        validation_alias="HERMES_DEEP_RESEARCH_ENABLED",
        description="Explicit opt-in for the supported Deep Research runtime.",
    )

    # 13 settings nuevas. Defaults críticos razonados:
    # - daily_budget=3.0: cubre ~75 jobs/dia a $0.04/job (post-user decision 2026-07-02,
    #   subido de 1.0 → 3.0 para evitar cap en mal momento). Single-tenant personal use.
    # - max_sources=5: sweet spot cost/quality per TDD critique §5.
    # - output_max_tokens=10000: LLM ~8-12K para research profundo; 10K es límite defensivo.
    # - per_source_max_tokens=3000: summaries 2-3K son sweet spot.
    # - recovery_drop_orphan_hours=168 (7d): un pending que lleva 7d sin arrancar es bug.

    # Budget caps
    deep_research_daily_budget_usd: float = Field(
        default=3.0,
        ge=0.0,
        le=100.0,
        validation_alias="HERMES_DEEP_RESEARCH_DAILY_BUDGET_USD",
        description="Daily cap agregado para research jobs (USD).",
    )
    deep_research_per_job_budget_usd: float = Field(
        default=5.0,
        ge=0.0,
        le=100.0,
        validation_alias="HERMES_DEEP_RESEARCH_PER_JOB_BUDGET_USD",
        description="Soft alert per job (USD); log warning si excede, no cancela.",
    )

    # Concurrency limits
    deep_research_max_concurrent_per_user: int = Field(
        default=3,
        ge=1,
        le=20,
        validation_alias="HERMES_DEEP_RESEARCH_MAX_CONCURRENT_PER_USER",
        description="Max jobs simultáneos por user_id (S14: sentinel 0).",
    )

    # TTL
    deep_research_data_jobs_ttl_days: int = Field(
        default=30,
        ge=1,
        le=365,
        validation_alias="HERMES_DEEP_RESEARCH_DATA_JOBS_TTL_DAYS",
        description="Cleanup de data/jobs/{id}/ tras N días.",
    )

    # Per-phase timeouts (segundos). Rationale por phase:
    # - phase1 (search Tavily): 30s suficiente para deep_research intent.
    # - phase2 (scrape HTTP per source): 30s; size guard 2MB ANTES de to_thread.
    # - phase3 (per-source LLM): 90s; ~5 calls en paralelo, MiniMax-M3.
    # - phase4 (final synthesis LLM): 120s; 1 call con hasta 10K tokens output.
    # - phase5 (file write): 5s; atomic write, debe ser casi instantáneo.
    deep_research_phase1_timeout_s: int = Field(
        default=30,
        ge=5,
        le=300,
        validation_alias="HERMES_DEEP_RESEARCH_PHASE1_TIMEOUT_S",
    )
    deep_research_phase2_timeout_s: int = Field(
        default=30,
        ge=5,
        le=300,
        validation_alias="HERMES_DEEP_RESEARCH_PHASE2_TIMEOUT_S",
    )
    deep_research_phase3_timeout_s: int = Field(
        default=90,
        ge=10,
        le=600,
        validation_alias="HERMES_DEEP_RESEARCH_PHASE3_TIMEOUT_S",
    )
    deep_research_phase4_timeout_s: int = Field(
        default=120,
        ge=10,
        le=600,
        validation_alias="HERMES_DEEP_RESEARCH_PHASE4_TIMEOUT_S",
    )
    deep_research_phase5_timeout_s: int = Field(
        default=5,
        ge=1,
        le=60,
        validation_alias="HERMES_DEEP_RESEARCH_PHASE5_TIMEOUT_S",
    )

    # Output limits
    deep_research_max_sources: int = Field(
        default=5,
        ge=1,
        le=20,
        validation_alias="HERMES_DEEP_RESEARCH_MAX_SOURCES",
        description="Cuántas URLs scrapeamos en phase 2.",
    )
    deep_research_output_max_tokens: int = Field(
        default=10000,
        ge=500,
        le=50000,
        validation_alias="HERMES_DEEP_RESEARCH_OUTPUT_MAX_TOKENS",
        description="Max output de phase 4 (final synthesis).",
    )
    deep_research_per_source_max_tokens: int = Field(
        default=3000,
        ge=500,
        le=10000,
        validation_alias="HERMES_DEEP_RESEARCH_PER_SOURCE_MAX_TOKENS",
        description="Max output de phase 3 (per-source synthesis).",
    )

    # === DR-Q1A-PRE1B: real Deep Research cancellation contract ===
    # How long the cancel endpoint (graceful=True) waits for the
    # local asyncio task to acknowledge cancellation before returning.
    # This is NOT a provider cancellation timeout, NOT a billing
    # guarantee, NOT a phase-completion timeout, NOT a hard monetary
    # boundary. It only bounds how long the HTTP caller waits.
    deep_research_cancel_wait_s: float = Field(
        default=5.0,
        ge=0.1,
        le=30.0,
        validation_alias="HERMES_DEEP_RESEARCH_CANCEL_WAIT_S",
        description=(
            "Bounded wait (seconds) for cancel endpoint when "
            "graceful=True. Controls how long the local asyncio "
            "task has to acknowledge cancellation. Not a provider "
            "or billing guarantee."
        ),
    )

    # Notification
    deep_research_notify_via_tg_default: bool = Field(
        default=True,
        validation_alias="HERMES_DEEP_RESEARCH_NOTIFY_VIA_TG_DEFAULT",
        description="Default del campo notify_via_tg en CreateJobRequest.",
    )

    # Recovery
    deep_research_recovery_drop_orphan_hours: int = Field(
        default=168,
        ge=1,
        le=8760,
        validation_alias="HERMES_DEEP_RESEARCH_RECOVERY_DROP_ORPHAN_HOURS",
        description="Pending sin started_at > N horas → drop en recovery (caso 1).",
    )
    deep_research_recovery_running_stuck_hours: int = Field(
        default=2,
        ge=1,
        le=48,
        validation_alias="HERMES_DEEP_RESEARCH_RECOVERY_RUNNING_STUCK_HOURS",
        description="Running sin output > N horas → reset pending + re-enqueue (caso 2).",
    )

    # Slice 1C2: report content retrieval settings. The read path is
    # ``<data_root>/<job_id>.md`` derived from the validated 12-char
    # hex job_id. ``max_report_bytes`` bounds the read so an oversize
    # file is rejected BEFORE the route reads it into memory.
    deep_research_data_root: Path = Field(
        default=Path("data/jobs"),
        validation_alias="HERMES_DEEP_RESEARCH_DATA_ROOT",
        description=(
            "Root directory for the report-content read path. "
            "Resolved at startup; created with mkdir -p if missing. "
            "Production deployments SHOULD set this to an absolute path."
        ),
    )
    deep_research_max_report_bytes: int = Field(
        default=5_242_880,
        ge=10_240,
        le=52_428_800,
        validation_alias="HERMES_DEEP_RESEARCH_MAX_REPORT_BYTES",
        description=(
            "Max bytes for the read path of "
            "``GET /v1/jobs/{id}/report``. Default 5 MiB. "
            "Oversize files are rejected (500 report_unavailable) "
            "BEFORE the body is read into memory."
        ),
    )

    # === Sprint 19.6+ Phase 4 (OpenAI Build Week): ChatGPT 5.6 frontier tier ===
    # The OpenAI Build Week hackathon (deadline 2026-07-21, $15k first
    # prize) requires use of ChatGPT 5.6 (OpenAI's new frontier model).
    # Oroimen's positioning is local-first with explicit frontier opt-in.
    # The smart-escalation path can call ChatGPT 5.6 after the configured
    # primary and fallback tiers fail technically. Difficulty or model confidence
    # never triggers frontier routing.
    #
    # Opt-in design (Sprint 19 north star: no automatic cloud calls):
    # - `enabled=False` (default): the frontier client is NOT instantiated
    #   and the model is silently skipped if it appears in a chain. No
    #   API key is required.
    # - `enabled=True` AND `api_key=""` → ValueError at startup
    #   (validator below). This prevents "I enabled it but it doesn't
    #   work" silent failures.
    # - `enabled=True` AND `api_key` set → client is instantiated and
    #   the model is used when it appears in the chain.
    #

    # Naming convention: <DOMAIN>__<FIELD> double-underscore, matching
    # the multi-tier embeddings pattern (EMBEDDING_TIER_<NAME>__FIELD).
    # The two underscores are the pydantic-settings convention for
    # nested namespaces (avoids collision with single-underscore
    # model aliases like oroimen-agent-fast).
    llm_text_frontier_enabled: bool = Field(
        default=False,
        validation_alias="LLM_TEXT_FRONTIER__ENABLED",
        description="Master switch for the ChatGPT 5.6 frontier tier. "
        "Default False (opt-in, Sprint 19 north star: no automatic "
        "cloud calls). When False, the frontier client is not "
        "instantiated and any frontier model in a chain is silently "
        "skipped. When True, LLM_TEXT_FRONTIER__API_KEY is required.",
    )
    llm_text_frontier_model: str = Field(
        default="gpt-5.6-sol",
        validation_alias="LLM_TEXT_FRONTIER__MODEL",
        description="Model name to send to the OpenAI API. Default: 'gpt-5.6-sol'.",
    )
    llm_text_frontier_api_key: str = Field(
        default="",
        validation_alias="LLM_TEXT_FRONTIER__API_KEY",
        description="OpenAI API key for the frontier tier. Empty "
        "(default) is allowed ONLY when llm_text_frontier_enabled=False. "
        "When the frontier is enabled, the key is required (validator "
        "rejects empty at startup). Never commit this value.",
    )
    llm_text_frontier_base_url: str = Field(
        default="https://api.openai.com/v1",
        validation_alias="LLM_TEXT_FRONTIER__BASE_URL",
        description="Base URL for the OpenAI-compatible API. Default "
        "https://api.openai.com/v1 (OpenAI standard). Override for "
        "OpenAI-compatible proxies (e.g., Azure OpenAI, local llama.cpp "
        "OpenAI-compat server hosting a frontier-tier model).",
    )
    llm_text_frontier_timeout_s: float = Field(
        default=60.0,
        gt=0.0,
        le=300.0,
        validation_alias="LLM_TEXT_FRONTIER__TIMEOUT_S",
        description="HTTP timeout for frontier calls. Default 60s "
        "(frontier models can have higher latency than local models; "
        "30s is too aggressive for the chain's last-resort tier).",
    )
    llm_text_frontier_max_tokens: int = Field(
        default=8192,
        ge=100,
        le=32768,
        validation_alias="LLM_TEXT_FRONTIER__MAX_TOKENS",
        description="Max output tokens per frontier call. Default 8192 "
        "(same as llm_max_tokens; frontier is last-resort so it can "
        "afford long outputs).",
    )
    # Per-tier circuit breaker overrides (optional). Default values
    # reuse the global CIRCUIT_BREAKER_FAIL_MAX / CIRCUIT_BREAKER_RESET_TIMEOUT
    # if these are not set. The frontier tier may want more aggressive
    # trip (cheaper to fall back to local than to retry a cloud call)
    # — operator choice.
    llm_text_frontier_breaker_fail_max: int = Field(
        default=3,
        ge=1,
        le=20,
        validation_alias="LLM_TEXT_FRONTIER__BREAKER_FAIL_MAX",
        description="Circuit breaker fail_max for the frontier tier. "
        "Default 3 (more aggressive than the global 5; the frontier "
        "is last-resort, cheaper to fall back than to retry).",
    )
    llm_text_frontier_breaker_reset_timeout_s: int = Field(
        default=120,
        ge=10,
        le=3600,
        validation_alias="LLM_TEXT_FRONTIER__BREAKER_RESET_TIMEOUT_S",
        description="Circuit breaker reset timeout for the frontier "
        "tier (seconds). Default 120s (longer than the global 60s; "
        "the frontier is more likely to have transient outages than "
        "local models).",
    )

    # PR #118 (Sprint 18 hardening): Pydantic validator rejects
    # vault_done_archive_after_days <= 0 at startup. Catches env-var
    # typos before the scheduler kicks in.
    @field_validator("http_api_api_key", mode="before")
    @classmethod
    def normalize_http_api_api_key(cls, value: Any) -> Any:
        """Treat blank .env values as disabled auth in every API router."""
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value

    @field_validator("http_api_cors_origins")
    @classmethod
    def validate_http_api_cors_origins(cls, value: str) -> str:
        """Reject wildcard CORS; an empty value disables browser CORS."""
        origins = [origin.strip() for origin in value.split(",") if origin.strip()]
        if "*" in origins:
            raise ValueError(
                "HERMES_API_CORS_ORIGINS cannot contain '*'; configure explicit origins"
            )
        return ",".join(origins)

    @field_validator("vault_done_archive_after_days")
    @classmethod
    def _validate_vacuum_days(cls, v: int) -> int:
        if v < 1:
            raise ValueError(
                f"vault_done_archive_after_days must be >= 1, got {v}. "
                f"Set VAULT_DONE_ARCHIVE_AFTER_DAYS to a positive integer."
            )
        return v

    # Sprint 19.6+ Phase 4 (OpenAI Build Week): the frontier tier
    # master switch (LLM_TEXT_FRONTIER__ENABLED=True) requires a
    # non-empty API key. Without this validator, a misconfigured
    # .env (e.g., enabled=True but the user forgot to set the key)
    # would silently skip the frontier at runtime — confusing.
    # We fail fast at startup instead.
    @field_validator("llm_text_frontier_api_key")
    @classmethod
    def _validate_frontier_api_key(cls, v: str, info: Any) -> str:
        # Cross-field check: if frontier is enabled, the API key
        # must be set. info.data has all previously validated fields.
        # Using Any for info to avoid the `from __future__ import
        # annotations` ordering issue (pydantic v2 ValidationInfo
        # may not be in scope at type-check time).
        enabled = info.data.get("llm_text_frontier_enabled", False)
        if enabled and not v.strip():
            raise ValueError(
                "LLM_TEXT_FRONTIER__API_KEY must be set when "
                "LLM_TEXT_FRONTIER__ENABLED=True. Either set the key "
                "(https://platform.openai.com/api-keys) or disable the "
                "frontier (LLM_TEXT_FRONTIER__ENABLED=false). The "
                "frontier is opt-in by design (Sprint 19 north star: "
                "no automatic cloud calls)."
            )
        return v

    # Sprint 19.6+ Phase 3 (TDD v0.7 §10.4.5 MAJOR-1): local_vision_base_url
    # MUST be a localhost URL. Prevents data exfiltration if .env is
    # tampered (Sprint 19 north star: no data leaves the NAS by default).
    # For higher-trust deployments with a remote Ollama, set up a tunnel
    # or VPN and override this validator explicitly.
    @field_validator("local_vision_base_url")
    @classmethod
    def _validate_local_vision_base_url(cls, v: str) -> str:
        parsed = urlparse(v)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(
                f"local_vision_base_url must use http or https, got "
                f"{parsed.scheme!r}. URL was {v!r}."
            )
        if parsed.hostname not in _LOCAL_VISION_ALLOWED_HOSTS:
            raise ValueError(
                f"local_vision_base_url host must be localhost (or "
                f"127.0.0.1, ::1), got {parsed.hostname!r}. URL was "
                f"{v!r}. Sprint 19 north star forbids data egress; "
                f"if you need a remote Ollama, set up a tunnel or VPN "
                f"and override this validator explicitly."
            )
        return v

    @field_validator("db_path", mode="before")
    @classmethod
    def _parse_db_path(cls, v: object) -> Path:
        # Solo validamos formato. La creación del directorio se hace
        # en Database.initialize() con los permisos del usuario real.
        # Aquí era un side effect que fallaba en el container (uid 1000
        # no puede crear /app/data/).
        return Path(str(v))

    @field_validator("backup_dir", mode="before")
    @classmethod
    def _parse_backup_dir(cls, v: object) -> Path:
        return Path(str(v))

    @field_validator("vault_path", mode="before")
    @classmethod
    def _parse_vault_path(cls, v: object) -> Path:
        return Path(str(v)).expanduser()

    @model_validator(mode="after")
    def _validate_models_against_allowlist(self) -> Settings:
        """Valida los 3 slots de modelo contra self.llm_allowed_models.

        Sprint 12+: la allowlist es per-instancia (no modulo-level constant)
        para que tests puedan inyectar su propia lista via env
        (LLM_ALLOWED_MODELS). Production deja el default y queda igual
        de estricto que antes.
        """
        allowed = set(self.llm_allowed_models)
        bad = [
            (name, value)
            for name, value in (
                ("llm_text_primary", self.llm_text_primary),
                ("llm_text_fallback", self.llm_text_fallback),
                ("llm_voice_primary", self.llm_voice_primary),
            )
            if value not in allowed
        ]
        if bad:
            names = ", ".join(f"{n}={v!r}" for n, v in bad)
            raise ValueError(
                f"Modelo(s) fuera de allowlist LLM_ALLOWED_MODELS: {names}. "
                f"Modelos permitidos: {sorted(allowed)}"
            )
        return self

    @field_validator("llm_model_overrides", mode="before")
    @classmethod
    def _parse_llm_model_overrides(cls, v: object) -> dict[str, list[str]]:
        """Parsea llm_model_overrides desde env (JSON string) o deja el dict tal cual.

        Pydantic-settings lee env vars como strings, pero llm_model_overrides
        es dict[str, list[str]]. Aceptamos ambos formatos y devolvemos
        siempre un dict. Valores invalidos (JSON malformado) se ignoran
        silenciosamente para no bloquear el startup por una config rota;
        se loggean en WARNING mas abajo (en el handler de chat).
        """
        if isinstance(v, dict):
            return {str(k): list(val) for k, val in v.items() if isinstance(val, list)}
        if isinstance(v, str):
            import json as _json

            v = v.strip()
            if not v:
                return {}
            try:
                parsed = _json.loads(v)
            except _json.JSONDecodeError:
                logger.warning(
                    "llm_model_overrides_parse_failed",
                    extra={"raw_value": v[:200]},
                )
                return {}
            if not isinstance(parsed, dict):
                return {}
            return {str(k): list(val) for k, val in parsed.items() if isinstance(val, list)}
        return {}

    @field_validator("llm_model_overrides")
    @classmethod
    def _normalize_llm_model_overrides(
        cls, overrides: dict[str, list[str]]
    ) -> dict[str, list[str]]:
        """Canonicalize public aliases and reject reserved/colliding IDs."""
        normalized: dict[str, list[str]] = {}
        for alias, chain in overrides.items():
            if not alias.strip() or alias != alias.strip():
                raise ValueError("Model override aliases must be non-empty and trimmed")
            if not chain or any(not model.strip() for model in chain):
                raise ValueError(f"Model override alias {alias!r} requires a non-empty chain")
            if alias == "hermes-agent" or alias.startswith("hermes-agent-"):
                canonical = f"oroimen-agent{alias[len('hermes-agent') :]}"
            else:
                canonical = alias

            if canonical in {"oroimen-agent", "oroimen-agent-frontier"}:
                raise ValueError(f"Model override alias {alias!r} is reserved")
            existing = normalized.get(canonical)
            if existing is not None and existing != chain:
                raise ValueError(f"Conflicting model override aliases resolve to {canonical!r}")
            normalized[canonical] = chain
        return normalized

    @property
    def allowed_user_ids_list(self) -> list[int]:
        return [int(x.strip()) for x in self.allowed_user_ids.split(",") if x.strip()]

    @property
    def oroimen_default_collections(self) -> list[tuple[str, str, int]]:
        """Sprint 19 Slice 4d v2: parse OROIMEN_DEFAULT_COLLECTIONS env var.

        Returns the PARA defaults to use for first-time seeding.

        True opt-in (Sprint 19 followup per user feedback 2026-07-12):
        - Empty/unset env var = returns empty list = NO defaults are
          seeded. The user must explicitly set OROIMEN_DEFAULT_COLLECTIONS
          if they want any collections pre-created on first run.
        - Set env var = returns the parsed list. If parse fails
          (malformed JSON, missing fields, etc.), logs a warning and
          returns empty list (graceful degradation, no crash).

        The hardcoded 4 PARA defaults from PR #142 are now ONLY used
        if the user explicitly passes them via seed_para_collections
        (defaults=DEFAULT_PARA_COLLECTIONS). Default call without
        defaults = empty (true opt-in).

        Format expected (JSON):
            [{"name": "01_Proyectos", "description": "...", "sort_order": 10}, ...]

        .env file format (SysAdmin constraint from Gemini 2026-07-12):
            - WRAP THE WHOLE VARIABLE in single quotes: OROIMEN_DEFAULT_COLLECTIONS='...'
            - Use strictly double quotes INSIDE the JSON: "name":"value"
            - DO NOT use single quotes inside the JSON values (apóstrofes
              will break the shell single-quoted string). E.g. NEVER:
              '...,"description":"Área d'estudio"...' (broken).
            - SAFE: keep apostrophes OUT of values, OR use json.dumps
              programmatically (e.g. in a Python setup script).
            - Docker Compose YAML handles JSON directly without shell
              quoting; safer for values with special chars.

        Raises:
            json.JSONDecodeError: if the env var is set but malformed.
            KeyError / ValueError: if entries are missing required fields.
        """
        if not self.oroimen_default_collections_json:
            # TRUE OPT-IN: empty env var = no defaults. The user must
            # set OROIMEN_DEFAULT_COLLECTIONS explicitly if they want
            # any PARA collections seeded.
            return []
        import json

        # R1 v0.6 M2 fix: graceful parse fallback. If the JSON is
        # malformed, log a warning and return an empty list (no
        # defaults). This avoids crashing the entire process on a
        # typo in the .env file. The user sees a clear log message
        # instead of a stack trace.
        try:
            parsed = json.loads(self.oroimen_default_collections_json)
        except (json.JSONDecodeError, ValueError) as e:
            import logging

            logging.getLogger(__name__).warning(
                "oroimen_default_collections_parse_failed",
                extra={
                    "error": str(e),
                    "raw_value": self.oroimen_default_collections_json[:200],
                },
            )
            return []
        if not isinstance(parsed, list):
            return []
        result: list[tuple[str, str, int]] = []
        for entry in parsed:
            if not isinstance(entry, dict):
                continue
            try:
                result.append((entry["name"], entry["description"], int(entry["sort_order"])))
            except (KeyError, TypeError, ValueError):
                continue
        return result

    @property
    def vault_monitor_roots(self) -> list[Path]:
        """Sprint 19 Slice 4d v2 (v0.7 §1 N1 + NB1 fix).

        Parse comma-separated VAULT_MONITOR_ROOTS into list[Path].
        Pydantic v2 does NOT auto-split env vars into list[X], so this
        manual split is required. Returns empty list if env var is empty
        or missing (caller treats empty as "no monitor mode").

        Per v0.7 §4 partition rule: no monitor root may be under
        VAULT_INBOX_ROOT. Validation happens in `_get_monitor_roots()`
        which raises ValueError on overlap.
        """
        if not self.vault_monitor_roots_str:
            return []
        return [Path(p.strip()) for p in self.vault_monitor_roots_str.split(",") if p.strip()]

    def _get_monitor_roots(self) -> list[Path]:
        """Sprint 19 Slice 4d v2 (v0.5 §4 + v0.6 m4.4): resolve monitor roots.

        Precedence:
        1. VAULT_MONITOR_ROOTS (list, via property) — takes priority
        2. VAULT_DROP_ROOT (single, deprecated) — converted to single list
        3. <VAULT_INBOX_ROOT>/drop — auto-derive (preserves v0.1 behavior)

        Raises ValueError on overlap with VAULT_INBOX_ROOT (forbidden
        config that would create M6 partition conflicts).
        """
        roots: list[Path] = []
        if self.vault_monitor_roots:
            roots = list(self.vault_monitor_roots)
        elif self.vault_drop_root:
            roots = [self.vault_drop_root]
        else:
            roots = [self.vault_inbox_root / "drop"]

        # Validate: no monitor root may be under VAULT_INBOX_ROOT
        inbox_resolved = self.vault_inbox_root.resolve()
        for root in roots:
            try:
                root.resolve().relative_to(inbox_resolved)
                raise ValueError(
                    f"VAULT_MONITOR_ROOTS contains {root} which is under "
                    f"VAULT_INBOX_ROOT ({inbox_resolved}). This creates an M6 "
                    f"partition conflict: v0.2 M6 would skip the file (it's "
                    f"under the legacy inbox), v0.1 M6 processes it. Remove "
                    f"this root from VAULT_MONITOR_ROOTS."
                )
            except ValueError as e:
                if "is under" in str(e):
                    raise
                # relative_to raises ValueError if not relative — that's the OK case
                continue

        return roots

    @property
    def text_chain(self) -> list[str]:
        """Chain por defecto para chat: [primary] o [primary, fallback].

        Sprint 19.6+ Phase 5 (OpenAI Build Week): chain LOCAL-FIRST.
        Comportamiento:
        - Si el operador tiene `OPENCODE_GO_API_KEY` configurada
          (>= 10 chars, el min del validator), el chain incluye el
          fallback MiniMax (cloud) ademas del primary.
        - Si NO hay API key, el chain es SOLO el primary (Ollama
          local). NO intenta llamar al cloud provider — eso fallaría
          con 401.

        Casos:
        - Default out-of-the-box: `OPENCODE_GO_API_KEY=""` (env file
          vacio) -> chain = ["qwen2.5:7b"] (Ollama local, no API key
          required). El "5-minute setup" de un juez clona el repo +
          `docker compose up` y el chat funciona sin tocar .env.
        - Operator adds OPENCODE_GO_API_KEY: chain = ["qwen2.5:7b",
          "MiniMax-M2.7-highspeed"]. El fallback cloud se activa
          automaticamente cuando el Ollama tier falla (breaker open
          o timeout). Esto es la "smart escalation UP" del plan:
          local por default, cloud cuando hace falta.
        - Operator sets LLM_TEXT_PRIMARY_PROVIDER=minimax +
          OPENCODE_GO_API_KEY=...: chain = ["MiniMax-M3",
          "MiniMax-M2.7-highspeed"] (comportamiento cloud-only
          Sprint 19.6+ Phase 4, backward compat).

        NOTA: la `opencode_go_api_key` es opcional desde Sprint 19.6+
        Phase 5 (mirror del patron Sprint 11 ADR-004 / `telegram_bot_token`).
        En el field validator, default es None (no required, no min_length).
        En `text_chain` solo comprobamos `is not None` y `.strip()` para
        aniadir el fallback cloud SOLO si la key esta set. Sin key → chain
        se queda en el primary (Ollama local, sin cloud calls). Esto evita
        un 401 ruidoso en runtime y match el principio "no automatic cloud
        calls" del Sprint 19 north star.
        """
        chain = [self.llm_text_primary]
        # Solo aniadir fallback si la API key del cloud provider esta
        # configurada. El fallback por default es un modelo MiniMax
        # que requiere API key — sin key, el chain se queda en el
        # primary (Ollama local). Esto evita un 401 ruidoso en
        # runtime y match el principio "no automatic cloud calls"
        # del Sprint 19 north star.
        # `is not None` primero para que no explote si el campo es None
        # (caso "judge clones, no .env" del Build Week polished subset).
        if (
            self.opencode_go_api_key is not None
            and self.opencode_go_api_key.strip()
            and self.llm_text_fallback
        ):
            chain.append(self.llm_text_fallback)
        return chain

    @property
    def text_chain_full(self) -> list[str]:
        """Chain completo con frontier tier: primary → fallback → frontier.

        Sprint 19.6+ Phase 4 (OpenAI Build Week): chain de 3 niveles
        para smart escalation. Orden de intento:
          1. Primary (llm_text_primary, default MiniMax-M3)
          2. Fallback (llm_text_fallback, default MiniMax-M2.7-highspeed)
          3. Frontier (llm_text_frontier_model, default gpt-5.6-sol)
             — SOLO si llm_text_frontier_enabled=True

        El frontier es opt-in (Sprint 19 north star: no automatic
        cloud calls). Si está deshabilitado, devuelve [primary, fallback]
        (igual que text_chain, para preservar backward compat).

        This is the router's normal chain. With frontier disabled it is
        exactly `text_chain`; enabling frontier appends the configured GPT
        model. A cloud fallback is present only when its own API key is set.
        """
        chain = list(self.text_chain)
        # Si el frontier está habilitado y el modelo está configurado,
        # lo añadimos como último recurso. La validación del API key
        # ocurre en __init__ (validator arriba), así que si enabled=True
        # pero no hay API key, el Settings no se construye (fail-fast).
        if self.llm_text_frontier_enabled and self.llm_text_frontier_model.strip():
            chain.append(self.llm_text_frontier_model)
        return chain

    @property
    def voice_chain(self) -> list[str]:
        # v1.2: voice_chain unificado con text_chain. Antes (legacy
        # opencode-go) había un chain separado para voz que dependia
        # de mimo-v2.5 (unico modelo de Go con capacidad de audio).
        # El bug opencode/opencode#30389 confirmo que mimo-v2.5 no
        # procesaba input_audio via Go. Ahora la voz pasa por STT
        # externo (Gemini Flash Lite) y luego al mismo chain de texto
        # (MiniMax-M3 → MiniMax-M2.7-highspeed). El STT externo se
        # mantiene por aislamiento de cuota y coste (ver Settings
        # arriba en stt_*).
        return list(self.text_chain)

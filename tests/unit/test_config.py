"""Tests para la configuración."""

from __future__ import annotations

import json as _json
from pathlib import Path

import pytest
from pydantic import ValidationError

from hermes.config import Settings


def test_settings_loads_minimal_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token-12345")
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "test-key-67890")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key-abcdef1234567890")
    monkeypatch.setenv("ALLOWED_USER_IDS", "123,456,789")
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))

    s = Settings(_env_file=None)
    assert s.telegram_bot_token == "test-token-12345"
    assert s.allowed_user_ids == "123,456,789"
    assert s.allowed_user_ids_list == [123, 456, 789]
    # Sprint 19.6+ Phase 5 (OpenAI Build Week): default chain is now
    # LOCAL-FIRST. `llm_text_primary` defaults to `qwen2.5:7b` (Ollama
    # local) and the provider hint is "ollama" — no API key required.
    # The fallback `MiniMax-M2.7-highspeed` is only added to the
    # chain when `OPENCODE_GO_API_KEY` is set (which it IS in this
    # test, so the chain has 2 elements). Before Sprint 19.6+
    # Phase 5 the default was MiniMax-M3 (cloud) and the chain was
    # always 2 elements.
    assert s.llm_text_primary == "qwen2.5:7b"
    assert s.llm_text_primary_provider == "ollama"
    assert s.llm_text_primary_base_url == "http://localhost:11434/v1"
    assert s.llm_text_primary_api_key == "ollama"
    assert s.llm_text_fallback == "MiniMax-M2.7-highspeed"
    assert s.llm_text_fallback_provider == "minimax"
    assert s.llm_voice_primary == "MiniMax-M3"
    # text_chain includes the fallback because the test sets
    # OPENCODE_GO_API_KEY. Without the key, the chain would be just
    # the primary (Ollama only). See `Settings.text_chain`.
    assert s.text_chain == ["qwen2.5:7b", "MiniMax-M2.7-highspeed"]
    # voice_chain == text_chain (unificado desde v1.2)
    assert s.voice_chain == ["qwen2.5:7b", "MiniMax-M2.7-highspeed"]
    # STT defaults
    assert s.gemini_api_key == "gemini-key-abcdef1234567890"
    assert s.stt_model == "gemini-3.1-flash-lite"
    assert s.stt_base_url == "https://generativelanguage.googleapis.com/v1beta"
    assert s.stt_max_concurrent == 2
    assert s.stt_per_minute == 12
    assert s.stt_queue_timeout_s == 60
    assert s.db_path == tmp_path / "test.db"


def test_settings_missing_required_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sprint 11 (ADR-004) + Sprint 19.6+ Phase 5: tokens cloud son opcionales.

    Antes (Sprint 11) `telegram_bot_token` era `Field(..., min_length=10)`
    (required) y bloqueaba deployments sin Telegram. Ahora `str | None`.

    Sprint 19.6+ Phase 5 (OpenAI Build Week) extiendio el mismo patron a
    `opencode_go_api_key` y `gemini_api_key`. Antes ambos eran `Field(...,
    min_length=10)` (required) y bloqueaban el flow "judge clones repo,
    `docker compose up`, chat works" porque hermes fallaba al cargar
    Settings sin .env o con .env.example (que los tiene vacios).

    Ahora los 3 tokens son `str | None = Field(default=None, ...)`. El
    runtime decide si la feature esta activa (sin token → sin fallback
    cloud, sin STT, sin Telegram). El field validator no rompe el startup.
    """
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("OPENCODE_GO_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    s = Settings(_env_file=None)
    assert s.telegram_bot_token is None
    assert s.opencode_go_api_key is None
    assert s.gemini_api_key is None
    # Sin cloud key, el chain se queda en el primary (Ollama local) —
    # NO automatic cloud calls, match del Sprint 19 north star.
    assert s.text_chain == ["qwen2.5:7b"]


def test_settings_empty_env_file_accepted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Sprint 19.6+ Phase 5: .env vacio (caso `cp .env.example .env` sin
    editar) NO debe romper el startup de Settings.

    Antes el field validator con `min_length=10` rechazaba string vacio
    con `String should have at least 10 characters`. Ahora el campo es
    `str | None` con default None, asi que empty string se acepta
    (Pydantic v2 acepta `""` como string valido a menos que el field
    use `min_length` o `constr(...)`).

    Esto es el flow REAL de "judge clones, cp .env.example .env,
    docker compose up" del Build Week polished subset: el .env existe
    (porque el cp lo creo) pero los tokens estan vacios (porque el
    juez no los relleno). El chain arranca con [primary] = [Ollama local].
    """
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OPENCODE_GO_API_KEY=\nGEMINI_API_KEY=\nTELEGRAM_BOT_TOKEN=\n",
        encoding="utf-8",
    )
    # Belt-and-suspenders: ensure env vars are NOT set in the test env
    # (they would override the .env file values).
    monkeypatch.delenv("OPENCODE_GO_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    s = Settings(_env_file=str(env_file))
    # Empty string is NOT None — Pydantic accepts "" as a valid string
    # for an Optional[str] field. The runtime checks (`is not None` and
    # `.strip()`) then treat "" as "no key configured" → chain = [primary].
    assert s.opencode_go_api_key == ""
    assert s.gemini_api_key == ""
    assert s.telegram_bot_token == ""
    assert s.text_chain == ["qwen2.5:7b"]


@pytest.mark.asyncio
async def test_db_path_creates_parent_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Database.initialize() crea el directorio padre. El validador Settings NO.

    Antes el validador tenia un side effect de mkdir, pero fallaba en
    el container (uid 1000 no puede crear /app/data/). Ahora la creacion
    del dir se hace en Database.initialize() con los permisos correctos.
    """
    from hermes.memory.db import Database

    nested = tmp_path / "a" / "b" / "c.db"
    assert not nested.parent.exists()
    d = Database(nested)
    await d.initialize()
    assert nested.parent.exists()
    assert nested.parent.is_dir()


def test_allowed_user_ids_csv_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token-12345")
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "test-key-67890")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key-abcdef1234567890")
    monkeypatch.delenv("ALLOWED_USER_IDS", raising=False)
    s = Settings(_env_file=None)
    assert s.allowed_user_ids == ""
    assert s.allowed_user_ids_list == []


def test_model_in_allowlist_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    # Sprint 12+: usar override de allowlist para admitir modelos
    # no-MiniMax. Verifica que el knob LLM_ALLOWED_MODELS funciona.
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token-12345")
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "test-key-67890")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key-abcdef1234567890")
    monkeypatch.setenv(
        "LLM_ALLOWED_MODELS", '["kimi-k2.6","qwen3.7-plus","MiniMax-M3","MiniMax-M2.7-highspeed"]'
    )
    monkeypatch.setenv("LLM_TEXT_PRIMARY", "kimi-k2.6")
    monkeypatch.setenv("LLM_TEXT_FALLBACK", "qwen3.7-plus")
    s = Settings(_env_file=None)
    assert s.llm_text_primary == "kimi-k2.6"
    assert s.llm_text_fallback == "qwen3.7-plus"
    # La allowlist overrideada se aplica al campo Settings:
    assert "kimi-k2.6" in s.llm_allowed_models


def test_model_not_in_allowlist_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Privacidad: rechaza modelos que no estén en la allowlist oficial de opencode-go.

    Esto previene usar modelos de /v1/models que podrían entrenar con datos del usuario
    o tener términos diferentes a Go.
    """
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token-12345")
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "test-key-67890")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key-abcdef1234567890")
    monkeypatch.setenv("LLM_TEXT_PRIMARY", "mimo-v2-omni")
    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None)
    assert "allowlist" in str(exc_info.value).lower()


def test_fallback_model_not_in_allowlist_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token-12345")
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "test-key-67890")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key-abcdef1234567890")
    monkeypatch.setenv("LLM_TEXT_FALLBACK", "hy3-preview")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


# ---------------------------------------------------------------------------
# Tests S2.1: GEMINI_API_KEY + STT_* (v1.2)
# ---------------------------------------------------------------------------


def test_gemini_api_key_optional_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sprint 19.6+ Phase 5: GEMINI_API_KEY sin set NO rompe startup.

    Antes (Sprint 11-19) `gemini_api_key` era `Field(..., min_length=10)`
    (required) y bloqueaba el startup del Settings sin GEMINI_API_KEY en
    el env. Ahora es `str | None = Field(default=None, ...)` (mirror
    del patron Sprint 11 ADR-004 / `telegram_bot_token`).

    El servicio de STT sigue requiriendo una key valida para transcribir
    audio — la guard `_fn` en `hermes/handlers/messages.py` no se
    construye si el caller no la pidio. Si la piden, `_transcribe()`
    en `hermes/stt/gemini.py:88-89` valida `if not api_key` y lanza
    STTError con mensaje claro. Sin key → STT feature inactiva, no crash.
    """
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token-12345")
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "test-key-67890")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    s = Settings(_env_file=None)
    assert s.gemini_api_key is None


def test_gemini_api_key_accepts_short_or_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sprint 19.6+ Phase 5: ya no hay `min_length=10` defensivo.

    Antes `Field(..., min_length=10)` rechazaba cualquier key <10 chars
    (defensivo contra typos). Ahora el field es `str | None` sin
    min_length — la key se acepta tal cual, y el servicio de STT
    rechaza con STTError("api_key is required and cannot be empty")
    si llega a transcribir con key vacia.

    Esto es importante para el flow "judge copia .env.example sin
    editar" del Build Week polished subset: la key puede ser "" en
    .env, Settings acepta, y el startup no rompe. Si el juez intenta
    transcribir audio, recibe un error claro (no un crash de startup).
    """
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token-12345")
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "test-key-67890")
    # Empty string
    monkeypatch.setenv("GEMINI_API_KEY", "")
    s = Settings(_env_file=None)
    assert s.gemini_api_key == ""
    # Short string (5 chars) — antes fallaba, ahora OK
    monkeypatch.setenv("GEMINI_API_KEY", "short")
    s = Settings(_env_file=None)
    assert s.gemini_api_key == "short"


def test_stt_model_can_be_overridden(monkeypatch: pytest.MonkeyPatch) -> None:
    """STT_MODEL env var sobrescribe el default."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token-12345")
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "test-key-67890")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key-abcdef1234567890")
    monkeypatch.setenv("STT_MODEL", "gemini-2.0-flash-exp")
    s = Settings(_env_file=None)
    assert s.stt_model == "gemini-2.0-flash-exp"


def test_stt_max_concurrent_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    """STT_MAX_CONCURRENT fuera de [1, 10] → ValidationError."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token-12345")
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "test-key-67890")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key-abcdef1234567890")
    # 0 es inválido
    monkeypatch.setenv("STT_MAX_CONCURRENT", "0")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)
    # 11 es inválido
    monkeypatch.setenv("STT_MAX_CONCURRENT", "11")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_stt_per_minute_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    """STT_PER_MINUTE fuera de [1, 60] → ValidationError."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token-12345")
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "test-key-67890")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key-abcdef1234567890")
    monkeypatch.setenv("STT_PER_MINUTE", "0")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)
    monkeypatch.setenv("STT_PER_MINUTE", "61")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_stt_queue_timeout_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    """STT_QUEUE_TIMEOUT_S fuera de [5, 300] → ValidationError."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token-12345")
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "test-key-67890")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key-abcdef1234567890")
    monkeypatch.setenv("STT_QUEUE_TIMEOUT_S", "4")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)
    monkeypatch.setenv("STT_QUEUE_TIMEOUT_S", "301")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_voice_chain_unified_with_text_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    """voice_chain == text_chain (unificado en v1.2).

    Antes había chains separados (texto vs voz). Tras el bug #30389
    que confirmó que mimo-v2.5 no procesa audio, voz y texto usan
    el mismo chain (minimax-m3 → deepseek-v4-flash).
    """
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token-12345")
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "test-key-67890")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key-abcdef1234567890")
    # Forzar chains: Sprint 12+ MiniMax defaults
    monkeypatch.setenv("LLM_TEXT_PRIMARY", "MiniMax-M3")
    monkeypatch.setenv("LLM_TEXT_FALLBACK", "MiniMax-M2.7-highspeed")
    s = Settings(_env_file=None)
    assert s.voice_chain == s.text_chain
    # Voz usa el mismo orden que texto
    assert s.voice_chain == ["MiniMax-M3", "MiniMax-M2.7-highspeed"]


def test_voice_primary_model_allowlist_validated(monkeypatch: pytest.MonkeyPatch) -> None:
    """LLM_VOICE_PRIMARY también pasa por la allowlist.

    El validador de allowlist sigue activo para evitar typos o modelos
    no soportados. Override LLM_ALLOWED_MODELS permite inyectar modelos
    custom (tests, modelos nuevos antes de añadirlos a la lista oficial).
    """
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token-12345")
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "test-key-67890")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key-abcdef1234567890")
    # mimo-v2-omni NO está en la allowlist oficial (de los 13)
    monkeypatch.setenv("LLM_VOICE_PRIMARY", "mimo-v2-omni")
    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None)
    assert "allowlist" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# Tests S11 (ADR-004): WebUI-primary + Telegram opt-in legacy
# ---------------------------------------------------------------------------


def test_enable_telegram_default_true(monkeypatch: pytest.MonkeyPatch) -> None:
    """S11.0: enable_telegram default = True (no romper deployments existentes).

    Sprint 12+ migrara a default False. Doc en TDD §9.3 DEC-S12-2.
    """
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token-12345")
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "test-key-67890")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key-abcdef1234567890")
    s = Settings(_env_file=None)
    assert s.enable_telegram is True


def test_enable_telegram_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """enable_telegram=False explicit deshabilita PollingReceiver."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token-12345")
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "test-key-67890")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key-abcdef1234567890")
    monkeypatch.setenv("ENABLE_TELEGRAM", "false")
    s = Settings(_env_file=None)
    assert s.enable_telegram is False


def test_telegram_bot_token_optional_when_features_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S11: telegram_bot_token ahora es `str | None` (default None).

    Antes era `str` required (min_length=10). Eso bloqueaba cualquier
    deployment sin Telegram. Ahora es opcional: si enable_telegram=False
    O el token no es necesario, hermes arranca con warning pero sin crash.
    """
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "test-key-67890")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key-abcdef1234567890")
    s = Settings(_env_file=None)
    assert s.telegram_bot_token is None


def test_telegram_bot_token_kept_when_provided(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S11: token sigue funcionando si esta set (retrocompat)."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token-12345")
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "test-key-67890")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key-abcdef1234567890")
    s = Settings(_env_file=None)
    assert s.telegram_bot_token == "test-token-12345"


def test_enable_http_api_default_true_after_s11(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S11: enable_http_api default cambio de False a True (WebUI primary).

    Antes default False (opt-in). Ahora default True (primary interface
    para open-webui + futura app Android nativa S11.1+). Si el user quiere
    legacy-only Telegram, pone ENABLE_HTTP_API=false.
    """
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token-12345")
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "test-key-67890")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key-abcdef1234567890")
    s = Settings(_env_file=None)
    assert s.enable_http_api is True


# Sprint 12 (ADR-007): llm_model_overrides parsing and behaviour.


def test_llm_model_overrides_default_empty(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Default sin env var: dict vacio (no aliases)."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "test-key-min-10")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini")
    monkeypatch.setenv("ALLOWED_USER_IDS", "1")
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("HERMES_LLM_MODEL_OVERRIDES", "")
    s = Settings(_env_file=None)
    assert s.llm_model_overrides == {}


def test_llm_model_overrides_parsed_from_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Sprint 12: env var con JSON valido se parsea a dict[str, list[str]]."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "test-key-min-10")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini")
    monkeypatch.setenv("ALLOWED_USER_IDS", "1")
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv(
        "HERMES_LLM_MODEL_OVERRIDES",
        _json.dumps({"oroimen-agent-fast": ["deepseek-v4-flash"]}),
    )
    s = Settings(_env_file=None)
    assert s.llm_model_overrides == {"oroimen-agent-fast": ["deepseek-v4-flash"]}


def test_llm_model_overrides_invalid_json_falls_back_to_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """JSON malformado no debe romper el startup. Loggea WARNING y devuelve dict vacio."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "test-key-min-10")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini")
    monkeypatch.setenv("ALLOWED_USER_IDS", "1")
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("HERMES_LLM_MODEL_OVERRIDES", "{not valid json")
    s = Settings(_env_file=None)
    assert s.llm_model_overrides == {}


# ---------------------------------------------------------------------------
# Sprint 16.8.2: llm_max_tokens default + override + validation
# ---------------------------------------------------------------------------
def test_llm_max_tokens_default_is_8192(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Sin env override, default = 8192 (matching Anthropic / Gemini defaults).

    Pre-Sprint 16 era 1024, que es muy bajo para memory facts + RAG.
    El test fija el default para que un cambio accidental futuro lo detecte.
    """
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "test-key-min-10")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini")
    monkeypatch.setenv("ALLOWED_USER_IDS", "1")
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    s = Settings(_env_file=None)
    assert s.llm_max_tokens == 8192


def test_llm_max_tokens_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """HERMES_LLM_MAX_TOKENS override funciona."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "test-key-min-10")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini")
    monkeypatch.setenv("ALLOWED_USER_IDS", "1")
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("HERMES_LLM_MAX_TOKENS", "4096")
    s = Settings(_env_file=None)
    assert s.llm_max_tokens == 4096


def test_llm_max_tokens_validation_rejects_out_of_range(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Si env var está fuera de rango (ge=100, le=32768), Pydantic debe rechazar."""
    from pydantic import ValidationError

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "test-key-min-10")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini")
    monkeypatch.setenv("ALLOWED_USER_IDS", "1")
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    # Muy chico: < 100
    monkeypatch.setenv("HERMES_LLM_MAX_TOKENS", "50")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)
    # Muy grande: > 32768
    monkeypatch.setenv("HERMES_LLM_MAX_TOKENS", "50000")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_http_api_cors_rejects_wildcard(monkeypatch: pytest.MonkeyPatch) -> None:
    """A wildcard must never reopen browser access to the local API."""
    monkeypatch.setenv("HERMES_API_CORS_ORIGINS", "*")
    with pytest.raises(ValidationError, match="cannot contain"):
        Settings(_env_file=None)


def test_blank_http_api_key_normalizes_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """A blank .env key means no auth for middleware and dependencies alike."""
    monkeypatch.setenv("HERMES_API_API_KEY", "   ")
    settings = Settings(_env_file=None)
    assert settings.http_api_api_key is None


def test_llm_model_overrides_normalizes_legacy_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "HERMES_LLM_MODEL_OVERRIDES",
        _json.dumps({"hermes-agent-fast": ["deepseek-v4-flash"]}),
    )
    settings = Settings(_env_file=None)
    assert settings.llm_model_overrides == {"oroimen-agent-fast": ["deepseek-v4-flash"]}


def test_llm_model_overrides_rejects_conflicting_spellings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "HERMES_LLM_MODEL_OVERRIDES",
        _json.dumps(
            {
                "hermes-agent-fast": ["legacy-chain"],
                "oroimen-agent-fast": ["public-chain"],
            }
        ),
    )
    with pytest.raises(ValidationError, match="Conflicting model override aliases"):
        Settings(_env_file=None)


@pytest.mark.parametrize("alias", ["hermes-agent-frontier", "oroimen-agent-frontier"])
def test_llm_model_overrides_rejects_reserved_frontier(
    monkeypatch: pytest.MonkeyPatch,
    alias: str,
) -> None:
    monkeypatch.setenv(
        "HERMES_LLM_MODEL_OVERRIDES",
        _json.dumps({alias: ["some-provider-model"]}),
    )
    with pytest.raises(ValidationError, match="is reserved"):
        Settings(_env_file=None)


@pytest.mark.parametrize("chain", [[], [" "]])
def test_llm_model_overrides_rejects_empty_chain(
    monkeypatch: pytest.MonkeyPatch,
    chain: list[str],
) -> None:
    monkeypatch.setenv(
        "HERMES_LLM_MODEL_OVERRIDES",
        _json.dumps({"oroimen-agent-fast": chain}),
    )
    with pytest.raises(ValidationError, match="requires a non-empty chain"):
        Settings(_env_file=None)

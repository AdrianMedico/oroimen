"""Tests de las tools builtin (Sprint 4 T2-T5).

Cubre:
- get_current_time: ISO format, zona horaria configurable
- get_weather: 2 calls HTTP (geocoding + forecast), error handling
- search_vault: matches, contexto, case-insensitive, max_results
- get_system_status: uptime, version, db health, breakers
- register_builtin_tools: registra las 4 tools con schemas correctos
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from hermes.llm.router import LLMRouter
from hermes.memory.db import Database
from hermes.tools.builtin import (
    GET_CURRENT_TIME_SCHEMA,
    GET_WEATHER_SCHEMA,
    SEARCH_VAULT_SCHEMA,
    get_current_time,
    get_system_status,
    get_weather,
    register_builtin_tools,
    search_vault,
)
from hermes.tools.registry import ToolRegistry

# ===========================================================================
# T2: get_current_time
# ===========================================================================


class TestGetCurrentTime:
    @pytest.mark.asyncio
    async def test_returns_iso_string(self, settings: object) -> None:
        result = await get_current_time(settings)
        # ISO 8601 format: "2026-06-22T14:30:00+02:00"
        assert isinstance(result, str)
        parsed = datetime.fromisoformat(result)
        assert parsed.tzinfo is not None

    @pytest.mark.asyncio
    async def test_uses_settings_timezone(self, settings: object) -> None:
        """Si settings.tz = 'Europe/Madrid', el offset debe ser +02:00 (CEST)
        o +01:00 (CET) dependiendo de la fecha."""
        result = await get_current_time(settings)
        # Madrid está en UTC+1 (invierno) o UTC+2 (verano)
        assert result.endswith("+01:00") or result.endswith("+02:00")

    @pytest.mark.asyncio
    async def test_handles_utc(self) -> None:
        """Si tz=UTC, el offset debe ser +00:00."""

        class _Settings:
            tz = "UTC"

        result = await get_current_time(_Settings())
        assert result.endswith("+00:00")

    def test_schema_is_valid(self) -> None:
        assert GET_CURRENT_TIME_SCHEMA["type"] == "object"
        assert "properties" in GET_CURRENT_TIME_SCHEMA
        # Workaround: properties no vacías (minimax-m3 rechaza properties={}).
        assert len(GET_CURRENT_TIME_SCHEMA["properties"]) >= 1
        assert GET_CURRENT_TIME_SCHEMA["required"] == []
        assert GET_CURRENT_TIME_SCHEMA.get("additionalProperties") is False


# ===========================================================================
# T3: get_weather
# ===========================================================================


class TestGetWeather:
    @pytest.mark.asyncio
    async def test_uses_geocoding_api(self, respx_mock: object) -> None:
        """Sprint 9.4: get_weather v2 con forecast + AQI + UV + cache."""
        respx_mock.get("https://geocoding-api.open-meteo.com/v1/search").mock(
            return_value=httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "name": "Madrid",
                            "latitude": 40.4168,
                            "longitude": -3.7038,
                            "country": "España",
                        }
                    ]
                },
            )
        )
        respx_mock.get("https://api.open-meteo.com/v1/forecast").mock(
            return_value=httpx.Response(
                200,
                json={
                    "current": {
                        "temperature_2m": 22.3,
                        "apparent_temperature": 21.5,
                        "relative_humidity_2m": 55,
                        "wind_speed_10m": 10,
                        "wind_gusts_10m": 15,
                        "wind_direction_10m": 180,
                        "weather_code": 1,
                        "uv_index": 5.0,
                        "precipitation": 0,
                    },
                    "daily": {
                        "time": ["2026-06-28", "2026-06-29"],
                        "temperature_2m_max": [25, 27],
                        "temperature_2m_min": [15, 16],
                        "precipitation_sum": [0, 2],
                        "precipitation_probability_max": [10, 60],
                        "uv_index_max": [7, 8],
                        "weather_code": [1, 3],
                    },
                },
            )
        )
        respx_mock.get("https://air-quality-api.open-meteo.com/v1/air-quality").mock(
            return_value=httpx.Response(
                200,
                json={
                    "current": {
                        "european_aqi": 35,
                        "pm2_5": 8.5,
                        "pm10": 12.3,
                        "ozone": 45,
                        "nitrogen_dioxide": 10,
                    }
                },
            )
        )

        result = await get_weather("Madrid")
        # Cabecera con ciudad + pais
        assert "Madrid" in result
        assert "España" in result
        # Actual
        assert "22.3°C" in result
        assert "21.5°C" in result  # feels_like
        assert "55%" in result  # humedad
        assert "UV 5.0" in result
        # Pronostico 7 dias
        assert "Pronostico 7 dias" in result
        assert "2026-06-28" in result
        assert "2026-06-29" in result
        # Calidad del aire
        assert "Calidad del aire" in result
        assert "PM2.5" in result
        assert "aceptable" in result  # AQI 35

    @pytest.mark.asyncio
    async def test_unknown_city_raises(self, respx_mock: object) -> None:
        """Ciudad inexistente → ValueError con mensaje claro."""
        respx_mock.get("https://geocoding-api.open-meteo.com/v1/search").mock(
            return_value=httpx.Response(200, json={"results": []})
        )

        with pytest.raises(ValueError, match="Ciudad no encontrada"):
            await get_weather("Atlantis")

    @pytest.mark.asyncio
    async def test_timeout_10s(self, respx_mock: object) -> None:
        """Si la API no responde, lanza httpx error (timeout configurado)."""
        import hermes.tools.builtin as bm

        bm._WEATHER_CACHE.clear()  # Reset cache (test_cache lo puebla)
        respx_mock.get("https://geocoding-api.open-meteo.com/v1/search").mock(
            side_effect=httpx.TimeoutException("timeout")
        )

        with pytest.raises(httpx.TimeoutException):
            await get_weather("Madrid")

    def test_schema_requires_city(self) -> None:
        assert "city" in GET_WEATHER_SCHEMA["required"]
        assert GET_WEATHER_SCHEMA["properties"]["city"]["type"] == "string"

    @pytest.mark.asyncio
    async def test_cache_avoids_repeated_api_calls(self, respx_mock: object) -> None:
        """Sprint 9.4: cache 15 min evita spamear APIs. 2 calls seguidos
        a la misma city solo hacen 1 set de requests."""
        import hermes.tools.builtin as bm

        bm._WEATHER_CACHE.clear()  # Reset cache

        geocoding_route = respx_mock.get("https://geocoding-api.open-meteo.com/v1/search").mock(
            return_value=httpx.Response(
                200,
                json={
                    "results": [
                        {"name": "Madrid", "latitude": 40.4, "longitude": -3.7, "country": "España"}
                    ]
                },
            )
        )
        forecast_route = respx_mock.get("https://api.open-meteo.com/v1/forecast").mock(
            return_value=httpx.Response(
                200,
                json={
                    "current": {"temperature_2m": 20, "weather_code": 0},
                    "daily": {
                        "time": ["2026-06-28"],
                        "temperature_2m_max": [25],
                        "temperature_2m_min": [15],
                        "precipitation_sum": [0],
                        "precipitation_probability_max": [0],
                        "uv_index_max": [5],
                        "weather_code": [0],
                    },
                },
            )
        )
        aq_route = respx_mock.get("https://air-quality-api.open-meteo.com/v1/air-quality").mock(
            return_value=httpx.Response(
                200,
                json={
                    "current": {
                        "european_aqi": 10,
                        "pm2_5": 5,
                        "pm10": 8,
                        "ozone": 30,
                        "nitrogen_dioxide": 5,
                    }
                },
            )
        )

        # 1ª llamada (3 requests HTTP)
        r1 = await get_weather("Madrid")
        # 2ª llamada (cache hit, 0 requests HTTP)
        r2 = await get_weather("Madrid")
        assert r1 == r2
        # Solo se hicieron los requests de la 1ª llamada
        assert geocoding_route.call_count == 1
        assert forecast_route.call_count == 1
        assert aq_route.call_count == 1

    @pytest.mark.asyncio
    async def test_air_quality_failure_is_graceful(self, respx_mock: object) -> None:
        """Sprint 9.4: si air-quality-api falla, get_weather sigue
        funcionando con forecast + UV (degradacion elegante)."""
        import hermes.tools.builtin as bm

        bm._WEATHER_CACHE.clear()

        respx_mock.get("https://geocoding-api.open-meteo.com/v1/search").mock(
            return_value=httpx.Response(
                200,
                json={
                    "results": [
                        {"name": "Madrid", "latitude": 40.4, "longitude": -3.7, "country": "España"}
                    ]
                },
            )
        )
        respx_mock.get("https://api.open-meteo.com/v1/forecast").mock(
            return_value=httpx.Response(
                200,
                json={
                    "current": {"temperature_2m": 20, "weather_code": 0, "uv_index": 3},
                    "daily": {
                        "time": ["2026-06-28"],
                        "temperature_2m_max": [25],
                        "temperature_2m_min": [15],
                        "precipitation_sum": [0],
                        "precipitation_probability_max": [0],
                        "uv_index_max": [5],
                        "weather_code": [0],
                    },
                },
            )
        )
        # air-quality returns 500
        respx_mock.get("https://air-quality-api.open-meteo.com/v1/air-quality").mock(
            return_value=httpx.Response(500, json={"error": "internal"})
        )

        result = await get_weather("Madrid")
        # Funciona aunque AQI falle
        assert "Madrid" in result
        assert "UV" in result
        # Pero NO incluye calidad del aire
        assert "Calidad del aire" not in result
        assert "PM2.5" not in result


# ===========================================================================
# T4: search_vault
# ===========================================================================


class TestSearchVault:
    @pytest.mark.asyncio
    async def test_finds_match(self, tmp_path: Path) -> None:
        """Busca 'Hermes' en vault con 3 notas → 3 matches."""
        (tmp_path / "note1.md").write_text("Hermes es el bot\nOtra línea", encoding="utf-8")
        (tmp_path / "note2.md").write_text("Texto sin match", encoding="utf-8")
        (tmp_path / "note3.md").write_text("Menciona Hermes otra vez", encoding="utf-8")

        result = await search_vault("Hermes", vault_path=tmp_path)
        assert "note1.md" in result
        assert "note2.md" not in result
        assert "note3.md" in result

    @pytest.mark.asyncio
    async def test_no_match_returns_friendly(self, tmp_path: Path) -> None:
        """Sin matches → mensaje user-friendly (no error)."""
        (tmp_path / "empty.md").write_text("Sin match", encoding="utf-8")
        result = await search_vault("xyz", vault_path=tmp_path)
        assert "No se encontraron" in result
        assert "xyz" in result

    @pytest.mark.asyncio
    async def test_case_insensitive(self, tmp_path: Path) -> None:
        """'hermes' matchea 'Hermes'."""
        (tmp_path / "note.md").write_text("Hermes es el bot", encoding="utf-8")
        result = await search_vault("hermes", vault_path=tmp_path)
        assert "note.md" in result

    @pytest.mark.asyncio
    async def test_includes_context(self, tmp_path: Path) -> None:
        """Cada match tiene líneas de contexto antes/después."""
        content = "\n".join(f"linea {i}" for i in range(1, 11))  # 10 líneas
        content = content.replace("linea 5", "linea 5 MATCH")
        (tmp_path / "note.md").write_text(content, encoding="utf-8")

        result = await search_vault("MATCH", vault_path=tmp_path)
        # El contexto debe incluir líneas antes y después
        assert "linea 3" in result  # 2 líneas antes
        assert "linea 4" in result
        assert "linea 6" in result
        assert "linea 7" in result

    @pytest.mark.asyncio
    async def test_respects_max_results(self, tmp_path: Path) -> None:
        """Si hay >max_results matches, trunca."""
        for i in range(10):
            (tmp_path / f"note{i}.md").write_text(f"match {i}", encoding="utf-8")

        result = await search_vault("match", vault_path=tmp_path, max_results=3)
        # Solo 3 archivos
        assert "note0.md" in result
        assert "note2.md" in result
        assert "note3.md" not in result

    @pytest.mark.asyncio
    async def test_nonexistent_vault_raises(self, tmp_path: Path) -> None:
        """Si vault_path no existe → FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Vault no encontrado"):
            await search_vault("query", vault_path=tmp_path / "nonexistent")

    @pytest.mark.asyncio
    async def test_real_vault_structure_with_subdirs(self, tmp_path: Path) -> None:
        """Simula la estructura real del vault (carpetas + subcarpetas + .md en raiz).

        El vault de Obsidian real tiene estructura jerárquica: 00_Inbox/,
        10_Studies/, 20_Projects/, etc. search_vault debe
        recorrer recursivamente (rglob) y encontrar matches en cualquier
        profundidad, ignorando archivos no-.md.
        """
        # Crear estructura similar al vault real
        (tmp_path / "00_Inbox").mkdir()
        (tmp_path / "00_Inbox" / "tarea.md").write_text(
            "Recordar: estudiar Hermes", encoding="utf-8"
        )
        (tmp_path / "10_Studies").mkdir()
        (tmp_path / "10_Studies" / "apuntes.md").write_text(
            "El parcial cubre Hermes y agentes", encoding="utf-8"
        )
        (tmp_path / "20_Projects").mkdir()
        (tmp_path / "20_Projects" / "cv.md").write_text("Experiencia con Hermes", encoding="utf-8")
        # Archivo no-.md (debe ignorarse)
        (tmp_path / "30_Tech_Lab").mkdir()
        (tmp_path / "30_Tech_Lab" / "config.yaml").write_text("name: hermes", encoding="utf-8")
        # .md en raiz
        (tmp_path / "README.md").write_text("Bienvenido al vault", encoding="utf-8")

        result = await search_vault("Hermes", vault_path=tmp_path)
        # 3 matches en .md de subdirs + 0 en config.yaml (ignorado)
        assert "00_Inbox" in result
        assert "10_Studies" in result
        assert "20_Projects" in result
        # El archivo yaml NO debe aparecer
        assert "config.yaml" not in result
        # El README tampoco (porque no contiene "Hermes")
        assert "README.md" not in result

    @pytest.mark.asyncio
    async def test_bind_mount_uses_settings_vault_path(
        self, settings: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """End-to-end: register_builtin_tools usa settings.vault_path.

        Replica el flujo real: settings.vault_path se configura via env
        (VAULT_PATH=/vault), el bind mount expone el vault en el
        container, y register_builtin_tools registra search_vault con
        esa ruta. Si el vault está accesible, search_vault funciona.
        """
        from hermes.tools.builtin import register_builtin_tools

        # Crear vault simulado en subdirectorio (como /vault en el container)
        vault = tmp_path / "vault"
        vault.mkdir()
        (vault / "test.md").write_text("Hola Hermes", encoding="utf-8")

        # Sobreescribir VAULT_PATH con el vault simulado
        monkeypatch.setenv("VAULT_PATH", str(vault))
        # Forzar recarga de settings con el nuevo VAULT_PATH
        from hermes.config import Settings

        s = Settings(_env_file=None)
        assert s.vault_path == vault

        # Registrar tools con esos settings (signature keyword-only)
        registry = ToolRegistry()
        register_builtin_tools(registry, settings=s, db=None, router=None, start_time=0.0)  # type: ignore[arg-type]

        # search_vault debe estar registrado y usar la ruta correcta
        result = await registry.execute("search_vault", {"query": "Hermes"})
        assert "test.md" in result

    def test_schema_requires_query(self) -> None:
        assert "query" in SEARCH_VAULT_SCHEMA["required"]
        assert SEARCH_VAULT_SCHEMA["properties"]["query"]["type"] == "string"


# ===========================================================================
# T5: get_system_status
# ===========================================================================


class TestGetSystemStatus:
    @pytest.mark.asyncio
    async def test_returns_multiline_status(
        self, settings: object, db: Database, settings_v12: object
    ) -> None:
        """Devuelve texto multi-línea con uptime, version, db, breakers."""
        router = LLMRouter(settings)
        try:
            result = await get_system_status(
                settings=settings_v12,
                db=db,
                router=router,
                start_time=100.0,  # 100s antes de now() → uptime ~0m
            )
            assert "Oroimen" in result
            assert "v0.4.0" in result or "version" in result.lower()
            assert "Uptime" in result
            assert "Database" in result
            assert "LLM chain" in result
            # 2 modelos en el chain (v1.2 unificado)
            assert "minimax" in result.lower() or "deepseek" in result.lower()
        finally:
            await router.aclose()

    @pytest.mark.asyncio
    async def test_reports_db_degraded(self, settings: object, settings_v12: object) -> None:
        """Si la DB falla, status = DEGRADED."""

        class _BrokenDB:
            async def ping(self) -> None:
                raise ConnectionError("DB down")

        router = LLMRouter(settings)
        try:
            result = await get_system_status(
                settings=settings_v12,
                db=_BrokenDB(),  # type: ignore[arg-type]
                router=router,
                start_time=100.0,
            )
            assert "DEGRADED" in result
        finally:
            await router.aclose()


# ===========================================================================
# register_builtin_tools
# ===========================================================================


class TestRegisterBuiltinTools:
    def test_registers_local_tools_by_default(self, settings: object, db: Database) -> None:
        """Default registry contains only local, non-egress tools."""
        registry = ToolRegistry()
        router = LLMRouter(settings)
        try:
            register_builtin_tools(
                registry,
                settings=settings,
                db=db,
                router=router,
                start_time=100.0,
            )
            tools = registry.list_tools()
            assert "get_current_time" in tools
            assert "search_vault" in tools
            assert "get_system_status" in tools
            assert "get_weather" not in tools
            assert "agent_reach_read" not in tools
            assert len(tools) == 8  # 3 local builtin + 5 collection tools
        finally:
            pass  # router se cierra al final del test

    def test_outbound_tools_require_explicit_opt_in(self, settings: object, db: Database) -> None:
        """Weather and Agent-Reach are registered only after explicit opt-in."""
        settings.outbound_tools_enabled = True  # type: ignore[attr-defined]
        registry = ToolRegistry()
        router = LLMRouter(settings)
        register_builtin_tools(
            registry,
            settings=settings,
            db=db,
            router=router,
            start_time=100.0,
        )
        tools = registry.list_tools()
        assert "get_weather" in tools
        assert "agent_reach_read" in tools
        assert len(tools) == 13

    def test_schemas_are_openai_format(self, settings: object, db: Database) -> None:
        """tool_schemas() devuelve formato OpenAI (type: function)."""
        registry = ToolRegistry()
        router = LLMRouter(settings)
        try:
            register_builtin_tools(
                registry,
                settings=settings,
                db=db,
                router=router,
                start_time=100.0,
            )
            schemas = registry.tool_schemas()
            assert len(schemas) == 8  # same count as local tools
            for schema in schemas:
                assert schema["type"] == "function"
                assert "name" in schema["function"]
                assert "description" in schema["function"]
                assert "parameters" in schema["function"]
        finally:
            pass

    def test_schemas_have_descriptions(self, settings: object, db: Database) -> None:
        """Cada tool tiene description no vacía (para que el LLM sepa cuándo usarla)."""
        registry = ToolRegistry()
        router = LLMRouter(settings)
        try:
            register_builtin_tools(
                registry,
                settings=settings,
                db=db,
                router=router,
                start_time=100.0,
            )
            for spec in registry.list_specs():
                assert spec.description, f"Tool {spec.name} sin description"
                assert len(spec.description) > 10
        finally:
            pass

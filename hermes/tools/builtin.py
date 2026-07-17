"""Tools internas (Nivel 1: sin red externa, sin subprocess).

Sprint 4 MVP-1: T2-T5.

Tools implementadas:
- get_current_time: hora actual en zona horaria del usuario
- get_weather: clima actual via Open-Meteo (HTTP nativo, sin API key)
- search_vault: grep en el Obsidian Vault local
- get_system_status: uptime, version, db health, circuit breakers

Seguridad: todas las tools pasan por el pipeline de seguridad de Nivel 1
(XML delimiters + truncate 2500 + pre-filtro regex) que se aplica en
T6. Esta capa (builtin.py) NO incluye esa lógica — la responsabilidad
es del AgentLoop o del ToolRegistry.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from hermes.config import Settings
    from hermes.llm.router import LLMRouter
    from hermes.memory.db import Database


# Timeout HTTP para tools externas (10s = suficiente para APIs rápidas,
# agresivo para detectar problemas rápido)
_HTTP_TIMEOUT_S = 10.0

# User-Agent identificable (algunos proxies bloquean UAs genéricos)
_USER_AGENT = "Oroimen/0.4 (https://github.com/AdrianMedico/oroimen)"


# ===========================================================================
# T2: get_current_time
# ===========================================================================


# Zona horaria por offset en horas (en vez de zoneinfo que requiere tzdata).
# Limitado a las zonas que realmente necesitamos (Madrid, UTC, NY, LA).
# Si necesitas una nueva zona, añade su offset aquí.
_TZ_OFFSETS_HOURS: dict[str, int] = {
    "UTC": 0,
    "Europe/Madrid": 1,  # CET (invierno). CEST (verano) se calcula abajo.
    "Europe/London": 0,
    "America/New_York": -5,  # EST (invierno). EDT (verario) se calcula abajo.
    "America/Los_Angeles": -8,  # PST (invierno). PDT (verano) se calcula abajo.
}


# Fechas aproximadas del cambio horario (simplificado).
# Última semana de marzo → +1h (DST start). Última semana de octubre → -1h.
def _is_dst_europe(dt: datetime) -> bool:
    """¿Estamos en horario de verano (CEST) en Europa?"""
    # Última domingo de marzo a última domingo de octubre.
    if dt.month < 3 or dt.month > 10:
        return False
    if 3 < dt.month < 10:
        return True
    # Meses de cambio (3 y 10): simplificado, asumimos DST desde día 25.
    return dt.day >= 25


def _is_dst_america(dt: datetime) -> bool:
    """¿Estamos en horario de verano (EDT/PDT) en América?"""
    if dt.month < 3 or dt.month > 11:
        return False
    if 3 < dt.month < 11:
        return True
    return dt.day >= 8  # 2do domingo aprox


async def get_current_time(settings: Settings) -> str:
    """Devuelve la hora actual en formato ISO 8601.

    Zona horaria configurable via settings.tz (default: Europe/Madrid).

    Implementación: usa datetime.now(UTC) + timedelta con offset
    configurable. No depende de `zoneinfo` (que requiere el paquete
    `tzdata` en Windows) ni de `pytz`.

    Limitaciones: solo soporta las zonas definidas en _TZ_OFFSETS_HOURS.
    Para zonas nuevas, añadir el offset en el dict.

    Returns:
        String ISO 8601, ej: "2026-06-22T14:30:00+02:00"
    """
    tz_name = settings.tz
    base_offset = _TZ_OFFSETS_HOURS.get(tz_name, 0)  # 0 = UTC fallback

    # Ajustar DST según zona
    now_utc = datetime.now(UTC)
    if tz_name.startswith("Europe/"):
        dst = 1 if _is_dst_europe(now_utc) else 0
    elif tz_name.startswith("America/"):
        dst = 1 if _is_dst_america(now_utc) else 0
    else:
        dst = 0  # UTC y otros: sin DST

    offset_hours = base_offset + dst
    local_tz = timezone(timedelta(hours=offset_hours))
    return datetime.now(local_tz).isoformat()


GET_CURRENT_TIME_SCHEMA: dict = {
    "type": "object",
    # Workaround: MiniMax-M3 (via endpoint Anthropic-compatible) rechaza
    # tools con properties={} ("parameters is empty (2013)"). Bug
    # históricamente atribuido al provider opencode-go + MiniMax
    # (issue 2013); se mantiene el workaround aunque ahora hablamos con
    # MiniMax API directo, por defensa. Añadimos un property opcional
    # dummy que el LLM puede ignorar.
    "properties": {
        "_unused": {
            "type": "string",
            "description": "ignored",
        },
    },
    "required": [],
    "additionalProperties": False,
}


# ===========================================================================
# T3: get_weather (Open-Meteo)
# ===========================================================================


# Sprint 9.4: cache de get_weather (15 min TTL) para no spamear
# Open-Meteo. Key: city lowercased.
_WEATHER_CACHE: dict[str, tuple[float, str]] = {}
_WEATHER_CACHE_TTL_S = 15 * 60  # 15 min


async def get_weather(city: str, *, client: httpx.AsyncClient | None = None) -> str:
    """Devuelve el clima actual + pronostico 7 dias + calidad del aire
    de una ciudad usando Open-Meteo (gratis, sin API key).

    Sprint 9.4: ahora incluye:
    - Temperatura actual + feels_like + humedad
    - Viento (velocidad + direccion + rafagas)
    - Pronostico 7 dias (max/min temp, precipitacion, codigo clima)
    - Calidad del aire (PM2.5, PM10, O3, NO2, indice EU)
    - Indice UV actual
    - Cache 15 min para evitar spamear APIs

    Flujo (3 requests HTTP):
    1. Geocoding: geocoding-api.open-meteo.com -> {lat, lon}
    2. Forecast: api.open-meteo.com/v1/forecast (current+hourly+daily)
    3. Air quality: air-quality-api.open-meteo.com/v1/air-quality

    Args:
        city: nombre de la ciudad (e.g. "Madrid", "Barcelona").
        client: cliente httpx opcional. Si es None, crea uno nuevo.

    Returns:
        Resumen compacto estructurado (ver formato abajo).

    Raises:
        ValueError: si la ciudad no se encuentra.
        httpx.HTTPError: si la API no responde.
    """
    # 0. Cache check
    import time as _time

    cache_key = city.lower().strip()
    cached = _WEATHER_CACHE.get(cache_key)
    if cached and (_time.time() - cached[0]) < _WEATHER_CACHE_TTL_S:
        return cached[1]

    own_client = client is None
    if own_client:
        http_client: httpx.AsyncClient = httpx.AsyncClient(
            timeout=_HTTP_TIMEOUT_S,
            headers={"User-Agent": _USER_AGENT},
        )
    else:
        assert client is not None, "client must not be None when own_client is False"
        http_client = client
    try:
        # 1. Geocoding
        geo_resp = await http_client.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "es", "format": "json"},
        )
        geo_resp.raise_for_status()
        geo_data = geo_resp.json()
        results = geo_data.get("results") or []
        if not results:
            raise ValueError(f"Ciudad no encontrada: {city!r}")
        loc = results[0]
        lat, lon = loc["latitude"], loc["longitude"]
        resolved_name = loc.get("name", city)
        country = loc.get("country", "?")

        # 2. Forecast completo (current + daily + uv_index_max)
        wx_resp = await http_client.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,apparent_temperature,is_day,"
                "precipitation,weather_code,wind_speed_10m,wind_direction_10m,wind_gusts_10m,uv_index",
                "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max,uv_index_max",
                "timezone": "auto",
                "forecast_days": 7,
            },
        )
        wx_resp.raise_for_status()
        wx_data = wx_resp.json()
        cw = wx_data.get("current") or {}
        daily = wx_data.get("daily") or {}

        # 3. Calidad del aire (Air Quality API, gratis sin key)
        try:
            aq_resp = await http_client.get(
                "https://air-quality-api.open-meteo.com/v1/air-quality",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current": "european_aqi,pm10,pm2_5,ozone,nitrogen_dioxide",
                    "timezone": "auto",
                },
            )
            aq_resp.raise_for_status()
            aq = aq_resp.json().get("current") or {}
        except Exception:
            aq = {}

        # Formatear resumen
        def _wmo_code_desc(code: int) -> str:
            """Mapeo WMO weather codes a descripcion humana (subset)."""
            mapping = {
                0: "despejado",
                1: "mayormente despejado",
                2: "parcialmente nublado",
                3: "nublado",
                45: "niebla",
                48: "niebla con escarcha",
                51: "llovizna ligera",
                53: "llovizna",
                55: "llovizna densa",
                61: "lluvia ligera",
                63: "lluvia",
                65: "lluvia densa",
                71: "nevada ligera",
                73: "nevada",
                75: "nevada densa",
                80: "chubascos ligeros",
                81: "chubascos",
                82: "chubascos violentos",
                95: "tormenta electrica",
                96: "tormenta con granizo",
                99: "tormenta con granizo fuerte",
            }
            return mapping.get(code, f"codigo {code}")

        parts = [f"**{resolved_name}, {country}**"]
        # Actual
        if cw:
            t = cw.get("temperature_2m", "?")
            feels = cw.get("apparent_temperature", "?")
            humid = cw.get("relative_humidity_2m", "?")
            wind = cw.get("wind_speed_10m", "?")
            gusts = cw.get("wind_gusts_10m", "?")
            wdir = cw.get("wind_direction_10m", "?")
            uv = cw.get("uv_index", "?")
            code = cw.get("weather_code", 0)
            precip = cw.get("precipitation", 0)
            parts.append(
                f"Ahora: {t}°C (sensacion {feels}°C), {humid}% humedad, "
                f"viento {wind} km/h (rafagas {gusts}) del {wdir}°, "
                f"UV {uv}, {_wmo_code_desc(code)}"
            )
            if precip > 0:
                parts.append(f"Precipitacion actual: {precip} mm")

        # Pronostico 7 dias
        if daily and daily.get("time"):
            parts.append("")
            parts.append("**Pronostico 7 dias:**")
            parts.append("| Fecha | Max | Min | Lluvia | Prob | UV | Clima |")
            parts.append("|-------|-----|-----|--------|------|-----|-------|")
            for i, date in enumerate(daily["time"][:7]):
                tmax = daily["temperature_2m_max"][i]
                tmin = daily["temperature_2m_min"][i]
                rain = daily.get("precipitation_sum", [0] * 7)[i]
                prob = daily.get("precipitation_probability_max", [0] * 7)[i]
                uvmx = daily.get("uv_index_max", [0] * 7)[i]
                code = daily.get("weather_code", [0] * 7)[i]
                parts.append(
                    f"| {date} | {tmax}°C | {tmin}°C | {rain}mm | {prob}% | {uvmx} | "
                    f"{_wmo_code_desc(code)} |"
                )

        # Calidad del aire
        if aq:
            aqi = aq.get("european_aqi", "?")
            pm25 = aq.get("pm2_5", "?")
            pm10 = aq.get("pm10", "?")
            o3 = aq.get("ozone", "?")
            no2 = aq.get("nitrogen_dioxide", "?")

            def _aqi_desc(v):
                try:
                    v = int(v)
                    if v <= 20:
                        return "buena"
                    if v <= 40:
                        return "aceptable"
                    if v <= 60:
                        return "moderada"
                    if v <= 80:
                        return "mala"
                    if v <= 100:
                        return "muy mala"
                    return "extremadamente mala"
                except (ValueError, TypeError):
                    return "?"

            parts.append("")
            parts.append(
                f"**Calidad del aire:** AQI {_aqi_desc(aqi)} ({aqi}), "
                f"PM2.5 {pm25} µg/m³, PM10 {pm10} µg/m³, O3 {o3} µg/m³, NO2 {no2} µg/m³"
            )

        result = "\n".join(parts)
        # Cache
        _WEATHER_CACHE[cache_key] = (_time.time(), result)
        return result
    finally:
        if own_client:
            await http_client.aclose()


GET_WEATHER_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "city": {
            "type": "string",
            "description": "Nombre de la ciudad (e.g. 'Madrid', 'Barcelona')",
        }
    },
    "required": ["city"],
}


# ===========================================================================
# T4: search_vault
# ===========================================================================


async def search_vault(
    query: str,
    *,
    vault_path: Path,
    max_results: int = 5,
    context_lines: int = 2,
) -> str:
    """Busca `query` en archivos .md del vault y devuelve matches con contexto.

    Búsqueda case-insensitive. Si no hay matches, devuelve mensaje
    user-friendly (NO error, el LLM debe saber que no encontró nada).

    Args:
        query: texto a buscar.
        vault_path: ruta al directorio del vault.
        max_results: máximo de matches a devolver.
        context_lines: líneas antes/después del match para contexto.

    Returns:
        Matches con contexto separados por "---", o "No se encontraron
        resultados" si no hay nada.

    Raises:
        FileNotFoundError: si vault_path no existe.
    """
    if not vault_path.exists():
        raise FileNotFoundError(f"Vault no encontrado: {vault_path}")

    pattern = re.compile(re.escape(query), re.IGNORECASE)
    matches: list[str] = []
    md_files = sorted(vault_path.rglob("*.md"))
    for md_file in md_files:
        try:
            content = md_file.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for line_num, line in enumerate(content.splitlines(), 1):
            if pattern.search(line):
                context = _extract_context(content, line_num, lines=context_lines)
                matches.append(f"📄 {md_file.relative_to(vault_path)}:{line_num}\n{context}")
                if len(matches) >= max_results:
                    break
        if len(matches) >= max_results:
            break
    if not matches:
        return f"No se encontraron resultados para {query!r}"
    return "\n---\n".join(matches)


def _extract_context(content: str, line_num: int, *, lines: int) -> str:
    """Extrae contexto alrededor de la línea N (1-indexed)."""
    all_lines = content.splitlines()
    start = max(0, line_num - 1 - lines)
    end = min(len(all_lines), line_num - 1 + lines + 1)
    chunk = all_lines[start:end]
    return "\n".join(chunk)


SEARCH_VAULT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Texto a buscar en el vault (case-insensitive)",
        }
    },
    "required": ["query"],
}


# ===========================================================================
# T5: get_system_status
# ===========================================================================


async def get_system_status(
    *,
    settings: Settings,
    db: Database,
    router: LLMRouter,
    start_time: float,
) -> str:
    """Devuelve un resumen multi-línea del estado del sistema.

    Secciones:
    - Uptime
    - Versión de Oroimen
    - Salud de la DB
    - Modelos LLM disponibles
    - Estado de circuit breakers

    Returns:
        Texto multi-línea con info relevante para diagnóstico.
    """
    import time

    uptime = _format_uptime(time.time() - start_time)
    db_ok = await _db_health(db)
    db_status = "OK" if db_ok else "DEGRADED"
    breakers = {m: router.breaker_state(m) for m in settings.text_chain}

    breaker_lines = "\n".join(f"  - {m}: {state}" for m, state in breakers.items())
    return (
        f"Oroimen v{settings.version}\n"
        f"Uptime: {uptime}\n"
        f"Database: {db_status}\n"
        f"LLM chain ({len(settings.text_chain)} modelos):\n"
        f"{breaker_lines}"
    )


def _format_uptime(seconds: float) -> str:
    """Formatea segundos en formato 'Xd Yh Zm'."""
    days, rem = divmod(int(seconds), 86400)
    hours, rem = divmod(rem, 3600)
    minutes, _ = divmod(rem, 60)
    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    parts.append(f"{minutes}m")
    return " ".join(parts)


async def _db_health(db: Database) -> bool:
    """Ping a la DB. Devuelve True si responde, False si no.

    Sprint 6 T53 v3.1: db.ping() ahora retorna bool directamente.
    Mantenemos el try/except como defensa en profundidad.
    """
    try:
        return bool(await db.ping())
    except Exception:
        return False


GET_SYSTEM_STATUS_SCHEMA: dict = {
    "type": "object",
    # Mismo workaround que get_current_time: MiniMax-M3 rechaza
    # tools con properties={} (ver comentario en GET_CURRENT_TIME_SCHEMA).
    "properties": {
        "_unused": {
            "type": "string",
            "description": "ignored",
        },
    },
    "required": [],
    "additionalProperties": False,
}


# ===========================================================================
# Función helper: registrar todas las tools en un ToolRegistry
# ===========================================================================


def register_builtin_tools(
    registry: Any,  # ToolRegistry (forward ref para evitar import circular)
    *,
    settings: Settings,
    db: Database,
    router: LLMRouter,
    start_time: float,
    embeddings_service: object | None = None,
    vault_embedder: object | None = None,
) -> None:
    """Registra las tools builtin en el registry.

    Args:
        registry: ToolRegistry donde registrar.
        settings: Settings con version y tz.
        db: Database para health check.
        router: LLMRouter para circuit breaker state.
        start_time: timestamp de inicio del proceso.
        embeddings_service: EmbeddingsService opcional para registrar
            `search_files`; initialization remains lazy.
        vault_embedder: chunk-level search implementation used by the
            drop-folder ingestion path.
    """

    async def _get_current_time() -> str:
        return await get_current_time(settings)

    async def _get_weather(city: str) -> str:
        return await get_weather(city)

    async def _search_vault(query: str) -> str:
        return await search_vault(query, vault_path=settings.vault_path)

    async def _get_system_status() -> str:
        return await get_system_status(
            settings=settings, db=db, router=router, start_time=start_time
        )

    registry.register(
        "get_current_time",
        _get_current_time,
        description="Devuelve la hora actual del servidor en zona horaria configurada",
        schema=GET_CURRENT_TIME_SCHEMA,
    )
    if settings.outbound_tools_enabled:
        registry.register(
            "get_weather",
            _get_weather,
            description="Devuelve el clima actual de una ciudad (vía Open-Meteo)",
            schema=GET_WEATHER_SCHEMA,
        )
    registry.register(
        "search_vault",
        _search_vault,
        description="Busca texto en el Obsidian Vault local (case-insensitive, con contexto)",
        schema=SEARCH_VAULT_SCHEMA,
    )
    registry.register(
        "get_system_status",
        _get_system_status,
        description="Devuelve estado del sistema: uptime, versión, salud DB, circuit breakers",
        schema=GET_SYSTEM_STATUS_SCHEMA,
    )

    # Sprint 9.1: search_files (RAG). Solo se registra si el
    # EmbeddingsService está habilitado. Si no, el LLM no ve esta tool
    # y no puede invocarla. Defense in depth: aunque la tool esté
    # registrada, valida que embeddings_service.is_enabled antes de
    # ejecutarse (un caller externo podría intentar llamarla via API).
    if embeddings_service is not None:
        from hermes.tools.search_files import (
            SEARCH_FILES_SCHEMA,
            search_files_tool_callable,
        )

        async def _search_files(query: str, top_k: int = 5) -> str:
            return await search_files_tool_callable(
                query=query,
                top_k=top_k,
                embeddings_service=embeddings_service,
                db=db,
                vault_embedder=vault_embedder,
            )

        registry.register(
            "search_files",
            _search_files,
            description=(
                "Busca archivos en la library del usuario por similaridad "
                "semantica. Usar cuando el user pregunta sobre documentos "
                "subidos previamente (e.g. 'que dicen mis papers sobre X?'). "
                "Retorna fragmentos relevantes con su file_id y filename."
            ),
            schema=SEARCH_FILES_SCHEMA,
            tool_category="read",
        )

    # Sprint 4 MVP-2 T11c: 4 tools de Agent-Reach (read, youtube,
    # github, rss). Wrapper minimalista: enum allowlist + asyncio
    # subprocess + regex. Ver hermes/tools/agent_reach.py.
    # Sprint 5 T49: marcadas como tool_category="read" para que el
    # security pipeline les asigne read_tool_max_chars (150K, no 2500).
    if settings.outbound_tools_enabled:
        from hermes.tools.agent_reach import make_agent_reach_tools

        for tool_spec in make_agent_reach_tools():
            registry.register(
                tool_spec["name"],
                tool_spec["callable"],
                description=tool_spec["description"],
                schema=tool_spec["schema"],
                tool_category="read",
            )

    # Sprint 19 Slice 3: 5 tools de vault collections (list, create,
    # add, remove, move). Todas con tool_category="system" (budget corto).
    # Las tools envuelven VaultCollectionsRepo directamente, no al HTTP API.
    from hermes.tools.collections import register_collections_tools

    register_collections_tools(registry, db=db)

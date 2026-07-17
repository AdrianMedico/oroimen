"""Cost + time helpers centralizados para research jobs.

Ver TDD_S14_DEEP_RESEARCH.md §1.5.1.1 y §8.1.

Punto único de decisión:
- format_now() / format_now_at(): UNICO punto de formatting de timestamps.
  Garantiza exactamente 3 dígitos de milisegundos con `:03d` (anti-trampa
  lexicografica SQLite: `.12` < `.120` falla en comparación TEXT).
- calculate_cost(): Decimal con 4 decimales, quantize ROUND_HALF_UP.
- PRICING_TABLE: input/output USD per 1M tokens por modelo.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal

# Pricing en USD por 1M tokens (input, output).
# Fuente: architecture v1.1 §2.8, TBD empíricamente S14 D2-3 measurement run.
# Sprint 14 inicia con valores estimados; se ajustan tras measurement run.
PRICING_TABLE: dict[str, tuple[Decimal, Decimal]] = {
    "MiniMax-M3": (Decimal("0.30"), Decimal("0.60")),
    "MiniMax-M2.7-highspeed": (Decimal("0.10"), Decimal("0.20")),  # estimado
    "deepseek-v3": (Decimal("0.05"), Decimal("0.10")),  # estimado, budget mode
}


def format_now() -> str:
    """Retorna timestamp UTC en formato 'YYYY-MM-DD HH:MM:SS.sss' con EXACTAMENTE 3 dígitos.

    Regla absoluta (TDD §1.5.1): TODO código S14 que escribe timestamps usa este helper.
    NUNCA usar f-string con integer math, NUNCA usar `[:-3]` sobre `%f`.

    Por qué NO `datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]`:
        Sintaxis opaca. Si alguien hace 'ah, lo simplifico' y cambia a
        `f"{dt:%Y-%m-%d %H:%M:%S}.{dt.microsecond // 1000}"`, el resultado
        es '.12' (2 dígitos) en lugar de '.120' (3 dígitos) → ordering bug silencioso.

    Por qué `:03d`:
        - Garantiza exactamente 3 dígitos con zero-padding ('.005' no '.5').
        - Si microsecond=10000 (10ms), f'{val:03d}' produce '010' (correcto).
        - Si microsecond=0, f'{val:03d}' produce '000' (correcto).
    """
    now = datetime.now(UTC)
    ms = now.microsecond // 1000
    return f"{now.strftime('%Y-%m-%d %H:%M:%S')}.{ms:03d}"


def format_now_at(dt: datetime) -> str:
    """Como `format_now()` pero con un `datetime` explícito en vez de "now".

    Útil para cutoffs de recovery (e.g. `now - 2h`). Mantiene MISMA regla
    de 3 dígitos de milisegundos.

    Args:
        dt: datetime (UTC o naive — se interpreta como UTC si naive).

    Returns:
        string formato 'YYYY-MM-DD HH:MM:SS.sss' con 3 dígitos de ms.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    elif dt.tzinfo != UTC:
        dt = dt.astimezone(UTC)
    ms = dt.microsecond // 1000
    return f"{dt.strftime('%Y-%m-%d %H:%M:%S')}.{ms:03d}"


def calculate_cost(model: str, tokens_in: int, tokens_out: int) -> Decimal:
    """Calcula cost en USD con 4 decimales de precisión ($0.0001).

    Pricing por 1M tokens (input, output) según PRICING_TABLE.

    Args:
        model: nombre del modelo (debe estar en PRICING_TABLE).
        tokens_in: tokens de input del LLM call.
        tokens_out: tokens de output del LLM call.

    Returns:
        Decimal cuantizado a 4 decimales con ROUND_HALF_UP.

    Raises:
        KeyError: si model no está en PRICING_TABLE. Fail-fast en dev
            (mejor detectar typo que silently asume $0).
    """
    in_rate, out_rate = PRICING_TABLE[model]
    cost_in = (Decimal(tokens_in) / Decimal(1_000_000)) * in_rate
    cost_out = (Decimal(tokens_out) / Decimal(1_000_000)) * out_rate
    return (cost_in + cost_out).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)


# Safety margin (Q3 verifier finding): estimación heurística para pre-submit.
# Cubre input tokens (no medibles antes de la primera LLM call) + retries.
# 30% conservador: el modelo real rinde input ~50% del output en longitud,
# más 2 retries promedio con la mitad del cost, da ~+25-35% por encima de
# la estimación naive de solo output. Ver deliverable.md §"Q3 decision".
_ESTIMATION_SAFETY_MARGIN_PCT = Decimal("1.30")


def estimate_research_cost(
    max_sources: int,
    per_source_max_tokens: int,
    output_max_tokens: int,
    pricing_table: dict[str, tuple[Decimal, Decimal]],
    primary_model: str,
    fallback_model: str = "deepseek-v3",
) -> Decimal:
    """Heurística conservadora para pre-submit cost estimation.

    Asume:
    - Per-source synth: ``per_source_max_tokens`` x output_rate del modelo primario.
    - Final synth: ``output_max_tokens`` x output_rate del modelo primario.
    - Search + scrape: 0 (no LLM).
    - Padding +30% por safety margin (input tokens estimados + retries).

    Si ``primary_model`` no está en ``pricing_table``, cae a ``fallback_model``
    (deepseek-v3) en lugar de fallar. Esto evita que un typo en settings
    rompa el submit; el notifier recibirá la estimación del modelo fallback,
    subóptima pero bounded.

    Args:
        max_sources: número de URLs a scrapear (settings.deep_research_max_sources).
        per_source_max_tokens: max output de Phase 3 per source
            (settings.deep_research_per_source_max_tokens).
        output_max_tokens: max output de Phase 4 final synth
            (settings.deep_research_output_max_tokens).
        pricing_table: tabla de precios (inyectada para tests; producción usa PRICING_TABLE).
        primary_model: modelo LLM primario (e.g. "MiniMax-M3").
        fallback_model: modelo al que caer si primary_model no está en pricing_table.

    Returns:
        Decimal cuantizado a 4 decimales con ROUND_HALF_UP.

    Raises:
        KeyError: si ni primary ni fallback están en pricing_table.
            (Imposible en producción — PRICING_TABLE garantiza fallback.)
    """
    model = primary_model if primary_model in pricing_table else fallback_model
    _in_rate, out_rate = pricing_table[model]

    # Solo output token rate: heurística conservadora (no medimos input antes
    # de la primera call, así que lo cubrimos con el safety margin +30%).
    per_source_cost = (
        Decimal(max_sources) * Decimal(per_source_max_tokens) / Decimal(1_000_000) * out_rate
    )
    final_synth_cost = Decimal(output_max_tokens) / Decimal(1_000_000) * out_rate

    raw_total = (per_source_cost + final_synth_cost) * _ESTIMATION_SAFETY_MARGIN_PCT
    return raw_total.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)

"""Unit tests for hermes.jobs.cost.estimate_research_cost (Q3 verifier finding).

Anti-regression checks (deliverable §Q3, verifier-report.md 2026-07-03):
- Default S14 config (5 sources * 3000 per_source * 10000 output) produces a
  sensible, bounded estimate (no mas de ~$0.05 para MiniMax-M3).
- Scale lineal con max_sources: 20 sources produce ~4x el cost de 5 sources
  (solo la parte per_source escala; final_synth queda constante).
- Unknown primary_model cae a fallback_model (deepseek-v3) sin crashear.
- Safety margin 1.30 se aplica antes de quantize (verificable comparando
  raw vs final).
- Quantize a 4 decimales con ROUND_HALF_UP -- el output es Decimal con
  exactamente 4 decimales, no float.
- Edge case: 0 sources, 0 tokens -> $0.0000.
- Custom pricing_table inyectable (aisla de PRICING_TABLE global para tests).
- Si ni primary ni fallback estan en pricing_table -> KeyError (fail-fast).
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

import pytest

from hermes.jobs.cost import estimate_research_cost

# MiniMax-M3: in $0.30/M, out $0.60/M (PRICING_TABLE oficial S14)
_PRICING_MINIMAX = {
    "MiniMax-M3": (Decimal("0.30"), Decimal("0.60")),
}
_PRICING_DEEPSEEK = {
    "deepseek-v3": (Decimal("0.05"), Decimal("0.10")),
}


def test_estimate_research_cost_5_sources_default() -> None:
    """S14 default config: 5 sources * 3000 tok * 10000 output on MiniMax-M3.

    per_source = 5 * 3000 * 0.60 / 1e6 = 0.009
    final_synth = 10000 * 0.60 / 1e6 = 0.006
    raw = (0.009 + 0.006) * 1.30 = 0.0195
    quantize -> 0.0195
    """
    cost = estimate_research_cost(
        max_sources=5,
        per_source_max_tokens=3000,
        output_max_tokens=10_000,
        pricing_table=_PRICING_MINIMAX,
        primary_model="MiniMax-M3",
    )
    assert isinstance(cost, Decimal), f"expected Decimal, got {type(cost)}"
    assert cost == Decimal("0.0195")
    # Quantize preserva 4 decimales exactos
    assert cost.as_tuple().exponent == -4  # type: ignore[attr-defined]


def test_estimate_research_cost_20_sources_scales_linearly() -> None:
    """20 sources producen exactamente 4x la parte per_source vs 5 sources.

    Comparando con 5 sources:
      per_source_5 = 5 * 3000 * 0.60 / 1e6 = 0.009
      per_source_20 = 20 * 3000 * 0.60 / 1e6 = 0.036  (= 4x per_source_5)
      final_synth = 0.006 (constante, no escala con sources)
      raw_5 = (0.009 + 0.006) * 1.30 = 0.0195
      raw_20 = (0.036 + 0.006) * 1.30 = 0.0546
    """
    cost_5 = estimate_research_cost(
        max_sources=5,
        per_source_max_tokens=3000,
        output_max_tokens=10_000,
        pricing_table=_PRICING_MINIMAX,
        primary_model="MiniMax-M3",
    )
    cost_20 = estimate_research_cost(
        max_sources=20,
        per_source_max_tokens=3000,
        output_max_tokens=10_000,
        pricing_table=_PRICING_MINIMAX,
        primary_model="MiniMax-M3",
    )
    # No es exactamente 4x porque final_synth es constante -- la diferencia
    # se diluye. Verificamos que escala EN LA DIRECCION CORRECTA y que la
    # diferencia entre per_source es exactamente 4x:
    per_source_5 = Decimal(5) * Decimal(3000) * Decimal("0.60") / Decimal(1_000_000)
    per_source_20 = Decimal(20) * Decimal(3000) * Decimal("0.60") / Decimal(1_000_000)
    assert per_source_20 / per_source_5 == Decimal("4")
    assert cost_20 > cost_5
    # Magnitud: cost_20 debe ser claramente mayor que cost_5
    assert cost_20 - cost_5 > Decimal("0.03")
    assert cost_20 == Decimal("0.0546")


def test_estimate_research_cost_unknown_primary_uses_fallback() -> None:
    """Primary model no esta en pricing_table -> cae a fallback (deepseek-v3).

    Pricing table solo tiene deepseek-v3 (NO MiniMax-M3). El primary_model
    es desconocido, asi que cae a deepseek-v3. Resultado: misma formula con
    out_rate=0.10.
    """
    cost_via_fallback = estimate_research_cost(
        max_sources=5,
        per_source_max_tokens=3000,
        output_max_tokens=10_000,
        pricing_table=_PRICING_DEEPSEEK,
        primary_model="gpt-99-turbo-unknown",
        fallback_model="deepseek-v3",
    )
    cost_direct = estimate_research_cost(
        max_sources=5,
        per_source_max_tokens=3000,
        output_max_tokens=10_000,
        pricing_table=_PRICING_DEEPSEEK,
        primary_model="deepseek-v3",
    )
    # Mismo calculo numerico (out_rate=0.10), misma formula -> mismo output
    assert cost_via_fallback == cost_direct
    # Sanity: deepseek-v3 es mas barato que MiniMax-M3 (precios distintos)
    cost_minimax = estimate_research_cost(
        max_sources=5,
        per_source_max_tokens=3000,
        output_max_tokens=10_000,
        pricing_table=_PRICING_MINIMAX,
        primary_model="MiniMax-M3",
    )
    assert cost_via_fallback < cost_minimax


def test_estimate_research_cost_quantize_4_decimals() -> None:
    """Quantize a exactamente 4 decimales con ROUND_HALF_UP.

    Caso con aritmetica que generaria muchos decimales: 7 sources * 1234 tok.
    per_source = 7 * 1234 * 0.60 / 1e6 = 5182.8 / 1e6 = 0.0051828
    final_synth = 10_000 * 0.60 / 1e6 = 0.006
    raw = (0.0051828 + 0.006) * 1.30 = 0.01453764
    quantize 4 dec -> 0.0145 (ROUND_HALF_UP)
    """
    cost = estimate_research_cost(
        max_sources=7,
        per_source_max_tokens=1234,
        output_max_tokens=10_000,
        pricing_table=_PRICING_MINIMAX,
        primary_model="MiniMax-M3",
    )
    # Verify exactly 4 decimal places (Decimal exponent == -4)
    _sign, _digits, exponent = cost.as_tuple()
    assert exponent == -4, f"expected 4 decimals, got exponent={exponent}, value={cost}"
    # ROUND_HALF_UP: 0.01453764 -> 0.0145
    assert cost == Decimal("0.0145")


def test_estimate_research_cost_safety_margin_applied() -> None:
    """El safety margin 1.30 se aplica ANTES del quantize.

    Verificamos comparando el output con (per_source + final_synth) * 1.30
    redondeado a 4 decimales con ROUND_HALF_UP. Si el margin no se aplicara,
    el cost seria ~30% menor.
    """
    cost = estimate_research_cost(
        max_sources=5,
        per_source_max_tokens=3000,
        output_max_tokens=10_000,
        pricing_table=_PRICING_MINIMAX,
        primary_model="MiniMax-M3",
    )
    per_source = Decimal(5) * Decimal(3000) * Decimal("0.60") / Decimal(1_000_000)
    final_synth = Decimal(10_000) * Decimal("0.60") / Decimal(1_000_000)
    raw = (per_source + final_synth) * Decimal("1.30")
    expected = raw.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
    assert cost == expected
    # Sanity: sin margin el cost seria ~$0.015; con margin debe ser ~$0.0195
    no_margin = (per_source + final_synth).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
    assert cost > no_margin
    # Ratio ~= 1.30
    ratio = cost / no_margin
    assert Decimal("1.29") <= ratio <= Decimal("1.31")


def test_estimate_research_cost_neither_primary_nor_fallback_raises() -> None:
    """Si ni primary ni fallback estan en pricing_table -> KeyError (fail-fast).

    Pricing table vacia: cualquier modelo -> KeyError.
    """
    with pytest.raises(KeyError):
        estimate_research_cost(
            max_sources=5,
            per_source_max_tokens=3000,
            output_max_tokens=10_000,
            pricing_table={},  # vacia
            primary_model="unknown-1",
            fallback_model="unknown-2",
        )


def test_estimate_research_cost_zero_tokens() -> None:
    """Edge case: 0 sources, 0 tokens -> cost = $0.0000 (no crash, no NaN)."""
    cost = estimate_research_cost(
        max_sources=0,
        per_source_max_tokens=0,
        output_max_tokens=0,
        pricing_table=_PRICING_MINIMAX,
        primary_model="MiniMax-M3",
    )
    assert cost == Decimal("0.0000")
    assert cost.as_tuple().exponent == -4  # type: ignore[attr-defined]


def test_estimate_research_cost_inject_pricing_table_isolates_global() -> None:
    """Inyectar pricing_table NO muta el PRICING_TABLE global (test isolation).

    Verificamos que pasar un custom pricing_table con un modelo unico no
    contamina PRICING_TABLE. Esto es importante para tests paralelos y
    para evitar que un test rompa otro.
    """
    import hermes.jobs.cost as cost_module

    custom_pricing = {
        "test-only-model": (Decimal("99.99"), Decimal("99.99")),
    }
    cost = estimate_research_cost(
        max_sources=5,
        per_source_max_tokens=3000,
        output_max_tokens=10_000,
        pricing_table=custom_pricing,
        primary_model="test-only-model",
        fallback_model="test-only-model",
    )
    assert cost > Decimal("0")
    # PRICING_TABLE global no debe contener nuestro modelo de test
    assert "test-only-model" not in cost_module.PRICING_TABLE
    # Y los modelos oficiales siguen ahi
    assert "MiniMax-M3" in cost_module.PRICING_TABLE

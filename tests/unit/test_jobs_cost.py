"""Unit tests for hermes.jobs.cost — PRICING_TABLE + calculate_cost.

Anti-regression checks (TDD §8.1):
- calculate_cost() returns Decimal quantized to 4 decimals.
- MiniMax-M3: $0.30/M in, $0.60/M out.
- deepseek-v3: $0.05/M in, $0.10/M out.
- Unknown model → KeyError (fail-fast).
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from hermes.jobs.cost import PRICING_TABLE, calculate_cost


def test_calculate_cost_mini_max_m3() -> None:
    """28000 in + 8500 out on MiniMax-M3 → exact Decimal.

    in  = 28000 / 1e6 * 0.30 = 0.0084
    out =  8500 / 1e6 * 0.60 = 0.0051
    total = 0.0135 (quantized ROUND_HALF_UP).
    """
    cost = calculate_cost("MiniMax-M3", 28_000, 8_500)
    assert isinstance(cost, Decimal), f"expected Decimal, got {type(cost)}"
    # Exactly 4 decimals (after quantization)
    assert cost == Decimal("0.0135")
    # No accidental drift to float
    assert str(cost) == "0.0135"


def test_calculate_cost_deepseek_v3() -> None:
    """100000 in + 50000 out on deepseek-v3 (cheap).

    in  = 100000 / 1e6 * 0.05 = 0.005
    out =  50000 / 1e6 * 0.10 = 0.005
    total = 0.0100 (quantized, with trailing zero).
    """
    cost = calculate_cost("deepseek-v3", 100_000, 50_000)
    assert cost == Decimal("0.0100")
    # Decimal quantize preserves trailing zeros — NOT a float 0.01
    assert str(cost) == "0.0100"


def test_calculate_cost_unknown_model() -> None:
    """Unknown model raises KeyError — fail-fast on typos.

    Verifies PRICING_TABLE doesn't silently default to $0.
    """
    with pytest.raises(KeyError) as exc_info:
        calculate_cost("gpt-99-turbo", 1000, 500)
    assert "gpt-99-turbo" in str(exc_info.value) or "gpt-99-turbo" in repr(exc_info.value)


def test_pricing_table_has_mini_max_and_deepseek() -> None:
    """Both Sprint 14 budget-relevant models present in PRICING_TABLE."""
    assert "MiniMax-M3" in PRICING_TABLE
    assert "deepseek-v3" in PRICING_TABLE
    # Each entry is (in_rate, out_rate) as Decimal
    for model, (in_rate, out_rate) in PRICING_TABLE.items():
        assert isinstance(in_rate, Decimal), f"{model}: in_rate not Decimal"
        assert isinstance(out_rate, Decimal), f"{model}: out_rate not Decimal"
        assert in_rate > 0 and out_rate > 0, f"{model}: rates must be > 0"


def test_calculate_cost_zero_tokens() -> None:
    """Edge case: 0 tokens in/out → cost = $0.0000."""
    cost = calculate_cost("MiniMax-M3", 0, 0)
    assert cost == Decimal("0.0000")

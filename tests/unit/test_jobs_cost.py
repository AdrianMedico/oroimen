"""Unit tests for hermes.jobs.cost — PRICING_TABLE + calculate_cost.

Anti-regression checks (TDD §8.1; DR-Q1A-PRE1A cost truth):
- calculate_cost() returns Decimal quantized to 4 decimals.
- MiniMax-M3: $0.30/M in, $1.20/M out (official standard tier,
  "Permanent 50% off" promo, <=512k input, verified 2026-07-21
  from https://platform.minimax.io/docs/guides/pricing-paygo).
- MiniMax-M2.7-highspeed: $0.60/M in, $2.40/M out (same source).
- deepseek-v3: $0.05/M in, $0.10/M out (preserved from prior
  estimate; not verified in this slice per DR-Q1A-PRE1A scope).
- Unknown model → KeyError (fail-fast).
- cost_usd is an estimated pay-as-you-go-equivalent amount, not
  actual provider billing.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from hermes.jobs.cost import (
    PRICING_AS_OF,
    PRICING_BASIS,
    PRICING_SOURCE,
    PRICING_TABLE,
    calculate_cost,
)


def test_calculate_cost_mini_max_m3() -> None:
    """28000 in + 8500 out on MiniMax-M3 → exact Decimal.

    With the verified rate (DR-Q1A-PRE1A):
      in  = 28000 / 1e6 * 0.30 = 0.0084
      out =  8500 / 1e6 * 1.20 = 0.0102
      total = 0.0186 (quantized ROUND_HALF_UP).
    """
    cost = calculate_cost("MiniMax-M3", 28_000, 8_500)
    assert isinstance(cost, Decimal), f"expected Decimal, got {type(cost)}"
    # Exactly 4 decimals (after quantization)
    assert cost == Decimal("0.0186")
    # No accidental drift to float
    assert str(cost) == "0.0186"


def test_calculate_cost_mini_max_m3_at_typical_dr_run() -> None:
    """5 sources x 3000 output + final 10000 output on MiniMax-M3.

    Per-source: 5 * 3000 * 1.20 / 1e6 = 0.018
    Final:       10000 * 1.20 / 1e6 = 0.012
    Total raw:   0.030
    """
    cost = calculate_cost("MiniMax-M3", 0, 30_000)
    assert cost == Decimal("0.0360")  # 30000 * 1.20 / 1e6 = 0.0360


def test_calculate_cost_mini_max_m2_7_highspeed() -> None:
    """1000 in + 500 out on MiniMax-M2.7-highspeed.

    With the verified rate (DR-Q1A-PRE1A):
      in  = 1000 / 1e6 * 0.60 = 0.0006
      out =  500 / 1e6 * 2.40 = 0.0012
      total = 0.0018 (quantized ROUND_HALF_UP).
    """
    cost = calculate_cost("MiniMax-M2.7-highspeed", 1_000, 500)
    assert isinstance(cost, Decimal)
    assert cost == Decimal("0.0018")


def test_calculate_cost_deepseek_v3() -> None:
    """100000 in + 50000 out on deepseek-v3 (cheap, preserved from
    prior estimate; not verified in this slice).

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
    """All S14 / S15 budget-relevant models present in PRICING_TABLE."""
    assert "MiniMax-M3" in PRICING_TABLE
    assert "MiniMax-M2.7-highspeed" in PRICING_TABLE
    assert "deepseek-v3" in PRICING_TABLE
    # Each entry is (in_rate, out_rate) as Decimal
    for model, (in_rate, out_rate) in PRICING_TABLE.items():
        assert isinstance(in_rate, Decimal), f"{model}: in_rate not Decimal"
        assert isinstance(out_rate, Decimal), f"{model}: out_rate not Decimal"
        assert in_rate > 0 and out_rate > 0, f"{model}: rates must be > 0"


def test_pricing_table_mini_max_m3_matches_verified_official_rate() -> None:
    """MiniMax-M3: $0.30/M in, $1.20/M out (as-of 2026-07-21).

    This is an as-of regression test. The values asserted here are
    the rates that the operator verified against the official
    MiniMax pricing page on 2026-07-21 (see PRICING_AS_OF). The
    test itself performs NO network access and cannot prove that
    the official page has not changed since then. A future slice
    must re-verify the rates against the official page and update
    BOTH PRICING_AS_OF and PRICING_TABLE together; this test will
    then be updated to the new pinned values.
    """
    in_rate, out_rate = PRICING_TABLE["MiniMax-M3"]
    assert in_rate == Decimal("0.30"), (
        f"MiniMax-M3 input rate pinned at $0.30/M as of PRICING_AS_OF "
        f"({PRICING_AS_OF}); got {in_rate}. If the official page has "
        f"genuinely changed, re-verify the rate, update PRICING_TABLE "
        f"AND PRICING_AS_OF, and update this assertion in the same "
        f"commit. See {PRICING_SOURCE} for the official source."
    )
    assert out_rate == Decimal("1.20"), (
        f"MiniMax-M3 output rate pinned at $1.20/M as of PRICING_AS_OF "
        f"({PRICING_AS_OF}); got {out_rate}. If the official page has "
        f"genuinely changed, re-verify the rate, update PRICING_TABLE "
        f"AND PRICING_AS_OF, and update this assertion in the same "
        f"commit. See {PRICING_SOURCE} for the official source."
    )


def test_pricing_table_mini_max_m2_7_highspeed_matches_verified_official_rate() -> None:
    """MiniMax-M2.7-highspeed: $0.60/M in, $2.40/M out (as-of 2026-07-21).

    Same as-of regression semantics as
    ``test_pricing_table_mini_max_m3_matches_verified_official_rate``.
    The test performs no network access.
    """
    in_rate, out_rate = PRICING_TABLE["MiniMax-M2.7-highspeed"]
    assert in_rate == Decimal("0.60"), (
        f"MiniMax-M2.7-highspeed input rate pinned at $0.60/M as of "
        f"PRICING_AS_OF ({PRICING_AS_OF}); got {in_rate}. See "
        f"{PRICING_SOURCE} for the official source."
    )
    assert out_rate == Decimal("2.40"), (
        f"MiniMax-M2.7-highspeed output rate pinned at $2.40/M as of "
        f"PRICING_AS_OF ({PRICING_AS_OF}); got {out_rate}. See "
        f"{PRICING_SOURCE} for the official source."
    )


def test_pricing_basis_is_paygo_equivalent() -> None:
    """PRICING_BASIS exposes the cost_usd semantics explicitly.

    cost_usd is an estimated pay-as-you-go-equivalent amount at the
    official standard rates. Operators using a subscription or
    quota-backed plan must treat this value as a relative cost proxy,
    NOT as a spend figure.
    """
    assert PRICING_BASIS == "official_paygo_equivalent"


def test_pricing_as_of_is_iso_date() -> None:
    """PRICING_AS_OF records the official pricing retrieval date.

    The date is the verification timestamp for the rates in
    PRICING_TABLE; if the official source changes, this date must
    be updated and the rates re-verified.
    """
    assert PRICING_AS_OF == "2026-07-21"
    # ISO date format check
    import re
    assert re.match(r"^\d{4}-\d{2}-\d{2}$", PRICING_AS_OF), (
        f"PRICING_AS_OF must be ISO date YYYY-MM-DD; got {PRICING_AS_OF}"
    )


def test_pricing_source_is_official_minimax_url() -> None:
    """PRICING_SOURCE points to the official MiniMax pricing page.

    The only acceptable source is the official primary MiniMax
    documentation. Reseller, blog, forum, or aggregator pricing is
    NOT acceptable per DR-Q1A-PRE1A scope.
    """
    assert PRICING_SOURCE == "https://platform.minimax.io/docs/guides/pricing-paygo"
    assert PRICING_SOURCE.startswith("https://platform.minimax.io/")


def test_calculate_cost_zero_tokens() -> None:
    """Edge case: 0 tokens in/out → cost = $0.0000."""
    cost = calculate_cost("MiniMax-M3", 0, 0)
    assert cost == Decimal("0.0000")

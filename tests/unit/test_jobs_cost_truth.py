"""DR-Q1A-PRE1A cost truth tests.

These tests prove the documented pay-as-you-go-equivalent semantics
of ``cost_usd`` and the verified official pricing rates. They are
the regression gate for the cost module: if any of these fail, the
rates are out of sync with the official MiniMax pricing page.

Anti-regression checks (DR-Q1A-PRE1A):
- ``PRICING_BASIS == "official_paygo_equivalent"``
- ``PRICING_AS_OF == "2026-07-21"``
- ``PRICING_SOURCE`` points to the official MiniMax pricing page
- ``PRICING_TABLE["MiniMax-M3"] == (Decimal("0.30"), Decimal("1.20"))``
- ``PRICING_TABLE["MiniMax-M2.7-highspeed"] == (Decimal("0.60"), Decimal("2.40"))``
- All call sites of ``calculate_cost`` and ``estimate_research_cost``
  receive the verified rates via the production PRICING_TABLE.
- Token counts are the only client-observable quantity; the cost
  estimate is a derived display field, NOT a spend figure.
"""

from __future__ import annotations

from decimal import Decimal

from hermes.jobs.cost import (
    PRICING_AS_OF,
    PRICING_BASIS,
    PRICING_SOURCE,
    PRICING_TABLE,
    calculate_cost,
    estimate_research_cost,
)


def test_pricing_basis_is_official_paygo_equivalent() -> None:
    """PRICING_BASIS exposes the cost_usd semantics explicitly.

    This is the single most important documentation guarantee of
    DR-Q1A-PRE1A. cost_usd is an estimated pay-as-you-go-equivalent
    amount; it is NOT actual provider billing.
    """
    assert PRICING_BASIS == "official_paygo_equivalent"
    # The string must be stable; downstream code or docs may
    # assert against it.
    assert isinstance(PRICING_BASIS, str)
    assert PRICING_BASIS == "official_paygo_equivalent"


def test_pricing_as_of_is_verified_retrieval_date() -> None:
    """PRICING_AS_OF records the official pricing retrieval date.

    The rate values in PRICING_TABLE are valid only as of this
    date. If the official source changes, this date and the rates
    must both be updated together.
    """
    assert PRICING_AS_OF == "2026-07-21"
    # ISO date format: YYYY-MM-DD
    assert len(PRICING_AS_OF) == 10
    assert PRICING_AS_OF[4] == "-"
    assert PRICING_AS_OF[7] == "-"


def test_pricing_source_is_official_minimax_url() -> None:
    """PRICING_SOURCE points to the official primary source.

    Per DR-Q1A-PRE1A scope: only the official primary MiniMax
    documentation is acceptable. Reseller, blog, forum, or
    aggregator pricing is NOT acceptable.
    """
    assert PRICING_SOURCE.startswith("https://platform.minimax.io/")
    assert "pricing-paygo" in PRICING_SOURCE


def test_pricing_table_mini_max_m3_input_rate() -> None:
    """MiniMax-M3 input rate = $0.30/M tokens (verified 2026-07-21)."""
    in_rate, _ = PRICING_TABLE["MiniMax-M3"]
    assert in_rate == Decimal("0.30"), (
        f"MiniMax-M3 input rate is now ${in_rate}/M; expected $0.30/M "
        f"per the official standard tier, <=512k input, 'Permanent 50% "
        f"off' promo (verified 2026-07-21 from {PRICING_SOURCE})."
    )


def test_pricing_table_mini_max_m3_output_rate() -> None:
    """MiniMax-M3 output rate = $1.20/M tokens (verified 2026-07-21)."""
    _in_rate, out_rate = PRICING_TABLE["MiniMax-M3"]
    assert out_rate == Decimal("1.20"), (
        f"MiniMax-M3 output rate is now ${out_rate}/M; expected $1.20/M "
        f"per the official standard tier, <=512k input, 'Permanent 50% "
        f"off' promo (verified 2026-07-21 from {PRICING_SOURCE})."
    )


def test_pricing_table_mini_max_m2_7_highspeed_input_rate() -> None:
    """MiniMax-M2.7-highspeed input rate = $0.60/M tokens (verified 2026-07-21)."""
    in_rate, _ = PRICING_TABLE["MiniMax-M2.7-highspeed"]
    assert in_rate == Decimal("0.60"), (
        f"MiniMax-M2.7-highspeed input rate is now ${in_rate}/M; expected "
        f"$0.60/M per the official standard tier (verified 2026-07-21 "
        f"from {PRICING_SOURCE})."
    )


def test_pricing_table_mini_max_m2_7_highspeed_output_rate() -> None:
    """MiniMax-M2.7-highspeed output rate = $2.40/M tokens (verified 2026-07-21)."""
    _in_rate, out_rate = PRICING_TABLE["MiniMax-M2.7-highspeed"]
    assert out_rate == Decimal("2.40"), (
        f"MiniMax-M2.7-highspeed output rate is now ${out_rate}/M; "
        f"expected $2.40/M per the official standard tier (verified "
        f"2026-07-21 from {PRICING_SOURCE})."
    )


def test_calculate_cost_on_production_pricing_table_uses_verified_rates() -> None:
    """calculate_cost(MiniMax-M3, ...) on the production PRICING_TABLE
    uses the verified rates (NOT the legacy $0.60/M output rate that
    existed before DR-Q1A-PRE1A)."""
    # Use a non-trivial token count so the math distinguishes the
    # legacy $0.60 output rate from the verified $1.20 rate.
    cost = calculate_cost("MiniMax-M3", 0, 10_000)
    # 10000 / 1e6 * 1.20 = 0.0120
    assert cost == Decimal("0.0120"), (
        f"calculate_cost on production PRICING_TABLE returned {cost}; "
        f"expected $0.0120 (= 10k output * $1.20/M). If this fails, "
        f"the production PRICING_TABLE['MiniMax-M3'] output rate is "
        f"NOT $1.20/M."
    )


def test_estimate_research_cost_on_production_pricing_table_uses_verified_rates() -> None:
    """estimate_research_cost on the production PRICING_TABLE uses the
    verified MiniMax-M3 output rate ($1.20/M)."""
    cost = estimate_research_cost(
        max_sources=5,
        per_source_max_tokens=3000,
        output_max_tokens=10_000,
        pricing_table=PRICING_TABLE,
        primary_model="MiniMax-M3",
    )
    # 5*3000*1.20/1e6 = 0.018 (per_source)
    # 10000*1.20/1e6 = 0.012 (final_synth)
    # (0.018 + 0.012) * 1.30 = 0.039
    assert cost == Decimal("0.0390"), (
        f"estimate_research_cost on production PRICING_TABLE returned "
        f"{cost}; expected $0.0390 for the S15 default config. If this "
        f"fails, the production PRICING_TABLE['MiniMax-M3'] output rate "
        f"is NOT $1.20/M."
    )


def test_cost_usd_is_decimal_with_4_decimals() -> None:
    """cost_usd is a Decimal quantized to 4 decimal places.

    This is the wire format invariant. Consumers of cost_usd can
    rely on at most 4 decimal places of precision.
    """
    cost = calculate_cost("MiniMax-M3", 1_234, 5_678)
    assert isinstance(cost, Decimal)
    # Exactly 4 decimal places
    _sign, _digits, exponent = cost.as_tuple()
    assert exponent == -4, (
        f"cost_usd must be quantized to 4 decimals; got exponent={exponent}, "
        f"value={cost}"
    )


def test_cost_usd_is_estimated_not_actual() -> None:
    """The cost module exposes the paygo-equivalent semantics via
    PRICING_BASIS. The function returns a Decimal; the SEMANTICS
    of that Decimal are explicit at module level.

    A subscription or quota-backed operator must not treat the
    returned value as a spend figure.
    """
    # Module-level declaration
    assert PRICING_BASIS == "official_paygo_equivalent"
    # Function returns a Decimal
    cost = calculate_cost("MiniMax-M3", 1_000, 500)
    assert isinstance(cost, Decimal)


def test_pricing_table_contains_all_budget_relevant_models() -> None:
    """All S15 pilot-corpus models are in PRICING_TABLE."""
    expected_models = {
        "MiniMax-M3",
        "MiniMax-M2.7-highspeed",
        "deepseek-v3",  # preserved from prior estimate; not verified
    }
    actual_models = set(PRICING_TABLE.keys())
    assert expected_models <= actual_models, (
        f"PRICING_TABLE missing models: {expected_models - actual_models}"
    )


def test_pricing_table_rates_are_all_positive_decimals() -> None:
    """Every rate in PRICING_TABLE is a positive Decimal."""
    for model, (in_rate, out_rate) in PRICING_TABLE.items():
        assert isinstance(in_rate, Decimal), f"{model}: in_rate not Decimal"
        assert isinstance(out_rate, Decimal), f"{model}: out_rate not Decimal"
        assert in_rate > 0, f"{model}: in_rate must be > 0; got {in_rate}"
        assert out_rate > 0, f"{model}: out_rate must be > 0; got {out_rate}"


def test_pricing_table_rates_are_not_legacy_estimates() -> None:
    """The DR-Q1A-PRE1A re-verified rates are NOT the legacy
    estimated rates. If a future change re-introduces the legacy
    estimated values, this test will fail.

    Legacy values (pre-PRE1A, in the original PRICING_TABLE):
      - MiniMax-M3 output: $0.60/M (legacy estimate, replaced with $1.20/M official)
      - MiniMax-M2.7-highspeed: $0.10/M in, $0.20/M out (legacy estimate,
        replaced with $0.60/M in, $2.40/M out official)
    """
    _in, out = PRICING_TABLE["MiniMax-M3"]
    assert out != Decimal("0.60"), (
        "MiniMax-M3 output rate regressed to legacy estimate $0.60/M. "
        "The verified rate is $1.20/M (2026-07-21). If the official "
        "source genuinely changed, update PRICING_AS_OF and this test."
    )
    in_m27, out_m27 = PRICING_TABLE["MiniMax-M2.7-highspeed"]
    assert (in_m27, out_m27) != (Decimal("0.10"), Decimal("0.20")), (
        "MiniMax-M2.7-highspeed rate regressed to legacy estimate. "
        "The verified rate is $0.60/M in, $2.40/M out (2026-07-21)."
    )

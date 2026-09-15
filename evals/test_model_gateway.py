"""
Component-level tests for model_gateway_decide -- the deterministic router
picking CHEAP_MODEL vs EXPENSIVE_MODEL. Called directly with hand-picked
inputs, no graph involved, so every boundary can be tested in isolation
instead of having to construct a whole claim that happens to land there.
"""
import sys

sys.path.insert(0, __file__.rsplit("/evals/", 1)[0])
import claims_auth_hybrid_rag_confidence_circuitbreaker as claims  # noqa: E402

SHORT_NARRATIVE = "x" * 50       # well under the 200-char signal
LONG_NARRATIVE = "x" * 201       # one over the 200-char signal


def _decide(amount, flags, similarity, narrative=SHORT_NARRATIVE):
    return claims.model_gateway_decide(amount, flags, similarity, narrative)


def test_clean_claim_routes_cheap_with_zero_score():
    d = _decide(amount=180.0, flags=[], similarity=0.47)
    assert d["tier"] == "cheap"
    assert d["complexity_score"] == 0
    assert d["selected_model"] == claims.CHEAP_MODEL


def test_ambiguous_similarity_band_boundaries():
    # just below the band: no signal
    assert _decide(amount=100.0, flags=[], similarity=0.549)["complexity_score"] == 0
    # entry of the band (inclusive): +3, crosses to expensive
    d = _decide(amount=100.0, flags=[], similarity=0.55)
    assert d["complexity_score"] == 3
    assert d["tier"] == "expensive"
    # just below the top of the band: still +3
    assert _decide(amount=100.0, flags=[], similarity=0.899)["complexity_score"] == 3
    # at/above the top of the band (exclusive): no signal -- a near-exact
    # match is easy to classify despite being high-stakes, per the
    # function's own docstring
    d = _decide(amount=100.0, flags=[], similarity=0.90)
    assert d["complexity_score"] == 0
    assert d["tier"] == "cheap"


def test_rule_flags_and_high_value_are_coupled_not_independent():
    """
    compute_rule_flags only ever sets a flag because claim_amount crossed
    10000 or 50000 -- both amount-driven -- so "rule flags present" (+2)
    and "high claim value" (+2) can never fire alone from a real claim:
    they fire together. This test pins that coupling down explicitly so a
    future change to compute_rule_flags' thresholds doesn't silently
    decouple them without anyone noticing the gateway's math assumed they
    moved together.
    """
    # rule flags present, amount also >10000 (the only way this actually happens)
    d = _decide(amount=10001.0, flags=["high_value_claim"], similarity=0.0)
    assert d["complexity_score"] == 4  # +2 (flags) + 2 (amount) -- not +2
    assert d["tier"] == "expensive"


def test_narrative_length_signal_never_flips_tier_alone():
    """
    The narrative-length signal (+1) can only matter if it pushes a
    complexity score from 2 to 3. But the other three signals only ever
    contribute 0, 3 (ambiguous similarity, alone) or 4 (rule flags + high
    value, which are coupled -- see the test above) -- a base score of
    exactly 2 is mathematically unreachable from this function's other
    inputs. So narrative length can move a score from 0->1 or 3->4 or
    4->5, but it can never be the deciding vote on which tier gets picked.
    This is a genuine property of the current threshold design, not a bug
    -- but it means the "long/detailed narrative" reason is, today, purely
    cosmetic. Documented here so it doesn't get relied on by accident.
    """
    for amount, flags, similarity in [
        (100.0, [], 0.0),        # base score 0
        (100.0, [], 0.70),       # base score 3 (already expensive)
        (10001.0, ["high_value_claim"], 0.0),  # base score 4 (already expensive)
    ]:
        short = _decide(amount, flags, similarity, SHORT_NARRATIVE)
        long_ = _decide(amount, flags, similarity, LONG_NARRATIVE)
        assert long_["complexity_score"] == short["complexity_score"] + 1
        assert long_["tier"] == short["tier"], (
            f"narrative length alone changed the tier for amount={amount} flags={flags} "
            f"similarity={similarity}: {short['tier']} -> {long_['tier']}"
        )


def test_high_claim_value_boundary():
    assert _decide(amount=10000.0, flags=[], similarity=0.0)["complexity_score"] == 0  # not > 10000
    assert _decide(amount=10000.01, flags=[], similarity=0.0)["complexity_score"] == 2  # > 10000, no rule flag passed here in isolation


def test_reasons_are_never_empty():
    d = _decide(amount=100.0, flags=[], similarity=0.0)
    assert d["reasons"], "a clean claim should still explain itself, not return an empty reasons list"
    assert "no complexity signals" in d["reasons"][0]

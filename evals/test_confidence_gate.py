"""
Component-level tests for confidence_decision_gate / route_after_confidence_gate --
the three-way auto-approve / auto-reject / human-review decision. Built directly
as ClaimState-shaped dicts, no graph invocation needed since these two functions
only read state, never call an external service.
"""
import sys

sys.path.insert(0, __file__.rsplit("/evals/", 1)[0])
import claims_auth_hybrid_rag_confidence_circuitbreaker as claims  # noqa: E402


def _state(action, confidence, top_sim=0.0, rule_flags=None):
    return {
        "llm_output": {"recommended_action": action, "rationale": "r", "confidence": confidence},
        "similar_cases": [{"id": "x", "dense_score": top_sim}] if top_sim else [],
        "rule_fraud_flags": rule_flags or [],
    }


def test_auto_approve_confidence_boundary():
    below = claims.route_after_confidence_gate(_state("approve", 0.849))
    at = claims.route_after_confidence_gate(_state("approve", 0.85))
    assert below == "notify_siu_zapier"  # falls to human review
    assert at == "approve_claim"


def test_auto_reject_requires_both_confidence_and_similarity():
    # high confidence, similarity just under the near-exact-match bar: human review, not auto-reject
    r1 = claims.route_after_confidence_gate(_state("flag_for_siu", 0.99, top_sim=0.969))
    assert r1 == "notify_siu_zapier"
    # high confidence AND similarity over the bar: auto-reject
    r2 = claims.route_after_confidence_gate(_state("flag_for_siu", 0.99, top_sim=0.97))
    assert r2 == "auto_reject_claim"
    # similarity satisfied but confidence just under its own bar: human review
    r3 = claims.route_after_confidence_gate(_state("flag_for_siu", 0.949, top_sim=0.99))
    assert r3 == "notify_siu_zapier"
    # both bars exactly met
    r4 = claims.route_after_confidence_gate(_state("flag_for_siu", 0.95, top_sim=0.97))
    assert r4 == "auto_reject_claim"


def test_flag_for_siu_with_no_similar_cases_is_human_review_not_auto_reject():
    """
    top_sim defaults to 0.0 when similar_cases is empty (e.g. a claim that
    got flagged by the billing specialist alone, with no FWA-case match at
    all) -- confirms that path can never accidentally satisfy
    AUTO_REJECT_MIN_SIMILARITY by falling through to some other default.
    """
    r = claims.route_after_confidence_gate(_state("flag_for_siu", 0.99, top_sim=0.0))
    assert r == "notify_siu_zapier"


def test_confidence_decision_gate_appends_flags_matching_the_route():
    approve_result = claims.confidence_decision_gate(_state("approve", 0.9))
    assert approve_result["fraud_flags"] == []  # nothing appended on a clean approve

    reject_result = claims.confidence_decision_gate(_state("flag_for_siu", 0.99, top_sim=0.99))
    assert any("llm_high_confidence_violation" in f for f in reject_result["fraud_flags"])

    review_result = claims.confidence_decision_gate(_state("flag_for_siu", 0.6, top_sim=0.6))
    assert any("llm_recommended_review" in f for f in review_result["fraud_flags"])


def test_confidence_gate_thresholds_match_documented_constants():
    """Pins the actual constant values -- fails loudly if someone recalibrates them without updating this suite."""
    assert claims.AUTO_APPROVE_CONFIDENCE == 0.85
    assert claims.AUTO_REJECT_CONFIDENCE == 0.95
    assert claims.AUTO_REJECT_MIN_SIMILARITY == 0.97

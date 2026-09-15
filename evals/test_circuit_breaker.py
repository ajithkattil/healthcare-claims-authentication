"""
Component-level tests for the circuit breaker's retry-then-trip mechanics,
calling check_similar_fraud_cases_hybrid and route_after_similarity directly
across simulated attempts. test_e2e_golden.py's circuit_breaker_trip case
covers the same mechanism through the full graph; this file isolates the
retry counting itself so a change to MAX_TOOL_ERRORS's meaning (e.g.
off-by-one) fails here first, with a much shorter trace to read.
"""
import sys

sys.path.insert(0, __file__.rsplit("/evals/", 1)[0])
import claims_auth_hybrid_rag_confidence_circuitbreaker as claims  # noqa: E402

FAILING_THREAD = "claim-K"  # the module-level _FORCE_TOOL_FAILURE_THREADS trigger
OK_THREAD = "claim-not-failing"


def _config(thread_id):
    return {"configurable": {"thread_id": thread_id}}


def test_max_tool_errors_constant_is_two():
    """Pins the constant this whole test file's math depends on."""
    assert claims.MAX_TOOL_ERRORS == 2


def test_first_failure_retries_not_escalates():
    state = {"embedding": [0.0] * 8, "narrative": "x", "tool_error_count": 0}
    result = claims.check_similar_fraud_cases_hybrid(state, _config(FAILING_THREAD))
    assert result["tool_call_failed"] is True
    assert result["tool_error_count"] == 1

    route = claims.route_after_similarity({**state, **result})
    assert route == "check_similar_fraud_cases_hybrid"  # retry, not escalate


def test_second_consecutive_failure_escalates():
    state = {"embedding": [0.0] * 8, "narrative": "x", "tool_error_count": 1}
    result = claims.check_similar_fraud_cases_hybrid(state, _config(FAILING_THREAD))
    assert result["tool_error_count"] == 2

    route = claims.route_after_similarity({**state, **result})
    assert route == "circuit_breaker_escalate"


def test_circuit_breaker_escalate_preserves_existing_rule_flags():
    state = {
        "rule_fraud_flags": ["high_value_claim"],
        "tool_error_count": 2,
        "tool_error_reason": "simulated Pinecone outage for circuit-breaker demonstration",
    }
    result = claims.circuit_breaker_escalate(state)
    assert result["circuit_breaker_tripped"] is True
    assert "high_value_claim" in result["fraud_flags"]
    assert any(f.startswith("circuit_breaker_tripped:") for f in result["fraud_flags"])


def test_non_failing_thread_succeeds_and_never_retries():
    state = {
        "embedding": claims.cohere_embed("routine clean narrative"),
        "narrative": "routine clean narrative",
        "tool_error_count": 0,
    }
    result = claims.check_similar_fraud_cases_hybrid(state, _config(OK_THREAD))
    assert result["tool_call_failed"] is False
    assert "similar_cases" in result

    route = claims.route_after_similarity({**state, **result})
    assert route == "retrieve_guidelines"

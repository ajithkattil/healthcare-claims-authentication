"""
End-to-end regression tests: drive the full compiled graph over the golden
claim dataset and assert the observable outcome at every layer -- model
tier, pause/no-pause, and (when finalized) the exact decision string.

This is the automated replacement for the manual "Testing checklist" in
README.md: same claims, same expectations, now asserted by pytest instead
of eyeballed in printed output.
"""
import sys

import pytest

sys.path.insert(0, __file__.rsplit("/evals/", 1)[0])
import claims_auth_hybrid_rag_confidence_circuitbreaker as claims  # noqa: E402

from conftest import invoke_claim  # noqa: E402
from golden_claims import GOLDEN_CLAIMS  # noqa: E402


def _resolve_narrative(case: dict) -> str:
    if case.get("narrative_fwa_case"):
        return claims._FWA_CASES[case["narrative_fwa_case"]]
    return case["narrative"]


@pytest.mark.parametrize("case", GOLDEN_CLAIMS, ids=[c["name"] for c in GOLDEN_CLAIMS])
def test_golden_claim(graph, case):
    narrative = _resolve_narrative(case)
    result, snapshot = invoke_claim(
        graph,
        claim_id=f"CLM-EVAL-{case['name'].upper()[:20]}",
        patient_id=case["patient_id"],
        claim_amount=case["claim_amount"],
        narrative=narrative,
        thread_id=case.get("thread_id"),
    )
    paused = bool(snapshot.next)

    # --- model tier ---
    expect_tier = case.get("expect_tier", "__unspecified__")
    if expect_tier != "__unspecified__":
        gw = result.get("model_gateway_decision")
        if expect_tier is None:
            assert gw is None, f"{case['name']}: expected the model gateway to never run, but it did: {gw}"
        else:
            assert gw is not None, f"{case['name']}: expected model gateway tier {expect_tier!r}, but it never ran"
            assert gw["tier"] == expect_tier, (
                f"{case['name']}: expected tier {expect_tier!r}, got {gw['tier']!r} "
                f"(complexity_score={gw['complexity_score']}, reasons={gw['reasons']})"
            )

    # --- pause state ---
    stage = case["expect_stage"]
    if stage in ("identity_reject", "guardrail_reject", "auto_approve", "auto_reject"):
        assert not paused, f"{case['name']}: expected a terminal decision with no pause, but the graph paused"
    elif stage in ("human_review", "circuit_breaker"):
        assert paused, f"{case['name']}: expected the graph to pause for SIU review, but it finished with: {result.get('decision')}"

    # --- decision string (only meaningful for non-paused terminal cases) ---
    if "expect_decision" in case:
        assert result.get("decision") == case["expect_decision"]
    if "expect_decision_contains" in case:
        decision = result.get("decision") or ""
        reason = result.get("input_guardrail_reason") or ""
        assert case["expect_decision_contains"] in decision or case["expect_decision_contains"] in reason, (
            f"{case['name']}: expected {case['expect_decision_contains']!r} in decision/reason, "
            f"got decision={decision!r} reason={reason!r}"
        )

    # --- circuit breaker specifics ---
    if stage == "circuit_breaker":
        assert result.get("circuit_breaker_tripped") is True
        assert any(f.startswith("circuit_breaker_tripped:") for f in result.get("fraud_flags", []))
        assert result.get("model_gateway_decision") is None, "circuit breaker should short-circuit before the gateway"
        assert result.get("billing_finding") is None, "circuit breaker should short-circuit before either specialist"
        assert result.get("narrative_finding") is None


def test_golden_dataset_names_are_unique():
    names = [c["name"] for c in GOLDEN_CLAIMS]
    assert len(names) == len(set(names)), "duplicate golden claim name -- pytest -k filtering would be ambiguous"


def test_deleting_checkpoint_produces_identical_output(tmp_path):
    """
    The manual checklist's "deleting the checkpoint file between runs produces
    identical output" item, automated: run the same claim against two
    independently-built graphs (simulating two fresh checkpoint files) and
    confirm the decision, tier, and pause state are identical.
    """
    from langgraph.checkpoint.sqlite import SqliteSaver

    claims._cache_init()
    outcomes = []
    for i in range(2):
        db_path = tmp_path / f"run_{i}.sqlite"
        with SqliteSaver.from_conn_string(str(db_path)) as checkpointer:
            g = claims.build_graph(checkpointer)
            result, snapshot = invoke_claim(
                g, claim_id="CLM-REPEAT", patient_id="PAT-1010", claim_amount=180.0,
                narrative="Routine annual wellness visit, CPT 99395, no complications, "
                          "consistent with patient's prior visit history.",
            )
            outcomes.append((result.get("decision"), bool(snapshot.next), result.get("model_gateway_decision", {}).get("tier")))

    assert outcomes[0] == outcomes[1], f"non-deterministic output across independent checkpoint files: {outcomes}"

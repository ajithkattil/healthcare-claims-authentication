"""
LangGraph POC #1 -- Healthcare Claims Authentication (Basic)

Pipeline:
  verify_patient_identity --[route]--> reject_claim (END)
                                  (or)--> check_insurance_coverage
                                            -> detect_fraud_signals --[route]--> approve_claim (END)
                                                                            (or)--> route_to_siu_review (HUMAN GATE, END)

SIU = Special Investigations Unit, the standard term for the team that
reviews claims flagged as potentially fraudulent or abusive in health
insurance operations.

`route_to_siu_review` is registered with interrupt_before, so when a claim
gets flagged, the graph run STOPS before that node executes. Execution is
suspended (checkpointed), control returns to your application code, a human
SIU reviewer inspects the state, injects a decision via graph.update_state(),
and only then does the run resume and the node execute with that human
input in hand.

Note on PHI: `patient_id` here is a synthetic identifier, not a real medical
record number or other direct identifier. In a real deployment touching
actual patient data, treat every field in ClaimState as Protected Health
Information (PHI) under HIPAA -- encrypt at rest and in transit, restrict
access by role, and avoid writing raw PHI into logs (the `log` field below
is illustrative only; a production version should log claim/decision IDs,
not clinical narrative text).

Run: python3 01_claims_auth_basic.py
"""

import operator
from typing import TypedDict, Annotated, Optional

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver


# ---------------------------------------------------------------------------
# 1. STATE SCHEMA
# ---------------------------------------------------------------------------
class ClaimState(TypedDict):
    claim_id: str
    patient_id: str
    claim_amount: float
    identity_verified: bool
    coverage_valid: bool
    fraud_flags: list[str]
    siu_decision: Optional[str]   # set by a human reviewer mid-flight
    decision: str
    log: Annotated[list[str], operator.add]


# ---------------------------------------------------------------------------
# 2. NODES  (all mock/deterministic logic, no LLM calls -- fast + repeatable)
# ---------------------------------------------------------------------------
def verify_patient_identity(state: ClaimState) -> dict:
    print("  -> [verify_patient_identity] executing")
    # Mock rule: patient_id must match the health plan's member ID format.
    verified = state["patient_id"].startswith("PAT-")
    return {
        "identity_verified": verified,
        "log": [f"verify_patient_identity: verified={verified}"],
    }


def route_after_identity(state: ClaimState) -> str:
    return "check_insurance_coverage" if state["identity_verified"] else "reject_claim"


def reject_claim(state: ClaimState) -> dict:
    print("  -> [reject_claim] executing")
    return {
        "decision": "REJECTED: patient identity/member verification failed",
        "log": ["reject_claim: terminal"],
    }


def check_insurance_coverage(state: ClaimState) -> dict:
    print("  -> [check_insurance_coverage] executing")
    # Mock rule: claim is within plan coverage unless amount is unusually high.
    valid = state["claim_amount"] <= 50000
    return {
        "coverage_valid": valid,
        "log": [f"check_insurance_coverage: coverage_valid={valid}"],
    }


def detect_fraud_signals(state: ClaimState) -> dict:
    print("  -> [detect_fraud_signals] executing")
    flags = []
    if not state["coverage_valid"]:
        flags.append("amount_exceeds_coverage_limit")
    if state["claim_amount"] > 10000:
        flags.append("high_value_claim")
    return {
        "fraud_flags": flags,
        "log": [f"detect_fraud_signals: flags={flags}"],
    }


def route_after_fraud(state: ClaimState) -> str:
    return "route_to_siu_review" if state["fraud_flags"] else "approve_claim"


def approve_claim(state: ClaimState) -> dict:
    print("  -> [approve_claim] executing")
    return {
        "decision": "APPROVED: auto-approved, no fraud/abuse signals",
        "log": ["approve_claim: terminal"],
    }


def route_to_siu_review(state: ClaimState) -> dict:
    # By the time this node actually runs, the graph already paused BEFORE
    # it (interrupt_before) and a human SIU reviewer has had the chance to
    # inject `siu_decision` via update_state(). This node just finalizes.
    print("  -> [route_to_siu_review] executing (post-human)")
    human_call = state.get("siu_decision") or "PENDING"
    return {
        "decision": f"SIU REVIEW: {human_call}",
        "log": [f"route_to_siu_review: finalized as {human_call}"],
    }


# ---------------------------------------------------------------------------
# 3. BUILD THE GRAPH
# ---------------------------------------------------------------------------
def build_graph(checkpointer):
    g = StateGraph(ClaimState)

    g.add_node("verify_patient_identity", verify_patient_identity)
    g.add_node("reject_claim", reject_claim)
    g.add_node("check_insurance_coverage", check_insurance_coverage)
    g.add_node("detect_fraud_signals", detect_fraud_signals)
    g.add_node("approve_claim", approve_claim)
    g.add_node("route_to_siu_review", route_to_siu_review)

    g.add_edge(START, "verify_patient_identity")
    g.add_conditional_edges(
        "verify_patient_identity",
        route_after_identity,
        {"check_insurance_coverage": "check_insurance_coverage", "reject_claim": "reject_claim"},
    )
    g.add_edge("reject_claim", END)
    g.add_edge("check_insurance_coverage", "detect_fraud_signals")
    g.add_conditional_edges(
        "detect_fraud_signals",
        route_after_fraud,
        {"approve_claim": "approve_claim", "route_to_siu_review": "route_to_siu_review"},
    )
    g.add_edge("approve_claim", END)
    g.add_edge("route_to_siu_review", END)

    # THE HUMAN GATE: pause execution right before this node runs.
    return g.compile(checkpointer=checkpointer, interrupt_before=["route_to_siu_review"])


# ---------------------------------------------------------------------------
# 4. RUN IT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    with SqliteSaver.from_conn_string("claims_checkpoints.sqlite") as checkpointer:
        graph = build_graph(checkpointer)

        # --- Scenario A: clean claim, no human needed ---------------------
        print("\n=== CLAIM A: clean, low-value -- expect auto-approve ===")
        config_a = {"configurable": {"thread_id": "claim-A"}}
        result_a = graph.invoke(
            {
                "claim_id": "CLM-1001", "patient_id": "PAT-7781", "claim_amount": 2500.0,
                "log": [], "fraud_flags": [], "siu_decision": None,
            },
            config=config_a,
        )
        print("Final decision:", result_a["decision"])

        # --- Scenario B: bad identity, rejected immediately ----------------
        print("\n=== CLAIM B: unrecognized patient id -- expect reject ===")
        config_b = {"configurable": {"thread_id": "claim-B"}}
        result_b = graph.invoke(
            {
                "claim_id": "CLM-1002", "patient_id": "UNKNOWN-99", "claim_amount": 500.0,
                "log": [], "fraud_flags": [], "siu_decision": None,
            },
            config=config_b,
        )
        print("Final decision:", result_b["decision"])

        # --- Scenario C: high-value, flagged -- HUMAN GATE fires ----------
        print("\n=== CLAIM C: high-value -- expect PAUSE for SIU review ===")
        config_c = {"configurable": {"thread_id": "claim-C"}}
        result_c = graph.invoke(
            {
                "claim_id": "CLM-1003", "patient_id": "PAT-4412", "claim_amount": 18000.0,
                "log": [], "fraud_flags": [], "siu_decision": None,
            },
            config=config_c,
        )
        snapshot = graph.get_state(config_c)
        print("Graph is paused. Next node queued:", snapshot.next)
        print("State at pause:", {k: v for k, v in snapshot.values.items() if k != "log"})
        print("(In a real app, this is where you'd surface a review UI to an SIU reviewer.)")

        # Simulate a human making the call, then injecting it into state.
        print("\n--- SIU reviewer reviews claim-C and decides: APPROVE ---")
        graph.update_state(config_c, {"siu_decision": "APPROVED_BY_SIU"})

        # Resume -- route_to_siu_review now runs with the human's input available.
        result_c_final = graph.invoke(None, config=config_c)
        print("Final decision:", result_c_final["decision"])

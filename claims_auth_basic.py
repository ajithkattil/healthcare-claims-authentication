"""
LangGraph POC #2 -- Claims Authentication

Pipeline:
  verify_identity --[route]--> reject_claim (END)
                          (or)--> check_policy_coverage
                                    -> detect_fraud_signals --[route]--> approve_claim (END)
                                                                    (or)--> route_to_investigator (HUMAN GATE, END)

New concept vs POC #1: a genuine human-in-the-loop interrupt.
`route_to_investigator` is registered with interrupt_before, so when a claim
gets flagged, the graph run STOPS before that node executes. Execution is
suspended (checkpointed), control returns to your application code, a human
reviews the state, injects a decision via graph.update_state(), and only
then does the run resume and the node execute with that human input in hand.

Run: python3 claims_auth_poc.py
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
    claimant_id: str
    claim_amount: float
    identity_verified: bool
    policy_valid: bool
    fraud_flags: list[str]
    investigator_decision: Optional[str]   # set by a human mid-flight
    decision: str
    log: Annotated[list[str], operator.add]


# ---------------------------------------------------------------------------
# 2. NODES  (all mock/deterministic logic, no LLM calls -- fast + repeatable)
# ---------------------------------------------------------------------------
def verify_identity(state: ClaimState) -> dict:
    print("  -> [verify_identity] executing")
    # Mock rule: claimant_id must look like a real internal ID (starts "CUST-")
    verified = state["claimant_id"].startswith("CUST-")
    return {
        "identity_verified": verified,
        "log": [f"verify_identity: verified={verified}"],
    }


def route_after_identity(state: ClaimState) -> str:
    return "check_policy_coverage" if state["identity_verified"] else "reject_claim"


def reject_claim(state: ClaimState) -> dict:
    print("  -> [reject_claim] executing")
    return {
        "decision": "REJECTED: identity verification failed",
        "log": ["reject_claim: terminal"],
    }


def check_policy_coverage(state: ClaimState) -> dict:
    print("  -> [check_policy_coverage] executing")
    # Mock rule: any claim is "covered" unless amount is absurdly high
    valid = state["claim_amount"] <= 50000
    return {
        "policy_valid": valid,
        "log": [f"check_policy_coverage: policy_valid={valid}"],
    }


def detect_fraud_signals(state: ClaimState) -> dict:
    print("  -> [detect_fraud_signals] executing")
    flags = []
    if not state["policy_valid"]:
        flags.append("amount_exceeds_policy_limit")
    if state["claim_amount"] > 10000:
        flags.append("high_value_claim")
    return {
        "fraud_flags": flags,
        "log": [f"detect_fraud_signals: flags={flags}"],
    }


def route_after_fraud(state: ClaimState) -> str:
    return "route_to_investigator" if state["fraud_flags"] else "approve_claim"


def approve_claim(state: ClaimState) -> dict:
    print("  -> [approve_claim] executing")
    return {
        "decision": "APPROVED: auto-approved, no fraud signals",
        "log": ["approve_claim: terminal"],
    }


def route_to_investigator(state: ClaimState) -> dict:
    # By the time this node actually runs, the graph already paused BEFORE
    # it (interrupt_before) and a human has had the chance to inject
    # `investigator_decision` via update_state(). This node just finalizes.
    print("  -> [route_to_investigator] executing (post-human)")
    human_call = state.get("investigator_decision") or "PENDING"
    return {
        "decision": f"INVESTIGATOR REVIEW: {human_call}",
        "log": [f"route_to_investigator: finalized as {human_call}"],
    }


# ---------------------------------------------------------------------------
# 3. BUILD THE GRAPH
# ---------------------------------------------------------------------------
def build_graph(checkpointer):
    g = StateGraph(ClaimState)

    g.add_node("verify_identity", verify_identity)
    g.add_node("reject_claim", reject_claim)
    g.add_node("check_policy_coverage", check_policy_coverage)
    g.add_node("detect_fraud_signals", detect_fraud_signals)
    g.add_node("approve_claim", approve_claim)
    g.add_node("route_to_investigator", route_to_investigator)

    g.add_edge(START, "verify_identity")
    g.add_conditional_edges(
        "verify_identity",
        route_after_identity,
        {"check_policy_coverage": "check_policy_coverage", "reject_claim": "reject_claim"},
    )
    g.add_edge("reject_claim", END)
    g.add_edge("check_policy_coverage", "detect_fraud_signals")
    g.add_conditional_edges(
        "detect_fraud_signals",
        route_after_fraud,
        {"approve_claim": "approve_claim", "route_to_investigator": "route_to_investigator"},
    )
    g.add_edge("approve_claim", END)
    g.add_edge("route_to_investigator", END)

    # THE HUMAN GATE: pause execution right before this node runs.
    return g.compile(checkpointer=checkpointer, interrupt_before=["route_to_investigator"])


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
                "claim_id": "CLM-1001",
                "claimant_id": "CUST-778",
                "claim_amount": 2500.0,
                "log": [],
                "fraud_flags": [],
                "investigator_decision": None,
            },
            config=config_a,
        )
        print("Final decision:", result_a["decision"])

        # --- Scenario B: bad identity, rejected immediately ----------------
        print("\n=== CLAIM B: bad claimant id -- expect reject ===")
        config_b = {"configurable": {"thread_id": "claim-B"}}
        result_b = graph.invoke(
            {
                "claim_id": "CLM-1002",
                "claimant_id": "UNKNOWN-99",
                "claim_amount": 500.0,
                "log": [],
                "fraud_flags": [],
                "investigator_decision": None,
            },
            config=config_b,
        )
        print("Final decision:", result_b["decision"])

        # --- Scenario C: high-value, flagged -- HUMAN GATE fires ----------
        print("\n=== CLAIM C: high-value -- expect PAUSE for human review ===")
        config_c = {"configurable": {"thread_id": "claim-C"}}
        result_c = graph.invoke(
            {
                "claim_id": "CLM-1003",
                "claimant_id": "CUST-441",
                "claim_amount": 18000.0,
                "log": [],
                "fraud_flags": [],
                "investigator_decision": None,
            },
            config=config_c,
        )
        snapshot = graph.get_state(config_c)
        print("Graph is paused. Next node queued:", snapshot.next)
        print("State at pause:", {k: v for k, v in snapshot.values.items() if k != "log"})
        print("(In a real app, this is where you'd surface a review UI to a human.)")

        # Simulate a human making the call, then injecting it into state.
        print("\n--- Human reviews claim-C and decides: APPROVE ---")
        graph.update_state(config_c, {"investigator_decision": "APPROVED_BY_INVESTIGATOR"})

        # Resume -- route_to_investigator now runs with the human's input available.
        result_c_final = graph.invoke(None, config=config_c)
        print("Final decision:", result_c_final["decision"])

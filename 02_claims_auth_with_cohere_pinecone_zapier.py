"""
LangGraph POC #3 -- Claims Authentication + Cohere + Pinecone + Zapier

Adds to POC #2's graph:

  ... check_policy_coverage
        -> embed_claim_narrative        (Cohere: text -> vector)
        -> check_similar_fraud_cases    (Pinecone: vector similarity search)
        -> detect_fraud_signals         (merge rule-based + similarity flags)
        --[route]--> approve_claim (END)
                (or)--> notify_investigator_zapier (Zapier: webhook/notification)
                          -> route_to_investigator (HUMAN GATE, END)

INTEGRATION MODE
-----------------
Set USE_LIVE_APIS = True and provide these env vars to hit the real services:
  COHERE_API_KEY
  PINECONE_API_KEY, PINECONE_INDEX_NAME
  ZAPIER_WEBHOOK_URL   (a "Catch Hook" trigger URL from a Zap)

This sandbox's network egress does not include api.cohere.ai, api.pinecone.io,
or hooks.zapier.com, so USE_LIVE_APIS defaults to False here and every call
is routed through a mock that mirrors the real SDK's return shape. The graph
logic, state schema, and node wiring are identical either way -- only the
three integration functions at the top change behavior when you flip the
flag and run this locally.

Run: python3 claims_auth_poc_v2.py
"""

import os
import math
import hashlib
import operator
from typing import TypedDict, Annotated, Optional

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

USE_LIVE_APIS = False  # flip to True + set env vars below to hit real services

COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "claims-fraud-cases")
ZAPIER_WEBHOOK_URL = os.environ.get("ZAPIER_WEBHOOK_URL")


# ---------------------------------------------------------------------------
# INTEGRATION LAYER -- real SDK calls when USE_LIVE_APIS=True, deterministic
# mocks otherwise. Node functions below call these and never know the
# difference.
# ---------------------------------------------------------------------------
def cohere_embed(text: str) -> list[float]:
    if USE_LIVE_APIS:
        import cohere
        client = cohere.Client(COHERE_API_KEY)
        resp = client.embed(texts=[text], model="embed-english-v3.0", input_type="search_document")
        return resp.embeddings[0]

    # Mock: deterministic pseudo-embedding derived from a hash of the text,
    # so the same narrative always produces the same vector (needed for the
    # similarity search below to behave consistently across runs).
    h = hashlib.sha256(text.encode()).digest()
    return [b / 255.0 for b in h[:16]]  # 16-dim toy vector


def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


# Mock "index" of embeddings for previously confirmed fraudulent claims.
# In live mode this data would already live in your Pinecone index;
# here we embed a couple of canonical fraud narratives once, at import time.
_MOCK_FRAUD_CASES = {
    "past-fraud-001": cohere_embed(
        "Claimant reported total loss of vehicle two days after policy start, "
        "no police report filed, inconsistent mileage on record."
    ),
    "past-fraud-002": cohere_embed(
        "Multiple claims filed under different claimant IDs from the same address "
        "within a 30 day window, all citing water damage."
    ),
}


def pinecone_query(embedding: list[float], top_k: int = 3) -> list[dict]:
    if USE_LIVE_APIS:
        from pinecone import Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)
        resp = index.query(vector=embedding, top_k=top_k, include_metadata=True)
        return [{"id": m.id, "score": m.score} for m in resp.matches]

    # Mock: brute-force cosine similarity against the in-memory fraud cases.
    scored = [
        {"id": case_id, "score": round(_cosine_sim(embedding, vec), 4)}
        for case_id, vec in _MOCK_FRAUD_CASES.items()
    ]
    return sorted(scored, key=lambda m: m["score"], reverse=True)[:top_k]


def zapier_notify(payload: dict) -> dict:
    if USE_LIVE_APIS:
        import requests
        resp = requests.post(ZAPIER_WEBHOOK_URL, json=payload, timeout=10)
        return {"status_code": resp.status_code}

    # Mock: just show what would have been sent.
    print(f"     [zapier mock] would POST to webhook: {payload}")
    return {"status_code": 200, "mock": True}


# ---------------------------------------------------------------------------
# 1. STATE SCHEMA
# ---------------------------------------------------------------------------
class ClaimState(TypedDict):
    claim_id: str
    claimant_id: str
    claim_amount: float
    narrative: str
    identity_verified: bool
    policy_valid: bool
    embedding: list[float]
    similar_cases: list[dict]
    fraud_flags: list[str]
    investigator_decision: Optional[str]
    decision: str
    log: Annotated[list[str], operator.add]


# ---------------------------------------------------------------------------
# 2. NODES
# ---------------------------------------------------------------------------
def verify_identity(state: ClaimState) -> dict:
    print("  -> [verify_identity] executing")
    verified = state["claimant_id"].startswith("CUST-")
    return {"identity_verified": verified, "log": [f"verify_identity: verified={verified}"]}


def route_after_identity(state: ClaimState) -> str:
    return "check_policy_coverage" if state["identity_verified"] else "reject_claim"


def reject_claim(state: ClaimState) -> dict:
    print("  -> [reject_claim] executing")
    return {"decision": "REJECTED: identity verification failed", "log": ["reject_claim: terminal"]}


def check_policy_coverage(state: ClaimState) -> dict:
    print("  -> [check_policy_coverage] executing")
    valid = state["claim_amount"] <= 50000
    return {"policy_valid": valid, "log": [f"check_policy_coverage: policy_valid={valid}"]}


def embed_claim_narrative(state: ClaimState) -> dict:
    print("  -> [embed_claim_narrative] executing (Cohere)")
    vec = cohere_embed(state["narrative"])
    return {"embedding": vec, "log": ["embed_claim_narrative: embedded via Cohere"]}


def check_similar_fraud_cases(state: ClaimState) -> dict:
    print("  -> [check_similar_fraud_cases] executing (Pinecone)")
    matches = pinecone_query(state["embedding"], top_k=2)
    return {"similar_cases": matches, "log": [f"check_similar_fraud_cases: top match={matches[0] if matches else None}"]}


def detect_fraud_signals(state: ClaimState) -> dict:
    print("  -> [detect_fraud_signals] executing")
    flags = []
    if not state["policy_valid"]:
        flags.append("amount_exceeds_policy_limit")
    if state["claim_amount"] > 10000:
        flags.append("high_value_claim")

    SIMILARITY_THRESHOLD = 0.85
    top = state["similar_cases"][0] if state["similar_cases"] else None
    if top and top["score"] >= SIMILARITY_THRESHOLD:
        flags.append(f"similar_to_known_fraud:{top['id']}(score={top['score']})")

    return {"fraud_flags": flags, "log": [f"detect_fraud_signals: flags={flags}"]}


def route_after_fraud(state: ClaimState) -> str:
    return "notify_investigator_zapier" if state["fraud_flags"] else "approve_claim"


def approve_claim(state: ClaimState) -> dict:
    print("  -> [approve_claim] executing")
    return {"decision": "APPROVED: auto-approved, no fraud signals", "log": ["approve_claim: terminal"]}


def notify_investigator_zapier(state: ClaimState) -> dict:
    print("  -> [notify_investigator_zapier] executing (Zapier)")
    result = zapier_notify({
        "claim_id": state["claim_id"],
        "claimant_id": state["claimant_id"],
        "fraud_flags": state["fraud_flags"],
        "message": "Claim flagged for investigator review",
    })
    return {"log": [f"notify_investigator_zapier: webhook result={result}"]}


def route_to_investigator(state: ClaimState) -> dict:
    print("  -> [route_to_investigator] executing (post-human)")
    human_call = state.get("investigator_decision") or "PENDING"
    return {"decision": f"INVESTIGATOR REVIEW: {human_call}", "log": [f"route_to_investigator: finalized as {human_call}"]}


# ---------------------------------------------------------------------------
# 3. BUILD THE GRAPH
# ---------------------------------------------------------------------------
def build_graph(checkpointer):
    g = StateGraph(ClaimState)

    for name, fn in [
        ("verify_identity", verify_identity),
        ("reject_claim", reject_claim),
        ("check_policy_coverage", check_policy_coverage),
        ("embed_claim_narrative", embed_claim_narrative),
        ("check_similar_fraud_cases", check_similar_fraud_cases),
        ("detect_fraud_signals", detect_fraud_signals),
        ("approve_claim", approve_claim),
        ("notify_investigator_zapier", notify_investigator_zapier),
        ("route_to_investigator", route_to_investigator),
    ]:
        g.add_node(name, fn)

    g.add_edge(START, "verify_identity")
    g.add_conditional_edges(
        "verify_identity", route_after_identity,
        {"check_policy_coverage": "check_policy_coverage", "reject_claim": "reject_claim"},
    )
    g.add_edge("reject_claim", END)
    g.add_edge("check_policy_coverage", "embed_claim_narrative")
    g.add_edge("embed_claim_narrative", "check_similar_fraud_cases")
    g.add_edge("check_similar_fraud_cases", "detect_fraud_signals")
    g.add_conditional_edges(
        "detect_fraud_signals", route_after_fraud,
        {"approve_claim": "approve_claim", "notify_investigator_zapier": "notify_investigator_zapier"},
    )
    g.add_edge("approve_claim", END)
    g.add_edge("notify_investigator_zapier", "route_to_investigator")
    g.add_edge("route_to_investigator", END)

    # Pause AFTER notifying the investigator (so they've already been
    # pinged) but BEFORE finalizing the decision.
    return g.compile(checkpointer=checkpointer, interrupt_before=["route_to_investigator"])


# ---------------------------------------------------------------------------
# 4. RUN IT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"USE_LIVE_APIS = {USE_LIVE_APIS} (mock mode)" if not USE_LIVE_APIS else "LIVE mode -- calling real APIs")

    with SqliteSaver.from_conn_string("claims_v2_checkpoints.sqlite") as checkpointer:
        graph = build_graph(checkpointer)

        print("\n=== CLAIM D: clean, low-value -- expect auto-approve ===")
        config_d = {"configurable": {"thread_id": "claim-D"}}
        result_d = graph.invoke(
            {
                "claim_id": "CLM-2001", "claimant_id": "CUST-101", "claim_amount": 1800.0,
                "narrative": "Minor fender bender in a parking lot, other driver at fault, police report attached.",
                "log": [], "fraud_flags": [], "investigator_decision": None,
            },
            config=config_d,
        )
        print("Final decision:", result_d["decision"])

        print("\n=== CLAIM E: narrative similar to known fraud case -- expect flag + pause ===")
        config_e = {"configurable": {"thread_id": "claim-E"}}
        result_e = graph.invoke(
            {
                "claim_id": "CLM-2002", "claimant_id": "CUST-205", "claim_amount": 4200.0,
                # Deliberately close wording to past-fraud-001 so the mock
                # cosine similarity trips the threshold.
                "narrative": "Claimant reported total loss of vehicle two days after policy start, "
                             "no police report filed, inconsistent mileage on record.",
                "log": [], "fraud_flags": [], "investigator_decision": None,
            },
            config=config_e,
        )
        snapshot = graph.get_state(config_e)
        print("Graph paused. Next node:", snapshot.next)
        print("Similar cases found:", snapshot.values["similar_cases"])
        print("Fraud flags:", snapshot.values["fraud_flags"])

        print("\n--- Human reviews claim-E and decides: DENY ---")
        graph.update_state(config_e, {"investigator_decision": "DENIED_BY_INVESTIGATOR"})
        result_e_final = graph.invoke(None, config=config_e)
        print("Final decision:", result_e_final["decision"])

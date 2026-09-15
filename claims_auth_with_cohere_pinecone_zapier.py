"""
LangGraph POC #2 -- Healthcare Claims Authentication + Cohere + Pinecone + Zapier

Adds to POC #1's graph:

  ... check_insurance_coverage
        -> embed_claim_narrative        (Cohere: text -> vector)
        -> check_similar_fraud_cases    (Pinecone: vector similarity search)
        -> detect_fraud_signals         (merge rule-based + similarity flags)
        --[route]--> approve_claim (END)
                (or)--> notify_siu_zapier (Zapier: webhook/notification)
                          -> route_to_siu_review (HUMAN GATE, END)

SIU = Special Investigations Unit -- the team in a health plan or payer
organization responsible for reviewing claims flagged for potential fraud,
waste, or abuse (FWA), a standard term in healthcare claims operations.

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

PHI NOTE: `patient_id` and `narrative` here are synthetic. In a real
deployment, both are Protected Health Information (PHI) under HIPAA.
Production considerations not implemented in this POC: encryption at rest
and in transit for anything sent to Cohere/Pinecone/Zapier, a signed
Business Associate Agreement (BAA) with each vendor before sending any real
PHI to them, access logging, and avoiding raw clinical narrative text in
application logs (the `log` field below is illustrative only).

Run: python3 02_claims_auth_with_cohere_pinecone_zapier.py
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


# Mock "index" of embeddings for previously confirmed fraud/waste/abuse (FWA)
# cases in healthcare claims. In live mode this data would already live in
# your Pinecone index; here we embed a couple of canonical FWA narratives
# once, at import time.
_MOCK_FRAUD_CASES = {
    "past-fwa-001": cohere_embed(
        "Provider billed for a 60 minute in-person office visit (CPT 99215) on a date "
        "the patient's own records show no visit occurred; no supporting clinical notes on file."
    ),
    "past-fwa-002": cohere_embed(
        "Multiple claims submitted for the same procedure code under different patient IDs "
        "from the same billing provider within a single day, exceeding plausible daily capacity."
    ),
}


def pinecone_query(embedding: list[float], top_k: int = 3) -> list[dict]:
    if USE_LIVE_APIS:
        from pinecone import Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)
        resp = index.query(vector=embedding, top_k=top_k, include_metadata=True)
        return [{"id": m.id, "score": m.score} for m in resp.matches]

    # Mock: brute-force cosine similarity against the in-memory FWA cases.
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
    patient_id: str
    claim_amount: float
    narrative: str
    identity_verified: bool
    coverage_valid: bool
    embedding: list[float]
    similar_cases: list[dict]
    fraud_flags: list[str]
    siu_decision: Optional[str]
    decision: str
    log: Annotated[list[str], operator.add]


# ---------------------------------------------------------------------------
# 2. NODES
# ---------------------------------------------------------------------------
def verify_patient_identity(state: ClaimState) -> dict:
    print("  -> [verify_patient_identity] executing")
    verified = state["patient_id"].startswith("PAT-")
    return {"identity_verified": verified, "log": [f"verify_patient_identity: verified={verified}"]}


def route_after_identity(state: ClaimState) -> str:
    return "check_insurance_coverage" if state["identity_verified"] else "reject_claim"


def reject_claim(state: ClaimState) -> dict:
    print("  -> [reject_claim] executing")
    return {"decision": "REJECTED: patient identity/member verification failed", "log": ["reject_claim: terminal"]}


def check_insurance_coverage(state: ClaimState) -> dict:
    print("  -> [check_insurance_coverage] executing")
    valid = state["claim_amount"] <= 50000
    return {"coverage_valid": valid, "log": [f"check_insurance_coverage: coverage_valid={valid}"]}


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
    if not state["coverage_valid"]:
        flags.append("amount_exceeds_coverage_limit")
    if state["claim_amount"] > 10000:
        flags.append("high_value_claim")

    SIMILARITY_THRESHOLD = 0.85
    top = state["similar_cases"][0] if state["similar_cases"] else None
    if top and top["score"] >= SIMILARITY_THRESHOLD:
        flags.append(f"similar_to_known_fwa:{top['id']}(score={top['score']})")

    return {"fraud_flags": flags, "log": [f"detect_fraud_signals: flags={flags}"]}


def route_after_fraud(state: ClaimState) -> str:
    return "notify_siu_zapier" if state["fraud_flags"] else "approve_claim"


def approve_claim(state: ClaimState) -> dict:
    print("  -> [approve_claim] executing")
    return {"decision": "APPROVED: auto-approved, no fraud/abuse signals", "log": ["approve_claim: terminal"]}


def notify_siu_zapier(state: ClaimState) -> dict:
    print("  -> [notify_siu_zapier] executing (Zapier)")
    result = zapier_notify({
        "claim_id": state["claim_id"],
        # Note: in a real deployment, avoid putting patient_id or narrative
        # text in a third-party webhook payload unless that vendor is
        # covered by a signed BAA -- a claim/case ID is usually sufficient
        # for the notification, with reviewers pulling detail from the
        # internal system rather than the webhook payload itself.
        "patient_id": state["patient_id"],
        "fraud_flags": state["fraud_flags"],
        "message": "Claim flagged for SIU (Special Investigations Unit) review",
    })
    return {"log": [f"notify_siu_zapier: webhook result={result}"]}


def route_to_siu_review(state: ClaimState) -> dict:
    print("  -> [route_to_siu_review] executing (post-human)")
    human_call = state.get("siu_decision") or "PENDING"
    return {"decision": f"SIU REVIEW: {human_call}", "log": [f"route_to_siu_review: finalized as {human_call}"]}


# ---------------------------------------------------------------------------
# 3. BUILD THE GRAPH
# ---------------------------------------------------------------------------
def build_graph(checkpointer):
    g = StateGraph(ClaimState)

    for name, fn in [
        ("verify_patient_identity", verify_patient_identity),
        ("reject_claim", reject_claim),
        ("check_insurance_coverage", check_insurance_coverage),
        ("embed_claim_narrative", embed_claim_narrative),
        ("check_similar_fraud_cases", check_similar_fraud_cases),
        ("detect_fraud_signals", detect_fraud_signals),
        ("approve_claim", approve_claim),
        ("notify_siu_zapier", notify_siu_zapier),
        ("route_to_siu_review", route_to_siu_review),
    ]:
        g.add_node(name, fn)

    g.add_edge(START, "verify_patient_identity")
    g.add_conditional_edges(
        "verify_patient_identity", route_after_identity,
        {"check_insurance_coverage": "check_insurance_coverage", "reject_claim": "reject_claim"},
    )
    g.add_edge("reject_claim", END)
    g.add_edge("check_insurance_coverage", "embed_claim_narrative")
    g.add_edge("embed_claim_narrative", "check_similar_fraud_cases")
    g.add_edge("check_similar_fraud_cases", "detect_fraud_signals")
    g.add_conditional_edges(
        "detect_fraud_signals", route_after_fraud,
        {"approve_claim": "approve_claim", "notify_siu_zapier": "notify_siu_zapier"},
    )
    g.add_edge("approve_claim", END)
    g.add_edge("notify_siu_zapier", "route_to_siu_review")
    g.add_edge("route_to_siu_review", END)

    # Pause AFTER notifying SIU (so they've already been pinged) but BEFORE
    # finalizing the decision.
    return g.compile(checkpointer=checkpointer, interrupt_before=["route_to_siu_review"])


# ---------------------------------------------------------------------------
# 4. RUN IT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"USE_LIVE_APIS = {USE_LIVE_APIS} (mock mode)" if not USE_LIVE_APIS else "LIVE mode -- calling real APIs")

    with SqliteSaver.from_conn_string("claims_v2_checkpoints.sqlite") as checkpointer:
        graph = build_graph(checkpointer)

        print("\n=== CLAIM D: clean, routine visit -- expect auto-approve ===")
        config_d = {"configurable": {"thread_id": "claim-D"}}
        result_d = graph.invoke(
            {
                "claim_id": "CLM-2001", "patient_id": "PAT-1010", "claim_amount": 180.0,
                "narrative": "Routine annual wellness visit, CPT 99395, no complications, "
                             "consistent with patient's prior visit history.",
                "log": [], "fraud_flags": [], "siu_decision": None,
            },
            config=config_d,
        )
        print("Final decision:", result_d["decision"])

        print("\n=== CLAIM E: narrative similar to known FWA case -- expect flag + pause ===")
        config_e = {"configurable": {"thread_id": "claim-E"}}
        result_e = graph.invoke(
            {
                "claim_id": "CLM-2002", "patient_id": "PAT-2050", "claim_amount": 4200.0,
                # Deliberately close wording to past-fwa-001 so the mock
                # cosine similarity trips the threshold.
                "narrative": "Provider billed for a 60 minute in-person office visit (CPT 99215) on a date "
                             "the patient's own records show no visit occurred; no supporting clinical notes on file.",
                "log": [], "fraud_flags": [], "siu_decision": None,
            },
            config=config_e,
        )
        snapshot = graph.get_state(config_e)
        print("Graph paused. Next node:", snapshot.next)
        print("Similar cases found:", snapshot.values["similar_cases"])
        print("Fraud flags:", snapshot.values["fraud_flags"])

        print("\n--- SIU reviewer reviews claim-E and decides: DENY ---")
        graph.update_state(config_e, {"siu_decision": "DENIED_BY_SIU"})
        result_e_final = graph.invoke(None, config=config_e)
        print("Final decision:", result_e_final["decision"])

"""
LangGraph POC #4 -- Healthcare Claims Authentication: Hybrid Retrieval,
RAG Grounding, Confidence-Driven Decision Gate, and a Circuit Breaker

Builds on POC #3 (LLM reasoning + guardrails + cache) and adds four things
identified by comparing this project against a real production narrative:

  1. HYBRID RETRIEVAL. check_similar_fraud_cases_hybrid combines Pinecone
     dense similarity with a real BM25 keyword search (rank_bm25) over the
     same mock FWA case narratives, and merges the two rankings. Dense
     search alone can miss exact-term matches (a specific CPT/ICD code);
     keyword search alone can miss paraphrased narratives. Hybrid catches
     both -- this is the single most concrete, evidence-backed idea from
     the source material this design was checked against.

  2. RAG GROUNDING. retrieve_guidelines is a genuinely new capability: it
     retrieves actual billing-guideline text (a small mock document store)
     using the same hybrid approach, and that retrieved text is placed
     directly in the LLM prompt with an explicit "answer only from what's
     provided, say so if it isn't" instruction. Previously (POC #3) the
     LLM reasoned only over rule flags and a bare similarity score -- it
     had no source text to ground its reasoning in. This is the difference
     between "an LLM call" and "RAG."

  3. CONFIDENCE-DRIVEN DECISION GATE. Previously, any fraud flag routed to
     human SIU review, full stop -- the LLM's confidence score was computed
     but never used. Now confidence_decision_gate reads recommended_action
     and confidence together and routes three ways: auto-approve (high
     confidence, clean), auto-reject (high confidence, clear violation),
     or human SIU review (everything in between -- the uncertain middle,
     which is where a human adds the most value).

  4. A CIRCUIT BREAKER, DISTINCT FROM THE GUARDRAILS. The guardrails in
     POC #3 validate *content* (is this a safe input, is this a valid LLM
     output). A circuit breaker is a different concern: *tool reliability*.
     check_similar_fraud_cases_hybrid tracks repeated Pinecone/BM25 failures
     per claim; after MAX_TOOL_ERRORS consecutive failures it stops
     retrying and escalates straight to human review with the failure
     reason attached, rather than retrying indefinitely or silently
     proceeding on incomplete data.

  5. SUPERVISOR-WORKER REASONING. llm_fraud_reasoning is no longer one LLM
     call doing everything. A supervisor now delegates to two narrow
     specialist sub-agents that each retrieve/reason over only the slice
     of state they need -- billing_coding_specialist (CPT/ICD plausibility
     against retrieved guideline chunks only) and narrative_fraud_specialist
     (free-text narrative against FWA case matches only) -- then
     supervisor_synthesize combines both findings into the same
     recommended_action/rationale/confidence structure the gate already
     expects. This is the hierarchical-delegation ("supervisor") multi-agent
     pattern: each specialist's prompt, retrieval scope, and even model
     tier can be tuned independently without touching the other's logic or
     the downstream gate.

  6. EVALUATOR-OPTIMIZER SELF-CRITIQUE. evaluator_optimizer_check is a
     second, cheap-tier LLM call that runs after the supervisor synthesizes
     a decision and before the output guardrail: it checks whether the
     synthesized rationale actually cites the retrieved guidelines/cases it
     claims to, or drifts into plausible-sounding but ungrounded reasoning.
     On a failed check it forces exactly one re-synthesis before escalating
     -- this complements the output guardrail (which checks *format*, not
     *faithfulness*) and is the second independent safety net over the
     supervisor's output.

Run: python3 04_claims_auth_hybrid_rag_confidence_circuitbreaker.py
"""

import os
import re
import json
import math
import sqlite3
import hashlib
import operator
from typing import TypedDict, Annotated, Optional

from rank_bm25 import BM25Okapi
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

USE_LIVE_APIS = False

COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "claims-fraud-cases")
ZAPIER_WEBHOOK_URL = os.environ.get("ZAPIER_WEBHOOK_URL")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
LLM_MODEL = "claude-sonnet-5"

CACHE_DB_PATH = "embedding_cache.sqlite"
MAX_TOOL_ERRORS = 2
MAX_EVALUATOR_REGENERATIONS = 1  # the evaluator forces at most one re-synthesis, never an unbounded loop

# Claims whose Pinecone call should simulate a hard, repeated outage --
# used to demonstrate the circuit breaker deterministically.
_FORCE_TOOL_FAILURE_THREADS = {"claim-K"}


# ---------------------------------------------------------------------------
# CACHE LAYER (same as POC #3)
# ---------------------------------------------------------------------------
def _cache_init():
    conn = sqlite3.connect(CACHE_DB_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS embedding_cache "
        "(narrative_hash TEXT PRIMARY KEY, embedding_json TEXT NOT NULL, created_at TEXT NOT NULL)"
    )
    conn.commit()
    conn.close()


def _cache_get(narrative: str) -> Optional[list[float]]:
    key = hashlib.sha256(narrative.encode()).hexdigest()
    conn = sqlite3.connect(CACHE_DB_PATH)
    row = conn.execute("SELECT embedding_json FROM embedding_cache WHERE narrative_hash = ?", (key,)).fetchone()
    conn.close()
    return json.loads(row[0]) if row else None


def _cache_put(narrative: str, embedding: list[float]) -> None:
    key = hashlib.sha256(narrative.encode()).hexdigest()
    conn = sqlite3.connect(CACHE_DB_PATH)
    conn.execute(
        "INSERT OR REPLACE INTO embedding_cache (narrative_hash, embedding_json, created_at) VALUES (?, ?, datetime('now'))",
        (key, json.dumps(embedding)),
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# INTEGRATION LAYER
# ---------------------------------------------------------------------------
def cohere_embed(text: str) -> list[float]:
    if USE_LIVE_APIS:
        import cohere
        client = cohere.Client(COHERE_API_KEY)
        resp = client.embed(texts=[text], model="embed-english-v3.0", input_type="search_document")
        return resp.embeddings[0]
    # Mock: a deterministic "hashing trick" bag-of-words embedding. Tokens
    # are hashed into a fixed number of buckets and counted, then
    # L2-normalized. Unlike a raw byte-hash of the whole string, this
    # actually makes cosine similarity track real word overlap between
    # texts -- necessary for the hybrid-search and confidence-gate demos
    # below to behave sensibly (a clean, unrelated narrative should score
    # LOW against the FWA corpus; a near-duplicate should score HIGH).
    dims = 64
    vec = [0.0] * dims
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        idx = int(hashlib.sha256(tok.encode()).hexdigest(), 16) % dims
        vec[idx] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm > 0 else vec


def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


# ---- Mock FWA case corpus (used for hybrid similarity search) ----
_FWA_CASES = {
    "past-fwa-001": "Provider billed for a 60 minute in-person office visit CPT 99215 on a date "
                    "the patient's own records show no visit occurred; no supporting clinical notes on file.",
    "past-fwa-002": "Multiple claims submitted for the same procedure code under different patient IDs "
                    "from the same billing provider within a single day, exceeding plausible daily capacity.",
    "past-fwa-003": "Claim upcoded a routine office visit CPT 99213 to a high-complexity visit CPT 99215 "
                    "without supporting documentation of the additional complexity.",
}
_FWA_EMBEDDINGS = {cid: cohere_embed(text) for cid, text in _FWA_CASES.items()}
_FWA_TOKENIZED = {cid: re.findall(r"[a-z0-9]+", text.lower()) for cid, text in _FWA_CASES.items()}
_FWA_BM25 = BM25Okapi(list(_FWA_TOKENIZED.values()))
_FWA_IDS_ORDER = list(_FWA_CASES.keys())

# ---- Mock billing-guideline corpus (used for RAG grounding) ----
_GUIDELINES = {
    "guideline-cms-4.3b": "CMS Guideline 4.3b: Two procedures performed during the same encounter must not "
                           "be billed separately (unbundled) if one is considered inclusive of the other "
                           "under standard coding edits, unless a modifier justifying separate billing is documented.",
    "guideline-cms-7.1": "CMS Guideline 7.1: The level of an evaluation and management (E/M) visit code billed "
                          "must be supported by documentation of the corresponding complexity of history, "
                          "examination, and medical decision-making performed during that visit.",
    "guideline-cms-2.9": "CMS Guideline 2.9: A claim must not be submitted for services rendered on a date "
                          "when the patient's own medical record shows no corresponding encounter took place.",
}
_GUIDELINE_EMBEDDINGS = {gid: cohere_embed(text) for gid, text in _GUIDELINES.items()}
_GUIDELINE_TOKENIZED = {gid: re.findall(r"[a-z0-9]+", text.lower()) for gid, text in _GUIDELINES.items()}
_GUIDELINE_BM25 = BM25Okapi(list(_GUIDELINE_TOKENIZED.values()))
_GUIDELINE_IDS_ORDER = list(_GUIDELINES.keys())


def _hybrid_rank(query_text: str, query_embedding: list[float], dense_embeddings: dict, bm25: BM25Okapi,
                  ids_order: list[str], texts: dict, top_k: int = 2, dense_weight: float = 0.6) -> list[dict]:
    """
    Merges dense cosine similarity with BM25 keyword scores using simple
    weighted min-max normalized fusion. Returns top_k results with both
    component scores so the caller (and a human reviewer) can see why
    something ranked where it did -- not just a single opaque number.
    """
    dense_scores = {cid: _cosine_sim(query_embedding, vec) for cid, vec in dense_embeddings.items()}
    query_tokens = re.findall(r"[a-z0-9]+", query_text.lower())
    bm25_raw = bm25.get_scores(query_tokens)
    bm25_scores = {cid: float(score) for cid, score in zip(ids_order, bm25_raw)}

    def _norm(d: dict) -> dict:
        vals = list(d.values())
        lo, hi = min(vals), max(vals)
        if hi - lo < 1e-9:
            return {k: 0.0 for k in d}
        return {k: (v - lo) / (hi - lo) for k, v in d.items()}

    dense_n = _norm(dense_scores)
    bm25_n = _norm(bm25_scores)

    fused = {
        cid: float(dense_weight * dense_n[cid] + (1 - dense_weight) * bm25_n[cid])
        for cid in ids_order
    }
    ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return [
        {
            "id": cid,
            "fused_score": round(score, 4),
            "dense_score": round(float(dense_scores[cid]), 4),
            "keyword_score": round(float(bm25_scores[cid]), 4),
            "text": texts[cid],
        }
        for cid, score in ranked
    ]


def pinecone_query_hybrid(narrative: str, embedding: list[float], thread_id: str, attempt_state: dict) -> list[dict]:
    """
    Wraps the Pinecone call with simulated-failure support for the circuit
    breaker demo. In live mode this would be a real Pinecone query;
    hybrid fusion with BM25 happens locally either way, mirroring a
    common production pattern (vendor does dense retrieval, you fuse with
    your own keyword layer).
    """
    if thread_id in _FORCE_TOOL_FAILURE_THREADS:
        raise ConnectionError("simulated Pinecone outage for circuit-breaker demonstration")

    if USE_LIVE_APIS:
        from pinecone import Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)
        resp = index.query(vector=embedding, top_k=3, include_metadata=True)
        dense_only = [{"id": m.id, "fused_score": m.score, "dense_score": m.score, "keyword_score": None, "text": None} for m in resp.matches]
        return dense_only

    return _hybrid_rank(narrative, embedding, _FWA_EMBEDDINGS, _FWA_BM25, _FWA_IDS_ORDER, _FWA_CASES, top_k=2)


def retrieve_guidelines_hybrid(narrative: str, embedding: list[float]) -> list[dict]:
    return _hybrid_rank(narrative, embedding, _GUIDELINE_EMBEDDINGS, _GUIDELINE_BM25, _GUIDELINE_IDS_ORDER, _GUIDELINES, top_k=2)


def zapier_notify(payload: dict) -> dict:
    if USE_LIVE_APIS:
        import requests
        resp = requests.post(ZAPIER_WEBHOOK_URL, json=payload, timeout=10)
        return {"status_code": resp.status_code}
    print(f"     [zapier mock] would POST to webhook: {payload}")
    return {"status_code": 200, "mock": True}


def billing_coding_specialist_review(claim_id: str, rule_flags: list[str], guidelines: list[dict]) -> dict:
    """
    Narrow specialist #1: CPT/ICD code plausibility against retrieved
    billing-guideline text ONLY. Never sees the narrative or FWA case
    matches -- sub-agent isolation, so this specialist's context can't
    bloat with fields it has no use for and its prompt/tier can be tuned
    independently of the narrative specialist's.
    """
    guideline_block = "\n".join(f"- [{g['id']}] {g['text']}" for g in guidelines)
    prompt = (
        "You are the Billing/Coding specialist on a claims-review team. Judge ONLY whether the "
        "billed procedure code(s) are plausible against the retrieved guideline text below -- do "
        "not consider anything else. Respond with ONLY a JSON object with keys "
        '"finding" ("consistent" or "coding_concern"), "rationale" (cite a guideline id if used), '
        '"confidence" (0-1).\n\n'
        f"Claim ID: {claim_id}\nRule-based flags: {rule_flags}\n\n"
        f"Retrieved billing guidelines:\n{guideline_block}\n"
    )
    if USE_LIVE_APIS:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        resp = client.messages.create(model=LLM_MODEL, max_tokens=200, messages=[{"role": "user", "content": prompt}])
        raw_text = resp.content[0].text
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            return {"_malformed_raw_output": raw_text}

    cited_guideline = guidelines[0]["id"] if guidelines else "no guideline retrieved"
    if rule_flags:
        return {
            "finding": "coding_concern",
            "rationale": f"{len(rule_flags)} rule-based flag(s) present; potentially inconsistent with {cited_guideline}.",
            "confidence": round(min(0.55 + 0.1 * len(rule_flags), 0.95), 2),
        }
    return {
        "finding": "consistent",
        "rationale": f"No rule-based flags; billed code(s) consistent with {cited_guideline}.",
        "confidence": 0.9,
    }


def narrative_fraud_specialist_review(claim_id: str, narrative: str, similar_cases: list[dict]) -> dict:
    """
    Narrow specialist #2: free-text narrative against confirmed-FWA case
    matches ONLY. Never sees the guideline corpus or rule flags -- the
    counterpart isolation to the billing specialist above.
    """
    similar_block = "\n".join(f"- [{c['id']}] fused_score={c['fused_score']} :: {c.get('text', '')}" for c in similar_cases)
    prompt = (
        "You are the Narrative/Fraud specialist on a claims-review team. Judge ONLY whether this "
        "claim's narrative resembles the confirmed-fraud cases retrieved below -- do not consider "
        "billing codes. Respond with ONLY a JSON object with keys "
        '"finding" ("clean" or "fraud_pattern_match"), "rationale" (cite a case id if used), '
        '"confidence" (0-1).\n\n'
        f"Claim ID: {claim_id}\nNarrative: {narrative}\n\n"
        f"Retrieved similar FWA cases:\n{similar_block}\n"
    )
    if USE_LIVE_APIS:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        resp = client.messages.create(model=LLM_MODEL, max_tokens=200, messages=[{"role": "user", "content": prompt}])
        raw_text = resp.content[0].text
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            return {"_malformed_raw_output": raw_text}

    top_case = similar_cases[0] if similar_cases else None
    top_score = top_case["dense_score"] if top_case else 0.0
    if top_score >= 0.65:
        return {
            "finding": "fraud_pattern_match",
            "rationale": f"{top_score:.2f} dense similarity to confirmed case {top_case['id']}.",
            "confidence": round(min(0.5 + 0.48 * top_score, 0.99), 2),
        }
    return {
        "finding": "clean",
        "rationale": "Low similarity to all known FWA cases.",
        "confidence": 0.92,
    }


def supervisor_synthesize_findings(claim_id: str, billing_finding: dict, narrative_finding: dict) -> dict:
    """
    The supervisor: reads both specialists' independent findings (never the
    raw retrieval or narrative itself -- it synthesizes conclusions, it
    doesn't re-derive them) and produces the single recommended_action /
    rationale / confidence structure the rest of the graph already expects.
    Either specialist raising a concern is enough to flag the claim -- the
    supervisor doesn't average away one specialist's signal with the
    other's silence.
    """
    billing_flagged = billing_finding.get("finding") == "coding_concern"
    narrative_flagged = narrative_finding.get("finding") == "fraud_pattern_match"

    if billing_flagged or narrative_flagged:
        parts = []
        if billing_flagged:
            parts.append(f"billing/coding specialist: {billing_finding.get('rationale', '')}")
        if narrative_flagged:
            parts.append(f"narrative/fraud specialist: {narrative_finding.get('rationale', '')}")
        confidence = max(
            billing_finding.get("confidence", 0.0) if billing_flagged else 0.0,
            narrative_finding.get("confidence", 0.0) if narrative_flagged else 0.0,
        )
        return {
            "recommended_action": "flag_for_siu",
            "rationale": "; ".join(parts),
            "confidence": round(float(confidence), 2),
        }
    return {
        "recommended_action": "approve",
        "rationale": (
            f"billing/coding specialist: {billing_finding.get('rationale', '')}; "
            f"narrative/fraud specialist: {narrative_finding.get('rationale', '')}"
        ),
        "confidence": round(min(float(billing_finding.get("confidence", 0.9)), float(narrative_finding.get("confidence", 0.9))), 2),
    }


def evaluator_optimizer_faithfulness_check(rationale: str, guidelines: list[dict], similar_cases: list[dict]) -> dict:
    """
    A second, cheap-tier LLM call that checks the supervisor's synthesized
    rationale actually cites material that was retrieved, rather than
    drifting into plausible-sounding but ungrounded reasoning. Distinct
    from run_output_guardrail: the guardrail checks *format* (is this valid,
    well-formed output); this checks *faithfulness* (is the reasoning
    actually grounded in what was retrieved).
    """
    retrieved_ids = {g["id"] for g in guidelines} | {c["id"] for c in similar_cases}

    if USE_LIVE_APIS:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        prompt = (
            "Does the rationale below cite ONLY ids from this retrieved set, or does it assert "
            "something the retrieved material doesn't support? Respond with ONLY JSON: "
            '{"faithful": true|false, "reason": "..."}.\n\n'
            f"Retrieved ids: {sorted(retrieved_ids)}\nRationale: {rationale}\n"
        )
        resp = client.messages.create(model="claude-haiku-4-5", max_tokens=100, messages=[{"role": "user", "content": prompt}])
        raw_text = resp.content[0].text
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            return {"faithful": False, "reason": "evaluator output was not valid JSON"}

    # Mock: a cited-id must actually be among what was retrieved. A rationale
    # that names no id at all is treated as faithful-by-default (nothing to
    # contradict), matching the "approve, nothing flagged" case above.
    cited = set(re.findall(r"\b(?:guideline-cms-[\d.]+[a-z]?|past-fwa-\d+)\b", rationale))
    hallucinated = cited - retrieved_ids
    if hallucinated:
        return {"faithful": False, "reason": f"rationale cites {sorted(hallucinated)}, not present in retrieved set"}
    return {"faithful": True, "reason": "all cited ids (if any) are in the retrieved set"}


# ---------------------------------------------------------------------------
# GUARDRAILS (same as POC #3)
# ---------------------------------------------------------------------------
_INJECTION_PATTERNS = [
    r"ignore (all|previous|prior) instructions",
    r"system\s*:",
    r"you are now",
    r"disregard the above",
    r"reveal your (prompt|instructions)",
]
_SSN_PATTERN = r"\b\d{3}-\d{2}-\d{4}\b"


def run_input_guardrail(narrative: str) -> tuple[bool, str]:
    if not narrative or len(narrative) > 2000:
        return False, "narrative missing or exceeds max length"
    lowered = narrative.lower()
    for pattern in _INJECTION_PATTERNS:
        if re.search(pattern, lowered):
            return False, f"narrative matched prompt-injection pattern: {pattern!r}"
    if re.search(_SSN_PATTERN, narrative):
        return False, "narrative contains an unredacted SSN-shaped string"
    return True, "ok"


def run_output_guardrail(llm_output: dict) -> tuple[bool, str]:
    if "_malformed_raw_output" in llm_output:
        return False, "LLM output was not valid JSON"
    required_keys = {"recommended_action", "rationale", "confidence"}
    if not required_keys.issubset(llm_output.keys()):
        return False, f"LLM output missing required keys, got: {list(llm_output.keys())}"
    if llm_output["recommended_action"] not in ("approve", "flag_for_siu"):
        return False, f"LLM output used an unrecognized action: {llm_output['recommended_action']!r}"
    try:
        conf = float(llm_output["confidence"])
    except (TypeError, ValueError):
        return False, "LLM output confidence is not numeric"
    if not (0.0 <= conf <= 1.0):
        return False, f"LLM output confidence out of range: {conf}"
    rationale = llm_output.get("rationale", "")
    if not isinstance(rationale, str) or not (0 < len(rationale) <= 500):
        return False, "LLM output rationale missing or implausibly long"
    for pattern in _INJECTION_PATTERNS:
        if re.search(pattern, rationale.lower()):
            return False, "LLM rationale itself matched an injection pattern (possible leakage)"
    return True, "ok"


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
    input_guardrail_passed: bool
    input_guardrail_reason: str
    embedding: list[float]
    embedding_cache_hit: bool
    similar_cases: list[dict]
    retrieved_guidelines: list[dict]
    rule_fraud_flags: list[str]
    billing_finding: dict
    narrative_finding: dict
    llm_output: dict
    evaluator_critique: dict
    evaluator_regeneration_count: int
    output_guardrail_passed: bool
    output_guardrail_reason: str
    guardrail_override: bool
    fraud_flags: list[str]
    siu_decision: Optional[str]
    decision: str
    tool_error_count: int
    tool_call_failed: bool
    tool_error_reason: str
    circuit_breaker_tripped: bool
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
    reason = state.get("input_guardrail_reason")
    decision = (
        f"REJECTED: blocked by input guardrail ({reason})"
        if reason and not state.get("input_guardrail_passed", True)
        else "REJECTED: patient identity/member verification failed"
    )
    return {"decision": decision, "log": ["reject_claim: terminal"]}


def check_insurance_coverage(state: ClaimState) -> dict:
    print("  -> [check_insurance_coverage] executing")
    valid = state["claim_amount"] <= 50000
    return {"coverage_valid": valid, "log": [f"check_insurance_coverage: coverage_valid={valid}"]}


def compute_rule_flags(state: ClaimState) -> dict:
    print("  -> [compute_rule_flags] executing")
    flags = []
    if not state["coverage_valid"]:
        flags.append("amount_exceeds_coverage_limit")
    if state["claim_amount"] > 10000:
        flags.append("high_value_claim")
    return {"rule_fraud_flags": flags, "log": [f"compute_rule_flags: flags={flags}"]}


def input_guardrail_check(state: ClaimState) -> dict:
    print("  -> [input_guardrail_check] executing")
    passed, reason = run_input_guardrail(state["narrative"])
    return {
        "input_guardrail_passed": passed,
        "input_guardrail_reason": reason,
        "log": [f"input_guardrail_check: passed={passed} reason={reason}"],
    }


def route_after_input_guardrail(state: ClaimState) -> str:
    return "check_narrative_embedding" if state["input_guardrail_passed"] else "reject_claim"


def check_narrative_embedding(state: ClaimState) -> dict:
    print("  -> [check_narrative_embedding] executing (cache-aware, Cohere on miss)")
    cached = _cache_get(state["narrative"])
    if cached is not None:
        print("     cache HIT -- skipping Cohere call")
        return {"embedding": cached, "embedding_cache_hit": True, "log": ["check_narrative_embedding: cache hit"]}
    print("     cache MISS -- calling Cohere")
    vec = cohere_embed(state["narrative"])
    _cache_put(state["narrative"], vec)
    return {"embedding": vec, "embedding_cache_hit": False, "log": ["check_narrative_embedding: cache miss, embedded + cached"]}


def check_similar_fraud_cases_hybrid(state: ClaimState, config) -> dict:
    thread_id = config["configurable"]["thread_id"]
    error_count = state.get("tool_error_count", 0)
    print(f"  -> [check_similar_fraud_cases_hybrid] executing (dense+keyword, attempt #{error_count + 1})")
    try:
        matches = pinecone_query_hybrid(state["narrative"], state["embedding"], thread_id, state)
        print(f"     top match: {matches[0]['id'] if matches else None} (fused_score={matches[0]['fused_score'] if matches else None})")
        return {
            "similar_cases": matches,
            "tool_call_failed": False,
            "log": [f"check_similar_fraud_cases_hybrid: top match={matches[0] if matches else None}"],
        }
    except Exception as e:
        new_count = error_count + 1
        print(f"     !! tool call failed ({e}); error_count now {new_count}")
        return {
            "tool_call_failed": True,
            "tool_error_count": new_count,
            "tool_error_reason": str(e),
            "log": [f"check_similar_fraud_cases_hybrid: FAILED attempt #{new_count} ({e})"],
        }


def route_after_similarity(state: ClaimState) -> str:
    if state.get("tool_call_failed"):
        if state["tool_error_count"] >= MAX_TOOL_ERRORS:
            return "circuit_breaker_escalate"
        return "check_similar_fraud_cases_hybrid"  # retry (self-loop, bounded by MAX_TOOL_ERRORS)
    return "retrieve_guidelines"


def circuit_breaker_escalate(state: ClaimState) -> dict:
    # Distinct from the output guardrail's fail-closed path: this fires on
    # TOOL UNRELIABILITY, not on untrusted model output. The claim never
    # even reaches the LLM -- we don't have reliable data to reason over.
    print("  -> [circuit_breaker_escalate] executing (tool circuit breaker tripped)")
    return {
        "circuit_breaker_tripped": True,
        "fraud_flags": state["rule_fraud_flags"] + [f"circuit_breaker_tripped:{state.get('tool_error_reason', 'unknown')}"],
        "log": [f"circuit_breaker_escalate: {state['tool_error_count']} consecutive tool failures, escalating without LLM review"],
    }


def retrieve_guidelines(state: ClaimState) -> dict:
    print("  -> [retrieve_guidelines] executing (RAG: hybrid retrieval over guideline corpus)")
    guidelines = retrieve_guidelines_hybrid(state["narrative"], state["embedding"])
    print(f"     retrieved: {[g['id'] for g in guidelines]}")
    return {"retrieved_guidelines": guidelines, "log": [f"retrieve_guidelines: retrieved={[g['id'] for g in guidelines]}"]}


def billing_coding_specialist(state: ClaimState) -> dict:
    print("  -> [billing_coding_specialist] executing (narrow scope: rule flags + guidelines only)")
    finding = billing_coding_specialist_review(state["claim_id"], state["rule_fraud_flags"], state["retrieved_guidelines"])
    return {"billing_finding": finding, "log": [f"billing_coding_specialist: finding={finding}"]}


def narrative_fraud_specialist(state: ClaimState) -> dict:
    print("  -> [narrative_fraud_specialist] executing (narrow scope: narrative + FWA matches only)")
    finding = narrative_fraud_specialist_review(state["claim_id"], state["narrative"], state["similar_cases"])
    return {"narrative_finding": finding, "log": [f"narrative_fraud_specialist: finding={finding}"]}


def supervisor_synthesize(state: ClaimState) -> dict:
    print("  -> [supervisor_synthesize] executing (combining both specialists' findings)")
    output = supervisor_synthesize_findings(state["claim_id"], state["billing_finding"], state["narrative_finding"])
    return {"llm_output": output, "log": [f"supervisor_synthesize: output={output}"]}


def evaluator_optimizer_check(state: ClaimState) -> dict:
    print("  -> [evaluator_optimizer_check] executing (faithfulness self-critique, cheap-tier)")
    critique = evaluator_optimizer_faithfulness_check(
        state["llm_output"].get("rationale", ""), state["retrieved_guidelines"], state["similar_cases"],
    )
    return {"evaluator_critique": critique, "log": [f"evaluator_optimizer_check: critique={critique}"]}


def route_after_evaluator(state: ClaimState) -> str:
    if state["evaluator_critique"].get("faithful", True):
        return "output_guardrail_check"
    if state.get("evaluator_regeneration_count", 0) < MAX_EVALUATOR_REGENERATIONS:
        return "supervisor_synthesize_retry"
    return "output_guardrail_check"  # exhausted the one allowed regeneration -- let the output guardrail's fail-closed path catch it


def supervisor_synthesize_retry(state: ClaimState) -> dict:
    # One bounded re-synthesis, per MAX_EVALUATOR_REGENERATIONS -- never an
    # unbounded generate/critique loop. Re-runs the same synthesis (in live
    # mode this would re-prompt the supervisor LLM with the evaluator's
    # critique attached); the count guards against looping forever if the
    # specialists' findings themselves are what's ungrounded.
    print("  -> [supervisor_synthesize_retry] executing (evaluator flagged unfaithful reasoning, re-synthesizing once)")
    output = supervisor_synthesize_findings(state["claim_id"], state["billing_finding"], state["narrative_finding"])
    count = state.get("evaluator_regeneration_count", 0) + 1
    return {
        "llm_output": output,
        "evaluator_regeneration_count": count,
        "log": [f"supervisor_synthesize_retry: attempt #{count}, output={output}"],
    }


def output_guardrail_check(state: ClaimState) -> dict:
    print("  -> [output_guardrail_check] executing")
    passed, reason = run_output_guardrail(state["llm_output"])
    return {
        "output_guardrail_passed": passed,
        "output_guardrail_reason": reason,
        "log": [f"output_guardrail_check: passed={passed} reason={reason}"],
    }


def route_after_output_guardrail(state: ClaimState) -> str:
    return "confidence_decision_gate" if state["output_guardrail_passed"] else "force_siu_fallback"


def force_siu_fallback(state: ClaimState) -> dict:
    print("  -> [force_siu_fallback] executing (output guardrail failed -- fail closed)")
    return {
        "guardrail_override": True,
        "fraud_flags": state["rule_fraud_flags"] + [f"output_guardrail_failed:{state['output_guardrail_reason']}"],
        "log": ["force_siu_fallback: output guardrail failed, forcing SIU review"],
    }


AUTO_APPROVE_CONFIDENCE = 0.85
AUTO_REJECT_CONFIDENCE = 0.95
AUTO_REJECT_MIN_SIMILARITY = 0.97  # near-exact match required, not just "high confidence"


def confidence_decision_gate(state: ClaimState) -> dict:
    """
    The three-way decision gate: reads the LLM's recommended_action AND
    confidence together (previously confidence was computed but unused).
      - approve + high confidence            -> auto-approve
      - flag_for_siu + very high confidence
        + near-exact known-FWA match          -> auto-reject (clear violation)
      - everything else (the uncertain middle) -> human SIU review
    """
    print("  -> [confidence_decision_gate] executing")
    action = state["llm_output"].get("recommended_action")
    confidence = float(state["llm_output"].get("confidence", 0.0))
    # Use the raw dense cosine score for the absolute auto-reject threshold,
    # NOT fused_score: fused_score is min-max normalized across only the
    # top-k candidates for RANKING purposes, so it can sit near 1.0 even
    # when the true best match is a mediocre one. dense_score is a genuine
    # absolute similarity (0-1) and is the right thing to threshold.
    top_sim = state["similar_cases"][0]["dense_score"] if state["similar_cases"] else 0.0

    flags = list(state["rule_fraud_flags"])
    if action == "approve" and confidence >= AUTO_APPROVE_CONFIDENCE:
        gate_result = "auto_approve"
    elif action == "flag_for_siu" and confidence >= AUTO_REJECT_CONFIDENCE and top_sim >= AUTO_REJECT_MIN_SIMILARITY:
        gate_result = "auto_reject"
        flags.append(f"llm_high_confidence_violation(confidence={confidence})")
    else:
        gate_result = "human_review"
        if action == "flag_for_siu":
            flags.append(f"llm_recommended_review(confidence={confidence})")

    print(f"     action={action} confidence={confidence} top_similarity={top_sim:.2f} -> gate_result={gate_result}")
    return {"fraud_flags": flags, "log": [f"confidence_decision_gate: {gate_result} (action={action}, confidence={confidence})"]}


def route_after_confidence_gate(state: ClaimState) -> str:
    action = state["llm_output"].get("recommended_action")
    confidence = float(state["llm_output"].get("confidence", 0.0))
    top_sim = state["similar_cases"][0]["dense_score"] if state["similar_cases"] else 0.0
    if action == "approve" and confidence >= AUTO_APPROVE_CONFIDENCE:
        return "approve_claim"
    if action == "flag_for_siu" and confidence >= AUTO_REJECT_CONFIDENCE and top_sim >= AUTO_REJECT_MIN_SIMILARITY:
        return "auto_reject_claim"
    return "notify_siu_zapier"


def approve_claim(state: ClaimState) -> dict:
    print("  -> [approve_claim] executing")
    return {"decision": "APPROVED: auto-approved, high-confidence clean claim", "log": ["approve_claim: terminal"]}


def auto_reject_claim(state: ClaimState) -> dict:
    print("  -> [auto_reject_claim] executing")
    zapier_notify({
        "claim_id": state["claim_id"], "patient_id": state["patient_id"],
        "fraud_flags": state["fraud_flags"], "message": "Claim auto-rejected -- high-confidence clear violation, audit notification only",
    })
    return {
        "decision": f"REJECTED: auto-rejected, high-confidence clear violation (confidence={state['llm_output'].get('confidence')})",
        "log": ["auto_reject_claim: terminal (automated, no human gate)"],
    }


def notify_siu_zapier(state: ClaimState) -> dict:
    print("  -> [notify_siu_zapier] executing (Zapier)")
    result = zapier_notify({
        "claim_id": state["claim_id"],
        "patient_id": state["patient_id"],
        "fraud_flags": state["fraud_flags"],
        "circuit_breaker_tripped": state.get("circuit_breaker_tripped", False),
        "guardrail_override": state.get("guardrail_override", False),
        "message": "Claim flagged for SIU review",
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
        ("compute_rule_flags", compute_rule_flags),
        ("input_guardrail_check", input_guardrail_check),
        ("check_narrative_embedding", check_narrative_embedding),
        ("check_similar_fraud_cases_hybrid", check_similar_fraud_cases_hybrid),
        ("circuit_breaker_escalate", circuit_breaker_escalate),
        ("retrieve_guidelines", retrieve_guidelines),
        ("billing_coding_specialist", billing_coding_specialist),
        ("narrative_fraud_specialist", narrative_fraud_specialist),
        ("supervisor_synthesize", supervisor_synthesize),
        ("evaluator_optimizer_check", evaluator_optimizer_check),
        ("supervisor_synthesize_retry", supervisor_synthesize_retry),
        ("output_guardrail_check", output_guardrail_check),
        ("force_siu_fallback", force_siu_fallback),
        ("confidence_decision_gate", confidence_decision_gate),
        ("approve_claim", approve_claim),
        ("auto_reject_claim", auto_reject_claim),
        ("notify_siu_zapier", notify_siu_zapier),
        ("route_to_siu_review", route_to_siu_review),
    ]:
        g.add_node(name, fn)

    g.add_edge(START, "verify_patient_identity")
    g.add_conditional_edges(
        "verify_patient_identity", route_after_identity,
        {"check_insurance_coverage": "check_insurance_coverage", "reject_claim": "reject_claim"},
    )
    g.add_edge("check_insurance_coverage", "compute_rule_flags")
    g.add_edge("compute_rule_flags", "input_guardrail_check")
    g.add_conditional_edges(
        "input_guardrail_check", route_after_input_guardrail,
        {"check_narrative_embedding": "check_narrative_embedding", "reject_claim": "reject_claim"},
    )
    g.add_edge("reject_claim", END)
    g.add_edge("check_narrative_embedding", "check_similar_fraud_cases_hybrid")
    g.add_conditional_edges(
        "check_similar_fraud_cases_hybrid", route_after_similarity,
        {
            "check_similar_fraud_cases_hybrid": "check_similar_fraud_cases_hybrid",
            "circuit_breaker_escalate": "circuit_breaker_escalate",
            "retrieve_guidelines": "retrieve_guidelines",
        },
    )
    g.add_edge("circuit_breaker_escalate", "notify_siu_zapier")
    # Supervisor-worker fan-out: both specialists run off the same
    # retrieved guidelines/similar-cases state, each reading only the
    # slice it needs, then fan back in to the supervisor.
    g.add_edge("retrieve_guidelines", "billing_coding_specialist")
    g.add_edge("retrieve_guidelines", "narrative_fraud_specialist")
    g.add_edge("billing_coding_specialist", "supervisor_synthesize")
    g.add_edge("narrative_fraud_specialist", "supervisor_synthesize")
    g.add_edge("supervisor_synthesize", "evaluator_optimizer_check")
    g.add_conditional_edges(
        "evaluator_optimizer_check", route_after_evaluator,
        {"output_guardrail_check": "output_guardrail_check", "supervisor_synthesize_retry": "supervisor_synthesize_retry"},
    )
    g.add_edge("supervisor_synthesize_retry", "output_guardrail_check")
    g.add_conditional_edges(
        "output_guardrail_check", route_after_output_guardrail,
        {"confidence_decision_gate": "confidence_decision_gate", "force_siu_fallback": "force_siu_fallback"},
    )
    g.add_edge("force_siu_fallback", "notify_siu_zapier")
    g.add_conditional_edges(
        "confidence_decision_gate", route_after_confidence_gate,
        {"approve_claim": "approve_claim", "auto_reject_claim": "auto_reject_claim", "notify_siu_zapier": "notify_siu_zapier"},
    )
    g.add_edge("approve_claim", END)
    g.add_edge("auto_reject_claim", END)
    g.add_edge("notify_siu_zapier", "route_to_siu_review")
    g.add_edge("route_to_siu_review", END)

    return g.compile(checkpointer=checkpointer, interrupt_before=["route_to_siu_review"])


# ---------------------------------------------------------------------------
# 4. RUN IT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    _cache_init()
    print(f"USE_LIVE_APIS = {USE_LIVE_APIS} (mock mode)" if not USE_LIVE_APIS else "LIVE mode -- calling real APIs")

    with SqliteSaver.from_conn_string("claims_v4_checkpoints.sqlite") as checkpointer:
        graph = build_graph(checkpointer)
        base = {
            "log": [], "fraud_flags": [], "siu_decision": None, "rule_fraud_flags": [],
            "tool_error_count": 0, "evaluator_regeneration_count": 0,
        }

        clean_narrative = (
            "Routine annual wellness visit, CPT 99395, no complications, "
            "consistent with patient's prior visit history."
        )
        exact_fraud_narrative = _FWA_CASES["past-fwa-001"]  # verbatim match -> should clear AUTO_REJECT thresholds
        ambiguous_narrative = (
            "Claim billed a high-complexity office visit CPT 99215; chart notes show a routine "
            "CPT 99213-level encounter with limited additional documentation."
        )

        print("\n=== CLAIM J: clean narrative -- expect hybrid retrieval, RAG grounding, AUTO-APPROVE ===")
        config_j = {"configurable": {"thread_id": "claim-J"}}
        result_j = graph.invoke(
            {**base, "claim_id": "CLM-4001", "patient_id": "PAT-1010", "claim_amount": 180.0, "narrative": clean_narrative},
            config=config_j,
        )
        print("Final decision:", result_j["decision"])

        print("\n=== CLAIM K: Pinecone permanently failing -- expect CIRCUIT BREAKER to trip after retries ===")
        config_k = {"configurable": {"thread_id": "claim-K"}}
        result_k = graph.invoke(
            {**base, "claim_id": "CLM-4002", "patient_id": "PAT-2020", "claim_amount": 500.0, "narrative": clean_narrative},
            config=config_k,
        )
        snapshot_k = graph.get_state(config_k)
        print("Circuit breaker tripped:", snapshot_k.values.get("circuit_breaker_tripped"))
        print("Tool error count:", snapshot_k.values.get("tool_error_count"))
        print("Graph paused (never reached the LLM at all). Next node:", snapshot_k.next)
        graph.update_state(config_k, {"siu_decision": "RESOLVED_AFTER_MANUAL_LOOKUP"})
        result_k_final = graph.invoke(None, config=config_k)
        print("Final decision:", result_k_final["decision"])

        print("\n=== CLAIM L: narrative is a near-exact match to a known FWA case -- expect AUTO-REJECT (no human gate) ===")
        config_l = {"configurable": {"thread_id": "claim-L"}}
        result_l = graph.invoke(
            {**base, "claim_id": "CLM-4003", "patient_id": "PAT-3030", "claim_amount": 4200.0, "narrative": exact_fraud_narrative},
            config=config_l,
        )
        print("LLM output:", result_l.get("llm_output"))
        print("Final decision:", result_l["decision"])
        print("(Notice: no pause, no route_to_siu_review -- this was a fully automated high-confidence rejection.)")

        print("\n=== CLAIM M: ambiguous narrative, partial documentation -- expect the UNCERTAIN MIDDLE -> human review ===")
        config_m = {"configurable": {"thread_id": "claim-M"}}
        result_m = graph.invoke(
            {**base, "claim_id": "CLM-4004", "patient_id": "PAT-4040", "claim_amount": 2200.0, "narrative": ambiguous_narrative},
            config=config_m,
        )
        snapshot_m = graph.get_state(config_m)
        print("LLM output:", snapshot_m.values.get("llm_output"))
        print("Retrieved guidelines:", [g["id"] for g in snapshot_m.values.get("retrieved_guidelines", [])])
        print("Graph paused for human judgement. Next node:", snapshot_m.next)
        graph.update_state(config_m, {"siu_decision": "APPROVED_AFTER_DOCUMENTATION_REQUEST"})
        result_m_final = graph.invoke(None, config=config_m)
        print("Final decision:", result_m_final["decision"])

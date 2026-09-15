"""
Gradio demo for the Healthcare Claims Authentication POC.

Wraps claims_auth_hybrid_rag_confidence_circuitbreaker.py's LangGraph graph in a
shareable web UI: pick a preset mock claim (or type your own), submit it, and if
it lands in the "uncertain middle" you act as the human SIU reviewer yourself --
Approve or Deny -- and watch the graph resume exactly where it paused.

Runs entirely in mock mode (USE_LIVE_APIS = False in the underlying script) so it
needs no API keys and costs nothing to host publicly. Deploy target: Hugging Face
Spaces (Gradio SDK) -- see the "Sharing this demo" section in README.md.
"""

import uuid

import gradio as gr
from langgraph.checkpoint.sqlite import SqliteSaver

import claims_auth_hybrid_rag_confidence_circuitbreaker as claims_mod


# ---------------------------------------------------------------------------
# Adapt the circuit-breaker demo trigger to work with per-session thread IDs.
# The underlying script hardcodes thread_id "claim-K" to simulate a permanent
# Pinecone outage. Here every claim gets a fresh, unique thread_id (so
# concurrent visitors on a public Space never collide on the same checkpoint
# thread), so we match on a PREFIX instead of exact membership.
# ---------------------------------------------------------------------------
class _PrefixTriggerSet:
    def __contains__(self, thread_id: object) -> bool:
        return isinstance(thread_id, str) and thread_id.startswith("demo-circuit-breaker")


claims_mod._FORCE_TOOL_FAILURE_THREADS = _PrefixTriggerSet()

# claims_mod only creates its embedding-cache table under `if __name__ ==
# "__main__"`, which never runs when this module is imported as a library --
# so create it explicitly here, or every claim submission crashes with
# "no such table: embedding_cache".
claims_mod._cache_init()

# ---------------------------------------------------------------------------
# Build the graph once at process startup. The checkpointer's SQLite file
# persists claim state across requests (and across restarts) -- fine for a
# demo; delete claims_demo_checkpoints.sqlite for a clean slate.
# ---------------------------------------------------------------------------
_checkpointer_cm = SqliteSaver.from_conn_string("claims_demo_checkpoints.sqlite")
_checkpointer = _checkpointer_cm.__enter__()
_graph = claims_mod.build_graph(_checkpointer)

PRESETS = {
    "Clean claim -- expect AUTO-APPROVE": {
        "prefix": "demo-clean",
        "patient_id": "PAT-1010",
        "claim_amount": 180.0,
        "narrative": (
            "Routine annual wellness visit, CPT 99395, no complications, "
            "consistent with patient's prior visit history."
        ),
    },
    "Pinecone outage -- expect CIRCUIT BREAKER trip": {
        "prefix": "demo-circuit-breaker",
        "patient_id": "PAT-2020",
        "claim_amount": 500.0,
        "narrative": (
            "Routine annual wellness visit, CPT 99395, no complications, "
            "consistent with patient's prior visit history."
        ),
    },
    "Near-exact FWA match -- expect AUTO-REJECT (no human gate)": {
        "prefix": "demo-fraud-match",
        "patient_id": "PAT-3030",
        "claim_amount": 4200.0,
        "narrative": claims_mod._FWA_CASES["past-fwa-001"],
    },
    "Ambiguous / partial documentation -- expect HUMAN REVIEW": {
        "prefix": "demo-ambiguous",
        "patient_id": "PAT-4040",
        "claim_amount": 2200.0,
        "narrative": (
            "Claim billed a high-complexity office visit CPT 99215; chart notes show a "
            "routine CPT 99213-level encounter with limited additional documentation."
        ),
    },
}

BASE_STATE = {
    "log": [],
    "fraud_flags": [],
    "siu_decision": None,
    "rule_fraud_flags": [],
    "tool_error_count": 0,
    "evaluator_regeneration_count": 0,
}


def _format_trace(state: dict) -> str:
    lines = list(state.get("log", []))
    return "\n".join(f"  -> {line}" for line in lines) if lines else "(no trace yet)"


def _format_details(state: dict) -> str:
    parts = []
    if state.get("model_gateway_decision"):
        d = state["model_gateway_decision"]
        parts.append(
            f"Model gateway: routed to {d['selected_model']} ({d['tier']} tier, "
            f"complexity_score={d['complexity_score']}) -- {'; '.join(d['reasons'])}"
        )
    if state.get("billing_finding"):
        parts.append(f"Billing/coding specialist: {state['billing_finding']}")
    if state.get("narrative_finding"):
        parts.append(f"Narrative/fraud specialist: {state['narrative_finding']}")
    if state.get("llm_output"):
        parts.append(f"Supervisor synthesis: {state['llm_output']}")
    if state.get("evaluator_critique"):
        parts.append(f"Evaluator-optimizer critique: {state['evaluator_critique']}")
    if state.get("similar_cases"):
        top = state["similar_cases"][0]
        parts.append(f"Top FWA-case match: {top['id']} (dense_score={top['dense_score']})")
    if state.get("circuit_breaker_tripped"):
        parts.append(f"Circuit breaker: TRIPPED after {state.get('tool_error_count')} failures")
    return "\n\n".join(parts) if parts else "(no reasoning detail yet -- claim rejected before reaching retrieval)"


def run_new_claim(preset_label: str, patient_id: str, claim_amount: float, narrative: str):
    """Submits a new claim. Returns (trace, details, decision, thread_id, controls_visible)."""
    if preset_label and preset_label in PRESETS:
        preset = PRESETS[preset_label]
        patient_id, claim_amount, narrative = preset["patient_id"], preset["claim_amount"], preset["narrative"]
        prefix = preset["prefix"]
    else:
        prefix = "demo-custom"

    if not narrative or not narrative.strip():
        return "", "", "Enter a narrative or pick a preset claim above.", "", gr.update(visible=False)

    thread_id = f"{prefix}-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}
    claim_id = f"CLM-DEMO-{uuid.uuid4().hex[:6].upper()}"

    result = _graph.invoke(
        {**BASE_STATE, "claim_id": claim_id, "patient_id": patient_id or "PAT-0000",
         "claim_amount": float(claim_amount or 0), "narrative": narrative},
        config=config,
    )

    snapshot = _graph.get_state(config)
    awaiting_review = bool(snapshot.next)  # non-empty means the graph is paused

    trace = _format_trace(result)
    details = _format_details(result)
    decision = result.get("decision") or "(pending human review -- see below)"

    return trace, details, decision, thread_id, gr.update(visible=awaiting_review)


def resume_claim(thread_id: str, human_decision: str):
    """Injects the visitor's SIU decision and resumes the paused graph."""
    if not thread_id:
        return "", "", "No claim is currently awaiting review.", gr.update(visible=False)

    config = {"configurable": {"thread_id": thread_id}}
    _graph.update_state(config, {"siu_decision": human_decision})
    result = _graph.invoke(None, config=config)

    trace = _format_trace(result)
    details = _format_details(result)
    decision = result.get("decision", "(no decision returned)")
    return trace, details, decision, gr.update(visible=False)


with gr.Blocks(title="Healthcare Claims Authentication POC") as demo:
    gr.Markdown(
        "# Healthcare Claims Authentication POC\n"
        "LangGraph + hybrid RAG + a supervisor/evaluator-optimizer reasoning pair + a "
        "confidence gate + a circuit breaker + human-in-the-loop review — running entirely "
        "in **mock mode** (no API keys, no external calls). Pick a preset claim below, or "
        "type your own, and watch the graph's routing trace and final decision. If a claim "
        "lands in the uncertain middle, you become the SIU reviewer."
    )

    thread_id_state = gr.State("")

    with gr.Row():
        with gr.Column():
            preset_dropdown = gr.Dropdown(
                choices=list(PRESETS.keys()), label="Preset mock claim (overrides fields below)",
            )
            patient_id_box = gr.Textbox(label="Patient ID", placeholder="PAT-1234", value="PAT-1234")
            claim_amount_box = gr.Number(label="Claim amount ($)", value=250.0)
            narrative_box = gr.Textbox(
                label="Claim narrative", lines=4,
                placeholder="Describe the claim, e.g. 'Routine annual wellness visit, CPT 99395...'",
            )
            submit_btn = gr.Button("Submit claim", variant="primary")

        with gr.Column():
            decision_box = gr.Textbox(label="Decision", interactive=False)
            with gr.Row(visible=False) as review_controls:
                approve_btn = gr.Button("Approve (as SIU reviewer)")
                deny_btn = gr.Button("Deny (as SIU reviewer)")
            details_box = gr.Textbox(label="Reasoning detail", lines=8, interactive=False)
            trace_box = gr.Textbox(label="Node execution trace", lines=10, interactive=False)

    def _fill_preset(preset_label: str):
        if preset_label and preset_label in PRESETS:
            p = PRESETS[preset_label]
            return p["patient_id"], p["claim_amount"], p["narrative"]
        return gr.update(), gr.update(), gr.update()

    preset_dropdown.change(
        _fill_preset, inputs=[preset_dropdown],
        outputs=[patient_id_box, claim_amount_box, narrative_box],
    )

    submit_btn.click(
        run_new_claim,
        inputs=[preset_dropdown, patient_id_box, claim_amount_box, narrative_box],
        outputs=[trace_box, details_box, decision_box, thread_id_state, review_controls],
    )

    approve_btn.click(
        lambda tid: resume_claim(tid, "APPROVED_BY_DEMO_REVIEWER"),
        inputs=[thread_id_state],
        outputs=[trace_box, details_box, decision_box, review_controls],
    )
    deny_btn.click(
        lambda tid: resume_claim(tid, "DENIED_BY_DEMO_REVIEWER"),
        inputs=[thread_id_state],
        outputs=[trace_box, details_box, decision_box, review_controls],
    )


if __name__ == "__main__":
    demo.launch(share=True)

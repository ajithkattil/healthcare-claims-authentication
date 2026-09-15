"""
Shared pytest fixtures for the evals suite.

Adds the repo root to sys.path (so `import claims_auth_hybrid_rag_confidence_circuitbreaker`
works no matter where pytest is invoked from) and provides a fresh graph + checkpointer
per test module, backed by a throwaway SQLite file that's deleted afterward.
"""
import os
import sys
import uuid

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import claims_auth_hybrid_rag_confidence_circuitbreaker as claims  # noqa: E402
from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402


def base_state(claim_id: str, patient_id: str, claim_amount: float, narrative: str) -> dict:
    """The same initial-state shape app.py and the script's own __main__ block use."""
    return {
        "log": [],
        "fraud_flags": [],
        "siu_decision": None,
        "rule_fraud_flags": [],
        "tool_error_count": 0,
        "evaluator_regeneration_count": 0,
        "claim_id": claim_id,
        "patient_id": patient_id,
        "claim_amount": float(claim_amount),
        "narrative": narrative,
    }


@pytest.fixture(scope="module")
def graph(tmp_path_factory):
    """
    One compiled graph per test module, backed by a throwaway checkpoint file.
    Module-scoped (not session-scoped) so test files don't share checkpoint
    state, and not function-scoped since building the graph + BM25 index has
    real (if small) setup cost.
    """
    db_path = tmp_path_factory.mktemp("evals") / f"checkpoints_{uuid.uuid4().hex[:8]}.sqlite"
    claims._cache_init()
    with SqliteSaver.from_conn_string(str(db_path)) as checkpointer:
        yield claims.build_graph(checkpointer)


def invoke_claim(graph, claim_id: str, patient_id: str, claim_amount: float, narrative: str,
                  thread_id: str | None = None) -> tuple[dict, object]:
    """Runs one claim through the graph and returns (result_state, state_snapshot)."""
    thread_id = thread_id or f"eval-{uuid.uuid4().hex[:10]}"
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(base_state(claim_id, patient_id, claim_amount, narrative), config=config)
    snapshot = graph.get_state(config)
    return result, snapshot

"""
LangGraph POC #0 -- Core Concepts: State, Conditional Edges, Checkpointed
Failure Recovery (Healthcare Records Pipeline)

This script is intentionally simple and NOT the claims authentication
workflow -- it's a small healthcare-record intake pipeline used purely to
demonstrate the LangGraph mechanics that the claims scripts (01, 02) build
on top of. Read this one first if you're new to LangGraph.

Pipeline: fetch_patient_record -> validate_record -> [route]
              -> process_record -> summarize_record
              (or) -> flag_incomplete_record -> END

`process_record` is deliberately flaky: it raises on its first invocation
per thread_id, then succeeds. This lets us prove that re-invoking the graph
with the SAME thread_id after a failure does NOT re-run fetch_patient_record
or validate_record (they already completed and were checkpointed) -- it
resumes directly at process_record.

Run: python3 00_concepts_demo_failure_recovery.py
"""

import operator
from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver


# ---------------------------------------------------------------------------
# 1. STATE SCHEMA
# ---------------------------------------------------------------------------
class RecordState(TypedDict):
    log: Annotated[list[str], operator.add]
    raw_record: str
    complete: bool
    processed_record: str
    summary: str


# ---------------------------------------------------------------------------
# 2. NODE FUNCTIONS
# ---------------------------------------------------------------------------
# Simulates a flaky downstream system (e.g. an EHR integration timing out).
_flaky_attempts: dict[str, int] = {}


def fetch_patient_record(state: RecordState) -> dict:
    print("  -> [fetch_patient_record] executing")
    data = state.get("raw_record") or "patient_id=PAT-4471, procedure_code=99213, diagnosis=complete"
    return {
        "raw_record": data,
        "log": ["fetch_patient_record: pulled intake record"],
    }


def validate_record(state: RecordState) -> dict:
    print("  -> [validate_record] executing")
    is_complete = "diagnosis=incomplete" not in state["raw_record"]
    return {
        "complete": is_complete,
        "log": [f"validate_record: complete={is_complete}"],
    }


def route_after_validation(state: RecordState) -> str:
    return "process_record" if state["complete"] else "flag_incomplete_record"


def flag_incomplete_record(state: RecordState) -> dict:
    print("  -> [flag_incomplete_record] executing")
    return {"log": ["flag_incomplete_record: sent back to intake for completion"]}


def process_record(state: RecordState, config) -> dict:
    thread_id = config["configurable"]["thread_id"]
    attempts = _flaky_attempts.get(thread_id, 0)
    _flaky_attempts[thread_id] = attempts + 1

    print(f"  -> [process_record] executing (attempt #{attempts + 1})")
    if attempts == 0:
        # Simulates a transient failure: EHR system timeout, coding-lookup
        # service 500, etc.
        raise RuntimeError("simulated transient failure in process_record")

    return {
        "processed_record": "patient_id=PAT-4471 normalized, procedure coded",
        "log": [f"process_record: succeeded on attempt #{attempts + 1}"],
    }


def summarize_record(state: RecordState) -> dict:
    print("  -> [summarize_record] executing")
    return {
        "summary": f"Record intake complete. Steps run: {len(state['log'])}",
        "log": ["summarize_record: done"],
    }


# ---------------------------------------------------------------------------
# 3. BUILD THE GRAPH
# ---------------------------------------------------------------------------
def build_graph(checkpointer):
    g = StateGraph(RecordState)

    g.add_node("fetch_patient_record", fetch_patient_record)
    g.add_node("validate_record", validate_record)
    g.add_node("flag_incomplete_record", flag_incomplete_record)
    g.add_node("process_record", process_record)
    g.add_node("summarize_record", summarize_record)

    g.add_edge(START, "fetch_patient_record")
    g.add_edge("fetch_patient_record", "validate_record")

    g.add_conditional_edges(
        "validate_record",
        route_after_validation,
        {"process_record": "process_record", "flag_incomplete_record": "flag_incomplete_record"},
    )

    g.add_edge("flag_incomplete_record", END)
    g.add_edge("process_record", "summarize_record")
    g.add_edge("summarize_record", END)

    return g.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# 4. RUN IT -- demonstrate failure + resume-only-the-failed-step
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    with SqliteSaver.from_conn_string("poc_checkpoints.sqlite") as checkpointer:
        graph = build_graph(checkpointer)

        thread_id = "demo-thread-1"
        config = {"configurable": {"thread_id": thread_id}}

        print("\n=== RUN 1 (expected to fail inside process_record) ===")
        try:
            result = graph.invoke({"log": []}, config=config)
            print("Result:", result)
        except Exception as e:
            print(f"  !! Run failed as expected: {e}")

        print("\n--- Checkpointed state after failure ---")
        snapshot = graph.get_state(config)
        print("  Next node(s) to run on resume:", snapshot.next)
        print("  State so far:", {k: v for k, v in snapshot.values.items() if k != "log"})
        print("  Log so far:", snapshot.values.get("log"))

        print("\n=== RUN 2 (same thread_id -> should RESUME, not restart) ===")
        # No fetch_patient_record / validate_record prints below -> proof
        # that only process_record (the failed node) and summarize_record
        # (downstream) re-run.
        result = graph.invoke(None, config=config)
        print("\nFinal result:", result)

        print("\n=== BONUS: a fresh thread_id hitting the incomplete-record branch ===")
        config2 = {"configurable": {"thread_id": "demo-thread-2"}}
        result2 = graph.invoke(
            {"log": [], "raw_record": "patient_id=PAT-9002, procedure_code=45378, diagnosis=incomplete"},
            config=config2,
        )
        print("Result:", result2)

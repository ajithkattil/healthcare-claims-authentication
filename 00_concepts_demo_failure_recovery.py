"""
LangGraph POC — checkpointing, conditional routing, and partial re-execution
after failure.

Pipeline: fetch_data -> validate_data -> [route] -> process_data -> summarize
                                              (or)-> handle_invalid -> END

`process_data` is deliberately flaky: it raises on its first invocation per
thread_id, then succeeds. This lets us show that when we re-invoke the graph
with the SAME thread_id after a failure, LangGraph does NOT re-run
fetch_data or validate_data (they already completed and were checkpointed)
-- it resumes directly at process_data.

Run: python3 langgraph_poc.py
"""

import operator
from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver


# ---------------------------------------------------------------------------
# 1. STATE SCHEMA
# ---------------------------------------------------------------------------
# `log` uses operator.add as a reducer -> every node's return value for
# `log` gets APPENDED to the existing list rather than overwriting it.
# `status`/`raw_data`/`validated`/`summary` are plain fields -> last write wins.
class PipelineState(TypedDict):
    log: Annotated[list[str], operator.add]
    raw_data: str
    validated: bool
    processed_data: str
    summary: str


# ---------------------------------------------------------------------------
# 2. NODE FUNCTIONS
# ---------------------------------------------------------------------------
# Simulates an external system that's flaky exactly once per thread.
# In a real POC this stands in for "the tool call that times out sometimes".
_flaky_attempts: dict[str, int] = {}


def fetch_data(state: PipelineState) -> dict:
    print("  -> [fetch_data] executing")
    # If the caller seeded raw_data (see demo-thread-2 below), keep it;
    # otherwise simulate pulling a normal, valid record.
    data = state.get("raw_data") or "order_id=42, amount=100"
    return {
        "raw_data": data,
        "log": ["fetch_data: pulled raw record"],
    }


def validate_data(state: PipelineState) -> dict:
    print("  -> [validate_data] executing")
    is_valid = "amount=-15" not in state["raw_data"]  # intentionally invalid
    return {
        "validated": is_valid,
        "log": [f"validate_data: valid={is_valid}"],
    }


def route_after_validation(state: PipelineState) -> str:
    # This function IS the "decision logic between agents/states".
    # It reads state and returns the name of the next node.
    return "process_data" if state["validated"] else "handle_invalid"


def handle_invalid(state: PipelineState) -> dict:
    print("  -> [handle_invalid] executing")
    return {"log": ["handle_invalid: flagged bad record, routed to review queue"]}


def process_data(state: PipelineState, config) -> dict:
    thread_id = config["configurable"]["thread_id"]
    attempts = _flaky_attempts.get(thread_id, 0)
    _flaky_attempts[thread_id] = attempts + 1

    print(f"  -> [process_data] executing (attempt #{attempts + 1})")
    if attempts == 0:
        # Simulates a transient failure: timeout, 500 from a downstream API, etc.
        raise RuntimeError("simulated transient failure in process_data")

    return {
        "processed_data": "order_id=42 normalized",
        "log": [f"process_data: succeeded on attempt #{attempts + 1}"],
    }


def summarize(state: PipelineState) -> dict:
    print("  -> [summarize] executing")
    return {
        "summary": f"Pipeline complete. Steps run: {len(state['log'])}",
        "log": ["summarize: done"],
    }


# ---------------------------------------------------------------------------
# 3. BUILD THE GRAPH
# ---------------------------------------------------------------------------
def build_graph(checkpointer):
    g = StateGraph(PipelineState)

    g.add_node("fetch_data", fetch_data)
    g.add_node("validate_data", validate_data)
    g.add_node("handle_invalid", handle_invalid)
    g.add_node("process_data", process_data)
    g.add_node("summarize", summarize)

    g.add_edge(START, "fetch_data")
    g.add_edge("fetch_data", "validate_data")

    # Conditional edge: router function decides the next node at runtime.
    g.add_conditional_edges(
        "validate_data",
        route_after_validation,
        {"process_data": "process_data", "handle_invalid": "handle_invalid"},
    )

    g.add_edge("handle_invalid", END)
    g.add_edge("process_data", "summarize")
    g.add_edge("summarize", END)

    return g.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# 4. RUN IT — demonstrate failure + resume-only-the-failed-step
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    with SqliteSaver.from_conn_string("poc_checkpoints.sqlite") as checkpointer:
        graph = build_graph(checkpointer)

        thread_id = "demo-thread-1"
        config = {"configurable": {"thread_id": thread_id}}

        print("\n=== RUN 1 (expected to fail inside process_data) ===")
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
        # Note: no fetch_data / validate_data prints below -> proof that only
        # process_data (the failed node) and summarize (downstream) re-run.
        result = graph.invoke(None, config=config)  # None input = resume as-is
        print("\nFinal result:", result)

        print("\n=== BONUS: a fresh thread_id hitting the invalid-data branch ===")
        # Same raw_data is hardcoded invalid in this POC, so this always
        # routes to handle_invalid -- showing the conditional edge in action.
        config2 = {"configurable": {"thread_id": "demo-thread-2"}}
        result2 = graph.invoke(
            {"log": [], "raw_data": "order_id=99, amount=-15"}, config=config2
        )
        print("Result:", result2)

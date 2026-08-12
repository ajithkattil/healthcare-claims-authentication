# healthcare-claims-authentication
This POC demonstrates an agentic orchestration approach to claims authentication
# Claims Authentication POC — LangGraph + Cohere + Pinecone + Zapier

A proof-of-concept agentic workflow for insurance claims authentication, built on
[LangGraph](https://github.com/langchain-ai/langgraph). Demonstrates state management,
conditional routing, checkpoint-based failure recovery, and a human-in-the-loop
approval gate, with optional live integrations to Cohere, Pinecone, and Zapier.

## Files in this project

| File | Purpose |
|---|---|
| `00_concepts_demo_failure_recovery.py` | Standalone LangGraph concepts demo: state, conditional edges, and — the core mechanic — proving that a re-invoked graph resumes only the failed node, not the whole run. Not claims-specific; read this first if you're new to LangGraph. |
| `01_claims_auth_basic.py` | The claims authentication graph (identity → policy → fraud → approve/investigate) with a human-in-the-loop interrupt, using purely mock/rule-based fraud detection. No external services required. |
| `02_claims_auth_with_cohere_pinecone_zapier.py` | The full version: adds Cohere embeddings of the claim narrative, a Pinecone similarity search against known fraud cases, and a Zapier webhook notification when a claim is flagged. This is the primary deliverable. |
| `architecture_diagram.png` | Architecture diagram referenced in the accompanying Word documents. |
| `requirements.txt` | Python dependencies. |
| `.env.example` | Template for API keys, only needed if you flip to live mode. |

## Prerequisites

- Python 3.10 or later (developed and tested on 3.12)
- pip
- No external accounts needed to run the POC as delivered (mock mode)
- Optional, for live mode: a Cohere API key, a Pinecone API key + index, and a Zapier webhook URL

## Install

```bash
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run — mock mode (default, no API keys needed)

```bash
python3 00_concepts_demo_failure_recovery.py
python3 01_claims_auth_basic.py
python3 02_claims_auth_with_cohere_pinecone_zapier.py
```

Each script is self-contained and prints its own trace of which node executed, in
order, plus the final decision for each sample claim it runs. Each also creates a
local SQLite file (e.g. `claims_v2_checkpoints.sqlite`) holding the persisted graph
state — safe to delete between runs if you want a clean slate.

### What to look for in each script's output

**`00_concepts_demo_failure_recovery.py`**
- `RUN 1` deliberately fails inside `process_data`.
- The printed "Next node(s) to run on resume" confirms the checkpoint knows exactly
  where execution stopped.
- `RUN 2` reuses the same `thread_id` — notice `fetch_data` and `validate_data` do
  **not** print again. Only the failed node and everything downstream re-execute.

**`01_claims_auth_basic.py`**
- Claim A: clean claim → auto-approved, no human involved.
- Claim B: bad claimant ID → rejected immediately, never reaches fraud checks.
- Claim C: high-value claim → flagged, graph **pauses** before `route_to_investigator`
  (`snapshot.next` shows the graph is genuinely suspended). We simulate a human
  decision via `graph.update_state()`, then resume with `graph.invoke(None, config)`.

**`02_claims_auth_with_cohere_pinecone_zapier.py`**
- Claim D: clean narrative → sails through embedding + similarity check → approved.
- Claim E: narrative text matches a mock "known fraud" case closely enough to clear
  the similarity threshold → flagged → Zapier mock prints the payload it would send →
  graph pauses for human review → we inject a `DENIED` decision → resume → finalized.

## Switching to live APIs

Live calls are **not** required to evaluate this POC's logic — mock mode exercises
every node, every routing decision, and the human-in-the-loop pause exactly as live
mode would, just with deterministic stand-ins for the three external calls.

If you do want to run against real services:

1. Copy `.env.example` to `.env` and fill in the four values.
2. Load them into your environment (e.g. `export $(cat .env | xargs)` on macOS/Linux,
   or use `python-dotenv` if you prefer to load them in-script).
3. In `02_claims_auth_with_cohere_pinecone_zapier.py`, change:
   ```python
   USE_LIVE_APIS = False
   ```
   to:
   ```python
   USE_LIVE_APIS = True
   ```
4. **Cohere**: no additional setup — the trial API key works out of the box for
   low-volume testing.
5. **Pinecone**: you must create an index before running live. The embedding model
   used (`embed-english-v3.0`) returns 1024-dimensional vectors, so create the index
   with `dimension=1024`, `metric="cosine"`. Example, using the Pinecone console or:
   ```python
   from pinecone import Pinecone, ServerlessSpec
   pc = Pinecone(api_key="...")
   pc.create_index(
       name="claims-fraud-cases",
       dimension=1024,
       metric="cosine",
       spec=ServerlessSpec(cloud="aws", region="us-east-1"),
   )
   ```
   You'll also need to upsert your real historical fraud-case embeddings into that
   index — the mock's in-memory dictionary of two sample cases is standing in for
   that data.
6. **Zapier**: create a Zap with a "Catch Hook" trigger step, copy its unique webhook
   URL into `ZAPIER_WEBHOOK_URL`. What happens downstream of that hook (Slack message,
   email, ticket creation) is configured entirely in Zapier's UI, not in this code.

## Testing checklist

Run through these to confirm the POC behaves as documented:

- [ ] `00_...py` Run 1 fails inside `process_data`, Run 2 shows only `process_data`
      and `summarize` re-executing (no repeated `fetch_data`/`validate_data` prints)
- [ ] `01_...py` Claim A prints `APPROVED`
- [ ] `01_...py` Claim B prints `REJECTED` and never prints `check_policy_coverage`
- [ ] `01_...py` Claim C prints "Graph is paused" with `Next node(s): ('route_to_investigator',)`
      before the human decision is injected, and prints the injected decision after resume
- [ ] `02_...py` Claim D shows `embed_claim_narrative` and `check_similar_fraud_cases`
      both executing, then `APPROVED`
- [ ] `02_...py` Claim E shows a non-empty `fraud_flags` containing
      `similar_to_known_fraud:...`, the Zapier mock print, the pause, and the final
      `DENIED_BY_INVESTIGATOR` decision after resume
- [ ] Deleting the `.sqlite` checkpoint files between runs produces identical output
      (proves no hidden state leaks between runs)
- [ ] Re-running a script twice **without** deleting the checkpoint file for a
      `thread_id` that already reached `END` is a no-op re-fetch of the final state,
      not a re-execution (LangGraph will not re-run a completed thread)

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `ModuleNotFoundError: No module named 'langgraph'` | Virtual environment not activated, or `pip install -r requirements.txt` not run |
| `pinecone-client` install error / deprecation exception on import | Use the `pinecone` package, not `pinecone-client` — this repo's `requirements.txt` already specifies the correct one |
| Graph "resumes" but re-runs everything from scratch | You changed the `thread_id` between calls, or deleted the `.sqlite` file — the checkpointer has no history for a new thread |
| `graph.invoke(None, config)` raises `KeyError` on thread | You must call the graph at least once with real input for a given `thread_id` before you can resume it with `None` |
| Live Cohere/Pinecone/Zapier calls fail with connection errors | Confirm `USE_LIVE_APIS = True`, your `.env` is loaded, and your network allows outbound HTTPS to `api.cohere.ai`, `api.pinecone.io`, and `hooks.zapier.com` |
| Pinecone query returns empty matches in live mode | Confirm you've upserted vectors into the index and the index dimension matches your embedding model's output dimension |

## Known limitations (by design, for POC scope)

- Fraud detection logic is deterministic/rule-based, not a trained model
- No authentication, encryption, or PII-handling hardening — do not point this at
  real claimant data as-is
- SQLite checkpointing is fine for a single-process POC; a concurrent production
  deployment should move to a Postgres-backed checkpointer
- The Zapier notification call is not idempotent — a checkpoint replay after a crash
  immediately following a successful Zapier call could send a duplicate notification

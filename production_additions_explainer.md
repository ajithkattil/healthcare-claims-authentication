# Production Additions — Implementation Notes

Covers the seven architecture elements from the design doc that aren't in POC scripts 00–05: **gateway/auth, multi-tenancy, PII tokenization, low-risk pre-filter, long-term memory, prompt/policy versioning, RAGAS/DeepEval.** Each section below sketches the implementation shape — key classes, where it sits in the graph, and the one or two design decisions worth calling out.

---

## 1. API / Model Gateway + Auth

**Where it sits:** a thin FastAPI service in front of the LangGraph app — every claim passes through it exactly once, before `intake`.

**Implementation shape:**
```python
class GatewayContext(BaseModel):
    tenant_id: str      # resolved from OIDC token claims, never request body
    trace_id: str        # generated here, threaded through every span
    model_tier: str       # decided by policy engine (see #4)

@app.middleware("http")
async def gateway_middleware(request, call_next):
    token = verify_oidc(request.headers["Authorization"])
    ctx = GatewayContext(
        tenant_id=token.claims["tenant_id"],   # trust boundary: token, not body
        trace_id=str(uuid4()),
    )
    request.state.ctx = ctx
    return await call_next(request)
```
**Design rationale:** `tenant_id` comes from the verified token claim, not a request field — a request body is user-controllable, so trusting it there would be a cross-tenant leak waiting to happen. Rate limiting (token-bucket per tenant) and the PII redaction proxy (see #3) also live here, so every node downstream inherits them for free instead of re-implementing per node.

---

## 2. Multi-Tenancy Isolation

**Where it sits:** `tenant_id` is set once in `ClaimState` at intake, then read (never re-derived) by every node that touches storage.

**Implementation shape:**
```python
class ClaimState(TypedDict):
    tenant_id: str
    ...

def get_vector_index(tenant_id: str) -> Pinecone.Index:
    return pinecone_client.Index(f"fwa-cases-{tenant_id}")  # separate index, not shared+filtered
```
**Design rationale:** separate Pinecone index per tenant, not one shared index with a `tenant_id` metadata filter. Costs more to operate, but it's the difference between "one payer's data is filtered out" and "one payer's data is physically unreachable" — the latter is what a HIPAA auditor wants to hear. Same logic applies to the Postgres claim-history table (row-level tenant scoping) and to observability (every span tagged `tenant_id` at instrumentation time).

---

## 3. PII/PHI Perimeter Tokenization

**Where it sits:** in the gateway (structured fields) and in the input-guardrail node (free-text narrative).

**Implementation shape:**
```python
class Tokenizer:
    """Backed by a vault service; reversible only via a separate, RBAC-gated resolve call."""
    def tokenize(self, raw: str, field: str) -> str: ...
    def resolve(self, token: str, requester_role: str) -> str:
        assert requester_role in ALLOWED_RESOLVER_ROLES  # separately audited
        ...
```
Every downstream service — cache, vector store, LLM prompt, Zapier payload — operates only on the token. The narrative-text guardrail additionally regex/NER-scans for PHI typed directly into free text (structural tokenization only catches known fields).

**Design rationale:** tokenize at the perimeter, operate on tokens everywhere inside, resolve back to PHI only at systems of record. Token resolution is its own audited RBAC operation — never an implicit side effect of another action.

---

## 4. Low-Risk Pre-Filter (Model-Tier Routing)

**Where it sits:** at the gateway, evaluated before the graph even reaches retrieval/LLM nodes.

**Implementation shape:**
```python
def assign_tier(claim: ClaimState) -> str:
    if claim["passed_all_rule_checks"] and claim["billed_amount"] < LOW_RISK_THRESHOLD:
        return "skip_llm"          # pre-filtered straight to auto-approve
    if claim["retrieval_confidence"] > 0.85 and claim["billed_amount"] < MID_THRESHOLD:
        return "haiku"
    if claim["provider_under_investigation"]:
        return "human"
    return "sonnet"
```
**Design rationale:** this is a cost lever, not just a routing convenience — most claim volume is low-risk and shouldn't touch the LLM at all. It's also why capacity scales: the LLM call is the bottleneck, and the pre-filter is what keeps it off the critical path for the majority of traffic.

---

## 5. Long-Term Memory (Claim History + FWA Feedback Loop)

**Where it sits:** two stores, separate from the LangGraph checkpointer (which is short-term, per-thread, disposable).

**Implementation shape:**
```python
# Relational — cross-claim pattern lookups
class ClaimHistory(Base):
    tenant_id: str
    tokenized_patient_id: str
    tokenized_provider_id: str
    decision: str
    timestamp: datetime

# Append-only vector namespace — grows only from confirmed human decisions
def on_escalation_closed(claim: ClaimState, reviewer_decision: str):
    if reviewer_decision == "confirmed_fraud":
        fwa_index.upsert(vector=claim["narrative_embedding"], metadata={"tenant_id": ...})
```
**Design rationale:** this is the feedback loop that makes the system get better over time — every SIU reviewer decision that confirms fraud becomes retrieval signal for every future claim, without retraining anything. It's also why it's architecturally separate from the checkpointer: checkpointing needs to be fast/disposable, long-term memory needs to be queryable across all threads.

---

## 6. Prompt & Policy Versioning

**Where it sits:** an external store (Postgres table or config service), loaded by name+version at runtime — not compiled into the script.

**Implementation shape:**
```python
class PromptVersion(Base):
    name: str
    version: int
    text: str
    author: str
    change_reason: str
    active: bool

def load_prompt(name: str) -> str:
    return db.query(PromptVersion).filter_by(name=name, active=True).one().text
```
Every trace records which prompt version was active, so a faithfulness-score change is attributable to a specific edit, not unexplained drift. A/B testing is just routing a percentage of traffic to a second `active` row and comparing evaluation scores before promoting.

**Design rationale:** decouples prompt tuning from CI/CD — a wording fix ships as a database write, not a redeploy, which matters because prompts change on a "we found a failure mode" cadence, faster than any release train.

---

## 7. Continuous Evaluation (RAGAS + DeepEval)

**Where it sits:** an offline weekly job, not inline in the request path.

**Implementation shape:**
```python
def weekly_eval_job():
    sample = sample_production_claims(pct=0.05)
    ragas_scores = ragas.evaluate(sample, metrics=[context_precision, faithfulness, answer_relevancy])
    bias_report = deepeval.evaluate_bias(sample, slice_by=["provider_type", "geography"])
    if ragas_scores["faithfulness"].mean() < FAITHFULNESS_FLOOR:
        alert_observability_team()
```
**Design rationale:** this turns "is the reasoning still grounded" into a trend line instead of a launch-day guess — and running it on a *sample*, offline, keeps it from adding latency or cost to every claim.

---

## Open questions worth tracking

- **Why aren't all seven built into the POC?** The POC's job (00–05) was to prove the orchestration and retrieval mechanics work; these seven are infra/platform concerns that represent real engineering effort beyond that scope — they're documented and designed here, not hand-waved or skipped silently. See ROADMAP.md for where each one is tracked.
- **Build order for a real rollout:** the gateway + multi-tenancy + tokenization first, because together they're the compliance floor — nothing else can go live without it.
- **The trickiest interaction to get right:** the confidence gate's interaction with the low-risk pre-filter and model-tier table — three independent three-way decisions (tier, gate, circuit breaker) that have to agree on what "escalate" means, or a claim can silently fall through a gap between them.

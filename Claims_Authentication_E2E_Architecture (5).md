# Healthcare Claims Authentication Platform
## Comprehensive End-to-End Architecture

---

## 1. STAR Narrative

**Situation.** Health-plan claims processing relies heavily on manual review for member identity verification, coverage validation, and fraud/waste/abuse (FWA) triage. Reviewer judgment is inconsistent, review capacity doesn't scale with claim volume, and a human reviewer has no fast way to check a new claim's narrative against thousands of prior confirmed FWA cases, or to reason over a claim against the actual current billing guideline text rather than just its coded fields. This is not a document-summarization problem or a single-call chatbot problem — it's a multi-step operational workflow requiring eligibility verification, code validation, and nuanced reasoning over billing patterns, at a volume no single LLM call can responsibly carry.

**Task.** Design and build a platform — not a chatbot — that automates the mechanical, high-volume parts of claims authentication while making its own confidence explicit: auto-deciding only where the evidence is strong, and routing everything else, including any tool or model failure, to a human Special Investigations Unit (SIU) reviewer. The platform must work across tenant boundaries with HIPAA-grade isolation, be auditable end-to-end, safe with PHI by construction, cost-aware across model tiers, continuously measurable rather than measured only at launch, and reusable as a template beyond claims.

**Action.** The platform is a LangGraph-orchestrated agentic workflow behind an identity-aware API/model gateway. Deterministic checks (identity format, coverage limits) run first and cheaply — no LLM involvement where none is needed. A hybrid retrieval layer (dense + BM25, fused) checks the claim narrative against known-FWA case embeddings and current billing-guideline text, grounding an LLM reasoning call in retrieved source text rather than the model's unaided judgment. A confidence-driven three-way gate — auto-approve, auto-reject, or escalate — reads the LLM's stated confidence alongside retrieval strength. A circuit breaker (tool reliability) and guardrails (content validity) are two independent, non-overlapping safety mechanisms, so a down vector database is never mistaken for a security event and a prompt-injection attempt is never treated as a retryable blip. Escalated claims pause the graph via a human-in-the-loop interrupt; a reviewer sees the full accumulated state and injects a decision, and the graph resumes from exactly that point — nothing upstream re-executes. Every tenant gets its own isolated retrieval index and its own slice of every log and trace. Prompts and model-tier policy are externalized and versioned, so tuning either is a config change, not a redeploy. A weekly sampled evaluation pass (RAGAS + DeepEval) turns "does the reasoning still hold up" into a trend line, not a guess.

**Result.** A claim moves from submission to an auditable decision without a human touching the routine cases, while every escalation carries the exact evidence a reviewer needs rather than a bare "flagged" label. Hybrid retrieval measurably improves what the LLM is grounded in over dense-only search. Unhandled failures become explicit escalations instead of silent wrong answers. Prompt and policy changes ship in hours, not days. Onboarding a new tenant is a provisioning exercise, not an engineering project. The same node/routing/middleware skeleton is reusable for other agentic workflows by swapping domain-specific nodes rather than rebuilding orchestration, gateway, guardrail, or observability code.

**The lesson underneath all of it:** production AI reliability is not an infrastructure problem — it's an architecture problem. You design for failure at the agent level, not just handle it at the infrastructure level, and you measure quality continuously, not just at launch.

---

## 2. Platform Feature Set

- **Identity & coverage verification** as discrete, auditable, deterministic steps — no LLM involvement, no ambiguity.
- **Hybrid retrieval** (dense vector similarity + BM25 keyword search, fused) against two separate corpora: known-FWA case embeddings, and current billing-guideline text.
- **RAG-grounded LLM reasoning** — the model answers only from retrieved guideline excerpts placed directly in its prompt, with an explicit instruction not to reason beyond what's provided.
- **Confidence-driven three-way decision gate** — auto-approve, auto-reject, or escalate to a human, based on the LLM's stated confidence and retrieval strength together, not either alone.
- **Input and output guardrails**, independent of each other, both fail-closed.
- **A circuit breaker** distinct from the guardrails, tracking tool (Pinecone/Cohere) reliability and escalating after repeated failures without retrying indefinitely, plus per-step timeout enforcement and structured fallback paths on every critical tool call.
- **A persistent, hash-keyed embedding cache** to avoid redundant embedding calls for repeated or resubmitted narratives.
- **Checkpointed state and failure recovery** — a run resumes at the exact node that failed, never from scratch.
- **Human-in-the-loop interrupts** that pause the graph before a routing decision and resume once a reviewer injects a decision, with the reviewer seeing full state, not a summary.
- **Structure-aware document chunking** for the guideline corpus, so one chunk maps to one numbered rule rather than an arbitrary word-count boundary.
- **Model-tier routing** at the gateway, plus a **low-risk pre-filter** so claims passing every rule-based check below a billed-amount threshold skip the LLM reasoning node entirely.
- **Tenant-isolated multi-tenancy** — separate retrieval indexes and tokenized identity per payer, enforced from the token claim at ingestion, not a request-body field.
- **Perimeter PII/PHI tokenization** — every internal service, cache, and third-party call operates on tokens, never raw identifiers.
- **A continuous evaluation loop** (RAGAS + DeepEval), not just launch-time testing — weekly sampled scoring with alert thresholds.
- **Prompt and policy versioning**, decoupled from application code, with A/B testing and per-trace attribution.
- **Full audit trail** — every node's execution, every routing decision, every tool call is traced and logged, partitioned per tenant.
- **Config-driven reusability** — the orchestration, gateway, guardrail, and observability layers are domain-agnostic; a new use case is a new state schema and node set, not new infrastructure.

---

## 3. Technology Stack

| Layer | Technology | Role |
|---|---|---|
| Orchestration | LangGraph (Python) | State machine, conditional routing, checkpointing |
| LLM reasoning | Claude (tiered: Haiku / Sonnet / Opus) | Guideline-grounded fraud/anomaly reasoning |
| Embeddings | Cohere `embed-english-v3.0` | Narrative and guideline chunk embeddings |
| Dense retrieval | Pinecone (namespaced per corpus, per tenant) | Vector similarity search |
| Keyword retrieval | `rank_bm25` (local, in-process) | Exact-term matching over the same chunk set |
| Fusion | Weighted min-max normalized sum + Reciprocal Rank Fusion | Combines dense + keyword rankings |
| Checkpoint / short-term state | Postgres (SQLite in POC) via LangGraph's checkpoint API | Per-claim run state, resumable on failure |
| Long-term memory | Postgres (claim history) + vector namespace (confirmed-FWA embeddings) | Cross-claim pattern memory |
| Cache | Postgres/Redis, hash-keyed | Embedding cache, idempotency, rate-limit counters |
| Notification | Zapier (webhook) | SIU notification, ID/flags only, no narrative |
| Gateway | Custom service (Envoy/API-gateway-fronted) | AuthN/Z, routing, PII redaction, model tier selection |
| Identity | OIDC/SAML SSO, mTLS service-to-service | User and service authentication |
| Evaluation | RAGAS + DeepEval | Context precision, faithfulness, answer relevancy, bias slicing |
| Observability | Arize Phoenix–style span tracing | LLM tracing, span-level debugging, per-tenant dashboards |
| Prompt/policy store | Versioned external store (name, version, text, author, timestamp, change reason) | Runtime-loaded, no-redeploy prompt and tier-policy changes |
| Deployment | EKS, ArgoCD, Terraform, GitLab CI, Backstage | Container orchestration, GitOps, IaC, CI, catalog |

---

## 4. Architecture Diagram

```
                                   ┌───────────────────────────┐
                                   │   Provider portal / EDI    │
                                   │   837 batch / Internal UI  │
                                   └─────────────┬───────────────┘
                                                 ▼
 ┌───────────────────────────────────────────────────────────────────────────┐
 │  IDENTITY & ACCESS          OIDC/SAML SSO · mTLS service auth ·           │
 │                              tenant_id resolved from token claims         │
 └─────────────────────────────────┬───────────────────────────────────────┘
                                   ▼
 ┌───────────────────────────────────────────────────────────────────────────┐
 │  API / MODEL GATEWAY                                                      │
 │   • rate limiting (per tenant, token bucket)                             │
 │   • PII/PHI redaction proxy → tokenized payload                          │
 │   • model-tier policy engine + low-risk pre-filter                      │
 │   • fallback chain across model providers/tiers                         │
 │   • trace_id + audit envelope attached, tagged with tenant_id            │
 └─────────────────────────────────┬───────────────────────────────────────┘
                                   ▼
 ┌───────────────────────────────────────────────────────────────────────────┐
 │  INPUT GUARDRAILS  (fail-closed, ordered)                                 │
 │   1. schema validation → 2. PHI-in-free-text scan →                     │
 │   3. prompt-injection detection → 4. policy allow/deny                   │
 └─────────────────────────────────┬───────────────────────────────────────┘
                                   ▼
 ┌───────────────────────────────────────────────────────────────────────────┐
 │  ORCHESTRATION (LangGraph StateGraph — patient-file-style shared state,  │
 │  checkpointed after every node; per-tenant Pinecone index)         │
 │                                                                            │
 │   intake → verify_identity → check_coverage ─┐                           │
 │                                                 ▼                          │
 │                                      code_validation (rules-based)       │
 │                                                 ▼                          │
 │                      ┌─────── low-risk pre-filter (skip LLM) ───────┐   │
 │                      ▼                                                │   │
 │              embed_narrative → check_similar_fraud_cases              │   │
 │                                → retrieve_guidelines (hybrid RAG)      │   │
 │                                 ▼                                       │   │
 │                          llm_fraud_reasoning (Claude, tiered)          │   │
 │                                 ▼                                       │   │
 │                          confidence_decision_gate ◄────────────────────┘   │
 │                        ┌────────────┼────────────┐                        │
 │                    APPROVE       REJECT       ESCALATE                    │
 └────────────────────────┴────────────┴────────────┴──────────────────────┘┘
        │                    │                    │                  │
        ▼                    ▼                    ▼                  ▼
 ┌───────────┐        ┌────────────┐       ┌─────────────┐    ┌───────────────┐
 │ SHORT-TERM│        │  RAG /     │       │  MODEL       │    │  HUMAN-IN-THE-│
 │ MEMORY    │        │  RETRIEVAL │       │  ROUTER      │    │  LOOP          │
 │ (thread   │        │  (chunked  │       │  (tier       │    │  interrupt_    │
 │ checkpoint│        │  guideline │       │  select +    │    │  before, SIU   │
 │ state,    │        │  corpus +  │       │  fallback    │    │  reviewer sees │
 │ Postgres) │        │  FWA case  │       │  chain)      │    │  full state,   │
 │           │        │  embeddings│       │              │    │  update_state()│
 └───────────┘        └────────────┘       └─────────────┘    └───────────────┘
        │                    │                    │                  │
        └────────────────────┴─────────┬──────────┴──────────────────┘
                                        ▼
                     ┌───────────────────────────────────────┐
                     │  LONG-TERM MEMORY / CACHE               │
                     │  embedding cache (hash-keyed) ·          │
                     │  per-patient/provider claim history ·    │
                     │  confirmed-FWA vector store (append-only)│
                     └───────────────────────┬───────────────────┘
                                            ▼
 ┌───────────────────────────────────────────────────────────────────────────┐
 │  OUTPUT GUARDRAILS  → schema-validate decision → fail-closed to human     │
 └─────────────────────────────────┬───────────────────────────────────────┘
                                   ▼
 ┌───────────────────────────────────────────────────────────────────────────┐
 │  EVALUATION & OBSERVABILITY                                               │
 │   RAGAS + DeepEval weekly sample · Arize-style span tracing ·           │
 │   append-only audit log, partitioned per tenant                         │
 └───────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Component Deep Dive

### 5.1 Channel & Ingestion

Claims enter through one of three channels — a provider self-service portal, a batch EDI 837 feed, or an internal reviewer UI for manually keyed claims. All three normalize into the same intake contract before anything else happens: a claim is parsed into its 14+ structured fields (patient identifier, provider identifier, CPT/ICD codes, claim amount, service dates, free-text narrative) and initialized into the shared `ClaimState`. The intake node also enriches the state with provider specialty and reimbursement rate — but only conditionally, fetched when a downstream node actually needs it, not upfront for every claim; keeping state lean at intake is a deliberate design decision, not an oversight. Keeping this parse step thin and channel-specific, while everything downstream is channel-agnostic, is what lets a new channel be added without touching the orchestration graph at all.

### 5.2 Identity & Access Management

Every request carries an OIDC token; the gateway validates it and resolves `tenant_id` **from the token's claims**, never from anything in the request body — a request body field is user-controllable and therefore not a trust boundary. Service-to-service calls (gateway → orchestrator → retrieval/LLM services) use mutual TLS with short-lived certificates rather than static API keys. Role-based access control gates two specific privileged operations distinctly from ordinary claim submission: **PII token resolution** and **`update_state()` for a paused claim** (the SIU reviewer action) — both audited individually, since both are points where the system's usual safeguards are intentionally bypassed by design.

### 5.3 API / Model Gateway

The gateway is the single choke point every claim passes through exactly once, and it does the jobs that would otherwise be duplicated across every node if left to the orchestrator: **rate limiting** (token-bucket, per tenant), **PII redaction** (Section 5.13), a **model-tier policy engine** (Section 5.8) including the **low-risk pre-filter**, a **fallback chain** across model providers/tiers if the assigned tier errors or times out, and an **audit envelope** — a `trace_id` stamped at the gateway and tagged with `tenant_id`, threading through every downstream span so one claim's full path is reconstructable from one ID, and one payer's dashboard never shows another payer's traces.

### 5.4 Orchestration Engine — The Hospital-Triage Model

A single LLM call takes input and produces output; it cannot pause, check something, make a conditional decision, loop back on failure, or hand off to another system. A claims workflow needs all of that — which is exactly the gap LangGraph closes by giving the workflow a state machine, not just a model call.

The clearest way to hold this: think of a hospital emergency room. A patient walks in — receptionist checks them in, a triage nurse assesses severity, critical cases go straight to the trauma team, moderate cases wait for a doctor, the doctor examines and orders tests, results come back and a decision is made: admit or discharge. Every staff member knows exactly what state the patient is in; the file travels with the patient; if a test fails, there's a defined escalation path; nobody improvises the process from scratch. LangGraph is that ER workflow for an AI agent: the **shared state** is the patient file, the **nodes** are the staff members, the **conditional edges** are the triage decisions.

Three concepts carry the whole model:
- **Nodes** — each one function doing one focused job (call a database, invoke the LLM, run a validation check), reading from shared state and writing results back.
- **Edges** — some unconditional, some conditional (`if error_count > 2, route to circuit breaker; else continue`), and the conditions are plain code, fully owned and testable.
- **State** — travels through the whole workflow; by the time the LLM reasoning node sees a claim, it already knows the identity result, the coverage result, and the error count so far.

| Dimension | ReAct loop | LangGraph |
|---|---|---|
| Control flow | Agent decides what to do next | Explicitly defined |
| Debuggability | Hard — input/output only | Easy — trace each node separately |
| Failure handling | Agent retries or gives up | Defined circuit breakers and escalation |
| Reliability | Unpredictable under edge cases | Deterministic — defined behavior for every state |
| Regulated-workflow fit | Not suitable | Auditable, inspectable, compliant |

For enterprise and healthcare workflows specifically, the deciding factor is deterministic control flow: knowing exactly what state the system is in at every point, with defined failure paths, rather than an agent freelancing its way to an answer.

### 5.5 Input Guardrails

Guardrails run strictly before any external call — embedding, retrieval, or LLM — and fail closed: a violation is a hard stop that routes the claim to human review, not a warning that gets logged and waved through. The checks run in a fixed order, each cheap enough to run on every claim: **(1) schema validation** — every required structured field present and correctly typed; **(2) PHI-in-free-text detection** — a narrative-focused scan catching identifiers typed directly into a notes field, which structural redaction alone can't see; **(3) prompt-injection / instruction-override detection** — on the narrative, before it's embedded or sent anywhere, so a caught attempt means the embedding and LLM nodes never execute for that claim; **(4) policy allow/deny** — claim types, provider IDs, or amounts outside the defined operating envelope are force-routed to human review regardless of what any model would have concluded.

### 5.6 Memory Architecture — Short-Term and Long-Term

**Short-term (thread/session state)** is the `ClaimState` for one claim's single run, checkpointed after every node in Postgres, keyed by `thread_id`. Its job is narrow: resume after a transient failure without re-executing completed nodes.

**Long-term (cross-claim memory)** is history that informs *future* claims, never mutated mid-run: a relational claim-history table (tenant-scoped, keyed by tokenized identifiers) for frequency-pattern lookups, and a separate, append-only vector namespace of confirmed-FWA case embeddings that grows every time a human SIU reviewer closes an escalation as confirmed fraud — turning every human decision into retrieval signal for every future claim. The two stay separate from the checkpointer because checkpointing needs to be fast, per-thread, and disposable, while long-term memory needs to be queryable across every thread and written to on a completely different cadence.

### 5.7 RAG & Retrieval — The Open-Book-Exam Model

An LLM answering from memory alone has two problems in this context: its knowledge is frozen at training cutoff — it doesn't know about a CMS billing guideline published last quarter — and when it doesn't know something confidently, it can generate plausible but incorrect reasoning. Both are disqualifying for a claims decision.

The mental model: a closed-book student, however smart, might guess wrong on something recent or obscure. An open-book student searches the textbook first, finds the relevant pages, reads them, then writes an answer grounded in the source material. RAG makes the LLM the open-book student — the model supplies reasoning, the vector database supplies up-to-date knowledge, and together the answer is both intelligent and grounded.

Concretely, retrieval serves two distinct purposes against two distinct corpora, deliberately kept separate rather than sharing one index or one threshold:

- **Corpus A — confirmed-FWA case embeddings.** A high similarity score here is a bad sign: this claim's narrative closely resembles a previously confirmed fraud case.
- **Corpus B — current billing-guideline text.** A high match here grounds the LLM's reasoning in actual policy language — this is the RAG half.

**Chunking (Corpus B)** is structure-aware — split on numbered-rule boundaries first, with overlap applied only to oversized individual rules — rather than naive fixed-size chunking, which can split a single rule mid-sentence across two chunks. Each chunk carries `chunk_id`, `doc_id`, `section`, `effective_date`, `word_count`; `effective_date` lets retrieval be filtered to the guideline version actually in force on the claim's service date, since billing guidelines update quarterly.

**Retrieval itself runs three ways, and the difference matters:**

| Dimension | Keyword (BM25) | Dense (vector) | Hybrid |
|---|---|---|---|
| Understands synonyms | No | Yes | Yes |
| Exact code matching | Yes — perfect | Weak | Yes |
| Natural language queries | Poor | Excellent | Excellent |
| Recommended for claims | Not alone | Not alone | Always |

Pure dense search misses exact terminology matches on clinical codes (`heart attack` won't match `myocardial infarction` the other way, and a specific CPT number needs exact matching BM25 is built for); pure keyword search misses paraphrase and synonym variation dense search is built for. Hybrid — both run in parallel, scores merged — is the only architecture that handles claims narratives, which mix free clinical text with precise billing codes, reliably.

**Fusion** is implemented two ways and compared rather than picked blind: **weighted min-max normalized sum** (`fused = w_dense × dense_norm + w_bm25 × bm25_norm`, degrading gracefully to dense-only when BM25 returns all zeros — a valid "no keyword overlap" result, not an error) and **Reciprocal Rank Fusion** (`RRF_score(d) = Σ 1 / (k + rank_i(d))`, typically `k = 60`, rank-based and scale-free, so it still ranks even when raw BM25 scores are all zero). The two can legitimately disagree on the top result, and that disagreement is surfaced, not silently resolved — it's a signal the query sits in a genuinely ambiguous zone.

**A relevance floor gate** checks the top dense score against a minimum threshold before anything reaches the LLM: below it, the claim routes to a human rather than letting the model reason over a weak "best of a bad set" match, since grounded-looking reasoning on a poor match is worse than no grounding at all.

**Generation** packages the retrieved chunks — never the full source document, never chunks below the floor — into the prompt with a hard instruction: answer only from what's provided, and say so if the answer isn't in the retrieved text. That single constraint is what prevents the model from citing a guideline that sounds plausible but was never actually retrieved.

### 5.8 Model Routing / LLM Reasoning Layer

Not every claim needs the full reasoning model. A low-risk pre-filter, evaluated at the gateway before the graph even reaches the retrieval/LLM nodes, routes claims that pass every deterministic check and sit below a billed-amount threshold directly toward auto-approval — reserving the LLM for claims that actually warrant reasoning. For everything else, the gateway's tier policy assigns a model:

| Signal at gateway time | Tier assigned |
|---|---|
| Passes all rule-based checks, below billed-amount threshold | Skip LLM entirely — pre-filtered toward approval |
| Low claim value, high retrieval confidence, no provider history flags | Cheap/fast tier |
| Mid-range retrieval confidence, or claim value above threshold | Full reasoning tier |
| Provider under active SIU investigation, or a previously-escalated-and-reversed claim type | Best-available tier, or straight to human |

The `llm_fraud_reasoning` node assembles the grounded prompt and calls whichever tier was assigned, returning a structured response: `anomaly_type`, `confidence`, plain-English reasoning, `recommended_action`. Confidence is read independently of the recommendation — "reject, low confidence" and "reject, high confidence" are treated differently, which a system reading only the recommendation would conflate.

### 5.9 Output Guardrails & Confidence Gate

The LLM's raw response is schema-validated before it's trusted — an unparseable or incomplete response fails closed to human review rather than being coerced into a best-effort decision. The confidence gate then reads `confidence` and `recommended_action` together: high confidence + clear recommendation auto-approves or auto-rejects; the ambiguous middle, a validation failure, or a tripped circuit breaker escalates. Same three-way logic as the tier-selection table above, applied one step later to the decision rather than the model choice.

### 5.10 Circuit Breaker & Reliability

The circuit breaker is a deliberately separate mechanism from guardrails, tracking tool reliability rather than content validity: retrieval nodes retry on failure up to a defined threshold, then trip the breaker and escalate to human review without ever reaching the LLM — a down dependency is never treated as a security event, and a guardrail failure is never treated as a retryable blip. Per-step timeouts and structured fallback paths sit on every critical tool call, and failure counts are tracked across claims and time, not just within one claim's run, so a genuinely down dependency is caught as a pattern rather than one slow claim at a time. This shifts the failure mode from *silent wrong answer* to *explicit escalation* — in a healthcare context that's a patient-safety requirement, not a reliability preference.

### 5.11 Human-in-the-Loop

`interrupt_before` pauses the graph immediately before the routing node executes whenever the confidence gate, the circuit breaker, or a guardrail sends a claim to escalation — a real suspension of execution, checkpointed like everything else. A reviewer opening a paused claim sees the entire accumulated state: identity result, coverage result, the specific retrieved guideline chunks, the fraud-similarity match and score, and the LLM's own stated reasoning and confidence. `update_state()` resumes the graph exactly where it paused. Every closed escalation confirmed as fraud feeds back into the long-term FWA vector store, closing the loop between human judgment and future automated retrieval. Auto-approve and auto-reject only ever happen above defined confidence thresholds; everything else reaches a human with full reasoning context — the accountability mechanism is architectural, not a policy statement layered on top.

### 5.12 Caching Architecture

**Short-term / request-scoped:** rate-limit counters, idempotency keys (so a checkpoint replay after a transient failure doesn't re-send a duplicate Zapier notification), and in-flight deduplication for identical concurrent submissions.

**Persistent embedding cache:** keyed by a hash of the narrative text, checked before every embedding call — a repeated or resubmitted narrative is a cache hit and never triggers redundant compute. Long-lived rather than TTL'd on a fixed schedule, bounded instead by pruning low-hit-count entries periodically.

**The distinction from long-term memory** is purpose, not mechanism: the embedding cache exists purely to avoid redundant compute and holds no decision-relevant history; long-term memory exists specifically to inform future decisions and is queried as part of the fraud-detection logic itself.

### 5.13 PII / PHI Handling — Cross-Cutting Deep Dive

Enforced at every boundary the claim crosses, deliberately redundant rather than relying on one control point:

- **At the gateway:** structured identifier fields are swapped for reversible tokens by a tokenization service backed by a vault in a private subnet; everything downstream operates on tokens, never the underlying identifier.
- **At the input guardrail:** free-text narrative fields are scanned for PHI patterns typed directly into a notes box — the control that catches what field-name-based tokenization misses.
- **At every external call boundary** (Cohere, Pinecone, Zapier, the LLM provider): only tokenized identifiers and de-identified narrative text leave the trust boundary, and every such vendor requires a signed Business Associate Agreement before any real PHI-adjacent data flows to it. The Zapier payload carries `claim_id` and fraud flags — never narrative text — so a reviewer pulls clinical detail from the internal system, not a third-party webhook log.
- **In the cache and long-term memory:** both keyed by tokenized identifiers, inside the same encrypted-at-rest, access-controlled boundary as the rest of the PHI estate.
- **In logs and traces:** claim and decision IDs only, never clinical narrative text.
- **Token resolution** is a distinct, separately-audited, RBAC-gated operation, never an implicit side effect of anything else.

The organizing principle: **tokenize at the perimeter, operate on tokens everywhere inside, resolve back to PHI only at systems of record that already carry HIPAA-appropriate access controls.**

### 5.14 Multi-Tenancy — The Apartment-Building Model

Two ways to serve multiple payers: fifty separate houses — each client gets their own platform, complete isolation, but fifty times the cost and maintenance — or one apartment building with fifty units — one shared infrastructure, shared maintenance, upgrades that benefit everyone, but the walls between units have to be load-bearing, because a data-isolation failure here is a HIPAA violation, not a design flaw. This platform is the apartment building, enforced through four decisions:

- **Tenant ID in state** — every request carries a tenant identifier from the moment it enters the system; the intake node sets `tenant_id`, and every subsequent node reads it before touching any data. One node forgetting to filter by `tenant_id` is a HIPAA violation, which is exactly why it's threaded through shared state rather than re-derived per node.
- **Separate retrieval indexes** — each tenant gets its own vector index, not a shared index with metadata filtering. More operationally complex, but unambiguously auditable: in a HIPAA audit, one payer's documents are physically unreachable by another payer's queries, not merely filtered out.
- **Isolated LLM context** — prompt construction pulls only from the current tenant's retrieved documents and state; claims from different tenants are never batched into one LLM call.
- **Partitioned observability** — every trace is tagged with `tenant_id` at instrumentation time, and dashboards are scoped by tenant, so investigating a failure for one payer never surfaces another payer's traces.

Multi-tenancy is a core design requirement from the start here, not a retrofit — which is what makes onboarding a new client a provisioning exercise rather than an engineering project.

### 5.15 Evaluation Framework — RAGAS + DeepEval

Standard logging tells you what happened; it doesn't tell you whether the reasoning was correct. Nobody can manually review thousands of claim decisions a week, so the platform runs an automated evaluation pass — RAGAS for retrieval/reasoning quality, DeepEval for bias slicing — on a weekly 5% sample of production claims:

- **Context precision** — of everything retrieved, how much was actually relevant to this specific claim? A cardiology claim retrieving oncology billing guidelines is a context-precision failure, and this is the metric that catches it. Hybrid retrieval measurably lifts this over dense-only search.
- **Faithfulness** — did the reasoning stay grounded in what was actually retrieved, or did the model cite a guideline that was never in the retrieved set (a hallucinated citation)? This is the hallucination check specifically, and it's the metric that improves most from prompt-tightening work informed by trace analysis.
- **Answer relevancy** — did the response address the specific question asked, or wander into an accurate-but-irrelevant tangent (e.g., a general patient history summary when the question was "is this claim overbilled")?
- **Bias slicing (DeepEval)** — the same evaluation sample sliced by provider type, patient demographic, and geography, to surface whether the system is systematically flagging certain groups at rates the evidence doesn't warrant, before it affects real patients.

Usage pattern: track all metrics as a trend, not a one-time score; set alert thresholds (e.g., faithfulness below a floor triggers an observability investigation); and after any architecture or prompt change, run evaluation on a larger sample to measure the delta before promoting the change to full traffic.

### 5.16 Observability

Every node execution, tool call, and routing decision is a span under one root trace per claim, keyed by the gateway's `trace_id` and tagged with `tenant_id`:

| Span | Type | Captures |
|---|---|---|
| Root claim span | Chain | Total wall-clock time, overall success/failure |
| Intake | Tool | Raw input → structured state, duration, parsing errors |
| Identity / coverage | Tool | Call parameters, response, latency — a latency spike here flags an upstream dependency issue |
| Retrieval | Retriever | Query, each chunk retrieved with its similarity score — low precision is visible directly here |
| LLM reasoning | LLM | Exact prompt sent, token counts, full response, evaluation scores for this trace |
| Decision gate | Chain | Final state, routing decision, downstream output |

Trace collection and anomaly detection (threshold breaches firing alerts) are fully automated; root-cause investigation is manual but targeted — a human opens a trace only when an alert fires, and the trace shows exactly which prompt, which retrieved chunks, and what the model said, side by side, typically resolving in minutes rather than days. The fix decision itself — what to actually change — always stays manual.

### 5.17 Prompt & Policy Versioning

Prompts (and the model-tier policy from Section 5.8) are the platform's policy layer — they encode what the system should do and how it should reason, and they change on a learning cadence (as evaluation surfaces a failure mode, as billing guidelines update quarterly), which is faster than any CI/CD cadence should have to accommodate. Coupling them to a code deployment means days of lead time for a wording change, no safe gradual rollout, no attribution between a score change and the specific change that caused it, and no way for a non-engineer domain expert to contribute.

The fix: an external, versioned prompt/policy store — each entry has a name, version number, text, author, timestamp, and change reason — loaded by name at runtime rather than compiled into application code. A change ships by writing a new version and flipping which one is active; A/B testing routes a percentage of traffic to the new version, evaluation scores both cohorts, and the change is promoted or rolled back by flipping a flag. Every trace records which version was active for that claim, so a score movement is always attributable to a specific change rather than treated as unexplained drift.

---

## 6. Reusable Framework Design

The platform is built as an instance of a config-driven agentic workflow template, not a one-off script, through three specific separations:

1. **Nodes are pure functions over a typed state schema, registered by name** — independently testable, with no knowledge of the graph they sit in. A different domain reuses the same node contract with domain-specific implementations swapped in.
2. **Graph topology is declarative** — routing functions read plain state fields, so the graph's shape can be expressed as versioned config rather than hardcoded, and the same scaffolding assembles any topology described that way.
3. **Cross-cutting concerns are middleware**, not per-node code — PII redaction, guardrail checks, circuit-breaker counters, tracing, and cache lookups wrap node execution generically, so a new node automatically inherits all of them.

The reusable unit is: **typed state schema + node registry + declarative routing config + middleware stack.** A new use case means a new schema and new node implementations; orchestration, gateway, guardrails, and observability don't change.

---

## 7. Responsible AI & Compliance

Three commitments run through the whole design, not bolted on afterward. **Transparency** — every decision, automated or human, carries the plain-English reasoning behind it: what was detected, why it was flagged, which guideline text was referenced, and what confidence was assigned. **Fairness** — DeepEval bias checks on the evaluation sample (Section 5.15) surface whether the system treats certain provider types or patient demographics differently before that disparity reaches real patients. **Human oversight as accountability, not just reliability** — uncertain decisions always escalate; auto-approve and auto-reject only fire above defined confidence thresholds; the cost of a wrong automated decision in a payment-integrity context is borne by a patient who needed care, which is why the ethical posture here is conservative by design: the platform accelerates human review, it doesn't replace it for the edge cases that matter most.

HIPAA specifically requires: PHI never appears in an external LLM prompt without a signed BAA covering that vendor; every PHI access is logged; cross-tenant leakage is a compliance violation, not a bug; and any model output used in a payment decision must be explainable — which Section 7's transparency commitment and Section 5.13's tokenization discipline together satisfy by construction rather than by policy memo.

---

## 8. Capacity, Testing, and Scaling

**Capacity** is bounded by the slowest node — almost always the LLM call, not the deterministic checks — and any stated claims/week figure is only meaningful alongside its assumed low-risk-pre-filter rate, model-tier mix, and cache hit rate, since each materially changes per-claim latency.

**Scaling to millions of claims** rests on three levers, in order of impact: the orchestration layer is stateless between claims, so horizontal scaling is a matter of adding workers; the low-risk pre-filter (Section 5.8) means not every claim reaches the LLM at all, reserving the expensive reasoning call for claims that actually warrant it; and the retrieval index scales horizontally with pre-warmed replicas. The LLM call is the bottleneck, which is exactly why the pre-filter and async processing patterns matter more than any single infrastructure upgrade.

**Stress testing at scale (e.g., 50,000 claims)** drives the graph directly with a synthetic load generator, first in mock mode with realistic latency injected, then against a live-adjacent staging environment; a ramp profile (baseline → ramp-up → sustained soak → spike) surfaces the real bottleneck, with metrics captured per node against explicit SLOs, not just end-to-end.

**Offline testing** is the correct default: a parameterized synthetic-claim generator (realistic claim-amount distributions, real CPT/ICD code tables, injected known-FWA-like and prompt-injection narratives at controlled rates, fixed random seed for reproducibility) removes any need to touch real PHI during test cycles, and a separate labeled golden dataset regression-tests decision correctness independently of load.

**Vertical scaling** is reserved for the checkpoint/cache Postgres instance, scaled up before scaled out (read replicas, then partitioning by tenant/date). The LLM tier itself scales via the gateway's tier-mix, pre-filter, and fallback-chain policy — never via infrastructure on our side.

---

## 9. Deployment Architecture

Orchestrator and gateway services run as stateless pods on **EKS**, horizontally autoscaled on queue depth rather than raw CPU, since the workload is I/O-bound. **ArgoCD** manages GitOps deployment of graph topology, prompt versions, and model-tier policy configs. **GitLab CI** runs the node/guardrail/circuit-breaker/evaluation test suite before any promotion. **Terraform** provisions the Postgres checkpoint/cache stores and the per-tenant vector index. **Backstage** is the catalog entry point through which a new domain team registers a new node set against the shared framework — onboarding as a provisioning exercise, not a new engineering project.

---


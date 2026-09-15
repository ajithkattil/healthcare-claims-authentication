# Agentic AI Pattern Review — Healthcare Claims Authentication Platform
### For teaching: mapping this project to 2026 agentic AI best practices, and where to extend it

---

## 1. How this project already scores against canonical patterns

Before adding anything, it's worth telling your students what's *already* textbook-correct here — this project is a genuinely strong teaching example, not just a claims demo.

| Canonical pattern (Anthropic's "Building Effective Agents" taxonomy) | Where it lives in this project |
|---|---|
| **Workflow, not a freewheeling agent** — predefined code paths orchestrating LLM calls, chosen deliberately over an open-ended agent loop | The whole LangGraph `StateGraph` — intake → verify → retrieve → reason → gate → route |
| **Routing** — classify input, send to specialized path | Low-risk pre-filter + model-tier policy (Section 5.8 of your doc); confidence gate's 3-way routing |
| **Prompt chaining** | identity → coverage → code validation → retrieval → reasoning, each stage consuming the prior stage's state |
| **Parallelization (sectioning)** | Hybrid retrieval running dense (Pinecone) and keyword (BM25) search independently, fused after |
| **Evaluator step** | Confidence-driven decision gate reading the LLM's own stated confidence against retrieval strength |
| **Tool documentation / ACI investment** | Structure-aware chunking so retrieval returns clean, well-scoped context to the reasoning node |

This matters pedagogically: most public "agent" tutorials are a single ReAct loop with unconstrained tool access. Your platform is the counter-example — the industry's actual 2026 consensus (Anthropic, LangChain, OpenAI Agents SDK docs) is that **deterministic workflows with narrow, well-scoped LLM calls beat open-ended agents for anything regulated or high-stakes.** That's a great opening slide.

*Sources: [Anthropic — Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)*

---

## 2. What's missing relative to the 2026 landscape — and how to teach it

These are the patterns that have become the default vocabulary in agentic AI discourse since your architecture doc was written. Each entry: what it is, why the field converged on it, and a concrete way to slot it into *this* codebase so it stays a living example rather than a slide.

### 2.1 Context Engineering (the successor discipline to prompt engineering)

**What changed:** the field's framing shifted from "how do I word the prompt" to "what set of tokens does the model actually need at this step" — Anthropic now describes this explicitly as *"the set of strategies for curating and maintaining the optimal set of tokens during LLM inference."* It has four pillars: **instructions, retrieval, memory, tools** — each deliberately budgeted, not just concatenated.

**Where your project already half-does this, and where to make it explicit:**
- Your RAG generation step already does #2 well (retrieved chunks only, never full documents — "context window management" and "retrieval & re-ranking" done right).
- What's missing: **compaction** and **tool-result summarization**. As a claim's state accumulates (identity result, coverage result, retrieved chunks, fraud-similarity match), nothing currently trims it. Add a `compact_state` node before the LLM reasoning call that summarizes anything not needed verbatim (e.g., collapse the full coverage-check payload to a one-line status) — a great "before/after token count" demo for students.
- **Sub-agent isolation**: if you add a second reasoning path (see 2.2), each sub-agent should retrieve against its own narrow scope and report a summary upward — never share the full accumulated `ClaimState` with a worker that only needs one field of it.

*Sources: [Sourcegraph — Context Engineering for AI Agents](https://sourcegraph.com/blog/context-engineering), [Atlan — Context Engineering vs Prompt Engineering](https://atlan.com/know/context-engineering-vs-prompt-engineering/)*

### 2.2 Multi-agent orchestration patterns (naming what you have, and what's next)

2026 industry consensus has converged on five named patterns. Worth teaching all five as a taxonomy, then showing which one your platform uses and why the others weren't chosen:

| Pattern | What it is | Fits your platform? |
|---|---|---|
| **Pipeline (sequential chain)** | Each stage's output feeds the next | **This is what you have today** — intake → verify → retrieve → reason → gate |
| **Supervisor (hierarchical delegation)** | A coordinator decomposes work across specialist sub-agents; called *"the 2026 production default"* | **Natural next step** — see below |
| **Fan-out (parallel scatter-gather)** | Independent tasks run simultaneously, results aggregated | You already do a *mini* version of this in hybrid retrieval (dense + BM25 in parallel) |
| **Debate (multi-perspective critique)** | Multiple agents get the same input, answer independently, a judge adjudicates | Worth teaching as the "why we didn't use this" case — ~2.5× cost, reserved for high-stakes externally-visible decisions |
| **Swarm (dynamic peer coordination)** | Open-ended agent population coordinating via shared memory, no fixed hierarchy | Overkill here — good contrast case for "when NOT to reach for the fanciest pattern" |

**Concrete extension — Supervisor pattern for the reasoning node.** Right now `llm_fraud_reasoning` is one LLM call. Split it into a supervisor that delegates to two narrow specialist sub-agents — a **coding/billing specialist** (checks CPT/ICD code plausibility against retrieved guideline text) and a **narrative specialist** (checks the free-text narrative against FWA case embeddings) — and has the supervisor synthesize their outputs into the same `anomaly_type` / `confidence` / `recommended_action` structure the gate already expects. This is a clean, low-risk way to demonstrate the supervisor-worker pattern without touching your downstream gate/circuit-breaker logic at all.

**Concrete extension — Evaluator-Optimizer as a second safety net.** Add a lightweight "self-critique" pass after `llm_fraud_reasoning`, before the confidence gate: a second, cheap-tier LLM call that checks "does this reasoning actually cite the retrieved chunks, or does it drift beyond them?" and can force a re-generation once before escalating. This is the evaluator-optimizer pattern, and it's a strong complement to your existing output guardrails (which check *format*, not *faithfulness*).

*Sources: [Digital Applied — Multi-Agent Orchestration: 5 Patterns That Work in 2026](https://www.digitalapplied.com/blog/multi-agent-orchestration-5-patterns-that-work), [Beam.ai — 6 Multi-Agent Orchestration Patterns for Production](https://beam.ai/agentic-insights/multi-agent-orchestration-patterns-production)*

### 2.3 Model Context Protocol (MCP) as the tool-integration layer

**What changed:** MCP has become the de facto standard for how agents talk to tools — described as *"the universal connector for AI agents."* The July 2026 spec update (2026-07-28) made the protocol **stateless at its core** (any request can land on any server instance behind a plain round-robin load balancer — no session affinity needed), added **list caching** with `ttlMs`/`cacheScope` so agents don't re-fetch tool/prompt lists every call, graduated **long-running task support** out of experimental status, and hardened auth (RFC 9207 issuer validation, issuer-bound client credentials).

**Why this matters for your project specifically:** right now Cohere, Pinecone, and Zapier are called via direct SDK integration inside graph nodes. Re-framing those three as **MCP servers** — one for embeddings/retrieval, one for the vector store, one for notifications — is a genuinely valuable teaching moment: it shows students the difference between "an agent with hardcoded tool integrations" (what almost every tutorial does) and "an agent with a standardized, swappable tool interface" (what production platforms are converging on). It also means the circuit breaker becomes a protocol-level concern (per-tool call reliability) rather than something you hand-roll per SDK.

*Sources: [MCP Blog — 2026-07-28 Specification](https://blog.modelcontextprotocol.io/posts/2026-07-28/), [ChatForest — The MCP Ecosystem in 2026](https://chatforest.com/guides/mcp-ecosystem-2026-state-of-the-standard/)*

### 2.4 Agent memory: naming your long-term memory gap with the current framework vocabulary

Your architecture doc already calls for long-term memory (claim history + append-only FWA vector store) — it's on the "not yet in POC" list from the interview explainer. Here's how to teach it with 2026-current vocabulary instead of a generic "vector store":

- The field now names **three memory types**: *episodic* (past interactions — your claim history table), *semantic* (facts/preferences — the confirmed-FWA case store), *procedural* (learned behavior — your prompt/policy versioning, arguably).
- **The critical gap your plain vector-store design has**: it can't do temporal reasoning. Zep/Graphiti-style memory stores facts as a *knowledge graph with validity windows* — "when was this true, when was it superseded" — which is exactly the problem your billing-guideline corpus already has (`effective_date` filtering, since guidelines update quarterly) but your FWA case memory doesn't yet address. A confirmed-fraud pattern from two years ago under an old coding scheme shouldn't carry the same retrieval weight as one from last month.
- **Teaching point:** show students the three-way comparison — **Mem0** (three-tier user/session/agent scopes, hybrid vector+graph+KV, good general-purpose default), **Zep/Graphiti** (best temporal accuracy, right fit for a domain where "when" changes the answer), **Letta** (OS-inspired tiering — core/archival/recall — good for showing *why* context budgets matter). For this platform, Zep/Graphiti's temporal-validity model is the most defensible choice to argue for, given the quarterly-guideline-update requirement already in your own doc.

*Sources: [Atlan — Best AI Agent Memory Frameworks in 2026](https://atlan.com/know/best-ai-agent-memory-frameworks-2026/)*

### 2.5 Agent evaluation: upgrading from "RAGAS weekly sample" to the 2026 three-level stack

Your doc already has RAGAS + DeepEval — good instinct, but the field has since formalized this into a **three-level evaluation stack**, which is a clean teaching structure:

1. **End-to-end** — did the agent accomplish the goal? (Your existing RAGAS answer-relevancy / faithfulness metrics live here.)
2. **Trajectory** — did it take a *good path* to get there, not just the right answer? This is new relative to your doc: **Tool Correctness** (were the right tools called with the right parameters — evaluated deterministically, not by LLM judgment) and **Step Efficiency** (fewest useful steps — catches an agent that retries retrieval three times when once would do, or calls the LLM reasoning node when the low-risk pre-filter should have skipped it).
3. **Component-level** — isolate exactly which node degraded (retrieval precision vs. reasoning faithfulness vs. gate calibration), rather than one blended end-to-end score.

**Concrete extension:** add trajectory-level assertions to your eval job — e.g., "a claim that hit the low-risk pre-filter should never show an LLM reasoning span" — these are cheap, deterministic, and catch a class of bug (silent pre-filter bypass) that RAGAS's content-quality metrics structurally can't see.

*Sources: [Confident AI — LLM Agent Evaluation Metrics in 2026](https://www.confident-ai.com/blog/llm-agent-evaluation-complete-guide)*

### 2.6 Agentic security: OWASP Top 10 for Agentic Applications (2026) as a structured checklist

This is genuinely new since your architecture doc, and it's excellent teaching material because your platform already mitigates roughly half of it by accident of good design — showing students *why* those design choices exist is more convincing than presenting the checklist cold.

| # | Risk | Already mitigated in your design? |
|---|---|---|
| ASI01 | **Agent Goal Hijack** — malicious instructions embedded in inputs redirect the agent's objective | ✅ Partially — input guardrails' prompt-injection detection catches this on the narrative field |
| ASI02 | **Tool Misuse & Exploitation** — legitimate tools chained in unsafe ways | ✅ Partially — deterministic checks run before any tool call; RAG generation is instructed not to reason beyond retrieved text |
| ASI03 | **Identity & Privilege Abuse** — cached credentials, delegation chains misused | ✅ RBAC-gated token resolution, mTLS service-to-service |
| ASI04 | **Agentic Supply Chain Vulnerabilities** — compromised third-party models/plugins/tools | ❌ Not addressed — worth adding as a teaching gap: how would you pin/verify a Cohere or Pinecone SDK version, or an MCP server, before trusting it in the claims path? |
| ASI05 | **Unexpected Code Execution (RCE)** | N/A today (no code-execution tool) — but flag it as "the moment you add a code-interpreter tool for, say, ad-hoc claims analytics, this risk activates" |
| ASI06 | **Memory & Context Poisoning** — corrupted memory biases future reasoning | ⚠️ Partial gap — your append-only FWA vector store only grows from *confirmed* human decisions, which is the right instinct, but nothing currently detects if a reviewer account itself is compromised and starts confirming false positives at scale |
| ASI07 | **Insecure Inter-Agent Communication** | N/A today (single-agent pipeline) — becomes relevant the moment you adopt the supervisor pattern in 2.2; teach it as "this is why MCP's 2026 auth hardening (issuer-bound credentials) matters once you have sub-agents talking to each other" |
| ASI08 | **Cascading Failures** | ✅ Circuit breaker + per-step timeouts are exactly the mitigation for this |
| ASI09 | **Human-Agent Trust Exploitation** — users over-trusting confident-sounding agent output | ⚠️ Partial — plain-English reasoning to the reviewer helps, but nothing currently flags *overconfidence* itself as a signal worth surfacing to the reviewer |
| ASI10 | **Rogue Agents** — deviation from intended purpose | ✅ Confidence thresholds + escalation-by-default for the ambiguous middle is the core mitigation |

**Teaching framing:** present this table as "four are solved, two don't apply yet but will the moment you add multi-agent or code-execution, and two (ASI04 supply chain, ASI06/09 trust calibration) are genuine open gaps worth a class discussion on how you'd close them."

*Sources: [Teleport — OWASP Top 10 for Agentic Applications 2026](https://goteleport.com/blog/owasp-top-10-agentic-applications/), [OWASP GenAI Security Project](https://genai.owasp.org/2025/12/09/owasp-genai-security-project-releases-top-10-risks-and-mitigations-for-agentic-ai-security/)*

---

## 3. If you only add three things before the class demo

Given limited build time, these three give the highest "this is current" signal per hour invested:

1. **Name your patterns explicitly in the diagram/narrative** — label the pipeline as a pipeline, note where routing and evaluator-optimizer already occur. Zero code change, immediate teaching value (Section 1 + 2.2's pattern-naming table).
2. **Add the evaluator-optimizer self-critique node** (2.2) — small, self-contained addition to the graph, demonstrates a second canonical pattern live, and strengthens your faithfulness story for the RAG section.
3. **Present the OWASP Agentic Top 10 table** (2.6) as-is in your slides — no code required, and it's the single most "did you know this exists" moment for students who've only seen prompt-injection discussed informally.

Everything else in Section 2 (MCP re-platforming, supervisor-worker split, Zep-style temporal memory, trajectory evals) is better framed as "here's the roadmap, here's why each is the current industry direction" than as code you rush to ship before class.

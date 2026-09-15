# Roadmap — Path to Production Grade

This is the one place to track what's left between this POC and a system that could
safely touch real claims data at scale. It consolidates three previously separate
sources — README's "Known limitations" and Agentic AI Pattern Mapping gaps,
`production_additions_explainer.md`'s seven designed-but-not-built elements, and
`agentic_patterns_review.md`'s 2026 pattern-currency review — plus a set of gaps none
of those name yet. Where an item already has real implementation detail elsewhere,
this file points there rather than repeating it; check items off here as they're built,
and treat the tier assignment as a living judgment call, not a fixed spec.

**How to read the tiers:** Tier 0 blocks pointing this at real claim data at all,
regardless of scale. Tier 1 is what breaks first at production volume or in a real
deployment pipeline. Tier 2 is the operational layer around the decision that a
regulated production system needs but a POC doesn't. Tier 3 is agentic-architecture
maturity — genuinely valuable, not urgent.

---

## Tier 0 — Trust & compliance floor

Nothing here should be skipped just because the orchestration logic works; none of it
is optional once real PHI or real claims decisions are involved.

- [ ] **Authentication and authorization on the app itself.** Verified gap: `app.py` has
      no auth today — anyone with the Gradio link can submit a claim *and* act as the
      SIU reviewer clicking Approve/Deny. This is distinct from the RBAC-gated PII
      *resolution* designed below; it's authorization on the decision action itself, and
      none of the existing docs name it.
- [ ] **Tamper-evident audit trail.** The `log` field and the SQLite checkpointer are
      debugging/resumability tools, not a compliance record. A real deployment needs an
      append-only store of who decided what, when, and from where — a legal artifact,
      not a side effect of state management.
- [ ] **Data retention & deletion policy.** HIPAA requires a defined retention schedule
      and the ability to purge PHI on request. Not addressed anywhere yet, even at the
      "designed, not built" level the items below get.
- [ ] **Perimeter PII tokenization + RBAC-gated resolution.** Designed in detail in
      `production_additions_explainer.md` #3 (the `Tokenizer` class, resolve-as-a-separately-audited-call
      pattern) — not built. See also README's [PII/PHI Handling](README.md#piiphi-handling)
      for what *is* implemented today (the one narrow SSN-regex input guardrail).
- [ ] **Multi-tenancy isolation.** Designed in `production_additions_explainer.md` #2
      (separate Pinecone index per tenant, not a shared index with a metadata filter) —
      not built. Blocks onboarding more than one payer/client safely.
- [ ] **Bias / disparate-impact testing as a first-class workstream.** Currently a single
      line of pseudocode buried in `production_additions_explainer.md` #7's weekly-eval
      sketch (`deepeval.evaluate_bias(..., slice_by=["provider_type", "geography"])`).
      For a system that auto-approves or auto-rejects healthcare claims, this deserves
      its own success criteria and a named owner, not a side clause in a continuous-eval
      job description.

## Tier 1 — Production engineering basics

- [ ] **CI pipeline running `evals/`.** Verified gap: no `.github/workflows` exists, so
      the 44-test suite built this session only runs when someone remembers to run it
      locally. Should gate every push/PR.
- [ ] **Containerization + deployment manifest.** Verified gap: no `Dockerfile`, no Helm
      chart — no containerized path to a cluster at all, notable given this project's
      own EKS/ArgoCD/Helm context.
- [ ] **Dependency pinning.** Verified gap: `requirements.txt` has `anthropic`,
      `rank_bm25`, `requests`, and `python-dotenv` completely unpinned, and
      `pinecone==7.*` as a wildcard. This is the concrete, fixable version of the
      "agentic supply chain vulnerabilities" risk (OWASP ASI04) that
      `agentic_patterns_review.md`'s Section 2.6 table already flags in the abstract.
- [ ] **Secrets management.** Verified gap: `.env` is the only mechanism today — no
      vault, secrets manager, or rotation story.
- [ ] **Concurrent-scale infrastructure** — Postgres-backed checkpointer, Redis-backed
      embedding cache, queue-backed worker autoscaling, per-provider rate
      limiting/backoff, and idempotent claim processing. Fully detailed already in
      README's [Scaling to Production Volume](README.md#scaling-to-production-volume) —
      not repeated here.
- [ ] **Trace-based observability** (LangSmith / Langfuse / Arize). Named as the single
      biggest real gap in README's Agentic AI Pattern Mapping Section 2 — today there's
      only the printed `log` trace.

## Tier 2 — Decision-quality & operational layer

- [ ] **Low-risk pre-filter.** Designed in `production_additions_explainer.md` #4 — a
      cost lever that routes the cleanest, lowest-value claims around the LLM entirely
      before they reach retrieval. Not built; distinct from the model gateway (which
      picks a model tier, not whether to call a model at all).
- [ ] **Reranking (cross-encoder).** Named as a real gap in README's Agentic AI Pattern
      Mapping Section 7 — ranking today is the dense+BM25 fusion score alone.
- [ ] **SIU case-management workflow.** Approve/Deny is a demo button today, not a queue
      with assignment, SLAs, escalation, or case notes a real fraud-investigation team
      would need.
- [ ] **Adverse-action explainability & appeal path.** A provider whose claim is
      auto-rejected has no defined way to see why or contest it. Not addressed in any
      existing doc.
- [ ] **Guideline corpus ingestion & versioning pipeline.** The billing-guideline corpus
      is a hardcoded Python dict. `agentic_patterns_review.md` Section 2.4 already notes
      guidelines update quarterly and that temporal validity matters, but there's no
      actual mechanism for updating them safely, tracking which version was active for a
      given decision, or rolling back a bad update.
- [ ] **Long-term/episodic memory + confirmed-fraud feedback loop.** Designed in
      `production_additions_explainer.md` #5 (a relational claim-history table plus an
      append-only FWA vector namespace that only grows from confirmed human decisions).
- [ ] **Prompt & policy versioning.** Designed in `production_additions_explainer.md` #6
      (externalized, versioned prompts so a wording fix ships as a database write, not a
      redeploy).
- [ ] **Continuous evaluation job** (RAGAS/DeepEval faithfulness + the bias sampling from
      Tier 0). Designed in `production_additions_explainer.md` #7 as a weekly offline
      job against a 5% production sample. Note README's
      [Evals Framework](README.md#evals-framework) covers pre-deployment regression
      testing, not this — they're complementary, not the same thing.

## Tier 3 — Advanced agentic maturity

Genuinely valuable, not urgent. Each of these is already scoped in real depth in
`agentic_patterns_review.md` — linked here rather than re-explained.

- [ ] **MCP re-platforming** for Cohere/Pinecone/Zapier, replacing direct SDK calls
      (`agentic_patterns_review.md` Section 2.3).
- [ ] **Context engineering: state compaction.** Nothing currently trims `ClaimState` as
      it accumulates through the pipeline (`agentic_patterns_review.md` Section 2.1).
- [ ] **Zep/Graphiti-style temporal memory** for the FWA case store, so a confirmed-fraud
      pattern from two years ago doesn't carry the same retrieval weight as one from last
      month (`agentic_patterns_review.md` Section 2.4).
- [ ] **Three-level trajectory eval stack.** Partially started already: `evals/`'s
      `test_e2e_golden.py` circuit-breaker case asserts the model gateway and both
      specialists never ran for that claim — exactly the "trajectory correctness" idea
      `agentic_patterns_review.md` Section 2.5 calls for. Worth extending with more
      path-correctness assertions (e.g. "a claim that hits the low-risk pre-filter should
      never show a specialist span") rather than treated as a separate future project.

## Housekeeping

- [ ] **Refresh `agentic_patterns_review.md`.** It still describes the supervisor-worker
      split and the evaluator-optimizer self-critique as *proposed* additions — both have
      since been fully built into `claims_auth_hybrid_rag_confidence_circuitbreaker.py`,
      along with the model gateway, which the doc doesn't mention at all. Low effort,
      prevents it being used as a stale design reference.

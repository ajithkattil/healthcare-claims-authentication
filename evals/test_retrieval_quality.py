"""
Retrieval-quality metrics (precision@1, recall@2) for the hybrid dense+BM25
ranker, against a hand-labeled set of which FWA case / guideline each probe
narrative should actually match.

Why this is a small hand-rolled metric instead of RAGAS's non-LLM context
metrics (NonLLMContextPrecisionWithReference / NonLLMContextRecall), which
are the obvious off-the-shelf choice for exactly this: installing `ragas`
into a clean environment alongside this repo's actual dependencies (in
particular langgraph) produced a real, reproducible import failure --
ragas's default import path pulls in `langchain_community.chat_models.vertexai`,
which no longer exists in current `langchain-community` releases, and pinning
an older `langchain-community` to fix that then collides with the
`langchain-core` version langgraph itself requires. That's not a hypothetical
concern; it was reproduced while building this suite. See README.md's Evals
Framework section for the full writeup and where RAGAS/DeepEval *are*
recommended (the live-mode, LLM-judged faithfulness layer, run from an
isolated environment specifically because of this conflict) -- the same
"buy vs. build" judgment call this repo already makes for the LLM gateway
itself, applied here for the same underlying reason: don't drag in a
provider-agnostic framework's full dependency tree for two metrics this
corpus is small enough to compute directly.

precision@1 and recall@2 are standard IR metrics regardless of which tool
computes them; the numbers here would be identical if RAGAS's non-LLM
metrics were substituted in once the dependency conflict is resolved
upstream.
"""
import sys

sys.path.insert(0, __file__.rsplit("/evals/", 1)[0])
import claims_auth_hybrid_rag_confidence_circuitbreaker as claims  # noqa: E402

# (probe narrative, expected top-ranked FWA case id)
FWA_RETRIEVAL_LABELS = [
    (
        "Provider billed for a 60 minute office visit but no visit occurred and there are "
        "no supporting clinical notes on file for that date.",
        "past-fwa-001",
    ),
    (
        "Multiple claims were submitted for the same procedure code under different patient "
        "IDs from the same billing provider within a single day, exceeding plausible daily "
        "capacity for that provider.",
        "past-fwa-002",
    ),
    (
        "The claim upcoded a routine office visit to a high-complexity visit without any "
        "supporting documentation of the additional complexity described.",
        "past-fwa-003",
    ),
]

# (probe narrative, expected top-ranked guideline id)
GUIDELINE_RETRIEVAL_LABELS = [
    (
        "Two procedures performed during the same encounter were billed separately as "
        "unbundled services without a modifier justifying the separate billing.",
        "guideline-cms-4.3b",
    ),
    (
        "The evaluation and management visit code billed needs to be supported by "
        "documentation of the complexity of history, examination, and medical "
        "decision-making performed.",
        "guideline-cms-7.1",
    ),
    (
        "A claim was submitted for services rendered on a date when the patient's own "
        "medical record shows no corresponding encounter took place.",
        "guideline-cms-2.9",
    ),
]


def _precision_at_1(retrieved_ids: list[str], expected_id: str) -> float:
    return 1.0 if retrieved_ids and retrieved_ids[0] == expected_id else 0.0


def _recall_at_k(retrieved_ids: list[str], expected_id: str) -> float:
    return 1.0 if expected_id in retrieved_ids else 0.0


def test_fwa_case_retrieval_precision_and_recall():
    precisions, recalls = [], []
    for narrative, expected_id in FWA_RETRIEVAL_LABELS:
        embedding = claims.cohere_embed(narrative)
        matches = claims.pinecone_query_hybrid(narrative, embedding, thread_id="eval-retrieval-not-failing", attempt_state={})
        ids = [m["id"] for m in matches]
        precisions.append(_precision_at_1(ids, expected_id))
        recalls.append(_recall_at_k(ids, expected_id))
        assert ids and ids[0] == expected_id, (
            f"narrative about {expected_id!r} ranked {ids} instead -- top result was "
            f"{matches[0] if matches else None}"
        )

    precision_at_1 = sum(precisions) / len(precisions)
    recall_at_2 = sum(recalls) / len(recalls)
    assert precision_at_1 == 1.0, f"FWA-case precision@1 = {precision_at_1:.2f}, expected 1.0"
    assert recall_at_2 == 1.0, f"FWA-case recall@2 = {recall_at_2:.2f}, expected 1.0"


def test_guideline_retrieval_precision_and_recall():
    precisions, recalls = [], []
    for narrative, expected_id in GUIDELINE_RETRIEVAL_LABELS:
        embedding = claims.cohere_embed(narrative)
        guidelines = claims.retrieve_guidelines_hybrid(narrative, embedding)
        ids = [g["id"] for g in guidelines]
        precisions.append(_precision_at_1(ids, expected_id))
        recalls.append(_recall_at_k(ids, expected_id))
        assert ids and ids[0] == expected_id, (
            f"narrative about {expected_id!r} ranked {ids} instead -- top result was "
            f"{guidelines[0] if guidelines else None}"
        )

    precision_at_1 = sum(precisions) / len(precisions)
    recall_at_2 = sum(recalls) / len(recalls)
    assert precision_at_1 == 1.0, f"guideline precision@1 = {precision_at_1:.2f}, expected 1.0"
    assert recall_at_2 == 1.0, f"guideline recall@2 = {recall_at_2:.2f}, expected 1.0"


def test_unrelated_narrative_does_not_falsely_match_any_fwa_case():
    """The negative case: a genuinely unrelated narrative should score low everywhere, not just 'lower'."""
    narrative = "Routine annual wellness visit, CPT 99395, no complications, consistent with prior history."
    embedding = claims.cohere_embed(narrative)
    matches = claims.pinecone_query_hybrid(narrative, embedding, thread_id="eval-retrieval-not-failing", attempt_state={})
    top_dense_score = matches[0]["dense_score"] if matches else 0.0
    assert top_dense_score < 0.55, (
        f"an unrelated narrative scored {top_dense_score:.2f} against the FWA corpus -- "
        f"expected it to stay below the model gateway's own ambiguous-band floor"
    )

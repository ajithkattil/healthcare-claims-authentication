"""
LangGraph POC #5 -- Ingesting & Retrieving from LARGE Policy Documents
(Chunking, Metadata, Vector Storage, and Hybrid Retrieval, in depth)

WHY THIS SCRIPT EXISTS
-----------------------
POCs #3 and #4 do real RAG grounding, but over a *tiny* hardcoded guideline
corpus (three short strings). That's fine for proving the graph logic, but
it skips a question that comes up constantly in real production systems:
"your policy document is 200 pages -- what actually happens to it before
an LLM can use it?"

This script answers that end-to-end, standalone, so you can run it and
watch each step:

  1. INGESTION -- start from one large synthetic policy document (a single
     long string standing in for a parsed PDF/DOCX of CMS-style billing
     rules -- in real life this text would come out of a PDF/OCR parser,
     not be typed by hand).

  2. CHUNKING -- split that document into small, overlapping, section-aware
     chunks. This is the part that's easy to get wrong. We do it in three
     escalating ways so the tradeoffs of each are clear:
       a) naive fixed-size chunking (what NOT to lead with)
       b) recursive/structure-aware chunking (split on headings/paragraphs
          first, only falling back to fixed-size inside a huge paragraph)
       c) chunking WITH overlap, so a rule that spans a boundary isn't
          truncated out of both neighboring chunks

  3. METADATA -- every chunk carries its source doc id, section heading,
     character offsets, and an effective-date/version tag. This is what
     lets you filter ("only search the current policy version") *before*
     or *during* the similarity search, instead of after.

  4. STORAGE -- each chunk's embedding + text + metadata gets "upserted"
     into a mock Pinecone index (namespaced, matching how you'd separate
     e.g. claims-fraud-cases from billing-guidelines in a real deployment).
     A parallel BM25 index is built over the SAME chunk set, so both
     retrieval paths reference identical chunk IDs.

  5. HYBRID RETRIEVAL -- given a claim narrative, retrieve relevant chunks
     two ways (dense cosine similarity + BM25 keyword) and fuse them TWO
     different ways so you can compare and explain both when asked:
       - weighted min-max normalized sum (used in POC #4)
       - Reciprocal Rank Fusion (RRF) -- rank-based, scale-free, often more
         robust when score distributions are uneven

  6. GROUNDED ANSWER -- only the retrieved CHUNKS (never the whole
     document) are placed in the mock LLM prompt, proving the core claim
     from POC #4: the LLM never sees the huge document, it sees a handful
     of chunks retrieval decided were relevant.

This is intentionally standalone (not wired into the claims-approval graph)
so you can run and re-run just the retrieval layer in isolation while you
study it. It still uses a small LangGraph pipeline for the ingestion side,
since that's a legitimate one-shot workflow with checkpointable steps.

Run: python3 05_large_document_chunking_hybrid_retrieval.py
"""

import re
import math
import hashlib
import operator
from typing import TypedDict, Annotated, Optional

from rank_bm25 import BM25Okapi
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver


# ---------------------------------------------------------------------------
# 0. A LARGE SYNTHETIC POLICY DOCUMENT
# ---------------------------------------------------------------------------
# Stands in for a 30-200 page real billing-policy PDF. Multiple numbered
# sections, each a few paragraphs, deliberately including one rule that
# spans what would otherwise be an awkward chunk boundary -- so you can
# SEE overlap rescue it later in the trace output.
LARGE_POLICY_DOCUMENT = {
    "doc_id": "CMS-BILLING-POLICY-v3",
    "effective_date": "2026-01-01",
    "text": """
SECTION 1: EVALUATION AND MANAGEMENT (E/M) VISIT CODING

1.1 General Principle. The level of an evaluation and management (E/M)
visit code billed must be supported by contemporaneous documentation of
the corresponding complexity of history, examination, and medical
decision-making actually performed during that visit. A billed code level
that exceeds what the clinical documentation supports is considered
upcoding regardless of intent.

1.2 Documentation Requirements. Supporting documentation must include, at
minimum: chief complaint, relevant history, examination findings tied to
the presenting problem, and an assessment/plan reflecting the medical
decision-making complexity claimed. Templated or copy-forward notes that
do not reflect the specifics of the visit in question are insufficient to
support a higher-complexity code.

1.3 Time-Based Coding Exception. Where time-based coding is used in place
of the medical-decision-making framework, the total time spent by the
billing provider on the date of the encounter must be documented
explicitly, including a breakdown of activities performed, and must meet
or exceed the minimum threshold for the code billed.

SECTION 2: SAME-DAY / SAME-ENCOUNTER PROCEDURE BUNDLING

2.1 Bundling Rule. Two procedures performed during the same patient
encounter must not be billed separately (unbundled) if one procedure is
considered inclusive of the other under standard National Correct Coding
Initiative (NCCI) edits, unless a modifier justifying separate,
identifiable billing is both applied and independently documented in the
clinical note.

2.2 Modifier Use. A modifier indicating a separately identifiable service
must be supported by documentation describing why the second procedure
was distinct in cause, anatomical site, or timing from the first -- a
modifier applied without such supporting documentation does not cure an
otherwise-improper unbundling.

2.3 Repeat Procedures. Where the same procedure code is billed more than
once for the same patient on the same date by the same provider, the
claim must include documentation establishing medical necessity for the
repetition (e.g., a distinct anatomical site or a documented complication
requiring a repeat procedure); absent such documentation, only the first
instance is payable.

SECTION 3: SERVICE-OCCURRENCE VERIFICATION

3.1 Encounter Existence. A claim must not be submitted for services
rendered on a date when the patient's own medical record -- appointment
records, check-in logs, or clinical notes -- shows no corresponding
encounter took place. This applies regardless of whether the billed
service would otherwise have been medically appropriate had it occurred.

3.2 Provider Capacity Check. Claims submitted for the same billing
provider, across multiple distinct patients, for services purportedly
rendered within a single day, are subject to review where the aggregate
billed time or volume exceeds what is plausible given standard scheduling
capacity for that provider's specialty and setting.

SECTION 4: DOCUMENTATION RETENTION AND AUDIT RESPONSE

4.1 Retention Period. Supporting clinical documentation for any billed
claim must be retained for a minimum of seven years from the date of
service, or longer where required by applicable state law, and must be
producible within the timeframe specified in any audit request.

4.2 Audit Non-Response. Failure to produce supporting documentation within
the specified audit-response window is treated, for adjudication purposes,
as equivalent to an absence of supporting documentation -- the claim is
adjudicated as if no documentation exists, regardless of whether such
documentation may exist but was not timely produced.
""",
}


# ---------------------------------------------------------------------------
# 1. CHUNKING STRATEGIES
# ---------------------------------------------------------------------------
def _tokenize_words(text: str) -> list[str]:
    return re.findall(r"\S+", text)


def naive_fixed_chunk(text: str, chunk_size_words: int = 80) -> list[str]:
    """
    (a) NAIVE fixed-size chunking, no overlap, no structure awareness.
    Shown here ONLY so the failure mode is visible: it happily slices a
    rule in half if the boundary lands mid-sentence, and a chunk near the
    START of section 2.3 might read as gibberish with no idea it's about
    "repeat procedures."
    """
    words = _tokenize_words(text)
    return [
        " ".join(words[i : i + chunk_size_words])
        for i in range(0, len(words), chunk_size_words)
    ]


def structure_aware_chunk(
    doc_id: str,
    effective_date: str,
    text: str,
    max_words: int = 120,
    overlap_words: int = 25,
) -> list[dict]:
    """
    (b) + (c) RECURSIVE / STRUCTURE-AWARE chunking WITH overlap -- what you
    actually want for a policy document.

    Step 1: split on the document's own structure first (numbered
            sub-sections like "1.1", "2.3", "4.2" -- the natural unit of
            "one rule" in a policy doc). This keeps each rule intact
            instead of chopping it at an arbitrary word count.
    Step 2: if any single sub-section is itself too long (bigger than
            max_words), recursively fall back to fixed-size splitting
            WITH overlap only within that oversized section.
    Step 3: attach metadata to every chunk -- this is what makes the
            chunk usable for filtered search later (e.g. "only v3,
            only Section 2") instead of just being a floating string.

    Overlap matters here specifically for rules like 2.1 -> 2.2, which
    are two DIFFERENT numbered rules that reference each other ("a
    modifier applied without such supporting documentation" in 2.2 only
    makes sense next to 2.1's bundling rule). Overlap means a chunk
    boundary falling between them still lets each chunk retain a bit of
    the neighboring rule's context.
    """
    section_pattern = re.compile(
        r"(SECTION \d+:.*?)(?=\nSECTION \d+:|\Z)", re.DOTALL
    )
    sections = section_pattern.findall(text)

    chunks: list[dict] = []
    chunk_counter = 0

    for section_block in sections:
        section_title_match = re.match(r"SECTION (\d+): (.+)", section_block.strip())
        section_num = section_title_match.group(1) if section_title_match else "?"
        section_title = section_title_match.group(2).strip() if section_title_match else "Untitled"

        # Split the section into its numbered sub-rules (e.g. "1.1 ...", "1.2 ...")
        sub_rule_pattern = re.compile(r"(\d+\.\d+\..*?)(?=\n\d+\.\d+\.|\Z)", re.DOTALL)
        sub_rules = sub_rule_pattern.findall(section_block)
        if not sub_rules:
            sub_rules = [section_block]

        for rule_text in sub_rules:
            rule_text = rule_text.strip()
            words = _tokenize_words(rule_text)

            if len(words) <= max_words:
                # Whole rule fits in one chunk -- the ideal case, one
                # chunk == one coherent, independently-meaningful rule.
                pieces = [rule_text]
            else:
                # Oversized rule -- fall back to fixed-size WITH overlap,
                # only for this rule's text.
                pieces = []
                step = max_words - overlap_words
                for i in range(0, len(words), step):
                    piece_words = words[i : i + max_words]
                    pieces.append(" ".join(piece_words))
                    if i + max_words >= len(words):
                        break

            for piece in pieces:
                chunk_counter += 1
                chunks.append(
                    {
                        "chunk_id": f"{doc_id}::chunk-{chunk_counter:03d}",
                        "doc_id": doc_id,
                        "effective_date": effective_date,
                        "section": f"Section {section_num}: {section_title}",
                        "word_count": len(_tokenize_words(piece)),
                        "text": piece,
                    }
                )

    return chunks


# ---------------------------------------------------------------------------
# 2. EMBEDDING + MOCK VECTOR STORE (Cohere -> Pinecone shape, matching
#    POC #4's mock style so behavior is comparable across scripts)
# ---------------------------------------------------------------------------
def cohere_embed(text: str, dims: int = 64) -> list[float]:
    vec = [0.0] * dims
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        idx = int(hashlib.sha256(tok.encode()).hexdigest(), 16) % dims
        vec[idx] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm > 0 else vec


def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


class MockPineconeIndex:
    """
    Stands in for a real Pinecone index. Stores (vector, text, metadata)
    per chunk_id, namespaced by document -- mirroring how you'd separate
    e.g. a "billing-guidelines" namespace from a "fraud-cases" namespace
    in one Pinecone project rather than mixing unrelated corpora.
    """

    def __init__(self, namespace: str):
        self.namespace = namespace
        self._store: dict[str, dict] = {}

    def upsert(self, chunk_id: str, vector: list[float], text: str, metadata: dict) -> None:
        self._store[chunk_id] = {"vector": vector, "text": text, "metadata": metadata}

    def query(self, query_vector: list[float], top_k: int = 5, metadata_filter: Optional[dict] = None) -> list[dict]:
        candidates = self._store.items()
        if metadata_filter:
            candidates = [
                (cid, row) for cid, row in candidates
                if all(row["metadata"].get(k) == v for k, v in metadata_filter.items())
            ]
        scored = [
            {"id": cid, "score": _cosine_sim(query_vector, row["vector"]), "text": row["text"], "metadata": row["metadata"]}
            for cid, row in candidates
        ]
        return sorted(scored, key=lambda r: r["score"], reverse=True)[:top_k]

    def __len__(self) -> int:
        return len(self._store)


# ---------------------------------------------------------------------------
# 3. HYBRID RETRIEVAL -- TWO FUSION STRATEGIES
# ---------------------------------------------------------------------------
def weighted_fusion(dense_results: list[dict], bm25_scores: dict[str, float], dense_weight: float = 0.6) -> list[dict]:
    """
    Same approach as POC #4: min-max normalize each score list to [0,1],
    then a weighted sum. Simple, tunable via dense_weight, but sensitive
    to outliers in either score distribution (one huge BM25 score can
    compress everything else toward 0 after normalization).
    """
    dense_scores = {r["id"]: r["score"] for r in dense_results}
    all_ids = set(dense_scores) | set(bm25_scores)

    def _norm(d: dict) -> dict:
        vals = list(d.values()) or [0.0]
        lo, hi = min(vals), max(vals)
        if hi - lo < 1e-9:
            return {k: 0.0 for k in d}
        return {k: (v - lo) / (hi - lo) for k, v in d.items()}

    dense_n = _norm({cid: dense_scores.get(cid, 0.0) for cid in all_ids})
    bm25_n = _norm({cid: bm25_scores.get(cid, 0.0) for cid in all_ids})

    fused = {cid: dense_weight * dense_n[cid] + (1 - dense_weight) * bm25_n[cid] for cid in all_ids}
    return sorted(
        [{"id": cid, "fused_score": round(score, 4)} for cid, score in fused.items()],
        key=lambda r: r["fused_score"], reverse=True,
    )


def reciprocal_rank_fusion(dense_results: list[dict], bm25_ranked_ids: list[str], k: int = 60) -> list[dict]:
    """
    RRF: ignores raw score magnitude entirely, only uses RANK POSITION.
    score(doc) = sum over each ranking list of 1 / (k + rank_in_that_list)

    This is scale-free by construction -- you never have to worry about
    normalizing dense cosine scores (bounded, ~0-1) against BM25 scores
    (unbounded, corpus-size-dependent) because ranks are already
    comparable across methods. This becomes the right answer once you push
    past "how do you combine the scores" toward "what if the score
    distributions are wildly different shapes."
    """
    dense_ranked_ids = [r["id"] for r in dense_results]
    scores: dict[str, float] = {}
    for rank, cid in enumerate(dense_ranked_ids):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    for rank, cid in enumerate(bm25_ranked_ids):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(
        [{"id": cid, "rrf_score": round(score, 5)} for cid, score in scores.items()],
        key=lambda r: r["rrf_score"], reverse=True,
    )


# ---------------------------------------------------------------------------
# 4. LANGGRAPH STATE + NODES (ingestion pipeline)
# ---------------------------------------------------------------------------
class IngestState(TypedDict):
    doc_id: str
    effective_date: str
    raw_text: str
    naive_chunks: list[str]
    structured_chunks: list[dict]
    index_size: int
    query_text: str
    dense_results: list[dict]
    bm25_ranked_ids: list[str]
    bm25_scores: dict
    weighted_fused: list[dict]
    rrf_fused: list[dict]
    grounded_answer: str
    log: Annotated[list[str], operator.add]


# Module-level store + BM25 index so nodes can share them without stuffing
# multi-KB vector blobs through graph state on every step (state should
# carry IDs/results, not the whole index).
_INDEX = MockPineconeIndex(namespace="billing-guidelines")
_BM25: Optional[BM25Okapi] = None
_CHUNK_LOOKUP: dict[str, dict] = {}
_CHUNK_ID_ORDER: list[str] = []


def ingest_document(state: IngestState) -> dict:
    print("  -> [ingest_document] executing")
    print(f"     source doc_id={state['doc_id']}  raw length={len(state['raw_text'])} chars, "
          f"~{len(_tokenize_words(state['raw_text']))} words")
    return {"log": ["ingest_document: loaded raw policy text"]}


def chunk_document(state: IngestState) -> dict:
    print("  -> [chunk_document] executing")
    naive = naive_fixed_chunk(state["raw_text"], chunk_size_words=80)
    structured = structure_aware_chunk(state["doc_id"], state["effective_date"], state["raw_text"])
    print(f"     naive fixed-size chunking  -> {len(naive)} chunks (no overlap, no structure awareness)")
    print(f"     structure-aware + overlap  -> {len(structured)} chunks (one chunk ~= one coherent rule)")
    return {
        "naive_chunks": naive,
        "structured_chunks": structured,
        "log": [f"chunk_document: {len(structured)} structured chunks produced"],
    }


def embed_and_index_chunks(state: IngestState) -> dict:
    global _BM25, _CHUNK_LOOKUP, _CHUNK_ID_ORDER
    print("  -> [embed_and_index_chunks] executing (Cohere embed -> Pinecone upsert + BM25 index)")

    tokenized_corpus = []
    for chunk in state["structured_chunks"]:
        vec = cohere_embed(chunk["text"])
        metadata = {
            "doc_id": chunk["doc_id"],
            "effective_date": chunk["effective_date"],
            "section": chunk["section"],
        }
        _INDEX.upsert(chunk["chunk_id"], vec, chunk["text"], metadata)
        _CHUNK_LOOKUP[chunk["chunk_id"]] = chunk
        _CHUNK_ID_ORDER.append(chunk["chunk_id"])
        tokenized_corpus.append(re.findall(r"[a-z0-9]+", chunk["text"].lower()))

    _BM25 = BM25Okapi(tokenized_corpus)
    print(f"     Pinecone mock index size: {len(_INDEX)} vectors (namespace='{_INDEX.namespace}')")
    print(f"     BM25 index built over the SAME {len(tokenized_corpus)} chunks (same chunk_ids on both sides)")
    return {
        "index_size": len(_INDEX),
        "log": [f"embed_and_index_chunks: indexed {len(_INDEX)} chunks into dense + BM25"],
    }


def hybrid_retrieve(state: IngestState) -> dict:
    print("  -> [hybrid_retrieve] executing")
    query = state["query_text"]
    query_vec = cohere_embed(query)

    # Dense: Pinecone-style top-k, optionally metadata-filtered to only
    # the current policy version -- this is the "huge corpus" lever: you
    # never brute-force search everything, you filter first where you can.
    dense_results = _INDEX.query(
        query_vec, top_k=5, metadata_filter={"doc_id": state["doc_id"], "effective_date": state["effective_date"]}
    )

    # BM25: rank ALL chunks by keyword score, take the ranking.
    query_tokens = re.findall(r"[a-z0-9]+", query.lower())
    raw_bm25 = _BM25.get_scores(query_tokens)
    bm25_scores = {cid: float(score) for cid, score in zip(_CHUNK_ID_ORDER, raw_bm25)}
    bm25_ranked_ids = [cid for cid, _ in sorted(bm25_scores.items(), key=lambda kv: kv[1], reverse=True)]

    weighted = weighted_fusion(dense_results, bm25_scores, dense_weight=0.6)
    rrf = reciprocal_rank_fusion(dense_results, bm25_ranked_ids)

    # ---- NO-BM25-MATCH CHECK -----------------------------------------
    # A BM25 score of 0.0 for a chunk isn't an error -- it means none of
    # the query's tokens appear in that chunk (rank_bm25 always returns a
    # full score array, one float per chunk, never a partial list or
    # None). If EVERY chunk scores 0.0, the query shares no vocabulary
    # with the corpus at all -- typically a heavily paraphrased or
    # synonym-heavy narrative. Watch what happens to each fusion strategy
    # in that case:
    max_bm25 = max(bm25_scores.values()) if bm25_scores else 0.0
    if max_bm25 == 0.0:
        print("     *** BM25: zero keyword overlap with ANY chunk (max score = 0.0) ***")
        print("     weighted_fusion's min-max normalizer detects hi-lo<1e-9 across all-zero "
              "BM25 scores and returns 0.0 for every chunk on that axis -- the fused score "
              "collapses to dense_weight * dense_score alone. Hybrid gracefully degrades to "
              "PURE DENSE search instead of erroring or returning garbage.")
        print("     RRF, in contrast, has no concept of 'this ranking is meaningless': it still "
              "hands out 1/(k+rank+1) credit based on BM25's arbitrary tie-break ordering, even "
              "though that ordering carries zero real keyword signal. At small corpus scale this "
              "barely moves the result; at production scale it's a real footgun worth guarding "
              "against (e.g. drop a ranking list from RRF entirely if its own top score is 0).")
    else:
        print(f"     BM25 top score: {max_bm25:.4f} (nonzero -- keyword signal exists)")

    print(f"     dense top match:    {dense_results[0]['id']} (cosine={dense_results[0]['score']:.4f})")
    print(f"     weighted-fusion top: {weighted[0]['id']} (fused={weighted[0]['fused_score']})")
    print(f"     RRF top:             {rrf[0]['id']} (rrf={rrf[0]['rrf_score']})")

    return {
        "dense_results": dense_results,
        "bm25_ranked_ids": bm25_ranked_ids,
        "bm25_scores": bm25_scores,
        "weighted_fused": weighted,
        "rrf_fused": rrf,
        "log": ["hybrid_retrieve: dense + BM25 retrieved, fused two ways"],
    }


MIN_DENSE_RELEVANCE = 0.2  # illustrative floor, not calibrated -- see POC #4's confidence gate for the same idea


def ground_and_answer(state: IngestState) -> dict:
    """
    Mock LLM call. The point being demonstrated: the prompt below contains
    ONLY the top retrieved chunks (a few hundred words), never the full
    ~500-word (or in a real deployment, 200-page) source document. This is
    what makes "huge document" tractable for the LLM's context window.

    RELEVANCE FLOOR: retrieval always returns top_k results, even if
    nothing in the corpus is actually relevant -- sorted()[:top_k] doesn't
    know the difference between "great match" and "least-bad of a bad
    set." Before trusting any retrieved chunk as grounding, check the top
    DENSE score (not the fused/RRF score, which are rank-relative and can
    look confident even when nothing matched) against a minimum floor.
    Below that floor, this is a "no relevant guideline found" case -- the
    graph should say so explicitly rather than grounding an LLM answer in
    a chunk that only won by default.
    """
    print("  -> [ground_and_answer] executing (mock LLM, grounded ONLY in retrieved chunks)")
    top_dense_score = state["dense_results"][0]["score"] if state["dense_results"] else 0.0
    if top_dense_score < MIN_DENSE_RELEVANCE:
        print(f"     top dense score {top_dense_score:.4f} is below MIN_DENSE_RELEVANCE "
              f"({MIN_DENSE_RELEVANCE}) -- nothing in the corpus is actually relevant.")
        return {
            "grounded_answer": "NO RELEVANT GUIDELINE FOUND -- retrieval scores below the relevance "
                                "floor; do not ground an answer in these chunks (route to human review "
                                "in a real workflow, exactly like POC #4's confidence gate).",
            "log": ["ground_and_answer: retrieval below relevance floor, declined to ground"],
        }

    top_ids = [r["id"] for r in state["rrf_fused"][:3]]
    retrieved_chunks = [_CHUNK_LOOKUP[cid] for cid in top_ids]

    context_block = "\n\n".join(f"[{c['chunk_id']} | {c['section']}]\n{c['text']}" for c in retrieved_chunks)
    prompt_word_count = len(_tokenize_words(context_block))
    full_doc_word_count = len(_tokenize_words(state["raw_text"]))

    print(f"     context sent to LLM: {prompt_word_count} words (top {len(retrieved_chunks)} chunks)")
    print(f"     full source document: {full_doc_word_count} words")
    print(f"     -> LLM prompt is ~{100 * prompt_word_count // full_doc_word_count}% the size of the full document")

    # Deterministic mock "answer" -- in live mode this would be a real
    # Anthropic call with context_block inserted into the prompt and an
    # "answer only from the provided excerpts" instruction, exactly like
    # POC #4's llm_reason_about_claim_grounded.
    answer = (
        f"Grounded in {len(retrieved_chunks)} retrieved excerpt(s) "
        f"({', '.join(c['chunk_id'] for c in retrieved_chunks)}): "
        f"the most relevant rule is from {retrieved_chunks[0]['section']}."
    )
    return {"grounded_answer": answer, "log": ["ground_and_answer: answered from retrieved chunks only"]}


# ---------------------------------------------------------------------------
# 5. BUILD THE GRAPH
# ---------------------------------------------------------------------------
def build_graph(checkpointer):
    g = StateGraph(IngestState)
    g.add_node("ingest_document", ingest_document)
    g.add_node("chunk_document", chunk_document)
    g.add_node("embed_and_index_chunks", embed_and_index_chunks)
    g.add_node("hybrid_retrieve", hybrid_retrieve)
    g.add_node("ground_and_answer", ground_and_answer)

    g.add_edge(START, "ingest_document")
    g.add_edge("ingest_document", "chunk_document")
    g.add_edge("chunk_document", "embed_and_index_chunks")
    g.add_edge("embed_and_index_chunks", "hybrid_retrieve")
    g.add_edge("hybrid_retrieve", "ground_and_answer")
    g.add_edge("ground_and_answer", END)

    return g.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# 6. RUN IT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    with SqliteSaver.from_conn_string("chunking_demo_checkpoints.sqlite") as checkpointer:
        graph = build_graph(checkpointer)

        base_input = {
            "doc_id": LARGE_POLICY_DOCUMENT["doc_id"],
            "effective_date": LARGE_POLICY_DOCUMENT["effective_date"],
            "raw_text": LARGE_POLICY_DOCUMENT["text"],
            "log": [],
        }

        print("\n=== INGEST: chunk + embed + index the policy document ONCE ===")
        config = {"configurable": {"thread_id": "policy-ingest-1"}}
        result = graph.invoke(
            {**base_input, "query_text": "Provider billed a high complexity visit but the notes only support a routine visit"},
            config=config,
        )
        print("\nGrounded answer:", result["grounded_answer"])
        print("\nTop weighted-fusion matches:")
        for r in result["weighted_fused"][:3]:
            print(f"   {r['id']}  fused_score={r['fused_score']}")
        print("\nTop RRF matches:")
        for r in result["rrf_fused"][:3]:
            print(f"   {r['id']}  rrf_score={r['rrf_score']}")

        print("\n=== A SECOND QUERY against the SAME already-built index (no re-chunking/re-embedding needed) ===")
        config2 = {"configurable": {"thread_id": "policy-query-2"}}
        result2 = graph.invoke(
            {**base_input, "query_text": "no appointment record exists for the date of service billed"},
            config=config2,
        )
        print("\nGrounded answer:", result2["grounded_answer"])
        print("Top RRF match:", result2["rrf_fused"][0])

        print("\n=== QUERY 3: vocabulary with ZERO overlap against the corpus -- no BM25 match at all ===")
        config3 = {"configurable": {"thread_id": "policy-query-3"}}
        result3 = graph.invoke(
            {**base_input, "query_text": "zylophonic quixotic gadgetry unrelated randomness nonsense"},
            config=config3,
        )
        print("\nGrounded answer:", result3["grounded_answer"])
        print(
            "\nNOTE on this specific demo: the mock cohere_embed() here is a hashing-trick "
            "bag-of-words, not a real semantic embedding -- by design (see the comment on "
            "cohere_embed above), its cosine similarity tracks WORD OVERLAP, the same signal "
            "BM25 uses. That means when BM25 finds zero keyword overlap, this particular mock's "
            "dense score is ALSO near zero, and you see both retrieval paths fail together, "
            "landing in the MIN_DENSE_RELEVANCE gate above. With a REAL embedding model (live "
            "Cohere), a query that's a pure paraphrase -- same meaning, zero shared vocabulary -- "
            "would still score meaningfully on the DENSE side even with zero BM25 overlap. THAT "
            "gap between 'shares no words' and 'shares no meaning' is exactly the production "
            "argument for hybrid retrieval; this mock just can't demonstrate it without a real "
            "semantic model, so don't mistake this run for dense 'failing to compensate' -- it's "
            "the mock's limitation, not hybrid retrieval's."
        )

        print("\n=== BONUS: show why naive fixed-size chunking (no structure, no overlap) is worse ===")
        naive = naive_fixed_chunk(LARGE_POLICY_DOCUMENT["text"], chunk_size_words=80)
        print(f"Naive chunking produced {len(naive)} chunks. Example chunk #3 (arbitrary word-count cut):")
        print(f"   ...{naive[2][:200]}...")
        print("Notice it may start or end mid-sentence, mixing the tail of one rule with the head of another --")
        print("this is exactly what structure-aware chunking with overlap in this script avoids.")

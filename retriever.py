# retriever.py
# Hybrid Search (BM25 + ChromaDB) + Cross-Encoder Reranking
# Drop-in upgrade for your existing RAG pipeline

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import os
import re

# ── Load Cross-Encoder once at startup (avoids reloading on every query) ──────
# This model scores (query, passage) pairs for relevance on a scale of ~-10 to +10
CROSS_ENCODER_MODEL = os.getenv(
    "CROSS_ENCODER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)
cross_encoder = None


def get_cross_encoder():
    """Load the reranker lazily so startup still works if the model is absent."""
    global cross_encoder
    if cross_encoder is None:
        cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    return cross_encoder


def tokenize(text: str) -> list[str]:
    """Lowercase and split text into word tokens for BM25."""
    return re.findall(r'\w+', text.lower())


def bm25_search(query: str, all_docs: list[dict], top_k: int = 10) -> list[dict]:
    """
    Keyword-based BM25 search over a list of document dicts.

    Each doc dict must have at least:
        { "text": "...", "metadata": {...}, "id": "..." }

    Returns the top_k docs ranked by BM25 score.

    WHY BM25?
    ---------
    ChromaDB uses vector/semantic search — great for meaning, but misses
    exact keywords (e.g. "AJKU admission 2024"). BM25 is the opposite:
    it excels at keyword matching. Combining both covers more ground.
    """
    corpus = [tokenize(doc["text"]) for doc in all_docs]
    bm25 = BM25Okapi(corpus)

    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)

    scored = sorted(zip(all_docs, scores), key=lambda x: x[1], reverse=True)
    results = []
    for doc, score in scored[:top_k]:
        if score > 0:
            enriched = dict(doc)
            enriched["bm25_score"] = float(score)
            results.append(enriched)
    return results


def hybrid_search(query: str, chroma_collection, all_docs: list[dict], top_k: int = 10) -> list[dict]:
    """
    Combines ChromaDB vector search + BM25 keyword search.

    Steps:
    1. Run ChromaDB semantic search  → top_k results
    2. Run BM25 keyword search       → top_k results
    3. Merge both, deduplicate by ID
    4. Return merged candidate pool for reranking

    WHY MERGE?
    ----------
    Each method has blind spots. A query like "fees structure for BS CS"
    might be found by BM25 (keyword: "fees", "BS CS") but not ranked
    high by vectors if the embedding drifted. Union covers both cases.
    """
    # ── 1. ChromaDB semantic search ───────────────────────────────────────────
    chroma_results = chroma_collection.query(query_texts=[query], n_results=top_k)

    chroma_docs = []
    if chroma_results and chroma_results["documents"]:
        distances = chroma_results.get("distances", [[]])[0]
        for idx, (text, meta, doc_id) in enumerate(zip(
            chroma_results["documents"][0],
            chroma_results["metadatas"][0],
            chroma_results["ids"][0],
        )):
            distance = distances[idx] if idx < len(distances) else None
            chroma_docs.append({
                "id": doc_id,
                "text": text,
                "metadata": meta,
                "vector_distance": distance,
                "vector_score": 1 / (1 + distance) if distance is not None else 0.0,
            })

    # ── 2. BM25 keyword search ────────────────────────────────────────────────
    bm25_docs = bm25_search(query, all_docs, top_k=top_k)

    # ── 3. Merge + deduplicate (ChromaDB results take priority in ordering) ───
    seen_ids = set()
    merged = []
    for doc in chroma_docs + bm25_docs:
        if doc["id"] not in seen_ids:
            seen_ids.add(doc["id"])
            merged.append(doc)
        else:
            existing = next(item for item in merged if item["id"] == doc["id"])
            existing.update({k: v for k, v in doc.items() if k.endswith("_score")})

    return merged  # Could be up to 2*top_k unique candidates


def rerank(query: str, candidates: list[dict], top_n: int = 5) -> list[dict]:
    """
    Re-scores candidate docs using a Cross-Encoder model.

    A Cross-Encoder reads (query + passage) together — unlike embeddings
    which encode them separately. This joint attention makes it much more
    accurate at judging relevance, but too slow to run on all docs.

    That's why we use it AFTER retrieval (as a reranker), not for initial search.

    HOW IT WORKS:
    -------------
    Input:  [(query, doc1_text), (query, doc2_text), ...]
    Output: [score1, score2, ...]   ← higher = more relevant

    We then sort by score and return the top_n docs.
    """
    if not candidates:
        return []

    # Build (query, passage) pairs — the format the model expects
    pairs = [(query, doc["text"]) for doc in candidates]

    try:
        scores = get_cross_encoder().predict(pairs)
        for doc, score in zip(candidates, scores):
            doc["rerank_score"] = float(score)
            doc["rerank_method"] = "cross_encoder"
    except Exception as exc:
        print("CrossEncoder unavailable, using hybrid fallback:", exc)
        for doc in candidates:
            doc["rerank_score"] = float(
                doc.get("bm25_score", 0.0) + doc.get("vector_score", 0.0)
            )
            doc["rerank_method"] = "hybrid_fallback"

    reranked = sorted(candidates, key=lambda x: x.get("rerank_score", 0.0), reverse=True)
    return reranked[:top_n]


def hybrid_rerank_search(
    query: str,
    chroma_collection,
    all_docs: list[dict],
    top_k: int = 10,
    top_n: int = 5,
) -> list[dict]:
    """
    MAIN ENTRY POINT — call this from agents.py

    Full pipeline:
        Query
          │
          ├─► ChromaDB Vector Search  ─┐
          │                            ├─► Merge + Dedupe
          └─► BM25 Keyword Search     ─┘
                                        │
                                        ▼
                               Cross-Encoder Reranking
                                        │
                                        ▼
                               Top-N chunks → Gemini

    Args:
        query           : user's question string
        chroma_collection: your existing ChromaDB collection object
        all_docs        : ALL docs from ChromaDB as list of dicts
                          (fetch once, cache it — see agents.py note below)
        top_k           : candidates to fetch from each search method
        top_n           : final chunks to return to Gemini

    Returns:
        List of top_n dicts: [{ "id", "text", "metadata", "rerank_score" }, ...]
    """
    candidates = hybrid_search(query, chroma_collection, all_docs, top_k=top_k)
    return rerank(query, candidates, top_n=top_n)

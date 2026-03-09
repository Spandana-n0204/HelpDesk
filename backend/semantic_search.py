import logging
import numpy as np

logger = logging.getLogger(__name__)

TOP_K_CANDIDATES = 10   # fetch this many before MMR
TOP_K_FINAL      = 4    # return this many to the LLM
MMR_LAMBDA       = 0.6  # 1.0 = pure relevance, 0.0 = pure diversity


# ── RRF ───────────────────────────────────────────────────────────────────────

def _reciprocal_rank_fusion(
    sem_scores: np.ndarray,
    kw_scores:  np.ndarray,
    k: int            = 60,
    sem_weight: float = 0.7,
) -> np.ndarray:
    """Combine two score arrays via Reciprocal Rank Fusion."""
    n         = len(sem_scores)
    sem_ranks = n - sem_scores.argsort().argsort()
    kw_ranks  = n - kw_scores.argsort().argsort()
    return (
        sem_weight       * (1.0 / (k + sem_ranks))
        + (1 - sem_weight) * (1.0 / (k + kw_ranks))
    )


# ── MMR ───────────────────────────────────────────────────────────────────────

def _mmr_rerank(
    query_emb:  np.ndarray,
    candidates: list,
    embeddings: np.ndarray,
    top_k:      int,
    lam:        float = MMR_LAMBDA,
) -> list:
    """Maximal Marginal Relevance — balance relevance with diversity."""
    selected  = []
    remaining = list(candidates)

    while remaining and len(selected) < top_k:
        if not selected:
            scores = [float(query_emb @ embeddings[i]) for i in remaining]
            best   = remaining[int(np.argmax(scores))]
        else:
            scores = []
            for idx in remaining:
                rel        = float(query_emb @ embeddings[idx])
                redundancy = max(float(embeddings[idx] @ embeddings[s]) for s in selected)
                scores.append(lam * rel - (1 - lam) * redundancy)
            best = remaining[int(np.argmax(scores))]

        selected.append(best)
        remaining.remove(best)

    return selected


# ── Public retrieval function ─────────────────────────────────────────────────

def retrieve(store, query: str, top_k_final: int = TOP_K_FINAL) -> list:
    """
    Full pipeline: RRF → candidate pool → MMR → final results.
    Returns list of {"text": str, "score": float}.
    """
    if store.embeddings is None or not store.chunks:
        return []

    sem_scores = store.semantic_scores(query)
    kw_scores  = store.keyword_scores(query)
    rrf_scores = _reciprocal_rank_fusion(sem_scores, kw_scores)

    candidate_idxs = rrf_scores.argsort()[::-1][:TOP_K_CANDIDATES].tolist()

    q_emb      = store.encode_query(query)
    final_idxs = _mmr_rerank(q_emb, candidate_idxs, store.embeddings, top_k_final)

    return [
        {"text": store.chunks[i], "score": float(rrf_scores[i])}
        for i in final_idxs
    ]
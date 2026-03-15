"""
semantic_search.py — Retrieval for DSCE HelpDesk
Pipeline: RRF(semantic + keyword) → MMR re-rank → clean context string
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

TOP_K_CANDIDATES = 10
TOP_K_FINAL      = 4      # chunks sent to LLM — keep small for speed
MMR_LAMBDA       = 0.65   # higher = more relevant, lower = more diverse


def _rrf(sem: np.ndarray, kw: np.ndarray, k: int = 60, w: float = 0.7) -> np.ndarray:
    n  = len(sem)
    rs = n - sem.argsort().argsort()
    rk = n - kw.argsort().argsort()
    return w / (k + rs) + (1 - w) / (k + rk)


def _mmr(q_emb: np.ndarray, candidates: list, embs: np.ndarray,
         top_k: int, lam: float = MMR_LAMBDA) -> list:
    selected, remaining = [], list(candidates)
    while remaining and len(selected) < top_k:
        if not selected:
            scores = [float(q_emb @ embs[i]) for i in remaining]
        else:
            scores = [
                lam * float(q_emb @ embs[i])
                - (1 - lam) * max(float(embs[i] @ embs[s]) for s in selected)
                for i in remaining
            ]
        best = remaining[int(np.argmax(scores))]
        selected.append(best)
        remaining.remove(best)
    return selected


def retrieve(store, query: str, top_k: int = TOP_K_FINAL) -> list:
    """Return list of {text, score} dicts."""
    if store.embeddings is None or not store.chunks:
        return []

    q_emb  = store.encode_query(query)
    sem    = store.semantic_scores(q_emb)
    kw     = store.keyword_scores(query)
    rrf    = _rrf(sem, kw)

    cands  = rrf.argsort()[::-1][:TOP_K_CANDIDATES].tolist()
    final  = _mmr(q_emb, cands, store.embeddings, top_k)

    return [{"text": store.chunks[i], "score": float(rrf[i])} for i in final]


def build_context(results: list) -> str:
    """Format chunks into a numbered context block for the LLM."""
    if not results:
        return ""
    return "\n\n".join(f"[{i+1}] {r['text'].strip()}" for i, r in enumerate(results))
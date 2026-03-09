import os
import math
import re
import pickle
import logging

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CACHE_FILE       = "vector_store.pkl"   # matches the .pkl already in your project


class VectorStore:
    def __init__(self, model_name: str = EMBED_MODEL_NAME, cache_path: str = CACHE_FILE):
        self.model_name  = model_name
        self.cache_path  = cache_path
        self.chunks: list      = []
        self.embeddings        = None   # np.ndarray once built
        self._model            = None   # lazy-loaded

    @property
    def model(self):
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    # ── Build ──────────────────────────────────────────────────────────────────

    def build(self, chunks: list) -> None:
        self.chunks = chunks
        logger.info(f"Encoding {len(chunks)} chunks…")
        self.embeddings = self.model.encode(
            chunks,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        self._save()

    def _save(self) -> None:
        with open(self.cache_path, "wb") as f:
            pickle.dump({"chunks": self.chunks, "embeddings": self.embeddings}, f)
        logger.info(f"Vector store saved → {self.cache_path}")

    def load(self) -> bool:
        if not os.path.exists(self.cache_path):
            return False
        try:
            with open(self.cache_path, "rb") as f:
                data = pickle.load(f)
            self.chunks     = data["chunks"]
            self.embeddings = data["embeddings"]
            logger.info(f"Loaded {len(self.chunks)} chunks from cache.")
            return True
        except Exception as e:
            logger.warning(f"Cache load failed ({e}), will rebuild.")
            return False

    # ── Scoring ────────────────────────────────────────────────────────────────

    def semantic_scores(self, query: str) -> np.ndarray:
        q_emb = self.model.encode(query, normalize_embeddings=True)
        return (self.embeddings @ q_emb).astype(float)

    def keyword_scores(self, query: str) -> np.ndarray:
        """BM25-lite: token overlap normalised by log(chunk length)."""
        query_tokens = set(re.findall(r'\w+', query.lower()))
        scores = np.zeros(len(self.chunks))
        for i, chunk in enumerate(self.chunks):
            tokens = re.findall(r'\w+', chunk.lower())
            if not tokens:
                continue
            matches = sum(1 for t in tokens if t in query_tokens)
            scores[i] = matches / (1 + math.log(len(tokens)))
        return scores

    def encode_query(self, query: str) -> np.ndarray:
        return self.model.encode(query, normalize_embeddings=True)
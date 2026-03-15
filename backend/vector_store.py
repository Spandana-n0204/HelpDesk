import os, math, re, pickle, logging
import numpy as np
from sentence_transformers import SentenceTransformer
 
logger = logging.getLogger(__name__)
 
EMBED_MODEL = "all-MiniLM-L6-v2"   # 80 MB, fast on CPU
CACHE_FILE  = "vector_store.pkl"
 
 
class VectorStore:
    def __init__(self, model_name: str = EMBED_MODEL, cache_path: str = CACHE_FILE):
        self.model_name  = model_name
        self.cache_path  = cache_path
        self.chunks: list       = []
        self.embeddings         = None   # np.ndarray (N, 384), float32
        self._model             = None   # loaded lazily
 
    # ── Model (lazy) ───────────────────────────────────────────────────────────
    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name} …")
            self._model = SentenceTransformer(self.model_name)
        return self._model
 
    # ── Build / Save / Load ────────────────────────────────────────────────────
    def build(self, chunks: list) -> None:
        self.chunks = chunks
        logger.info(f"Encoding {len(chunks)} chunks …")
        self.embeddings = self.model.encode(
            chunks,
            batch_size=128,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")
        self._save()
        logger.info("Vector store built and cached.")
 
    def _save(self) -> None:
        with open(self.cache_path, "wb") as f:
            pickle.dump({"chunks": self.chunks, "embeddings": self.embeddings}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
 
    def load(self) -> bool:
        if not os.path.exists(self.cache_path):
            return False
        try:
            with open(self.cache_path, "rb") as f:
                data = pickle.load(f)
            self.chunks     = data["chunks"]
            self.embeddings = data["embeddings"].astype("float32")
            logger.info(f"Loaded {len(self.chunks)} chunks from cache.")
            return True
        except Exception as e:
            logger.warning(f"Cache load failed ({e}) — will rebuild.")
            return False
 
    # ── Scoring ────────────────────────────────────────────────────────────────
    def encode_query(self, query: str) -> np.ndarray:
        return self.model.encode(
            query, normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")
 
    def semantic_scores(self, query_emb: np.ndarray) -> np.ndarray:
        # Fast dot product (embeddings already normalised → = cosine)
        return (self.embeddings @ query_emb).astype("float64")
 
    def keyword_scores(self, query: str) -> np.ndarray:
        """BM25-lite: token overlap / log(chunk_len)."""
        qt     = set(re.findall(r'\w+', query.lower()))
        scores = np.zeros(len(self.chunks), dtype="float64")
        for i, chunk in enumerate(self.chunks):
            toks = re.findall(r'\w+', chunk.lower())
            if toks:
                scores[i] = sum(1 for t in toks if t in qt) / (1 + math.log(len(toks)))
        return scores
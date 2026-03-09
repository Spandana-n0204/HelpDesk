import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from document_loader import load_all_chunks
from vector_store     import VectorStore
from semantic_search  import retrieve
from context_manager  import build_context
from llm_generator    import generate_answer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DOCS_DIRS = [
    os.path.join(BASE_DIR, "..", "documents"),
    os.path.join(BASE_DIR, "..", "extracted_data"),
    os.path.join(BASE_DIR, "data"),
]

# ── Global store ───────────────────────────────────────────────────────────────
store = VectorStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load or build the vector store once at startup."""
    global store
    logger.info("Starting DSCE HelpDesk…")
    if not store.load():
        logger.info("No cache found — building index from documents…")
        try:
            chunks = load_all_chunks(DOCS_DIRS)
            store.build(chunks)
        except Exception as e:
            logger.error(f"Index build failed: {e}")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="DSCE HelpDesk API",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ─────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    question: str
    answer:   str
    context:  str = None   # only populated by /chat/debug


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "service": "DSCE HelpDesk API v2"}


@app.get("/health")
def health():
    return {
        "status":        "ok",
        "chunks_loaded": len(store.chunks),
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    results = retrieve(store, question)
    context = build_context(results)
    answer  = generate_answer(question, context)

    return ChatResponse(question=question, answer=answer)


@app.post("/chat/debug", response_model=ChatResponse)
def chat_debug(req: ChatRequest):
    """Same as /chat but also returns the retrieved context for inspection."""
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    results = retrieve(store, question)
    context = build_context(results)
    answer  = generate_answer(question, context)

    return ChatResponse(question=question, answer=answer, context=context)


@app.get("/debug/search")
def debug_search(q: str, top_k: int = 6):
    """
    Inspect raw retrieval results without calling the LLM.
    Use this to verify chunking & ranking quality.

    Example: GET /debug/search?q=KCET+fee+DSCE
    """
    results = retrieve(store, q, top_k_final=top_k)
    return {"query": q, "results": results}


@app.post("/rebuild")
def rebuild():
    """Force a full re-index. Call after adding/updating documents."""
    global store
    try:
        chunks = load_all_chunks(DOCS_DIRS)
        store.build(chunks)
        return {"message": "Rebuilt successfully.", "chunk_count": len(store.chunks)}
    except Exception as e:
        logger.error(f"Rebuild failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
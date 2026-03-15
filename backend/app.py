"""
app.py — FastAPI backend for DSCE HelpDesk
Fix: DOCS_DIRS now includes backend/data so FAQs are indexed
"""

import logging, os, uuid
from collections import defaultdict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from document_loader import load_all_chunks
from vector_store    import VectorStore
from semantic_search import retrieve, build_context
from llm             import generate_answer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))

# ── IMPORTANT: lists every folder that contains your documents ─────────────────
DOCS_DIRS = [
    os.path.join(BASE_DIR, "data"),               # backend/data  ← your FAQs live here
    os.path.join(BASE_DIR, "data", "faqs"),        # backend/data/faqs explicitly
    os.path.join(BASE_DIR, "..", "docs"),          # project root /docs
    os.path.join(BASE_DIR, "..", "documents"),     # project root /documents
    os.path.join(BASE_DIR, "..", "extracted_data"),
]

store         = VectorStore()
chat_sessions = defaultdict(list)
MAX_HISTORY   = 8


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting DSCE HelpDesk …")
    if not store.load():
        logger.info("No cache — building index …")
        try:
            chunks = load_all_chunks(DOCS_DIRS)
            store.build(chunks)
        except Exception as e:
            logger.error(f"Index build failed: {e}")
    logger.info(f"Ready — {len(store.chunks)} chunks loaded.")
    yield


app = FastAPI(title="DSCE HelpDesk API", version="3.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question:   str
    session_id: str = ""

class ChatResponse(BaseModel):
    session_id: str
    question:   str
    answer:     str
    context:    str = None


@app.get("/")
def root():
    return {"status": "ok", "chunks_loaded": len(store.chunks)}

@app.get("/health")
def health():
    return {"status": "ok", "chunks_loaded": len(store.chunks)}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(400, "Question must not be empty.")

    sid     = req.session_id or str(uuid.uuid4())
    history = chat_sessions[sid]

    results = retrieve(store, question)
    context = build_context(results)
    answer  = generate_answer(question, context, history)

    history.append({"role": "user",      "content": question})
    history.append({"role": "assistant", "content": answer})
    if len(history) > MAX_HISTORY:
        chat_sessions[sid] = history[-MAX_HISTORY:]

    return ChatResponse(session_id=sid, question=question, answer=answer)


@app.post("/chat/debug", response_model=ChatResponse)
def chat_debug(req: ChatRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(400, "Question must not be empty.")

    sid     = req.session_id or str(uuid.uuid4())
    history = chat_sessions[sid]

    results = retrieve(store, question)
    context = build_context(results)
    answer  = generate_answer(question, context, history)

    history.append({"role": "user",      "content": question})
    history.append({"role": "assistant", "content": answer})

    return ChatResponse(session_id=sid, question=question, answer=answer, context=context)


@app.get("/debug/search")
def debug_search(q: str, top_k: int = 5):
    """Check retrieval without calling LLM. GET /debug/search?q=diploma+admission"""
    results = retrieve(store, q, top_k=top_k)
    return {"query": q, "results": results}


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    chat_sessions.pop(session_id, None)
    return {"message": "Session cleared."}


@app.post("/rebuild")
def rebuild():
    """Force re-index after adding/updating documents."""
    try:
        chunks = load_all_chunks(DOCS_DIRS)
        store.build(chunks)
        return {"message": "Rebuilt.", "chunk_count": len(store.chunks)}
    except Exception as e:
        raise HTTPException(500, str(e))
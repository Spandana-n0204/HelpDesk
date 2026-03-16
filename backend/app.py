"""
app.py — FastAPI backend for DSCE HelpDesk
Now with persistent chat history via SQLite.
New endpoints:
  GET  /conversations?device_id=xxx        — list all past chats
  GET  /conversations/{id}/messages        — load a specific chat
  DELETE /conversations/{id}               — delete a chat
"""

import logging, os, uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from document_loader import load_all_chunks
from vector_store    import VectorStore
from semantic_search import retrieve, build_context
from llm             import generate_answer
from database        import (
    init_db, create_conversation, get_conversations,
    get_messages, get_recent_messages, save_message,
    update_conversation_title, delete_conversation,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DOCS_DIRS = [
    os.path.join(BASE_DIR, "data"),
    os.path.join(BASE_DIR, "data", "faqs"),
    os.path.join(BASE_DIR, "..", "docs"),
    os.path.join(BASE_DIR, "..", "documents"),
    os.path.join(BASE_DIR, "..", "extracted_data"),
]

store = VectorStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting DSCE HelpDesk …")
    # Init SQLite database
    init_db()
    logger.info("Database ready.")
    # Load vector store
    if not store.load():
        logger.info("No cache — building index …")
        try:
            chunks = load_all_chunks(DOCS_DIRS)
            store.build(chunks)
        except Exception as e:
            logger.error(f"Index build failed: {e}")
    logger.info(f"Ready — {len(store.chunks)} chunks loaded.")
    yield


app = FastAPI(title="DSCE HelpDesk API", version="4.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ──────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question:        str
    conversation_id: str = ""   # empty = start new conversation
    device_id:       str = ""   # identifies the user's browser


class ChatResponse(BaseModel):
    conversation_id: str
    question:        str
    answer:          str


# ── Chat endpoint ──────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "chunks_loaded": len(store.chunks)}


@app.get("/health")
def health():
    return {"status": "ok", "chunks_loaded": len(store.chunks)}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    question  = req.question.strip()
    device_id = req.device_id or "anonymous"

    if not question:
        raise HTTPException(400, "Question must not be empty.")

    # Create new conversation if needed
    conv_id = req.conversation_id or str(uuid.uuid4())
    existing = get_messages(conv_id)
    if not existing:
        create_conversation(conv_id, device_id)
        # Use first question as conversation title
        update_conversation_title(conv_id, question)

    # Get recent history for context memory
    history = get_recent_messages(conv_id, limit=8)

    # Retrieve relevant chunks
    results = retrieve(store, question)
    context = build_context(results)

    # Generate answer
    answer = generate_answer(question, context, history)

    # Save both messages to database
    save_message(conv_id, "user",      question)
    save_message(conv_id, "assistant", answer)

    return ChatResponse(conversation_id=conv_id, question=question, answer=answer)


# ── History endpoints ──────────────────────────────────────────────────────────

@app.get("/conversations")
def list_conversations(device_id: str = "anonymous"):
    """
    Get all past conversations for a device.
    Frontend calls this on load to populate the sidebar.
    GET /conversations?device_id=abc123
    """
    return get_conversations(device_id)


@app.get("/conversations/{conversation_id}/messages")
def load_conversation(conversation_id: str):
    """
    Load all messages in a conversation.
    Frontend calls this when user clicks a past chat in sidebar.
    GET /conversations/abc-123/messages
    """
    messages = get_messages(conversation_id)
    if not messages:
        raise HTTPException(404, "Conversation not found.")
    return messages


@app.delete("/conversations/{conversation_id}")
def remove_conversation(conversation_id: str):
    """Delete a conversation and all its messages."""
    delete_conversation(conversation_id)
    return {"message": "Deleted."}


# ── Debug endpoints ────────────────────────────────────────────────────────────

@app.get("/debug/search")
def debug_search(q: str, top_k: int = 5):
    """Check retrieval without calling LLM. GET /debug/search?q=KCET+fee"""
    results = retrieve(store, q, top_k=top_k)
    return {"query": q, "results": results}


@app.post("/rebuild")
def rebuild():
    """Force re-index all documents after updating data files."""
    try:
        chunks = load_all_chunks(DOCS_DIRS)
        store.build(chunks)
        return {"message": "Rebuilt.", "chunk_count": len(store.chunks)}
    except Exception as e:
        raise HTTPException(500, str(e))
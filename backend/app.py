import logging, os, uuid
from contextlib import asynccontextmanager
from typing import Optional

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
    init_db()
    logger.info("Database ready.")
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


class ChatRequest(BaseModel):
    question:        str
    conversation_id: Optional[str] = None   # None = start new conversation
    device_id:       Optional[str] = ""


class ChatResponse(BaseModel):
    conversation_id: str
    question:        str
    answer:          str


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
    conv_id  = req.conversation_id or str(uuid.uuid4())
    existing = get_messages(conv_id)
    if not existing:
        create_conversation(conv_id, device_id)
        update_conversation_title(conv_id, question)

    history = get_recent_messages(conv_id, limit=8)
    results = retrieve(store, question)
    context = build_context(results)
    answer  = generate_answer(question, context, history)

    save_message(conv_id, "user",      question)
    save_message(conv_id, "assistant", answer)

    return ChatResponse(conversation_id=conv_id, question=question, answer=answer)


@app.get("/conversations")
def list_conversations(device_id: str = "anonymous"):
    return get_conversations(device_id)


@app.get("/conversations/{conversation_id}/messages")
def load_conversation(conversation_id: str):
    messages = get_messages(conversation_id)
    if not messages:
        raise HTTPException(404, "Conversation not found.")
    return messages


@app.delete("/conversations/{conversation_id}")
def remove_conversation(conversation_id: str):
    delete_conversation(conversation_id)
    return {"message": "Deleted."}


@app.get("/debug/search")
def debug_search(q: str, top_k: int = 5):
    results = retrieve(store, q, top_k=top_k)
    return {"query": q, "results": results}


@app.post("/rebuild")
def rebuild():
    try:
        chunks = load_all_chunks(DOCS_DIRS)
        store.build(chunks)
        return {"message": "Rebuilt.", "chunk_count": len(store.chunks)}
    except Exception as e:
        raise HTTPException(500, str(e))


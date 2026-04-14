
import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_history.db")


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row   # lets us access columns by name
    return conn


def init_db():
    """Create tables if they don't exist. Called once at startup."""
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS conversations (
            id         TEXT PRIMARY KEY,
            device_id  TEXT NOT NULL,
            title      TEXT NOT NULL DEFAULT 'New Chat',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS messages (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            role            TEXT NOT NULL,   -- 'user' or 'assistant'
            content         TEXT NOT NULL,
            created_at      TEXT NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        );

        CREATE INDEX IF NOT EXISTS idx_conv_device ON conversations(device_id);
        CREATE INDEX IF NOT EXISTS idx_msg_conv    ON messages(conversation_id);
    """)
    conn.commit()
    conn.close()


# ── Conversations ──────────────────────────────────────────────────────────────

def create_conversation(conversation_id: str, device_id: str) -> dict:
    now  = datetime.utcnow().isoformat()
    conn = get_conn()
    conn.execute(
        "INSERT INTO conversations (id, device_id, title, created_at, updated_at) VALUES (?,?,?,?,?)",
        (conversation_id, device_id, "New Chat", now, now)
    )
    conn.commit()
    conn.close()
    return {"id": conversation_id, "title": "New Chat", "created_at": now}


def update_conversation_title(conversation_id: str, title: str):
    """Set title from first user message (trimmed to 50 chars)."""
    conn = get_conn()
    conn.execute(
        "UPDATE conversations SET title=?, updated_at=? WHERE id=?",
        (title[:50], datetime.utcnow().isoformat(), conversation_id)
    )
    conn.commit()
    conn.close()


def get_conversations(device_id: str) -> list:
    """Return all conversations for a device, newest first."""
    conn  = get_conn()
    rows  = conn.execute(
        "SELECT id, title, created_at, updated_at FROM conversations "
        "WHERE device_id=? ORDER BY updated_at DESC",
        (device_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_conversation(conversation_id: str):
    conn = get_conn()
    conn.execute("DELETE FROM messages      WHERE conversation_id=?", (conversation_id,))
    conn.execute("DELETE FROM conversations WHERE id=?",              (conversation_id,))
    conn.commit()
    conn.close()


# ── Messages ───────────────────────────────────────────────────────────────────

def save_message(conversation_id: str, role: str, content: str):
    now  = datetime.utcnow().isoformat()
    conn = get_conn()
    conn.execute(
        "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?,?,?,?)",
        (conversation_id, role, content, now)
    )
    # Update conversation's updated_at timestamp
    conn.execute(
        "UPDATE conversations SET updated_at=? WHERE id=?",
        (now, conversation_id)
    )
    conn.commit()
    conn.close()


def get_messages(conversation_id: str) -> list:
    """Return all messages in a conversation, oldest first."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT role, content, created_at FROM messages "
        "WHERE conversation_id=? ORDER BY id ASC",
        (conversation_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_recent_messages(conversation_id: str, limit: int = 8) -> list:
    """Return last N messages as {role, content} for LLM history."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT role, content FROM messages "
        "WHERE conversation_id=? ORDER BY id DESC LIMIT ?",
        (conversation_id, limit)
    ).fetchall()
    conn.close()
    return [dict(r) for r in reversed(rows)]
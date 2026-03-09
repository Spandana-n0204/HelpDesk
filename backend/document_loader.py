import json
import os
import re
import hashlib
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MIN_CHUNK_CHARS = 60   # drop anything shorter than this


# ── Helpers ────────────────────────────────────────────────────────────────────

def _chunk_hash(text: str) -> str:
    return hashlib.md5(text.strip().lower().encode()).hexdigest()


def _is_noise(text: str) -> bool:
    """Return True for keyword-soup or too-short chunks."""
    text = text.strip()
    if len(text) < MIN_CHUNK_CHARS:
        return True
    words = text.split()
    # Short phrase with no sentence-like punctuation → likely a keyword tag
    if len(words) <= 4 and not any(c in text for c in (".", ",", ":", "-", "?")):
        return True
    return False


# ── JSON → Natural Language ────────────────────────────────────────────────────

def _json_to_chunks(data: Any, context_prefix: str = "") -> list:
    """
    Recursively convert a JSON structure into readable natural-language chunks.
    Each chunk carries its full context path so it's meaningful in isolation.
    """
    chunks = []

    if isinstance(data, dict):
        for key, value in data.items():
            human_key = key.replace("_", " ").replace("-", " ").strip()
            child_prefix = f"{context_prefix} > {human_key}" if context_prefix else human_key

            if isinstance(value, (str, int, float, bool)):
                chunks.append(f"{child_prefix}: {value}")

            elif isinstance(value, list):
                if all(isinstance(v, (str, int, float)) for v in value):
                    items = ", ".join(str(v) for v in value)
                    chunks.append(f"{child_prefix}: {items}")
                else:
                    for item in value:
                        chunks.extend(_json_to_chunks(item, child_prefix))

            elif isinstance(value, dict):
                # Small flat dicts → one readable sentence
                if all(isinstance(v, (str, int, float, bool)) for v in value.values()) and len(value) <= 6:
                    parts = [f"{k.replace('_', ' ')}: {v}" for k, v in value.items()]
                    chunks.append(f"{child_prefix} — " + "; ".join(parts))
                else:
                    chunks.extend(_json_to_chunks(value, child_prefix))

    elif isinstance(data, list):
        for item in data:
            chunks.extend(_json_to_chunks(item, context_prefix))

    elif isinstance(data, (str, int, float, bool)):
        if context_prefix:
            chunks.append(f"{context_prefix}: {data}")
        else:
            chunks.append(str(data))

    return chunks


def _flatten_json_file(path: str) -> list:
    """Load a structured JSON file and return natural-language chunks."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    topic = Path(path).stem.replace("_", " ").replace("-", " ").title()
    raw_chunks = _json_to_chunks(data, topic)

    # Combine tiny sibling chunks into richer paragraphs
    combined = []
    buffer = ""
    for chunk in raw_chunks:
        if len(buffer) + len(chunk) < 350:
            buffer = (buffer + " | " + chunk).strip(" |")
        else:
            if buffer:
                combined.append(buffer)
            buffer = chunk
    if buffer:
        combined.append(buffer)

    return combined


def _load_faq_json(path: str) -> list:
    """Parse FAQ JSON: list of {question, answer} or dict of q:a pairs."""
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                q = item.get("question", item.get("q", ""))
                a = item.get("answer",   item.get("a", ""))
                if q and a:
                    chunks.append(f"Q: {q}\nA: {a}")
                else:
                    chunks.extend(_json_to_chunks(item))
    elif isinstance(data, dict):
        for q, a in data.items():
            if isinstance(a, str):
                chunks.append(f"Q: {q}\nA: {a}")
            else:
                chunks.extend(_json_to_chunks(a, q))

    return chunks


# ── Plain Text Chunker ─────────────────────────────────────────────────────────

def _chunk_text(text: str, max_chars: int = 400, overlap: int = 80) -> list:
    """Sentence-aware sliding-window chunker."""
    sentences = re.split(r'(?<=[.?!])\s+', text.strip())
    chunks = []
    current = ""

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if len(current) + len(sent) + 1 <= max_chars:
            current = (current + " " + sent).strip()
        else:
            if current:
                chunks.append(current)
            overlap_text = current[-overlap:] if len(current) > overlap else current
            current = (overlap_text + " " + sent).strip()

    if current:
        chunks.append(current)

    return chunks


# ── Public API ─────────────────────────────────────────────────────────────────

def load_all_chunks(docs_dirs: list) -> list:
    """
    Walk directories, load every supported file, return clean deduplicated chunks.
    """
    raw_chunks = []

    for docs_dir in docs_dirs:
        if not os.path.exists(docs_dir):
            logger.warning(f"Directory not found, skipping: {docs_dir}")
            continue

        for root, _, files in os.walk(docs_dir):
            for fname in files:
                fpath = os.path.join(root, fname)
                try:
                    if fname.endswith(".json"):
                        stem = fname.lower()
                        if "faq" in stem or "qa" in stem:
                            raw_chunks.extend(_load_faq_json(fpath))
                        else:
                            raw_chunks.extend(_flatten_json_file(fpath))
                    elif fname.endswith((".txt", ".md")):
                        with open(fpath, "r", encoding="utf-8") as f:
                            raw_chunks.extend(_chunk_text(f.read()))
                except Exception as e:
                    logger.warning(f"Could not load {fpath}: {e}")

    # Filter noise + deduplicate
    seen = set()
    clean = []
    for chunk in raw_chunks:
        chunk = chunk.strip()
        if _is_noise(chunk):
            continue
        h = _chunk_hash(chunk)
        if h not in seen:
            seen.add(h)
            clean.append(chunk)

    logger.info(f"Loaded {len(clean)} clean chunks from {len(raw_chunks)} raw fragments.")
    return clean
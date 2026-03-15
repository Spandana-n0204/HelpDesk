"""
document_loader.py — Load and chunk DSCE documents
Fix: skips master_index.json (table of contents — pollutes retrieval)
"""

import json, os, re, hashlib, logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MIN_CHUNK_CHARS = 60

# Skip these files — they are index/manifest files, not real content
SKIP_FILES = {"master_index.json"}


def _hash(text: str) -> str:
    return hashlib.md5(text.strip().lower().encode()).hexdigest()


def _is_noise(text: str) -> bool:
    text = text.strip()
    if len(text) < MIN_CHUNK_CHARS:
        return True
    words = text.split()
    if len(words) <= 4 and not any(c in text for c in (".", ",", ":", "-", "?")):
        return True
    return False


def _json_to_chunks(data: Any, prefix: str = "") -> list:
    chunks = []
    if isinstance(data, dict):
        for key, value in data.items():
            hkey  = key.replace("_", " ").replace("-", " ").strip()
            child = f"{prefix} > {hkey}" if prefix else hkey
            if isinstance(value, (str, int, float, bool)):
                chunks.append(f"{child}: {value}")
            elif isinstance(value, list):
                if all(isinstance(v, (str, int, float)) for v in value):
                    chunks.append(f"{child}: {', '.join(str(v) for v in value)}")
                else:
                    for item in value:
                        chunks.extend(_json_to_chunks(item, child))
            elif isinstance(value, dict):
                if all(isinstance(v, (str, int, float, bool)) for v in value.values()) and len(value) <= 6:
                    parts = [f"{k.replace('_', ' ')}: {v}" for k, v in value.items()]
                    chunks.append(f"{child} — " + "; ".join(parts))
                else:
                    chunks.extend(_json_to_chunks(value, child))
    elif isinstance(data, list):
        for item in data:
            chunks.extend(_json_to_chunks(item, prefix))
    elif isinstance(data, (str, int, float, bool)):
        chunks.append(f"{prefix}: {data}" if prefix else str(data))
    return chunks


def _load_structured_json(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    topic  = Path(path).stem.replace("_", " ").replace("-", " ").title()
    raw    = _json_to_chunks(data, topic)
    merged, buf = [], ""
    for c in raw:
        if len(buf) + len(c) < 380:
            buf = (buf + " | " + c).strip(" |")
        else:
            if buf:
                merged.append(buf)
            buf = c
    if buf:
        merged.append(buf)
    return merged


def _load_faq_json(path: str) -> list:
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


def _chunk_text(text: str, max_chars: int = 420, overlap: int = 80) -> list:
    sentences = re.split(r'(?<=[.?!])\s+', text.strip())
    chunks, cur = [], ""
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if len(cur) + len(s) + 1 <= max_chars:
            cur = (cur + " " + s).strip()
        else:
            if cur:
                chunks.append(cur)
            tail = cur[-overlap:] if len(cur) > overlap else cur
            cur  = (tail + " " + s).strip()
    if cur:
        chunks.append(cur)
    return chunks


def load_all_chunks(docs_dirs: list) -> list:
    raw = []
    for d in docs_dirs:
        if not os.path.exists(d):
            logger.warning(f"Skipping missing dir: {d}")
            continue
        for root, _, files in os.walk(d):
            for fname in files:
                if fname.lower() in SKIP_FILES:
                    logger.info(f"Skipping: {fname}")
                    continue
                fpath = os.path.join(root, fname)
                try:
                    if fname.endswith(".json"):
                        if any(x in fname.lower() for x in ("faq", "qa")):
                            raw.extend(_load_faq_json(fpath))
                        else:
                            raw.extend(_load_structured_json(fpath))
                    elif fname.endswith((".txt", ".md")):
                        raw.extend(_chunk_text(open(fpath, encoding="utf-8").read()))
                except Exception as e:
                    logger.warning(f"Could not load {fpath}: {e}")

    seen, clean = set(), []
    for c in raw:
        c = c.strip()
        if _is_noise(c):
            continue
        h = _hash(c)
        if h not in seen:
            seen.add(h)
            clean.append(c)

    logger.info(f"Loaded {len(clean)} clean chunks (from {len(raw)} raw).")
    return clean
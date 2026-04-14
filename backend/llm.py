"""
llm.py — LLM via Groq API for DSCE HelpDesk
"""

import os
import logging
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "llama-3.1-8b-instant"

SYSTEM_PROMPT = """You are the DSCE HelpDesk assistant for Dayananda Sagar College of Engineering (DSCE), Bengaluru.

Answer questions using ONLY the information in the CONTEXT provided.

STRICT RULES — follow every one of these:
1. NEVER say "based on the provided context", "according to the context", "as per the context", "the context mentions", "the provided context" or any similar phrase. Just answer directly.
2. NEVER say "I would recommend contacting the admissions office for up-to-date information" unless the context truly has no relevant information at all.
3. NEVER say "the context does not provide" — if you don't have the info, just say "I don't have that information. Please contact DSCE at admissions@dsce.edu.in"
4. Give detailed, helpful answers using everything available in the context.
5. If the question is a follow-up, use conversation history to understand what topic they mean.
6. For admissions: list steps clearly.
7. For fees: always include category (KCET/Management/NRI) and year if available.
8. For courses/branches: list them clearly.
9. Answer naturally like a helpful college assistant — not like a system reading from a database.
10. Always format answers with bullet points for lists, bold for important terms, and clear sections when answering multi-part questions. Never give answers as one big paragraph."""


def generate_answer(question: str, context: str, chat_history: list = None) -> str:
    if not context or not context.strip():
        return (
            "I don't have specific information on that. "
            "Please contact DSCE at admissions@dsce.edu.in or visit www.dsce.edu.in"
        )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Last 4 turns of conversation for context memory
    if chat_history:
        for msg in chat_history[-4:]:
            role = "user" if msg["role"] == "user" else "assistant"
            content = msg.get("content", "").strip()
            if content:
                messages.append({"role": role, "content": content})

    messages.append({
        "role":    "user",
        "content": f"Context:\n{context[:1500]}\n\nQuestion: {question}",
    })

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":       GROQ_MODEL,
        "messages":    messages,
        "max_tokens":  600,
        "temperature": 0.2,
    }

    try:
        resp = requests.post(GROQ_URL, json=payload, headers=headers, timeout=30)

        if not resp.ok:
            logger.error(f"Groq {resp.status_code}: {resp.text}")

        if resp.status_code == 401:
            return "Invalid Groq API key. Please check your .env file."
        if resp.status_code == 400:
            return "Groq API request error. Check terminal for details."
        if resp.status_code == 429:
            return "Rate limit reached. Please wait a moment and try again."

        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    except requests.exceptions.Timeout:
        return "Request timed out. Please try again."
    except Exception as e:
        logger.error(f"Groq error: {e}")
        return "An error occurred. Please try again."
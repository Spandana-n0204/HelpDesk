"""
llm.py — LLM via Groq API
Fixes: longer answers, better system prompt, chat history working
"""

import os
import logging
import requests

logger = logging.getLogger(__name__)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

GROQ_URL   = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.1-8b-instant"

SYSTEM_PROMPT = """You are the DSCE HelpDesk assistant for Dayananda Sagar College of Engineering (DSCE), Bengaluru.

Your job is to answer questions helpfully and in detail using the CONTEXT provided.

Rules:
1. Answer using ONLY the information in the CONTEXT. Do not make up facts.
2. Give detailed, helpful answers — explain fully, don't just list one line.
3. If the question is a follow-up (e.g. "tell me more", "what about fees"), use the conversation history to understand what topic they mean.
4. If the context does not contain enough information, say: "I don't have complete information on that. Please contact DSCE at admissions@dsce.edu.in or visit www.dsce.edu.in"
5. For admissions: mention the process steps clearly.
6. For fees: always include the category (KCET/Management/NRI) and academic year if available.
7. For courses/branches: list them clearly.
8. Never say "Based on the context" — just answer directly and naturally."""


def generate_answer(question: str, context: str, chat_history: list = None) -> str:
    """
    Generate a detailed answer using Groq.
    chat_history: list of {"role": "user"/"assistant", "content": str}
    """
    if not context or not context.strip():
        return (
            "I don't have specific information on that. "
            "Please contact DSCE at admissions@dsce.edu.in or visit www.dsce.edu.in"
        )

    # Build messages list
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Inject last 4 turns of conversation so follow-up questions work
    if chat_history:
        for msg in chat_history[-4:]:
            role = "user" if msg["role"] == "user" else "assistant"
            content = msg.get("content", "").strip()
            if content:  # skip empty messages — Groq rejects them
                messages.append({"role": role, "content": content})

    # Add current question with retrieved context
    messages.append({
        "role": "user",
        "content": f"Context:\n{context[:1500]}\n\nQuestion: {question}",
    })

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":       GROQ_MODEL,
        "messages":    messages,
        "max_tokens":  600,    # raised from 300 — allows detailed answers
        "temperature": 0.2,    # slight creativity for more natural responses
    }

    try:
        resp = requests.post(GROQ_URL, json=payload, headers=headers, timeout=30)

        if not resp.ok:
            logger.error(f"Groq {resp.status_code}: {resp.text}")

        if resp.status_code == 401:
            return "Invalid Groq API key. Please check llm.py"
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
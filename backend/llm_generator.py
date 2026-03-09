import json
import logging
import requests

logger = logging.getLogger(__name__)

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:1b"

SYSTEM_PROMPT = """You are the official HelpDesk assistant for Dayananda Sagar College of Engineering (DSCE), Bengaluru.

Answer questions using ONLY the information in the CONTEXT block provided. Follow these rules strictly:

1. Base your answer exclusively on the CONTEXT. Do not invent facts.
2. For fee questions, always state the academic year and allotment category (KCET / Management / NRI) when present in context.
3. For document questions, list the items clearly.
4. If the CONTEXT does not contain enough information, respond with:
   "I don't have that information. Please contact DSCE at admissions@dsce.edu.in or visit https://www.dsce.edu.in"
5. Keep answers concise and helpful. Do not repeat the question back.
"""


def _build_prompt(question: str, context: str) -> str:
    return (
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        f"ANSWER:"
    )


def generate_answer(question: str, context: str) -> str:
    """
    Send question + retrieved context to Ollama, return the answer string.
    """
    if not context:
        return (
            "I don't have specific information about that. "
            "Please contact DSCE at admissions@dsce.edu.in or visit https://www.dsce.edu.in"
        )

    payload = {
        "model":  OLLAMA_MODEL,
        "system": SYSTEM_PROMPT,
        "prompt": _build_prompt(question, context),
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p":       0.9,
            "num_predict": 512,
        },
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        answer = resp.json().get("response", "").strip()
        return answer or "I was unable to generate a response. Please rephrase your question."

    except requests.exceptions.ConnectionError:
        logger.error("Ollama not reachable — is it running?")
        return "The AI service is currently unavailable. Please try again shortly."
    except requests.exceptions.Timeout:
        logger.error("Ollama request timed out.")
        return "The request timed out. Please try again."
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return "An error occurred. Please try again."


def generate_answer_streaming(question: str, context: str):
    """Generator that yields tokens for streaming responses."""
    payload = {
        "model":  OLLAMA_MODEL,
        "system": SYSTEM_PROMPT,
        "prompt": _build_prompt(question, context),
        "stream": True,
        "options": {"temperature": 0.1, "top_p": 0.9, "num_predict": 512},
    }
    try:
        with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=120) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line:
                    chunk = json.loads(line)
                    yield chunk.get("response", "")
                    if chunk.get("done"):
                        break
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        yield "\n[Error generating response]"
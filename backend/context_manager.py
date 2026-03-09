def build_context(results: list) -> str:
    """
    Format retrieved chunks into a numbered context block for the LLM.
    Each chunk is separated clearly so the model can reference them.
    """
    if not results:
        return ""

    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r['text'].strip()}")

    return "\n\n".join(lines)
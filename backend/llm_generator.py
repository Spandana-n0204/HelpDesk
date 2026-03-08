import ollama


def clean_context(context_docs):
    cleaned_lines = []

    for doc in context_docs:
        line = doc.strip()

        # remove common metadata prefixes
        prefixes = [
            "faqs question:",
            "faqs keywords:",
            "faqs short_answer:",
            "faqs detailed_answer:",
            "files keywords:",
            "files topics:",
            "files description:"
        ]

        for p in prefixes:
            if line.lower().startswith(p):
                line = line[len(p):].strip()

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def generate_response(question: str, context_docs):

    print("\nRetrieved docs:")
    for doc in context_docs:
        print(doc)

    context = clean_context(context_docs)

    print("\nContext sent to LLM:")
    print(context)

    response = ollama.chat(
        model="llama3.2:1b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an assistant answering questions about "
                    "Dayananda Sagar College of Engineering (DSCE). "
                    "Use the provided context to answer clearly. "
                    "If the answer is present in the context, state it directly. "
                    "If it is not present, say: "
                    "'I don't have enough information about that.'"
                )
            },
            {
                "role": "user",
                "content": f"""
Context:
{context}

Question:
{question}

Answer clearly using the context above.
"""
            }
        ]
    )

    return response["message"]["content"]
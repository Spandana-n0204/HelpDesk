import ollama


def generate_response(question: str, context_docs):

    # join retrieved docs
    context = "\n\n".join(context_docs)

    print("\n Retrieved docs:")
    for doc in context_docs:
        print(doc)
    print(context)
    prompt = f"""
You are a college information assistant for Dayananda Sagar College of Engineering.

Answer the question ONLY using the information from the context below.

IMPORTANT RULES:
- Do NOT guess numbers.
- Do NOT create information.
- If the answer contains numbers (fees, amounts, dates), copy them EXACTLY from the context.
- If the information is not present in the context, reply exactly:
"I don't have enough information about that."

Context:
{context}

Question:
{question}

Answer:
"""

    response = ollama.chat(
        model="llama3.2:1b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]
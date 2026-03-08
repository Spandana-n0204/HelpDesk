import pickle
import numpy as np
import faiss

# Load FAISS index
index = faiss.read_index("faiss_index.index")

# Load vectorizer
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load documents
with open("documents.pkl", "rb") as f:
    documents = pickle.load(f)


def format_content(content):

    if isinstance(content, list):
        return "\n".join([format_content(item) for item in content])

    elif isinstance(content, dict):
        return "\n".join([f"{key}: {format_content(value)}" for key, value in content.items()])

    else:
        return str(content)


def generate_response(question: str):

    question_vector = vectorizer.transform([question]).toarray().astype("float32")

    D, I = index.search(question_vector, k=12)

    retrieved_docs = [documents[i] for i in I[0]]

    formatted_docs = []

    for doc in retrieved_docs:
        formatted_docs.append(format_content(doc))

    return formatted_docs
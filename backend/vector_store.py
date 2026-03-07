import faiss
import pickle
import numpy as np

index = faiss.read_index("faiss_index.index")

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("documents.pkl", "rb") as f:
    documents = pickle.load(f)


def search(query, k=3):
    query_vector = vectorizer.transform([query]).toarray().astype("float32")
    distances, indices = index.search(query_vector, k)

    results = [documents[i] for i in indices[0]]
    return results
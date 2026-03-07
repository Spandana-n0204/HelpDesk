import os
import json
import pickle
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_FOLDER = "data"

documents = []

for file in os.listdir(DATA_FOLDER):
    if file.endswith(".json"):
        with open(os.path.join(DATA_FOLDER, file), "r", encoding="utf-8") as f:
            data = json.load(f)

            # Convert JSON to structured text chunks
            def extract_chunks(obj, parent_key=""):
                chunks = []

                if isinstance(obj, dict):
                    for key, value in obj.items():
                        new_key = f"{parent_key} {key}".strip()
                        chunks.extend(extract_chunks(value, new_key))

                elif isinstance(obj, list):
                    for item in obj:
                        chunks.extend(extract_chunks(item, parent_key))

                else:
                    chunks.append(f"{parent_key}: {obj}")

                return chunks

            documents.extend(extract_chunks(data))

print(f"Total chunks created: {len(documents)}")

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(documents)

dense_vectors = vectors.toarray().astype("float32")

index = faiss.IndexFlatL2(dense_vectors.shape[1])
index.add(dense_vectors)

faiss.write_index(index, "faiss_index.index")

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("documents.pkl", "wb") as f:
    pickle.dump(documents, f)

print("Index rebuilt successfully.")
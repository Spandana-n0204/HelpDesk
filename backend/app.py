from fastapi import FastAPI
from pydantic import BaseModel
from semantic_engine import generate_response as semantic_search
from llm_generator import generate_response as llm_generate

app = FastAPI()


class Query(BaseModel):
    question: str


@app.post("/chat")
def chat(query: Query):

    question = query.question

    # retrieve documents using FAISS
    docs = semantic_search(question)

    # send retrieved docs to LLM
    answer = llm_generate(question, docs)

    return {"answer": answer}
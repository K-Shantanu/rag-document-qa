from fastapi import FastAPI
from app.ingestion import load_documents
from app.embedding import embed_texts, model
from app.retrieval import create_faiss_index, search
from app.llm import generate_answer

import faiss
import os

app = FastAPI()

INDEX_PATH = "faiss_index.bin"


chunks = load_documents("documents")

texts = [chunk["text"] for chunk in chunks]

if os.path.exists(INDEX_PATH):
    index = faiss.read_index(INDEX_PATH)
else:
    embeddings = embed_texts(texts)
    index = create_faiss_index(embeddings)
    faiss.write_index(index, INDEX_PATH)


@app.post("/ask")
def ask_question(question: str):

    query_embedding = model.encode([question])[0]

    results = search(index, query_embedding, chunks)

    retrieved_texts = [r["text"] for r in results]

    context = " ".join(retrieved_texts)[:1500]

    answer = generate_answer(context, question)

    return {
        "question": question,
        "answer": answer,
        "retrieved_context": results
    }
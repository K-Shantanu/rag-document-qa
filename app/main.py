from fastapi import FastAPI
from app.ingestion import load_pdf, chunk_text
from app.embedding import embed_texts, model
from app.retrieval import create_faiss_index, search
from app.llm import generate_answer

app = FastAPI()

text = load_pdf("data/sample.pdf")
chunks = chunk_text(text)
embeddings = embed_texts(chunks)
index = create_faiss_index(embeddings)


@app.post("/ask")
def ask_question(question: str):
    query_embedding = model.encode([question])[0]
    indices = search(index, query_embedding)

    retrieved_chunks = [chunks[i] for i in indices]
    context = " ".join(retrieved_chunks)[:1500]

    answer = generate_answer(context, question)

    return {
        "question": question,
        "answer": answer,
        "retrieved_context": retrieved_chunks
    }
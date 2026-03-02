# RAG-Based Document Question Answering System

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system for question answering over PDF documents.

The application allows users to ask natural language questions about a document. Instead of relying purely on a language model, the system retrieves relevant sections of the document using semantic search and generates answers grounded in the retrieved context.

The entire pipeline runs locally and does not require paid external APIs.

---

## Motivation

Large language models often produce incorrect or hallucinated responses when they do not have access to supporting information. RAG addresses this issue by retrieving relevant content before generating an answer.

This project was built to understand and implement the core components of a RAG pipeline from scratch, including document ingestion, embedding generation, vector search, context retrieval, and answer generation.

---

## System Architecture

The system follows this pipeline:

1. Extract text from a PDF document  
2. Split text into overlapping chunks  
3. Convert chunks into vector embeddings  
4. Store embeddings in a FAISS index  
5. Convert the user query into an embedding  
6. Retrieve the most similar chunks  
7. Inject retrieved context into a prompt  
8. Generate an answer using a local language model  
9. Serve responses through a FastAPI endpoint  

This architecture mirrors the design used in production document QA systems.

---

## Tech Stack

- Python
- FastAPI
- SentenceTransformers
- FAISS (CPU)
- HuggingFace Transformers (Flan-T5)
- PyPDF
- Docker

---

## Project Structure

```
rag_system/
│
├── app/
│   ├── main.py          # FastAPI entry point
│   ├── ingestion.py     # PDF loading and chunking
│   ├── embedding.py     # Embedding generation
│   ├── retrieval.py     # FAISS indexing and similarity search
│   └── llm.py           # Answer generation logic
│
├── data/
│   └── sample.pdf
│
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## How It Works

### Document Ingestion
Text is extracted from PDF files using PyPDF and combined into a single document string.

### Chunking
The text is split into fixed-size chunks with overlap to preserve semantic continuity across chunk boundaries.

### Embeddings
Each chunk is converted into a dense vector using a SentenceTransformers model (`all-MiniLM-L6-v2`). These embeddings capture semantic meaning rather than simple keyword matches.

### Vector Index
Embeddings are stored in a FAISS index to enable efficient similarity search.

### Retrieval
When a question is submitted:
- The query is embedded
- The top-k most similar chunks are retrieved
- Retrieved text is combined into contextual input for the language model

### Generation
A local HuggingFace seq2seq model (Flan-T5) generates an answer based only on the retrieved context.

The answer is returned through a REST API built with FastAPI.

---

## Running Locally

### 1. Create a virtual environment

```
python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Start the server

```
uvicorn app.main:app --reload
```

Open the API documentation at:

```
http://localhost:8000/docs
```

---

## Running with Docker

### Build the Docker image

```
docker build -t rag-system .
```

### Run the container

```
docker run -p 8000:8000 rag-system
```

Access the API at:

```
http://localhost:8000/docs
```

---

## Example API Request

Endpoint:

POST /ask

Request body:

```json
{
  "question": "Who is the wife mentioned in the document?"
}
```

Response format:

```json
{
  "question": "...",
  "answer": "...",
  "retrieved_context": [...]
}
```

---

## Limitations

- Uses a small local language model for demonstration
- Answer quality depends on model size and context length
- No reranking or advanced retrieval optimization
- Intended as an educational and portfolio project

---

## Possible Improvements

- Integrate a larger open-source model
- Add reranking for improved retrieval accuracy
- Implement sentence-aware chunking
- Add embedding caching
- Deploy to a cloud environment
- Add streaming responses

---

## Learning Outcomes

This project demonstrates:

- End-to-end RAG implementation
- Embedding-based semantic retrieval
- Vector database integration
- Context-grounded answer generation
- REST API deployment
- Containerization using Docker
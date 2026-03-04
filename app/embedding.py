from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_texts(texts):
    embeddings = model.encode(texts)
    return embeddings


def build_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index

def save_index(index, path="index.faiss"):
    faiss.write_index(index, path)

def load_index(path="index.faiss"):
    return faiss.read_index(path)
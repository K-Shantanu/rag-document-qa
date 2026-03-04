import faiss
import numpy as np


def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index


def search(index, query_embedding, chunks, k=3):

    distances, indices = index.search(
        np.array([query_embedding]), k
    )

    results = []

    for i in indices[0]:
        results.append({
            "text": chunks[i]["text"],
            "source": chunks[i]["source"]
        })

    return results
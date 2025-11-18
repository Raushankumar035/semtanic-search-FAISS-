import numpy as np
import faiss
import pickle
import re
from sentence_transformers import SentenceTransformer
import textwrap

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = pickle.load(open("chunks.pkl", "rb"))
sources = pickle.load(open("sources.pkl", "rb"))
embeddings = np.load("embeddings.npy")

index = faiss.read_index("index.faiss")


def clean_text(t):
    t = re.sub(r'\s+', ' ', t)
    return t.strip()


def semantic_search(query, top_k=3, score_threshold=0.40):
    q_emb = model.encode([query]).astype("float32")
    faiss.normalize_L2(q_emb)

    D, I = index.search(q_emb, top_k)

    results = []
    for score, idx in zip(D[0], I[0]):
        if score < score_threshold:
            continue
        results.append({
            "score": float(score),
            "source": sources[idx],
            "text": clean_text(documents[idx])
        })

    return results

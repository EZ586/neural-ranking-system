import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class FaissRetriever:
    def __init__(self, documents):
        self.documents = documents
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        texts = [doc["text"] for doc in documents]
        embeddings = self.model.encode(texts).astype("float32")

        faiss.normalize_L2(embeddings)

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def retrieve(self, query, k=5):
        q = self.model.encode([query]).astype("float32")
        faiss.normalize_L2(q)

        scores, indices = self.index.search(q, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            doc = dict(self.documents[idx])
            doc["score"] = float(score)
            results.append(doc)

        return results
from sentence_transformers import CrossEncoder

class CrossEncoderRanker:
    def __init__(self):
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def rerank(self, query, documents):
        pairs = [[query, doc["text"]] for doc in documents]
        scores = self.model.predict(pairs)

        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)

        return sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
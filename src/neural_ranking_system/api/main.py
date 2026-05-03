from fastapi import FastAPI
from src.neural_ranking_system.retrieval.faiss_retriever import FaissRetriever
from src.neural_ranking_system.ranking.cross_encoder import CrossEncoderRanker
from src.neural_ranking_system.pipeline.search_pipeline import SearchPipeline

app = FastAPI()

documents = [
    {"id": 1, "text": "Neural ranking improves search relevance."},
    {"id": 2, "text": "FAISS enables efficient vector search."},
    {"id": 3, "text": "BM25 is a strong baseline for retrieval."},
    {"id": 4, "text": "Transformers can rerank search results."},
]

retriever = FaissRetriever(documents)
ranker = CrossEncoderRanker()
pipeline = SearchPipeline(retriever, ranker)

@app.get("/search")
def search(q: str):
    return pipeline.run(q)
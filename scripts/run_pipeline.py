from src.retrieval.faiss_retriever import FaissRetriever
from src.ranking.cross_encoder import CrossEncoderRanker
from src.pipeline.search_pipeline import SearchPipeline

documents = [
    {"id": 1, "text": "Neural ranking improves search relevance."},
    {"id": 2, "text": "FAISS enables efficient vector search."},
    {"id": 3, "text": "BM25 is a strong baseline."},
    {"id": 4, "text": "Transformers rerank search results."},
]

retriever = FaissRetriever(documents)
ranker = CrossEncoderRanker()
pipeline = SearchPipeline(retriever, ranker)

results = pipeline.run("how do transformers help ranking?")

for r in results:
    print(r)
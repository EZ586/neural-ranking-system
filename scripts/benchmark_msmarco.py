from neural_ranking_system.data.msmarco_loader import load_msmarco_dev_subset
from neural_ranking_system.retrieval.faiss_retriever import FaissRetriever
from neural_ranking_system.ranking.cross_encoder import CrossEncoderRanker
from neural_ranking_system.pipeline.search_pipeline import SearchPipeline
from neural_ranking_system.evaluation.metrics import recall_at_k


def mrr_at_k(relevant_ids, predicted_ids, k):
    relevant = set(relevant_ids)

    for i, doc_id in enumerate(predicted_ids[:k]):
        if doc_id in relevant:
            return 1.0 / (i + 1)

    return 0.0


def main():
    docs, queries = load_msmarco_dev_subset(
        max_queries=100,
        max_docs=50000,
    )

    retriever = FaissRetriever(docs)
    ranker = CrossEncoderRanker()
    pipeline = SearchPipeline(retriever=retriever, ranker=ranker)

    recall_scores = []
    mrr_scores = []

    for item in queries:
        results = pipeline.run(
            query=item["query"],
            retrieve_k=50,
            final_k=10,
        )

        predicted_ids = [str(r["id"]) for r in results]
        relevant_ids = [str(x) for x in item["relevant_doc_ids"]]

        recall_scores.append(recall_at_k(relevant_ids, predicted_ids, k=10))
        mrr_scores.append(mrr_at_k(relevant_ids, predicted_ids, k=10))

    print(f"Queries evaluated: {len(queries)}")
    print(f"Recall@10: {sum(recall_scores) / len(recall_scores):.4f}")
    print(f"MRR@10: {sum(mrr_scores) / len(mrr_scores):.4f}")


if __name__ == "__main__":
    main()
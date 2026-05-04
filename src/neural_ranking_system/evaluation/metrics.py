import math
from typing import Iterable, List, Sequence, Set, Union


Id = Union[str, int]


def _to_set(ids: Iterable[Id]) -> Set[str]:
    return {str(x) for x in ids}


def _to_list(ids: Sequence[Id]) -> List[str]:
    return [str(x) for x in ids]


def precision_at_k(relevant_ids: Iterable[Id], predicted_ids: Sequence[Id], k: int) -> float:
    """
    Precision@K = relevant retrieved in top K / K
    """
    if k <= 0:
        return 0.0

    relevant = _to_set(relevant_ids)
    predicted = _to_list(predicted_ids)[:k]

    if not predicted:
        return 0.0

    hits = sum(1 for doc_id in predicted if doc_id in relevant)
    return hits / k


def recall_at_k(relevant_ids: Iterable[Id], predicted_ids: Sequence[Id], k: int) -> float:
    """
    Recall@K = relevant retrieved in top K / total relevant documents
    """
    if k <= 0:
        return 0.0

    relevant = _to_set(relevant_ids)
    predicted = _to_list(predicted_ids)[:k]

    if not relevant:
        return 0.0

    hits = sum(1 for doc_id in predicted if doc_id in relevant)
    return hits / len(relevant)


def hit_at_k(relevant_ids: Iterable[Id], predicted_ids: Sequence[Id], k: int) -> float:
    """
    Hit@K = 1 if at least one relevant document appears in top K, else 0
    """
    if k <= 0:
        return 0.0

    relevant = _to_set(relevant_ids)
    predicted = _to_list(predicted_ids)[:k]

    return 1.0 if any(doc_id in relevant for doc_id in predicted) else 0.0


def reciprocal_rank_at_k(
    relevant_ids: Iterable[Id],
    predicted_ids: Sequence[Id],
    k: int,
) -> float:
    """
    Reciprocal Rank@K = 1 / rank of first relevant result
    """
    if k <= 0:
        return 0.0

    relevant = _to_set(relevant_ids)
    predicted = _to_list(predicted_ids)[:k]

    for rank, doc_id in enumerate(predicted, start=1):
        if doc_id in relevant:
            return 1.0 / rank

    return 0.0


def mrr_at_k(
    all_relevant_ids: Sequence[Iterable[Id]],
    all_predicted_ids: Sequence[Sequence[Id]],
    k: int,
) -> float:
    """
    Mean Reciprocal Rank@K across multiple queries.
    """
    if not all_relevant_ids:
        return 0.0

    scores = [
        reciprocal_rank_at_k(relevant_ids, predicted_ids, k)
        for relevant_ids, predicted_ids in zip(all_relevant_ids, all_predicted_ids)
    ]

    return sum(scores) / len(scores) if scores else 0.0


def average_precision_at_k(
    relevant_ids: Iterable[Id],
    predicted_ids: Sequence[Id],
    k: int,
) -> float:
    """
    Average Precision@K.

    Rewards putting relevant documents earlier in the ranked list.
    """
    if k <= 0:
        return 0.0

    relevant = _to_set(relevant_ids)
    predicted = _to_list(predicted_ids)[:k]

    if not relevant:
        return 0.0

    hits = 0
    precision_sum = 0.0

    for i, doc_id in enumerate(predicted, start=1):
        if doc_id in relevant:
            hits += 1
            precision_sum += hits / i

    return precision_sum / min(len(relevant), k)


def map_at_k(
    all_relevant_ids: Sequence[Iterable[Id]],
    all_predicted_ids: Sequence[Sequence[Id]],
    k: int,
) -> float:
    """
    Mean Average Precision@K across multiple queries.
    """
    if not all_relevant_ids:
        return 0.0

    scores = [
        average_precision_at_k(relevant_ids, predicted_ids, k)
        for relevant_ids, predicted_ids in zip(all_relevant_ids, all_predicted_ids)
    ]

    return sum(scores) / len(scores) if scores else 0.0


def dcg_at_k(relevance_scores: Sequence[float], k: int) -> float:
    """
    Discounted Cumulative Gain@K.

    relevance_scores should be ordered by predicted rank.
    Example: [1, 0, 1, 0]
    """
    if k <= 0:
        return 0.0

    score = 0.0

    for i, rel in enumerate(relevance_scores[:k]):
        score += rel / math.log2(i + 2)

    return score


def ndcg_at_k_from_scores(
    relevance_scores: Sequence[float],
    k: int,
) -> float:
    """
    NDCG@K from an already-ranked list of relevance scores.
    """
    if k <= 0:
        return 0.0

    dcg = dcg_at_k(relevance_scores, k)
    ideal_scores = sorted(relevance_scores, reverse=True)
    idcg = dcg_at_k(ideal_scores, k)

    return dcg / idcg if idcg > 0 else 0.0


def ndcg_at_k(
    relevant_ids: Iterable[Id],
    predicted_ids: Sequence[Id],
    k: int,
) -> float:
    """
    Binary NDCG@K using relevant doc ids.

    Relevant documents get score 1.
    Non-relevant documents get score 0.
    """
    if k <= 0:
        return 0.0

    relevant = _to_set(relevant_ids)
    predicted = _to_list(predicted_ids)[:k]

    relevance_scores = [1.0 if doc_id in relevant else 0.0 for doc_id in predicted]

    # Add missing relevant docs to ideal ranking if there are more relevant docs than predicted docs.
    ideal_relevant_count = min(len(relevant), k)
    ideal_scores = [1.0] * ideal_relevant_count

    dcg = dcg_at_k(relevance_scores, k)
    idcg = dcg_at_k(ideal_scores, k)

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_single_query(
    relevant_ids: Iterable[Id],
    predicted_ids: Sequence[Id],
    k_values: Sequence[int] = (1, 3, 5, 10),
) -> dict:
    """
    Returns common ranking metrics for one query.
    """
    results = {}

    for k in k_values:
        results[f"precision@{k}"] = precision_at_k(relevant_ids, predicted_ids, k)
        results[f"recall@{k}"] = recall_at_k(relevant_ids, predicted_ids, k)
        results[f"hit@{k}"] = hit_at_k(relevant_ids, predicted_ids, k)
        results[f"mrr@{k}"] = reciprocal_rank_at_k(relevant_ids, predicted_ids, k)
        results[f"ndcg@{k}"] = ndcg_at_k(relevant_ids, predicted_ids, k)

    return results


def evaluate_queries(
    all_relevant_ids: Sequence[Iterable[Id]],
    all_predicted_ids: Sequence[Sequence[Id]],
    k_values: Sequence[int] = (1, 3, 5, 10),
) -> dict:
    """
    Returns average ranking metrics across multiple queries.
    """
    if len(all_relevant_ids) != len(all_predicted_ids):
        raise ValueError("all_relevant_ids and all_predicted_ids must have the same length.")

    if not all_relevant_ids:
        return {}

    totals = {}

    for relevant_ids, predicted_ids in zip(all_relevant_ids, all_predicted_ids):
        single = evaluate_single_query(relevant_ids, predicted_ids, k_values)

        for metric_name, value in single.items():
            totals[metric_name] = totals.get(metric_name, 0.0) + value

    num_queries = len(all_relevant_ids)

    return {
        metric_name: value / num_queries
        for metric_name, value in totals.items()
    }
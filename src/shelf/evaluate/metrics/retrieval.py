"""Retrieval metrics for SHELF evaluation.

All metrics are pure functions with no side effects.
Metrics follow standard IR definitions from BEIR/MTEB.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def ndcg_at_k(
    ranked_ids: list[str],
    relevant_ids: set[str],
    k: int = 10,
) -> float:
    """Compute Normalized Discounted Cumulative Gain at k.

    NDCG measures ranking quality by comparing the relevance of items
    at each position against an ideal ranking.

    Args:
        ranked_ids: Ordered list of document IDs (most relevant first)
        relevant_ids: Set of relevant document IDs (ground truth)
        k: Cutoff position

    Returns:
        NDCG@k score in [0, 1]. Returns 0 if no relevant documents exist.
    """
    if not relevant_ids:
        return 0.0

    # DCG: sum of relevance / log2(position + 1)
    dcg = 0.0
    for i, doc_id in enumerate(ranked_ids[:k]):
        if doc_id in relevant_ids:
            # Binary relevance: rel = 1 if relevant, 0 otherwise
            # Using gain = 1 (not 2^rel - 1 since binary)
            dcg += 1.0 / math.log2(
                i + 2
            )  # +2 because positions are 1-indexed in formula

    # IDCG: ideal DCG if all relevant docs were ranked first
    num_relevant_in_k = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(num_relevant_in_k))

    if idcg == 0:
        return 0.0

    return dcg / idcg


def mrr(
    ranked_ids: list[str],
    relevant_ids: set[str],
) -> float:
    """Compute Mean Reciprocal Rank for a single query.

    MRR measures how quickly the first relevant document appears.

    Args:
        ranked_ids: Ordered list of document IDs (most relevant first)
        relevant_ids: Set of relevant document IDs

    Returns:
        Reciprocal rank (1/position of first relevant doc).
        Returns 0 if no relevant document found.
    """
    for i, doc_id in enumerate(ranked_ids):
        if doc_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(
    ranked_ids: list[str],
    relevant_ids: set[str],
    k: int = 10,
) -> float:
    """Compute Recall at k.

    Recall@k measures what fraction of relevant documents appear in top-k.

    Args:
        ranked_ids: Ordered list of document IDs (most relevant first)
        relevant_ids: Set of relevant document IDs
        k: Cutoff position

    Returns:
        Recall@k score in [0, 1]. Returns 0 if no relevant documents exist.
    """
    if not relevant_ids:
        return 0.0

    retrieved_relevant = sum(1 for doc_id in ranked_ids[:k] if doc_id in relevant_ids)
    return retrieved_relevant / len(relevant_ids)


def precision_at_k(
    ranked_ids: list[str],
    relevant_ids: set[str],
    k: int = 10,
) -> float:
    """Compute Precision at k.

    Precision@k measures what fraction of top-k documents are relevant.

    Args:
        ranked_ids: Ordered list of document IDs (most relevant first)
        relevant_ids: Set of relevant document IDs
        k: Cutoff position

    Returns:
        Precision@k score in [0, 1].
    """
    if k == 0:
        return 0.0

    top_k = ranked_ids[:k]
    if not top_k:
        return 0.0

    retrieved_relevant = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return retrieved_relevant / len(top_k)


def average_precision(
    ranked_ids: list[str],
    relevant_ids: set[str],
    k: int | None = None,
) -> float:
    """Compute Average Precision for a single query.

    AP is the average of precision values at each relevant document position.

    Args:
        ranked_ids: Ordered list of document IDs (most relevant first)
        relevant_ids: Set of relevant document IDs
        k: Optional cutoff (None = use full ranking)

    Returns:
        AP score in [0, 1]. Returns 0 if no relevant documents exist.
    """
    if not relevant_ids:
        return 0.0

    if k is not None:
        ranked_ids = ranked_ids[:k]

    num_relevant_seen = 0
    precision_sum = 0.0

    for i, doc_id in enumerate(ranked_ids):
        if doc_id in relevant_ids:
            num_relevant_seen += 1
            precision_at_i = num_relevant_seen / (i + 1)
            precision_sum += precision_at_i

    if num_relevant_seen == 0:
        return 0.0

    # Divide by total relevant (not just those seen) per standard definition
    return precision_sum / len(relevant_ids)


def map_at_k(
    ranked_ids: list[str],
    relevant_ids: set[str],
    k: int = 10,
) -> float:
    """Compute Mean Average Precision at k for a single query.

    This is AP computed with a cutoff at k.

    Args:
        ranked_ids: Ordered list of document IDs (most relevant first)
        relevant_ids: Set of relevant document IDs
        k: Cutoff position

    Returns:
        MAP@k score in [0, 1].
    """
    return average_precision(ranked_ids, relevant_ids, k=k)


def compute_retrieval_metrics(
    results: dict[str, list[str]],
    relevance: dict[str, set[str]],
    k_values: list[int] | None = None,
    compute_per_query: bool = False,
) -> dict[str, Any]:
    """Compute all retrieval metrics aggregated over queries.

    Args:
        results: Mapping from query_id to ranked list of document IDs
        relevance: Mapping from query_id to set of relevant document IDs
        k_values: List of k values for @k metrics (default: [1, 5, 10, 50, 100])
        compute_per_query: Whether to include per-query metric breakdowns

    Returns:
        Dictionary with all metrics:
        {
            "ndcg@10": 0.72,
            "mrr": 0.81,
            "recall@10": 0.65,
            "precision@10": 0.32,
            "map@10": 0.68,
            "num_queries": 100,
            "per_query": {...}  # Only if compute_per_query=True
        }
    """
    if k_values is None:
        k_values = [1, 5, 10, 50, 100]

    # Validate inputs
    if not results:
        raise ValueError("results cannot be empty")

    query_ids = set(results.keys())
    relevance_ids = set(relevance.keys())

    missing_relevance = query_ids - relevance_ids
    if missing_relevance:
        raise ValueError(
            f"Missing relevance judgments for queries: {list(missing_relevance)[:5]}"
        )

    # Initialize accumulators
    ndcg_scores: dict[int, list[float]] = {k: [] for k in k_values}
    recall_scores: dict[int, list[float]] = {k: [] for k in k_values}
    precision_scores: dict[int, list[float]] = {k: [] for k in k_values}
    map_scores: dict[int, list[float]] = {k: [] for k in k_values}
    mrr_scores: list[float] = []

    per_query_metrics: dict[str, dict[str, float]] = {}

    for query_id, ranked_ids in results.items():
        relevant_ids = relevance.get(query_id, set())

        # Skip queries with no relevant documents (optional, but standard)
        if not relevant_ids:
            continue

        # Compute MRR
        query_mrr = mrr(ranked_ids, relevant_ids)
        mrr_scores.append(query_mrr)

        # Compute @k metrics
        query_metrics: dict[str, float] = {"mrr": query_mrr}

        for k in k_values:
            ndcg_k = ndcg_at_k(ranked_ids, relevant_ids, k)
            recall_k = recall_at_k(ranked_ids, relevant_ids, k)
            precision_k = precision_at_k(ranked_ids, relevant_ids, k)
            map_k = map_at_k(ranked_ids, relevant_ids, k)

            ndcg_scores[k].append(ndcg_k)
            recall_scores[k].append(recall_k)
            precision_scores[k].append(precision_k)
            map_scores[k].append(map_k)

            query_metrics[f"ndcg@{k}"] = ndcg_k
            query_metrics[f"recall@{k}"] = recall_k
            query_metrics[f"precision@{k}"] = precision_k
            query_metrics[f"map@{k}"] = map_k

        if compute_per_query:
            per_query_metrics[query_id] = query_metrics

    # Aggregate (mean across queries)
    metrics: dict[str, Any] = {
        "mrr": float(np.mean(mrr_scores)) if mrr_scores else 0.0,
        "num_queries": len(mrr_scores),
    }

    for k in k_values:
        if ndcg_scores[k]:
            metrics[f"ndcg@{k}"] = float(np.mean(ndcg_scores[k]))
            metrics[f"recall@{k}"] = float(np.mean(recall_scores[k]))
            metrics[f"precision@{k}"] = float(np.mean(precision_scores[k]))
            metrics[f"map@{k}"] = float(np.mean(map_scores[k]))
        else:
            metrics[f"ndcg@{k}"] = 0.0
            metrics[f"recall@{k}"] = 0.0
            metrics[f"precision@{k}"] = 0.0
            metrics[f"map@{k}"] = 0.0

    if compute_per_query:
        metrics["per_query"] = per_query_metrics

    return metrics

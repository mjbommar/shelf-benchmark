"""Retrieval metrics for SHELF evaluation.

All metrics are pure functions with no side effects.
Metrics follow standard IR definitions from BEIR/MTEB.

**Graded relevance.** SHELF's retrieval tasks originally judged a corpus
document relevant iff it carried the query's label. With 21 LCC classes over a
34k corpus that makes ~5% of the corpus relevant to *every* query, so NDCG@10
saturates and stops discriminating: a run that returns ten documents from a
neighbouring class scores the same as one that returns ten unrelated ones. The
LC taxonomies supply an ordinal relation ladder for free (see
``shelf.evaluate.strata`` and ``docs/data_plan_v0.4.md`` sections 11.1/11.4), so
this module accepts *graded* judgments as well as binary ones.

Anything that takes ``relevant_ids: set[str]`` also takes a
``Mapping[str, float]`` of per-document gains; a binary set behaves exactly as
before, because gain 1 for every relevant document is the binary special case.
:func:`compute_graded_retrieval_metrics` is the batched entry point and reports
``graded_ndcg@k`` alongside the binary metrics rather than in place of them.

Gains are used linearly by default (``gain = rel``). The exponential form
``gain = 2**rel - 1`` from Burges et al. is available via ``exponential=True``;
the two agree exactly on binary judgments and diverge only when a judgment
carries a partial-credit level, where the exponential form all but erases
partial credit.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _gain_map(relevance: Any) -> dict[str, float]:
    """Coerce a graded-relevance mapping to ``{doc_id: float}``."""
    return {str(doc_id): float(level) for doc_id, level in relevance.items()}


def gain_of(relevance_level: float, *, exponential: bool = False) -> float:
    """Convert a relevance level to a DCG gain.

    Args:
        relevance_level: Graded relevance (0 = not relevant, higher = closer).
        exponential: Use ``2**rel - 1`` (Burges et al.) instead of ``rel``.

    Returns:
        The gain contributed by a document at this relevance level. The two
        forms coincide on binary judgments, where ``rel`` is 0 or 1.
    """
    level = float(relevance_level)
    if not exponential:
        return level
    return float(2.0**level - 1.0)


def dcg_from_gains(
    ranked_gains: Sequence[float],
    k: int = 10,
    *,
    exponential: bool = False,
) -> float:
    """Discounted cumulative gain of an already-resolved gain sequence.

    Args:
        ranked_gains: Relevance level of each retrieved document, in rank order.
        k: Cutoff position.
        exponential: Use exponential gains.

    Returns:
        DCG@k. Unnormalized, so it is only comparable within a single query.
    """
    return sum(
        gain_of(level, exponential=exponential) / math.log2(i + 2)
        for i, level in enumerate(ranked_gains[:k])
    )


def ndcg_from_gains(
    ranked_gains: Sequence[float],
    ideal_gains: Sequence[float],
    k: int = 10,
    *,
    exponential: bool = False,
) -> float:
    """NDCG@k from a ranked gain sequence and the ideal gain sequence.

    This is the primitive the graded path is built on. ``ideal_gains`` must be
    the relevance levels of the best possible ranking (descending); it is sorted
    defensively so a caller cannot silently inflate the score by passing them in
    the wrong order.

    Args:
        ranked_gains: Relevance level of each retrieved document, in rank order.
        ideal_gains: Relevance levels available for this query, best first.
        k: Cutoff position.
        exponential: Use exponential gains.

    Returns:
        NDCG@k in [0, 1]. Returns 0.0 when no judged document has positive gain.
    """
    ideal_sorted = sorted((float(g) for g in ideal_gains), reverse=True)
    idcg = dcg_from_gains(ideal_sorted, k, exponential=exponential)
    if idcg <= 0.0:
        return 0.0

    dcg = dcg_from_gains(ranked_gains, k, exponential=exponential)
    return dcg / idcg


def ideal_gains_from_tiers(
    tiers: Sequence[tuple[float, int]],
    k: int,
) -> list[float]:
    """Expand ``(gain, count)`` tiers into the top-k ideal gain sequence.

    Materializing one entry per judged document is wasteful when a single tier
    holds thousands of documents and only the first ``k`` positions can score.
    Callers therefore report tier sizes and this expands just enough of them.

    Args:
        tiers: ``(relevance_level, number_of_documents)`` pairs, any order.
        k: Cutoff position.

    Returns:
        Up to ``k`` relevance levels in descending order.

    Raises:
        ValueError: If a tier reports a negative count.
    """
    if k <= 0:
        return []

    ideal: list[float] = []
    for level, count in sorted(tiers, key=lambda t: t[0], reverse=True):
        if count < 0:
            raise ValueError(f"tier count must be non-negative, got {count}")
        if level <= 0.0 or count == 0:
            continue
        take = min(count, k - len(ideal))
        ideal.extend([float(level)] * take)
        if len(ideal) >= k:
            break
    return ideal


def ndcg_at_k(
    ranked_ids: list[str],
    relevant_ids: set[str] | Mapping[str, float],
    k: int = 10,
    *,
    exponential: bool = False,
) -> float:
    """Compute Normalized Discounted Cumulative Gain at k.

    NDCG measures ranking quality by comparing the relevance of items
    at each position against an ideal ranking.

    Accepts binary *or* graded judgments. A ``set`` is the binary case and is
    scored exactly as before (every relevant document has gain 1). A mapping of
    ``doc_id -> relevance level`` is the graded case: documents absent from the
    mapping score 0, and the ideal ranking is built from the mapping's own
    levels, so a run that surfaces same-class documents outranks one that
    surfaces merely same-category documents.

    Args:
        ranked_ids: Ordered list of document IDs (most relevant first)
        relevant_ids: Set of relevant document IDs, or a mapping from document
            ID to graded relevance level.
        k: Cutoff position
        exponential: Use ``2**rel - 1`` gains. No effect on binary judgments.

    Returns:
        NDCG@k score in [0, 1]. Returns 0 if no relevant documents exist.
    """
    if not relevant_ids:
        return 0.0

    if isinstance(relevant_ids, Mapping):
        graded = _gain_map(relevant_ids)
        ranked_gains = [graded.get(doc_id, 0.0) for doc_id in ranked_ids]
        ideal_gains = [level for level in graded.values() if level > 0.0]
        return ndcg_from_gains(ranked_gains, ideal_gains, k, exponential=exponential)

    # Binary relevance: rel = 1 if relevant, 0 otherwise.
    ranked_gains = [1.0 if doc_id in relevant_ids else 0.0 for doc_id in ranked_ids]
    ideal_gains = [1.0] * min(len(relevant_ids), k)
    return ndcg_from_gains(ranked_gains, ideal_gains, k, exponential=exponential)


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


def compute_graded_retrieval_metrics(
    results: dict[str, list[str]],
    gains: Mapping[str, Mapping[str, float]],
    ideal_tiers: Mapping[str, Sequence[tuple[float, int]]],
    k_values: list[int] | None = None,
    *,
    exponential: bool = False,
    compute_per_query: bool = False,
) -> dict[str, Any]:
    """Compute graded NDCG aggregated over queries.

    This is the graded counterpart to :func:`compute_retrieval_metrics` and is
    reported *alongside* it, never instead of it: the binary numbers keep older
    runs comparable, while the graded numbers are the ones that discriminate.

    ``gains`` only has to cover documents the run actually retrieved. The size
    of each relevance tier comes from ``ideal_tiers`` instead, because a query
    whose top tier holds 1,600 corpus documents needs none of them enumerated to
    normalize an NDCG@10.

    Args:
        results: Mapping from query_id to ranked list of document IDs.
        gains: Mapping from query_id to ``{doc_id: relevance level}`` for the
            retrieved documents. Missing documents count as 0.
        ideal_tiers: Mapping from query_id to ``(relevance level, count)`` pairs
            describing the whole judged set for that query.
        k_values: List of k values for @k metrics (default: [1, 5, 10, 50, 100]).
        exponential: Use ``2**rel - 1`` gains.
        compute_per_query: Whether to include per-query breakdowns.

    Returns:
        Dictionary with ``graded_ndcg@k`` for each k, ``num_queries``, and
        ``per_query`` when requested.

    Raises:
        ValueError: If ``results`` is empty.
    """
    if k_values is None:
        k_values = [1, 5, 10, 50, 100]

    if not results:
        raise ValueError("results cannot be empty")

    max_k = max(k_values)
    scores: dict[int, list[float]] = {k: [] for k in k_values}
    per_query_metrics: dict[str, dict[str, float]] = {}

    for query_id, ranked_ids in results.items():
        tiers = ideal_tiers.get(query_id)
        if not tiers:
            # No judged document with positive gain: scoring it would average in
            # a 0.0 that reflects the judgments, not the run.
            continue

        ideal_full = ideal_gains_from_tiers(tiers, max_k)
        if not ideal_full:
            continue

        query_gains = gains.get(query_id, {})
        ranked_gains = [float(query_gains.get(doc_id, 0.0)) for doc_id in ranked_ids]

        query_metrics: dict[str, float] = {}
        for k in k_values:
            score = ndcg_from_gains(
                ranked_gains,
                ideal_full[:k],
                k,
                exponential=exponential,
            )
            scores[k].append(score)
            query_metrics[f"graded_ndcg@{k}"] = score

        if compute_per_query:
            per_query_metrics[query_id] = query_metrics

    metrics: dict[str, Any] = {
        "num_queries": len(scores[k_values[0]]) if k_values else 0,
    }
    for k in k_values:
        metrics[f"graded_ndcg@{k}"] = float(np.mean(scores[k])) if scores[k] else 0.0

    if compute_per_query:
        metrics["per_query"] = per_query_metrics

    return metrics

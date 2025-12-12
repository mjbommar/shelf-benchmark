"""Metric functions for SHELF evaluation.

All metrics are pure functions with no side effects.
"""

from shelf.evaluate.metrics.classification import (
    accuracy,
    compute_classification_metrics,
    confusion_matrix,
    macro_f1,
    micro_f1,
    per_class_f1,
    per_class_metrics,
    weighted_f1,
)
from shelf.evaluate.metrics.clustering import (
    adjusted_rand_index,
    completeness,
    compute_clustering_metrics,
    homogeneity,
    normalized_mutual_info,
    v_measure,
)
from shelf.evaluate.metrics.pair import (
    compute_pair_metrics,
    find_best_threshold,
    pair_accuracy,
    pair_average_precision,
    pair_auc_roc,
    pair_f1,
    pair_precision,
    pair_recall,
)
from shelf.evaluate.metrics.retrieval import (
    compute_retrieval_metrics,
    map_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

__all__ = [
    # Classification
    "accuracy",
    "macro_f1",
    "micro_f1",
    "weighted_f1",
    "per_class_f1",
    "per_class_metrics",
    "confusion_matrix",
    "compute_classification_metrics",
    # Clustering
    "v_measure",
    "normalized_mutual_info",
    "adjusted_rand_index",
    "homogeneity",
    "completeness",
    "compute_clustering_metrics",
    # Pair Classification
    "pair_f1",
    "pair_accuracy",
    "pair_precision",
    "pair_recall",
    "pair_auc_roc",
    "pair_average_precision",
    "find_best_threshold",
    "compute_pair_metrics",
    # Retrieval
    "ndcg_at_k",
    "mrr",
    "recall_at_k",
    "precision_at_k",
    "map_at_k",
    "compute_retrieval_metrics",
]

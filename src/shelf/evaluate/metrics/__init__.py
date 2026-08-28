"""Metric functions for SHELF evaluation.

All metrics are pure functions with no side effects.
"""

from shelf.evaluate.metrics.classification import (
    accuracy,
    compute_classification_metrics,
    compute_stratified_confusion_matrices,
    confusion_matrix,
    extract_top_confusions,
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
    compute_discovery_metrics,
    homogeneity,
    normalized_mutual_info,
    v_measure,
)
from shelf.evaluate.metrics.multilabel import (
    binarize_labels,
    compute_multilabel_metrics,
    hamming_loss,
    label_cardinality,
    label_coverage_error,
    label_ranking_average_precision,
    mean_average_precision,
    multilabel_f1,
    per_label_metrics,
    subset_accuracy,
)
from shelf.evaluate.metrics.pair import (
    compute_pair_metrics,
    find_best_threshold,
    pair_accuracy,
    pair_auc_roc,
    pair_average_precision,
    pair_f1,
    pair_precision,
    pair_recall,
)
from shelf.evaluate.metrics.retrieval import (
    compute_graded_retrieval_metrics,
    compute_retrieval_metrics,
    dcg_from_gains,
    gain_of,
    ideal_gains_from_tiers,
    map_at_k,
    mrr,
    ndcg_at_k,
    ndcg_from_gains,
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
    "compute_stratified_confusion_matrices",
    "extract_top_confusions",
    # Multi-label Classification
    "binarize_labels",
    "multilabel_f1",
    "subset_accuracy",
    "hamming_loss",
    "label_ranking_average_precision",
    "mean_average_precision",
    "label_coverage_error",
    "per_label_metrics",
    "label_cardinality",
    "compute_multilabel_metrics",
    # Clustering
    "v_measure",
    "normalized_mutual_info",
    "adjusted_rand_index",
    "homogeneity",
    "completeness",
    "compute_clustering_metrics",
    "compute_discovery_metrics",
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
    # Retrieval (graded relevance)
    "gain_of",
    "dcg_from_gains",
    "ndcg_from_gains",
    "ideal_gains_from_tiers",
    "compute_graded_retrieval_metrics",
]

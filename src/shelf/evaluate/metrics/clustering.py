"""Clustering metrics for SHELF evaluation.

All metrics are pure functions with no side effects.
Uses sklearn for core computations.
"""

from __future__ import annotations

from typing import Any

from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    v_measure_score,
)


def v_measure(
    labels_true: list[str] | list[int],
    labels_pred: list[int],
    beta: float = 1.0,
) -> float:
    """Compute V-measure (harmonic mean of homogeneity and completeness).

    V-measure is a clustering evaluation metric that balances two criteria:
    - Homogeneity: each cluster contains only members of a single class
    - Completeness: all members of a class are assigned to the same cluster

    Args:
        labels_true: Ground truth class labels
        labels_pred: Predicted cluster assignments
        beta: Weight of homogeneity vs completeness (1.0 = equal weight)

    Returns:
        V-measure score in [0, 1]. Higher is better.
    """
    return float(v_measure_score(labels_true, labels_pred, beta=beta))


def normalized_mutual_info(
    labels_true: list[str] | list[int],
    labels_pred: list[int],
    average_method: str = "arithmetic",
) -> float:
    """Compute Normalized Mutual Information (NMI).

    NMI measures the mutual information between the true and predicted
    labelings, normalized to be in [0, 1].

    Args:
        labels_true: Ground truth class labels
        labels_pred: Predicted cluster assignments
        average_method: How to normalize ('arithmetic', 'geometric', 'min', 'max')

    Returns:
        NMI score in [0, 1]. Higher is better.
    """
    return float(
        normalized_mutual_info_score(
            labels_true, labels_pred, average_method=average_method
        )
    )


def adjusted_rand_index(
    labels_true: list[str] | list[int],
    labels_pred: list[int],
) -> float:
    """Compute Adjusted Rand Index (ARI).

    ARI measures similarity between two clusterings, adjusted for chance.
    It considers all pairs of samples and counts pairs that are assigned
    in the same or different clusters in both labelings.

    Args:
        labels_true: Ground truth class labels
        labels_pred: Predicted cluster assignments

    Returns:
        ARI score in [-1, 1]. 1 = perfect, 0 = random, negative = worse than random.
    """
    return float(adjusted_rand_score(labels_true, labels_pred))


def homogeneity(
    labels_true: list[str] | list[int],
    labels_pred: list[int],
) -> float:
    """Compute homogeneity score.

    A clustering is homogeneous if all of its clusters contain only
    data points which are members of a single class.

    Args:
        labels_true: Ground truth class labels
        labels_pred: Predicted cluster assignments

    Returns:
        Homogeneity score in [0, 1]. Higher is better.
    """
    return float(homogeneity_score(labels_true, labels_pred))


def completeness(
    labels_true: list[str] | list[int],
    labels_pred: list[int],
) -> float:
    """Compute completeness score.

    A clustering is complete if all data points that are members of
    a given class are elements of the same cluster.

    Args:
        labels_true: Ground truth class labels
        labels_pred: Predicted cluster assignments

    Returns:
        Completeness score in [0, 1]. Higher is better.
    """
    return float(completeness_score(labels_true, labels_pred))


def compute_clustering_metrics(
    labels_true: list[str] | list[int],
    labels_pred: list[int],
) -> dict[str, Any]:
    """Compute all clustering metrics at once.

    Args:
        labels_true: Ground truth class labels
        labels_pred: Predicted cluster assignments

    Returns:
        Dictionary with all metrics:
        {
            "v_measure": 0.72,
            "nmi": 0.68,
            "ari": 0.55,
            "homogeneity": 0.75,
            "completeness": 0.70,
            "num_samples": 1000,
            "num_clusters_true": 21,
            "num_clusters_pred": 21,
        }
    """
    if not labels_true or not labels_pred:
        raise ValueError("labels_true and labels_pred cannot be empty")

    if len(labels_true) != len(labels_pred):
        raise ValueError(
            f"Length mismatch: labels_true has {len(labels_true)}, "
            f"labels_pred has {len(labels_pred)}"
        )

    # Compute all metrics
    metrics: dict[str, Any] = {
        "v_measure": v_measure(labels_true, labels_pred),
        "nmi": normalized_mutual_info(labels_true, labels_pred),
        "ari": adjusted_rand_index(labels_true, labels_pred),
        "homogeneity": homogeneity(labels_true, labels_pred),
        "completeness": completeness(labels_true, labels_pred),
        "num_samples": len(labels_true),
        "num_clusters_true": len(set(labels_true)),
        "num_clusters_pred": len(set(labels_pred)),
    }

    return metrics

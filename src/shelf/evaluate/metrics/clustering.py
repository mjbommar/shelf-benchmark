"""Clustering metrics for SHELF evaluation.

All metrics are pure functions with no side effects.
Uses sklearn for core computations.
"""

from __future__ import annotations

from typing import Any, Sequence, cast

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


def compute_discovery_metrics(
    labels_true: Sequence[int | str],
    labels_pred: Sequence[int],
    noise_label: int = -1,
) -> dict[str, Any]:
    """Compute metrics for cluster discovery evaluation.

    This extends standard clustering metrics with discovery-specific metrics:
    - noise_ratio: Fraction of points labeled as noise
    - num_clusters_pred: Number of clusters found (excluding noise)
    - num_clusters_true: Number of true clusters
    - cluster_k_error: Relative error in cluster count prediction

    Args:
        labels_true: Ground truth labels
        labels_pred: Predicted cluster assignments (may include -1 for noise)
        noise_label: Label used for noise points (default: -1)

    Returns:
        Dict with all clustering metrics plus discovery metrics
    """
    labels_true = list(labels_true)
    labels_pred = list(labels_pred)

    # Identify noise points
    noise_mask = [p == noise_label for p in labels_pred]
    num_noise = sum(noise_mask)
    noise_ratio = num_noise / len(labels_pred) if labels_pred else 0.0

    # Count clusters (excluding noise)
    unique_pred = set(labels_pred)
    if noise_label in unique_pred:
        unique_pred.remove(noise_label)
    num_clusters_pred = len(unique_pred)
    num_clusters_true = len(set(labels_true))

    # Compute k error
    if num_clusters_true > 0:
        cluster_k_error = abs(num_clusters_pred - num_clusters_true) / num_clusters_true
    else:
        cluster_k_error = float("inf") if num_clusters_pred > 0 else 0.0

    # Filter out noise points for standard metrics
    labels_true_filtered: list[int | str]
    labels_pred_filtered: list[int]

    if num_noise > 0 and num_noise < len(labels_pred):
        labels_true_filtered = [t for t, n in zip(labels_true, noise_mask) if not n]
        labels_pred_filtered = [p for p, n in zip(labels_pred, noise_mask) if not n]
    else:
        labels_true_filtered = labels_true
        labels_pred_filtered = labels_pred

    # Compute standard metrics on non-noise points
    if labels_true_filtered and labels_pred_filtered:
        # Cast to satisfy type checker - the lists will be homogeneous
        metrics = compute_clustering_metrics(
            cast("list[str] | list[int]", labels_true_filtered), labels_pred_filtered
        )
    else:
        # All points are noise - return zeros
        metrics = {
            "v_measure": 0.0,
            "nmi": 0.0,
            "ari": 0.0,
            "homogeneity": 0.0,
            "completeness": 0.0,
            "num_samples": len(labels_true),
            "num_clusters_true": num_clusters_true,
            "num_clusters_pred": 0,
        }

    # Add discovery metrics
    metrics["noise_count"] = num_noise
    metrics["noise_ratio"] = noise_ratio
    metrics["num_clusters_pred"] = num_clusters_pred
    metrics["cluster_k_error"] = cluster_k_error
    metrics["num_samples_clustered"] = len(labels_pred) - num_noise

    return metrics

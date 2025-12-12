"""Pair classification metrics for SHELF evaluation.

All metrics are pure functions with no side effects.
Uses sklearn for core computations.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def pair_accuracy(
    y_true: list[int],
    y_pred: list[int],
) -> float:
    """Compute accuracy for pair classification.

    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted labels (0 or 1)

    Returns:
        Accuracy in [0, 1]
    """
    return float(accuracy_score(y_true, y_pred))


def pair_f1(
    y_true: list[int],
    y_pred: list[int],
) -> float:
    """Compute F1 score for positive class (pairs that match).

    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted labels (0 or 1)

    Returns:
        F1 score in [0, 1]
    """
    return float(f1_score(y_true, y_pred, zero_division=0.0))


def pair_precision(
    y_true: list[int],
    y_pred: list[int],
) -> float:
    """Compute precision for positive class.

    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted labels (0 or 1)

    Returns:
        Precision in [0, 1]
    """
    return float(precision_score(y_true, y_pred, zero_division=0.0))


def pair_recall(
    y_true: list[int],
    y_pred: list[int],
) -> float:
    """Compute recall for positive class.

    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted labels (0 or 1)

    Returns:
        Recall in [0, 1]
    """
    return float(recall_score(y_true, y_pred, zero_division=0.0))


def pair_auc_roc(
    y_true: list[int],
    y_scores: list[float],
) -> float:
    """Compute Area Under ROC Curve.

    Args:
        y_true: Ground truth labels (0 or 1)
        y_scores: Predicted scores/probabilities

    Returns:
        AUC-ROC in [0, 1]
    """
    # Need both classes present
    if len(set(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_scores))


def pair_average_precision(
    y_true: list[int],
    y_scores: list[float],
) -> float:
    """Compute Average Precision (Area Under PR Curve).

    Args:
        y_true: Ground truth labels (0 or 1)
        y_scores: Predicted scores/probabilities

    Returns:
        Average precision in [0, 1]
    """
    # Need positive class present
    if sum(y_true) == 0:
        return 0.0
    return float(average_precision_score(y_true, y_scores))


def find_best_threshold(
    y_true: list[int],
    y_scores: list[float],
    metric: str = "f1",
) -> tuple[float, float]:
    """Find the threshold that maximizes a given metric.

    Args:
        y_true: Ground truth labels (0 or 1)
        y_scores: Predicted scores/probabilities
        metric: Metric to optimize ("f1", "accuracy")

    Returns:
        Tuple of (best_threshold, best_score)
    """
    y_true_arr = np.array(y_true)
    y_scores_arr = np.array(y_scores)

    # Try thresholds from min to max score
    thresholds = np.unique(y_scores_arr)
    if len(thresholds) > 100:
        # Sample thresholds for efficiency
        thresholds = np.percentile(y_scores_arr, np.linspace(0, 100, 101))

    best_threshold = 0.5
    best_score = 0.0

    for thresh in thresholds:
        y_pred = (y_scores_arr >= thresh).astype(int)

        if metric == "f1":
            score = f1_score(y_true_arr, y_pred, zero_division=0.0)
        elif metric == "accuracy":
            score = accuracy_score(y_true_arr, y_pred)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if score > best_score:
            best_score = score
            best_threshold = thresh

    return float(best_threshold), float(best_score)


def compute_pair_metrics(
    y_true: list[int],
    y_scores: list[float],
    threshold: float | None = None,
) -> dict[str, Any]:
    """Compute all pair classification metrics at once.

    Args:
        y_true: Ground truth labels (0 or 1)
        y_scores: Predicted similarity scores
        threshold: Classification threshold. If None, finds optimal threshold.

    Returns:
        Dictionary with all metrics:
        {
            "f1": 0.72,
            "accuracy": 0.85,
            "precision": 0.80,
            "recall": 0.65,
            "auc_roc": 0.88,
            "average_precision": 0.82,
            "threshold": 0.65,
            "num_samples": 1000,
            "num_positive": 500,
            "num_negative": 500,
        }
    """
    if not y_true or not y_scores:
        raise ValueError("y_true and y_scores cannot be empty")

    if len(y_true) != len(y_scores):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)}, y_scores has {len(y_scores)}"
        )

    # Find best threshold if not provided
    if threshold is None:
        threshold, _ = find_best_threshold(y_true, y_scores, metric="f1")

    # Convert scores to predictions using threshold
    y_pred = [1 if s >= threshold else 0 for s in y_scores]

    # Compute all metrics
    metrics: dict[str, Any] = {
        "f1": pair_f1(y_true, y_pred),
        "accuracy": pair_accuracy(y_true, y_pred),
        "precision": pair_precision(y_true, y_pred),
        "recall": pair_recall(y_true, y_pred),
        "auc_roc": pair_auc_roc(y_true, y_scores),
        "average_precision": pair_average_precision(y_true, y_scores),
        "threshold": threshold,
        "num_samples": len(y_true),
        "num_positive": sum(y_true),
        "num_negative": len(y_true) - sum(y_true),
    }

    return metrics

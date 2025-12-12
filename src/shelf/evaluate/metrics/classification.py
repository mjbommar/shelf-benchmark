"""Classification metrics for SHELF evaluation.

All metrics are pure functions with no side effects.
Uses sklearn for core computations with explicit zero_division handling.
"""

from __future__ import annotations

from typing import Any

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix as sklearn_confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def macro_f1(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str] | None = None,
) -> float:
    """Compute macro-averaged F1 score.

    Macro F1 computes F1 for each class and averages them equally,
    regardless of class frequency. Good for imbalanced datasets.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: Label set (optional, inferred if not provided)

    Returns:
        Macro F1 score in [0, 1]
    """
    return float(
        f1_score(
            y_true,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0.0,
        )
    )


def micro_f1(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str] | None = None,
) -> float:
    """Compute micro-averaged F1 score.

    Micro F1 aggregates TP, FP, FN across all classes before computing F1.
    Equivalent to accuracy for single-label classification.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: Label set (optional)

    Returns:
        Micro F1 score in [0, 1]
    """
    return float(
        f1_score(
            y_true,
            y_pred,
            labels=labels,
            average="micro",
            zero_division=0.0,
        )
    )


def weighted_f1(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str] | None = None,
) -> float:
    """Compute weighted F1 score.

    Weighted F1 computes F1 for each class and averages weighted by
    class support (number of samples per class).

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: Label set (optional)

    Returns:
        Weighted F1 score in [0, 1]
    """
    return float(
        f1_score(
            y_true,
            y_pred,
            labels=labels,
            average="weighted",
            zero_division=0.0,
        )
    )


def accuracy(
    y_true: list[str],
    y_pred: list[str],
) -> float:
    """Compute simple accuracy.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Accuracy in [0, 1]
    """
    return float(accuracy_score(y_true, y_pred))


def per_class_f1(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
) -> dict[str, float]:
    """Compute F1 score for each class.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: Ordered list of class labels

    Returns:
        Dictionary mapping label to F1 score
    """
    scores = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0.0,
    )
    return {label: float(score) for label, score in zip(labels, scores)}


def per_class_metrics(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
) -> dict[str, dict[str, float]]:
    """Compute precision, recall, F1, and support for each class.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: Ordered list of class labels

    Returns:
        Dictionary mapping label to {precision, recall, f1, support}
    """
    precision_scores = precision_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0.0,
    )
    recall_scores = recall_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0.0,
    )
    f1_scores = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0.0,
    )

    # Compute support (count of each class in y_true)
    label_counts = {}
    for label in labels:
        label_counts[label] = 0
    for label in y_true:
        if label in label_counts:
            label_counts[label] += 1

    result = {}
    for i, label in enumerate(labels):
        result[label] = {
            "precision": float(precision_scores[i]),
            "recall": float(recall_scores[i]),
            "f1": float(f1_scores[i]),
            "support": label_counts[label],
        }

    return result


def confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
) -> list[list[int]]:
    """Compute confusion matrix.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: Ordered list of class labels

    Returns:
        Confusion matrix as nested list (row=true, col=pred)
    """
    cm = sklearn_confusion_matrix(y_true, y_pred, labels=labels)
    return cm.tolist()


def classification_summary(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str] | None = None,
) -> str:
    """Generate sklearn classification report as string.

    Useful for human-readable output.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: Label set (optional)

    Returns:
        Formatted classification report string
    """
    return classification_report(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0.0,
    )


def compute_classification_metrics(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str] | None = None,
    compute_per_class: bool = True,
    compute_confusion_matrix: bool = True,
) -> dict[str, Any]:
    """Compute all classification metrics at once.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: Ordered list of class labels. If None, inferred from data.
        compute_per_class: Whether to include per-class breakdown
        compute_confusion_matrix: Whether to include confusion matrix

    Returns:
        Dictionary with all metrics:
        {
            "accuracy": 0.85,
            "macro_f1": 0.72,
            "micro_f1": 0.85,
            "weighted_f1": 0.83,
            "num_samples": 1000,
            "num_correct": 850,
            "num_classes": 21,
            "per_class": {...},  # if compute_per_class
            "confusion_matrix": [[...], ...],  # if compute_confusion_matrix
        }
    """
    if not y_true or not y_pred:
        raise ValueError("y_true and y_pred cannot be empty")

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)}, y_pred has {len(y_pred)}"
        )

    # Infer labels if not provided
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    # Core metrics
    metrics: dict[str, Any] = {
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred, labels),
        "micro_f1": micro_f1(y_true, y_pred, labels),
        "weighted_f1": weighted_f1(y_true, y_pred, labels),
        "num_samples": len(y_true),
        "num_correct": sum(1 for t, p in zip(y_true, y_pred) if t == p),
        "num_classes": len(labels),
    }

    # Per-class breakdown
    if compute_per_class:
        metrics["per_class"] = per_class_metrics(y_true, y_pred, labels)

    # Confusion matrix
    if compute_confusion_matrix:
        metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels)
        metrics["labels"] = labels  # Include label order for interpreting matrix

    return metrics

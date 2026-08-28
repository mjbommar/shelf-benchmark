"""Multi-label classification metrics for SHELF evaluation.

Used by the ``topic_classification`` task, where each document carries 1-4
LCSH topical terms drawn from a 112-term vocabulary.

All metrics are pure functions with no side effects. Uses sklearn for core
computations with explicit ``zero_division=0.0`` handling (see CLAUDE.md).

Metric selection
----------------
Single-label accuracy is meaningless here, so the reported set covers three
distinct failure modes:

1. **Averaging choice** - ``micro_f1`` pools TP/FP/FN over all (document, label)
   cells and is dominated by frequent topics; ``macro_f1`` averages per-label F1
   and therefore exposes rare-topic failure; ``samples_f1`` averages per-document
   F1 and answers "how good is a typical prediction set"; ``weighted_f1`` is
   macro weighted by support. Reporting all four prevents cherry-picking (see
   the "multiple metrics always" principle in CLAUDE.md).
2. **Set-level correctness** - ``subset_accuracy`` (exact match ratio) is the
   strictest possible score, and ``hamming_loss`` is the most forgiving,
   measuring the per-cell error rate. Together they bracket the F1 family.
3. **Threshold-free ranking** - ``lrap`` (label ranking average precision),
   ``map_micro`` / ``map_macro`` (mean average precision), and
   ``coverage_error`` score the *ranking* the model induces over labels and are
   therefore independent of the decision threshold. A model can look bad on F1
   purely because of a badly calibrated threshold; these separate the two.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    coverage_error,
    f1_score,
    label_ranking_average_precision_score,
    precision_score,
    recall_score,
)
from sklearn.metrics import (
    hamming_loss as sklearn_hamming_loss,
)

# Averaging strategies reported for every multi-label evaluation.
F1_AVERAGES = ("micro", "macro", "samples", "weighted")


def binarize_labels(
    label_sets: list[list[str]],
    labels: list[str],
) -> np.ndarray:
    """Convert lists of label names into a binary indicator matrix.

    Unlike ``sklearn.preprocessing.MultiLabelBinarizer`` this uses a caller
    supplied, fixed label ordering so that ground truth and predictions are
    always aligned on the same columns, and silently ignores labels outside
    the vocabulary (callers validate the label space separately).

    Args:
        label_sets: One list of label names per sample. May be empty.
        labels: Ordered label vocabulary (defines the column order)

    Returns:
        Binary indicator array of shape (n_samples, n_labels)

    Example:
        >>> binarize_labels([["a"], ["a", "b"], []], ["a", "b"]).tolist()
        [[1, 0], [1, 1], [0, 0]]
    """
    index = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(label_sets), len(labels)), dtype=np.int8)
    for row, label_set in enumerate(label_sets):
        for label in label_set:
            col = index.get(label)
            if col is not None:
                matrix[row, col] = 1
    return matrix


def _validate_shapes(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Raise if the two indicator matrices are not aligned."""
    if y_true.size == 0 or y_pred.size == 0:
        raise ValueError("y_true and y_pred cannot be empty")
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true has {y_true.shape}, y_pred has {y_pred.shape}"
        )


def multilabel_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "micro",
) -> float:
    """Compute an averaged F1 score for multi-label predictions.

    Args:
        y_true: Binary indicator matrix of ground truth labels
        y_pred: Binary indicator matrix of predicted labels
        average: One of "micro", "macro", "samples", "weighted"

    Returns:
        F1 score in [0, 1]
    """
    return float(
        f1_score(
            y_true,
            y_pred,
            average=average,
            zero_division=0.0,
        )
    )


def subset_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute exact match ratio (fraction of rows predicted perfectly).

    This is the strictest multi-label metric: a single wrong or missing label
    makes the whole row wrong.

    Args:
        y_true: Binary indicator matrix of ground truth labels
        y_pred: Binary indicator matrix of predicted labels

    Returns:
        Subset accuracy in [0, 1]
    """
    return float(accuracy_score(y_true, y_pred))


def hamming_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Hamming loss (fraction of incorrect (sample, label) cells).

    Lower is better. Note this is dominated by the majority-negative cells when
    the label vocabulary is large, so it is reported alongside F1 rather than
    instead of it.

    Args:
        y_true: Binary indicator matrix of ground truth labels
        y_pred: Binary indicator matrix of predicted labels

    Returns:
        Hamming loss in [0, 1]
    """
    return float(sklearn_hamming_loss(y_true, y_pred))


def label_ranking_average_precision(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> float:
    """Compute LRAP: threshold-free quality of the per-document label ranking.

    For each document, averages over its true labels the fraction of
    higher-ranked labels that are also true. 1.0 means every true label was
    ranked above every false one.

    Rows with no true labels are undefined for LRAP and are dropped; if no row
    has a true label, returns 0.0.

    Args:
        y_true: Binary indicator matrix of ground truth labels
        y_score: Continuous score matrix (higher = more likely)

    Returns:
        LRAP in [0, 1]
    """
    mask = np.asarray(y_true).sum(axis=1) > 0
    if not mask.any():
        return 0.0
    return float(
        label_ranking_average_precision_score(
            np.asarray(y_true)[mask],
            np.asarray(y_score)[mask],
        )
    )


def mean_average_precision(
    y_true: np.ndarray,
    y_score: np.ndarray,
    average: str = "micro",
) -> float:
    """Compute mean average precision over label scores.

    Args:
        y_true: Binary indicator matrix of ground truth labels
        y_score: Continuous score matrix (higher = more likely)
        average: "micro" (pool all cells) or "macro" (average per-label AP)

    Returns:
        mAP in [0, 1]. Returns 0.0 when no positive cell exists.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.sum() == 0:
        return 0.0

    if average == "macro":
        # A label with no positives has undefined AP; sklearn emits nan. Drop
        # those columns rather than letting nan poison the average.
        keep = y_true.sum(axis=0) > 0
        if not keep.any():
            return 0.0
        y_true = y_true[:, keep]
        y_score = y_score[:, keep]

    score = average_precision_score(y_true, y_score, average=average)
    value = float(np.nanmean(np.asarray(score, dtype=float)))
    return 0.0 if np.isnan(value) else value


def label_coverage_error(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute coverage error: mean rank depth needed to cover all true labels.

    The best achievable value equals the mean number of true labels per
    document (2.49 for SHELF topics), not 0.

    Rows with no true labels are dropped. Returns 0.0 if none remain.

    Args:
        y_true: Binary indicator matrix of ground truth labels
        y_score: Continuous score matrix (higher = more likely)

    Returns:
        Coverage error (>= 1.0 when defined)
    """
    mask = np.asarray(y_true).sum(axis=1) > 0
    if not mask.any():
        return 0.0
    return float(coverage_error(np.asarray(y_true)[mask], np.asarray(y_score)[mask]))


def per_label_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
) -> dict[str, dict[str, float]]:
    """Compute precision, recall, F1 and support for each label.

    Args:
        y_true: Binary indicator matrix of ground truth labels
        y_pred: Binary indicator matrix of predicted labels
        labels: Ordered label vocabulary matching the matrix columns

    Returns:
        Dictionary mapping label to {precision, recall, f1, support, num_predicted}
    """
    if y_true.shape[1] != len(labels):
        raise ValueError(
            f"Labels length ({len(labels)}) must match matrix columns "
            f"({y_true.shape[1]})"
        )

    precision_scores = precision_score(y_true, y_pred, average=None, zero_division=0.0)
    recall_scores = recall_score(y_true, y_pred, average=None, zero_division=0.0)
    f1_scores = f1_score(y_true, y_pred, average=None, zero_division=0.0)
    support = np.asarray(y_true).sum(axis=0)
    predicted = np.asarray(y_pred).sum(axis=0)

    return {
        label: {
            "precision": float(precision_scores[i]),
            "recall": float(recall_scores[i]),
            "f1": float(f1_scores[i]),
            "support": int(support[i]),
            "num_predicted": int(predicted[i]),
        }
        for i, label in enumerate(labels)
    }


def label_cardinality(y: np.ndarray) -> float:
    """Compute mean number of positive labels per sample."""
    y = np.asarray(y)
    if y.shape[0] == 0:
        return 0.0
    return float(y.sum(axis=1).mean())


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    y_score: np.ndarray | None = None,
    compute_per_label: bool = True,
) -> dict[str, Any]:
    """Compute the full multi-label metric set at once.

    Args:
        y_true: Binary indicator matrix of ground truth labels
        y_pred: Binary indicator matrix of predicted labels
        labels: Ordered label vocabulary matching the matrix columns
        y_score: Optional continuous score matrix enabling the threshold-free
            ranking metrics (lrap, map_micro, map_macro, coverage_error)
        compute_per_label: Whether to include the per-label breakdown

    Returns:
        Dictionary with all metrics:
        {
            "micro_f1": 0.52,
            "macro_f1": 0.41,
            "samples_f1": 0.50,
            "weighted_f1": 0.51,
            "subset_accuracy": 0.11,
            "hamming_loss": 0.02,
            "num_samples": 8507,
            "num_exact_match": 936,
            "num_labels": 112,
            "label_cardinality_true": 2.49,
            "label_cardinality_pred": 3.10,
            "empty_prediction_rate": 0.01,
            "lrap": 0.70,            # if y_score given
            "map_micro": 0.60,       # if y_score given
            "map_macro": 0.55,       # if y_score given
            "coverage_error": 12.4,  # if y_score given
            "per_label": {...},      # if compute_per_label
            "labels": [...],         # if compute_per_label
        }
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    _validate_shapes(y_true, y_pred)

    if y_true.shape[1] != len(labels):
        raise ValueError(
            f"Labels length ({len(labels)}) must match matrix columns "
            f"({y_true.shape[1]})"
        )

    num_exact_match = int((y_true == y_pred).all(axis=1).sum())

    metrics: dict[str, Any] = {
        "micro_f1": multilabel_f1(y_true, y_pred, average="micro"),
        "macro_f1": multilabel_f1(y_true, y_pred, average="macro"),
        "samples_f1": multilabel_f1(y_true, y_pred, average="samples"),
        "weighted_f1": multilabel_f1(y_true, y_pred, average="weighted"),
        "subset_accuracy": subset_accuracy(y_true, y_pred),
        "hamming_loss": hamming_loss(y_true, y_pred),
        "num_samples": int(y_true.shape[0]),
        "num_exact_match": num_exact_match,
        "num_labels": len(labels),
        "label_cardinality_true": label_cardinality(y_true),
        "label_cardinality_pred": label_cardinality(y_pred),
        "empty_prediction_rate": float((y_pred.sum(axis=1) == 0).mean()),
    }

    if y_score is not None:
        y_score = np.asarray(y_score, dtype=float)
        if y_score.shape != y_true.shape:
            raise ValueError(
                f"Shape mismatch: y_true has {y_true.shape}, "
                f"y_score has {y_score.shape}"
            )
        metrics["lrap"] = label_ranking_average_precision(y_true, y_score)
        metrics["map_micro"] = mean_average_precision(y_true, y_score, "micro")
        metrics["map_macro"] = mean_average_precision(y_true, y_score, "macro")
        metrics["coverage_error"] = label_coverage_error(y_true, y_score)

    if compute_per_label:
        metrics["per_label"] = per_label_metrics(y_true, y_pred, labels)
        metrics["labels"] = list(labels)

    return metrics

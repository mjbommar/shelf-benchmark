"""Unit tests for shelf.evaluate.metrics.classification module.

Tests cover:
- Individual metric functions (macro_f1, micro_f1, etc.)
- Per-class metrics computation
- Confusion matrix generation
- compute_classification_metrics aggregation
- Edge cases: zero division, single class, empty predictions
"""

from __future__ import annotations

import pytest

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


class TestAccuracy:
    """Tests for accuracy metric."""

    def test_perfect_accuracy(self, perfect_classification):
        """Test 100% accuracy."""
        y_true, y_pred = perfect_classification
        assert accuracy(y_true, y_pred) == pytest.approx(1.0)

    def test_zero_accuracy(self):
        """Test 0% accuracy."""
        y_true = ["A", "B", "C", "D"]
        y_pred = ["B", "C", "D", "A"]
        assert accuracy(y_true, y_pred) == pytest.approx(0.0)

    def test_partial_accuracy(self, partial_classification):
        """Test partial accuracy."""
        y_true, y_pred = partial_classification
        acc = accuracy(y_true, y_pred)
        assert 0 < acc < 1
        # 4 out of 10 correct = 0.4
        assert acc == pytest.approx(0.4)


class TestF1Scores:
    """Tests for F1 score variants."""

    def test_perfect_f1(self, perfect_classification):
        """Test perfect F1 scores."""
        y_true, y_pred = perfect_classification
        assert macro_f1(y_true, y_pred) == pytest.approx(1.0)
        assert micro_f1(y_true, y_pred) == pytest.approx(1.0)
        assert weighted_f1(y_true, y_pred) == pytest.approx(1.0)

    def test_macro_vs_micro_imbalanced(self, imbalanced_classification):
        """Test difference between macro and micro F1 on imbalanced data."""
        y_true, y_pred = imbalanced_classification
        macro = macro_f1(y_true, y_pred)
        micro = micro_f1(y_true, y_pred)

        # Micro tends to favor majority class
        # Both should be between 0 and 1
        assert 0 < macro <= 1
        assert 0 < micro <= 1

    def test_zero_f1_for_missing_class(self):
        """Test F1 when a class is never predicted."""
        y_true = ["A", "A", "B", "B", "C", "C"]
        y_pred = ["A", "A", "A", "A", "A", "A"]  # Never predicts B or C

        # Should handle zero division gracefully
        f1 = macro_f1(y_true, y_pred)
        assert 0 <= f1 <= 1

    def test_f1_with_explicit_labels(self):
        """Test F1 scores with explicit label list."""
        y_true = ["A", "A", "B", "B"]
        y_pred = ["A", "A", "B", "B"]
        labels = ["A", "B", "C"]  # Include C even though not in data

        macro = macro_f1(y_true, y_pred, labels=labels)
        # C has F1=0 (zero_division=0), so macro = (1.0 + 1.0 + 0.0) / 3 = 0.667
        assert macro == pytest.approx(2.0 / 3.0)


class TestPerClassMetrics:
    """Tests for per-class metric computation."""

    def test_per_class_f1(self, perfect_classification):
        """Test per-class F1 computation."""
        y_true, y_pred = perfect_classification
        labels = ["A", "B", "C", "D"]

        result = per_class_f1(y_true, y_pred, labels)

        assert isinstance(result, dict)
        assert set(result.keys()) == {"A", "B", "C", "D"}
        # Perfect classification - all F1 scores should be 1.0
        for label in labels:
            assert result[label] == pytest.approx(1.0)

    def test_per_class_metrics_full(self, partial_classification):
        """Test full per-class metrics (precision, recall, F1, support)."""
        y_true, y_pred = partial_classification
        labels = ["A", "B", "C", "D"]

        result = per_class_metrics(y_true, y_pred, labels)

        assert isinstance(result, dict)
        for label in labels:
            assert label in result
            assert "precision" in result[label]
            assert "recall" in result[label]
            assert "f1" in result[label]
            assert "support" in result[label]
        # Support should be count in y_true (A:3, B:3, C:2, D:2 in fixture)
        assert result["A"]["support"] == 3
        assert result["B"]["support"] == 3
        assert result["C"]["support"] == 2
        assert result["D"]["support"] == 2

    def test_per_class_metrics_imbalanced(self, imbalanced_classification):
        """Test per-class metrics on imbalanced data."""
        y_true, y_pred = imbalanced_classification
        labels = ["A", "B"]

        result = per_class_metrics(y_true, y_pred, labels)

        # Check supports match y_true counts (fixture: A:90, B:10)
        assert result["A"]["support"] == 90
        assert result["B"]["support"] == 10


class TestConfusionMatrix:
    """Tests for confusion matrix generation."""

    def test_confusion_matrix_perfect(self, perfect_classification):
        """Test confusion matrix for perfect predictions."""
        y_true, y_pred = perfect_classification
        labels = ["A", "B", "C", "D"]

        cm = confusion_matrix(y_true, y_pred, labels)

        # Should be identity matrix (diagonal only)
        assert isinstance(cm, list)
        assert len(cm) == 4
        for i, row in enumerate(cm):
            assert len(row) == 4
            for j, count in enumerate(row):
                if i != j:
                    assert count == 0  # Off-diagonal should be 0
        # Fixture has A:3, B:3, C:3, D:1
        assert cm[0][0] == 3  # A
        assert cm[1][1] == 3  # B
        assert cm[2][2] == 3  # C
        assert cm[3][3] == 1  # D

    def test_confusion_matrix_structure(self, partial_classification):
        """Test confusion matrix structure."""
        y_true, y_pred = partial_classification
        labels = ["A", "B", "C", "D"]

        cm = confusion_matrix(y_true, y_pred, labels)

        # Check dimensions
        assert len(cm) == 4
        for row in cm:
            assert len(row) == 4

        # Sum of all elements should equal total samples
        total = sum(sum(row) for row in cm)
        assert total == len(y_true)


class TestComputeClassificationMetrics:
    """Tests for the aggregated compute_classification_metrics function."""

    def test_all_metrics_returned(self, perfect_classification):
        """Test that all expected metrics are returned."""
        y_true, y_pred = perfect_classification
        result = compute_classification_metrics(y_true, y_pred)

        assert "macro_f1" in result
        assert "micro_f1" in result
        assert "weighted_f1" in result
        assert "accuracy" in result
        assert "num_samples" in result
        assert "num_correct" in result
        assert "num_classes" in result

    def test_per_class_metrics(self, perfect_classification):
        """Test per-class metrics computation."""
        y_true, y_pred = perfect_classification
        result = compute_classification_metrics(
            y_true,
            y_pred,
            compute_per_class=True,
        )

        assert "per_class" in result
        per_class = result["per_class"]

        # Should have metrics for each class
        for cls in set(y_true):
            assert cls in per_class
            assert "f1" in per_class[cls]
            assert "precision" in per_class[cls]
            assert "recall" in per_class[cls]

    def test_confusion_matrix(self, perfect_classification):
        """Test confusion matrix generation."""
        y_true, y_pred = perfect_classification
        result = compute_classification_metrics(
            y_true,
            y_pred,
            compute_confusion_matrix=True,
        )

        assert "confusion_matrix" in result
        cm = result["confusion_matrix"]

        # For perfect predictions, diagonal should equal class counts
        assert isinstance(cm, list)
        assert "labels" in result  # Labels should be included for interpretation

    def test_with_labels(self, perfect_classification):
        """Test providing explicit label list."""
        y_true, y_pred = perfect_classification
        labels = ["A", "B", "C", "D", "E"]  # Include extra class E

        result = compute_classification_metrics(
            y_true,
            y_pred,
            labels=labels,
            compute_per_class=True,
        )

        # Should include all specified labels
        assert result["num_classes"] == 5
        assert "labels" in result or "per_class" in result

    def test_num_samples_and_correct(self, partial_classification):
        """Test sample counts."""
        y_true, y_pred = partial_classification
        result = compute_classification_metrics(y_true, y_pred)

        assert result["num_samples"] == len(y_true)
        assert result["num_correct"] == sum(1 for t, p in zip(y_true, y_pred) if t == p)
        # For partial_classification: 10 samples, 4 correct (0.4 accuracy)
        assert result["num_correct"] == 4
        assert result["num_samples"] == 10

    def test_disable_optional_computations(self, perfect_classification):
        """Test disabling per-class and confusion matrix."""
        y_true, y_pred = perfect_classification
        result = compute_classification_metrics(
            y_true,
            y_pred,
            compute_per_class=False,
            compute_confusion_matrix=False,
        )

        assert "per_class" not in result
        assert "confusion_matrix" not in result
        assert "labels" not in result


class TestClassificationEdgeCases:
    """Edge case tests for classification metrics."""

    def test_single_class(self):
        """Test with only one class in data."""
        import warnings

        y_true = ["A", "A", "A", "A"]
        y_pred = ["A", "A", "A", "A"]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = compute_classification_metrics(y_true, y_pred)
        assert result["accuracy"] == pytest.approx(1.0)
        assert result["num_classes"] == 1

    def test_binary_classification(self):
        """Test binary classification."""
        y_true = ["positive", "negative", "positive", "negative"]
        y_pred = ["positive", "negative", "negative", "negative"]

        result = compute_classification_metrics(y_true, y_pred)
        assert "macro_f1" in result
        assert result["num_classes"] == 2

    def test_numeric_labels(self):
        """Test with numeric labels."""
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 0, 1, 1]

        result = compute_classification_metrics(y_true, y_pred)
        assert 0 < result["accuracy"] < 1
        # 5 out of 6 correct
        assert result["accuracy"] == pytest.approx(5.0 / 6.0)

    def test_empty_inputs_raises(self):
        """Test that empty inputs raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            compute_classification_metrics([], [])

    def test_mismatched_lengths_raises(self):
        """Test that mismatched lengths raise ValueError."""
        y_true = ["A", "B", "C"]
        y_pred = ["A", "B"]

        with pytest.raises(ValueError, match="Length mismatch"):
            compute_classification_metrics(y_true, y_pred)

    def test_predicted_class_not_in_true(self):
        """Test when predicted class doesn't appear in true labels."""
        y_true = ["A", "A", "B", "B"]
        y_pred = ["A", "C", "B", "C"]  # C is predicted but not in y_true

        result = compute_classification_metrics(y_true, y_pred)
        # Should handle gracefully - labels inferred from union
        assert result["num_classes"] == 3

    def test_all_wrong_predictions(self):
        """Test when all predictions are wrong."""
        y_true = ["A", "A", "A", "A"]
        y_pred = ["B", "B", "B", "B"]

        result = compute_classification_metrics(y_true, y_pred)
        assert result["accuracy"] == pytest.approx(0.0)
        assert result["num_correct"] == 0

    def test_micro_equals_accuracy(self, partial_classification):
        """Test that micro F1 equals accuracy for single-label classification."""
        y_true, y_pred = partial_classification

        result = compute_classification_metrics(y_true, y_pred)
        # For single-label classification, micro F1 should equal accuracy
        assert result["micro_f1"] == pytest.approx(result["accuracy"])

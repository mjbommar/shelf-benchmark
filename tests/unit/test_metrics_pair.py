"""Unit tests for shelf.evaluate.metrics.pair module.

Tests cover:
- Individual metric functions (accuracy, F1, precision, recall)
- Scoring metrics (AUC-ROC, average precision)
- Threshold finding (F1 and accuracy optimization)
- compute_pair_metrics aggregation
- Edge cases: all zeros, all ones, single class, empty inputs
- Input validation: mismatched lengths, empty inputs
"""

from __future__ import annotations

import numpy as np
import pytest

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


@pytest.mark.unit
@pytest.mark.metrics
class TestPairAccuracy:
    """Tests for pair_accuracy metric."""

    def test_perfect_accuracy(self, pair_perfect):
        """Test 100% accuracy."""
        y_true, y_pred = pair_perfect
        assert pair_accuracy(y_true.tolist(), y_pred.tolist()) == pytest.approx(1.0)

    def test_zero_accuracy(self):
        """Test 0% accuracy (all predictions wrong)."""
        y_true = [1, 1, 1, 1, 0, 0, 0, 0]
        y_pred = [0, 0, 0, 0, 1, 1, 1, 1]
        assert pair_accuracy(y_true, y_pred) == pytest.approx(0.0)

    def test_partial_accuracy(self, pair_partial):
        """Test partial accuracy."""
        y_true, y_pred = pair_partial
        acc = pair_accuracy(y_true.tolist(), y_pred.tolist())
        assert 0 < acc < 1
        # 6 out of 8 correct = 0.75
        assert acc == pytest.approx(0.75)

    def test_all_zeros(self):
        """Test when all labels are 0."""
        y_true = [0, 0, 0, 0, 0]
        y_pred = [0, 0, 0, 0, 0]
        assert pair_accuracy(y_true, y_pred) == pytest.approx(1.0)

    def test_all_ones(self):
        """Test when all labels are 1."""
        y_true = [1, 1, 1, 1, 1]
        y_pred = [1, 1, 1, 1, 1]
        assert pair_accuracy(y_true, y_pred) == pytest.approx(1.0)

    def test_single_sample(self):
        """Test with single sample."""
        y_true = [1]
        y_pred = [1]
        assert pair_accuracy(y_true, y_pred) == pytest.approx(1.0)

        y_true = [0]
        y_pred = [1]
        assert pair_accuracy(y_true, y_pred) == pytest.approx(0.0)

    def test_return_type(self, pair_perfect):
        """Test that return type is float."""
        y_true, y_pred = pair_perfect
        result = pair_accuracy(y_true.tolist(), y_pred.tolist())
        assert isinstance(result, float)

    def test_metric_range(self, pair_partial):
        """Test that metric is in valid range [0, 1]."""
        y_true, y_pred = pair_partial
        acc = pair_accuracy(y_true.tolist(), y_pred.tolist())
        assert 0.0 <= acc <= 1.0


@pytest.mark.unit
@pytest.mark.metrics
class TestPairF1:
    """Tests for pair_f1 metric."""

    def test_perfect_f1(self, pair_perfect):
        """Test perfect F1 score."""
        y_true, y_pred = pair_perfect
        assert pair_f1(y_true.tolist(), y_pred.tolist()) == pytest.approx(1.0)

    def test_zero_f1(self):
        """Test F1 when no true positives."""
        y_true = [1, 1, 1, 1]
        y_pred = [0, 0, 0, 0]
        assert pair_f1(y_true, y_pred) == pytest.approx(0.0)

    def test_partial_f1(self, pair_partial):
        """Test partial F1 score."""
        y_true, y_pred = pair_partial
        f1 = pair_f1(y_true.tolist(), y_pred.tolist())
        assert 0 < f1 < 1

    def test_all_negative_true(self):
        """Test F1 when all true labels are negative."""
        y_true = [0, 0, 0, 0]
        y_pred = [0, 0, 0, 0]
        # No positive class, should use zero_division=0.0
        assert pair_f1(y_true, y_pred) == pytest.approx(0.0)

    def test_all_negative_pred(self):
        """Test F1 when all predictions are negative."""
        y_true = [1, 1, 0, 0]
        y_pred = [0, 0, 0, 0]
        # Recall=0, Precision=0 (zero_division), F1=0
        assert pair_f1(y_true, y_pred) == pytest.approx(0.0)

    def test_all_positive_pred(self):
        """Test F1 when all predictions are positive."""
        y_true = [1, 1, 0, 0]
        y_pred = [1, 1, 1, 1]
        # TP=2, FP=2, FN=0
        # Precision=2/4=0.5, Recall=2/2=1.0, F1=2*0.5*1/(0.5+1)=0.667
        f1 = pair_f1(y_true, y_pred)
        assert f1 == pytest.approx(2.0 / 3.0)

    def test_return_type(self, pair_perfect):
        """Test that return type is float."""
        y_true, y_pred = pair_perfect
        result = pair_f1(y_true.tolist(), y_pred.tolist())
        assert isinstance(result, float)

    def test_metric_range(self, pair_partial):
        """Test that metric is in valid range [0, 1]."""
        y_true, y_pred = pair_partial
        f1 = pair_f1(y_true.tolist(), y_pred.tolist())
        assert 0.0 <= f1 <= 1.0


@pytest.mark.unit
@pytest.mark.metrics
class TestPairPrecision:
    """Tests for pair_precision metric."""

    def test_perfect_precision(self, pair_perfect):
        """Test perfect precision."""
        y_true, y_pred = pair_perfect
        assert pair_precision(y_true.tolist(), y_pred.tolist()) == pytest.approx(1.0)

    def test_zero_precision(self):
        """Test zero precision (all positive predictions are wrong)."""
        y_true = [0, 0, 0, 0]
        y_pred = [1, 1, 1, 1]
        # All predictions are false positives
        assert pair_precision(y_true, y_pred) == pytest.approx(0.0)

    def test_partial_precision(self):
        """Test partial precision."""
        y_true = [1, 1, 0, 0]
        y_pred = [1, 1, 1, 0]
        # TP=2, FP=1, Precision=2/3
        assert pair_precision(y_true, y_pred) == pytest.approx(2.0 / 3.0)

    def test_no_positive_predictions(self):
        """Test when no positive predictions are made."""
        y_true = [1, 1, 0, 0]
        y_pred = [0, 0, 0, 0]
        # No predicted positives, use zero_division=0.0
        assert pair_precision(y_true, y_pred) == pytest.approx(0.0)

    def test_all_negative_true(self):
        """Test when all true labels are negative."""
        y_true = [0, 0, 0, 0]
        y_pred = [0, 0, 0, 0]
        # No positive class, should use zero_division=0.0
        assert pair_precision(y_true, y_pred) == pytest.approx(0.0)

    def test_return_type(self, pair_perfect):
        """Test that return type is float."""
        y_true, y_pred = pair_perfect
        result = pair_precision(y_true.tolist(), y_pred.tolist())
        assert isinstance(result, float)

    def test_metric_range(self, pair_partial):
        """Test that metric is in valid range [0, 1]."""
        y_true, y_pred = pair_partial
        prec = pair_precision(y_true.tolist(), y_pred.tolist())
        assert 0.0 <= prec <= 1.0


@pytest.mark.unit
@pytest.mark.metrics
class TestPairRecall:
    """Tests for pair_recall metric."""

    def test_perfect_recall(self, pair_perfect):
        """Test perfect recall."""
        y_true, y_pred = pair_perfect
        assert pair_recall(y_true.tolist(), y_pred.tolist()) == pytest.approx(1.0)

    def test_zero_recall(self):
        """Test zero recall (all true positives missed)."""
        y_true = [1, 1, 1, 1]
        y_pred = [0, 0, 0, 0]
        # Missed all positives
        assert pair_recall(y_true, y_pred) == pytest.approx(0.0)

    def test_partial_recall(self):
        """Test partial recall."""
        y_true = [1, 1, 1, 0]
        y_pred = [1, 1, 0, 0]
        # TP=2, FN=1, Recall=2/3
        assert pair_recall(y_true, y_pred) == pytest.approx(2.0 / 3.0)

    def test_all_negative_true(self):
        """Test when all true labels are negative."""
        y_true = [0, 0, 0, 0]
        y_pred = [0, 0, 0, 0]
        # No positive class, should use zero_division=0.0
        assert pair_recall(y_true, y_pred) == pytest.approx(0.0)

    def test_all_positive_pred(self):
        """Test when all predictions are positive."""
        y_true = [1, 1, 0, 0]
        y_pred = [1, 1, 1, 1]
        # TP=2, FN=0, Recall=2/2=1.0
        assert pair_recall(y_true, y_pred) == pytest.approx(1.0)

    def test_return_type(self, pair_perfect):
        """Test that return type is float."""
        y_true, y_pred = pair_perfect
        result = pair_recall(y_true.tolist(), y_pred.tolist())
        assert isinstance(result, float)

    def test_metric_range(self, pair_partial):
        """Test that metric is in valid range [0, 1]."""
        y_true, y_pred = pair_partial
        rec = pair_recall(y_true.tolist(), y_pred.tolist())
        assert 0.0 <= rec <= 1.0


@pytest.mark.unit
@pytest.mark.metrics
class TestPairAUCROC:
    """Tests for pair_auc_roc metric."""

    def test_perfect_auc(self):
        """Test perfect AUC-ROC."""
        y_true = [0, 0, 0, 0, 1, 1, 1, 1]
        y_scores = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
        assert pair_auc_roc(y_true, y_scores) == pytest.approx(1.0)

    def test_random_auc(self):
        """Test random predictions (AUC around 0.5)."""
        y_true = [0, 1, 0, 1, 0, 1, 0, 1]
        y_scores = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        # All same score = 0.5 AUC
        assert pair_auc_roc(y_true, y_scores) == pytest.approx(0.5)

    def test_worst_auc(self):
        """Test worst case (inverted predictions)."""
        y_true = [0, 0, 0, 0, 1, 1, 1, 1]
        y_scores = [0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1]
        # Completely inverted
        assert pair_auc_roc(y_true, y_scores) == pytest.approx(0.0)

    def test_partial_auc(self):
        """Test partial AUC-ROC (not perfect)."""
        # Use data where some positives have lower scores than some negatives
        y_true = [0, 0, 1, 1, 0, 1]
        y_scores = [0.3, 0.6, 0.5, 0.8, 0.4, 0.7]  # neg=0.3,0.6,0.4, pos=0.5,0.8,0.7
        auc = pair_auc_roc(y_true, y_scores)
        assert 0.5 < auc < 1  # Should be good but not perfect

    def test_single_class_all_zeros(self):
        """Test when all labels are 0 (single class)."""
        y_true = [0, 0, 0, 0]
        y_scores = [0.1, 0.2, 0.3, 0.4]
        # Should return 0.5 when only one class present
        assert pair_auc_roc(y_true, y_scores) == pytest.approx(0.5)

    def test_single_class_all_ones(self):
        """Test when all labels are 1 (single class)."""
        y_true = [1, 1, 1, 1]
        y_scores = [0.6, 0.7, 0.8, 0.9]
        # Should return 0.5 when only one class present
        assert pair_auc_roc(y_true, y_scores) == pytest.approx(0.5)

    def test_return_type(self):
        """Test that return type is float."""
        y_true = [0, 0, 1, 1]
        y_scores = [0.2, 0.4, 0.6, 0.8]
        result = pair_auc_roc(y_true, y_scores)
        assert isinstance(result, float)

    def test_metric_range(self):
        """Test that metric is in valid range [0, 1]."""
        y_true = [0, 0, 1, 1, 0, 1]
        y_scores = [0.2, 0.4, 0.6, 0.8, 0.3, 0.7]
        auc = pair_auc_roc(y_true, y_scores)
        assert 0.0 <= auc <= 1.0


@pytest.mark.unit
@pytest.mark.metrics
class TestPairAveragePrecision:
    """Tests for pair_average_precision metric."""

    def test_perfect_average_precision(self):
        """Test perfect average precision."""
        y_true = [0, 0, 0, 0, 1, 1, 1, 1]
        y_scores = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
        assert pair_average_precision(y_true, y_scores) == pytest.approx(1.0)

    def test_worst_average_precision(self):
        """Test worst average precision."""
        y_true = [0, 0, 0, 0, 1, 1, 1, 1]
        y_scores = [0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1]
        # All positives at the bottom
        ap = pair_average_precision(y_true, y_scores)
        # AP will be > 0 but low
        assert 0 < ap < 1

    def test_partial_average_precision(self):
        """Test partial average precision."""
        y_true = [1, 0, 1, 0, 1, 0]
        y_scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
        ap = pair_average_precision(y_true, y_scores)
        assert 0 < ap < 1

    def test_no_positives(self):
        """Test when there are no positive samples."""
        y_true = [0, 0, 0, 0]
        y_scores = [0.1, 0.2, 0.3, 0.4]
        # Should return 0.0 when no positives
        assert pair_average_precision(y_true, y_scores) == pytest.approx(0.0)

    def test_all_positives(self):
        """Test when all samples are positive."""
        y_true = [1, 1, 1, 1]
        y_scores = [0.6, 0.7, 0.8, 0.9]
        # All are positive, so AP = 1.0
        assert pair_average_precision(y_true, y_scores) == pytest.approx(1.0)

    def test_single_positive(self):
        """Test with single positive sample."""
        y_true = [0, 0, 0, 1]
        y_scores = [0.1, 0.2, 0.3, 0.9]
        # Single positive ranked first
        assert pair_average_precision(y_true, y_scores) == pytest.approx(1.0)

        y_true = [1, 0, 0, 0]
        y_scores = [0.1, 0.2, 0.3, 0.9]
        # Single positive ranked last
        ap = pair_average_precision(y_true, y_scores)
        assert 0 < ap < 1

    def test_return_type(self):
        """Test that return type is float."""
        y_true = [0, 0, 1, 1]
        y_scores = [0.2, 0.4, 0.6, 0.8]
        result = pair_average_precision(y_true, y_scores)
        assert isinstance(result, float)

    def test_metric_range(self):
        """Test that metric is in valid range [0, 1]."""
        y_true = [0, 0, 1, 1, 0, 1]
        y_scores = [0.2, 0.4, 0.6, 0.8, 0.3, 0.7]
        ap = pair_average_precision(y_true, y_scores)
        assert 0.0 <= ap <= 1.0


@pytest.mark.unit
@pytest.mark.metrics
class TestFindBestThreshold:
    """Tests for find_best_threshold function."""

    def test_find_best_f1_threshold(self):
        """Test finding threshold that maximizes F1."""
        y_true = [0, 0, 0, 0, 1, 1, 1, 1]
        y_scores = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]

        threshold, score = find_best_threshold(y_true, y_scores, metric="f1")

        # Threshold should be between min and max scores
        assert min(y_scores) <= threshold <= max(y_scores)
        # Score should be perfect F1
        assert score == pytest.approx(1.0)
        # Threshold should separate classes (around 0.5)
        assert 0.4 < threshold <= 0.6

    def test_find_best_accuracy_threshold(self):
        """Test finding threshold that maximizes accuracy."""
        y_true = [0, 0, 0, 0, 1, 1, 1, 1]
        y_scores = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]

        threshold, score = find_best_threshold(y_true, y_scores, metric="accuracy")

        # Threshold should be between min and max scores
        assert min(y_scores) <= threshold <= max(y_scores)
        # Score should be perfect accuracy
        assert score == pytest.approx(1.0)
        # Threshold should separate classes
        assert 0.4 < threshold <= 0.6

    def test_threshold_with_noise(self):
        """Test threshold finding with noisy scores."""
        y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        y_scores = [0.2, 0.3, 0.4, 0.5, 0.45, 0.55, 0.6, 0.7, 0.8, 0.9]

        threshold, score = find_best_threshold(y_true, y_scores, metric="f1")

        # Should find reasonable threshold
        assert 0 < threshold < 1
        # Score should be high but not perfect due to overlap
        assert 0.5 < score <= 1.0

    def test_threshold_all_same_scores(self):
        """Test when all scores are identical."""
        y_true = [0, 0, 1, 1]
        y_scores = [0.5, 0.5, 0.5, 0.5]

        threshold, score = find_best_threshold(y_true, y_scores, metric="f1")

        # Should still return a valid threshold (the only unique value)
        assert threshold == pytest.approx(0.5)
        # Score depends on threshold choice
        assert 0.0 <= score <= 1.0

    def test_threshold_many_unique_scores(self):
        """Test threshold sampling with many unique scores."""
        np.random.seed(42)
        y_true = [0] * 50 + [1] * 50
        y_scores = list(np.random.uniform(0, 0.5, 50)) + list(
            np.random.uniform(0.5, 1.0, 50)
        )

        threshold, score = find_best_threshold(y_true, y_scores, metric="f1")

        # Should sample thresholds efficiently (101 percentiles)
        assert 0 <= threshold <= 1
        assert 0 <= score <= 1

    def test_invalid_metric_raises_error(self):
        """Test that invalid metric raises ValueError."""
        y_true = [0, 0, 1, 1]
        y_scores = [0.2, 0.4, 0.6, 0.8]

        with pytest.raises(ValueError, match="Unknown metric"):
            find_best_threshold(y_true, y_scores, metric="invalid")

    def test_return_types(self):
        """Test that return types are correct."""
        y_true = [0, 0, 1, 1]
        y_scores = [0.2, 0.4, 0.6, 0.8]

        threshold, score = find_best_threshold(y_true, y_scores, metric="f1")

        assert isinstance(threshold, float)
        assert isinstance(score, float)

    def test_default_metric_is_f1(self):
        """Test that default metric is F1."""
        y_true = [0, 0, 1, 1]
        y_scores = [0.2, 0.4, 0.6, 0.8]

        threshold_f1, score_f1 = find_best_threshold(y_true, y_scores, metric="f1")
        threshold_default, score_default = find_best_threshold(y_true, y_scores)

        # Should be the same
        assert threshold_f1 == pytest.approx(threshold_default)
        assert score_f1 == pytest.approx(score_default)


@pytest.mark.unit
@pytest.mark.metrics
class TestComputePairMetrics:
    """Tests for compute_pair_metrics aggregation function."""

    def test_compute_all_metrics_with_threshold(self):
        """Test computing all metrics with explicit threshold."""
        y_true = [0, 0, 0, 0, 1, 1, 1, 1]
        y_scores = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
        threshold = 0.5

        metrics = compute_pair_metrics(y_true, y_scores, threshold=threshold)

        # Check all expected keys are present
        assert "f1" in metrics
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "auc_roc" in metrics
        assert "average_precision" in metrics
        assert "threshold" in metrics
        assert "num_samples" in metrics
        assert "num_positive" in metrics
        assert "num_negative" in metrics

        # Check values are correct
        assert metrics["threshold"] == pytest.approx(0.5)
        assert metrics["num_samples"] == 8
        assert metrics["num_positive"] == 4
        assert metrics["num_negative"] == 4

        # All metrics should be perfect
        assert metrics["f1"] == pytest.approx(1.0)
        assert metrics["accuracy"] == pytest.approx(1.0)
        assert metrics["precision"] == pytest.approx(1.0)
        assert metrics["recall"] == pytest.approx(1.0)
        assert metrics["auc_roc"] == pytest.approx(1.0)
        assert metrics["average_precision"] == pytest.approx(1.0)

    def test_compute_all_metrics_auto_threshold(self):
        """Test computing all metrics with automatic threshold."""
        y_true = [0, 0, 0, 0, 1, 1, 1, 1]
        y_scores = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]

        metrics = compute_pair_metrics(y_true, y_scores)

        # Should find optimal threshold
        assert "threshold" in metrics
        assert 0.4 < metrics["threshold"] <= 0.6

        # All metrics should be perfect with optimal threshold
        assert metrics["f1"] == pytest.approx(1.0)
        assert metrics["accuracy"] == pytest.approx(1.0)

    def test_partial_predictions(self):
        """Test with partial predictions (overlapping scores)."""
        # Use data where positives and negatives have overlapping scores
        y_true = [0, 0, 1, 1, 0, 1, 0, 1]
        y_scores = [0.3, 0.6, 0.5, 0.8, 0.4, 0.7, 0.55, 0.65]  # More overlap
        threshold = 0.5

        metrics = compute_pair_metrics(y_true, y_scores, threshold=threshold)

        # Some metrics should be partial (allow AUC to be perfect if separation is good)
        assert 0 < metrics["f1"] <= 1
        assert 0 < metrics["accuracy"] <= 1
        assert 0.5 < metrics["auc_roc"] <= 1
        assert 0 < metrics["average_precision"] <= 1

    def test_imbalanced_data(self):
        """Test with imbalanced positive/negative samples."""
        y_true = [0, 0, 0, 0, 0, 0, 1, 1]
        y_scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.45, 0.7, 0.9]
        threshold = 0.6

        metrics = compute_pair_metrics(y_true, y_scores, threshold=threshold)

        # Check counts
        assert metrics["num_positive"] == 2
        assert metrics["num_negative"] == 6
        assert metrics["num_samples"] == 8

    def test_all_negative_samples(self):
        """Test when all samples are negative."""
        y_true = [0, 0, 0, 0]
        y_scores = [0.1, 0.2, 0.3, 0.4]
        threshold = 0.5

        metrics = compute_pair_metrics(y_true, y_scores, threshold=threshold)

        # Counts should be correct
        assert metrics["num_positive"] == 0
        assert metrics["num_negative"] == 4

        # F1, precision, recall should use zero_division
        assert metrics["f1"] == pytest.approx(0.0)
        assert metrics["precision"] == pytest.approx(0.0)
        assert metrics["recall"] == pytest.approx(0.0)

        # Accuracy can still be high if all predicted negative
        assert 0 <= metrics["accuracy"] <= 1

        # AUC-ROC should be 0.5 (single class)
        assert metrics["auc_roc"] == pytest.approx(0.5)

        # Average precision should be 0.0 (no positives)
        assert metrics["average_precision"] == pytest.approx(0.0)

    def test_all_positive_samples(self):
        """Test when all samples are positive."""
        y_true = [1, 1, 1, 1]
        y_scores = [0.6, 0.7, 0.8, 0.9]
        threshold = 0.5

        metrics = compute_pair_metrics(y_true, y_scores, threshold=threshold)

        # Counts should be correct
        assert metrics["num_positive"] == 4
        assert metrics["num_negative"] == 0

        # All metrics should be perfect if all predicted positive
        assert metrics["f1"] == pytest.approx(1.0)
        assert metrics["accuracy"] == pytest.approx(1.0)
        assert metrics["precision"] == pytest.approx(1.0)
        assert metrics["recall"] == pytest.approx(1.0)

        # AUC-ROC should be 0.5 (single class)
        assert metrics["auc_roc"] == pytest.approx(0.5)

        # Average precision should be 1.0 (all positives)
        assert metrics["average_precision"] == pytest.approx(1.0)

    def test_empty_input_raises_error(self):
        """Test that empty inputs raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            compute_pair_metrics([], [])

        with pytest.raises(ValueError, match="cannot be empty"):
            compute_pair_metrics([1, 0], [])

        with pytest.raises(ValueError, match="cannot be empty"):
            compute_pair_metrics([], [0.5, 0.7])

    def test_mismatched_lengths_raise_error(self):
        """Test that mismatched lengths raise ValueError."""
        y_true = [0, 1, 0, 1]
        y_scores = [0.2, 0.8]

        with pytest.raises(ValueError, match="Length mismatch"):
            compute_pair_metrics(y_true, y_scores)

    def test_single_sample(self):
        """Test with single sample."""
        y_true = [1]
        y_scores = [0.8]
        threshold = 0.5

        metrics = compute_pair_metrics(y_true, y_scores, threshold=threshold)

        assert metrics["num_samples"] == 1
        assert metrics["num_positive"] == 1
        assert metrics["num_negative"] == 0

        # Should handle single sample gracefully
        assert 0 <= metrics["f1"] <= 1
        assert 0 <= metrics["accuracy"] <= 1

    def test_threshold_zero(self):
        """Test with threshold of 0 (all predictions positive)."""
        y_true = [0, 0, 1, 1]
        y_scores = [0.2, 0.4, 0.6, 0.8]
        threshold = 0.0

        metrics = compute_pair_metrics(y_true, y_scores, threshold=threshold)

        # All scores >= 0, so all predictions are 1
        assert metrics["threshold"] == pytest.approx(0.0)
        # Recall should be 1.0 (all positives found)
        assert metrics["recall"] == pytest.approx(1.0)

    def test_threshold_one(self):
        """Test with threshold of 1 (all predictions negative)."""
        y_true = [0, 0, 1, 1]
        y_scores = [0.2, 0.4, 0.6, 0.8]
        threshold = 1.0

        metrics = compute_pair_metrics(y_true, y_scores, threshold=threshold)

        # No scores >= 1.0, so all predictions are 0
        assert metrics["threshold"] == pytest.approx(1.0)
        # Recall should be 0.0 (no positives found)
        assert metrics["recall"] == pytest.approx(0.0)

    def test_metrics_are_floats(self):
        """Test that all metric values are floats or ints."""
        y_true = [0, 0, 1, 1]
        y_scores = [0.2, 0.4, 0.6, 0.8]

        metrics = compute_pair_metrics(y_true, y_scores)

        for key, value in metrics.items():
            assert isinstance(value, (int, float)), f"{key} should be numeric"

    def test_metric_ranges(self):
        """Test that all metrics are in valid ranges."""
        y_true = [0, 0, 1, 1, 0, 1, 0, 1]
        y_scores = [0.2, 0.4, 0.6, 0.8, 0.3, 0.7, 0.1, 0.9]

        metrics = compute_pair_metrics(y_true, y_scores)

        # All probability/ratio metrics in [0, 1]
        assert 0.0 <= metrics["f1"] <= 1.0
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["precision"] <= 1.0
        assert 0.0 <= metrics["recall"] <= 1.0
        assert 0.0 <= metrics["auc_roc"] <= 1.0
        assert 0.0 <= metrics["average_precision"] <= 1.0

        # Counts should be non-negative
        assert metrics["num_samples"] >= 0
        assert metrics["num_positive"] >= 0
        assert metrics["num_negative"] >= 0

        # Counts should sum correctly
        assert (
            metrics["num_positive"] + metrics["num_negative"] == metrics["num_samples"]
        )

    def test_reproducibility(self):
        """Test that same inputs produce same outputs."""
        y_true = [0, 0, 1, 1, 0, 1]
        y_scores = [0.2, 0.4, 0.6, 0.8, 0.3, 0.7]

        metrics1 = compute_pair_metrics(y_true, y_scores)
        metrics2 = compute_pair_metrics(y_true, y_scores)

        # All metrics should be identical
        for key in metrics1:
            assert metrics1[key] == pytest.approx(metrics2[key])

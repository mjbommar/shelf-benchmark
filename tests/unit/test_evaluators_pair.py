"""Unit tests for shelf.evaluate.evaluators.pair module.

Tests cover:
- PairClassificationEvaluator initialization
- evaluate() method with various prediction scenarios
- Prediction validation and error handling
- Threshold optimization
- Metrics computation (F1, accuracy, precision, recall, AUC-ROC, average precision)
- Handling imbalanced pair data
- Edge cases: perfect predictions, partial predictions, all same class
- Input validation: missing predictions, invalid formats

Note: This tests the PairClassificationEvaluator class directly.
Integration tests with actual embedders are in tests/integration/.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import pytest

from shelf.evaluate.evaluators.pair import PairClassificationEvaluator
from shelf.evaluate.schemas import ValidationError
from shelf.evaluate.tasks import TaskSpec, TaskType


# ===========================================================================
# Fixtures for PairClassificationEvaluator Tests
# ===========================================================================


@pytest.fixture
def pair_task_spec() -> TaskSpec:
    """Task specification for pair classification."""
    return TaskSpec(
        name="test_pair_classification",
        task_type=TaskType.PAIR_CLASSIFICATION,
        description="Test pair classification task",
        text_field="text",
        label_field="label",
        id_field="pair_id",
        label_space=None,  # Binary: 0 or 1
        primary_metric="f1",
        secondary_metrics=(
            "accuracy",
            "precision",
            "recall",
            "auc_roc",
            "average_precision",
        ),
        dataset_name="test/dataset",
        dataset_config="same_lcc_pairs",
        default_split="test",
    )


@pytest.fixture
def pair_ground_truth() -> pl.DataFrame:
    """Ground truth for pair classification with balanced classes."""
    return pl.DataFrame(
        {
            "pair_id": [f"pair_{i:03d}" for i in range(10)],
            "doc_a_title": [f"Title A{i}" for i in range(10)],
            "doc_a_body": [f"Body A{i}" for i in range(10)],
            "doc_b_title": [f"Title B{i}" for i in range(10)],
            "doc_b_body": [f"Body B{i}" for i in range(10)],
            "label": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],  # 5 positive, 5 negative
        }
    )


@pytest.fixture
def imbalanced_pair_ground_truth() -> pl.DataFrame:
    """Ground truth for pair classification with imbalanced classes."""
    return pl.DataFrame(
        {
            "pair_id": [f"pair_{i:03d}" for i in range(20)],
            "doc_a_title": [f"Title A{i}" for i in range(20)],
            "doc_a_body": [f"Body A{i}" for i in range(20)],
            "doc_b_title": [f"Title B{i}" for i in range(20)],
            "doc_b_body": [f"Body B{i}" for i in range(20)],
            "label": [1] * 2 + [0] * 18,  # 2 positive, 18 negative (10% imbalance)
        }
    )


@pytest.fixture
def perfect_pair_predictions() -> list[dict[str, Any]]:
    """Perfect predictions using scores that perfectly separate classes."""
    return [{"pair_id": f"pair_{i:03d}", "score": 0.9} for i in range(5)] + [
        {"pair_id": f"pair_{i:03d}", "score": 0.1} for i in range(5, 10)
    ]


@pytest.fixture
def partial_pair_predictions() -> list[dict[str, Any]]:
    """Partial predictions with some errors (75% accuracy)."""
    return [
        {"pair_id": "pair_000", "score": 0.9},  # TP
        {"pair_id": "pair_001", "score": 0.8},  # TP
        {"pair_id": "pair_002", "score": 0.7},  # TP
        {"pair_id": "pair_003", "score": 0.2},  # FN (should be 1, scored low)
        {"pair_id": "pair_004", "score": 0.85},  # TP
        {"pair_id": "pair_005", "score": 0.1},  # TN
        {"pair_id": "pair_006", "score": 0.15},  # TN
        {"pair_id": "pair_007", "score": 0.75},  # FP (should be 0, scored high)
        {"pair_id": "pair_008", "score": 0.2},  # TN
        {"pair_id": "pair_009", "score": 0.1},  # TN
    ]


@pytest.fixture
def binary_pair_predictions() -> list[dict[str, Any]]:
    """Predictions using binary labels instead of scores."""
    return [{"pair_id": f"pair_{i:03d}", "prediction": 1} for i in range(5)] + [
        {"pair_id": f"pair_{i:03d}", "prediction": 0} for i in range(5, 10)
    ]


@pytest.fixture
def mixed_format_predictions() -> list[dict[str, Any]]:
    """Predictions with mixed score and binary formats."""
    return [
        {"pair_id": "pair_000", "score": 0.9},
        {"pair_id": "pair_001", "prediction": 1},
        {"pair_id": "pair_002", "score": 0.8, "prediction": 1},
        {"pair_id": "pair_003", "score": 0.3},
        {"pair_id": "pair_004", "prediction": 1},
        {"pair_id": "pair_005", "score": 0.1},
        {"pair_id": "pair_006", "prediction": 0},
        {"pair_id": "pair_007", "score": 0.2},
        {"pair_id": "pair_008", "prediction": 0},
        {"pair_id": "pair_009", "score": 0.05, "prediction": 0},
    ]


# ===========================================================================
# Tests for PairClassificationEvaluator Initialization
# ===========================================================================


@pytest.mark.unit
@pytest.mark.evaluator
class TestPairClassificationEvaluatorInit:
    """Tests for PairClassificationEvaluator initialization."""

    def test_init_basic(self, pair_task_spec):
        """Test basic initialization."""
        evaluator = PairClassificationEvaluator(pair_task_spec)

        assert evaluator.task_spec == pair_task_spec
        assert evaluator.random_seed == 42

    def test_init_custom_seed(self, pair_task_spec):
        """Test initialization with custom random seed."""
        evaluator = PairClassificationEvaluator(pair_task_spec, random_seed=123)

        assert evaluator.random_seed == 123

    def test_init_stores_task_type(self, pair_task_spec):
        """Test that evaluator stores correct task type."""
        evaluator = PairClassificationEvaluator(pair_task_spec)

        assert evaluator.task_spec.task_type == TaskType.PAIR_CLASSIFICATION

    def test_init_stores_primary_metric(self, pair_task_spec):
        """Test that evaluator stores primary metric."""
        evaluator = PairClassificationEvaluator(pair_task_spec)

        assert evaluator.task_spec.primary_metric == "f1"


# ===========================================================================
# Tests for evaluate() Method
# ===========================================================================


@pytest.mark.unit
@pytest.mark.evaluator
class TestPairClassificationEvaluatorEvaluate:
    """Tests for PairClassificationEvaluator.evaluate() method."""

    def test_evaluate_perfect_predictions(
        self, pair_task_spec, pair_ground_truth, perfect_pair_predictions
    ):
        """Test evaluation with perfect predictions."""
        evaluator = PairClassificationEvaluator(pair_task_spec)
        result = evaluator.evaluate(perfect_pair_predictions, pair_ground_truth)

        # Check result structure
        assert result.task == "test_pair_classification"
        assert result.task_type == "pair_classification"
        assert result.primary_metric == "f1"
        assert result.num_samples == 10

        # Check metrics - should be perfect
        assert result.metrics["f1"] == pytest.approx(1.0)
        assert result.metrics["accuracy"] == pytest.approx(1.0)
        assert result.metrics["precision"] == pytest.approx(1.0)
        assert result.metrics["recall"] == pytest.approx(1.0)
        assert result.metrics["auc_roc"] == pytest.approx(1.0)
        assert result.metrics["average_precision"] == pytest.approx(1.0)

        # Check that threshold was computed
        assert "threshold" in result.metrics
        assert 0.0 <= result.metrics["threshold"] <= 1.0

        # Check sample counts
        assert result.metrics["num_samples"] == 10
        assert result.metrics["num_positive"] == 5
        assert result.metrics["num_negative"] == 5

    def test_evaluate_partial_predictions(
        self, pair_task_spec, pair_ground_truth, partial_pair_predictions
    ):
        """Test evaluation with partial predictions (some errors)."""
        evaluator = PairClassificationEvaluator(pair_task_spec)
        result = evaluator.evaluate(partial_pair_predictions, pair_ground_truth)

        # Check metrics - should be imperfect
        assert 0.5 < result.metrics["f1"] < 1.0
        assert 0.5 < result.metrics["accuracy"] < 1.0
        assert result.metrics["auc_roc"] > 0.5  # Better than random

        # Counts should still be correct
        assert result.metrics["num_samples"] == 10
        assert result.metrics["num_positive"] == 5
        assert result.metrics["num_negative"] == 5

    def test_evaluate_binary_predictions(
        self, pair_task_spec, pair_ground_truth, binary_pair_predictions
    ):
        """Test evaluation with binary predictions instead of scores."""
        evaluator = PairClassificationEvaluator(pair_task_spec)
        result = evaluator.evaluate(binary_pair_predictions, pair_ground_truth)

        # Should handle binary predictions
        assert result.metrics["f1"] == pytest.approx(1.0)
        assert result.metrics["accuracy"] == pytest.approx(1.0)

        # Note: AUC-ROC computed from binary values (0/1) may not be meaningful
        # but should still be in valid range
        assert 0.0 <= result.metrics["auc_roc"] <= 1.0

    def test_evaluate_mixed_format(
        self, pair_task_spec, pair_ground_truth, mixed_format_predictions
    ):
        """Test evaluation with mixed score and binary predictions."""
        evaluator = PairClassificationEvaluator(pair_task_spec)
        result = evaluator.evaluate(mixed_format_predictions, pair_ground_truth)

        # Should handle mixed formats (prefers score over prediction)
        assert result.num_samples == 10
        assert "f1" in result.metrics
        assert "accuracy" in result.metrics

    def test_evaluate_imbalanced_data(
        self, pair_task_spec, imbalanced_pair_ground_truth
    ):
        """Test evaluation with highly imbalanced data."""
        # Create predictions that are all negative (majority class baseline)
        predictions = [{"pair_id": f"pair_{i:03d}", "score": 0.1} for i in range(20)]

        evaluator = PairClassificationEvaluator(pair_task_spec)
        result = evaluator.evaluate(predictions, imbalanced_pair_ground_truth)

        # Check counts
        assert result.metrics["num_positive"] == 2
        assert result.metrics["num_negative"] == 18

        # With optimal threshold, even low scores can work
        # Just check that metrics are computed
        assert 0.0 <= result.metrics["accuracy"] <= 1.0
        assert 0.0 <= result.metrics["f1"] <= 1.0

    def test_evaluate_all_same_scores(self, pair_task_spec, pair_ground_truth):
        """Test evaluation when all predictions have same score."""
        predictions = [{"pair_id": f"pair_{i:03d}", "score": 0.5} for i in range(10)]

        evaluator = PairClassificationEvaluator(pair_task_spec)
        result = evaluator.evaluate(predictions, pair_ground_truth)

        # Should handle gracefully even though all scores are identical
        assert result.num_samples == 10
        assert "threshold" in result.metrics
        # AUC-ROC should be 0.5 for constant predictions
        assert result.metrics["auc_roc"] == pytest.approx(0.5)

    def test_evaluate_all_positive_labels(self, pair_task_spec):
        """Test evaluation when all ground truth labels are positive."""
        ground_truth = pl.DataFrame(
            {
                "pair_id": [f"pair_{i:03d}" for i in range(5)],
                "doc_a_title": [f"Title A{i}" for i in range(5)],
                "doc_a_body": [f"Body A{i}" for i in range(5)],
                "doc_b_title": [f"Title B{i}" for i in range(5)],
                "doc_b_body": [f"Body B{i}" for i in range(5)],
                "label": [1, 1, 1, 1, 1],
            }
        )

        predictions = [{"pair_id": f"pair_{i:03d}", "score": 0.9} for i in range(5)]

        evaluator = PairClassificationEvaluator(pair_task_spec)
        result = evaluator.evaluate(predictions, ground_truth)

        # Check counts
        assert result.metrics["num_positive"] == 5
        assert result.metrics["num_negative"] == 0

        # Should handle single-class scenario
        # AUC-ROC should be 0.5 (single class)
        assert result.metrics["auc_roc"] == pytest.approx(0.5)
        # Average precision should be 1.0 (all positives)
        assert result.metrics["average_precision"] == pytest.approx(1.0)

    def test_evaluate_all_negative_labels(self, pair_task_spec):
        """Test evaluation when all ground truth labels are negative."""
        ground_truth = pl.DataFrame(
            {
                "pair_id": [f"pair_{i:03d}" for i in range(5)],
                "doc_a_title": [f"Title A{i}" for i in range(5)],
                "doc_a_body": [f"Body A{i}" for i in range(5)],
                "doc_b_title": [f"Title B{i}" for i in range(5)],
                "doc_b_body": [f"Body B{i}" for i in range(5)],
                "label": [0, 0, 0, 0, 0],
            }
        )

        predictions = [{"pair_id": f"pair_{i:03d}", "score": 0.1} for i in range(5)]

        evaluator = PairClassificationEvaluator(pair_task_spec)
        result = evaluator.evaluate(predictions, ground_truth)

        # Check counts
        assert result.metrics["num_positive"] == 0
        assert result.metrics["num_negative"] == 5

        # Should handle single-class scenario
        # AUC-ROC should be 0.5 (single class)
        assert result.metrics["auc_roc"] == pytest.approx(0.5)
        # Average precision should be 0.0 (no positives)
        assert result.metrics["average_precision"] == pytest.approx(0.0)
        # F1, precision, recall should be 0.0 (zero_division)
        assert result.metrics["f1"] == pytest.approx(0.0)

    def test_evaluate_computes_primary_score(
        self, pair_task_spec, pair_ground_truth, perfect_pair_predictions
    ):
        """Test that primary_score is set correctly."""
        evaluator = PairClassificationEvaluator(pair_task_spec)
        result = evaluator.evaluate(perfect_pair_predictions, pair_ground_truth)

        # Primary metric is f1
        assert result.primary_score == result.metrics["f1"]
        assert result.primary_score == pytest.approx(1.0)

    def test_evaluate_creates_context(
        self, pair_task_spec, pair_ground_truth, perfect_pair_predictions
    ):
        """Test that evaluation context is created."""
        evaluator = PairClassificationEvaluator(pair_task_spec)
        result = evaluator.evaluate(perfect_pair_predictions, pair_ground_truth)

        assert result.context is not None
        assert result.context.random_seed == 42
        assert result.context.dataset_checksum is not None

    def test_evaluate_with_id_field(self, pair_task_spec, pair_ground_truth):
        """Test evaluation when predictions use 'id' instead of 'pair_id'."""
        # Some predictions may use generic 'id' field
        predictions = [
            {"id": f"pair_{i:03d}", "score": 0.9 if i < 5 else 0.1} for i in range(10)
        ]

        evaluator = PairClassificationEvaluator(pair_task_spec)
        result = evaluator.evaluate(predictions, pair_ground_truth)

        # Should work with 'id' field
        assert result.num_samples == 10
        assert result.metrics["f1"] == pytest.approx(1.0)


# ===========================================================================
# Tests for Threshold Optimization
# ===========================================================================


@pytest.mark.unit
@pytest.mark.evaluator
class TestPairClassificationThresholdOptimization:
    """Tests for threshold optimization in pair classification."""

    def test_threshold_is_optimized(
        self, pair_task_spec, pair_ground_truth, partial_pair_predictions
    ):
        """Test that threshold is found to maximize F1."""
        evaluator = PairClassificationEvaluator(pair_task_spec)
        result = evaluator.evaluate(partial_pair_predictions, pair_ground_truth)

        # Threshold should be computed
        assert "threshold" in result.metrics
        threshold = result.metrics["threshold"]

        # Should be a reasonable value (not at extremes unless optimal)
        assert 0.0 <= threshold <= 1.0

        # F1 score should be optimized for this threshold
        assert result.metrics["f1"] > 0.0

    def test_threshold_with_clear_separation(self, pair_task_spec, pair_ground_truth):
        """Test threshold finding with clearly separated classes."""
        # Positives all have high scores, negatives all have low scores
        predictions = [
            {"pair_id": f"pair_{i:03d}", "score": 0.8 + i * 0.02} for i in range(5)
        ] + [
            {"pair_id": f"pair_{i:03d}", "score": 0.1 + (i - 5) * 0.02}
            for i in range(5, 10)
        ]

        evaluator = PairClassificationEvaluator(pair_task_spec)
        result = evaluator.evaluate(predictions, pair_ground_truth)

        # Threshold should separate the classes well
        # With clear separation, F1 should be perfect or near-perfect
        assert result.metrics["f1"] >= 0.95
        # Threshold should be found (any value is valid with clear separation)
        assert 0.0 <= result.metrics["threshold"] <= 1.0

    def test_threshold_with_overlap(self, pair_task_spec, pair_ground_truth):
        """Test threshold finding with overlapping score distributions."""
        # Create scores with overlap between classes
        np.random.seed(42)
        predictions = []
        for i in range(5):  # Positive class
            score = np.random.uniform(0.4, 0.9)
            predictions.append({"pair_id": f"pair_{i:03d}", "score": score})
        for i in range(5, 10):  # Negative class
            score = np.random.uniform(0.1, 0.6)
            predictions.append({"pair_id": f"pair_{i:03d}", "score": score})

        evaluator = PairClassificationEvaluator(pair_task_spec)
        result = evaluator.evaluate(predictions, pair_ground_truth)

        # Threshold should be found even with overlap
        assert "threshold" in result.metrics
        # F1 should be > random (0.5)
        assert result.metrics["f1"] > 0.5


# ===========================================================================
# Tests for Validation and Error Handling
# ===========================================================================


@pytest.mark.unit
@pytest.mark.evaluator
class TestPairClassificationValidation:
    """Tests for prediction validation and error handling."""

    def test_missing_predictions_raises_validation_error(
        self, pair_task_spec, pair_ground_truth
    ):
        """Test that missing predictions raise validation error."""
        # Only provide predictions for half the pairs
        predictions = [{"pair_id": f"pair_{i:03d}", "score": 0.5} for i in range(5)]

        evaluator = PairClassificationEvaluator(pair_task_spec)

        # Validation should fail with missing predictions
        with pytest.raises(ValidationError, match="Missing predictions"):
            evaluator.evaluate(predictions, pair_ground_truth)

    def test_extra_predictions_raises_validation_error(
        self, pair_task_spec, pair_ground_truth
    ):
        """Test that extra predictions not in ground truth raise validation error."""
        # Include predictions for pairs not in ground truth
        predictions = [{"pair_id": f"pair_{i:03d}", "score": 0.5} for i in range(15)]

        evaluator = PairClassificationEvaluator(pair_task_spec)

        # Validation should fail with unknown pair_ids
        with pytest.raises(ValidationError, match="unknown pair_id"):
            evaluator.evaluate(predictions, pair_ground_truth)

    def test_wrong_ids_raise_validation_error(self, pair_task_spec, pair_ground_truth):
        """Test that wrong IDs raise validation error."""
        # Predictions with IDs that don't match ground truth
        predictions = [{"pair_id": "wrong_id_001", "score": 0.5}]

        evaluator = PairClassificationEvaluator(pair_task_spec)

        # Validation should fail with unknown pair_id and missing predictions
        with pytest.raises(ValidationError):
            evaluator.evaluate(predictions, pair_ground_truth)

    def test_invalid_prediction_format_raises_error(
        self, pair_task_spec, pair_ground_truth
    ):
        """Test that invalid prediction format raises validation error."""
        # Missing both score and prediction fields
        predictions = [{"pair_id": f"pair_{i:03d}"} for i in range(10)]

        evaluator = PairClassificationEvaluator(pair_task_spec)

        # Should raise ValidationError due to missing required fields
        with pytest.raises(ValidationError):
            evaluator.evaluate(predictions, pair_ground_truth)

    def test_prediction_without_id_raises_validation_error(
        self, pair_task_spec, pair_ground_truth
    ):
        """Test that predictions without ID raise validation error."""
        # Mix of valid predictions and ones without ID
        predictions = [
            {"pair_id": "pair_000", "score": 0.9},
            {"score": 0.8},  # Missing ID - should fail validation
            {"pair_id": "pair_002", "score": 0.7},
        ]

        evaluator = PairClassificationEvaluator(pair_task_spec)

        # Validation should fail due to missing pair_id field
        with pytest.raises(ValidationError):
            evaluator.evaluate(predictions, pair_ground_truth)

    def test_score_defaults_to_zero_if_none(self, pair_task_spec, pair_ground_truth):
        """Test that score defaults to 0.0 if both score and prediction are None."""
        predictions = [
            {"pair_id": f"pair_{i:03d}", "score": 0.5 if i < 5 else None}
            for i in range(10)
        ]

        evaluator = PairClassificationEvaluator(pair_task_spec)

        # Should raise validation error for missing score/prediction
        with pytest.raises(ValidationError):
            evaluator.evaluate(predictions, pair_ground_truth)


# ===========================================================================
# Tests for Metric Computation
# ===========================================================================


@pytest.mark.unit
@pytest.mark.evaluator
class TestPairClassificationMetrics:
    """Tests for metric computation in pair classification."""

    def test_all_metrics_present(
        self, pair_task_spec, pair_ground_truth, perfect_pair_predictions
    ):
        """Test that all expected metrics are present in results."""
        evaluator = PairClassificationEvaluator(pair_task_spec)
        result = evaluator.evaluate(perfect_pair_predictions, pair_ground_truth)

        expected_metrics = [
            "f1",
            "accuracy",
            "precision",
            "recall",
            "auc_roc",
            "average_precision",
            "threshold",
            "num_samples",
            "num_positive",
            "num_negative",
        ]

        for metric in expected_metrics:
            assert metric in result.metrics

    def test_metrics_are_numeric(
        self, pair_task_spec, pair_ground_truth, perfect_pair_predictions
    ):
        """Test that all metrics are numeric."""
        evaluator = PairClassificationEvaluator(pair_task_spec)
        result = evaluator.evaluate(perfect_pair_predictions, pair_ground_truth)

        for metric, value in result.metrics.items():
            assert isinstance(value, (int, float)), f"{metric} should be numeric"

    def test_metrics_in_valid_range(
        self, pair_task_spec, pair_ground_truth, partial_pair_predictions
    ):
        """Test that all metrics are in valid ranges."""
        evaluator = PairClassificationEvaluator(pair_task_spec)
        result = evaluator.evaluate(partial_pair_predictions, pair_ground_truth)

        # Probability/ratio metrics should be in [0, 1]
        for metric in [
            "f1",
            "accuracy",
            "precision",
            "recall",
            "auc_roc",
            "average_precision",
        ]:
            assert 0.0 <= result.metrics[metric] <= 1.0

        # Counts should be non-negative
        assert result.metrics["num_samples"] >= 0
        assert result.metrics["num_positive"] >= 0
        assert result.metrics["num_negative"] >= 0

    def test_counts_sum_correctly(
        self, pair_task_spec, pair_ground_truth, perfect_pair_predictions
    ):
        """Test that positive and negative counts sum to total samples."""
        evaluator = PairClassificationEvaluator(pair_task_spec)
        result = evaluator.evaluate(perfect_pair_predictions, pair_ground_truth)

        assert (
            result.metrics["num_positive"] + result.metrics["num_negative"]
            == result.metrics["num_samples"]
        )

    def test_auc_roc_computation(
        self, pair_task_spec, pair_ground_truth, perfect_pair_predictions
    ):
        """Test AUC-ROC computation with perfect separation."""
        evaluator = PairClassificationEvaluator(pair_task_spec)
        result = evaluator.evaluate(perfect_pair_predictions, pair_ground_truth)

        # Perfect predictions should have AUC-ROC = 1.0
        assert result.metrics["auc_roc"] == pytest.approx(1.0)

    def test_average_precision_computation(
        self, pair_task_spec, pair_ground_truth, perfect_pair_predictions
    ):
        """Test average precision computation."""
        evaluator = PairClassificationEvaluator(pair_task_spec)
        result = evaluator.evaluate(perfect_pair_predictions, pair_ground_truth)

        # Perfect predictions should have average precision = 1.0
        assert result.metrics["average_precision"] == pytest.approx(1.0)


# ===========================================================================
# Tests for Edge Cases
# ===========================================================================


@pytest.mark.unit
@pytest.mark.evaluator
class TestPairClassificationEdgeCases:
    """Tests for edge cases in pair classification."""

    def test_single_pair(self, pair_task_spec):
        """Test evaluation with single pair."""
        ground_truth = pl.DataFrame(
            {
                "pair_id": ["pair_000"],
                "doc_a_title": ["Title A"],
                "doc_a_body": ["Body A"],
                "doc_b_title": ["Title B"],
                "doc_b_body": ["Body B"],
                "label": [1],
            }
        )

        predictions = [{"pair_id": "pair_000", "score": 0.9}]

        evaluator = PairClassificationEvaluator(pair_task_spec)
        result = evaluator.evaluate(predictions, ground_truth)

        assert result.num_samples == 1
        assert result.metrics["num_positive"] == 1
        assert result.metrics["num_negative"] == 0

    def test_very_small_dataset(self, pair_task_spec):
        """Test evaluation with very small dataset (2 pairs)."""
        ground_truth = pl.DataFrame(
            {
                "pair_id": ["pair_000", "pair_001"],
                "doc_a_title": ["Title A0", "Title A1"],
                "doc_a_body": ["Body A0", "Body A1"],
                "doc_b_title": ["Title B0", "Title B1"],
                "doc_b_body": ["Body B0", "Body B1"],
                "label": [1, 0],
            }
        )

        predictions = [
            {"pair_id": "pair_000", "score": 0.9},
            {"pair_id": "pair_001", "score": 0.1},
        ]

        evaluator = PairClassificationEvaluator(pair_task_spec)
        result = evaluator.evaluate(predictions, ground_truth)

        assert result.num_samples == 2
        # Should handle gracefully even with minimal data
        assert "f1" in result.metrics
        assert "auc_roc" in result.metrics

    def test_reproducibility(
        self, pair_task_spec, pair_ground_truth, partial_pair_predictions
    ):
        """Test that same inputs produce same outputs."""
        evaluator1 = PairClassificationEvaluator(pair_task_spec, random_seed=42)
        evaluator2 = PairClassificationEvaluator(pair_task_spec, random_seed=42)

        result1 = evaluator1.evaluate(partial_pair_predictions, pair_ground_truth)
        result2 = evaluator2.evaluate(partial_pair_predictions, pair_ground_truth)

        # All metrics should be identical
        for key in result1.metrics:
            assert result1.metrics[key] == pytest.approx(result2.metrics[key])

    def test_different_seed_same_results(
        self, pair_task_spec, pair_ground_truth, perfect_pair_predictions
    ):
        """Test that different seeds don't affect deterministic evaluation."""
        evaluator1 = PairClassificationEvaluator(pair_task_spec, random_seed=42)
        evaluator2 = PairClassificationEvaluator(pair_task_spec, random_seed=999)

        result1 = evaluator1.evaluate(perfect_pair_predictions, pair_ground_truth)
        result2 = evaluator2.evaluate(perfect_pair_predictions, pair_ground_truth)

        # Evaluation should be deterministic regardless of seed
        # (seed is for bootstrap CI, not used in basic evaluate)
        for key in result1.metrics:
            assert result1.metrics[key] == pytest.approx(result2.metrics[key])


# ===========================================================================
# Tests for Result Structure
# ===========================================================================


@pytest.mark.unit
@pytest.mark.evaluator
class TestPairClassificationResultStructure:
    """Tests for result structure and metadata."""

    def test_result_has_task_name(
        self, pair_task_spec, pair_ground_truth, perfect_pair_predictions
    ):
        """Test that result contains task name."""
        evaluator = PairClassificationEvaluator(pair_task_spec)
        result = evaluator.evaluate(perfect_pair_predictions, pair_ground_truth)

        assert result.task == "test_pair_classification"

    def test_result_has_task_type(
        self, pair_task_spec, pair_ground_truth, perfect_pair_predictions
    ):
        """Test that result contains task type."""
        evaluator = PairClassificationEvaluator(pair_task_spec)
        result = evaluator.evaluate(perfect_pair_predictions, pair_ground_truth)

        assert result.task_type == "pair_classification"

    def test_result_has_primary_metric(
        self, pair_task_spec, pair_ground_truth, perfect_pair_predictions
    ):
        """Test that result contains primary metric name."""
        evaluator = PairClassificationEvaluator(pair_task_spec)
        result = evaluator.evaluate(perfect_pair_predictions, pair_ground_truth)

        assert result.primary_metric == "f1"

    def test_result_has_num_samples(
        self, pair_task_spec, pair_ground_truth, perfect_pair_predictions
    ):
        """Test that result contains number of samples."""
        evaluator = PairClassificationEvaluator(pair_task_spec)
        result = evaluator.evaluate(perfect_pair_predictions, pair_ground_truth)

        assert result.num_samples == 10

    def test_result_has_data_provenance(
        self, pair_task_spec, pair_ground_truth, perfect_pair_predictions
    ):
        """Test that result contains data provenance."""
        evaluator = PairClassificationEvaluator(pair_task_spec)
        result = evaluator.evaluate(perfect_pair_predictions, pair_ground_truth)

        assert result.data_provenance is not None

    def test_result_summary_string(
        self, pair_task_spec, pair_ground_truth, perfect_pair_predictions
    ):
        """Test that result has summary() method."""
        evaluator = PairClassificationEvaluator(pair_task_spec)
        result = evaluator.evaluate(perfect_pair_predictions, pair_ground_truth)

        summary = result.summary()
        assert isinstance(summary, str)
        assert "f1" in summary.lower()
        assert "test_pair_classification" in summary


# ===========================================================================
# Tests for Alternative Text Fields
# ===========================================================================


@pytest.mark.unit
@pytest.mark.evaluator
class TestPairClassificationTextFields:
    """Tests for handling different text field formats."""

    def test_fallback_to_text_fields(self, pair_task_spec):
        """Test fallback to text_a/text_b fields when doc_a_body missing."""
        # Ground truth without doc_a_body/doc_b_body (uses text_a/text_b)
        ground_truth = pl.DataFrame(
            {
                "pair_id": [f"pair_{i:03d}" for i in range(5)],
                "text_a": [f"Text A{i}" for i in range(5)],
                "text_b": [f"Text B{i}" for i in range(5)],
                "label": [1, 1, 1, 0, 0],
            }
        )

        predictions = [
            {"pair_id": f"pair_{i:03d}", "score": 0.9 if i < 3 else 0.1}
            for i in range(5)
        ]

        evaluator = PairClassificationEvaluator(pair_task_spec)
        result = evaluator.evaluate(predictions, ground_truth)

        # Should work with text_a/text_b fields
        assert result.num_samples == 5
        assert result.metrics["f1"] == pytest.approx(1.0)

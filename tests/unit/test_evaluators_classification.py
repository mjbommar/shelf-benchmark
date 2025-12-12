"""Unit tests for ClassificationEvaluator.

Tests cover:
1. ClassificationEvaluator initialization
2. evaluate() method with various scenarios
3. Per-class metrics computation
4. Confusion matrix generation
5. Stratified metrics (if applicable)
6. Bootstrap confidence intervals (compute_ci=True)
7. Error handling for mismatched predictions/ground truth
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from shelf.evaluate.evaluators.classification import ClassificationEvaluator
from shelf.evaluate.results import EvaluationResult
from shelf.evaluate.schemas import ValidationError
from shelf.evaluate.tasks import TaskSpec, TaskType


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def classification_task_spec():
    """Create a sample classification task spec."""
    return TaskSpec(
        name="test_lcc_classification",
        task_type=TaskType.CLASSIFICATION,
        description="Test LCC classification task",
        text_field="text",
        label_field="lcc",
        id_field="id",
        label_space=("A", "B", "C", "D"),
        primary_metric="macro_f1",
        secondary_metrics=("micro_f1", "accuracy", "weighted_f1"),
        dataset_name="test/dataset",
        dataset_config="default",
        default_split="test",
    )


@pytest.fixture
def ground_truth_df():
    """Create ground truth DataFrame."""
    return pl.DataFrame(
        {
            "id": ["doc_001", "doc_002", "doc_003", "doc_004", "doc_005", "doc_006"],
            "text": [
                "Sample text A",
                "Sample text B",
                "Sample text C",
                "Sample text D",
                "Sample text A2",
                "Sample text B2",
            ],
            "lcc": ["A", "B", "C", "D", "A", "B"],
            "form": ["lecture", "map", "essay", "lecture", "map", "essay"],
            "git_commit": ["abc123"] * 6,
            "model": ["gpt-5.1"] * 6,
        }
    )


@pytest.fixture
def ground_truth_df_larger():
    """Create larger ground truth DataFrame for stratification tests."""
    return pl.DataFrame(
        {
            "id": [f"doc_{i:03d}" for i in range(1, 21)],
            "text": [f"Sample text {i}" for i in range(1, 21)],
            "lcc": ["A"] * 5 + ["B"] * 5 + ["C"] * 5 + ["D"] * 5,
            "form": ["lecture", "map", "essay", "prayer", "letter"] * 4,
            "register": ["academic", "casual"] * 10,
            "git_commit": ["abc123"] * 20,
            "model": ["gpt-5.1"] * 20,
        }
    )


@pytest.fixture
def perfect_predictions():
    """Perfect predictions (100% accuracy)."""
    return [
        {"id": "doc_001", "prediction": "A"},
        {"id": "doc_002", "prediction": "B"},
        {"id": "doc_003", "prediction": "C"},
        {"id": "doc_004", "prediction": "D"},
        {"id": "doc_005", "prediction": "A"},
        {"id": "doc_006", "prediction": "B"},
    ]


@pytest.fixture
def partial_predictions():
    """Partial predictions (50% accuracy)."""
    return [
        {"id": "doc_001", "prediction": "A"},  # correct
        {"id": "doc_002", "prediction": "A"},  # wrong
        {"id": "doc_003", "prediction": "C"},  # correct
        {"id": "doc_004", "prediction": "A"},  # wrong
        {"id": "doc_005", "prediction": "A"},  # correct
        {"id": "doc_006", "prediction": "A"},  # wrong
    ]


@pytest.fixture
def predictions_with_confidence():
    """Predictions with confidence scores."""
    return [
        {"id": "doc_001", "prediction": "A", "confidence": 0.95},
        {"id": "doc_002", "prediction": "B", "confidence": 0.87},
        {"id": "doc_003", "prediction": "C", "confidence": 0.72},
        {"id": "doc_004", "prediction": "D", "confidence": 0.91},
        {"id": "doc_005", "prediction": "A", "confidence": 0.88},
        {"id": "doc_006", "prediction": "B", "confidence": 0.79},
    ]


# ===========================================================================
# Initialization Tests
# ===========================================================================


@pytest.mark.unit
@pytest.mark.evaluator
class TestClassificationEvaluatorInit:
    """Test ClassificationEvaluator initialization."""

    def test_init_basic(self, classification_task_spec):
        """Test basic initialization with task spec."""
        evaluator = ClassificationEvaluator(classification_task_spec)

        assert evaluator.task_spec == classification_task_spec
        assert evaluator.random_seed == 42
        assert evaluator.filter_by == {}
        assert evaluator.stratify_by == []

    def test_init_with_random_seed(self, classification_task_spec):
        """Test initialization with custom random seed."""
        evaluator = ClassificationEvaluator(classification_task_spec, random_seed=123)

        assert evaluator.random_seed == 123

    def test_init_stores_task_spec(self, classification_task_spec):
        """Test that task spec is properly stored."""
        evaluator = ClassificationEvaluator(classification_task_spec)

        assert evaluator.task_spec.name == "test_lcc_classification"
        assert evaluator.task_spec.task_type == TaskType.CLASSIFICATION
        assert evaluator.task_spec.primary_metric == "macro_f1"
        assert evaluator.task_spec.label_space == ("A", "B", "C", "D")


# ===========================================================================
# evaluate() Method Tests - Perfect Predictions
# ===========================================================================


@pytest.mark.unit
@pytest.mark.evaluator
class TestClassificationEvaluatorPerfect:
    """Test ClassificationEvaluator with perfect predictions."""

    def test_evaluate_perfect_predictions(
        self, classification_task_spec, ground_truth_df, perfect_predictions
    ):
        """Test evaluate with perfect predictions (100% accuracy)."""
        evaluator = ClassificationEvaluator(classification_task_spec)
        result = evaluator.evaluate(perfect_predictions, ground_truth_df)

        # Check result structure
        assert isinstance(result, EvaluationResult)
        assert result.task == "test_lcc_classification"
        assert result.task_type == "classification"
        assert result.num_samples == 6

        # Check metrics
        assert result.metrics["accuracy"] == 1.0
        assert result.metrics["macro_f1"] == 1.0
        assert result.metrics["micro_f1"] == 1.0
        assert result.metrics["weighted_f1"] == 1.0
        assert result.num_correct == 6

        # Check primary score
        assert result.primary_metric == "macro_f1"
        assert result.primary_score == 1.0

    def test_perfect_predictions_no_misclassified(
        self, classification_task_spec, ground_truth_df, perfect_predictions
    ):
        """Test that perfect predictions have no misclassified IDs."""
        evaluator = ClassificationEvaluator(classification_task_spec)
        result = evaluator.evaluate(perfect_predictions, ground_truth_df)

        assert result.misclassified_ids == []

    def test_perfect_predictions_confusion_matrix(
        self, classification_task_spec, ground_truth_df, perfect_predictions
    ):
        """Test confusion matrix for perfect predictions."""
        evaluator = ClassificationEvaluator(classification_task_spec)
        result = evaluator.evaluate(perfect_predictions, ground_truth_df)

        # Check confusion matrix exists
        assert result.confusion_matrix is not None

        # For perfect predictions, confusion matrix should be diagonal
        # With 2 A's, 2 B's, 1 C, 1 D:
        # Row 0 (A): [2, 0, 0, 0]
        # Row 1 (B): [0, 2, 0, 0]
        # Row 2 (C): [0, 0, 1, 0]
        # Row 3 (D): [0, 0, 0, 1]
        cm = result.confusion_matrix
        assert cm[0][0] == 2  # A predicted as A
        assert cm[1][1] == 2  # B predicted as B
        assert cm[2][2] == 1  # C predicted as C
        assert cm[3][3] == 1  # D predicted as D

        # All off-diagonal elements should be 0
        for i in range(4):
            for j in range(4):
                if i != j:
                    assert cm[i][j] == 0


# ===========================================================================
# evaluate() Method Tests - Partial Predictions
# ===========================================================================


@pytest.mark.unit
@pytest.mark.evaluator
class TestClassificationEvaluatorPartial:
    """Test ClassificationEvaluator with partial predictions."""

    def test_evaluate_partial_predictions(
        self, classification_task_spec, ground_truth_df, partial_predictions
    ):
        """Test evaluate with partial predictions (50% accuracy)."""
        evaluator = ClassificationEvaluator(classification_task_spec)
        result = evaluator.evaluate(partial_predictions, ground_truth_df)

        # Check metrics
        assert result.metrics["accuracy"] == 0.5
        assert result.num_correct == 3

        # Macro F1 should be between 0 and 1
        assert 0.0 <= result.metrics["macro_f1"] <= 1.0

    def test_partial_predictions_misclassified_ids(
        self, classification_task_spec, ground_truth_df, partial_predictions
    ):
        """Test that misclassified IDs are correctly identified."""
        evaluator = ClassificationEvaluator(classification_task_spec)
        result = evaluator.evaluate(partial_predictions, ground_truth_df)

        # We have 3 wrong predictions: doc_002, doc_004, doc_006
        assert result.misclassified_ids is not None
        assert len(result.misclassified_ids) == 3
        assert "doc_002" in result.misclassified_ids
        assert "doc_004" in result.misclassified_ids
        assert "doc_006" in result.misclassified_ids

    def test_partial_predictions_per_class_metrics(
        self, classification_task_spec, ground_truth_df, partial_predictions
    ):
        """Test per-class metrics computation."""
        evaluator = ClassificationEvaluator(classification_task_spec)
        result = evaluator.evaluate(partial_predictions, ground_truth_df)

        # Check per-class metrics exist
        assert result.per_class_metrics is not None

        # All labels should be present
        assert "A" in result.per_class_metrics
        assert "B" in result.per_class_metrics
        assert "C" in result.per_class_metrics
        assert "D" in result.per_class_metrics

        # Each class should have precision, recall, f1, support
        for label in ["A", "B", "C", "D"]:
            metrics = result.per_class_metrics[label]
            assert "precision" in metrics
            assert "recall" in metrics
            assert "f1" in metrics
            assert "support" in metrics

            # All metrics should be in [0, 1]
            assert 0.0 <= metrics["precision"] <= 1.0
            assert 0.0 <= metrics["recall"] <= 1.0
            assert 0.0 <= metrics["f1"] <= 1.0
            assert metrics["support"] >= 0

    def test_partial_predictions_confusion_matrix(
        self, classification_task_spec, ground_truth_df, partial_predictions
    ):
        """Test confusion matrix for partial predictions."""
        evaluator = ClassificationEvaluator(classification_task_spec)
        result = evaluator.evaluate(partial_predictions, ground_truth_df)

        # Check confusion matrix exists and has correct shape
        assert result.confusion_matrix is not None
        assert len(result.confusion_matrix) == 4  # 4 classes
        assert all(len(row) == 4 for row in result.confusion_matrix)

        # Total should equal number of samples
        total = sum(sum(row) for row in result.confusion_matrix)
        assert total == 6


# ===========================================================================
# evaluate() Method Tests - Edge Cases
# ===========================================================================


@pytest.mark.unit
@pytest.mark.evaluator
class TestClassificationEvaluatorEdgeCases:
    """Test ClassificationEvaluator edge cases."""

    def test_evaluate_with_confidence_scores(
        self, classification_task_spec, ground_truth_df, predictions_with_confidence
    ):
        """Test that confidence scores are accepted but don't affect metrics."""
        evaluator = ClassificationEvaluator(classification_task_spec)
        result = evaluator.evaluate(predictions_with_confidence, ground_truth_df)

        # Should work fine and give same results as perfect predictions
        assert result.metrics["accuracy"] == 1.0
        assert result.num_correct == 6

    def test_evaluate_missing_prediction_raises_validation_error(
        self, classification_task_spec, ground_truth_df
    ):
        """Test that missing predictions raise ValidationError."""
        evaluator = ClassificationEvaluator(classification_task_spec)

        # Only 4 predictions out of 6
        incomplete_predictions = [
            {"id": "doc_001", "prediction": "A"},
            {"id": "doc_002", "prediction": "B"},
            {"id": "doc_003", "prediction": "C"},
            {"id": "doc_004", "prediction": "D"},
        ]

        with pytest.raises(ValidationError) as exc_info:
            evaluator.evaluate(incomplete_predictions, ground_truth_df)

        # Check error message mentions missing predictions
        assert "Missing predictions" in str(exc_info.value)

    def test_evaluate_duplicate_id_raises_validation_error(
        self, classification_task_spec, ground_truth_df
    ):
        """Test that duplicate IDs raise ValidationError."""
        evaluator = ClassificationEvaluator(classification_task_spec)

        # Duplicate doc_001
        duplicate_predictions = [
            {"id": "doc_001", "prediction": "A"},
            {"id": "doc_001", "prediction": "B"},  # duplicate!
            {"id": "doc_002", "prediction": "B"},
            {"id": "doc_003", "prediction": "C"},
            {"id": "doc_004", "prediction": "D"},
            {"id": "doc_005", "prediction": "A"},
            {"id": "doc_006", "prediction": "B"},
        ]

        with pytest.raises(ValidationError) as exc_info:
            evaluator.evaluate(duplicate_predictions, ground_truth_df)

        assert "duplicate" in str(exc_info.value).lower()

    def test_evaluate_unknown_id_raises_validation_error(
        self, classification_task_spec, ground_truth_df
    ):
        """Test that unknown document IDs raise ValidationError."""
        evaluator = ClassificationEvaluator(classification_task_spec)

        # Unknown ID
        unknown_id_predictions = [
            {"id": "doc_001", "prediction": "A"},
            {"id": "doc_002", "prediction": "B"},
            {"id": "doc_003", "prediction": "C"},
            {"id": "doc_004", "prediction": "D"},
            {"id": "doc_005", "prediction": "A"},
            {"id": "doc_999", "prediction": "B"},  # unknown!
        ]

        with pytest.raises(ValidationError) as exc_info:
            evaluator.evaluate(unknown_id_predictions, ground_truth_df)

        assert "unknown" in str(exc_info.value).lower()

    def test_evaluate_invalid_label_raises_validation_error(
        self, classification_task_spec, ground_truth_df
    ):
        """Test that invalid labels raise ValidationError."""
        evaluator = ClassificationEvaluator(classification_task_spec)

        # Invalid label "Z" (not in label space)
        invalid_label_predictions = [
            {"id": "doc_001", "prediction": "A"},
            {"id": "doc_002", "prediction": "B"},
            {"id": "doc_003", "prediction": "Z"},  # invalid!
            {"id": "doc_004", "prediction": "D"},
            {"id": "doc_005", "prediction": "A"},
            {"id": "doc_006", "prediction": "B"},
        ]

        with pytest.raises(ValidationError) as exc_info:
            evaluator.evaluate(invalid_label_predictions, ground_truth_df)

        assert "invalid label" in str(exc_info.value).lower()

    def test_evaluate_single_class(self, classification_task_spec):
        """Test evaluation with only one class."""
        # Create ground truth with only one class
        single_class_df = pl.DataFrame(
            {
                "id": ["doc_001", "doc_002", "doc_003"],
                "text": ["Text 1", "Text 2", "Text 3"],
                "lcc": ["A", "A", "A"],
            }
        )

        predictions = [
            {"id": "doc_001", "prediction": "A"},
            {"id": "doc_002", "prediction": "A"},
            {"id": "doc_003", "prediction": "A"},
        ]

        evaluator = ClassificationEvaluator(classification_task_spec)
        result = evaluator.evaluate(predictions, single_class_df)

        # Should work and have perfect accuracy
        assert result.metrics["accuracy"] == 1.0
        assert result.num_correct == 3

    def test_evaluate_all_wrong(self, classification_task_spec, ground_truth_df):
        """Test evaluation where all predictions are wrong."""
        # All predictions are wrong
        all_wrong_predictions = [
            {"id": "doc_001", "prediction": "B"},  # true: A
            {"id": "doc_002", "prediction": "A"},  # true: B
            {"id": "doc_003", "prediction": "A"},  # true: C
            {"id": "doc_004", "prediction": "A"},  # true: D
            {"id": "doc_005", "prediction": "B"},  # true: A
            {"id": "doc_006", "prediction": "A"},  # true: B
        ]

        evaluator = ClassificationEvaluator(classification_task_spec)
        result = evaluator.evaluate(all_wrong_predictions, ground_truth_df)

        assert result.metrics["accuracy"] == 0.0
        assert result.num_correct == 0
        assert len(result.misclassified_ids) == 6

    def test_evaluate_misclassified_ids_limited_to_100(self, classification_task_spec):
        """Test that misclassified IDs are limited to first 100."""
        # Create a large dataset with 150 documents, all misclassified
        large_df = pl.DataFrame(
            {
                "id": [f"doc_{i:03d}" for i in range(150)],
                "text": [f"Text {i}" for i in range(150)],
                "lcc": ["A"] * 150,
            }
        )

        # All wrong predictions
        predictions = [{"id": f"doc_{i:03d}", "prediction": "B"} for i in range(150)]

        evaluator = ClassificationEvaluator(classification_task_spec)
        result = evaluator.evaluate(predictions, large_df)

        # Should have exactly 100 misclassified IDs (limited)
        assert len(result.misclassified_ids) == 100


# ===========================================================================
# Provenance and Context Tests
# ===========================================================================


@pytest.mark.unit
@pytest.mark.evaluator
class TestClassificationEvaluatorProvenance:
    """Test ClassificationEvaluator provenance tracking."""

    def test_evaluate_includes_context(
        self, classification_task_spec, ground_truth_df, perfect_predictions
    ):
        """Test that evaluation result includes context."""
        evaluator = ClassificationEvaluator(classification_task_spec)
        result = evaluator.evaluate(perfect_predictions, ground_truth_df)

        # Check context exists
        assert result.context is not None
        assert result.context.random_seed == 42
        assert result.context.dataset_checksum is not None

    def test_evaluate_includes_provenance(
        self, classification_task_spec, ground_truth_df, perfect_predictions
    ):
        """Test that evaluation result includes data provenance."""
        evaluator = ClassificationEvaluator(classification_task_spec)
        result = evaluator.evaluate(perfect_predictions, ground_truth_df)

        # Check provenance exists
        assert result.data_provenance is not None
        assert result.data_provenance.unique_commits == ["abc123"]
        assert result.data_provenance.unique_models == ["gpt-5.1"]


# ===========================================================================
# Stratified Metrics Tests
# ===========================================================================


@pytest.mark.unit
@pytest.mark.evaluator
class TestClassificationEvaluatorStratified:
    """Test ClassificationEvaluator stratified metrics (future feature)."""

    def test_evaluate_without_stratification(
        self, classification_task_spec, ground_truth_df, perfect_predictions
    ):
        """Test that stratified metrics are None when not requested."""
        evaluator = ClassificationEvaluator(classification_task_spec)
        result = evaluator.evaluate(perfect_predictions, ground_truth_df)

        # No stratification requested
        assert result.stratified_metrics is None
        assert result.stratify_by is None


# ===========================================================================
# Bootstrap CI Tests (Future Feature)
# ===========================================================================


@pytest.mark.unit
@pytest.mark.evaluator
class TestClassificationEvaluatorBootstrap:
    """Test ClassificationEvaluator bootstrap confidence intervals."""

    def test_evaluate_compute_ci_false(
        self, classification_task_spec, ground_truth_df, perfect_predictions
    ):
        """Test that CIs are None when compute_ci=False."""
        evaluator = ClassificationEvaluator(classification_task_spec)
        result = evaluator.evaluate(
            perfect_predictions, ground_truth_df, compute_ci=False
        )

        # CIs should not be computed
        assert result.confidence_intervals is None

    def test_evaluate_compute_ci_true_not_implemented(
        self, classification_task_spec, ground_truth_df, perfect_predictions
    ):
        """Test that compute_ci=True parameter is accepted (implementation TBD)."""
        evaluator = ClassificationEvaluator(classification_task_spec)

        # Should not raise an error even though CIs aren't implemented yet
        result = evaluator.evaluate(
            perfect_predictions, ground_truth_df, compute_ci=True
        )

        # For now, CIs are still None (not implemented)
        # This test documents the expected behavior
        assert result.confidence_intervals is None


# ===========================================================================
# evaluate_classifier() Method Tests
# ===========================================================================


@pytest.mark.unit
@pytest.mark.evaluator
class TestClassificationEvaluatorClassifier:
    """Test ClassificationEvaluator.evaluate_classifier() method."""

    def test_evaluate_classifier_perfect(
        self, classification_task_spec, ground_truth_df
    ):
        """Test evaluate_classifier with a perfect mock classifier."""
        # Mock classifier that returns ground truth
        mock_classifier = MagicMock()
        mock_classifier.predict.return_value = ["A", "B", "C", "D", "A", "B"]
        mock_classifier.model_name = "MockPerfectClassifier"

        evaluator = ClassificationEvaluator(classification_task_spec)

        # Mock _load_ground_truth to return our test data
        with patch.object(
            evaluator, "_load_ground_truth", return_value=ground_truth_df
        ):
            result = evaluator.evaluate_classifier(mock_classifier, split="test")

        # Check that classifier was called correctly
        mock_classifier.predict.assert_called_once()
        call_args = mock_classifier.predict.call_args
        assert len(call_args[0][0]) == 6  # 6 texts

        # Check results
        assert result.metrics["accuracy"] == 1.0
        assert result.num_correct == 6

    def test_evaluate_classifier_partial(
        self, classification_task_spec, ground_truth_df
    ):
        """Test evaluate_classifier with partial accuracy."""
        # Mock classifier with 50% accuracy
        mock_classifier = MagicMock()
        mock_classifier.predict.return_value = ["A", "A", "C", "A", "A", "A"]
        mock_classifier.model_name = "MockPartialClassifier"

        evaluator = ClassificationEvaluator(classification_task_spec)

        with patch.object(
            evaluator, "_load_ground_truth", return_value=ground_truth_df
        ):
            result = evaluator.evaluate_classifier(mock_classifier, split="test")

        assert result.metrics["accuracy"] == 0.5
        assert result.num_correct == 3

    def test_evaluate_classifier_wrong_prediction_count_raises(
        self, classification_task_spec, ground_truth_df
    ):
        """Test that mismatched prediction count raises ValueError."""
        # Mock classifier that returns wrong number of predictions
        mock_classifier = MagicMock()
        mock_classifier.predict.return_value = ["A", "B"]  # Only 2 instead of 6

        evaluator = ClassificationEvaluator(classification_task_spec)

        with patch.object(
            evaluator, "_load_ground_truth", return_value=ground_truth_df
        ):
            with pytest.raises(ValueError) as exc_info:
                evaluator.evaluate_classifier(mock_classifier, split="test")

            assert "returned 2 predictions for 6 documents" in str(exc_info.value)

    def test_evaluate_classifier_uses_batch_size(
        self, classification_task_spec, ground_truth_df
    ):
        """Test that batch_size parameter is passed to classifier."""
        mock_classifier = MagicMock()
        mock_classifier.predict.return_value = ["A", "B", "C", "D", "A", "B"]

        evaluator = ClassificationEvaluator(classification_task_spec)

        with patch.object(
            evaluator, "_load_ground_truth", return_value=ground_truth_df
        ):
            evaluator.evaluate_classifier(mock_classifier, split="test", batch_size=16)

        # Check batch_size was passed
        call_kwargs = mock_classifier.predict.call_args[1]
        assert call_kwargs["batch_size"] == 16


# ===========================================================================
# evaluate_embedder_with_classifier() Method Tests
# ===========================================================================


@pytest.mark.unit
@pytest.mark.evaluator
class TestClassificationEvaluatorEmbedder:
    """Test ClassificationEvaluator.evaluate_embedder_with_classifier() method."""

    def test_evaluate_embedder_with_classifier(
        self, classification_task_spec, ground_truth_df
    ):
        """Test evaluate_embedder_with_classifier with mock embedder."""
        import numpy as np

        # Mock embedder
        mock_embedder = MagicMock()
        mock_embedder.model_name = "MockEmbedder"
        mock_embedder.embedding_dim = 128

        # Return random embeddings
        def mock_encode(texts, batch_size=32, show_progress=True):
            return np.random.randn(len(texts), 128)

        mock_embedder.encode = mock_encode

        evaluator = ClassificationEvaluator(classification_task_spec)

        # Mock data loading
        with patch.object(
            evaluator,
            "_load_ground_truth",
            side_effect=[ground_truth_df, ground_truth_df],
        ):
            result = evaluator.evaluate_embedder_with_classifier(
                mock_embedder, split="test", train_split="train"
            )

        # Check result
        assert isinstance(result, EvaluationResult)
        assert result.context.extra["embedding_dim"] == 128
        assert result.context.extra["classifier"] == "LogisticRegression"
        assert result.context.extra["train_size"] == 6

    def test_evaluate_embedder_wrong_train_embedding_count_raises(
        self, classification_task_spec, ground_truth_df
    ):
        """Test that mismatched train embedding count raises ValueError."""
        import numpy as np

        mock_embedder = MagicMock()
        mock_embedder.model_name = "MockEmbedder"
        mock_embedder.embedding_dim = 128

        # Return wrong number of embeddings for train
        def mock_encode(texts, batch_size=32, show_progress=True):
            # Return only 2 embeddings instead of 6
            return np.random.randn(2, 128)

        mock_embedder.encode = mock_encode

        evaluator = ClassificationEvaluator(classification_task_spec)

        with patch.object(
            evaluator,
            "_load_ground_truth",
            side_effect=[ground_truth_df, ground_truth_df],
        ):
            with pytest.raises(ValueError) as exc_info:
                evaluator.evaluate_embedder_with_classifier(
                    mock_embedder, split="test", train_split="train"
                )

            assert "returned 2 train embeddings for 6 texts" in str(exc_info.value)

    def test_evaluate_embedder_wrong_test_embedding_count_raises(
        self, classification_task_spec, ground_truth_df
    ):
        """Test that mismatched test embedding count raises ValueError."""
        import numpy as np

        mock_embedder = MagicMock()
        mock_embedder.model_name = "MockEmbedder"
        mock_embedder.embedding_dim = 128

        call_count = [0]

        def mock_encode(texts, batch_size=32, show_progress=True):
            call_count[0] += 1
            if call_count[0] == 1:  # train
                return np.random.randn(len(texts), 128)
            else:  # test - wrong size
                return np.random.randn(2, 128)  # Only 2 instead of 6

        mock_embedder.encode = mock_encode

        evaluator = ClassificationEvaluator(classification_task_spec)

        with patch.object(
            evaluator,
            "_load_ground_truth",
            side_effect=[ground_truth_df, ground_truth_df],
        ):
            with pytest.raises(ValueError) as exc_info:
                evaluator.evaluate_embedder_with_classifier(
                    mock_embedder, split="test", train_split="train"
                )

            assert "returned 2 test embeddings for 6 texts" in str(exc_info.value)


# ===========================================================================
# Integration Tests
# ===========================================================================


@pytest.mark.unit
@pytest.mark.evaluator
class TestClassificationEvaluatorIntegration:
    """Integration tests for ClassificationEvaluator."""

    def test_end_to_end_perfect(
        self, classification_task_spec, ground_truth_df, perfect_predictions
    ):
        """Test complete end-to-end evaluation with perfect predictions."""
        evaluator = ClassificationEvaluator(classification_task_spec, random_seed=42)
        result = evaluator.evaluate(perfect_predictions, ground_truth_df)

        # Verify all components
        assert result.task == "test_lcc_classification"
        assert result.task_type == "classification"
        assert result.num_samples == 6
        assert result.num_correct == 6
        assert result.metrics["accuracy"] == 1.0
        assert result.metrics["macro_f1"] == 1.0
        assert result.per_class_metrics is not None
        assert result.confusion_matrix is not None
        assert result.data_provenance is not None
        assert result.context is not None

    def test_end_to_end_partial(
        self, classification_task_spec, ground_truth_df, partial_predictions
    ):
        """Test complete end-to-end evaluation with partial predictions."""
        evaluator = ClassificationEvaluator(classification_task_spec, random_seed=42)
        result = evaluator.evaluate(partial_predictions, ground_truth_df)

        # Verify all components
        assert result.task == "test_lcc_classification"
        assert result.num_samples == 6
        assert result.num_correct == 3
        assert result.metrics["accuracy"] == 0.5
        assert len(result.misclassified_ids) == 3
        assert result.per_class_metrics is not None
        assert result.confusion_matrix is not None

    def test_result_serialization(
        self, classification_task_spec, ground_truth_df, perfect_predictions, tmp_path
    ):
        """Test that results can be serialized to JSON."""
        evaluator = ClassificationEvaluator(classification_task_spec)
        result = evaluator.evaluate(perfect_predictions, ground_truth_df)

        # Save to JSON
        json_path = tmp_path / "result.json"
        result.to_json(json_path)

        # Verify file exists
        assert json_path.exists()

        # Load back
        loaded_result = EvaluationResult.from_json(json_path)
        assert loaded_result.task == result.task
        assert loaded_result.metrics["accuracy"] == result.metrics["accuracy"]

    def test_result_to_dict(
        self, classification_task_spec, ground_truth_df, perfect_predictions
    ):
        """Test that results can be converted to dict."""
        evaluator = ClassificationEvaluator(classification_task_spec)
        result = evaluator.evaluate(perfect_predictions, ground_truth_df)

        # Convert to dict
        result_dict = result.to_dict()

        # Verify structure
        assert result_dict["task"] == "test_lcc_classification"
        assert result_dict["metrics"]["accuracy"] == 1.0
        assert "per_class_metrics" in result_dict
        assert "confusion_matrix" in result_dict
        assert "context" in result_dict
        assert "data_provenance" in result_dict

    def test_result_summary(
        self, classification_task_spec, ground_truth_df, perfect_predictions
    ):
        """Test that result summary can be generated."""
        evaluator = ClassificationEvaluator(classification_task_spec)
        result = evaluator.evaluate(perfect_predictions, ground_truth_df)

        # Get summary
        summary = result.summary()

        # Verify summary contains key info
        assert "test_lcc_classification" in summary
        assert "Samples: 6" in summary
        assert "accuracy: 1.0000" in summary
        assert "macro_f1: 1.0000" in summary


# ===========================================================================
# Label Space Tests
# ===========================================================================


@pytest.mark.unit
@pytest.mark.evaluator
class TestClassificationEvaluatorLabelSpace:
    """Test ClassificationEvaluator with different label spaces."""

    def test_evaluate_with_explicit_label_space(
        self, classification_task_spec, ground_truth_df, perfect_predictions
    ):
        """Test that explicit label space is used."""
        evaluator = ClassificationEvaluator(classification_task_spec)
        result = evaluator.evaluate(perfect_predictions, ground_truth_df)

        # Should have 4 classes from label_space
        assert result.metrics["num_classes"] == 4

    def test_evaluate_with_no_label_space(self, ground_truth_df, perfect_predictions):
        """Test evaluation when task spec has no label space."""
        # Create task spec without label space
        task_spec = TaskSpec(
            name="test_open_vocab",
            task_type=TaskType.CLASSIFICATION,
            description="Test open vocabulary task",
            text_field="text",
            label_field="lcc",
            id_field="id",
            label_space=None,  # No label space
            primary_metric="macro_f1",
            secondary_metrics=("micro_f1", "accuracy"),
            dataset_name="test/dataset",
            dataset_config="default",
            default_split="test",
        )

        evaluator = ClassificationEvaluator(task_spec)
        result = evaluator.evaluate(perfect_predictions, ground_truth_df)

        # Should infer labels from data (4 unique labels: A, B, C, D)
        assert result.metrics["num_classes"] == 4
        assert result.metrics["accuracy"] == 1.0

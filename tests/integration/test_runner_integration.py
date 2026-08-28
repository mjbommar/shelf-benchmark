"""Integration tests for shelf.evaluate.runner module.

These tests exercise real code paths without heavy mocking:
- Real evaluator instantiation and selection
- Schema validation
- Checksum wiring and context serialization
- File I/O for predictions and results
- Task discovery

Marked with @pytest.mark.integration for separate execution.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

# `_create_evaluator` is a private helper defined in the legacy flat
# `runner.py` module, loaded by the `shelf.evaluate.runner` package into
# `sys.modules["shelf.evaluate._runner_impl"]`. The package's curated
# `__all__` does not re-export private helpers, so import it from the
# real implementation module instead.
from shelf.evaluate._runner_impl import _create_evaluator
from shelf.evaluate.evaluators.classification import ClassificationEvaluator
from shelf.evaluate.evaluators.clustering import ClusteringEvaluator
from shelf.evaluate.evaluators.retrieval import RetrievalEvaluator
from shelf.evaluate.registry import get_task, list_tasks
from shelf.evaluate.results import EvaluationResult
from shelf.evaluate.runner import evaluate_all
from shelf.evaluate.tasks import TaskType

# ===========================================================================
# Integration Tests for evaluate() with Real Evaluators
# ===========================================================================


@pytest.mark.integration
class TestEvaluateIntegration:
    """Integration tests for evaluate() with real evaluators (minimal mocking)."""

    def test_evaluate_classification_with_predictions_list(
        self, small_classification_dataset, classification_predictions
    ):
        """Test evaluate() with real ClassificationEvaluator and prediction list.

        Only mocks the data loading, not the evaluation logic.
        """
        task_spec = get_task("lcc_classification")
        evaluator = ClassificationEvaluator(task_spec)

        # Patch only the data loading to use our fixture
        with patch.object(
            evaluator, "_load_ground_truth", return_value=small_classification_dataset
        ):
            result = evaluator.evaluate(
                classification_predictions, small_classification_dataset
            )

        # Verify real metrics were computed
        assert isinstance(result, EvaluationResult)
        assert result.task == "lcc_classification"
        assert result.task_type == "classification"
        assert "macro_f1" in result.metrics
        assert "micro_f1" in result.metrics
        assert "accuracy" in result.metrics
        assert 0 <= result.primary_score <= 1
        assert result.num_samples == 20

        # Verify per-class metrics exist
        assert result.per_class_metrics is not None
        assert len(result.per_class_metrics) > 0

        # Verify context was captured
        assert result.context is not None
        assert result.context.shelf_version is not None
        assert result.context.python_version is not None
        assert result.context.sklearn_version is not None

    def test_evaluate_classification_with_predictions_file(
        self, small_classification_dataset, predictions_jsonl_file
    ):
        """Test evaluate() with prediction file path (tests checksum wiring)."""
        task_spec = get_task("lcc_classification")
        evaluator = ClassificationEvaluator(task_spec)

        # Use evaluate_from_file which handles file loading and checksum
        with patch.object(
            evaluator, "_load_ground_truth", return_value=small_classification_dataset
        ):
            result = evaluator.evaluate_from_file(predictions_jsonl_file, split="test")

        # Verify result
        assert isinstance(result, EvaluationResult)
        assert result.num_samples == 20

        # Verify prediction file checksum was set
        assert result.context is not None
        assert result.context.prediction_file_checksum is not None
        assert len(result.context.prediction_file_checksum) == 32  # MD5 hex digest

    def test_evaluate_retrieval_with_predictions(self, small_retrieval_dataset):
        """Test evaluate() with real RetrievalEvaluator."""
        task_spec = get_task("lcc_retrieval")
        evaluator = RetrievalEvaluator(task_spec)

        # Create predictions for all documents (each doc is a query)
        doc_ids = small_retrieval_dataset["id"].to_list()
        retrieval_preds = []
        for query_id in doc_ids:
            # Rank all other docs (exclude self)
            ranked = [d for d in doc_ids if d != query_id]
            retrieval_preds.append(
                {
                    "query_id": query_id,
                    "ranked_doc_ids": ranked[:5],  # Top 5
                }
            )

        with patch.object(
            evaluator, "_load_ground_truth", return_value=small_retrieval_dataset
        ):
            result = evaluator.evaluate(retrieval_preds, small_retrieval_dataset)

        # Verify real retrieval metrics
        assert isinstance(result, EvaluationResult)
        assert result.task == "lcc_retrieval"
        assert result.task_type == "retrieval"
        assert "ndcg@10" in result.metrics or "ndcg_at_10" in result.metrics
        assert result.num_samples > 0

    def test_evaluate_result_serialization(
        self, small_classification_dataset, classification_predictions, output_dir
    ):
        """Test that results serialize correctly to JSON."""
        task_spec = get_task("lcc_classification")
        evaluator = ClassificationEvaluator(task_spec)

        with patch.object(
            evaluator, "_load_ground_truth", return_value=small_classification_dataset
        ):
            result = evaluator.evaluate(
                classification_predictions, small_classification_dataset
            )

        # Save to file
        output_file = output_dir / "results.json"
        result.to_json(output_file)

        # Verify file exists and is valid JSON
        assert output_file.exists()
        with open(output_file) as f:
            data = json.load(f)

        # Verify key fields
        assert data["task"] == "lcc_classification"
        assert data["task_type"] == "classification"
        assert "metrics" in data
        assert "primary_score" in data
        assert "num_samples" in data

        # Verify context was serialized
        assert "context" in data
        assert data["context"]["shelf_version"] is not None

        # Verify can deserialize back
        loaded_result = EvaluationResult.from_json(output_file)
        assert loaded_result.task == result.task
        assert loaded_result.primary_score == result.primary_score

    def test_evaluate_computes_correct_accuracy(
        self, small_classification_dataset, classification_predictions
    ):
        """Test that evaluation computes expected accuracy.

        Our fixture has 16/20 correct predictions = 80% accuracy.
        """
        task_spec = get_task("lcc_classification")
        evaluator = ClassificationEvaluator(task_spec)

        with patch.object(
            evaluator, "_load_ground_truth", return_value=small_classification_dataset
        ):
            result = evaluator.evaluate(
                classification_predictions, small_classification_dataset
            )

        # We have 4 wrong predictions out of 20
        expected_accuracy = 16 / 20
        assert result.metrics["accuracy"] == pytest.approx(expected_accuracy, rel=0.01)


@pytest.mark.integration
class TestCreateEvaluatorIntegration:
    """Integration tests for _create_evaluator() with real evaluator classes."""

    def test_creates_correct_evaluator_types(self):
        """Test that _create_evaluator returns correct evaluator classes."""
        # Classification
        task_spec = get_task("lcc_classification")
        evaluator = _create_evaluator(task_spec)
        assert isinstance(evaluator, ClassificationEvaluator)

        # Retrieval
        task_spec = get_task("lcc_retrieval")
        evaluator = _create_evaluator(task_spec)
        assert isinstance(evaluator, RetrievalEvaluator)

        # Clustering
        task_spec = get_task("lcc_clustering")
        evaluator = _create_evaluator(task_spec)
        assert isinstance(evaluator, ClusteringEvaluator)

    def test_evaluator_inherits_task_spec(self):
        """Test that created evaluator has correct task spec."""
        task_spec = get_task("lcgft_category_classification")
        evaluator = _create_evaluator(task_spec)

        assert evaluator.task_spec == task_spec
        assert evaluator.task_spec.name == "lcgft_category_classification"
        assert evaluator.task_spec.task_type == TaskType.CLASSIFICATION


@pytest.mark.integration
class TestEvaluateAllIntegration:
    """Integration tests for evaluate_all() with real task discovery."""

    def test_evaluate_all_discovers_tasks(self, mock_embedder):
        """Test that evaluate_all discovers tasks from registry."""
        # Get expected tasks
        retrieval_tasks = list_tasks(TaskType.RETRIEVAL)
        assert len(retrieval_tasks) > 0

        # Mock evaluate to avoid actual evaluation but verify task discovery
        call_count = {"count": 0}

        def mock_eval(task, **kwargs):
            call_count["count"] += 1
            return EvaluationResult(
                task=task,
                task_type="retrieval",
                split="test",
                primary_metric="ndcg@10",
                primary_score=0.75,
                metrics={"ndcg@10": 0.75},
                num_samples=100,
            )

        with patch("shelf.evaluate._runner_impl.evaluate", side_effect=mock_eval):
            results = evaluate_all(mock_embedder)

        # Verify all retrieval tasks were discovered
        assert set(results.keys()) == set(retrieval_tasks)
        assert call_count["count"] == len(retrieval_tasks)

    def test_evaluate_all_creates_output_files(self, mock_embedder, output_dir):
        """Test that evaluate_all creates output files when output_dir specified."""

        def mock_eval(task, **kwargs):
            return EvaluationResult(
                task=task,
                task_type="retrieval",
                split="test",
                primary_metric="ndcg@10",
                primary_score=0.75,
                metrics={"ndcg@10": 0.75},
                num_samples=100,
            )

        with patch("shelf.evaluate._runner_impl.evaluate", side_effect=mock_eval):
            results = evaluate_all(
                mock_embedder,
                tasks=["lcc_retrieval", "form_retrieval"],
                output_dir=output_dir,
            )

        # Verify files were created
        assert (output_dir / "lcc_retrieval.json").exists()
        assert (output_dir / "form_retrieval.json").exists()

        # Verify file content
        with open(output_dir / "lcc_retrieval.json") as f:
            data = json.load(f)
            assert data["task"] == "lcc_retrieval"

    def test_evaluate_all_filters_by_task_type(self, mock_embedder):
        """Test that evaluate_all correctly filters by task_types."""

        def mock_eval(task, **kwargs):
            task_spec = get_task(task)
            return EvaluationResult(
                task=task,
                task_type=task_spec.task_type.value,
                split="test",
                primary_metric="macro_f1",
                primary_score=0.85,
                metrics={"macro_f1": 0.85},
                num_samples=100,
            )

        with patch("shelf.evaluate._runner_impl.evaluate", side_effect=mock_eval):
            results = evaluate_all(
                mock_embedder,
                task_types=[TaskType.CLASSIFICATION],
            )

        # Verify only classification tasks were evaluated
        classification_tasks = list_tasks(TaskType.CLASSIFICATION)
        assert set(results.keys()) == set(classification_tasks)


@pytest.mark.integration
class TestContextAndChecksum:
    """Integration tests for context capture and checksum computation."""

    def test_context_captures_environment(
        self, small_classification_dataset, classification_predictions
    ):
        """Test that EvaluationContext correctly captures environment."""
        task_spec = get_task("lcc_classification")
        evaluator = ClassificationEvaluator(task_spec)

        with patch.object(
            evaluator, "_load_ground_truth", return_value=small_classification_dataset
        ):
            result = evaluator.evaluate(
                classification_predictions, small_classification_dataset
            )

        ctx = result.context
        assert ctx is not None

        # Verify version strings are populated
        assert ctx.shelf_version is not None
        assert ctx.python_version is not None
        assert ctx.sklearn_version is not None
        assert ctx.numpy_version is not None
        assert ctx.polars_version is not None

        # Verify platform info
        assert ctx.platform_info is not None
        assert len(ctx.platform_info) > 0

        # Verify timestamp is ISO format
        assert ctx.timestamp is not None
        assert "T" in ctx.timestamp  # ISO format has T separator

        # Verify random seed
        assert ctx.random_seed == 42  # Default seed

    def test_prediction_file_checksum_differs_by_content(
        self, small_classification_dataset, tmp_path
    ):
        """Test that different prediction files get different checksums."""
        task_spec = get_task("lcc_classification")
        evaluator = ClassificationEvaluator(task_spec)

        # Create two different prediction files
        preds1 = [{"id": f"doc_{i:03d}", "prediction": "A"} for i in range(20)]
        preds2 = [{"id": f"doc_{i:03d}", "prediction": "B"} for i in range(20)]

        file1 = tmp_path / "preds1.jsonl"
        file2 = tmp_path / "preds2.jsonl"

        for f, preds in [(file1, preds1), (file2, preds2)]:
            with open(f, "w") as fp:
                for p in preds:
                    fp.write(json.dumps(p) + "\n")

        with patch.object(
            evaluator, "_load_ground_truth", return_value=small_classification_dataset
        ):
            result1 = evaluator.evaluate_from_file(file1)
            result2 = evaluator.evaluate_from_file(file2)

        # Checksums should differ
        assert (
            result1.context.prediction_file_checksum
            != result2.context.prediction_file_checksum
        )


@pytest.mark.integration
class TestSchemaValidation:
    """Integration tests for prediction schema validation."""

    def test_invalid_prediction_id_raises(self, small_classification_dataset):
        """Test that invalid prediction format raises validation error."""
        task_spec = get_task("lcc_classification")
        evaluator = ClassificationEvaluator(task_spec)

        # Missing required 'id' field
        invalid_predictions = [{"prediction": "A"}]

        with (
            patch.object(
                evaluator,
                "_load_ground_truth",
                return_value=small_classification_dataset,
            ),
            pytest.raises(Exception),
        ):  # Could be ValidationError or KeyError
            evaluator.evaluate(invalid_predictions, small_classification_dataset)

    def test_mismatched_ids_handled(self, small_classification_dataset):
        """Test handling of predictions with IDs not in ground truth.

        The evaluator should raise a ValidationError when predictions
        reference unknown document IDs.
        """
        from shelf.evaluate.schemas import ValidationError

        task_spec = get_task("lcc_classification")
        evaluator = ClassificationEvaluator(task_spec)

        # Predictions with non-existent IDs
        predictions = [
            {"id": "nonexistent_001", "prediction": "A"},
            {"id": "nonexistent_002", "prediction": "B"},
        ]

        with patch.object(
            evaluator, "_load_ground_truth", return_value=small_classification_dataset
        ):
            # Should raise ValidationError for unknown IDs
            with pytest.raises(ValidationError) as exc_info:
                evaluator.evaluate(predictions, small_classification_dataset)

            # Verify error message mentions unknown IDs
            assert "unknown id" in str(exc_info.value).lower() or "nonexistent" in str(
                exc_info.value
            )

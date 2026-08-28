"""Unit tests for shelf.evaluate.runner module.

Tests cover:
- evaluate() function with predictions file
- evaluate() function with predictions list
- evaluate() function with model (various task types)
- evaluate_all() function
- _create_evaluator() helper
- Error handling for invalid tasks/predictions
- Output file saving
- Prediction file checksum tracking
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

# `_create_evaluator` is a private helper that lives in the legacy flat
# `runner.py` module. The `shelf.evaluate.runner` package (config/context/
# events/orchestrator) intentionally curates its public surface in
# `__all__` and does not re-export private helpers, but it does load the
# flat module into `sys.modules["shelf.evaluate._runner_impl"]` so that
# `evaluate`/`evaluate_all` keep working. Import (and patch) the helper
# from that real implementation module rather than the package shim.
from shelf.evaluate._runner_impl import _create_evaluator
from shelf.evaluate.results import EvaluationContext, EvaluationResult
from shelf.evaluate.runner import evaluate, evaluate_all
from shelf.evaluate.tasks import TaskType

# ===========================================================================
# Mock Models and Evaluators
# ===========================================================================


class MockEmbedder:
    """Mock embedder for testing."""

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return random embeddings."""
        return np.random.randn(len(texts), 384)


class MockClassifier:
    """Mock classifier with predict method."""

    def predict(self, texts: list[str]) -> list[str]:
        """Return random predictions."""
        return ["A"] * len(texts)


class MockRetriever:
    """Mock retriever with retrieve method."""

    def retrieve(self, query: str, top_k: int = 10) -> list[str]:
        """Return random document IDs."""
        return [f"doc_{i}" for i in range(top_k)]


# ===========================================================================
# Helper Functions
# ===========================================================================


def create_temp_predictions_file(predictions: list[dict[str, Any]]) -> Path:
    """Create a temporary JSONL predictions file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")
        return Path(f.name)


def create_mock_result(
    task: str,
    task_type: str = "classification",
    primary_metric: str = "macro_f1",
    primary_score: float = 0.85,
    metrics: dict[str, float] | None = None,
    num_samples: int = 100,
    split: str = "test",
) -> EvaluationResult:
    """Create a mock EvaluationResult for testing."""
    return EvaluationResult(
        task=task,
        task_type=task_type,
        split=split,
        primary_metric=primary_metric,
        primary_score=primary_score,
        metrics=metrics or {primary_metric: primary_score},
        num_samples=num_samples,
    )


# ===========================================================================
# Tests for _create_evaluator()
# ===========================================================================


class TestCreateEvaluator:
    """Tests for _create_evaluator() helper function."""

    @pytest.mark.unit
    def test_create_retrieval_evaluator(self):
        """Test creating RetrievalEvaluator."""
        from shelf.evaluate.evaluators.retrieval import RetrievalEvaluator
        from shelf.evaluate.registry import get_task

        task_spec = get_task("lcc_retrieval")
        evaluator = _create_evaluator(task_spec)
        assert isinstance(evaluator, RetrievalEvaluator)
        assert evaluator.task_spec == task_spec

    @pytest.mark.unit
    def test_create_classification_evaluator(self):
        """Test creating ClassificationEvaluator."""
        from shelf.evaluate.evaluators.classification import ClassificationEvaluator
        from shelf.evaluate.registry import get_task

        task_spec = get_task("lcc_classification")
        evaluator = _create_evaluator(task_spec)
        assert isinstance(evaluator, ClassificationEvaluator)
        assert evaluator.task_spec == task_spec

    @pytest.mark.unit
    def test_create_clustering_evaluator(self):
        """Test creating ClusteringEvaluator."""
        from shelf.evaluate.evaluators.clustering import ClusteringEvaluator
        from shelf.evaluate.registry import get_task

        task_spec = get_task("lcc_clustering")
        evaluator = _create_evaluator(task_spec)
        assert isinstance(evaluator, ClusteringEvaluator)
        assert evaluator.task_spec == task_spec

    @pytest.mark.unit
    def test_create_pair_classification_evaluator(self):
        """Test creating PairClassificationEvaluator."""
        from shelf.evaluate.evaluators.pair import PairClassificationEvaluator
        from shelf.evaluate.registry import get_task

        task_spec = get_task("same_lcc_pairs")
        evaluator = _create_evaluator(task_spec)
        assert isinstance(evaluator, PairClassificationEvaluator)
        assert evaluator.task_spec == task_spec

    @pytest.mark.unit
    def test_create_evaluator_multilabel(self):
        """Multilabel tasks now dispatch to MultiLabelClassificationEvaluator."""
        from shelf.evaluate.tasks import TaskSpec

        task_spec = TaskSpec(
            name="test_multilabel",
            task_type=TaskType.MULTILABEL,
            description="Test multilabel task",
            text_field="text",
            label_field="labels",
            id_field="id",
            label_space=None,
            primary_metric="f1_micro",
            secondary_metrics=(),
            dataset_name="test",
            dataset_config=None,
            default_split="test",
        )

        from shelf.evaluate.evaluators import MultiLabelClassificationEvaluator

        evaluator = _create_evaluator(task_spec)
        assert isinstance(evaluator, MultiLabelClassificationEvaluator)
        assert evaluator.task_spec == task_spec

    @pytest.mark.unit
    def test_create_evaluator_unknown_type_raises(self):
        """Test creating evaluator for unknown task type raises ValueError."""
        from shelf.evaluate.tasks import TaskSpec

        # Create a mock task spec with invalid type
        task_spec = Mock(spec=TaskSpec)
        task_spec.task_type = "invalid_type"

        with pytest.raises(ValueError, match="Unknown task type"):
            _create_evaluator(task_spec)


# ===========================================================================
# Tests for evaluate() with Predictions
# ===========================================================================


class TestEvaluateWithPredictions:
    """Tests for evaluate() function using predictions (not models)."""

    @pytest.mark.unit
    @patch("shelf.evaluate._runner_impl.ClassificationEvaluator")
    def test_evaluate_with_predictions_list(self, mock_evaluator_cls):
        """Test evaluate() with predictions as list of dicts."""
        # Setup mock evaluator
        mock_evaluator = MagicMock()
        mock_result = EvaluationResult(
            task="lcc_classification",
            task_type="classification",
            split="test",
            primary_metric="macro_f1",
            primary_score=0.85,
            metrics={"macro_f1": 0.85, "micro_f1": 0.87},
            num_samples=100,
        )
        mock_evaluator.evaluate.return_value = mock_result
        mock_evaluator._load_ground_truth.return_value = []
        mock_evaluator_cls.return_value = mock_evaluator

        predictions = [
            {"id": "doc_001", "prediction": "A"},
            {"id": "doc_002", "prediction": "B"},
        ]

        result = evaluate("lcc_classification", predictions=predictions)

        assert result.task == "lcc_classification"
        assert result.primary_score == 0.85
        mock_evaluator.evaluate.assert_called_once()

    @pytest.mark.unit
    @patch("shelf.evaluate._runner_impl.ClassificationEvaluator")
    def test_evaluate_with_predictions_file(self, mock_evaluator_cls):
        """Test evaluate() with predictions file path."""
        # Setup mock evaluator
        mock_evaluator = MagicMock()
        mock_result = create_mock_result("lcc_classification")
        mock_evaluator.evaluate.return_value = mock_result
        mock_evaluator._load_ground_truth.return_value = []
        mock_evaluator_cls.return_value = mock_evaluator

        # Create predictions file
        predictions = [
            {"id": "doc_001", "prediction": "A"},
            {"id": "doc_002", "prediction": "B"},
        ]
        pred_file = create_temp_predictions_file(predictions)

        try:
            result = evaluate("lcc_classification", predictions=pred_file)

            assert result.task == "lcc_classification"
            assert result.primary_score == 0.85
            # Note: context and checksum verification is tested in TestPredictionFileChecksum
            mock_evaluator.evaluate.assert_called_once()
        finally:
            pred_file.unlink()

    @pytest.mark.unit
    @patch("shelf.evaluate._runner_impl.ClassificationEvaluator")
    def test_evaluate_with_predictions_string_path(self, mock_evaluator_cls):
        """Test evaluate() with predictions path as string."""
        # Setup mock evaluator
        mock_evaluator = MagicMock()
        mock_result = create_mock_result("lcc_classification")
        mock_evaluator.evaluate.return_value = mock_result
        mock_evaluator._load_ground_truth.return_value = []
        mock_evaluator_cls.return_value = mock_evaluator

        # Create predictions file
        predictions = [{"id": "doc_001", "prediction": "A"}]
        pred_file = create_temp_predictions_file(predictions)

        try:
            # Pass as string (not Path)
            result = evaluate("lcc_classification", predictions=str(pred_file))

            assert result.task == "lcc_classification"
            assert result.primary_score == 0.85
        finally:
            pred_file.unlink()

    @pytest.mark.unit
    def test_evaluate_neither_predictions_nor_model_raises(self):
        """Test evaluate() raises ValueError when neither predictions nor model provided."""
        with pytest.raises(
            ValueError, match="Must provide either predictions or model"
        ):
            evaluate("lcc_classification")


# ===========================================================================
# Tests for evaluate() with Models
# ===========================================================================


class TestEvaluateWithModels:
    """Tests for evaluate() function using models directly."""

    @pytest.mark.unit
    @patch("shelf.evaluate._runner_impl._create_evaluator")
    def test_evaluate_retrieval_with_embedder(self, mock_create_eval):
        """Test evaluate() for retrieval task with embedder model."""
        from shelf.evaluate.evaluators.retrieval import RetrievalEvaluator

        # Setup mock evaluator with correct spec
        mock_evaluator = MagicMock(spec=RetrievalEvaluator)
        mock_result = create_mock_result(
            "lcc_retrieval",
            task_type="retrieval",
            primary_metric="ndcg@10",
            primary_score=0.75,
        )
        mock_evaluator.evaluate_embedder.return_value = mock_result
        mock_create_eval.return_value = mock_evaluator

        embedder = MockEmbedder()
        result = evaluate("lcc_retrieval", model=embedder)

        assert result.task == "lcc_retrieval"
        assert result.primary_score == 0.75
        mock_evaluator.evaluate_embedder.assert_called_once()

    @pytest.mark.unit
    @patch("shelf.evaluate._runner_impl._create_evaluator")
    def test_evaluate_retrieval_with_retriever(self, mock_create_eval):
        """Test evaluate() for retrieval task with retriever model (has retrieve method)."""
        from shelf.evaluate.evaluators.retrieval import RetrievalEvaluator

        # Setup mock evaluator with correct spec
        mock_evaluator = MagicMock(spec=RetrievalEvaluator)
        mock_result = create_mock_result(
            "lcc_retrieval",
            task_type="retrieval",
            primary_metric="ndcg@10",
            primary_score=0.80,
        )
        mock_evaluator.evaluate_retriever.return_value = mock_result
        mock_create_eval.return_value = mock_evaluator

        retriever = MockRetriever()
        result = evaluate("lcc_retrieval", model=retriever)

        assert result.task == "lcc_retrieval"
        assert result.primary_score == 0.80
        mock_evaluator.evaluate_retriever.assert_called_once()

    @pytest.mark.unit
    @patch("shelf.evaluate._runner_impl._create_evaluator")
    def test_evaluate_classification_with_classifier(self, mock_create_eval):
        """Test evaluate() for classification with classifier model (has predict method)."""
        from shelf.evaluate.evaluators.classification import ClassificationEvaluator

        # Setup mock evaluator with correct spec
        mock_evaluator = MagicMock(spec=ClassificationEvaluator)
        mock_result = create_mock_result("lcc_classification", primary_score=0.88)
        mock_evaluator.evaluate_classifier.return_value = mock_result
        mock_create_eval.return_value = mock_evaluator

        classifier = MockClassifier()
        result = evaluate("lcc_classification", model=classifier)

        assert result.task == "lcc_classification"
        assert result.primary_score == 0.88
        mock_evaluator.evaluate_classifier.assert_called_once()

    @pytest.mark.unit
    @patch("shelf.evaluate._runner_impl._create_evaluator")
    def test_evaluate_classification_with_embedder(self, mock_create_eval):
        """Test evaluate() for classification with embedder (trains LogisticRegression)."""
        from shelf.evaluate.evaluators.classification import ClassificationEvaluator

        # Setup mock evaluator with correct spec
        mock_evaluator = MagicMock(spec=ClassificationEvaluator)
        mock_result = create_mock_result("lcc_classification", primary_score=0.82)
        mock_evaluator.evaluate_embedder_with_classifier.return_value = mock_result
        mock_create_eval.return_value = mock_evaluator

        embedder = MockEmbedder()
        result = evaluate("lcc_classification", model=embedder)

        assert result.task == "lcc_classification"
        assert result.primary_score == 0.82
        mock_evaluator.evaluate_embedder_with_classifier.assert_called_once()

    @pytest.mark.unit
    @patch("shelf.evaluate._runner_impl._create_evaluator")
    def test_evaluate_clustering_with_embedder(self, mock_create_eval):
        """Test evaluate() for clustering with embedder."""
        from shelf.evaluate.evaluators.clustering import ClusteringEvaluator

        # Setup mock evaluator with correct spec
        mock_evaluator = MagicMock(spec=ClusteringEvaluator)
        mock_result = create_mock_result(
            "lcc_clustering",
            task_type="clustering",
            primary_metric="v_measure",
            primary_score=0.65,
        )
        mock_evaluator.evaluate_embedder.return_value = mock_result
        mock_create_eval.return_value = mock_evaluator

        embedder = MockEmbedder()
        result = evaluate("lcc_clustering", model=embedder)

        assert result.task == "lcc_clustering"
        assert result.primary_score == 0.65
        mock_evaluator.evaluate_embedder.assert_called_once()

    @pytest.mark.unit
    @patch("shelf.evaluate._runner_impl._create_evaluator")
    def test_evaluate_pair_classification_with_embedder(self, mock_create_eval):
        """Test evaluate() for pair classification with embedder."""
        from shelf.evaluate.evaluators.pair import PairClassificationEvaluator

        # Setup mock evaluator with correct spec
        mock_evaluator = MagicMock(spec=PairClassificationEvaluator)
        mock_result = create_mock_result(
            "same_lcc_pairs",
            task_type="pair_classification",
            primary_metric="f1",
            primary_score=0.78,
        )
        mock_evaluator.evaluate_embedder.return_value = mock_result
        mock_create_eval.return_value = mock_evaluator

        embedder = MockEmbedder()
        result = evaluate("same_lcc_pairs", model=embedder)

        assert result.task == "same_lcc_pairs"
        assert result.primary_score == 0.78
        mock_evaluator.evaluate_embedder.assert_called_once()

    @pytest.mark.unit
    @patch("shelf.evaluate._runner_impl._create_evaluator")
    def test_evaluate_with_model_custom_split(self, mock_create_eval):
        """Test evaluate() with custom split parameter."""
        from shelf.evaluate.evaluators.retrieval import RetrievalEvaluator

        # Setup mock evaluator with correct spec
        mock_evaluator = MagicMock(spec=RetrievalEvaluator)
        mock_result = create_mock_result(
            "lcc_retrieval",
            task_type="retrieval",
            primary_metric="ndcg@10",
            primary_score=0.75,
        )
        mock_evaluator.evaluate_embedder.return_value = mock_result
        mock_create_eval.return_value = mock_evaluator

        embedder = MockEmbedder()
        result = evaluate("lcc_retrieval", model=embedder, split="validation")

        # Verify split was passed to evaluator
        call_kwargs = mock_evaluator.evaluate_embedder.call_args[1]
        assert call_kwargs["split"] == "validation"

    @pytest.mark.unit
    @patch("shelf.evaluate._runner_impl._create_evaluator")
    def test_evaluate_with_model_max_queries(self, mock_create_eval):
        """Test evaluate() with max_queries parameter for retrieval."""
        from shelf.evaluate.evaluators.retrieval import RetrievalEvaluator

        # Setup mock evaluator with correct spec
        mock_evaluator = MagicMock(spec=RetrievalEvaluator)
        mock_result = create_mock_result(
            "lcc_retrieval",
            task_type="retrieval",
            primary_metric="ndcg@10",
            primary_score=0.75,
        )
        mock_evaluator.evaluate_embedder.return_value = mock_result
        mock_create_eval.return_value = mock_evaluator

        embedder = MockEmbedder()
        result = evaluate("lcc_retrieval", model=embedder, max_queries=100)

        # Verify max_queries was passed to evaluator
        call_kwargs = mock_evaluator.evaluate_embedder.call_args[1]
        assert call_kwargs["max_queries"] == 100

    @pytest.mark.unit
    @patch("shelf.evaluate._runner_impl._create_evaluator")
    def test_evaluate_with_model_batch_size(self, mock_create_eval):
        """Test evaluate() with custom batch_size parameter."""
        from shelf.evaluate.evaluators.classification import ClassificationEvaluator

        # Setup mock evaluator with correct spec
        mock_evaluator = MagicMock(spec=ClassificationEvaluator)
        mock_result = create_mock_result("lcc_classification", primary_score=0.82)
        mock_evaluator.evaluate_embedder_with_classifier.return_value = mock_result
        mock_create_eval.return_value = mock_evaluator

        embedder = MockEmbedder()
        result = evaluate("lcc_classification", model=embedder, batch_size=64)

        # Verify batch_size was passed to evaluator
        call_kwargs = mock_evaluator.evaluate_embedder_with_classifier.call_args[1]
        assert call_kwargs["batch_size"] == 64

    @pytest.mark.unit
    @patch("shelf.evaluate._runner_impl._create_evaluator")
    def test_evaluate_with_model_show_progress_false(self, mock_create_eval):
        """Test evaluate() with show_progress=False."""
        from shelf.evaluate.evaluators.retrieval import RetrievalEvaluator

        # Setup mock evaluator with correct spec
        mock_evaluator = MagicMock(spec=RetrievalEvaluator)
        mock_result = create_mock_result(
            "lcc_retrieval",
            task_type="retrieval",
            primary_metric="ndcg@10",
            primary_score=0.75,
        )
        mock_evaluator.evaluate_embedder.return_value = mock_result
        mock_create_eval.return_value = mock_evaluator

        embedder = MockEmbedder()
        result = evaluate("lcc_retrieval", model=embedder, show_progress=False)

        # Verify show_progress was passed to evaluator
        call_kwargs = mock_evaluator.evaluate_embedder.call_args[1]
        assert call_kwargs["show_progress"] is False


# ===========================================================================
# Tests for evaluate() Error Handling
# ===========================================================================


class TestEvaluateErrorHandling:
    """Tests for error handling in evaluate() function."""

    @pytest.mark.unit
    def test_evaluate_invalid_task_raises(self):
        """Test evaluate() raises ValueError for invalid task name."""
        with pytest.raises(ValueError, match="Unknown task"):
            evaluate("nonexistent_task", predictions=[])

    @pytest.mark.unit
    @patch("shelf.evaluate._runner_impl._create_evaluator")
    def test_evaluate_wrong_evaluator_type_raises(self, mock_create_eval):
        """Test evaluate() raises ValueError if evaluator type doesn't match task."""
        # Mock evaluator to return wrong type (not a RetrievalEvaluator)
        mock_evaluator = Mock()
        mock_evaluator.__class__.__name__ = "WrongEvaluator"
        mock_create_eval.return_value = mock_evaluator

        embedder = MockEmbedder()

        with pytest.raises(ValueError, match="Expected RetrievalEvaluator"):
            evaluate("lcc_retrieval", model=embedder)


# ===========================================================================
# Tests for evaluate() Output Saving
# ===========================================================================


class TestEvaluateOutputSaving:
    """Tests for output_path parameter in evaluate()."""

    @pytest.mark.unit
    @patch("shelf.evaluate._runner_impl.ClassificationEvaluator")
    def test_evaluate_saves_output_json(self, mock_evaluator_cls, tmp_path):
        """Test evaluate() saves results to JSON when output_path provided."""
        # Setup mock evaluator
        mock_evaluator = MagicMock()
        mock_result = create_mock_result("lcc_classification")
        mock_evaluator.evaluate.return_value = mock_result
        mock_evaluator._load_ground_truth.return_value = []
        mock_evaluator_cls.return_value = mock_evaluator

        predictions = [{"id": "doc_001", "prediction": "A"}]
        output_path = tmp_path / "results.json"

        result = evaluate(
            "lcc_classification", predictions=predictions, output_path=output_path
        )

        # Verify file was created
        assert output_path.exists()

        # Verify content
        with open(output_path) as f:
            data = json.load(f)
            assert data["task"] == "lcc_classification"
            assert data["primary_score"] == 0.85

    @pytest.mark.unit
    @patch("shelf.evaluate._runner_impl.ClassificationEvaluator")
    def test_evaluate_output_path_as_string(self, mock_evaluator_cls, tmp_path):
        """Test evaluate() accepts output_path as string."""
        # Setup mock evaluator
        mock_evaluator = MagicMock()
        mock_result = create_mock_result("lcc_classification")
        mock_evaluator.evaluate.return_value = mock_result
        mock_evaluator._load_ground_truth.return_value = []
        mock_evaluator_cls.return_value = mock_evaluator

        predictions = [{"id": "doc_001", "prediction": "A"}]
        output_path = str(tmp_path / "results.json")

        result = evaluate(
            "lcc_classification", predictions=predictions, output_path=output_path
        )

        # Verify file was created
        assert Path(output_path).exists()


# ===========================================================================
# Tests for evaluate_all()
# ===========================================================================


class TestEvaluateAll:
    """Tests for evaluate_all() function."""

    @pytest.mark.unit
    @patch("shelf.evaluate._runner_impl.evaluate")
    def test_evaluate_all_default_retrieval_tasks(self, mock_evaluate):
        """Test evaluate_all() defaults to retrieval tasks."""

        # Setup mock evaluate to return results
        def mock_eval(task, **kwargs):
            return create_mock_result(
                task,
                task_type="retrieval",
                primary_metric="ndcg@10",
                primary_score=0.75,
            )

        mock_evaluate.side_effect = mock_eval

        embedder = MockEmbedder()
        results = evaluate_all(embedder)

        # Should evaluate all retrieval tasks by default
        assert len(results) > 0
        # Verify all results are from retrieval tasks
        from shelf.evaluate.registry import list_retrieval_tasks

        expected_tasks = set(list_retrieval_tasks())
        assert set(results.keys()) == expected_tasks

    @pytest.mark.unit
    @patch("shelf.evaluate._runner_impl.evaluate")
    def test_evaluate_all_specific_tasks(self, mock_evaluate):
        """Test evaluate_all() with specific task list."""

        # Setup mock evaluate
        def mock_eval(task, **kwargs):
            return create_mock_result(
                task,
                task_type="retrieval",
                primary_metric="ndcg@10",
                primary_score=0.75,
            )

        mock_evaluate.side_effect = mock_eval

        embedder = MockEmbedder()
        tasks = ["lcc_retrieval", "form_retrieval"]
        results = evaluate_all(embedder, tasks=tasks)

        assert len(results) == 2
        assert "lcc_retrieval" in results
        assert "form_retrieval" in results

    @pytest.mark.unit
    @patch("shelf.evaluate._runner_impl.evaluate")
    def test_evaluate_all_by_task_types(self, mock_evaluate):
        """Test evaluate_all() with task_types filter."""

        # Setup mock evaluate
        def mock_eval(task, **kwargs):
            return create_mock_result(task, primary_score=0.82)

        mock_evaluate.side_effect = mock_eval

        embedder = MockEmbedder()
        results = evaluate_all(embedder, task_types=[TaskType.CLASSIFICATION])

        # Should evaluate all classification tasks
        from shelf.evaluate.registry import list_classification_tasks

        expected_tasks = set(list_classification_tasks())
        assert set(results.keys()) == expected_tasks

    @pytest.mark.unit
    @patch("shelf.evaluate._runner_impl.evaluate")
    def test_evaluate_all_multiple_task_types(self, mock_evaluate):
        """Test evaluate_all() with multiple task types."""

        # Setup mock evaluate
        def mock_eval(task, **kwargs):
            return create_mock_result(
                task, primary_metric="test_metric", primary_score=0.80
            )

        mock_evaluate.side_effect = mock_eval

        embedder = MockEmbedder()
        results = evaluate_all(
            embedder, task_types=[TaskType.RETRIEVAL, TaskType.CLUSTERING]
        )

        # Should have both retrieval and clustering tasks
        from shelf.evaluate.registry import list_clustering_tasks, list_retrieval_tasks

        expected_tasks = set(list_retrieval_tasks() + list_clustering_tasks())
        assert set(results.keys()) == expected_tasks

    @pytest.mark.unit
    @patch("shelf.evaluate._runner_impl.evaluate")
    def test_evaluate_all_saves_individual_results(self, mock_evaluate, tmp_path):
        """Test evaluate_all() saves individual result files when output_dir specified."""

        # Setup mock evaluate
        def mock_eval(task, **kwargs):
            return create_mock_result(
                task,
                task_type="retrieval",
                primary_metric="ndcg@10",
                primary_score=0.75,
            )

        mock_evaluate.side_effect = mock_eval

        embedder = MockEmbedder()
        tasks = ["lcc_retrieval", "form_retrieval"]
        output_dir = tmp_path / "results"

        results = evaluate_all(embedder, tasks=tasks, output_dir=output_dir)

        # Verify directory was created
        assert output_dir.exists()

        # Verify individual result files exist
        assert (output_dir / "lcc_retrieval.json").exists()
        assert (output_dir / "form_retrieval.json").exists()

    @pytest.mark.unit
    @patch("shelf.evaluate._runner_impl.evaluate")
    def test_evaluate_all_passes_split_parameter(self, mock_evaluate):
        """Test evaluate_all() passes split parameter to evaluate()."""
        # Setup mock evaluate
        mock_evaluate.return_value = create_mock_result(
            "lcc_retrieval",
            task_type="retrieval",
            primary_metric="ndcg@10",
            primary_score=0.75,
        )

        embedder = MockEmbedder()
        tasks = ["lcc_retrieval"]
        results = evaluate_all(embedder, tasks=tasks, split="validation")

        # Verify split was passed to evaluate()
        call_kwargs = mock_evaluate.call_args[1]
        assert call_kwargs["split"] == "validation"

    @pytest.mark.unit
    @patch("shelf.evaluate._runner_impl.evaluate")
    def test_evaluate_all_passes_kwargs(self, mock_evaluate):
        """Test evaluate_all() passes additional kwargs to evaluate()."""
        # Setup mock evaluate
        mock_evaluate.return_value = create_mock_result(
            "lcc_retrieval",
            task_type="retrieval",
            primary_metric="ndcg@10",
            primary_score=0.75,
        )

        embedder = MockEmbedder()
        tasks = ["lcc_retrieval"]
        results = evaluate_all(
            embedder, tasks=tasks, batch_size=64, show_progress=False, max_queries=100
        )

        # Verify kwargs were passed to evaluate()
        call_kwargs = mock_evaluate.call_args[1]
        assert call_kwargs["batch_size"] == 64
        assert call_kwargs["show_progress"] is False
        assert call_kwargs["max_queries"] == 100

    @pytest.mark.unit
    @patch("shelf.evaluate._runner_impl.evaluate")
    def test_evaluate_all_skips_not_implemented(self, mock_evaluate, caplog):
        """Test evaluate_all() skips tasks that raise NotImplementedError."""

        # Setup mock to raise NotImplementedError for one task
        def mock_eval(task, **kwargs):
            if task == "lcc_retrieval":
                return create_mock_result(
                    task,
                    task_type="retrieval",
                    primary_metric="ndcg@10",
                    primary_score=0.75,
                )
            else:
                raise NotImplementedError("Not implemented yet")

        mock_evaluate.side_effect = mock_eval

        embedder = MockEmbedder()
        tasks = ["lcc_retrieval", "form_retrieval"]

        with caplog.at_level("WARNING"):
            results = evaluate_all(embedder, tasks=tasks)

        # Should only have one result (the successful one)
        assert len(results) == 1
        assert "lcc_retrieval" in results

        # Should have warning logged
        assert "Skipping" in caplog.text

    @pytest.mark.unit
    @patch("shelf.evaluate._runner_impl.evaluate")
    def test_evaluate_all_propagates_errors(self, mock_evaluate):
        """Test evaluate_all() propagates non-NotImplementedError exceptions."""
        # Setup mock to raise general exception
        mock_evaluate.side_effect = RuntimeError("Test error")

        embedder = MockEmbedder()
        tasks = ["lcc_retrieval"]

        with pytest.raises(RuntimeError, match="Test error"):
            evaluate_all(embedder, tasks=tasks)

    @pytest.mark.unit
    @patch("shelf.evaluate._runner_impl.evaluate")
    def test_evaluate_all_returns_dict(self, mock_evaluate):
        """Test evaluate_all() returns dict mapping task names to results."""

        # Setup mock evaluate
        def mock_eval(task, **kwargs):
            return create_mock_result(
                task,
                task_type="retrieval",
                primary_metric="ndcg@10",
                primary_score=0.75,
            )

        mock_evaluate.side_effect = mock_eval

        embedder = MockEmbedder()
        tasks = ["lcc_retrieval", "form_retrieval"]
        results = evaluate_all(embedder, tasks=tasks)

        # Verify result structure
        assert isinstance(results, dict)
        assert len(results) == 2
        for task_name, result in results.items():
            assert isinstance(result, EvaluationResult)
            assert result.task == task_name


# ===========================================================================
# Tests for Prediction File Checksum
# ===========================================================================


class TestPredictionFileChecksum:
    """Tests for prediction file checksum tracking."""

    @pytest.mark.unit
    @patch("shelf.evaluate._runner_impl.ClassificationEvaluator")
    def test_checksum_stored_for_file_predictions(self, mock_evaluator_cls):
        """Test that prediction file checksum is stored in context."""
        # Setup mock evaluator
        mock_evaluator = MagicMock()
        mock_context = EvaluationContext(
            shelf_version="0.2.0",
            python_version="3.13.0",
            sklearn_version="1.3.0",
            numpy_version="1.26.0",
            polars_version="1.0.0",
            dataset_checksum="test_dataset",
            prediction_file_checksum=None,
            random_seed=42,
            platform_info="Linux",
            timestamp="2024-01-01T00:00:00Z",
        )
        mock_result = create_mock_result("lcc_classification")
        mock_result.context = mock_context
        mock_evaluator.evaluate.return_value = mock_result
        mock_evaluator._load_ground_truth.return_value = []
        mock_evaluator_cls.return_value = mock_evaluator

        # Create predictions file
        predictions = [{"id": "doc_001", "prediction": "A"}]
        pred_file = create_temp_predictions_file(predictions)

        try:
            result = evaluate("lcc_classification", predictions=pred_file)

            # Verify checksum was added to context
            assert result.context is not None
            assert result.context.prediction_file_checksum is not None
            assert len(result.context.prediction_file_checksum) > 0
        finally:
            pred_file.unlink()

    @pytest.mark.unit
    @patch("shelf.evaluate._runner_impl.ClassificationEvaluator")
    def test_no_checksum_for_list_predictions(self, mock_evaluator_cls):
        """Test that no checksum is stored for predictions list (not file)."""
        # Setup mock evaluator
        mock_evaluator = MagicMock()
        mock_result = create_mock_result("lcc_classification")
        mock_evaluator.evaluate.return_value = mock_result
        mock_evaluator._load_ground_truth.return_value = []
        mock_evaluator_cls.return_value = mock_evaluator

        predictions = [{"id": "doc_001", "prediction": "A"}]
        result = evaluate("lcc_classification", predictions=predictions)

        # No checksum for list predictions
        # (context might be None or have no prediction_file_checksum)
        if result.context is not None:
            assert result.context.prediction_file_checksum is None

"""End-to-end tests for SHELF evaluation framework.

These tests exercise the complete evaluation flow:
- Registry lookup
- Evaluator creation and dispatch
- Real metric computation
- Result serialization

Only data loading is mocked to avoid requiring HuggingFace datasets.

Marked with @pytest.mark.integration for separate execution.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest

from shelf.evaluate.runner import evaluate, evaluate_all
from shelf.evaluate.results import EvaluationResult
from shelf.evaluate.tasks import TaskType


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def classification_dataset() -> pl.DataFrame:
    """Dataset for classification tasks with all required fields."""
    return pl.DataFrame(
        {
            "id": [f"doc_{i:03d}" for i in range(50)],
            "text": [
                f"Document {i} about {'science' if i % 3 == 0 else 'history' if i % 3 == 1 else 'arts'}"
                for i in range(50)
            ],
            "lcc_code": [
                ["Q", "D", "N"][i % 3]  # Science, History, Fine Arts
                for i in range(50)
            ],
            "lcgft_category": [
                ["Informational works", "Creative nonfiction", "Literature"][i % 3]
                for i in range(50)
            ],
            "lcgft_form": ["lecture"] * 50,
            "register": [["academic", "formal", "casual"][i % 3] for i in range(50)],
            "topic": ["General"] * 50,
            "region": ["North America"] * 50,
            "audience": ["General audience"] * 50,
            "geographic_region": ["North America"] * 50,
        }
    )


@pytest.fixture
def classification_predictions() -> list[dict[str, Any]]:
    """Predictions for classification tasks."""
    # 80% correct predictions
    predictions = []
    for i in range(50):
        true_label = ["Q", "D", "N"][i % 3]
        # Make 20% wrong (indices 5, 15, 25, 35, 45)
        if i % 10 == 5:
            pred_label = "A"  # Wrong
        else:
            pred_label = true_label  # Correct
        predictions.append({"id": f"doc_{i:03d}", "prediction": pred_label})
    return predictions


@pytest.fixture
def retrieval_dataset() -> pl.DataFrame:
    """Dataset for retrieval tasks."""
    return pl.DataFrame(
        {
            "id": [f"doc_{i:03d}" for i in range(30)],
            "text": [
                f"Document {i} about {'physics' if i % 3 == 0 else 'chemistry' if i % 3 == 1 else 'biology'}"
                for i in range(30)
            ],
            "lcc_code": [
                ["Q", "Q", "Q"][i % 3]  # All Science for simplicity
                for i in range(30)
            ],
            "lcgft_form": [["lecture", "article", "review"][i % 3] for i in range(30)],
            "lcgft_category": ["Informational works"] * 30,
        }
    )


@pytest.fixture
def clustering_dataset() -> pl.DataFrame:
    """Dataset for clustering tasks."""
    return pl.DataFrame(
        {
            "id": [f"doc_{i:03d}" for i in range(40)],
            "text": [f"Document {i} about topic {i % 4}" for i in range(40)],
            "lcc_code": [["Q", "D", "N", "P"][i % 4] for i in range(40)],
            "lcgft_category": [
                ["Informational works", "Creative nonfiction", "Literature", "Music"][
                    i % 4
                ]
                for i in range(40)
            ],
            "register": [
                ["academic", "formal", "casual", "technical"][i % 4] for i in range(40)
            ],
            "geographic_region": [
                ["North America", "Europe", "East Asia", "South America"][i % 4]
                for i in range(40)
            ],
        }
    )


@pytest.fixture
def clustering_predictions() -> list[dict[str, Any]]:
    """Predictions for clustering tasks."""
    # Good clustering: mostly follows ground truth pattern
    predictions = []
    for i in range(40):
        # Assign clusters that roughly match the LCC pattern
        cluster = i % 4
        # Add some noise
        if i % 10 == 7:
            cluster = (cluster + 1) % 4
        predictions.append({"id": f"doc_{i:03d}", "cluster": cluster})
    return predictions


@pytest.fixture
def pair_dataset() -> pl.DataFrame:
    """Dataset for pair classification tasks."""
    pairs = []
    for i in range(20):
        label = 1 if i % 2 == 0 else 0  # Alternating same/different
        pairs.append(
            {
                "pair_id": f"pair_{i:03d}",
                "text_a": f"First document about topic {i % 3}",
                "text_b": f"Second document about topic {i % 3 if label == 1 else (i + 1) % 3}",
                "label": label,
            }
        )
    return pl.DataFrame(pairs)


@pytest.fixture
def pair_predictions() -> list[dict[str, Any]]:
    """Predictions for pair classification tasks."""
    predictions = []
    for i in range(20):
        # Good predictions that mostly correlate with label
        score = 0.8 if i % 2 == 0 else 0.2
        # Add some noise
        if i % 5 == 3:
            score = 0.5
        predictions.append({"pair_id": f"pair_{i:03d}", "score": score})
    return predictions


class MockEmbedder:
    """Mock embedder for testing model-based evaluation."""

    def __init__(self, dim: int = 64):
        self.dim = dim
        self.embedding_dim = dim
        self.model_name = "mock-embedder"
        self._rng = np.random.default_rng(42)

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Generate deterministic embeddings based on text content."""
        embeddings = []
        for text in texts:
            # Use hash of text for deterministic embeddings
            seed = hash(text[:30]) % (2**32)
            rng = np.random.default_rng(seed)
            emb = rng.standard_normal(self.dim)
            emb = emb / np.linalg.norm(emb)
            embeddings.append(emb)
        return np.array(embeddings, dtype=np.float32)


@pytest.fixture
def mock_embedder() -> MockEmbedder:
    """Provide mock embedder for testing."""
    return MockEmbedder(dim=64)


# ===========================================================================
# E2E Tests for evaluate() via Public API
# ===========================================================================


@pytest.mark.integration
class TestEvaluateE2EClassification:
    """E2E tests for classification via evaluate() public API."""

    def test_evaluate_lcc_classification_from_predictions(
        self, classification_dataset, classification_predictions
    ):
        """Test full flow: evaluate() -> registry -> evaluator -> metrics."""
        with patch(
            "shelf.evaluate.evaluators.base.TaskEvaluator._load_ground_truth",
            return_value=classification_dataset,
        ):
            result = evaluate(
                task="lcc_classification",
                predictions=classification_predictions,
                split="test",
            )

        # Verify result structure
        assert isinstance(result, EvaluationResult)
        assert result.task == "lcc_classification"
        assert result.task_type == "classification"
        assert result.split == "test"

        # Verify metrics computed
        assert "macro_f1" in result.metrics
        assert "micro_f1" in result.metrics
        assert "accuracy" in result.metrics
        assert result.primary_metric == "macro_f1"
        assert 0 <= result.primary_score <= 1

        # Verify counts
        assert result.num_samples == 50

        # Verify context captured
        assert result.context is not None
        assert result.context.shelf_version is not None
        assert result.context.random_seed == 42

    def test_evaluate_lcgft_category_classification(self, classification_dataset):
        """Test LCGFT category classification task."""
        predictions = [
            {
                "id": f"doc_{i:03d}",
                "prediction": classification_dataset["lcgft_category"][i],
            }
            for i in range(50)
        ]

        with patch(
            "shelf.evaluate.evaluators.base.TaskEvaluator._load_ground_truth",
            return_value=classification_dataset,
        ):
            result = evaluate(
                task="lcgft_category_classification",
                predictions=predictions,
                split="test",
            )

        assert result.task == "lcgft_category_classification"
        # Perfect predictions should have high accuracy
        assert result.metrics["accuracy"] == pytest.approx(1.0, rel=0.01)

    def test_evaluate_register_classification(self, classification_dataset):
        """Test register classification task."""
        predictions = [
            {"id": f"doc_{i:03d}", "prediction": classification_dataset["register"][i]}
            for i in range(50)
        ]

        with patch(
            "shelf.evaluate.evaluators.base.TaskEvaluator._load_ground_truth",
            return_value=classification_dataset,
        ):
            result = evaluate(
                task="register_classification",
                predictions=predictions,
                split="test",
            )

        assert result.task == "register_classification"
        assert result.metrics["accuracy"] == pytest.approx(1.0, rel=0.01)

    def test_evaluate_classification_from_file(
        self, classification_dataset, classification_predictions, tmp_path
    ):
        """Test evaluation from predictions file path."""
        pred_file = tmp_path / "predictions.jsonl"
        with open(pred_file, "w") as f:
            for pred in classification_predictions:
                f.write(json.dumps(pred) + "\n")

        with patch(
            "shelf.evaluate.evaluators.base.TaskEvaluator._load_ground_truth",
            return_value=classification_dataset,
        ):
            result = evaluate(
                task="lcc_classification",
                predictions=str(pred_file),
                split="test",
            )

        assert result.task == "lcc_classification"
        assert result.context.prediction_file_checksum is not None
        assert len(result.context.prediction_file_checksum) == 32  # MD5


@pytest.mark.integration
class TestEvaluateE2ERetrieval:
    """E2E tests for retrieval via evaluate() public API."""

    def test_evaluate_retrieval_with_embedder(self, retrieval_dataset, mock_embedder):
        """Test retrieval evaluation with embedder model."""
        with patch(
            "shelf.evaluate.evaluators.base.TaskEvaluator._load_ground_truth",
            return_value=retrieval_dataset,
        ):
            result = evaluate(
                task="lcc_retrieval",
                model=mock_embedder,
                split="test",
                max_queries=10,  # Limit for faster test
                show_progress=False,
            )

        assert isinstance(result, EvaluationResult)
        assert result.task == "lcc_retrieval"
        assert result.task_type == "retrieval"

        # Verify retrieval metrics
        assert "ndcg@10" in result.metrics or "ndcg_at_10" in result.metrics
        assert "mrr" in result.metrics
        assert result.num_samples > 0

    def test_evaluate_form_retrieval(self, retrieval_dataset, mock_embedder):
        """Test form retrieval task."""
        with patch(
            "shelf.evaluate.evaluators.base.TaskEvaluator._load_ground_truth",
            return_value=retrieval_dataset,
        ):
            result = evaluate(
                task="form_retrieval",
                model=mock_embedder,
                split="test",
                max_queries=5,
                show_progress=False,
            )

        assert result.task == "form_retrieval"
        assert result.task_type == "retrieval"


@pytest.mark.integration
class TestEvaluateE2EClustering:
    """E2E tests for clustering via evaluate() public API."""

    def test_evaluate_clustering_from_predictions(
        self, clustering_dataset, clustering_predictions
    ):
        """Test clustering with prediction list."""
        with patch(
            "shelf.evaluate.evaluators.base.TaskEvaluator._load_ground_truth",
            return_value=clustering_dataset,
        ):
            result = evaluate(
                task="lcc_clustering",
                predictions=clustering_predictions,
                split="test",
            )

        assert isinstance(result, EvaluationResult)
        assert result.task == "lcc_clustering"
        assert result.task_type == "clustering"

        # Verify clustering metrics
        assert "v_measure" in result.metrics
        assert "nmi" in result.metrics
        assert "ari" in result.metrics
        assert 0 <= result.primary_score <= 1

    def test_evaluate_clustering_with_embedder(self, clustering_dataset, mock_embedder):
        """Test clustering with embedder (runs k-means internally)."""
        with patch(
            "shelf.evaluate.evaluators.base.TaskEvaluator._load_ground_truth",
            return_value=clustering_dataset,
        ):
            result = evaluate(
                task="lcc_clustering",
                model=mock_embedder,
                split="test",
                show_progress=False,
            )

        assert result.task == "lcc_clustering"
        assert result.task_type == "clustering"
        assert "v_measure" in result.metrics

    def test_evaluate_lcgft_clustering(self, clustering_dataset, mock_embedder):
        """Test LCGFT clustering task."""
        with patch(
            "shelf.evaluate.evaluators.base.TaskEvaluator._load_ground_truth",
            return_value=clustering_dataset,
        ):
            result = evaluate(
                task="lcgft_clustering",
                model=mock_embedder,
                split="test",
                show_progress=False,
            )

        assert result.task == "lcgft_clustering"
        assert "v_measure" in result.metrics


@pytest.mark.integration
class TestEvaluateE2EPairClassification:
    """E2E tests for pair classification via evaluate() public API."""

    def test_evaluate_pair_from_predictions(self, pair_dataset, pair_predictions):
        """Test pair classification with prediction list."""
        with patch(
            "shelf.evaluate.evaluators.pair.PairClassificationEvaluator._load_ground_truth",
            return_value=pair_dataset,
        ):
            result = evaluate(
                task="same_lcc_pairs",
                predictions=pair_predictions,
                split="test",
            )

        assert isinstance(result, EvaluationResult)
        assert result.task == "same_lcc_pairs"
        assert result.task_type == "pair_classification"

        # Verify pair metrics
        assert "f1" in result.metrics
        assert "accuracy" in result.metrics
        assert 0 <= result.primary_score <= 1

    def test_evaluate_pair_with_embedder(self, pair_dataset, mock_embedder):
        """Test pair classification with embedder."""
        with patch(
            "shelf.evaluate.evaluators.pair.PairClassificationEvaluator._load_ground_truth",
            return_value=pair_dataset,
        ):
            result = evaluate(
                task="same_lcc_pairs",
                model=mock_embedder,
                split="test",
                show_progress=False,
            )

        assert result.task == "same_lcc_pairs"
        assert "f1" in result.metrics


# ===========================================================================
# E2E Tests for evaluate_all()
# ===========================================================================


@pytest.mark.integration
class TestEvaluateAllE2E:
    """E2E tests for evaluate_all() with real execution."""

    def test_evaluate_all_retrieval_tasks(self, retrieval_dataset, mock_embedder):
        """Test evaluate_all runs multiple retrieval tasks."""
        with patch(
            "shelf.evaluate.evaluators.base.TaskEvaluator._load_ground_truth",
            return_value=retrieval_dataset,
        ):
            results = evaluate_all(
                model=mock_embedder,
                tasks=["lcc_retrieval", "form_retrieval"],
                split="test",
                max_queries=5,
                show_progress=False,
            )

        assert len(results) == 2
        assert "lcc_retrieval" in results
        assert "form_retrieval" in results

        for task_name, result in results.items():
            assert isinstance(result, EvaluationResult)
            assert result.task == task_name
            assert result.task_type == "retrieval"

    def test_evaluate_all_with_output_dir(
        self, retrieval_dataset, mock_embedder, tmp_path
    ):
        """Test that evaluate_all creates output files."""
        output_dir = tmp_path / "results"

        with patch(
            "shelf.evaluate.evaluators.base.TaskEvaluator._load_ground_truth",
            return_value=retrieval_dataset,
        ):
            results = evaluate_all(
                model=mock_embedder,
                tasks=["lcc_retrieval"],
                split="test",
                output_dir=output_dir,
                max_queries=5,
                show_progress=False,
            )

        # Verify output file created
        assert (output_dir / "lcc_retrieval.json").exists()

        # Verify file content
        with open(output_dir / "lcc_retrieval.json") as f:
            data = json.load(f)
            assert data["task"] == "lcc_retrieval"
            assert "metrics" in data

    def test_evaluate_all_clustering_tasks(self, clustering_dataset, mock_embedder):
        """Test evaluate_all with clustering task types."""
        with patch(
            "shelf.evaluate.evaluators.base.TaskEvaluator._load_ground_truth",
            return_value=clustering_dataset,
        ):
            results = evaluate_all(
                model=mock_embedder,
                task_types=[TaskType.CLUSTERING],
                tasks=["lcc_clustering", "lcgft_clustering"],
                split="test",
                show_progress=False,
            )

        assert len(results) == 2
        for result in results.values():
            assert result.task_type == "clustering"
            assert "v_measure" in result.metrics


# ===========================================================================
# E2E Tests for Output File Handling
# ===========================================================================


@pytest.mark.integration
class TestOutputFileHandling:
    """E2E tests for result serialization and output paths."""

    def test_evaluate_saves_to_output_path(
        self, classification_dataset, classification_predictions, tmp_path
    ):
        """Test that evaluate() saves results when output_path specified."""
        output_file = tmp_path / "results.json"

        with patch(
            "shelf.evaluate.evaluators.base.TaskEvaluator._load_ground_truth",
            return_value=classification_dataset,
        ):
            result = evaluate(
                task="lcc_classification",
                predictions=classification_predictions,
                split="test",
                output_path=output_file,
            )

        # Verify file created
        assert output_file.exists()

        # Verify content matches result
        with open(output_file) as f:
            data = json.load(f)

        assert data["task"] == result.task
        assert data["primary_score"] == pytest.approx(result.primary_score, rel=0.001)
        assert data["metrics"]["accuracy"] == pytest.approx(
            result.metrics["accuracy"], rel=0.001
        )

    def test_result_round_trip_serialization(
        self, classification_dataset, classification_predictions, tmp_path
    ):
        """Test that results can be serialized and deserialized."""
        with patch(
            "shelf.evaluate.evaluators.base.TaskEvaluator._load_ground_truth",
            return_value=classification_dataset,
        ):
            original = evaluate(
                task="lcc_classification",
                predictions=classification_predictions,
                split="test",
            )

        # Save and load
        output_file = tmp_path / "result.json"
        original.to_json(output_file)
        loaded = EvaluationResult.from_json(output_file)

        # Verify all fields preserved
        assert loaded.task == original.task
        assert loaded.task_type == original.task_type
        assert loaded.split == original.split
        assert loaded.primary_metric == original.primary_metric
        assert loaded.primary_score == pytest.approx(original.primary_score, rel=0.001)
        assert loaded.num_samples == original.num_samples
        assert loaded.context is not None
        assert loaded.context.shelf_version == original.context.shelf_version


# ===========================================================================
# E2E Tests for Error Handling
# ===========================================================================


@pytest.mark.integration
class TestEvaluateErrorHandling:
    """E2E tests for error handling in evaluate()."""

    def test_evaluate_unknown_task_raises(self):
        """Test that unknown task raises ValueError."""
        with pytest.raises(ValueError, match="Unknown task"):
            evaluate(task="nonexistent_task", predictions=[])

    def test_evaluate_no_predictions_or_model_raises(self):
        """Test that missing predictions and model raises ValueError."""
        with pytest.raises(
            ValueError, match="Must provide either predictions or model"
        ):
            evaluate(task="lcc_classification")

    def test_evaluate_invalid_predictions_raises(self, classification_dataset):
        """Test that invalid predictions raise validation error."""
        # Predictions with unknown IDs
        invalid_predictions = [
            {"id": "unknown_001", "prediction": "A"},
            {"id": "unknown_002", "prediction": "B"},
        ]

        with patch(
            "shelf.evaluate.evaluators.base.TaskEvaluator._load_ground_truth",
            return_value=classification_dataset,
        ):
            with pytest.raises(Exception):  # ValidationError or similar
                evaluate(
                    task="lcc_classification",
                    predictions=invalid_predictions,
                    split="test",
                )

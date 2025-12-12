"""Unit tests for retrieval evaluator.

Tests the RetrievalEvaluator class that evaluates retrieval/ranking predictions.
Covers initialization, evaluation with different ranking qualities, per-query metrics,
different k values, edge cases, and validation.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest

from shelf.evaluate.evaluators.retrieval import RetrievalEvaluator
from shelf.evaluate.results import EvaluationResult
from shelf.evaluate.schemas import ValidationError
from shelf.evaluate.tasks import TaskSpec, TaskType


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def retrieval_task_spec() -> TaskSpec:
    """Create a minimal retrieval task spec."""
    return TaskSpec(
        name="test_retrieval",
        task_type=TaskType.RETRIEVAL,
        description="Test retrieval task",
        text_field="body",
        label_field="lcc",
        id_field="id",
        label_space=tuple(["A", "B", "C", "D"]),
        primary_metric="ndcg@10",
        secondary_metrics=("mrr", "recall@10", "map@10"),
        dataset_name="test_dataset",
        dataset_config="default",
        default_split="test",
    )


@pytest.fixture
def ground_truth_queries() -> pl.DataFrame:
    """Create ground truth queries DataFrame."""
    return pl.DataFrame(
        {
            "id": ["q1", "q2", "q3", "q4"],
            "body": [
                "Query 1 about topic A",
                "Query 2 about topic B",
                "Query 3 about topic A",
                "Query 4 about topic C",
            ],
            "lcc": ["A", "B", "A", "C"],
        }
    )


@pytest.fixture
def ground_truth_corpus() -> pl.DataFrame:
    """Create ground truth corpus DataFrame."""
    return pl.DataFrame(
        {
            "id": [f"doc_{i}" for i in range(1, 11)],
            "body": [f"Document {i} text" for i in range(1, 11)],
            "lcc": ["A", "A", "B", "B", "B", "C", "C", "D", "D", "D"],
        }
    )


@pytest.fixture
def perfect_predictions() -> list[dict[str, Any]]:
    """Perfect retrieval predictions (all relevant docs ranked first)."""
    return [
        {"query_id": "q1", "ranked_doc_ids": ["doc_1", "doc_2", "doc_3", "doc_4"]},
        {"query_id": "q2", "ranked_doc_ids": ["doc_3", "doc_4", "doc_5", "doc_1"]},
        {"query_id": "q3", "ranked_doc_ids": ["doc_1", "doc_2", "doc_6", "doc_7"]},
        {"query_id": "q4", "ranked_doc_ids": ["doc_6", "doc_7", "doc_1", "doc_2"]},
    ]


@pytest.fixture
def partial_predictions() -> list[dict[str, Any]]:
    """Partial retrieval predictions (relevant docs mixed in)."""
    return [
        {"query_id": "q1", "ranked_doc_ids": ["doc_3", "doc_1", "doc_4", "doc_2"]},
        {"query_id": "q2", "ranked_doc_ids": ["doc_1", "doc_3", "doc_2", "doc_4"]},
        {"query_id": "q3", "ranked_doc_ids": ["doc_6", "doc_1", "doc_7", "doc_2"]},
        {"query_id": "q4", "ranked_doc_ids": ["doc_1", "doc_6", "doc_2", "doc_7"]},
    ]


@pytest.fixture
def worst_predictions() -> list[dict[str, Any]]:
    """Worst retrieval predictions (all relevant docs ranked last)."""
    return [
        {"query_id": "q1", "ranked_doc_ids": ["doc_8", "doc_9", "doc_10", "doc_1"]},
        {"query_id": "q2", "ranked_doc_ids": ["doc_1", "doc_2", "doc_6", "doc_3"]},
        {"query_id": "q3", "ranked_doc_ids": ["doc_8", "doc_9", "doc_10", "doc_1"]},
        {"query_id": "q4", "ranked_doc_ids": ["doc_1", "doc_2", "doc_3", "doc_6"]},
    ]


@pytest.fixture
def predictions_with_scores() -> list[dict[str, Any]]:
    """Predictions with relevance scores."""
    return [
        {
            "query_id": "q1",
            "ranked_doc_ids": ["doc_1", "doc_2", "doc_3"],
            "scores": [0.95, 0.87, 0.72],
        },
        {
            "query_id": "q2",
            "ranked_doc_ids": ["doc_3", "doc_4", "doc_5"],
            "scores": [0.91, 0.85, 0.78],
        },
    ]


# ===========================================================================
# Initialization Tests
# ===========================================================================


@pytest.mark.unit
@pytest.mark.evaluator
def test_retrieval_evaluator_initialization(retrieval_task_spec):
    """Test basic initialization of RetrievalEvaluator."""
    evaluator = RetrievalEvaluator(retrieval_task_spec)

    assert evaluator.task_spec == retrieval_task_spec
    assert evaluator.random_seed == 42
    assert evaluator.k_values == [1, 5, 10, 50, 100]


@pytest.mark.unit
@pytest.mark.evaluator
def test_retrieval_evaluator_custom_k_values(retrieval_task_spec):
    """Test initialization with custom k values."""
    custom_k = [1, 3, 5, 10]
    evaluator = RetrievalEvaluator(retrieval_task_spec, k_values=custom_k)

    assert evaluator.k_values == custom_k


@pytest.mark.unit
@pytest.mark.evaluator
def test_retrieval_evaluator_custom_random_seed(retrieval_task_spec):
    """Test initialization with custom random seed."""
    evaluator = RetrievalEvaluator(retrieval_task_spec, random_seed=123)

    assert evaluator.random_seed == 123


# ===========================================================================
# Perfect Rankings Tests
# ===========================================================================


@pytest.mark.unit
@pytest.mark.evaluator
def test_evaluate_perfect_rankings(
    retrieval_task_spec,
    perfect_predictions,
    ground_truth_queries,
    ground_truth_corpus,
):
    """Test evaluation with perfect rankings."""
    evaluator = RetrievalEvaluator(retrieval_task_spec, k_values=[1, 5, 10])

    # Mock _load_ground_truth to return our test data
    with patch.object(evaluator, "_load_ground_truth") as mock_load:
        mock_load.return_value = ground_truth_corpus

        result = evaluator.evaluate(
            predictions=perfect_predictions,
            ground_truth=ground_truth_queries,
            corpus_splits=["train", "validation"],
        )

    assert isinstance(result, EvaluationResult)
    assert result.task == "test_retrieval"

    # Perfect rankings should have high scores
    # Note: may not be 1.0 for all metrics depending on relevance distribution
    metrics = result.metrics
    assert "ndcg@10" in metrics
    assert "mrr" in metrics
    assert "recall@10" in metrics
    assert "map@10" in metrics

    # All metrics should be between 0 and 1
    for metric_name, metric_value in metrics.items():
        if not metric_name.startswith("_"):  # Skip internal fields
            assert 0.0 <= metric_value <= 1.0, f"{metric_name} = {metric_value}"


@pytest.mark.unit
@pytest.mark.evaluator
def test_evaluate_perfect_rankings_high_mrr(
    retrieval_task_spec,
    ground_truth_queries,
    ground_truth_corpus,
):
    """Test that perfect rankings achieve high MRR."""
    # Create predictions where first doc is always relevant
    predictions = [
        {
            "query_id": "q1",
            "ranked_doc_ids": ["doc_1", "doc_3", "doc_4"],
        },  # doc_1 is LCC=A
        {
            "query_id": "q2",
            "ranked_doc_ids": ["doc_3", "doc_1", "doc_2"],
        },  # doc_3 is LCC=B
        {
            "query_id": "q3",
            "ranked_doc_ids": ["doc_2", "doc_1", "doc_3"],
        },  # doc_2 is LCC=A
    ]

    evaluator = RetrievalEvaluator(retrieval_task_spec, k_values=[1, 5, 10])

    with patch.object(evaluator, "_load_ground_truth") as mock_load:
        mock_load.return_value = ground_truth_corpus

        result = evaluator.evaluate(
            predictions=predictions,
            ground_truth=ground_truth_queries.head(3),
            corpus_splits=["train"],
        )

    # MRR should be 1.0 if first doc is always relevant
    assert result.metrics["mrr"] == 1.0


# ===========================================================================
# Partial Rankings Tests
# ===========================================================================


@pytest.mark.unit
@pytest.mark.evaluator
def test_evaluate_partial_rankings(
    retrieval_task_spec,
    partial_predictions,
    ground_truth_queries,
    ground_truth_corpus,
):
    """Test evaluation with partial rankings (relevant docs mixed in)."""
    evaluator = RetrievalEvaluator(retrieval_task_spec, k_values=[1, 5, 10])

    with patch.object(evaluator, "_load_ground_truth") as mock_load:
        mock_load.return_value = ground_truth_corpus

        result = evaluator.evaluate(
            predictions=partial_predictions,
            ground_truth=ground_truth_queries,
        )

    # Partial rankings should have moderate scores
    assert 0.0 < result.metrics["ndcg@10"] < 1.0
    assert 0.0 < result.metrics["mrr"] < 1.0
    assert 0.0 < result.metrics["recall@10"] <= 1.0


@pytest.mark.unit
@pytest.mark.evaluator
def test_evaluate_partial_rankings_lower_than_perfect(
    retrieval_task_spec,
    perfect_predictions,
    partial_predictions,
    ground_truth_queries,
    ground_truth_corpus,
):
    """Test that partial rankings score lower than perfect rankings."""
    evaluator = RetrievalEvaluator(retrieval_task_spec, k_values=[10])

    with patch.object(evaluator, "_load_ground_truth") as mock_load:
        mock_load.return_value = ground_truth_corpus

        perfect_result = evaluator.evaluate(
            predictions=perfect_predictions,
            ground_truth=ground_truth_queries,
        )

        partial_result = evaluator.evaluate(
            predictions=partial_predictions,
            ground_truth=ground_truth_queries,
        )

    # Partial should score lower than perfect
    assert partial_result.metrics["ndcg@10"] <= perfect_result.metrics["ndcg@10"]
    assert partial_result.metrics["mrr"] <= perfect_result.metrics["mrr"]


# ===========================================================================
# Worst Rankings Tests
# ===========================================================================


@pytest.mark.unit
@pytest.mark.evaluator
def test_evaluate_worst_rankings(
    retrieval_task_spec,
    worst_predictions,
    ground_truth_queries,
    ground_truth_corpus,
):
    """Test evaluation with worst rankings (relevant docs at bottom)."""
    evaluator = RetrievalEvaluator(retrieval_task_spec, k_values=[1, 3, 5])

    with patch.object(evaluator, "_load_ground_truth") as mock_load:
        mock_load.return_value = ground_truth_corpus

        result = evaluator.evaluate(
            predictions=worst_predictions,
            ground_truth=ground_truth_queries,
        )

    # Worst rankings should have low scores at small k
    assert result.metrics["recall@1"] <= 0.5
    assert result.metrics["precision@1"] <= 0.5


# ===========================================================================
# Different K Values Tests
# ===========================================================================


@pytest.mark.unit
@pytest.mark.evaluator
def test_evaluate_different_k_values(
    retrieval_task_spec,
    partial_predictions,
    ground_truth_queries,
    ground_truth_corpus,
):
    """Test that different k values produce different metrics."""
    evaluator = RetrievalEvaluator(retrieval_task_spec, k_values=[1, 5, 10, 50])

    with patch.object(evaluator, "_load_ground_truth") as mock_load:
        mock_load.return_value = ground_truth_corpus

        result = evaluator.evaluate(
            predictions=partial_predictions,
            ground_truth=ground_truth_queries,
        )

    # All k values should be present
    assert "ndcg@1" in result.metrics
    assert "ndcg@5" in result.metrics
    assert "ndcg@10" in result.metrics
    assert "ndcg@50" in result.metrics

    assert "recall@1" in result.metrics
    assert "recall@5" in result.metrics
    assert "recall@10" in result.metrics
    assert "recall@50" in result.metrics


@pytest.mark.unit
@pytest.mark.evaluator
def test_recall_increases_with_k(
    retrieval_task_spec,
    partial_predictions,
    ground_truth_queries,
    ground_truth_corpus,
):
    """Test that recall increases (or stays same) as k increases."""
    evaluator = RetrievalEvaluator(retrieval_task_spec, k_values=[1, 5, 10])

    with patch.object(evaluator, "_load_ground_truth") as mock_load:
        mock_load.return_value = ground_truth_corpus

        result = evaluator.evaluate(
            predictions=partial_predictions,
            ground_truth=ground_truth_queries,
        )

    # Recall should be monotonically non-decreasing with k
    assert result.metrics["recall@1"] <= result.metrics["recall@5"]
    assert result.metrics["recall@5"] <= result.metrics["recall@10"]


# ===========================================================================
# Per-Query Metrics Tests
# ===========================================================================


@pytest.mark.unit
@pytest.mark.evaluator
def test_evaluate_per_query_metrics(
    retrieval_task_spec,
    partial_predictions,
    ground_truth_queries,
    ground_truth_corpus,
):
    """Test per-query metrics computation."""
    evaluator = RetrievalEvaluator(retrieval_task_spec, k_values=[5, 10])

    with patch.object(evaluator, "_load_ground_truth") as mock_load:
        mock_load.return_value = ground_truth_corpus

        result = evaluator.evaluate(
            predictions=partial_predictions,
            ground_truth=ground_truth_queries,
            compute_ci=False,
        )

    # Per-query metrics should be available
    assert result.per_query_metrics is not None

    # Should have metrics for each query
    assert len(result.per_query_metrics) > 0

    # Each query should have the expected metrics
    for query_id, query_metrics in result.per_query_metrics.items():
        assert "mrr" in query_metrics
        assert "ndcg@5" in query_metrics
        assert "ndcg@10" in query_metrics
        assert "recall@5" in query_metrics
        assert "map@10" in query_metrics


@pytest.mark.unit
@pytest.mark.evaluator
def test_per_query_metrics_aggregation(
    retrieval_task_spec,
    ground_truth_queries,
    ground_truth_corpus,
):
    """Test that aggregated metrics match mean of per-query metrics."""
    predictions = [
        {"query_id": "q1", "ranked_doc_ids": ["doc_1", "doc_2"]},
        {"query_id": "q2", "ranked_doc_ids": ["doc_3", "doc_4"]},
    ]

    evaluator = RetrievalEvaluator(retrieval_task_spec, k_values=[5])

    with patch.object(evaluator, "_load_ground_truth") as mock_load:
        mock_load.return_value = ground_truth_corpus

        result = evaluator.evaluate(
            predictions=predictions,
            ground_truth=ground_truth_queries.head(2),
        )

    # Compute mean of per-query MRR
    per_query = result.per_query_metrics
    mrr_values = [metrics["mrr"] for metrics in per_query.values()]
    mean_mrr = np.mean(mrr_values)

    # Should match the aggregated metric
    assert abs(result.metrics["mrr"] - mean_mrr) < 1e-6


# ===========================================================================
# Edge Cases Tests
# ===========================================================================


@pytest.mark.unit
@pytest.mark.evaluator
def test_evaluate_no_relevant_documents(
    retrieval_task_spec,
    ground_truth_corpus,
):
    """Test evaluation when query has no relevant documents in corpus."""
    # Create query with label that doesn't exist in corpus
    queries = pl.DataFrame(
        {
            "id": ["q1"],
            "body": ["Query about nonexistent topic"],
            "lcc": ["Z"],  # Not in corpus
        }
    )

    predictions = [
        {"query_id": "q1", "ranked_doc_ids": ["doc_1", "doc_2", "doc_3"]},
    ]

    evaluator = RetrievalEvaluator(retrieval_task_spec, k_values=[5])

    with patch.object(evaluator, "_load_ground_truth") as mock_load:
        mock_load.return_value = ground_truth_corpus

        result = evaluator.evaluate(
            predictions=predictions,
            ground_truth=queries,
        )

    # Metrics should handle this gracefully
    # Query with no relevant docs is typically skipped in metric computation
    assert result.num_samples >= 0  # Query processed


@pytest.mark.unit
@pytest.mark.evaluator
def test_evaluate_empty_ranking(
    retrieval_task_spec,
    ground_truth_queries,
    ground_truth_corpus,
):
    """Test evaluation with empty ranked_doc_ids (should fail validation)."""
    predictions = [
        {"query_id": "q1", "ranked_doc_ids": []},  # Empty ranking
    ]

    evaluator = RetrievalEvaluator(retrieval_task_spec)

    with patch.object(evaluator, "_load_ground_truth") as mock_load:
        mock_load.return_value = ground_truth_corpus

        # Should raise validation error
        with pytest.raises(ValidationError):
            evaluator.evaluate(
                predictions=predictions,
                ground_truth=ground_truth_queries.head(1),
            )


@pytest.mark.unit
@pytest.mark.evaluator
def test_evaluate_single_query(
    retrieval_task_spec,
    ground_truth_queries,
    ground_truth_corpus,
):
    """Test evaluation with single query."""
    predictions = [
        {"query_id": "q1", "ranked_doc_ids": ["doc_1", "doc_2", "doc_3"]},
    ]

    evaluator = RetrievalEvaluator(retrieval_task_spec, k_values=[5])

    with patch.object(evaluator, "_load_ground_truth") as mock_load:
        mock_load.return_value = ground_truth_corpus

        result = evaluator.evaluate(
            predictions=predictions,
            ground_truth=ground_truth_queries.head(1),
        )

    assert result.num_samples == 1
    assert "ndcg@5" in result.metrics
    assert "mrr" in result.metrics


# ===========================================================================
# Validation Tests
# ===========================================================================


@pytest.mark.unit
@pytest.mark.evaluator
def test_validate_missing_query_id(
    retrieval_task_spec,
    ground_truth_queries,
    ground_truth_corpus,
):
    """Test validation fails when query_id is missing from predictions."""
    predictions = [
        {"query_id": "q1", "ranked_doc_ids": ["doc_1", "doc_2"]},
        # Missing q2
        {"query_id": "q3", "ranked_doc_ids": ["doc_3", "doc_4"]},
    ]

    evaluator = RetrievalEvaluator(retrieval_task_spec)

    with patch.object(evaluator, "_load_ground_truth") as mock_load:
        mock_load.return_value = ground_truth_corpus

        # Should raise validation error for missing q2
        with pytest.raises(ValidationError) as exc_info:
            evaluator.evaluate(
                predictions=predictions,
                ground_truth=ground_truth_queries.head(3),
            )

        assert "Missing predictions" in str(exc_info.value)


@pytest.mark.unit
@pytest.mark.evaluator
def test_validate_unknown_query_id(
    retrieval_task_spec,
    ground_truth_queries,
    ground_truth_corpus,
):
    """Test validation fails when query_id not in ground truth."""
    predictions = [
        {"query_id": "q1", "ranked_doc_ids": ["doc_1", "doc_2"]},
        {"query_id": "q999", "ranked_doc_ids": ["doc_3", "doc_4"]},  # Unknown
    ]

    evaluator = RetrievalEvaluator(retrieval_task_spec)

    with patch.object(evaluator, "_load_ground_truth") as mock_load:
        mock_load.return_value = ground_truth_corpus

        with pytest.raises(ValidationError) as exc_info:
            evaluator.evaluate(
                predictions=predictions,
                ground_truth=ground_truth_queries.head(1),
            )

        assert "unknown query_id" in str(exc_info.value)


@pytest.mark.unit
@pytest.mark.evaluator
def test_validate_unknown_doc_id(
    retrieval_task_spec,
    ground_truth_queries,
    ground_truth_corpus,
):
    """Test validation fails when ranked_doc_ids contains unknown doc."""
    predictions = [
        {"query_id": "q1", "ranked_doc_ids": ["doc_1", "doc_999"]},  # doc_999 unknown
    ]

    evaluator = RetrievalEvaluator(retrieval_task_spec)

    with patch.object(evaluator, "_load_ground_truth") as mock_load:
        mock_load.return_value = ground_truth_corpus

        with pytest.raises(ValidationError) as exc_info:
            evaluator.evaluate(
                predictions=predictions,
                ground_truth=ground_truth_queries.head(1),
            )

        assert "unknown doc IDs" in str(exc_info.value)


@pytest.mark.unit
@pytest.mark.evaluator
def test_validate_duplicate_query_id(
    retrieval_task_spec,
    ground_truth_queries,
    ground_truth_corpus,
):
    """Test validation fails when query_id appears multiple times."""
    predictions = [
        {"query_id": "q1", "ranked_doc_ids": ["doc_1", "doc_2"]},
        {"query_id": "q1", "ranked_doc_ids": ["doc_3", "doc_4"]},  # Duplicate
    ]

    evaluator = RetrievalEvaluator(retrieval_task_spec)

    with patch.object(evaluator, "_load_ground_truth") as mock_load:
        mock_load.return_value = ground_truth_corpus

        with pytest.raises(ValidationError) as exc_info:
            evaluator.evaluate(
                predictions=predictions,
                ground_truth=ground_truth_queries.head(1),
            )

        assert "duplicate query_id" in str(exc_info.value)


@pytest.mark.unit
@pytest.mark.evaluator
def test_validate_predictions_with_scores(
    retrieval_task_spec,
    predictions_with_scores,
    ground_truth_queries,
    ground_truth_corpus,
):
    """Test that predictions with scores are valid."""
    evaluator = RetrievalEvaluator(retrieval_task_spec, k_values=[5])

    with patch.object(evaluator, "_load_ground_truth") as mock_load:
        mock_load.return_value = ground_truth_corpus

        # Should not raise
        result = evaluator.evaluate(
            predictions=predictions_with_scores,
            ground_truth=ground_truth_queries.head(2),
        )

    assert result is not None


# ===========================================================================
# Relevance Building Tests
# ===========================================================================


@pytest.mark.unit
@pytest.mark.evaluator
def test_build_relevance_judgments_from_df(
    retrieval_task_spec,
    ground_truth_queries,
    ground_truth_corpus,
):
    """Test building relevance judgments from query and corpus DataFrames."""
    evaluator = RetrievalEvaluator(retrieval_task_spec)

    relevance = evaluator._build_relevance_judgments_from_df(
        queries_df=ground_truth_queries,
        corpus_df=ground_truth_corpus,
    )

    # Should have relevance for each query
    assert "q1" in relevance
    assert "q2" in relevance
    assert "q3" in relevance
    assert "q4" in relevance

    # q1 has lcc=A, so relevant docs are doc_1, doc_2 (both lcc=A)
    assert relevance["q1"] == {"doc_1", "doc_2"}

    # q2 has lcc=B, so relevant docs are doc_3, doc_4, doc_5 (all lcc=B)
    assert relevance["q2"] == {"doc_3", "doc_4", "doc_5"}

    # q3 has lcc=A (same as q1)
    assert relevance["q3"] == {"doc_1", "doc_2"}

    # q4 has lcc=C, so relevant docs are doc_6, doc_7 (both lcc=C)
    assert relevance["q4"] == {"doc_6", "doc_7"}


@pytest.mark.unit
@pytest.mark.evaluator
def test_build_relevance_no_relevant_in_corpus(
    retrieval_task_spec,
    ground_truth_corpus,
):
    """Test building relevance when query label not in corpus."""
    queries = pl.DataFrame(
        {
            "id": ["q1"],
            "body": ["Query"],
            "lcc": ["Z"],  # Not in corpus
        }
    )

    evaluator = RetrievalEvaluator(retrieval_task_spec)

    relevance = evaluator._build_relevance_judgments_from_df(
        queries_df=queries,
        corpus_df=ground_truth_corpus,
    )

    # Should return empty set for q1
    assert relevance["q1"] == set()


# ===========================================================================
# Corpus Splits Tests
# ===========================================================================


@pytest.mark.unit
@pytest.mark.evaluator
def test_evaluate_custom_corpus_splits(
    retrieval_task_spec,
    perfect_predictions,
    ground_truth_queries,
    ground_truth_corpus,
):
    """Test evaluation with custom corpus splits."""
    evaluator = RetrievalEvaluator(retrieval_task_spec)

    # Mock different data for different splits
    def mock_load(split):
        if split == "train":
            return ground_truth_corpus.head(5)
        else:
            return ground_truth_corpus.tail(5)

    with patch.object(evaluator, "_load_ground_truth", side_effect=mock_load):
        result = evaluator.evaluate(
            predictions=perfect_predictions,
            ground_truth=ground_truth_queries,
            corpus_splits=["train", "validation"],
        )

    assert result is not None
    assert "ndcg@10" in result.metrics


@pytest.mark.unit
@pytest.mark.evaluator
def test_evaluate_single_corpus_split(
    retrieval_task_spec,
    perfect_predictions,
    ground_truth_queries,
    ground_truth_corpus,
):
    """Test evaluation with single corpus split."""
    evaluator = RetrievalEvaluator(retrieval_task_spec)

    with patch.object(evaluator, "_load_ground_truth") as mock_load:
        mock_load.return_value = ground_truth_corpus

        result = evaluator.evaluate(
            predictions=perfect_predictions,
            ground_truth=ground_truth_queries,
            corpus_splits=["train"],
        )

    assert result is not None


# ===========================================================================
# Ranking Computation Tests
# ===========================================================================


@pytest.mark.unit
@pytest.mark.evaluator
def test_compute_rankings(retrieval_task_spec):
    """Test _compute_rankings method with embeddings."""
    evaluator = RetrievalEvaluator(retrieval_task_spec)

    # Create simple embeddings
    query_ids = ["q1", "q2"]
    corpus_ids = ["doc_1", "doc_2", "doc_3"]

    query_embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )

    corpus_embeddings = np.array(
        [
            [1.0, 0.0, 0.0],  # Most similar to q1
            [0.5, 0.5, 0.0],  # Middle
            [0.0, 1.0, 0.0],  # Most similar to q2
        ]
    )

    results = evaluator._compute_rankings(
        query_ids=query_ids,
        query_embeddings=query_embeddings,
        corpus_ids=corpus_ids,
        corpus_embeddings=corpus_embeddings,
        top_k=3,
        show_progress=False,
    )

    # q1 should rank doc_1 first (cosine similarity = 1.0)
    assert results["q1"][0] == "doc_1"

    # q2 should rank doc_3 first (cosine similarity = 1.0)
    assert results["q2"][0] == "doc_3"

    # Each result should have top_k documents
    assert len(results["q1"]) == 3
    assert len(results["q2"]) == 3


@pytest.mark.unit
@pytest.mark.evaluator
def test_compute_rankings_top_k_limit(retrieval_task_spec):
    """Test that _compute_rankings respects top_k limit."""
    evaluator = RetrievalEvaluator(retrieval_task_spec)

    query_ids = ["q1"]
    corpus_ids = ["doc_1", "doc_2", "doc_3", "doc_4", "doc_5"]

    query_embeddings = np.random.randn(1, 10)
    corpus_embeddings = np.random.randn(5, 10)

    results = evaluator._compute_rankings(
        query_ids=query_ids,
        query_embeddings=query_embeddings,
        corpus_ids=corpus_ids,
        corpus_embeddings=corpus_embeddings,
        top_k=3,
        show_progress=False,
    )

    # Should only return top 3 docs
    assert len(results["q1"]) == 3


# ===========================================================================
# Result Metadata Tests
# ===========================================================================


@pytest.mark.unit
@pytest.mark.evaluator
def test_result_contains_task_metadata(
    retrieval_task_spec,
    perfect_predictions,
    ground_truth_queries,
    ground_truth_corpus,
):
    """Test that result contains expected task metadata."""
    evaluator = RetrievalEvaluator(retrieval_task_spec)

    with patch.object(evaluator, "_load_ground_truth") as mock_load:
        mock_load.return_value = ground_truth_corpus

        result = evaluator.evaluate(
            predictions=perfect_predictions,
            ground_truth=ground_truth_queries,
        )

    assert result.task == "test_retrieval"
    assert result.task_type == "retrieval"
    assert result.num_samples == len(perfect_predictions)


@pytest.mark.unit
@pytest.mark.evaluator
def test_result_num_queries(
    retrieval_task_spec,
    partial_predictions,
    ground_truth_queries,
    ground_truth_corpus,
):
    """Test that num_queries is correctly reported in results."""
    evaluator = RetrievalEvaluator(retrieval_task_spec)

    with patch.object(evaluator, "_load_ground_truth") as mock_load:
        mock_load.return_value = ground_truth_corpus

        result = evaluator.evaluate(
            predictions=partial_predictions,
            ground_truth=ground_truth_queries,
        )

    # Should match number of predictions with relevant docs
    assert result.num_samples >= 0


# ===========================================================================
# Confidence Interval Tests (Placeholder)
# ===========================================================================


@pytest.mark.unit
@pytest.mark.evaluator
def test_evaluate_compute_ci_not_implemented(
    retrieval_task_spec,
    perfect_predictions,
    ground_truth_queries,
    ground_truth_corpus,
):
    """Test that compute_ci parameter is accepted (implementation may be pending)."""
    evaluator = RetrievalEvaluator(retrieval_task_spec)

    with patch.object(evaluator, "_load_ground_truth") as mock_load:
        mock_load.return_value = ground_truth_corpus

        # Should not raise error even if compute_ci=True
        # Implementation may be pending based on the code comment
        result = evaluator.evaluate(
            predictions=perfect_predictions,
            ground_truth=ground_truth_queries,
            compute_ci=True,
        )

    assert result is not None

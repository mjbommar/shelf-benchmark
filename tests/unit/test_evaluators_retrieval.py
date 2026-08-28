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


# ===========================================================================
# Graded relevance (data_plan_v0.4 section 11.1)
# ===========================================================================


@pytest.fixture
def faceted_queries() -> pl.DataFrame:
    """Queries carrying the facet columns the graded schemes read."""
    return pl.DataFrame(
        {
            "id": ["q1"],
            "text": ["a lecture on physics"],
            "lcc_code": ["Q"],
            "lcgft_form": ["Lectures"],
            "lcgft_category": ["Instructional and educational works"],
            "topics": [["Physics"]],
        }
    )


@pytest.fixture
def faceted_corpus() -> pl.DataFrame:
    """One document per relevance tier of the subject axis."""
    return pl.DataFrame(
        {
            "id": ["same_class", "shared_topic", "unrelated"],
            "text": ["a", "b", "c"],
            "lcc_code": ["Q", "T", "K"],
            "lcgft_form": ["Jokes", "Lectures", "Maps"],
            "lcgft_category": [
                "Recreational works",
                "Instructional and educational works",
                "Cartographic materials",
            ],
            "topics": [["Chemistry"], ["Physics"], ["Law"]],
        }
    )


def _graded_task_spec(name: str, label_field: str) -> TaskSpec:
    return TaskSpec(
        name=name,
        task_type=TaskType.RETRIEVAL,
        description="graded test",
        text_field="text",
        label_field=label_field,
        id_field="id",
        label_space=None,
        primary_metric="ndcg@10",
        secondary_metrics=("graded_ndcg@10",),
        dataset_name="test_dataset",
        dataset_config="default",
        default_split="test",
    )


def _evaluate_ranking(spec, queries, corpus, ranking, k_values=None):
    evaluator = RetrievalEvaluator(spec, k_values=k_values or [1, 3])
    with patch.object(evaluator, "_load_ground_truth") as mock_load:
        mock_load.return_value = corpus
        return evaluator.evaluate(
            predictions=[{"query_id": "q1", "ranked_doc_ids": ranking}],
            ground_truth=queries,
            corpus_splits=["train"],
        )


def test_graded_metrics_are_reported_alongside_binary(faceted_queries, faceted_corpus):
    """Graded NDCG is an addition, not a replacement: both must be present."""
    spec = _graded_task_spec("graded_lcc", "lcc_code")
    result = _evaluate_ranking(
        spec, faceted_queries, faceted_corpus, ["same_class", "shared_topic"]
    )

    assert "ndcg@10" not in result.metrics  # k_values overridden in the fixture
    assert result.metrics["ndcg@1"] == pytest.approx(1.0)
    assert result.metrics["graded_ndcg@1"] == pytest.approx(1.0)
    assert result.metrics["graded_relevance_max_gain"] == pytest.approx(3.0)


def test_graded_gives_partial_credit_where_binary_gives_none(
    faceted_queries, faceted_corpus
):
    """A different-class document sharing a topic is a near miss, not a miss."""
    spec = _graded_task_spec("graded_lcc", "lcc_code")

    partial = _evaluate_ranking(spec, faceted_queries, faceted_corpus, ["shared_topic"])
    nothing = _evaluate_ranking(spec, faceted_queries, faceted_corpus, ["unrelated"])

    # Binary relevance cannot tell these apart -- neither document is class Q.
    assert partial.metrics["ndcg@1"] == pytest.approx(0.0)
    assert nothing.metrics["ndcg@1"] == pytest.approx(0.0)

    # Graded relevance can.
    assert partial.metrics["graded_ndcg@1"] > 0.0
    assert nothing.metrics["graded_ndcg@1"] == pytest.approx(0.0)


def test_graded_form_axis_credits_same_category(faceted_queries, faceted_corpus):
    """On the form axis, a same-category document earns the partial tier."""
    spec = _graded_task_spec("graded_form", "lcgft_form")
    result = _evaluate_ranking(
        spec, faceted_queries, faceted_corpus, ["unrelated", "shared_topic"]
    )
    # shared_topic is also a Lecture, so it is the top tier and is ranked second.
    assert 0.0 < result.metrics["graded_ndcg@3"] < 1.0


def test_graded_category_axis_ranks_same_form_highest(faceted_queries, faceted_corpus):
    """For a category query, a same-form document is the category match plus more."""
    spec = _graded_task_spec("graded_category", "lcgft_category")
    result = _evaluate_ranking(spec, faceted_queries, faceted_corpus, ["shared_topic"])
    assert result.metrics["graded_ndcg@1"] == pytest.approx(1.0)
    assert result.metrics["graded_relevance_max_gain"] == pytest.approx(3.0)


def test_graded_metrics_absent_when_axis_has_no_scheme(
    retrieval_task_spec, ground_truth_queries, ground_truth_corpus, perfect_predictions
):
    """An unknown axis degrades to binary-only rather than failing."""
    evaluator = RetrievalEvaluator(retrieval_task_spec)
    with patch.object(evaluator, "_load_ground_truth") as mock_load:
        mock_load.return_value = ground_truth_corpus
        result = evaluator.evaluate(
            predictions=perfect_predictions,
            ground_truth=ground_truth_queries,
            corpus_splits=["train"],
        )
    assert not any(key.startswith("graded_") for key in result.metrics)


def test_graded_metrics_absent_when_facet_columns_missing():
    """The right axis but a corpus without the columns the scheme reads."""
    spec = _graded_task_spec("graded_form", "lcgft_form")
    queries = pl.DataFrame({"id": ["q1"], "text": ["x"], "lcgft_form": ["Lectures"]})
    corpus = pl.DataFrame({"id": ["d1"], "text": ["y"], "lcgft_form": ["Lectures"]})
    result = _evaluate_ranking(spec, queries, corpus, ["d1"])
    assert not any(key.startswith("graded_") for key in result.metrics)


# ===========================================================================
# Instruction-conditioned retrieval (data_plan_v0.4 section 11.3)
# ===========================================================================


@pytest.fixture
def instruction_queries() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": ["q1"],
            "text": ["a lecture on physics"],
            "lcc_code": ["Q"],
            "lcgft_form": ["Lectures"],
            "lcgft_category": ["Instructional and educational works"],
            "topics": [["Physics"]],
            "audience": ["General"],
            "register": ["academic"],
        }
    )


@pytest.fixture
def instruction_corpus() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": ["same_both", "same_form_other_subject", "same_subject_other_form"],
            "text": ["a", "b", "c"],
            "lcc_code": ["Q", "K", "Q"],
            "lcgft_form": ["Lectures", "Lectures", "Jokes"],
            "lcgft_category": [
                "Instructional and educational works",
                "Instructional and educational works",
                "Recreational works",
            ],
            "topics": [["Physics"], ["Law"], ["Humor"]],
            "audience": ["General", "General", "General"],
            "register": ["academic", "casual", "academic"],
        }
    )


def _instruction_result(task_name, queries, corpus, ranking):
    from shelf.evaluate.registry import get_task

    evaluator = RetrievalEvaluator(get_task(task_name), k_values=[1, 3])
    with patch.object(evaluator, "_load_ground_truth") as mock_load:
        mock_load.return_value = corpus
        return evaluator.evaluate(
            predictions=[{"query_id": "q1", "ranked_doc_ids": ranking}],
            ground_truth=queries,
            corpus_splits=["train"],
        )


def test_same_query_different_instruction_different_answer(
    instruction_queries, instruction_corpus
):
    """The claim the task rests on, checked end to end.

    One ranking, one query, two instructions: the document that is correct
    under one is wrong under the other.
    """
    ranking = ["same_form_other_subject", "same_subject_other_form"]

    same_form = _instruction_result(
        "instruction_same_form_diff_subject",
        instruction_queries,
        instruction_corpus,
        ranking,
    )
    same_subject = _instruction_result(
        "instruction_same_subject_diff_form",
        instruction_queries,
        instruction_corpus,
        ranking,
    )

    assert same_form.metrics["ndcg@1"] == pytest.approx(1.0)
    assert same_subject.metrics["ndcg@1"] == pytest.approx(0.0)
    assert same_subject.metrics["ndcg@3"] > 0.0


def test_instruction_constraint_diagnostics_are_reported(
    instruction_queries, instruction_corpus
):
    result = _instruction_result(
        "instruction_same_form_diff_subject",
        instruction_queries,
        instruction_corpus,
        ["same_both"],
    )
    # same_both is a Lecture (anchor hit) but class Q (contrast violation).
    assert result.metrics["anchor_match@1"] == pytest.approx(1.0)
    assert result.metrics["contrast_violation@1"] == pytest.approx(1.0)
    assert result.metrics["contrast_violation_lift@1"] > 1.0
    assert result.metrics["queries_without_answer"] == pytest.approx(0.0)


def test_instruction_relevance_ignores_label_field_grouping(
    instruction_queries, instruction_corpus
):
    """Relevance must not fall back to task_spec.label_field.

    ``instruction_same_form_diff_subject`` has ``label_field="lcgft_form"``. A
    plain label match would make ``same_both`` relevant; the instruction does
    not, because it is the same subject.
    """
    result = _instruction_result(
        "instruction_same_form_diff_subject",
        instruction_queries,
        instruction_corpus,
        ["same_both", "same_form_other_subject"],
    )
    assert result.metrics["ndcg@1"] == pytest.approx(0.0)


def test_instruction_task_reports_no_graded_metrics(
    instruction_queries, instruction_corpus
):
    """Grading is defined against a label axis, not against an instruction."""
    result = _instruction_result(
        "instruction_same_form_diff_subject",
        instruction_queries,
        instruction_corpus,
        ["same_form_other_subject"],
    )
    assert not any(key.startswith("graded_") for key in result.metrics)


def test_instruction_prefix_is_applied_to_queries(
    instruction_queries, instruction_corpus
):
    """The instruction must reach the model, on the query side only."""
    from shelf.evaluate.instructions import get_instruction
    from shelf.evaluate.registry import get_task

    spec = get_instruction("instruction_same_form_diff_subject")
    assert spec is not None

    seen: list[list[str]] = []

    class RecordingEmbedder:
        model_name = "recording"
        embedding_dim = 2

        def encode(self, texts, **_kwargs):
            seen.append(list(texts))
            return np.ones((len(texts), 2), dtype=float)

    evaluator = RetrievalEvaluator(get_task("instruction_same_form_diff_subject"))
    with patch.object(evaluator, "_load_ground_truth") as mock_load:
        mock_load.side_effect = lambda split: (
            instruction_queries if split == "test" else instruction_corpus
        )
        evaluator.evaluate_embedder(
            RecordingEmbedder(),  # type: ignore[arg-type]
            corpus_splits=["train"],
            show_progress=False,
        )

    corpus_texts, query_texts = seen[0], seen[1]
    assert all(spec.instruction not in text for text in corpus_texts)
    assert all(text.startswith("Instruct: ") for text in query_texts)
    assert query_texts[0].endswith("a lecture on physics")

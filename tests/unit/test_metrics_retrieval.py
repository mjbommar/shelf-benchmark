"""Unit tests for shelf.evaluate.metrics.retrieval module.

Tests cover:
- NDCG@k computation
- MRR (Mean Reciprocal Rank)
- Recall@k and Precision@k
- MAP (Mean Average Precision)
- compute_retrieval_metrics aggregation
- Edge cases: no relevant docs, k > results, etc.
"""

from __future__ import annotations

import pytest

from shelf.evaluate.metrics.retrieval import (
    average_precision,
    compute_retrieval_metrics,
    map_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


class TestNDCG:
    """Tests for NDCG@k metric."""

    def test_perfect_ranking(self, retrieval_perfect):
        """Test perfect ranking gives NDCG = 1.0."""
        relevant_ids, ranked_ids, _ = retrieval_perfect
        ndcg = ndcg_at_k(ranked_ids, set(relevant_ids), k=3)
        assert ndcg == pytest.approx(1.0)

    def test_perfect_ranking_larger_k(self, retrieval_perfect):
        """Test perfect ranking with k larger than relevant docs."""
        relevant_ids, ranked_ids, _ = retrieval_perfect
        # Only 3 relevant docs, but asking for top 5
        ndcg = ndcg_at_k(ranked_ids, set(relevant_ids), k=5)
        assert ndcg == pytest.approx(1.0)

    def test_partial_ranking(self, retrieval_partial):
        """Test partial ranking gives intermediate NDCG."""
        relevant_ids, ranked_ids, _ = retrieval_partial
        ndcg = ndcg_at_k(ranked_ids, set(relevant_ids), k=5)
        assert 0 < ndcg < 1

    def test_worst_ranking(self, retrieval_worst):
        """Test worst ranking gives low NDCG."""
        relevant_ids, ranked_ids, _ = retrieval_worst
        ndcg_at_3 = ndcg_at_k(ranked_ids, set(relevant_ids), k=3)
        ndcg_at_6 = ndcg_at_k(ranked_ids, set(relevant_ids), k=6)

        # At k=3, none are relevant
        assert ndcg_at_3 == pytest.approx(0.0)
        # At k=6, all 3 relevant docs are included
        assert ndcg_at_6 > 0

    def test_no_relevant_docs(self):
        """Test with no relevant documents."""
        ranked_ids = ["doc_1", "doc_2", "doc_3"]
        relevant_ids = set()  # No relevant docs

        ndcg = ndcg_at_k(ranked_ids, relevant_ids, k=3)
        # Edge case handling - should be 0
        assert ndcg == pytest.approx(0.0)

    def test_empty_ranked_list(self):
        """Test with empty ranked list."""
        ranked_ids = []
        relevant_ids = {"doc_1", "doc_2"}

        ndcg = ndcg_at_k(ranked_ids, relevant_ids, k=5)
        assert ndcg == pytest.approx(0.0)


class TestMRR:
    """Tests for Mean Reciprocal Rank."""

    def test_first_result_relevant(self, retrieval_perfect):
        """Test MRR = 1 when first result is relevant."""
        relevant_ids, ranked_ids, _ = retrieval_perfect
        assert mrr(ranked_ids, set(relevant_ids)) == pytest.approx(1.0)

    def test_second_result_relevant(self):
        """Test MRR = 0.5 when second result is first relevant."""
        ranked_ids = ["doc_4", "doc_1", "doc_5"]
        relevant_ids = {"doc_1", "doc_2"}

        assert mrr(ranked_ids, relevant_ids) == pytest.approx(0.5)

    def test_third_result_relevant(self):
        """Test MRR = 1/3 when third result is first relevant."""
        ranked_ids = ["doc_4", "doc_5", "doc_1", "doc_2"]
        relevant_ids = {"doc_1", "doc_2"}

        assert mrr(ranked_ids, relevant_ids) == pytest.approx(1.0 / 3.0)

    def test_no_relevant_in_results(self):
        """Test MRR when no relevant docs in results."""
        ranked_ids = ["doc_4", "doc_5", "doc_6"]
        relevant_ids = {"doc_1", "doc_2", "doc_3"}

        # MRR should be 0
        assert mrr(ranked_ids, relevant_ids) == pytest.approx(0.0)

    def test_empty_results(self):
        """Test MRR with empty results."""
        ranked_ids = []
        relevant_ids = {"doc_1", "doc_2"}

        assert mrr(ranked_ids, relevant_ids) == pytest.approx(0.0)


class TestRecallAndPrecision:
    """Tests for Recall@k and Precision@k."""

    def test_perfect_recall(self, retrieval_perfect):
        """Test perfect recall at sufficient k."""
        relevant_ids, ranked_ids, _ = retrieval_perfect
        recall = recall_at_k(ranked_ids, set(relevant_ids), k=3)
        assert recall == pytest.approx(1.0)

    def test_partial_recall(self, retrieval_partial):
        """Test partial recall."""
        relevant_ids, ranked_ids, _ = retrieval_partial
        recall_2 = recall_at_k(ranked_ids, set(relevant_ids), k=2)
        recall_5 = recall_at_k(ranked_ids, set(relevant_ids), k=5)

        # More results should give higher or equal recall
        assert recall_2 <= recall_5
        # At k=2, should have 1 of 3 relevant = 1/3
        assert recall_2 == pytest.approx(1.0 / 3.0)
        # At k=5, should have all 3 relevant = 1.0
        assert recall_5 == pytest.approx(1.0)

    def test_precision_at_k(self, retrieval_perfect):
        """Test precision at different k values."""
        relevant_ids, ranked_ids, _ = retrieval_perfect

        prec_3 = precision_at_k(ranked_ids, set(relevant_ids), k=3)
        prec_5 = precision_at_k(ranked_ids, set(relevant_ids), k=5)

        # For perfect ranking at k=3, all 3 are relevant
        assert prec_3 == pytest.approx(1.0)
        # At k=5, 3 of 5 are relevant
        assert prec_5 == pytest.approx(0.6)

    def test_recall_no_relevant_docs(self):
        """Test recall when no relevant docs exist."""
        ranked_ids = ["doc_1", "doc_2", "doc_3"]
        relevant_ids = set()

        recall = recall_at_k(ranked_ids, relevant_ids, k=3)
        assert recall == pytest.approx(0.0)

    def test_precision_k_zero(self):
        """Test precision when k=0."""
        ranked_ids = ["doc_1", "doc_2", "doc_3"]
        relevant_ids = {"doc_1"}

        prec = precision_at_k(ranked_ids, relevant_ids, k=0)
        assert prec == pytest.approx(0.0)


class TestAveragePrecision:
    """Tests for Average Precision."""

    def test_perfect_ap(self, retrieval_perfect):
        """Test AP for perfect ranking."""
        relevant_ids, ranked_ids, _ = retrieval_perfect
        ap = average_precision(ranked_ids, set(relevant_ids))
        assert ap == pytest.approx(1.0)

    def test_ap_calculation(self):
        """Test AP calculation with known example."""
        # Example: relevant docs at positions 1, 3, 5 out of 6 results
        ranked_ids = ["doc_1", "doc_4", "doc_2", "doc_5", "doc_3", "doc_6"]
        relevant_ids = {"doc_1", "doc_2", "doc_3"}

        # AP = (1/1 + 2/3 + 3/5) / 3
        expected_ap = (1.0 + 2.0 / 3.0 + 3.0 / 5.0) / 3.0
        ap = average_precision(ranked_ids, relevant_ids)
        assert ap == pytest.approx(expected_ap)

    def test_ap_with_cutoff(self, retrieval_perfect):
        """Test AP with cutoff k."""
        relevant_ids, ranked_ids, _ = retrieval_perfect

        # Without cutoff
        ap_full = average_precision(ranked_ids, set(relevant_ids))
        # With cutoff at k=3
        ap_k3 = average_precision(ranked_ids, set(relevant_ids), k=3)

        assert ap_full == pytest.approx(1.0)
        assert ap_k3 == pytest.approx(1.0)

    def test_ap_no_relevant_docs(self):
        """Test AP when no relevant docs exist."""
        ranked_ids = ["doc_1", "doc_2", "doc_3"]
        relevant_ids = set()

        ap = average_precision(ranked_ids, relevant_ids)
        assert ap == pytest.approx(0.0)

    def test_map_at_k_equals_ap_with_k(self):
        """Test that MAP@k equals AP with cutoff."""
        ranked_ids = ["doc_1", "doc_2", "doc_3", "doc_4", "doc_5"]
        relevant_ids = {"doc_1", "doc_3"}

        map_k = map_at_k(ranked_ids, relevant_ids, k=3)
        ap_k = average_precision(ranked_ids, relevant_ids, k=3)

        assert map_k == pytest.approx(ap_k)


class TestComputeRetrievalMetrics:
    """Tests for aggregated compute_retrieval_metrics function."""

    def test_all_metrics_returned(self, retrieval_multi_query):
        """Test that all expected metrics are returned."""
        results, relevance = retrieval_multi_query

        metrics = compute_retrieval_metrics(
            results=results,
            relevance=relevance,
            k_values=[1, 5, 10],
        )

        # Should have NDCG, recall, precision, MAP at each k
        assert "ndcg@1" in metrics
        assert "ndcg@5" in metrics
        assert "ndcg@10" in metrics
        assert "recall@1" in metrics
        assert "recall@5" in metrics
        assert "precision@1" in metrics
        assert "map@1" in metrics
        assert "mrr" in metrics
        assert "num_queries" in metrics

    def test_num_queries(self, retrieval_multi_query):
        """Test query count."""
        results, relevance = retrieval_multi_query

        metrics = compute_retrieval_metrics(
            results=results,
            relevance=relevance,
        )

        assert metrics["num_queries"] == 3

    def test_default_k_values(self, retrieval_multi_query):
        """Test default k values."""
        results, relevance = retrieval_multi_query

        metrics = compute_retrieval_metrics(
            results=results,
            relevance=relevance,
        )

        # Default k_values = [1, 5, 10, 50, 100]
        assert "ndcg@1" in metrics
        assert "ndcg@5" in metrics
        assert "ndcg@10" in metrics
        assert "ndcg@50" in metrics
        assert "ndcg@100" in metrics

    def test_per_query_metrics(self, retrieval_multi_query):
        """Test per-query metric breakdown."""
        results, relevance = retrieval_multi_query

        metrics = compute_retrieval_metrics(
            results=results,
            relevance=relevance,
            k_values=[5, 10],
            compute_per_query=True,
        )

        assert "per_query" in metrics
        per_query = metrics["per_query"]

        # Should have metrics for each query
        for query_id in results.keys():
            assert query_id in per_query
            assert "mrr" in per_query[query_id]
            assert "ndcg@5" in per_query[query_id]
            assert "recall@5" in per_query[query_id]

    def test_average_computation(self):
        """Test that metrics are averaged correctly."""
        # Two queries with known values
        results = {
            "q1": ["doc_1", "doc_2", "doc_3"],
            "q2": ["doc_4", "doc_5", "doc_6"],
        }
        relevance = {
            "q1": {"doc_1"},  # MRR = 1.0
            "q2": {"doc_5"},  # MRR = 0.5
        }

        metrics = compute_retrieval_metrics(
            results=results,
            relevance=relevance,
            k_values=[1, 3],
        )

        # Average MRR should be (1.0 + 0.5) / 2 = 0.75
        assert metrics["mrr"] == pytest.approx(0.75)


class TestRetrievalEdgeCases:
    """Edge case tests for retrieval metrics."""

    def test_k_larger_than_results(self, retrieval_perfect):
        """Test k larger than result list."""
        relevant_ids, ranked_ids, _ = retrieval_perfect

        # k=10 but only 5 results
        ndcg = ndcg_at_k(ranked_ids, set(relevant_ids), k=10)
        recall = recall_at_k(ranked_ids, set(relevant_ids), k=10)

        # Should handle gracefully
        assert 0 <= ndcg <= 1
        assert 0 <= recall <= 1

    def test_empty_results(self):
        """Test with empty result list."""
        ranked_ids = []
        relevant_ids = {"doc_1", "doc_2"}

        # Should handle gracefully
        ndcg = ndcg_at_k(ranked_ids, relevant_ids, k=5)
        recall = recall_at_k(ranked_ids, relevant_ids, k=5)
        prec = precision_at_k(ranked_ids, relevant_ids, k=5)

        assert ndcg == pytest.approx(0.0)
        assert recall == pytest.approx(0.0)
        assert prec == pytest.approx(0.0)

    def test_single_relevant_doc(self):
        """Test with single relevant document."""
        ranked_ids = ["doc_1", "doc_2", "doc_3"]
        relevant_ids = {"doc_2"}

        recall = recall_at_k(ranked_ids, relevant_ids, k=2)
        assert recall == pytest.approx(1.0)  # Found doc_2 at position 2

        mrr_score = mrr(ranked_ids, relevant_ids)
        assert mrr_score == pytest.approx(0.5)  # Position 2

    def test_empty_results_dict_raises(self):
        """Test that empty results dict raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            compute_retrieval_metrics(
                results={},
                relevance={"q1": {"doc_1"}},
            )

    def test_missing_relevance_raises(self):
        """Test that missing relevance judgments raise ValueError."""
        results = {"q1": ["doc_1", "doc_2"], "q2": ["doc_3", "doc_4"]}
        relevance = {"q1": {"doc_1"}}  # Missing q2

        with pytest.raises(ValueError, match="Missing relevance judgments"):
            compute_retrieval_metrics(
                results=results,
                relevance=relevance,
            )

    def test_query_with_no_relevant_docs_skipped(self):
        """Test that queries with no relevant docs are skipped."""
        results = {
            "q1": ["doc_1", "doc_2"],
            "q2": ["doc_3", "doc_4"],
        }
        relevance = {
            "q1": {"doc_1"},
            "q2": set(),  # No relevant docs
        }

        metrics = compute_retrieval_metrics(
            results=results,
            relevance=relevance,
            k_values=[1, 5],
        )

        # Only q1 should be counted
        assert metrics["num_queries"] == 1

    def test_all_irrelevant_results(self):
        """Test when all retrieved docs are irrelevant."""
        ranked_ids = ["doc_4", "doc_5", "doc_6"]
        relevant_ids = {"doc_1", "doc_2", "doc_3"}

        ndcg = ndcg_at_k(ranked_ids, relevant_ids, k=3)
        recall = recall_at_k(ranked_ids, relevant_ids, k=3)
        prec = precision_at_k(ranked_ids, relevant_ids, k=3)
        mrr_score = mrr(ranked_ids, relevant_ids)
        ap = average_precision(ranked_ids, relevant_ids)

        assert ndcg == pytest.approx(0.0)
        assert recall == pytest.approx(0.0)
        assert prec == pytest.approx(0.0)
        assert mrr_score == pytest.approx(0.0)
        assert ap == pytest.approx(0.0)

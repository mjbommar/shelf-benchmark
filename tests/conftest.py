"""Shared test fixtures for SHELF benchmark tests.

This conftest.py provides common fixtures used across test modules:
- Sample data (labels, predictions, scores)
- Classification fixtures
- Retrieval fixtures
- Statistical test fixtures
- Schema validation fixtures
- Factory fixtures for test data generation
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any

import numpy as np
import pytest


# ===========================================================================
# Autouse Fixtures (Applied to All Tests)
# ===========================================================================


@pytest.fixture(autouse=True)
def reset_random_seed() -> Generator[None, None, None]:
    """Reset random seed before each test for reproducibility."""
    np.random.seed(42)
    yield
    # No teardown needed


# ===========================================================================
# Label and Prediction Fixtures
# ===========================================================================


@pytest.fixture
def binary_labels() -> list[str]:
    """Binary classification labels."""
    return ["positive", "negative"]


@pytest.fixture
def multiclass_labels() -> list[str]:
    """Multiclass labels (LCC-like)."""
    return ["A", "B", "C", "D", "E"]


@pytest.fixture
def lcc_labels() -> list[str]:
    """Full LCC classification labels."""
    return list("ABCDEFGHJKLMNPQRSTUVZ")  # 21 classes


# ===========================================================================
# Classification Fixtures
# ===========================================================================


@pytest.fixture
def perfect_classification() -> tuple[list[str], list[str]]:
    """Perfect classification predictions (100% accuracy)."""
    y_true = ["A", "B", "C", "A", "B", "C", "A", "B", "C", "D"]
    y_pred = ["A", "B", "C", "A", "B", "C", "A", "B", "C", "D"]
    return y_true, y_pred


@pytest.fixture
def partial_classification() -> tuple[list[str], list[str]]:
    """Partial classification (50% accuracy, balanced errors)."""
    y_true = ["A", "B", "C", "D", "A", "B", "C", "D", "A", "B"]
    y_pred = ["A", "B", "C", "D", "B", "A", "D", "C", "B", "A"]
    return y_true, y_pred


@pytest.fixture
def random_classification() -> tuple[list[str], list[str]]:
    """Random predictions (baseline)."""
    np.random.seed(42)
    labels = ["A", "B", "C", "D"]
    y_true = [labels[i % 4] for i in range(100)]
    y_pred = [np.random.choice(labels) for _ in range(100)]
    return y_true, y_pred


@pytest.fixture
def imbalanced_classification() -> tuple[list[str], list[str]]:
    """Imbalanced class distribution (90% class A)."""
    y_true = ["A"] * 90 + ["B"] * 10
    y_pred = ["A"] * 85 + ["B"] * 5 + ["A"] * 5 + ["B"] * 5
    return y_true, y_pred


# ===========================================================================
# Statistical Test Fixtures
# ===========================================================================


@pytest.fixture
def paired_scores_different() -> tuple[np.ndarray, np.ndarray]:
    """Two score arrays with significant difference."""
    np.random.seed(42)
    scores_a = np.random.normal(0.85, 0.05, 50)
    scores_b = np.random.normal(0.75, 0.05, 50)
    return scores_a, scores_b


@pytest.fixture
def paired_scores_similar() -> tuple[np.ndarray, np.ndarray]:
    """Two score arrays with no significant difference."""
    np.random.seed(42)
    base = np.random.normal(0.80, 0.05, 50)
    noise_a = np.random.normal(0, 0.01, 50)
    noise_b = np.random.normal(0, 0.01, 50)
    return base + noise_a, base + noise_b


@pytest.fixture
def multiple_model_scores() -> dict[str, np.ndarray]:
    """Scores from multiple models for Friedman test."""
    np.random.seed(42)
    return {
        "ModelA": np.random.normal(0.85, 0.03, 10),
        "ModelB": np.random.normal(0.82, 0.04, 10),
        "ModelC": np.random.normal(0.78, 0.05, 10),
        "ModelD": np.random.normal(0.75, 0.04, 10),
    }


# ===========================================================================
# Retrieval Fixtures
# ===========================================================================


@pytest.fixture
def retrieval_perfect() -> tuple[list[str], list[str], list[str]]:
    """Perfect retrieval ranking."""
    relevant_ids = ["doc_1", "doc_2", "doc_3"]
    ranked_ids = ["doc_1", "doc_2", "doc_3", "doc_4", "doc_5"]
    all_ids = ["doc_1", "doc_2", "doc_3", "doc_4", "doc_5", "doc_6"]
    return relevant_ids, ranked_ids, all_ids


@pytest.fixture
def retrieval_partial() -> tuple[list[str], list[str], list[str]]:
    """Partial retrieval (relevant docs mixed in)."""
    relevant_ids = ["doc_1", "doc_3", "doc_5"]
    ranked_ids = ["doc_2", "doc_1", "doc_4", "doc_3", "doc_5"]
    all_ids = ["doc_1", "doc_2", "doc_3", "doc_4", "doc_5", "doc_6"]
    return relevant_ids, ranked_ids, all_ids


@pytest.fixture
def retrieval_worst() -> tuple[list[str], list[str], list[str]]:
    """Worst retrieval (relevant docs at bottom)."""
    relevant_ids = ["doc_1", "doc_2", "doc_3"]
    ranked_ids = ["doc_4", "doc_5", "doc_6", "doc_1", "doc_2", "doc_3"]
    all_ids = ["doc_1", "doc_2", "doc_3", "doc_4", "doc_5", "doc_6"]
    return relevant_ids, ranked_ids, all_ids


@pytest.fixture
def retrieval_multi_query() -> tuple[dict[str, list[str]], dict[str, set[str]]]:
    """Multiple queries for aggregated metrics."""
    results = {
        "q1": ["doc_1", "doc_2", "doc_3", "doc_4", "doc_5"],
        "q2": ["doc_6", "doc_7", "doc_8", "doc_9", "doc_10"],
        "q3": ["doc_11", "doc_12", "doc_13", "doc_14", "doc_15"],
    }
    relevance = {
        "q1": {"doc_1", "doc_3"},
        "q2": {"doc_6", "doc_7", "doc_8"},
        "q3": {"doc_15"},
    }
    return results, relevance


# ===========================================================================
# Data Fixtures
# ===========================================================================


@pytest.fixture
def sample_records() -> list[dict[str, Any]]:
    """Sample data records for checksum/provenance tests."""
    return [
        {
            "id": "doc_001",
            "lcc": "A",
            "form": "lecture",
            "git_commit": "abc123",
            "model": "gpt-5.1",
        },
        {
            "id": "doc_002",
            "lcc": "B",
            "form": "map",
            "git_commit": "abc123",
            "model": "gpt-5.1",
        },
        {
            "id": "doc_003",
            "lcc": "C",
            "form": "essay",
            "git_commit": "def456",
            "model": "gpt-5.2",
        },
    ]


@pytest.fixture
def temp_json_file(sample_records) -> Path:
    """Create a temporary JSON file with sample data."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sample_records, f)
        return Path(f.name)


@pytest.fixture
def temp_jsonl_file() -> Path:
    """Create a temporary JSONL file with predictions."""
    predictions = [
        {"id": "doc_001", "prediction": "A"},
        {"id": "doc_002", "prediction": "B"},
        {"id": "doc_003", "prediction": "C"},
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")
        return Path(f.name)


# ===========================================================================
# Bootstrap Fixtures
# ===========================================================================


@pytest.fixture
def bootstrap_data() -> np.ndarray:
    """Sample data for bootstrap tests."""
    np.random.seed(42)
    return np.random.normal(0.80, 0.10, 100)


# ===========================================================================
# Clustering Fixtures
# ===========================================================================


@pytest.fixture
def clustering_perfect() -> tuple[np.ndarray, np.ndarray]:
    """Perfect clustering (labels match exactly)."""
    labels_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    labels_pred = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    return labels_true, labels_pred


@pytest.fixture
def clustering_random() -> tuple[np.ndarray, np.ndarray]:
    """Random clustering."""
    np.random.seed(42)
    labels_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    labels_pred = np.random.randint(0, 3, 9)
    return labels_true, labels_pred


# ===========================================================================
# Pair Classification Fixtures
# ===========================================================================


@pytest.fixture
def pair_perfect() -> tuple[np.ndarray, np.ndarray]:
    """Perfect pair classification."""
    y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    return y_true, y_pred


@pytest.fixture
def pair_partial() -> tuple[np.ndarray, np.ndarray]:
    """Partial pair classification (75% accuracy)."""
    y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 1, 0, 1, 1, 0, 0, 0])
    return y_true, y_pred


# ===========================================================================
# Schema Validation Fixtures
# ===========================================================================


@pytest.fixture
def valid_classification_predictions() -> list[dict[str, Any]]:
    """Valid classification predictions for schema tests."""
    return [
        {"id": "doc_001", "prediction": "A", "confidence": 0.95},
        {"id": "doc_002", "prediction": "B", "confidence": 0.87},
        {"id": "doc_003", "prediction": "C", "confidence": 0.72},
    ]


@pytest.fixture
def valid_multilabel_predictions() -> list[dict[str, Any]]:
    """Valid multi-label predictions for schema tests."""
    return [
        {
            "id": "doc_001",
            "predictions": ["topic1", "topic2"],
            "confidences": [0.9, 0.8],
        },
        {"id": "doc_002", "predictions": ["topic3"], "confidences": [0.95]},
        {"id": "doc_003", "predictions": ["topic1", "topic4", "topic5"]},
    ]


@pytest.fixture
def valid_retrieval_predictions() -> list[dict[str, Any]]:
    """Valid retrieval predictions for schema tests."""
    return [
        {
            "query_id": "q1",
            "ranked_doc_ids": ["doc_1", "doc_2", "doc_3"],
            "scores": [0.9, 0.7, 0.5],
        },
        {"query_id": "q2", "ranked_doc_ids": ["doc_4", "doc_5"], "scores": [0.8, 0.6]},
        {"query_id": "q3", "ranked_doc_ids": ["doc_1", "doc_6", "doc_7", "doc_8"]},
    ]


@pytest.fixture
def valid_clustering_predictions() -> list[dict[str, Any]]:
    """Valid clustering predictions for schema tests."""
    return [
        {"id": "doc_001", "cluster": 0},
        {"id": "doc_002", "cluster": 1},
        {"id": "doc_003", "cluster": 0},
        {"id": "doc_004", "cluster": 2},
    ]


@pytest.fixture
def valid_pair_predictions() -> list[dict[str, Any]]:
    """Valid pair predictions for schema tests."""
    return [
        {"pair_id": "pair_001", "score": 0.95, "prediction": 1},
        {"pair_id": "pair_002", "score": 0.12, "prediction": 0},
        {"pair_id": "pair_003", "prediction": 1},
        {"pair_id": "pair_004", "score": 0.55},
    ]


@pytest.fixture
def ground_truth_ids() -> set[str]:
    """Ground truth document IDs for validation tests."""
    return {"doc_001", "doc_002", "doc_003"}


@pytest.fixture
def classification_label_space() -> set[str]:
    """Valid label space for classification validation."""
    return {"A", "B", "C", "D"}


# ===========================================================================
# Factory Fixtures
# ===========================================================================


@pytest.fixture
def make_temp_jsonl_file(tmp_path: Path):
    """Factory fixture for creating temporary JSONL files."""
    created_files: list[Path] = []

    def _make_file(
        predictions: list[dict[str, Any]], filename: str = "predictions.jsonl"
    ) -> Path:
        filepath = tmp_path / filename
        with open(filepath, "w") as f:
            for pred in predictions:
                f.write(json.dumps(pred) + "\n")
        created_files.append(filepath)
        return filepath

    yield _make_file

    # Cleanup
    for filepath in created_files:
        if filepath.exists():
            filepath.unlink()


@pytest.fixture
def make_classification_data():
    """Factory fixture for generating classification data with various scenarios."""

    def _make_data(
        n_samples: int = 100,
        n_classes: int = 4,
        accuracy: float = 0.8,
        seed: int = 42,
    ) -> tuple[list[str], list[str]]:
        np.random.seed(seed)
        labels = [chr(ord("A") + i) for i in range(n_classes)]
        y_true = [labels[i % n_classes] for i in range(n_samples)]

        # Generate predictions with target accuracy
        y_pred = []
        for true_label in y_true:
            if np.random.random() < accuracy:
                y_pred.append(true_label)
            else:
                # Pick a random different label
                wrong_labels = [l for l in labels if l != true_label]
                y_pred.append(np.random.choice(wrong_labels))
        return y_true, y_pred

    return _make_data


# ===========================================================================
# Task and Registry Fixtures
# ===========================================================================


@pytest.fixture
def sample_task_spec() -> dict[str, Any]:
    """Sample task specification dictionary."""
    return {
        "name": "test_classification",
        "task_type": "classification",
        "description": "Test classification task",
        "text_field": "body",
        "label_field": "lcc",
        "id_field": "id",
        "label_space": ["A", "B", "C", "D"],
        "metrics": ["macro_f1", "micro_f1", "accuracy"],
    }


# ===========================================================================
# Benchmark Fixtures
# ===========================================================================


@pytest.fixture
def large_classification_data() -> tuple[list[str], list[str]]:
    """Large dataset for benchmark tests."""
    np.random.seed(42)
    n_samples = 10000
    labels = list("ABCDEFGHJKLMNPQRSTUVZ")  # 21 LCC classes
    y_true = [np.random.choice(labels) for _ in range(n_samples)]
    y_pred = [np.random.choice(labels) for _ in range(n_samples)]
    return y_true, y_pred


@pytest.fixture
def large_retrieval_data() -> tuple[dict[str, list[str]], dict[str, set[str]]]:
    """Large retrieval dataset for benchmark tests."""
    np.random.seed(42)
    n_queries = 100
    corpus_size = 1000
    results_per_query = 50

    corpus_ids = [f"doc_{i:04d}" for i in range(corpus_size)]
    results = {}
    relevance = {}

    for q in range(n_queries):
        query_id = f"query_{q:03d}"
        # Random ranking
        results[query_id] = list(
            np.random.choice(corpus_ids, results_per_query, replace=False)
        )
        # Random relevance (3-10 relevant docs)
        n_relevant = np.random.randint(3, 11)
        relevance[query_id] = set(
            np.random.choice(corpus_ids, n_relevant, replace=False)
        )

    return results, relevance

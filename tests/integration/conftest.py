"""Integration test fixtures for SHELF evaluation.

Provides small in-memory datasets and prediction files for testing
real evaluator behavior without mocking.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest


@pytest.fixture
def small_classification_dataset() -> pl.DataFrame:
    """Create a small dataset for classification testing.

    Returns a Polars DataFrame with the required fields for classification tasks.
    """
    return pl.DataFrame(
        {
            "id": [f"doc_{i:03d}" for i in range(20)],
            "text": [
                "The theory of evolution explains biological diversity",
                "Quantum mechanics describes subatomic particle behavior",
                "Shakespeare wrote many famous plays and sonnets",
                "The stock market crashed in 1929",
                "Beethoven composed nine symphonies",
                "The heart pumps blood through the circulatory system",
                "Congress passed the Civil Rights Act in 1964",
                "Machine learning uses algorithms to find patterns",
                "The Roman Empire fell in 476 AD",
                "DNA contains genetic information for living organisms",
                "Impressionism emerged as an art movement in France",
                "The Constitution defines the structure of government",
                "Plants convert sunlight to energy through photosynthesis",
                "The Industrial Revolution transformed manufacturing",
                "Mozart was a child prodigy composer",
                "Atoms are the building blocks of matter",
                "The Renaissance began in Italy",
                "Vaccines help prevent infectious diseases",
                "The Supreme Court interprets constitutional law",
                "Newton discovered the laws of motion",
            ],
            "lcc_code": [
                "Q",
                "Q",
                "P",
                "H",
                "M",  # Science, Science, Literature, Social Sciences, Music
                "R",
                "J",
                "Q",
                "D",
                "Q",  # Medicine, Political Science, Science, World History, Science
                "N",
                "J",
                "Q",
                "T",
                "M",  # Fine Arts, Political Science, Science, Technology, Music
                "Q",
                "D",
                "R",
                "K",
                "Q",  # Science, World History, Medicine, Law, Science
            ],
            "form": ["lecture"] * 20,
            "form_category": ["Instructional and educational works"] * 20,
            "topic": ["General"] * 20,
            "region": ["North America"] * 20,
            "audience": ["General audience"] * 20,
            "register": ["academic"] * 20,
        }
    )


@pytest.fixture
def small_retrieval_dataset() -> pl.DataFrame:
    """Create a small dataset for retrieval testing."""
    return pl.DataFrame(
        {
            "id": [f"doc_{i:03d}" for i in range(10)],
            "text": [
                "Biology studies living organisms and their interactions",
                "Chemistry examines the composition of matter",
                "Literature explores human experience through writing",
                "Economics analyzes production and consumption",
                "Music theory covers harmony and composition",
                "Medical research advances healthcare treatments",
                "Political theory examines power and governance",
                "Computer science studies computation and algorithms",
                "History chronicles past events and civilizations",
                "Physics investigates the fundamental forces of nature",
            ],
            "lcc_code": ["Q", "Q", "P", "H", "M", "R", "J", "Q", "D", "Q"],
            "form": ["lecture"] * 10,
        }
    )


@pytest.fixture
def classification_predictions() -> list[dict[str, Any]]:
    """Create sample classification predictions."""
    # Predictions with some correct, some incorrect
    return [
        {"id": "doc_000", "prediction": "Q"},  # Correct
        {"id": "doc_001", "prediction": "Q"},  # Correct
        {"id": "doc_002", "prediction": "P"},  # Correct
        {"id": "doc_003", "prediction": "H"},  # Correct
        {"id": "doc_004", "prediction": "M"},  # Correct
        {"id": "doc_005", "prediction": "Q"},  # Wrong (should be R)
        {"id": "doc_006", "prediction": "J"},  # Correct
        {"id": "doc_007", "prediction": "T"},  # Wrong (should be Q)
        {"id": "doc_008", "prediction": "D"},  # Correct
        {"id": "doc_009", "prediction": "Q"},  # Correct
        {"id": "doc_010", "prediction": "N"},  # Correct
        {"id": "doc_011", "prediction": "K"},  # Wrong (should be J)
        {"id": "doc_012", "prediction": "Q"},  # Correct
        {"id": "doc_013", "prediction": "T"},  # Correct
        {"id": "doc_014", "prediction": "M"},  # Correct
        {"id": "doc_015", "prediction": "Q"},  # Correct
        {"id": "doc_016", "prediction": "D"},  # Correct
        {"id": "doc_017", "prediction": "R"},  # Correct
        {"id": "doc_018", "prediction": "J"},  # Wrong (should be K)
        {"id": "doc_019", "prediction": "Q"},  # Correct
    ]


@pytest.fixture
def retrieval_predictions() -> list[dict[str, Any]]:
    """Create sample retrieval predictions (ranked doc IDs per query)."""
    return [
        {
            "query_id": "doc_000",
            "ranked_doc_ids": ["doc_001", "doc_007", "doc_009", "doc_002", "doc_003"],
        },
        {
            "query_id": "doc_001",
            "ranked_doc_ids": ["doc_000", "doc_007", "doc_009", "doc_005", "doc_008"],
        },
        {
            "query_id": "doc_002",
            "ranked_doc_ids": ["doc_003", "doc_004", "doc_000", "doc_001", "doc_005"],
        },
    ]


@pytest.fixture
def predictions_jsonl_file(classification_predictions, tmp_path) -> Path:
    """Create a temporary JSONL file with classification predictions."""
    pred_file = tmp_path / "predictions.jsonl"
    with open(pred_file, "w") as f:
        for pred in classification_predictions:
            f.write(json.dumps(pred) + "\n")
    return pred_file


@pytest.fixture
def retrieval_predictions_jsonl_file(retrieval_predictions, tmp_path) -> Path:
    """Create a temporary JSONL file with retrieval predictions."""
    pred_file = tmp_path / "retrieval_predictions.jsonl"
    with open(pred_file, "w") as f:
        for pred in retrieval_predictions:
            f.write(json.dumps(pred) + "\n")
    return pred_file


class MockEmbedder:
    """Simple mock embedder that returns deterministic embeddings.

    Uses document text hash to generate consistent embeddings,
    ensuring similar texts get similar embeddings.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim
        self._rng = np.random.default_rng(42)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Generate deterministic embeddings based on text content."""
        embeddings = []
        for text in texts:
            # Use hash of text for deterministic but content-based embeddings
            seed = hash(text[:50]) % (2**32)
            rng = np.random.default_rng(seed)
            emb = rng.standard_normal(self.dim)
            # Normalize
            emb = emb / np.linalg.norm(emb)
            embeddings.append(emb)
        return np.array(embeddings, dtype=np.float32)


@pytest.fixture
def mock_embedder() -> MockEmbedder:
    """Provide a mock embedder for testing."""
    return MockEmbedder(dim=64)


@pytest.fixture
def output_dir(tmp_path) -> Path:
    """Provide a temporary output directory."""
    out = tmp_path / "output"
    out.mkdir()
    return out

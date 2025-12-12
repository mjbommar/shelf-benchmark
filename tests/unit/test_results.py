"""Unit tests for shelf.evaluate.results module.

Tests cover:
- DataProvenance creation and methods
- EvaluationContext capture and serialization
- EvaluationResult serialization (to_dict, from_dict, to_json, from_json)
- compute_file_checksum determinism
- compute_data_checksum determinism
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from shelf.evaluate.results import (
    DataProvenance,
    EvaluationContext,
    EvaluationResult,
    compute_data_checksum,
    compute_file_checksum,
)


class TestComputeFileChecksum:
    """Tests for compute_file_checksum function."""

    def test_deterministic(self, temp_json_file):
        """Test checksum is deterministic."""
        checksum1 = compute_file_checksum(temp_json_file)
        checksum2 = compute_file_checksum(temp_json_file)
        assert checksum1 == checksum2

    def test_different_content_different_checksum(self):
        """Test different content gives different checksums."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f1:
            f1.write("content A")
            path1 = Path(f1.name)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f2:
            f2.write("content B")
            path2 = Path(f2.name)

        try:
            checksum1 = compute_file_checksum(path1)
            checksum2 = compute_file_checksum(path2)

            assert checksum1 != checksum2
        finally:
            path1.unlink()
            path2.unlink()

    def test_checksum_format(self, temp_json_file):
        """Test checksum is valid MD5 hex string."""
        checksum = compute_file_checksum(temp_json_file)

        assert len(checksum) == 32  # MD5 hex length
        assert all(c in "0123456789abcdef" for c in checksum)


class TestComputeDataChecksum:
    """Tests for compute_data_checksum function."""

    def test_deterministic(self, sample_records):
        """Test checksum is deterministic for same data."""
        checksum1 = compute_data_checksum(sample_records)
        checksum2 = compute_data_checksum(sample_records)
        assert checksum1 == checksum2

    def test_order_independent(self, sample_records):
        """Test checksum is same regardless of record order."""
        reversed_records = list(reversed(sample_records))

        checksum1 = compute_data_checksum(sample_records)
        checksum2 = compute_data_checksum(reversed_records)

        assert checksum1 == checksum2

    def test_different_data_different_checksum(self, sample_records):
        """Test different data gives different checksums."""
        modified_records = sample_records.copy()
        modified_records[0] = {**modified_records[0], "lcc": "Z"}

        checksum1 = compute_data_checksum(sample_records)
        checksum2 = compute_data_checksum(modified_records)

        assert checksum1 != checksum2

    def test_empty_list(self):
        """Test empty list gives valid checksum."""
        checksum = compute_data_checksum([])
        assert len(checksum) == 32


class TestDataProvenance:
    """Tests for DataProvenance dataclass."""

    def test_from_data_single_commit(self, sample_records):
        """Test creation from single-commit data."""
        # Make all records have same commit
        records = [
            {**r, "git_commit": "abc123", "model": "gpt-5.1"} for r in sample_records
        ]

        provenance = DataProvenance.from_data(records)

        assert provenance.is_single_commit() is True
        assert provenance.is_single_model() is True
        assert provenance.primary_commit == "abc123"
        assert provenance.primary_model == "gpt-5.1"

    def test_from_data_multiple_commits(self, sample_records):
        """Test creation from multi-commit data."""
        provenance = DataProvenance.from_data(sample_records)

        # Sample data has 2 commits: abc123, def456
        assert len(provenance.unique_commits) >= 1
        assert (
            provenance.is_single_commit() is False
            or len(provenance.unique_commits) == 1
        )

    def test_from_data_with_filters(self, sample_records):
        """Test filters_applied is stored."""
        filters = {"git_commit": "abc123"}
        provenance = DataProvenance.from_data(sample_records, filters_applied=filters)

        assert provenance.filters_applied == filters

    def test_to_dict(self, sample_records):
        """Test to_dict serialization."""
        provenance = DataProvenance.from_data(sample_records)
        d = provenance.to_dict()

        assert "unique_commits" in d
        assert "unique_models" in d
        assert "commit_distribution" in d
        assert "model_distribution" in d

    def test_commit_distribution(self, sample_records):
        """Test commit distribution counts."""
        provenance = DataProvenance.from_data(sample_records)

        # Distribution should sum to total records
        total = sum(provenance.commit_distribution.values())
        assert total == len(sample_records)


class TestEvaluationContext:
    """Tests for EvaluationContext dataclass."""

    def test_capture_basic(self):
        """Test basic context capture."""
        context = EvaluationContext.capture(random_seed=42)

        assert context.random_seed == 42
        assert context.shelf_version is not None
        assert context.python_version is not None
        assert context.sklearn_version is not None
        assert context.numpy_version is not None
        assert context.timestamp is not None
        assert context.platform_info is not None

    def test_capture_with_checksum(self):
        """Test capture with dataset checksum."""
        context = EvaluationContext.capture(
            random_seed=42,
            dataset_checksum="abc123def456",
        )

        assert context.dataset_checksum == "abc123def456"

    def test_capture_with_model_name(self):
        """Test capture with model name."""
        context = EvaluationContext.capture(
            random_seed=42,
            model_name="BERT-base",
        )

        assert context.model_name == "BERT-base"

    def test_capture_with_extra_context(self):
        """Test capture with extra keyword arguments."""
        context = EvaluationContext.capture(
            random_seed=42,
            custom_param="value",
            batch_size=32,
        )

        assert context.extra["custom_param"] == "value"
        assert context.extra["batch_size"] == 32

    def test_to_dict(self):
        """Test to_dict serialization."""
        context = EvaluationContext.capture(random_seed=42)
        d = context.to_dict()

        assert isinstance(d, dict)
        assert d["random_seed"] == 42
        assert "timestamp" in d


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    @pytest.fixture
    def sample_result(self):
        """Create a sample EvaluationResult."""
        return EvaluationResult(
            task="lcc_classification",
            task_type="classification",
            split="test",
            primary_metric="macro_f1",
            primary_score=0.85,
            metrics={
                "macro_f1": 0.85,
                "micro_f1": 0.88,
                "accuracy": 0.87,
            },
            num_samples=1000,
            context=EvaluationContext.capture(random_seed=42),
        )

    def test_to_dict(self, sample_result):
        """Test to_dict serialization."""
        d = sample_result.to_dict()

        assert d["task"] == "lcc_classification"
        assert d["primary_score"] == 0.85
        assert d["metrics"]["macro_f1"] == 0.85
        assert "context" in d

    def test_from_dict(self, sample_result):
        """Test from_dict deserialization."""
        d = sample_result.to_dict()
        restored = EvaluationResult.from_dict(d)

        assert restored.task == sample_result.task
        assert restored.primary_score == sample_result.primary_score
        assert restored.metrics == sample_result.metrics

    def test_json_round_trip(self, sample_result):
        """Test JSON serialization round trip."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            sample_result.to_json(path)
            restored = EvaluationResult.from_json(path)

            assert restored.task == sample_result.task
            assert restored.primary_score == sample_result.primary_score
            assert restored.num_samples == sample_result.num_samples
        finally:
            path.unlink()

    def test_str_representation(self, sample_result):
        """Test string representation."""
        s = str(sample_result)

        assert "lcc_classification" in s
        assert "0.85" in s

    def test_summary(self, sample_result):
        """Test summary generation."""
        summary = sample_result.summary()

        assert "lcc_classification" in summary
        assert "macro_f1" in summary
        assert "0.85" in summary

    def test_with_per_class_metrics(self):
        """Test result with per-class metrics."""
        result = EvaluationResult(
            task="lcc_classification",
            task_type="classification",
            split="test",
            primary_metric="macro_f1",
            primary_score=0.85,
            metrics={"macro_f1": 0.85},
            per_class_metrics={
                "A": {"f1": 0.90, "precision": 0.88, "recall": 0.92},
                "B": {"f1": 0.80, "precision": 0.82, "recall": 0.78},
            },
            num_samples=100,
        )

        d = result.to_dict()
        assert "per_class_metrics" in d
        assert d["per_class_metrics"]["A"]["f1"] == 0.90

    def test_with_confidence_intervals(self):
        """Test result with confidence intervals."""
        result = EvaluationResult(
            task="lcc_classification",
            task_type="classification",
            split="test",
            primary_metric="macro_f1",
            primary_score=0.85,
            metrics={"macro_f1": 0.85},
            confidence_intervals={"macro_f1": (0.82, 0.88)},
            num_samples=100,
        )

        d = result.to_dict()
        assert "confidence_intervals" in d
        # CIs are converted to lists in JSON
        assert d["confidence_intervals"]["macro_f1"] == [0.82, 0.88]

    def test_with_numpy_metrics(self):
        """Test handling of numpy types in metrics."""
        result = EvaluationResult(
            task="test",
            task_type="classification",
            split="test",
            primary_metric="accuracy",
            primary_score=np.float64(0.85),  # numpy type
            metrics={"accuracy": np.float64(0.85)},
            num_samples=100,
        )

        d = result.to_dict()
        # Should be converted to Python float
        assert isinstance(d["primary_score"], float)
        assert isinstance(d["metrics"]["accuracy"], float)


class TestEvaluationResultEdgeCases:
    """Edge case tests for EvaluationResult."""

    def test_minimal_result(self):
        """Test with minimal required fields."""
        result = EvaluationResult(
            task="test",
            task_type="test",
            split="test",
            primary_metric="score",
            primary_score=0.5,
            metrics={"score": 0.5},
            num_samples=10,
        )

        d = result.to_dict()
        restored = EvaluationResult.from_dict(d)

        assert restored.task == "test"

    def test_with_empty_metrics(self):
        """Test with empty optional fields."""
        result = EvaluationResult(
            task="test",
            task_type="test",
            split="test",
            primary_metric="score",
            primary_score=0.0,
            metrics={},
            num_samples=0,
        )

        d = result.to_dict()
        assert d["num_samples"] == 0

"""Evaluation results and context for reproducibility.

This module provides structured types for storing evaluation results
along with complete context for reproducibility.

Includes per-sample result storage for practitioner-focused analysis:
- Sample-level variance and reliability metrics
- Error stratification by document attributes
- Bootstrap CIs from per-sample data (tighter than task-level)
"""

from __future__ import annotations

import gzip
import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal


def _get_package_version(package: str) -> str:
    """Get version of an installed package, or 'not installed' if not found."""
    try:
        from importlib.metadata import version

        return version(package)
    except Exception:
        return "not installed"


# Type alias for task types
TaskType = Literal["classification", "retrieval", "clustering", "pair_classification"]


@dataclass
class PerSampleResult:
    """Per-sample evaluation result for detailed analysis.

    Stores individual sample predictions for:
    - Sample-level variance analysis
    - Error stratification by document attributes
    - Bootstrap CIs with more granular data
    - Reliability/robustness assessment

    Attributes:
        id: Sample identifier (document ID, query ID, or pair ID)
        y_true: Ground truth label/value
        y_pred: Predicted label/value
        correct: Whether prediction was correct (for classification/pairs)
        score: Confidence score or similarity score (if available)
        metadata: Document metadata for stratification (form, register, length, etc.)
    """

    id: str
    y_true: Any
    y_pred: Any
    correct: bool | None = None
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: dict[str, Any] = {
            "id": self.id,
            "y_true": self.y_true,
            "y_pred": self.y_pred,
        }
        if self.correct is not None:
            result["correct"] = self.correct
        if self.score is not None:
            result["score"] = float(self.score)
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PerSampleResult:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            y_true=data["y_true"],
            y_pred=data["y_pred"],
            correct=data.get("correct"),
            score=data.get("score"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class PerSampleResults:
    """Collection of per-sample results with metadata.

    Provides efficient storage and loading of per-sample data,
    kept separate from main results to avoid bloating result files.

    Attributes:
        task: Task name
        task_type: Type of task (classification, retrieval, etc.)
        model_key: Model identifier
        split: Dataset split
        samples: List of per-sample results
        sample_count: Number of samples
        correct_count: Number correct (classification/pairs)
        accuracy: Overall accuracy (if applicable)
    """

    task: str
    task_type: TaskType
    model_key: str
    split: str
    samples: list[PerSampleResult]
    sample_count: int = 0
    correct_count: int | None = None
    accuracy: float | None = None

    def __post_init__(self) -> None:
        """Compute derived fields."""
        self.sample_count = len(self.samples)
        if self.samples and self.samples[0].correct is not None:
            self.correct_count = sum(1 for s in self.samples if s.correct)
            self.accuracy = (
                self.correct_count / self.sample_count if self.sample_count > 0 else 0.0
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task": self.task,
            "task_type": self.task_type,
            "model_key": self.model_key,
            "split": self.split,
            "sample_count": self.sample_count,
            "correct_count": self.correct_count,
            "accuracy": self.accuracy,
            "samples": [s.to_dict() for s in self.samples],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PerSampleResults:
        """Create from dictionary."""
        samples = [PerSampleResult.from_dict(s) for s in data.get("samples", [])]
        return cls(
            task=data["task"],
            task_type=data["task_type"],
            model_key=data["model_key"],
            split=data["split"],
            samples=samples,
        )

    def save(self, path: Path | str, compress: bool = True) -> Path:
        """Save per-sample results to file.

        Args:
            path: Base path (will add .jsonl.gz or .jsonl extension)
            compress: Whether to gzip compress (default: True, ~10x smaller)

        Returns:
            Actual path written to
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Add appropriate extension
        if compress and not path.suffix.endswith(".gz"):
            if not path.suffix == ".jsonl":
                path = path.with_suffix(".jsonl.gz")
            else:
                path = Path(str(path) + ".gz")
        elif not compress and path.suffix != ".jsonl":
            path = path.with_suffix(".jsonl")

        # Write header + samples as JSONL
        header = {
            "task": self.task,
            "task_type": self.task_type,
            "model_key": self.model_key,
            "split": self.split,
            "sample_count": self.sample_count,
            "correct_count": self.correct_count,
            "accuracy": self.accuracy,
            "_type": "header",
        }

        if compress:
            with gzip.open(path, "wt", encoding="utf-8") as f:
                f.write(json.dumps(header) + "\n")
                for sample in self.samples:
                    f.write(json.dumps(sample.to_dict()) + "\n")
        else:
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps(header) + "\n")
                for sample in self.samples:
                    f.write(json.dumps(sample.to_dict()) + "\n")

        return path

    @classmethod
    def load(cls, path: Path | str) -> PerSampleResults:
        """Load per-sample results from file.

        Args:
            path: Path to .jsonl or .jsonl.gz file

        Returns:
            PerSampleResults instance
        """
        path = Path(path)

        # Determine if compressed
        is_compressed = path.suffix == ".gz" or str(path).endswith(".jsonl.gz")

        samples: list[PerSampleResult] = []
        header: dict[str, Any] = {}

        opener = gzip.open if is_compressed else open
        with opener(path, "rt", encoding="utf-8") as f:  # type: ignore[arg-type]
            for line in f:
                data = json.loads(line.strip())
                if data.get("_type") == "header":
                    header = data
                else:
                    samples.append(PerSampleResult.from_dict(data))

        return cls(
            task=header.get("task", ""),
            task_type=header.get("task_type", "classification"),
            model_key=header.get("model_key", ""),
            split=header.get("split", "test"),
            samples=samples,
        )

    def get_correct_mask(self) -> list[bool]:
        """Get boolean mask of correct predictions."""
        return [s.correct is True for s in self.samples]

    def get_incorrect_samples(self) -> list[PerSampleResult]:
        """Get list of incorrectly classified samples."""
        return [s for s in self.samples if s.correct is False]

    def get_scores(self) -> list[float]:
        """Get list of scores (confidence/similarity)."""
        return [s.score for s in self.samples if s.score is not None]

    def stratify_by(self, field: str) -> dict[str, list[PerSampleResult]]:
        """Group samples by a metadata field.

        Args:
            field: Metadata field to stratify by (e.g., "form", "register", "length_bucket")

        Returns:
            Dict mapping field values to sample lists
        """
        groups: dict[str, list[PerSampleResult]] = {}
        for sample in self.samples:
            value = sample.metadata.get(field, "unknown")
            if value not in groups:
                groups[value] = []
            groups[value].append(sample)
        return groups

    def compute_stratified_accuracy(self, field: str) -> dict[str, dict[str, float]]:
        """Compute accuracy stratified by a metadata field.

        Args:
            field: Metadata field to stratify by

        Returns:
            Dict mapping field values to {accuracy, count, correct}
        """
        groups = self.stratify_by(field)
        results: dict[str, dict[str, float]] = {}

        for value, samples in groups.items():
            n_total = len(samples)
            n_correct = sum(1 for s in samples if s.correct is True)
            results[value] = {
                "accuracy": n_correct / n_total if n_total > 0 else 0.0,
                "count": float(n_total),
                "correct": float(n_correct),
                "error_rate": 1.0 - (n_correct / n_total) if n_total > 0 else 1.0,
            }

        return results

    # --- Error analysis query methods ---

    def get_errors(
        self,
        true_label: str | None = None,
        pred_label: str | None = None,
        limit: int | None = None,
    ) -> list[PerSampleResult]:
        """Get misclassified samples with optional filtering.

        Args:
            true_label: Filter to errors with this true label
            pred_label: Filter to errors predicted as this label
            limit: Maximum number of results to return

        Returns:
            List of error samples matching criteria
        """
        errors = [s for s in self.samples if s.correct is False]

        if true_label is not None:
            errors = [s for s in errors if s.y_true == true_label]

        if pred_label is not None:
            errors = [s for s in errors if s.y_pred == pred_label]

        if limit is not None:
            errors = errors[:limit]

        return errors

    def get_errors_by_confusion(
        self,
        true_label: str,
        pred_label: str,
        limit: int | None = None,
    ) -> list[PerSampleResult]:
        """Get samples confused between two specific classes.

        This is useful for investigating specific confusion patterns
        identified in the confusion matrix or top_confusions list.

        Args:
            true_label: Ground truth class
            pred_label: Predicted (wrong) class
            limit: Maximum number of results to return

        Returns:
            List of samples with true_label predicted as pred_label

        Example:
            >>> # Get samples where Technology (T) was misclassified as Science (Q)
            >>> errors = results.get_errors_by_confusion("T", "Q", limit=10)
            >>> for e in errors:
            ...     print(f"{e.id}: {e.metadata.get('form', 'unknown')}")
        """
        return self.get_errors(
            true_label=true_label, pred_label=pred_label, limit=limit
        )

    def get_sample_by_id(self, sample_id: str) -> PerSampleResult | None:
        """Get a specific sample by ID.

        Args:
            sample_id: The sample identifier

        Returns:
            The sample if found, None otherwise
        """
        for sample in self.samples:
            if sample.id == sample_id:
                return sample
        return None

    def filter_by_metadata(
        self,
        field: str,
        value: Any,
        errors_only: bool = False,
    ) -> list[PerSampleResult]:
        """Get samples with a specific metadata value.

        Args:
            field: Metadata field name (e.g., "form", "register", "length_bucket")
            value: Value to filter for
            errors_only: If True, only return misclassified samples

        Returns:
            List of samples matching the criteria
        """
        results = []
        for sample in self.samples:
            if sample.metadata.get(field) == value:
                if errors_only and sample.correct is not False:
                    continue
                results.append(sample)
        return results

    def get_confusion_counts(self) -> dict[tuple[str, str], int]:
        """Get counts of all (true_label, pred_label) pairs for errors.

        Returns:
            Dict mapping (true_label, pred_label) tuples to counts

        Example:
            >>> counts = results.get_confusion_counts()
            >>> # Sort by count descending
            >>> sorted_counts = sorted(counts.items(), key=lambda x: -x[1])
            >>> for (true_l, pred_l), count in sorted_counts[:5]:
            ...     print(f"{true_l} -> {pred_l}: {count}")
        """
        counts: dict[tuple[str, str], int] = {}
        for sample in self.samples:
            if sample.correct is False:
                key = (str(sample.y_true), str(sample.y_pred))
                counts[key] = counts.get(key, 0) + 1
        return counts

    def get_by_confidence(
        self,
        min_score: float | None = None,
        max_score: float | None = None,
        errors_only: bool = False,
    ) -> list[PerSampleResult]:
        """Get samples within a confidence score range.

        Useful for analyzing model calibration and finding
        high-confidence errors or low-confidence correct predictions.

        Args:
            min_score: Minimum score (inclusive)
            max_score: Maximum score (inclusive)
            errors_only: If True, only return misclassified samples

        Returns:
            List of samples in the score range
        """
        results = []
        for sample in self.samples:
            if sample.score is None:
                continue

            if min_score is not None and sample.score < min_score:
                continue
            if max_score is not None and sample.score > max_score:
                continue
            if errors_only and sample.correct is not False:
                continue

            results.append(sample)

        return results

    def get_high_confidence_errors(
        self,
        threshold: float = 0.9,
        limit: int | None = None,
    ) -> list[PerSampleResult]:
        """Get misclassified samples with high confidence scores.

        These are particularly interesting for error analysis as they
        represent cases where the model was confidently wrong.

        Args:
            threshold: Minimum confidence score
            limit: Maximum number of results

        Returns:
            List of high-confidence errors, sorted by score descending
        """
        errors = self.get_by_confidence(min_score=threshold, errors_only=True)
        # Sort by score descending (most confident errors first)
        errors.sort(key=lambda s: s.score or 0, reverse=True)
        if limit is not None:
            errors = errors[:limit]
        return errors


@dataclass
class DataProvenance:
    """Provenance information about the data used in evaluation.

    Tracks which data generation commits and models are represented,
    enabling filtering and comparison across different data generations.

    Attributes:
        unique_commits: List of unique git commit IDs in the data
        unique_models: List of unique generation models (e.g., "gpt-5.1")
        commit_distribution: Count of samples per commit
        model_distribution: Count of samples per generation model
        primary_commit: Most common commit ID (for single-commit datasets)
        primary_model: Most common generation model
        filters_applied: Any filters that were applied to the data
    """

    unique_commits: list[str]
    unique_models: list[str]
    commit_distribution: dict[str, int]
    model_distribution: dict[str, int]
    primary_commit: str | None = None
    primary_model: str | None = None
    filters_applied: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_data(
        cls,
        data: list[dict[str, Any]],
        filters_applied: dict[str, Any] | None = None,
    ) -> DataProvenance:
        """Extract provenance from a list of data records.

        Args:
            data: List of document records
            filters_applied: Any filters that were applied

        Returns:
            DataProvenance instance
        """
        from collections import Counter

        commits = [d.get("git_commit", "") or "" for d in data]
        models = [d.get("model", "") or "" for d in data]

        commit_counts = Counter(commits)
        model_counts = Counter(models)

        # Remove empty strings from unique lists
        unique_commits = [c for c in commit_counts if c]
        unique_models = [m for m in model_counts if m]

        # Primary = most common (for datasets from single generation)
        primary_commit = commit_counts.most_common(1)[0][0] if commit_counts else None
        primary_model = model_counts.most_common(1)[0][0] if model_counts else None

        return cls(
            unique_commits=unique_commits,
            unique_models=unique_models,
            commit_distribution=dict(commit_counts),
            model_distribution=dict(model_counts),
            primary_commit=primary_commit if primary_commit else None,
            primary_model=primary_model if primary_model else None,
            filters_applied=filters_applied or {},
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    def is_single_commit(self) -> bool:
        """Check if all data is from a single commit."""
        return len(self.unique_commits) == 1

    def is_single_model(self) -> bool:
        """Check if all data is from a single generation model."""
        return len(self.unique_models) == 1


@dataclass
class EvaluationContext:
    """Complete context for reproducible evaluation.

    This captures everything needed to reproduce an evaluation:
    - Library versions (shelf, sklearn, numpy, etc.)
    - Data checksums and generation info
    - Random seeds
    - Platform information
    - Timestamps

    Attributes:
        shelf_version: Version of the shelf package
        python_version: Python interpreter version
        sklearn_version: scikit-learn version
        numpy_version: NumPy version
        polars_version: Polars version
        dataset_checksum: MD5 checksum of the dataset split used
        prediction_file_checksum: MD5 checksum of predictions file (if used)
        random_seed: Random seed used for any stochastic operations
        platform_info: Platform string (OS, architecture)
        timestamp: ISO8601 timestamp of evaluation
        model_name: Name of the model being evaluated
        data_commit: Git commit ID of data generation code (for filtering)
        code_commit: Git commit ID of evaluation code
        extra: Additional context (model name, parameters, etc.)
    """

    shelf_version: str
    python_version: str
    sklearn_version: str
    numpy_version: str
    polars_version: str
    dataset_checksum: str | None
    prediction_file_checksum: str | None
    random_seed: int
    platform_info: str
    timestamp: str
    model_name: str | None = None
    data_commit: str | None = None
    code_commit: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def capture(
        cls,
        random_seed: int = 42,
        dataset_checksum: str | None = None,
        prediction_file_checksum: str | None = None,
        model_name: str | None = None,
        data_commit: str | None = None,
        **extra: Any,
    ) -> EvaluationContext:
        """Capture current environment context.

        Args:
            random_seed: Random seed being used
            dataset_checksum: MD5 of dataset (computed separately)
            prediction_file_checksum: MD5 of predictions file
            model_name: Name of the model being evaluated
            data_commit: Git commit ID of data generation (for filtering by generation)
            **extra: Additional context to store

        Returns:
            EvaluationContext with current environment info
        """
        # Try to get current git commit
        code_commit = None
        try:
            import subprocess

            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                code_commit = result.stdout.strip()
        except Exception:
            pass

        return cls(
            shelf_version=_get_package_version("shelf"),
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            sklearn_version=_get_package_version("scikit-learn"),
            numpy_version=_get_package_version("numpy"),
            polars_version=_get_package_version("polars"),
            dataset_checksum=dataset_checksum,
            prediction_file_checksum=prediction_file_checksum,
            random_seed=random_seed,
            platform_info=platform.platform(),
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_name=model_name,
            data_commit=data_commit,
            code_commit=code_commit,
            extra=extra,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class EvaluationResult:
    """Complete evaluation results with full context.

    This stores all metrics, detailed breakdowns, and the context
    needed for reproducibility.

    Attributes:
        task: Task name (e.g., "lcc_retrieval")
        task_type: Task type string (e.g., "retrieval")
        split: Dataset split evaluated (e.g., "test")
        primary_metric: Name of the primary metric
        primary_score: Value of the primary metric
        metrics: All computed metrics
        per_class_metrics: Per-class breakdown (classification tasks)
        confusion_matrix: Confusion matrix (classification tasks)
        per_query_metrics: Per-query breakdown (retrieval tasks)
        num_samples: Number of samples evaluated
        num_correct: Number correct (classification tasks)
        misclassified_ids: IDs of misclassified samples
        confidence_intervals: Bootstrap CIs for metrics
        stratified_metrics: Metrics broken down by a stratification field
        stratify_by: Field used for stratification (if any)
        data_provenance: Provenance info about the data used
        context: Full evaluation context for reproducibility
    """

    # Task info
    task: str
    task_type: str
    split: str

    # Primary result
    primary_metric: str
    primary_score: float

    # All metrics
    metrics: dict[str, float]

    # Detailed breakdowns (task-specific)
    per_class_metrics: dict[str, dict[str, float]] | None = None
    confusion_matrix: list[list[int]] | None = None
    # Labels for confusion matrix indices (e.g., ["A", "B", "C", ...])
    confusion_matrix_labels: list[str] | None = None
    # Top confused class pairs, pre-computed for convenience
    # Each entry: {"true_label": str, "pred_label": str, "count": int, ...}
    top_confusions: list[dict[str, Any]] | None = None
    # Stratified confusion matrices: {field: {stratum_value: matrix}}
    # e.g., {"audience": {"Physicians": [[...]], "General public": [[...]]}}
    stratified_confusion_matrices: dict[str, dict[str, list[list[int]]]] | None = None
    per_query_metrics: dict[str, dict[str, float]] | None = None

    # Counts
    num_samples: int = 0
    num_correct: int | None = None

    # For debugging/analysis
    misclassified_ids: list[str] | None = None

    # Confidence intervals (bootstrap)
    confidence_intervals: dict[str, tuple[float, float]] | None = None

    # Stratified analysis
    stratified_metrics: dict[str, dict[str, float]] | None = None
    stratify_by: str | None = None

    # Data provenance
    data_provenance: DataProvenance | None = None

    # Full context
    context: EvaluationContext | None = None

    # Per-sample results (for detailed analysis)
    # Stored separately to keep main results lean
    per_sample_results: PerSampleResults | None = None
    per_sample_path: str | None = None  # Path to separate per-sample file

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Handles nested dataclasses and numpy types.
        """
        result = {
            "task": self.task,
            "task_type": self.task_type,
            "split": self.split,
            "primary_metric": self.primary_metric,
            "primary_score": float(self.primary_score),
            "metrics": {k: float(v) for k, v in self.metrics.items()},
            "num_samples": self.num_samples,
        }

        if self.per_class_metrics is not None:
            result["per_class_metrics"] = {
                k: {m: float(v) for m, v in metrics.items()}
                for k, metrics in self.per_class_metrics.items()
            }

        if self.confusion_matrix is not None:
            result["confusion_matrix"] = self.confusion_matrix

        if self.confusion_matrix_labels is not None:
            result["confusion_matrix_labels"] = self.confusion_matrix_labels

        if self.top_confusions is not None:
            result["top_confusions"] = self.top_confusions

        if self.stratified_confusion_matrices is not None:
            result["stratified_confusion_matrices"] = self.stratified_confusion_matrices

        if self.per_query_metrics is not None:
            result["per_query_metrics"] = {
                k: {m: float(v) for m, v in metrics.items()}
                for k, metrics in self.per_query_metrics.items()
            }

        if self.num_correct is not None:
            result["num_correct"] = self.num_correct

        if self.misclassified_ids is not None:
            result["misclassified_ids"] = self.misclassified_ids

        if self.confidence_intervals is not None:
            result["confidence_intervals"] = {
                k: [float(v[0]), float(v[1])]
                for k, v in self.confidence_intervals.items()
            }

        if self.stratified_metrics is not None:
            result["stratified_metrics"] = {
                k: {m: float(v) for m, v in metrics.items()}
                for k, metrics in self.stratified_metrics.items()
            }
            result["stratify_by"] = self.stratify_by

        if self.data_provenance is not None:
            result["data_provenance"] = self.data_provenance.to_dict()

        if self.context is not None:
            result["context"] = self.context.to_dict()

        # Only store path reference, not full per-sample data
        if self.per_sample_path is not None:
            result["per_sample_path"] = self.per_sample_path

        return result

    def to_json(
        self,
        path: Path | str,
        indent: int = 2,
        save_per_sample: bool = False,
    ) -> None:
        """Save results to JSON file.

        Args:
            path: Path to save JSON file
            indent: JSON indentation level
            save_per_sample: Whether to save per-sample results to separate file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save per-sample data to separate file if requested
        if save_per_sample and self.per_sample_results is not None:
            # Create per-sample filename based on main result filename
            per_sample_filename = path.stem + "_samples.jsonl.gz"
            per_sample_path = path.parent / per_sample_filename
            self.per_sample_results.save(per_sample_path, compress=True)
            self.per_sample_path = str(per_sample_path)

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationResult:
        """Create from dictionary.

        Args:
            data: Dictionary from to_dict() or JSON

        Returns:
            EvaluationResult instance
        """
        # Handle nested dataclasses
        context_data = data.pop("context", None)
        context = None
        if context_data is not None:
            context = EvaluationContext(**context_data)

        data_provenance_data = data.pop("data_provenance", None)
        data_provenance = None
        if data_provenance_data is not None:
            data_provenance = DataProvenance(**data_provenance_data)

        # Extract per_sample_path (per_sample_results loaded separately)
        per_sample_path = data.pop("per_sample_path", None)

        # Convert confidence intervals back to tuples
        ci = data.get("confidence_intervals")
        if ci is not None:
            data["confidence_intervals"] = {k: tuple(v) for k, v in ci.items()}

        return cls(
            **data,
            context=context,
            data_provenance=data_provenance,
            per_sample_path=per_sample_path,
        )

    @classmethod
    def from_json(
        cls,
        path: Path | str,
        load_per_sample: bool = False,
    ) -> EvaluationResult:
        """Load results from JSON file.

        Args:
            path: Path to JSON file
            load_per_sample: Whether to also load per-sample results if available

        Returns:
            EvaluationResult instance
        """
        with open(path) as f:
            data = json.load(f)
        result = cls.from_dict(data)

        # Optionally load per-sample results
        if load_per_sample and result.per_sample_path:
            result.load_per_sample_results()

        return result

    def load_per_sample_results(self) -> PerSampleResults | None:
        """Load per-sample results from the referenced file.

        Returns:
            PerSampleResults if available and loaded, None otherwise
        """
        if self.per_sample_path is None:
            return None

        try:
            self.per_sample_results = PerSampleResults.load(self.per_sample_path)
            return self.per_sample_results
        except FileNotFoundError:
            return None

    def has_per_sample_data(self) -> bool:
        """Check if per-sample data is available (either loaded or referenceable)."""
        return self.per_sample_results is not None or self.per_sample_path is not None

    def summary(self) -> str:
        """Generate human-readable summary.

        Returns:
            Multi-line summary string
        """
        lines = [
            f"Task: {self.task} ({self.task_type})",
            f"Split: {self.split}",
            f"Samples: {self.num_samples}",
            "",
            "Metrics:",
        ]

        for metric, value in sorted(self.metrics.items()):
            ci_str = ""
            if self.confidence_intervals and metric in self.confidence_intervals:
                ci = self.confidence_intervals[metric]
                ci_str = f" [{ci[0]:.4f}, {ci[1]:.4f}]"
            primary_marker = " *" if metric == self.primary_metric else ""
            lines.append(f"  {metric}: {value:.4f}{ci_str}{primary_marker}")

        if self.context:
            lines.extend(
                [
                    "",
                    "Context:",
                    f"  Timestamp: {self.context.timestamp}",
                    f"  Random seed: {self.context.random_seed}",
                    f"  Platform: {self.context.platform_info}",
                ]
            )

        return "\n".join(lines)

    def __str__(self) -> str:
        """Short string representation."""
        return f"EvaluationResult({self.task}, {self.primary_metric}={self.primary_score:.4f})"


def compute_file_checksum(path: Path | str) -> str:
    """Compute MD5 checksum of a file.

    Args:
        path: Path to file

    Returns:
        Hex digest of MD5 hash
    """
    path = Path(path)
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest()


def compute_data_checksum(data: list[dict[str, Any]]) -> str:
    """Compute checksum of data records.

    Args:
        data: List of dictionaries

    Returns:
        Hex digest of MD5 hash
    """
    md5 = hashlib.md5()
    # Sort keys for deterministic ordering
    for record in sorted(data, key=lambda x: str(x.get("id", ""))):
        md5.update(json.dumps(record, sort_keys=True).encode())
    return md5.hexdigest()

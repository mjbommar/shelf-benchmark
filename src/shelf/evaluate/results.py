"""Evaluation results and context for reproducibility.

This module provides structured types for storing evaluation results
along with complete context for reproducibility.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _get_package_version(package: str) -> str:
    """Get version of an installed package, or 'not installed' if not found."""
    try:
        from importlib.metadata import version

        return version(package)
    except Exception:
        return "not installed"


@dataclass
class EvaluationContext:
    """Complete context for reproducible evaluation.

    This captures everything needed to reproduce an evaluation:
    - Library versions (shelf, sklearn, numpy, etc.)
    - Data checksums
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
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def capture(
        cls,
        random_seed: int = 42,
        dataset_checksum: str | None = None,
        prediction_file_checksum: str | None = None,
        **extra: Any,
    ) -> EvaluationContext:
        """Capture current environment context.

        Args:
            random_seed: Random seed being used
            dataset_checksum: MD5 of dataset (computed separately)
            prediction_file_checksum: MD5 of predictions file
            **extra: Additional context to store

        Returns:
            EvaluationContext with current environment info
        """
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
    per_query_metrics: dict[str, dict[str, float]] | None = None

    # Counts
    num_samples: int = 0
    num_correct: int | None = None

    # For debugging/analysis
    misclassified_ids: list[str] | None = None

    # Confidence intervals (bootstrap)
    confidence_intervals: dict[str, tuple[float, float]] | None = None

    # Full context
    context: EvaluationContext | None = None

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

        if self.context is not None:
            result["context"] = self.context.to_dict()

        return result

    def to_json(self, path: Path | str, indent: int = 2) -> None:
        """Save results to JSON file.

        Args:
            path: Path to save JSON file
            indent: JSON indentation level
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
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
        context_data = data.pop("context", None)
        context = None
        if context_data is not None:
            context = EvaluationContext(**context_data)

        # Convert confidence intervals back to tuples
        ci = data.get("confidence_intervals")
        if ci is not None:
            data["confidence_intervals"] = {k: tuple(v) for k, v in ci.items()}

        return cls(**data, context=context)

    @classmethod
    def from_json(cls, path: Path | str) -> EvaluationResult:
        """Load results from JSON file.

        Args:
            path: Path to JSON file

        Returns:
            EvaluationResult instance
        """
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

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

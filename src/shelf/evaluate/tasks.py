"""Task types and specifications for SHELF evaluation.

This module defines the core types that specify what tasks are available
and how they should be evaluated.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TaskType(Enum):
    """Types of evaluation tasks supported by SHELF."""

    CLASSIFICATION = "classification"
    MULTILABEL = "multilabel"
    RETRIEVAL = "retrieval"
    CLUSTERING = "clustering"
    PAIR_CLASSIFICATION = "pair_classification"


@dataclass(frozen=True)
class TaskSpec:
    """Immutable specification for an evaluation task.

    This defines everything needed to evaluate a task:
    - What data to load
    - What fields contain text, labels, and IDs
    - What metrics to compute
    - What the valid label space is

    Attributes:
        name: Unique task identifier (e.g., "lcc_classification")
        task_type: Type of task (classification, retrieval, etc.)
        description: Human-readable description of the task
        text_field: Field name containing the text to evaluate
        label_field: Field name containing ground truth labels
        id_field: Field name containing document IDs
        label_space: Valid labels (None = open vocabulary)
        primary_metric: Main metric for ranking/comparison
        secondary_metrics: Additional metrics to report
        dataset_name: HuggingFace dataset name
        dataset_config: Dataset configuration (e.g., "default", "same_lcc_pairs")
        default_split: Default split to evaluate on
    """

    name: str
    task_type: TaskType
    description: str

    # Data fields
    text_field: str
    label_field: str
    id_field: str

    # Label space (None = open vocabulary, e.g., for topics)
    label_space: tuple[str, ...] | None

    # Metrics
    primary_metric: str
    secondary_metrics: tuple[str, ...]

    # Dataset configuration
    dataset_name: str
    dataset_config: str | None
    default_split: str

    @property
    def num_classes(self) -> int | None:
        """Number of classes if label space is defined."""
        return len(self.label_space) if self.label_space else None

    def __str__(self) -> str:
        """Human-readable representation."""
        return f"TaskSpec({self.name}, type={self.task_type.value}, metric={self.primary_metric})"

    def __repr__(self) -> str:
        """Detailed representation."""
        return (
            f"TaskSpec(name={self.name!r}, task_type={self.task_type}, "
            f"label_field={self.label_field!r}, primary_metric={self.primary_metric!r})"
        )

"""Event dataclasses for the evaluation runner.

This module defines all event types emitted by the EvaluationOrchestrator.
Events follow an event-driven architecture that cleanly separates orchestration
logic from output/presentation concerns.

Events are consumed by OutputHandler implementations (RichHandler, JSONLHandler,
LoggerHandler) to provide unified logging across multiple formats.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunStarted:
    """Emitted when evaluation run begins.

    Attributes:
        run_id: Unique identifier for this run (typically timestamp-based)
        models: List of model keys to evaluate
        tasks: List of (task_name, task_type) tuples to execute
        total_combinations: Total number of model × task combinations
        context: Run context including version info, git state, etc.
    """

    run_id: str
    models: list[str]
    tasks: list[tuple[str, str]]  # (task_name, task_type)
    total_combinations: int
    context: dict[str, Any]  # version info, git, checksums, etc.


@dataclass(frozen=True)
class ModelStarted:
    """Emitted when model evaluation begins.

    Attributes:
        model_key: Model identifier (e.g., 'minilm', 'bge_small')
        model_name: Human-readable model name
        model_type: Model type ('sentence_transformer', 'openai', etc.)
        tasks_to_run: Number of tasks scheduled for this model
        tasks_skipped: Number of tasks skipped (already exist, not applicable)
    """

    model_key: str
    model_name: str
    model_type: str
    tasks_to_run: int
    tasks_skipped: int


@dataclass(frozen=True)
class CacheBuilding:
    """Emitted when embedding cache is being built.

    For dense models (sentence transformers), embeddings are cached upfront
    to avoid recomputing them for each task.

    Attributes:
        model_key: Model identifier
        num_texts: Number of texts to embed
    """

    model_key: str
    num_texts: int


@dataclass(frozen=True)
class CacheBuilt:
    """Emitted when embedding cache is complete.

    Attributes:
        model_key: Model identifier
        num_entries: Number of embeddings cached
        memory_mb: Memory usage of cache in megabytes
        duration_seconds: Time taken to build cache
        num_batches: Number of batches processed
        batch_times: List of per-batch durations (seconds)
        mean_batch_time: Mean time per batch (seconds)
        std_batch_time: Standard deviation of batch times (seconds)
        throughput: Samples per second
        total_bytes: Total input bytes processed
        throughput_bytes_sec: Bytes processed per second
        num_params_torch: Actual parameter count via torch.numel()
        hidden_size: Model hidden dimension
        context_window: Max position embeddings
    """

    model_key: str
    num_entries: int
    memory_mb: float
    duration_seconds: float
    num_batches: int = 0
    batch_times: tuple[float, ...] = ()
    mean_batch_time: float = 0.0
    std_batch_time: float = 0.0
    throughput: float = 0.0
    # New fields for enhanced metrics
    total_bytes: int = 0
    throughput_bytes_sec: float = 0.0
    num_params_torch: int | None = None
    hidden_size: int | None = None
    context_window: int | None = None


@dataclass(frozen=True)
class EmbeddingProgress:
    """Emitted during embedding for progress tracking.

    This event is emitted periodically during batch embedding to enable
    progress bars and status updates.

    Attributes:
        model_key: Model identifier
        current: Number of texts embedded so far
        total: Total number of texts to embed
    """

    model_key: str
    current: int
    total: int


@dataclass(frozen=True)
class TaskStarted:
    """Emitted when task evaluation begins.

    Attributes:
        model_key: Model identifier
        task_name: Task name (e.g., 'lcc_classification', 'form_retrieval')
        task_type: Task type ('classification', 'retrieval', 'clustering', etc.)
    """

    model_key: str
    task_name: str
    task_type: str


@dataclass(frozen=True)
class TaskCompleted:
    """Emitted when task evaluation completes successfully.

    Attributes:
        model_key: Model identifier
        task_name: Task name
        task_type: Task type
        primary_metric: Name of the primary metric (e.g., 'macro_f1', 'ndcg@10')
        primary_score: Value of the primary metric
        metrics: All metrics computed for this task
        duration_seconds: Time taken to evaluate task
        result_path: Path to saved result file (None if not saved)
    """

    model_key: str
    task_name: str
    task_type: str
    primary_metric: str
    primary_score: float
    metrics: dict[str, float]
    duration_seconds: float
    result_path: Path | None


@dataclass(frozen=True)
class TaskFailed:
    """Emitted when task evaluation fails.

    Attributes:
        model_key: Model identifier
        task_name: Task name
        task_type: Task type
        error: Error message
        traceback: Full traceback string (None if not available)
    """

    model_key: str
    task_name: str
    task_type: str
    error: str
    traceback: str | None


@dataclass(frozen=True)
class TaskSkipped:
    """Emitted when task is skipped.

    Tasks may be skipped because results already exist, the model doesn't
    support the task type, or configuration excludes the task.

    Attributes:
        model_key: Model identifier
        task_name: Task name
        reason: Explanation of why task was skipped
    """

    model_key: str
    task_name: str
    reason: str


@dataclass(frozen=True)
class ModelCompleted:
    """Emitted when all tasks for a model are complete.

    Attributes:
        model_key: Model identifier
        tasks_completed: Number of tasks successfully completed
        tasks_failed: Number of tasks that failed
        tasks_skipped: Number of tasks skipped
        duration_seconds: Total time spent on this model
    """

    model_key: str
    tasks_completed: int
    tasks_failed: int
    tasks_skipped: int
    duration_seconds: float


@dataclass(frozen=True)
class RunCompleted:
    """Emitted when entire evaluation run is complete.

    Attributes:
        run_id: Run identifier
        shelf_scores: SHELF scores for each model
        efficiency_metrics: Efficiency metrics (SHELF_eff, etc.) for each model
        total_tasks: Total number of task combinations attempted
        completed_tasks: Number of tasks successfully completed
        failed_tasks: Number of tasks that failed
        skipped_tasks: Number of tasks skipped
        duration_seconds: Total run duration
        summary_path: Path to summary JSON file (None if not saved)
    """

    run_id: str
    shelf_scores: dict[str, float]
    efficiency_metrics: dict[str, dict[str, Any]]
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    skipped_tasks: int
    duration_seconds: float
    summary_path: Path | None

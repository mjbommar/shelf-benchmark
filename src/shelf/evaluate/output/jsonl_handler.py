"""JSONL output handler for structured logging of evaluation events.

This module provides JSONLHandler, which writes evaluation events to a JSONL
(JSON Lines) file for analysis and monitoring. Each line is a complete JSON
record with timestamp, event type, and event-specific data.

JSONL format is ideal for:
- Streaming analysis (process events as they arrive)
- Easy import into data analysis tools (pandas, jq, etc.)
- Append-only logging without requiring JSON array closing
- Efficient parsing of large log files (no need to load entire file)

Example JSONL output:
    {"timestamp": "2025-01-15T10:30:00Z", "event": "run_started", "run_id": "20250115_103000", "models": ["minilm", "bge_small"], ...}
    {"timestamp": "2025-01-15T10:30:01Z", "event": "model_started", "model_key": "minilm", ...}
    {"timestamp": "2025-01-15T10:30:15Z", "event": "task_completed", "task_name": "lcc_classification", "primary_score": 0.8542, ...}
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

if TYPE_CHECKING:
    from shelf.evaluate.runner.events import (
        CacheBuilding,
        CacheBuilt,
        EmbeddingProgress,
        ModelCompleted,
        ModelStarted,
        RunCompleted,
        RunStarted,
        TaskCompleted,
        TaskFailed,
        TaskSkipped,
        TaskStarted,
    )


class JSONLHandler:
    """Output handler that writes structured events to a JSONL file.

    Each event is written as a single JSON line containing:
    - timestamp: ISO 8601 timestamp in UTC
    - event: Event type name (e.g., "run_started", "task_completed")
    - Event-specific fields (flattened into the top-level record)

    The handler automatically converts Path objects to strings and handles
    None values. Files are opened in append mode and flushed after each write
    to ensure events are persisted immediately.

    Example:
        ```python
        with JSONLHandler(Path("eval.jsonl")) as handler:
            orchestrator = EvaluationOrchestrator(config, output=handler)
            orchestrator.run()
        # File contains one JSON line per event
        ```

    Note:
        The handler skips EmbeddingProgress events as they are too noisy for
        logs (emitted frequently during batch embedding).
    """

    def __init__(self, path: Path) -> None:
        """Initialize JSONL handler.

        Args:
            path: Path to the JSONL file to write to (will be created if
                  it doesn't exist, appended to if it does)
        """
        self.path = path
        self._file: TextIO = open(path, "a", encoding="utf-8")

    def _write(self, event_type: str, data: dict[str, Any]) -> None:
        """Write an event record to the JSONL file.

        Creates a record with timestamp, event type, and flattened event data,
        then writes it as a single JSON line and flushes.

        Args:
            event_type: Type of event (e.g., "run_started", "task_completed")
            data: Event-specific data to include in the record
        """
        record: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
        }

        # Flatten data into record, converting Path to str
        for key, value in data.items():
            if isinstance(value, Path):
                record[key] = str(value)
            elif isinstance(value, list):
                # Convert list elements (handle Path objects in lists)
                record[key] = [
                    str(item) if isinstance(item, Path) else item for item in value
                ]
            else:
                record[key] = value

        # Write JSON line and flush immediately
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

    def on_run_started(self, event: RunStarted) -> None:
        """Handle run started event.

        Logs: run_id, models, tasks (as list of task names), total_combinations,
        and context (version info, git state, checksums).

        Args:
            event: Run started event
        """
        self._write(
            "run_started",
            {
                "run_id": event.run_id,
                "models": event.models,
                "tasks": [task_name for task_name, _ in event.tasks],
                "total_combinations": event.total_combinations,
                "context": event.context,
            },
        )

    def on_model_started(self, event: ModelStarted) -> None:
        """Handle model started event.

        Logs: model_key, model_name, model_type, tasks_to_run, tasks_skipped.

        Args:
            event: Model started event
        """
        self._write(
            "model_started",
            {
                "model_key": event.model_key,
                "model_name": event.model_name,
                "model_type": event.model_type,
                "tasks_to_run": event.tasks_to_run,
                "tasks_skipped": event.tasks_skipped,
            },
        )

    def on_cache_building(self, event: CacheBuilding) -> None:
        """Handle cache building event.

        Logs: model_key, num_texts.

        Args:
            event: Cache building event
        """
        self._write(
            "cache_building",
            {
                "model_key": event.model_key,
                "num_texts": event.num_texts,
            },
        )

    def on_cache_built(self, event: CacheBuilt) -> None:
        """Handle cache built event.

        Logs: model_key, num_entries, memory_mb, duration_seconds,
        batch timing statistics (num_batches, mean_batch_time, std_batch_time, throughput).

        Args:
            event: Cache built event
        """
        self._write(
            "cache_built",
            {
                "model_key": event.model_key,
                "num_entries": event.num_entries,
                "memory_mb": event.memory_mb,
                "duration_seconds": event.duration_seconds,
                "num_batches": event.num_batches,
                "mean_batch_time": event.mean_batch_time,
                "std_batch_time": event.std_batch_time,
                "throughput": event.throughput,
            },
        )

    def on_embedding_progress(self, event: EmbeddingProgress) -> None:
        """Handle embedding progress event.

        This event is skipped (too noisy for logs - emitted frequently during
        batch embedding).

        Args:
            event: Embedding progress event (ignored)
        """
        # Skip - too noisy for logs
        pass

    def on_task_started(self, event: TaskStarted) -> None:
        """Handle task started event.

        Logs: model_key, task_name, task_type.

        Args:
            event: Task started event
        """
        self._write(
            "task_started",
            {
                "model_key": event.model_key,
                "task_name": event.task_name,
                "task_type": event.task_type,
            },
        )

    def on_task_completed(self, event: TaskCompleted) -> None:
        """Handle task completed event.

        Logs: model_key, task_name, task_type, primary_metric, primary_score,
        duration_seconds, result_path.

        Args:
            event: Task completed event
        """
        self._write(
            "task_completed",
            {
                "model_key": event.model_key,
                "task_name": event.task_name,
                "task_type": event.task_type,
                "primary_metric": event.primary_metric,
                "primary_score": event.primary_score,
                "duration_seconds": event.duration_seconds,
                "result_path": event.result_path,
            },
        )

    def on_task_failed(self, event: TaskFailed) -> None:
        """Handle task failed event.

        Logs: model_key, task_name, task_type, error (truncated traceback).

        Args:
            event: Task failed event
        """
        # Truncate traceback to first 500 chars to keep logs manageable
        error = event.error
        if event.traceback:
            error = f"{error}\n{event.traceback[:500]}"
            if len(event.traceback) > 500:
                error += "... (truncated)"

        self._write(
            "task_failed",
            {
                "model_key": event.model_key,
                "task_name": event.task_name,
                "task_type": event.task_type,
                "error": error,
            },
        )

    def on_task_skipped(self, event: TaskSkipped) -> None:
        """Handle task skipped event.

        Logs: model_key, task_name, reason.

        Args:
            event: Task skipped event
        """
        self._write(
            "task_skipped",
            {
                "model_key": event.model_key,
                "task_name": event.task_name,
                "reason": event.reason,
            },
        )

    def on_model_completed(self, event: ModelCompleted) -> None:
        """Handle model completed event.

        Logs: model_key, tasks_completed, tasks_failed, tasks_skipped,
        duration_seconds.

        Args:
            event: Model completed event
        """
        self._write(
            "model_completed",
            {
                "model_key": event.model_key,
                "tasks_completed": event.tasks_completed,
                "tasks_failed": event.tasks_failed,
                "tasks_skipped": event.tasks_skipped,
                "duration_seconds": event.duration_seconds,
            },
        )

    def on_run_completed(self, event: RunCompleted) -> None:
        """Handle run completed event.

        Logs: run_id, total_tasks, completed_tasks, failed_tasks, skipped_tasks,
        duration_seconds, shelf_scores, efficiency_metrics, summary_path.

        Args:
            event: Run completed event
        """
        self._write(
            "run_completed",
            {
                "run_id": event.run_id,
                "total_tasks": event.total_tasks,
                "completed_tasks": event.completed_tasks,
                "failed_tasks": event.failed_tasks,
                "skipped_tasks": event.skipped_tasks,
                "duration_seconds": event.duration_seconds,
                "shelf_scores": event.shelf_scores,
                "efficiency_metrics": event.efficiency_metrics,
                "summary_path": event.summary_path,
            },
        )

    def close(self) -> None:
        """Close the JSONL file handle.

        Should be called when done writing events, or use the handler as a
        context manager to ensure automatic cleanup.
        """
        if hasattr(self, "_file") and not self._file.closed:
            self._file.close()

    def __enter__(self) -> JSONLHandler:
        """Enter context manager.

        Returns:
            Self for use in with statement
        """
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager and close file.

        Args:
            exc_type: Exception type (if any)
            exc_val: Exception value (if any)
            exc_tb: Exception traceback (if any)
        """
        self.close()

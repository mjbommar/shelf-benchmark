"""Python logging adapter for evaluation output.

This handler uses Python's standard logging module for traditional log output.
Useful for integration with existing logging infrastructure, file-based logging,
or environments where rich console output is not desired.

The handler maps evaluation events to appropriate log levels:
- INFO: Normal progress (run started, task completed, model completed)
- WARNING: Skipped tasks
- ERROR: Task failures
- DEBUG: Fine-grained progress (task started, embedding progress)

Example:
    ```python
    import logging
    from shelf.evaluate.output.logger_handler import LoggerHandler

    # Configure root logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    # Use in orchestrator
    handler = LoggerHandler()
    orchestrator = EvaluationOrchestrator(config, output=handler)
    result = orchestrator.run()
    ```
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

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


class LoggerHandler:
    """Output handler that uses Python's standard logging module.

    This handler formats evaluation events as log messages at appropriate levels.
    It's designed for traditional logging workflows and integrates seamlessly with
    existing Python logging infrastructure.

    Attributes:
        logger: Python logger instance to use for output
        level: Default log level for normal events (defaults to INFO)
    """

    def __init__(
        self,
        logger: logging.Logger | None = None,
        level: int = logging.INFO,
    ) -> None:
        """Initialize the logger handler.

        Args:
            logger: Logger to use for output. If None, creates logger for "shelf.evaluate"
            level: Default log level for normal events (defaults to INFO)
        """
        self.logger = (
            logger if logger is not None else logging.getLogger("shelf.evaluate")
        )
        self.level = level
        self._embedding_log_interval = 100  # Log every N embeddings to reduce noise

    def on_run_started(self, event: RunStarted) -> None:
        """Log run started event at INFO level.

        Args:
            event: Run started event
        """
        num_models = len(event.models)
        num_tasks = len(event.tasks)
        self.logger.info(
            f"Starting evaluation run {event.run_id}: {num_models} models, {num_tasks} tasks"
        )

    def on_model_started(self, event: ModelStarted) -> None:
        """Log model started event at INFO level.

        Args:
            event: Model started event
        """
        self.logger.info(
            f"Evaluating {event.model_name} ({event.model_type}): {event.tasks_to_run} tasks"
        )

    def on_cache_building(self, event: CacheBuilding) -> None:
        """Log cache building event at INFO level.

        Args:
            event: Cache building event
        """
        self.logger.info(
            f"Building embedding cache for {event.model_key}: {event.num_texts} texts"
        )

    def on_cache_built(self, event: CacheBuilt) -> None:
        """Log cache built event at INFO level.

        Args:
            event: Cache built event
        """
        self.logger.info(
            f"Cache built: {event.num_entries} entries, {event.memory_mb:.1f} MB "
            f"in {event.duration_seconds:.1f}s"
        )
        if event.num_batches > 0:
            self.logger.info(
                f"Batch stats: {event.num_batches} batches, "
                f"{event.mean_batch_time:.3f}s ± {event.std_batch_time:.3f}s per batch, "
                f"{event.throughput:.1f} samples/s"
            )

    def on_embedding_progress(self, event: EmbeddingProgress) -> None:
        """Log embedding progress at DEBUG level (throttled).

        Only logs every N embeddings to reduce noise.

        Args:
            event: Embedding progress event
        """
        # Only log periodically to avoid spam
        if (
            event.current % self._embedding_log_interval == 0
            or event.current == event.total
        ):
            self.logger.debug(f"Embedding progress: {event.current}/{event.total}")

    def on_task_started(self, event: TaskStarted) -> None:
        """Log task started event at DEBUG level.

        Args:
            event: Task started event
        """
        self.logger.debug(f"Starting {event.model_key}/{event.task_name}")

    def on_task_completed(self, event: TaskCompleted) -> None:
        """Log task completed event at INFO level.

        Args:
            event: Task completed event
        """
        self.logger.info(
            f"{event.model_key}/{event.task_name}: {event.primary_metric}={event.primary_score:.4f} "
            f"({event.duration_seconds:.1f}s)"
        )

    def on_task_failed(self, event: TaskFailed) -> None:
        """Log task failed event at ERROR level.

        Logs error message at ERROR level and full traceback at DEBUG level if available.

        Args:
            event: Task failed event
        """
        self.logger.error(f"{event.model_key}/{event.task_name} failed: {event.error}")
        if event.traceback is not None:
            self.logger.debug(
                f"Traceback for {event.model_key}/{event.task_name}:\n{event.traceback}"
            )

    def on_task_skipped(self, event: TaskSkipped) -> None:
        """Log task skipped event at INFO level.

        Args:
            event: Task skipped event
        """
        self.logger.info(
            f"{event.model_key}/{event.task_name}: skipped ({event.reason})"
        )

    def on_model_completed(self, event: ModelCompleted) -> None:
        """Log model completed event at INFO level.

        Args:
            event: Model completed event
        """
        self.logger.info(
            f"{event.model_key} complete: {event.tasks_completed} tasks, "
            f"{event.tasks_failed} failed ({event.duration_seconds:.1f}s)"
        )

    def on_run_completed(self, event: RunCompleted) -> None:
        """Log run completed event at INFO level.

        Args:
            event: Run completed event
        """
        self.logger.info(
            f"Run complete: {event.total_tasks} tasks, {event.failed_tasks} failed "
            f"in {event.duration_seconds:.1f}s"
        )
        if event.summary_path is not None:
            self.logger.info(f"Summary saved to: {event.summary_path}")

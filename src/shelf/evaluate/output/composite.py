"""Composite handler for routing events to multiple handlers simultaneously.

This module provides CompositeHandler, which fans out events to multiple OutputHandlers.
This enables combining different output formats (e.g., rich console + JSONL logging)
without coupling the orchestrator to multiple handlers.

Key design decisions:
- Individual handler failures don't break others (fault isolation)
- Handlers are called in registration order (deterministic)
- Supports dynamic handler management (add/remove during run)
- Type-safe through OutputHandler protocol

Example:
    ```python
    # Combine rich console output with JSONL logging
    composite = create_composite(
        RichHandler(),
        JSONLHandler("eval.jsonl"),
        StructlogHandler()
    )
    orchestrator = EvaluationOrchestrator(config, output=composite)
    ```
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from shelf.evaluate.output.base import OutputHandler

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

logger = logging.getLogger(__name__)


class CompositeHandler:
    """Fan-out handler that routes events to multiple handlers.

    Routes all events to a list of child handlers, with fault isolation to prevent
    one handler's failure from affecting others. Useful for combining multiple output
    formats (console, file, logging) in a single evaluation run.

    Handlers are called in registration order. Exceptions are caught and logged but
    not propagated, ensuring robustness.

    Attributes:
        handlers: List of OutputHandlers to route events to

    Example:
        ```python
        handler = CompositeHandler([
            RichHandler(),
            JSONLHandler("results.jsonl")
        ])

        # Add another handler dynamically
        handler.add(StructlogHandler())

        # Use with orchestrator
        orchestrator = EvaluationOrchestrator(config, output=handler)
        ```
    """

    def __init__(self, handlers: list[OutputHandler]) -> None:
        """Initialize composite handler with list of handlers.

        Args:
            handlers: List of OutputHandlers to fan out to. Can be empty.
        """
        self.handlers = handlers

    def on_run_started(self, event: RunStarted) -> None:
        """Route run started event to all handlers.

        Args:
            event: Run started event with configuration and context
        """
        for handler in self.handlers:
            try:
                handler.on_run_started(event)
            except Exception as e:
                logger.exception(
                    "Handler %s failed on_run_started: %s",
                    type(handler).__name__,
                    e,
                )

    def on_model_started(self, event: ModelStarted) -> None:
        """Route model started event to all handlers.

        Args:
            event: Model started event with model info and task counts
        """
        for handler in self.handlers:
            try:
                handler.on_model_started(event)
            except Exception as e:
                logger.exception(
                    "Handler %s failed on_model_started: %s",
                    type(handler).__name__,
                    e,
                )

    def on_cache_building(self, event: CacheBuilding) -> None:
        """Route cache building event to all handlers.

        Args:
            event: Cache building event with model key and text count
        """
        for handler in self.handlers:
            try:
                handler.on_cache_building(event)
            except Exception as e:
                logger.exception(
                    "Handler %s failed on_cache_building: %s",
                    type(handler).__name__,
                    e,
                )

    def on_cache_built(self, event: CacheBuilt) -> None:
        """Route cache built event to all handlers.

        Args:
            event: Cache built event with performance metrics
        """
        for handler in self.handlers:
            try:
                handler.on_cache_built(event)
            except Exception as e:
                logger.exception(
                    "Handler %s failed on_cache_built: %s",
                    type(handler).__name__,
                    e,
                )

    def on_embedding_progress(self, event: EmbeddingProgress) -> None:
        """Route embedding progress event to all handlers.

        High-frequency event - handlers should be efficient.

        Args:
            event: Embedding progress event with current/total counts
        """
        for handler in self.handlers:
            try:
                handler.on_embedding_progress(event)
            except Exception as e:
                logger.exception(
                    "Handler %s failed on_embedding_progress: %s",
                    type(handler).__name__,
                    e,
                )

    def on_task_started(self, event: TaskStarted) -> None:
        """Route task started event to all handlers.

        Args:
            event: Task started event with task and model identifiers
        """
        for handler in self.handlers:
            try:
                handler.on_task_started(event)
            except Exception as e:
                logger.exception(
                    "Handler %s failed on_task_started: %s",
                    type(handler).__name__,
                    e,
                )

    def on_task_completed(self, event: TaskCompleted) -> None:
        """Route task completed event to all handlers.

        Args:
            event: Task completed event with metrics and result path
        """
        for handler in self.handlers:
            try:
                handler.on_task_completed(event)
            except Exception as e:
                logger.exception(
                    "Handler %s failed on_task_completed: %s",
                    type(handler).__name__,
                    e,
                )

    def on_task_failed(self, event: TaskFailed) -> None:
        """Route task failed event to all handlers.

        Args:
            event: Task failed event with error details
        """
        for handler in self.handlers:
            try:
                handler.on_task_failed(event)
            except Exception as e:
                logger.exception(
                    "Handler %s failed on_task_failed: %s",
                    type(handler).__name__,
                    e,
                )

    def on_task_skipped(self, event: TaskSkipped) -> None:
        """Route task skipped event to all handlers.

        Args:
            event: Task skipped event with skip reason
        """
        for handler in self.handlers:
            try:
                handler.on_task_skipped(event)
            except Exception as e:
                logger.exception(
                    "Handler %s failed on_task_skipped: %s",
                    type(handler).__name__,
                    e,
                )

    def on_model_completed(self, event: ModelCompleted) -> None:
        """Route model completed event to all handlers.

        Args:
            event: Model completed event with task counts and duration
        """
        for handler in self.handlers:
            try:
                handler.on_model_completed(event)
            except Exception as e:
                logger.exception(
                    "Handler %s failed on_model_completed: %s",
                    type(handler).__name__,
                    e,
                )

    def on_run_completed(self, event: RunCompleted) -> None:
        """Route run completed event to all handlers.

        Args:
            event: Run completed event with final scores and summary path
        """
        for handler in self.handlers:
            try:
                handler.on_run_completed(event)
            except Exception as e:
                logger.exception(
                    "Handler %s failed on_run_completed: %s",
                    type(handler).__name__,
                    e,
                )

    def add(self, handler: OutputHandler) -> None:
        """Add a handler to the composite.

        The new handler will receive all subsequent events. It will not receive
        events that were emitted before it was added.

        Args:
            handler: OutputHandler to add
        """
        self.handlers.append(handler)

    def remove(self, handler: OutputHandler) -> None:
        """Remove a handler from the composite.

        The handler will no longer receive events. If the handler is not in the
        composite, this is a no-op (does not raise an error).

        Args:
            handler: OutputHandler to remove
        """
        try:
            self.handlers.remove(handler)
        except ValueError:
            # Handler not in list - this is fine, just log it
            logger.debug(
                "Attempted to remove handler %s that wasn't in composite",
                type(handler).__name__,
            )

    def __len__(self) -> int:
        """Return the number of handlers in the composite.

        Returns:
            Number of handlers currently registered
        """
        return len(self.handlers)


def create_composite(*handlers: OutputHandler) -> CompositeHandler:
    """Create a composite handler from multiple handlers.

    Convenience factory function for creating a CompositeHandler from a variable
    number of handler arguments.

    Args:
        *handlers: Variable number of OutputHandlers to combine

    Returns:
        CompositeHandler that routes to all provided handlers

    Example:
        ```python
        handler = create_composite(
            RichHandler(),
            JSONLHandler("log.jsonl"),
            StructlogHandler()
        )
        ```
    """
    return CompositeHandler(list(handlers))

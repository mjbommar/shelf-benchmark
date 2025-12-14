"""Base protocol and null implementation for evaluation output handlers.

This module defines the OutputHandler protocol that all output handlers must implement,
as well as a NullHandler for testing and silent mode.

The event-driven design separates orchestration logic from presentation/logging:
- Orchestrator emits events (what happens)
- OutputHandlers consume events (how it's reported)

This enables multiple output formats (rich console, JSONL, Python logging) without
coupling the orchestrator to any specific output mechanism.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

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


class OutputHandler(Protocol):
    """Protocol for handling evaluation events.

    All output handlers must implement these methods to receive events from the
    EvaluationOrchestrator. This protocol-based approach allows for:

    1. Multiple output formats without modifying orchestrator code
    2. Composition via CompositeHandler to fan out to multiple handlers
    3. Easy testing with mock/null handlers
    4. Type-safe event handling

    Each method receives a specific event type containing all relevant data for that
    event. Methods should return None and handle any errors internally (logging,
    ignoring, or raising as appropriate for the handler type).

    Example:
        ```python
        class MyHandler:
            def on_run_started(self, event: RunStarted) -> None:
                print(f"Starting run {event.run_id} with {len(event.models)} models")

            def on_task_completed(self, event: TaskCompleted) -> None:
                print(f"{event.task_name}: {event.primary_score:.4f}")

            # ... implement all other methods ...
        ```
    """

    def on_run_started(self, event: RunStarted) -> None:
        """Handle run started event.

        Emitted when an evaluation run begins. Contains run configuration,
        environment context, and list of models/tasks to evaluate.

        Args:
            event: Run started event with configuration and context
        """
        ...

    def on_model_started(self, event: ModelStarted) -> None:
        """Handle model started event.

        Emitted when evaluation begins for a specific model. Contains model
        metadata and count of tasks to run/skip.

        Args:
            event: Model started event with model info and task counts
        """
        ...

    def on_cache_building(self, event: CacheBuilding) -> None:
        """Handle cache building event.

        Emitted when embedding cache construction begins for a dense model.
        Useful for starting progress indicators.

        Args:
            event: Cache building event with model key and text count
        """
        ...

    def on_cache_built(self, event: CacheBuilt) -> None:
        """Handle cache built event.

        Emitted when embedding cache construction completes. Contains
        performance metrics (duration, memory usage, entry count).

        Args:
            event: Cache built event with performance metrics
        """
        ...

    def on_embedding_progress(self, event: EmbeddingProgress) -> None:
        """Handle embedding progress event.

        Emitted periodically during embedding to update progress indicators.
        High-frequency event - handlers should be efficient.

        Args:
            event: Embedding progress event with current/total counts
        """
        ...

    def on_task_started(self, event: TaskStarted) -> None:
        """Handle task started event.

        Emitted when evaluation begins for a specific task on a model.

        Args:
            event: Task started event with task and model identifiers
        """
        ...

    def on_task_completed(self, event: TaskCompleted) -> None:
        """Handle task completed event.

        Emitted when a task evaluation completes successfully. Contains
        primary metric score, duration, and result file path.

        Args:
            event: Task completed event with metrics and result path
        """
        ...

    def on_task_failed(self, event: TaskFailed) -> None:
        """Handle task failed event.

        Emitted when a task evaluation fails with an error. Contains
        error message and optional traceback.

        Args:
            event: Task failed event with error details
        """
        ...

    def on_task_skipped(self, event: TaskSkipped) -> None:
        """Handle task skipped event.

        Emitted when a task is skipped (e.g., result already exists and
        skip_existing is enabled).

        Args:
            event: Task skipped event with skip reason
        """
        ...

    def on_model_completed(self, event: ModelCompleted) -> None:
        """Handle model completed event.

        Emitted when all tasks for a model have finished (success or failure).
        Contains aggregate statistics for the model's tasks.

        Args:
            event: Model completed event with task counts and duration
        """
        ...

    def on_run_completed(self, event: RunCompleted) -> None:
        """Handle run completed event.

        Emitted when the entire evaluation run finishes. Contains final
        SHELF scores, efficiency metrics, and summary statistics.

        Args:
            event: Run completed event with final scores and summary path
        """
        ...


class NullHandler:
    """No-op output handler for testing or silent mode.

    This handler implements the OutputHandler protocol but performs no actions.
    Useful for:
    - Testing orchestrator logic without output noise
    - Silent/headless evaluation runs
    - Baseline implementation to copy when creating new handlers

    Example:
        ```python
        # Silent evaluation
        orchestrator = EvaluationOrchestrator(config, output=NullHandler())
        result = orchestrator.run()
        ```
    """

    def on_run_started(self, event: RunStarted) -> None:
        """No-op implementation."""
        pass

    def on_model_started(self, event: ModelStarted) -> None:
        """No-op implementation."""
        pass

    def on_cache_building(self, event: CacheBuilding) -> None:
        """No-op implementation."""
        pass

    def on_cache_built(self, event: CacheBuilt) -> None:
        """No-op implementation."""
        pass

    def on_embedding_progress(self, event: EmbeddingProgress) -> None:
        """No-op implementation."""
        pass

    def on_task_started(self, event: TaskStarted) -> None:
        """No-op implementation."""
        pass

    def on_task_completed(self, event: TaskCompleted) -> None:
        """No-op implementation."""
        pass

    def on_task_failed(self, event: TaskFailed) -> None:
        """No-op implementation."""
        pass

    def on_task_skipped(self, event: TaskSkipped) -> None:
        """No-op implementation."""
        pass

    def on_model_completed(self, event: ModelCompleted) -> None:
        """No-op implementation."""
        pass

    def on_run_completed(self, event: RunCompleted) -> None:
        """No-op implementation."""
        pass

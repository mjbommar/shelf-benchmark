"""Output handlers for SHELF evaluation framework.

This package provides output handlers that consume evaluation events and
produce various output formats (console, JSONL, logging).

The event-driven design separates orchestration logic from presentation:
- Orchestrator emits events (what happens)
- OutputHandlers consume events (how it's reported)

Example:
    ```python
    from shelf.evaluate.output import (
        RichHandler,
        JSONLHandler,
        create_composite,
    )

    # Use multiple output formats
    handler = create_composite(
        RichHandler(),
        JSONLHandler("eval.jsonl"),
    )
    orchestrator = EvaluationOrchestrator(config, output=handler)
    result = orchestrator.run()
    ```
"""

from shelf.evaluate.output.base import NullHandler, OutputHandler
from shelf.evaluate.output.composite import CompositeHandler, create_composite
from shelf.evaluate.output.jsonl_handler import JSONLHandler
from shelf.evaluate.output.logger_handler import LoggerHandler
from shelf.evaluate.output.rich_handler import RichHandler

__all__ = [
    "OutputHandler",
    "NullHandler",
    "RichHandler",
    "JSONLHandler",
    "LoggerHandler",
    "CompositeHandler",
    "create_composite",
]

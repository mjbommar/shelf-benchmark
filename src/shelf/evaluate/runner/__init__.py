"""Runner support modules for SHELF evaluation framework.

This package provides supporting functionality for the evaluation runner,
including context tracking, configuration, event handling, and orchestration.

Note: The main evaluation runner functions (evaluate, evaluate_all) are defined
in a runner.py file at the evaluate package level. Because Python gives priority
to the runner/ directory over runner.py when both exist, we need to explicitly
import from the .py file using importlib.
"""

from shelf.evaluate.runner.config import RunConfig
from shelf.evaluate.runner.context import (
    RunContext,
    compute_dataset_checksum,
    get_git_info,
    get_version_info,
)
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
from shelf.evaluate.runner.orchestrator import (
    EvaluationOrchestrator,
    OrchestrationResult,
)

# Import runner functions from the sibling runner.py module
# We use importlib.util to load the .py file explicitly since this directory
# shadows it in normal imports
import importlib.util
import sys
from pathlib import Path

_runner_py_path = Path(__file__).parent.parent / "runner.py"
_spec = importlib.util.spec_from_file_location(
    "shelf.evaluate._runner_impl", _runner_py_path
)
if _spec and _spec.loader:
    _runner_module = importlib.util.module_from_spec(_spec)
    sys.modules["shelf.evaluate._runner_impl"] = _runner_module
    _spec.loader.exec_module(_runner_module)
    evaluate = _runner_module.evaluate
    evaluate_all = _runner_module.evaluate_all
else:
    raise ImportError(f"Could not load runner.py from {_runner_py_path}")

__all__ = [
    # Configuration
    "RunConfig",
    # Context utilities
    "RunContext",
    "compute_dataset_checksum",
    "get_git_info",
    "get_version_info",
    # Events
    "CacheBuilding",
    "CacheBuilt",
    "EmbeddingProgress",
    "ModelCompleted",
    "ModelStarted",
    "RunCompleted",
    "RunStarted",
    "TaskCompleted",
    "TaskFailed",
    "TaskSkipped",
    "TaskStarted",
    # Orchestration
    "EvaluationOrchestrator",
    "OrchestrationResult",
    # Runner functions (from ../runner.py)
    "evaluate",
    "evaluate_all",
]

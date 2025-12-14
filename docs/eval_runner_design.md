# SHELF Evaluation Runner Architecture

## Overview

This document describes the redesigned evaluation runner architecture that:
1. Separates orchestration logic from output/presentation
2. Provides unified logging (rich console, JSONL, Python logger)
3. Is reusable and testable
4. Integrates cleanly with the CLI

## Current State

The current `scripts/baselines/run_all.py` is a monolithic 890-line script that:
- Mixes orchestration, output, and logging
- Uses Python's `logging` module for all output
- Has no separation between "what to do" and "how to report it"
- Is called via subprocess from the CLI (a hack)

## Proposed Architecture

```
src/shelf/evaluate/
├── runner/                    # NEW: Orchestration module
│   ├── __init__.py           # Public API
│   ├── orchestrator.py       # EvaluationOrchestrator class
│   ├── config.py             # RunConfig, ModelSpec, TaskSpec
│   ├── events.py             # Event types (started, progress, completed, error)
│   └── context.py            # RunContext (versions, git, checksums)
│
├── output/                    # NEW: Output system
│   ├── __init__.py           # Public API
│   ├── base.py               # OutputHandler protocol
│   ├── rich_handler.py       # Rich console output (tables, progress bars)
│   ├── jsonl_handler.py      # JSONL structured logs
│   ├── logger_handler.py     # Python logging adapter
│   └── composite.py          # CompositeHandler (fan-out to multiple)
│
├── adapters/                  # Existing (unchanged)
├── evaluators/                # Existing (unchanged)
├── metrics/                   # Existing (unchanged)
└── ...
```

## Core Design: Event-Driven Architecture

The orchestrator emits **events** that output handlers consume. This cleanly separates:
- **What happens** (orchestrator logic)
- **How it's reported** (output handlers)

### Event Types

```python
@dataclass
class RunStarted:
    """Emitted when evaluation run begins."""
    run_id: str
    config: RunConfig
    context: RunContext
    models: list[str]
    tasks: list[str]
    total_combinations: int

@dataclass
class ModelStarted:
    """Emitted when model evaluation begins."""
    model_key: str
    model_name: str
    model_type: str
    tasks_to_run: int
    tasks_skipped: int

@dataclass
class CacheBuilding:
    """Emitted when embedding cache is being built."""
    model_key: str
    num_texts: int

@dataclass
class CacheBuilt:
    """Emitted when embedding cache is complete."""
    model_key: str
    num_entries: int
    memory_mb: float
    duration_seconds: float

@dataclass
class TaskStarted:
    """Emitted when task evaluation begins."""
    model_key: str
    task_name: str
    task_type: str

@dataclass
class TaskCompleted:
    """Emitted when task evaluation completes."""
    model_key: str
    task_name: str
    task_type: str
    primary_metric: str
    primary_score: float
    duration_seconds: float
    result_path: Path

@dataclass
class TaskFailed:
    """Emitted when task evaluation fails."""
    model_key: str
    task_name: str
    error: str
    traceback: str | None

@dataclass
class ModelCompleted:
    """Emitted when all tasks for a model are done."""
    model_key: str
    tasks_completed: int
    tasks_failed: int
    duration_seconds: float

@dataclass
class RunCompleted:
    """Emitted when entire run is complete."""
    run_id: str
    shelf_scores: dict[str, float]
    efficiency_metrics: dict[str, dict]
    total_tasks: int
    failed_tasks: int
    duration_seconds: float
    summary_path: Path

@dataclass
class EmbeddingProgress:
    """Emitted during embedding (for progress bars)."""
    model_key: str
    current: int
    total: int
```

### OutputHandler Protocol

```python
from typing import Protocol

class OutputHandler(Protocol):
    """Protocol for handling evaluation events."""

    def on_run_started(self, event: RunStarted) -> None: ...
    def on_model_started(self, event: ModelStarted) -> None: ...
    def on_cache_building(self, event: CacheBuilding) -> None: ...
    def on_cache_built(self, event: CacheBuilt) -> None: ...
    def on_task_started(self, event: TaskStarted) -> None: ...
    def on_task_completed(self, event: TaskCompleted) -> None: ...
    def on_task_failed(self, event: TaskFailed) -> None: ...
    def on_model_completed(self, event: ModelCompleted) -> None: ...
    def on_run_completed(self, event: RunCompleted) -> None: ...
    def on_embedding_progress(self, event: EmbeddingProgress) -> None: ...
```

## Output Handlers

### RichHandler

Beautiful console output using rich:

```python
class RichHandler:
    """Rich console output with tables and progress bars."""

    def __init__(self, console: Console | None = None):
        self.console = console or Console()
        self._progress: Progress | None = None
        self._task_id: TaskID | None = None

    def on_run_started(self, event: RunStarted) -> None:
        # Show header panel
        self.console.print(Panel(
            f"[bold]SHELF Baseline Evaluation[/bold]\n"
            f"Dataset: v{event.context.dataset_version}\n"
            f"Models: {len(event.models)} | Tasks: {len(event.tasks)}",
            border_style="cyan",
        ))

        # Show version info table
        table = Table(title="Environment", show_header=False)
        table.add_row("Python", event.context.python_version)
        table.add_row("sklearn", event.context.sklearn_version)
        table.add_row("Git", f"{event.context.git_commit} (dirty={event.context.git_dirty})")
        self.console.print(table)

    def on_cache_building(self, event: CacheBuilding) -> None:
        # Start progress bar
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console,
        )
        self._progress.start()
        self._task_id = self._progress.add_task(
            f"Embedding {event.num_texts} texts...",
            total=event.num_texts
        )

    def on_embedding_progress(self, event: EmbeddingProgress) -> None:
        if self._progress and self._task_id:
            self._progress.update(self._task_id, completed=event.current)

    def on_task_completed(self, event: TaskCompleted) -> None:
        self.console.print(
            f"  [green]✓[/green] {event.task_name}: "
            f"{event.primary_metric}=[cyan]{event.primary_score:.4f}[/cyan] "
            f"[dim]({event.duration_seconds:.1f}s)[/dim]"
        )

    def on_run_completed(self, event: RunCompleted) -> None:
        # Show SHELF scores table
        table = Table(title="SHELF Score Rankings")
        table.add_column("Rank", justify="right", style="dim")
        table.add_column("Model", style="cyan")
        table.add_column("SHELF", justify="right", style="green")
        table.add_column("SHELF_eff", justify="right", style="yellow")

        sorted_scores = sorted(
            event.shelf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        for rank, (model, score) in enumerate(sorted_scores, 1):
            eff = event.efficiency_metrics.get(model, {}).get("shelf_eff")
            table.add_row(
                str(rank),
                model,
                f"{score:.4f}",
                f"{eff:.2f}" if eff else "-",
            )

        self.console.print(table)
```

### JSONLHandler

Structured logging for analysis:

```python
class JSONLHandler:
    """JSONL structured event logging."""

    def __init__(self, path: Path):
        self.path = path
        self._file = open(path, "a")

    def _write(self, event_type: str, data: dict) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            **data,
        }
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

    def on_run_started(self, event: RunStarted) -> None:
        self._write("run_started", {
            "run_id": event.run_id,
            "models": event.models,
            "tasks": event.tasks,
            "total_combinations": event.total_combinations,
        })

    def on_task_completed(self, event: TaskCompleted) -> None:
        self._write("task_completed", {
            "model_key": event.model_key,
            "task_name": event.task_name,
            "task_type": event.task_type,
            "primary_metric": event.primary_metric,
            "primary_score": event.primary_score,
            "duration_seconds": event.duration_seconds,
        })

    # ... etc
```

### LoggerHandler

Traditional Python logging:

```python
class LoggerHandler:
    """Python logging adapter."""

    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger("shelf.evaluate")

    def on_run_started(self, event: RunStarted) -> None:
        self.logger.info(
            "Starting evaluation run %s: %d models, %d tasks",
            event.run_id, len(event.models), len(event.tasks)
        )

    def on_task_completed(self, event: TaskCompleted) -> None:
        self.logger.info(
            "%s/%s: %s=%.4f (%.1fs)",
            event.model_key, event.task_name,
            event.primary_metric, event.primary_score,
            event.duration_seconds
        )
```

### CompositeHandler

Fan-out to multiple handlers:

```python
class CompositeHandler:
    """Routes events to multiple handlers."""

    def __init__(self, handlers: list[OutputHandler]):
        self.handlers = handlers

    def on_run_started(self, event: RunStarted) -> None:
        for h in self.handlers:
            h.on_run_started(event)

    # ... same pattern for all events
```

## EvaluationOrchestrator

The core orchestrator that does the work:

```python
class EvaluationOrchestrator:
    """Orchestrates evaluation runs with event-based output."""

    def __init__(
        self,
        config: RunConfig,
        output: OutputHandler,
    ):
        self.config = config
        self.output = output
        self.context = RunContext.capture()  # versions, git, etc.

    def run(self) -> RunResult:
        """Execute the evaluation run."""
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        start_time = time.time()

        # Emit run started
        self.output.on_run_started(RunStarted(
            run_id=run_id,
            config=self.config,
            context=self.context,
            models=self.config.models,
            tasks=self.config.tasks,
            total_combinations=self._count_combinations(),
        ))

        # Collect texts if needed
        all_texts = None
        if self.config.use_cache and self._has_dense_models():
            all_texts = self._collect_texts()

        results = {}
        shelf_scores = {}

        for model_key in self.config.models:
            model_result = self._evaluate_model(model_key, all_texts)
            results.update(model_result)

        # Compute SHELF scores
        shelf_scores = self._compute_shelf_scores(results)
        efficiency = self._compute_efficiency(shelf_scores)

        # Emit completion
        self.output.on_run_completed(RunCompleted(
            run_id=run_id,
            shelf_scores=shelf_scores,
            efficiency_metrics=efficiency,
            total_tasks=len(results),
            failed_tasks=sum(1 for r in results.values() if "error" in r),
            duration_seconds=time.time() - start_time,
            summary_path=self.config.output_dir / "summary.json",
        ))

        return RunResult(
            shelf_scores=shelf_scores,
            efficiency=efficiency,
            results=results,
        )

    def _evaluate_model(self, model_key: str, all_texts: list[str] | None):
        """Evaluate all tasks for a single model."""
        model_config = self.config.models_config[model_key]

        # Determine tasks to run
        tasks_to_run, tasks_skipped = self._get_tasks_for_model(model_key)

        self.output.on_model_started(ModelStarted(
            model_key=model_key,
            model_name=model_config["name"],
            model_type=model_config["type"],
            tasks_to_run=len(tasks_to_run),
            tasks_skipped=tasks_skipped,
        ))

        if not tasks_to_run:
            return {}

        # Load model
        model = self._create_model(model_config)

        # Build cache if dense model
        if model_config["type"] == "sentence_transformer" and all_texts:
            model = self._build_cache(model, model_key, all_texts)

        # Evaluate tasks
        results = {}
        for task_name, task_type in tasks_to_run:
            result = self._evaluate_task(model, model_key, task_name, task_type)
            results[f"{model_key}_{task_name}"] = result

        return results
```

## CLI Integration

The CLI becomes a thin layer:

```python
@app.command("run")
def cmd_run(
    config: Path = DEFAULT_CONFIG_PATH,
    models: list[str] | None = None,
    skip_existing: bool = False,
    quiet: bool = False,
    jsonl_log: Path | None = None,
):
    """Run baseline evaluations."""
    # Build run config
    run_config = RunConfig.from_yaml(config)
    if models:
        run_config = run_config.with_models(models)
    if skip_existing:
        run_config = run_config.with_skip_existing(True)

    # Build output handler
    handlers = []

    if not quiet:
        handlers.append(RichHandler(console))

    if jsonl_log:
        handlers.append(JSONLHandler(jsonl_log))

    # Always log to Python logger
    handlers.append(LoggerHandler())

    output = CompositeHandler(handlers) if len(handlers) > 1 else handlers[0]

    # Run orchestrator
    orchestrator = EvaluationOrchestrator(run_config, output)
    result = orchestrator.run()

    # Exit with error if any tasks failed
    if result.failed_tasks > 0:
        raise typer.Exit(1)
```

## Benefits

1. **Separation of Concerns**: Orchestration logic is completely separate from output formatting
2. **Testability**: Orchestrator can be tested with a mock output handler
3. **Flexibility**: Easy to add new output formats (Slack notifications, database logging, etc.)
4. **CLI Integration**: No subprocess hack - direct Python call
5. **Rich Output**: Beautiful terminal UI with progress bars and tables
6. **Structured Logs**: JSONL for automated analysis and monitoring
7. **Backwards Compatible**: Python logger still works for traditional logging

## Migration Path

1. Create `src/shelf/evaluate/runner/` with new architecture
2. Create `src/shelf/evaluate/output/` with handlers
3. Update `src/shelf/cli_cmds/eval.py` to use orchestrator directly
4. Keep `scripts/baselines/run_all.py` as deprecated wrapper (calls new code)
5. Remove subprocess call from CLI

## Example JSONL Output

```jsonl
{"timestamp": "2025-12-14T05:30:00Z", "event": "run_started", "run_id": "20251214_053000", "models": ["minilm", "bge_small"], "tasks": 16}
{"timestamp": "2025-12-14T05:30:01Z", "event": "model_started", "model_key": "minilm", "tasks_to_run": 16}
{"timestamp": "2025-12-14T05:30:05Z", "event": "cache_built", "model_key": "minilm", "num_entries": 50508, "memory_mb": 78.4}
{"timestamp": "2025-12-14T05:30:10Z", "event": "task_completed", "model_key": "minilm", "task_name": "lcc_retrieval", "primary_score": 0.4523}
...
{"timestamp": "2025-12-14T05:45:00Z", "event": "run_completed", "total_tasks": 32, "failed_tasks": 0, "duration_seconds": 900}
```

## Example Rich Output

```
╭──────────────────────────────────────────────────────────────────────────────╮
│ SHELF Baseline Evaluation                                                    │
│ Dataset: v0.3.0 | Models: 2 | Tasks: 16                                      │
╰──────────────────────────────────────────────────────────────────────────────╯

Environment
├── Python    3.13.7
├── sklearn   1.8.0
└── Git       06303cb (dirty=True)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 MiniLM-L6 (22M params, 384 dims)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Embedding 50508 texts... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:05:23

  ✓ lcc_retrieval: ndcg@10=0.4523 (2.3s)
  ✓ form_retrieval: ndcg@10=0.3891 (2.1s)
  ✓ lcc_classification: macro_f1=0.5234 (1.8s)
  ...

                      SHELF Score Rankings
┏━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┓
┃ Rank ┃ Model           ┃  SHELF ┃ SHELF_eff ┃ Pareto ┃
┡━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━┩
│    1 │ BGE-small       │ 0.4964 │     65.98 │   ✓    │
│    2 │ MiniLM-L6       │ 0.4647 │     63.17 │   ✓    │
└──────┴─────────────────┴────────┴───────────┴────────┘

Summary saved to: results/v0.3.0/baselines/summary.json
Total time: 15m 23s | 32 tasks | 0 errors
```

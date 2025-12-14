"""Beautiful Rich console output handler for SHELF evaluations.

This module provides colorful, interactive terminal output using the rich library.
The RichHandler displays evaluation progress with progress bars, tables, panels,
and color-coded status messages.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
)
from rich.rule import Rule
from rich.table import Table

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


class RichHandler:
    """Rich console output handler for SHELF baseline evaluations.

    Provides beautiful, interactive terminal output with:
    - Header panels with run configuration
    - Progress bars for embedding/caching
    - Color-coded task status (green check, red X, dim circle)
    - Summary tables with SHELF scores and rankings
    - Environment metadata display

    Attributes:
        console: Rich Console instance for output
        progress: Progress bar instance (None when not active)
        progress_task_id: Current progress task ID (None when not active)
    """

    def __init__(self, console: Console | None = None) -> None:
        """Initialize the Rich handler.

        Args:
            console: Console instance to use. If None, creates a new one.
        """
        self.console = console if console is not None else Console()
        self.progress: Progress | None = None
        self.progress_task_id: TaskID | None = None

    def on_run_started(self, event: RunStarted) -> None:
        """Display run header with configuration and environment.

        Shows:
        - SHELF Baseline Evaluation panel header
        - Dataset version, model count, task count
        - Environment table (Python, sklearn, Git commit, etc.)

        Args:
            event: Run started event with configuration
        """
        try:
            # Header panel
            header_text = "[bold cyan]SHELF Baseline Evaluation[/bold cyan]"
            self.console.print(Panel(header_text, expand=False))

            # Configuration summary
            dataset_version = event.context.get("dataset_version", "unknown")
            model_count = len(event.models)
            task_count = len(event.tasks)

            self.console.print(
                f"[bold]Dataset:[/bold] {dataset_version} | "
                f"[bold]Models:[/bold] {model_count} | "
                f"[bold]Tasks:[/bold] {task_count}"
            )
            self.console.print()

            # Environment table
            env_table = Table(title="Environment", show_header=False, box=None)
            env_table.add_column("Key", style="bold")
            env_table.add_column("Value")

            python_version = event.context.get("python_version", "unknown")
            sklearn_version = event.context.get("sklearn_version", "unknown")
            git_commit = event.context.get("git_commit", "unknown")
            git_dirty = event.context.get("git_dirty", False)

            env_table.add_row("Python", python_version)
            env_table.add_row("sklearn", sklearn_version)
            env_table.add_row(
                "Git commit",
                f"{git_commit}{' (dirty)' if git_dirty else ''}",
            )

            self.console.print(env_table)
            self.console.print()

        except Exception as e:
            # Fallback to simple output if rich formatting fails
            self.console.print(f"[red]Error formatting run header: {e}[/red]")
            self.console.print(f"Starting evaluation run {event.run_id}")

    def on_model_started(self, event: ModelStarted) -> None:
        """Display model header with task summary.

        Shows:
        - Horizontal rule with model name and type
        - "X tasks to run, Y skipped" summary

        Args:
            event: Model started event with model info
        """
        try:
            # Model header rule
            model_display = f"{event.model_name} ({event.model_type})"
            self.console.print(Rule(f"[bold blue]{model_display}[/bold blue]"))

            # Task summary
            self.console.print(
                f"  [bold]{event.tasks_to_run}[/bold] tasks to run, "
                f"[dim]{event.tasks_skipped}[/dim] skipped"
            )

        except Exception as e:
            self.console.print(f"[red]Error formatting model header: {e}[/red]")
            self.console.print(f"Evaluating model: {event.model_key}")

    def on_cache_building(self, event: CacheBuilding) -> None:
        """Start progress bar for embedding cache construction.

        Creates a progress bar with:
        - Spinner column
        - Text description
        - Bar column
        - Task progress column (percentage, current/total)

        Args:
            event: Cache building event with text count
        """
        try:
            # Create progress bar
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=self.console,
            )

            # Start progress and add task
            self.progress.start()
            self.progress_task_id = self.progress.add_task(
                f"Embedding {event.num_texts:,} texts...",
                total=event.num_texts,
            )

        except Exception as e:
            self.console.print(f"[red]Error starting progress bar: {e}[/red]")
            self.console.print(f"  Building cache for {event.num_texts:,} texts...")

    def on_embedding_progress(self, event: EmbeddingProgress) -> None:
        """Update embedding progress bar.

        Args:
            event: Embedding progress event with current/total counts
        """
        try:
            if self.progress is not None and self.progress_task_id is not None:
                self.progress.update(self.progress_task_id, completed=event.current)

        except Exception:
            # Silently ignore progress update errors to avoid flooding console
            pass

    def on_cache_built(self, event: CacheBuilt) -> None:
        """Stop progress bar and display cache statistics.

        Shows:
        - Cache entry count
        - Memory usage in MB
        - Duration in seconds
        - Batch timing statistics (mean, std, throughput)

        Args:
            event: Cache built event with performance metrics
        """
        try:
            # Stop progress bar
            if self.progress is not None:
                self.progress.stop()
                self.progress = None
                self.progress_task_id = None

            # Display cache stats
            self.console.print(
                f"  [dim]Cache: {event.num_entries:,} entries, "
                f"{event.memory_mb:.1f} MB ({event.duration_seconds:.1f}s)[/dim]"
            )

            # Display batch timing stats if available
            if event.num_batches > 0:
                self.console.print(
                    f"  [dim]Batches: {event.num_batches:,} | "
                    f"Time/batch: {event.mean_batch_time:.3f}s ± {event.std_batch_time:.3f}s | "
                    f"Throughput: {event.throughput:.1f} samples/s[/dim]"
                )

        except Exception as e:
            self.console.print(f"[red]Error displaying cache stats: {e}[/red]")

    def on_task_started(self, event: TaskStarted) -> None:
        """Handle task started event.

        Currently a no-op since task completion will show the result.
        Could be extended to show a spinner for long-running tasks.

        Args:
            event: Task started event
        """
        # No-op for now - task completion will show result
        pass

    def on_task_completed(self, event: TaskCompleted) -> None:
        """Display task completion with green checkmark.

        Format: "  ✓ task_name: metric=score (duration)"

        Args:
            event: Task completed event with metrics
        """
        try:
            self.console.print(
                f"  [green]✓[/green] {event.task_name}: "
                f"{event.primary_metric}={event.primary_score:.4f} "
                f"({event.duration_seconds:.1f}s)"
            )

        except Exception as e:
            self.console.print(f"[red]Error formatting task result: {e}[/red]")
            self.console.print(f"  Completed: {event.task_name}")

    def on_task_failed(self, event: TaskFailed) -> None:
        """Display task failure with red X.

        Format: "  ✗ task_name: error"

        Args:
            event: Task failed event with error details
        """
        try:
            self.console.print(f"  [red]✗[/red] {event.task_name}: {event.error}")

        except Exception as e:
            self.console.print(f"Error formatting task failure: {e}")
            self.console.print(f"  Failed: {event.task_name} - {event.error}")

    def on_task_skipped(self, event: TaskSkipped) -> None:
        """Display task skip with dim circle.

        Format: "  ○ task_name: skipped (reason)"

        Args:
            event: Task skipped event with reason
        """
        try:
            self.console.print(
                f"  [dim]○ {event.task_name}: skipped ({event.reason})[/dim]"
            )

        except Exception as e:
            self.console.print(f"Error formatting task skip: {e}")
            self.console.print(f"  Skipped: {event.task_name} - {event.reason}")

    def on_model_completed(self, event: ModelCompleted) -> None:
        """Display model completion summary.

        Shows count of completed/failed/skipped tasks and total duration.

        Args:
            event: Model completed event with task counts
        """
        try:
            self.console.print(
                f"  [bold]Completed:[/bold] {event.tasks_completed} tasks, "
                f"{event.tasks_failed} failed, {event.tasks_skipped} skipped "
                f"({event.duration_seconds:.1f}s)"
            )
            self.console.print()

        except Exception as e:
            self.console.print(f"[red]Error formatting model summary: {e}[/red]")
            self.console.print(
                f"  Model completed: {event.tasks_completed} tasks "
                f"in {event.duration_seconds:.1f}s"
            )
            self.console.print()

    def on_run_completed(self, event: RunCompleted) -> None:
        """Display final summary with SHELF scores and rankings.

        Shows:
        - SHELF Score Rankings table (Rank, Model, SHELF, SHELF_eff, Pareto, Size)
        - Best by Size Category table
        - Total time, task count, error count
        - Path to summary file

        Args:
            event: Run completed event with final scores
        """
        try:
            # SHELF Score Rankings table
            scores_table = Table(title="SHELF Score Rankings", show_header=True)
            scores_table.add_column("Rank", justify="right", style="cyan")
            scores_table.add_column("Model", style="bold")
            scores_table.add_column("SHELF", justify="right", style="green")
            scores_table.add_column("SHELF_eff", justify="right", style="yellow")
            scores_table.add_column("Pareto", justify="center")
            scores_table.add_column("Size", justify="right")

            # Sort models by SHELF score descending
            sorted_models = sorted(
                event.shelf_scores.items(),
                key=lambda x: x[1],
                reverse=True,
            )

            for rank, (model_key, shelf_score) in enumerate(sorted_models, start=1):
                eff_metrics = event.efficiency_metrics.get(model_key, {})
                shelf_eff = eff_metrics.get("shelf_eff", 0.0)
                is_pareto = eff_metrics.get("is_pareto", False)
                size_category = eff_metrics.get("size_category", "unknown")

                scores_table.add_row(
                    str(rank),
                    model_key,
                    f"{shelf_score:.4f}",
                    f"{shelf_eff:.4f}",
                    "✓" if is_pareto else "",
                    size_category,
                )

            self.console.print()
            self.console.print(scores_table)
            self.console.print()

            # Best by Size Category table
            size_best: dict[str, tuple[str, float]] = {}
            for model_key, shelf_score in event.shelf_scores.items():
                eff_metrics = event.efficiency_metrics.get(model_key, {})
                size_category = eff_metrics.get("size_category", "unknown")

                if (
                    size_category not in size_best
                    or shelf_score > size_best[size_category][1]
                ):
                    size_best[size_category] = (model_key, shelf_score)

            if size_best:
                size_table = Table(title="Best by Size Category", show_header=True)
                size_table.add_column("Size", style="cyan")
                size_table.add_column("Model", style="bold")
                size_table.add_column("SHELF", justify="right", style="green")

                # Sort by size category
                for size_category in sorted(size_best.keys()):
                    model_key, shelf_score = size_best[size_category]
                    size_table.add_row(
                        size_category,
                        model_key,
                        f"{shelf_score:.4f}",
                    )

                self.console.print(size_table)
                self.console.print()

            # Summary statistics
            total_minutes = int(event.duration_seconds // 60)
            total_seconds = int(event.duration_seconds % 60)

            self.console.print(
                f"[bold]Total time:[/bold] {total_minutes}m {total_seconds}s | "
                f"[bold]{event.completed_tasks}[/bold] tasks | "
                f"[red]{event.failed_tasks}[/red] errors"
            )

            # Summary file path
            if event.summary_path is not None:
                self.console.print(f"[dim]Summary: {event.summary_path}[/dim]")

            self.console.print()

        except Exception as e:
            self.console.print(f"[red]Error formatting final summary: {e}[/red]")
            # Fallback to simple summary
            self.console.print(
                f"Evaluation complete: {event.completed_tasks} tasks completed, "
                f"{event.failed_tasks} failed in {event.duration_seconds:.1f}s"
            )
            if event.summary_path is not None:
                self.console.print(f"Summary: {event.summary_path}")

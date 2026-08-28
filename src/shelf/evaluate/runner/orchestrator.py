"""Evaluation orchestrator for SHELF baseline evaluations.

This module provides the EvaluationOrchestrator class which coordinates
the evaluation of models across tasks, emitting events to OutputHandlers
for unified reporting.

The orchestrator:
1. Loads and validates configuration
2. Captures environment context for reproducibility
3. Manages model lifecycle (loading, caching, cleanup)
4. Evaluates each model on applicable tasks
5. Computes SHELF scores and efficiency metrics
6. Saves results and manifests

Example:
    ```python
    from shelf.evaluate.runner.config import RunConfig
    from shelf.evaluate.runner.orchestrator import EvaluationOrchestrator
    from shelf.evaluate.output import RichHandler

    config = RunConfig.from_yaml("config.yaml")
    orchestrator = EvaluationOrchestrator(config, output=RichHandler())
    result = orchestrator.run()
    ```
"""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from shelf.evaluate.output.base import OutputHandler
    from shelf.evaluate.runner.config import RunConfig


@dataclass
class OrchestrationResult:
    """Result of an evaluation run.

    Attributes:
        run_id: Unique identifier for this run
        shelf_scores: SHELF scores by model key
        efficiency_metrics: Efficiency metrics by model key
        results: All task results (model_task -> result dict)
        completed_tasks: Number of successfully completed tasks
        failed_tasks: Number of failed tasks
        skipped_tasks: Number of skipped tasks
        duration_seconds: Total run duration
        summary_path: Path to summary JSON (None if not saved)
    """

    run_id: str
    shelf_scores: dict[str, float]
    efficiency_metrics: dict[str, dict[str, Any]]
    results: dict[str, dict[str, Any]] = field(default_factory=dict)
    completed_tasks: int = 0
    failed_tasks: int = 0
    skipped_tasks: int = 0
    duration_seconds: float = 0.0
    summary_path: Path | None = None


class EvaluationOrchestrator:
    """Orchestrates SHELF baseline evaluations with event-driven output.

    The orchestrator coordinates model evaluation across tasks, emitting events
    to OutputHandlers at each step. This separates evaluation logic from output
    presentation, enabling multiple output formats (Rich console, JSONL, logging).

    Attributes:
        config: Run configuration
        output: Output handler for events
        dry_run: If True, only simulate without running evaluations
    """

    def __init__(
        self,
        config: RunConfig,
        output: OutputHandler,
        *,
        dry_run: bool = False,
    ) -> None:
        """Initialize the orchestrator.

        Args:
            config: Run configuration loaded from YAML
            output: Output handler for emitting events
            dry_run: If True, report what would run without executing
        """
        self.config = config
        self.output = output
        self.dry_run = dry_run

        # Import RunContext here to avoid circular imports at module level
        from shelf.evaluate.runner.context import RunContext

        self._context = RunContext.capture(config.dataset_version)

        # Store model info (num_params_torch, hidden_size, etc.) keyed by model_key
        self._model_info: dict[str, dict[str, Any]] = {}

    def run(self) -> OrchestrationResult:
        """Execute the evaluation run.

        Returns:
            OrchestrationResult with scores, metrics, and results
        """
        from shelf.evaluate.runner.events import (
            ModelCompleted,
            ModelStarted,
            RunCompleted,
            RunStarted,
            TaskCompleted,
            TaskFailed,
            TaskSkipped,
            TaskStarted,
        )

        start_time = datetime.now(timezone.utc)
        run_id = start_time.strftime("%Y%m%d_%H%M%S")

        # Calculate total combinations
        total_combinations = self._calculate_total_combinations()

        # Emit run started
        self.output.on_run_started(
            RunStarted(
                run_id=run_id,
                models=self.config.models,
                tasks=self.config.tasks,
                total_combinations=total_combinations,
                context=self._context.to_dict(),
            )
        )

        if self.dry_run:
            return self._dry_run_result(run_id)

        # Setup output directory
        output_dir = self._setup_output_dir()

        # Collect all texts for caching if needed
        all_texts: list[str] | None = None
        if self.config.use_cache and self.config.has_dense_models():
            all_texts = self._collect_all_texts()

        # Track results
        all_results: dict[str, dict[str, Any]] = {}
        total_completed = 0
        total_failed = 0
        total_skipped = 0
        multiple_classifiers = self._has_multiple_classifiers()

        # Evaluate each model
        for model_key in self.config.models:
            model_config = self.config.get_model_config(model_key)
            model_name = model_config["name"]
            model_type = model_config["type"]

            # Get tasks for this model
            model_tasks = self.config.get_tasks_for_model(model_key)

            # Check which tasks need to run
            tasks_to_run, tasks_to_skip = self._filter_tasks(
                model_key, model_tasks, output_dir, model_type=model_type
            )

            # Load existing results for skipped tasks
            for task_name, reason in tasks_to_skip:
                if reason == "already exists" and multiple_classifiers:
                    for head in self.config.classification_heads:
                        result_key = self._get_result_key(
                            model_key, task_name, classifier=head, multi_heads=True
                        )
                        result_path = self._get_result_path(
                            model_key,
                            task_name,
                            classifier=head,
                            multi_heads=True,
                            output_dir=output_dir,
                        )
                        if result_path.exists():
                            with open(result_path) as f:
                                all_results[result_key] = json.load(f)
                else:
                    result_key = self._get_result_key(model_key, task_name)
                    result_path = self._get_result_path(
                        model_key, task_name, output_dir=output_dir
                    )
                    if result_path.exists():
                        with open(result_path) as f:
                            all_results[result_key] = json.load(f)

            # Emit model started
            self.output.on_model_started(
                ModelStarted(
                    model_key=model_key,
                    model_name=model_name,
                    model_type=model_type,
                    tasks_to_run=len(tasks_to_run),
                    tasks_skipped=len(tasks_to_skip),
                )
            )

            # Emit skip events
            for task_name, reason in tasks_to_skip:
                self.output.on_task_skipped(
                    TaskSkipped(
                        model_key=model_key,
                        task_name=task_name,
                        reason=reason,
                    )
                )
                total_skipped += 1

            # Skip model loading if nothing to run
            if not tasks_to_run:
                self.output.on_model_completed(
                    ModelCompleted(
                        model_key=model_key,
                        tasks_completed=0,
                        tasks_failed=0,
                        tasks_skipped=len(tasks_to_skip),
                        duration_seconds=0.0,
                    )
                )
                continue

            model_start_time = time.time()

            # Load model
            try:
                model = self._create_model(model_config)
            except Exception as e:
                # Failed to load model - fail all tasks
                for task_name, task_type in tasks_to_run:
                    self.output.on_task_failed(
                        TaskFailed(
                            model_key=model_key,
                            task_name=task_name,
                            task_type=task_type,
                            error=f"Failed to load model: {e}",
                            traceback=traceback.format_exc(),
                        )
                    )
                    total_failed += 1

                self.output.on_model_completed(
                    ModelCompleted(
                        model_key=model_key,
                        tasks_completed=0,
                        tasks_failed=len(tasks_to_run),
                        tasks_skipped=len(tasks_to_skip),
                        duration_seconds=time.time() - model_start_time,
                    )
                )
                continue

            # Build cache for dense models
            eval_model = model
            if (
                self.config.use_cache
                and model_type == "sentence_transformer"
                and all_texts is not None
            ):
                eval_model = self._build_cache(model, model_key, model_name, all_texts)

            # Evaluate each task
            model_completed = 0
            model_failed = 0

            for task_name, task_type in tasks_to_run:
                # Emit task started
                self.output.on_task_started(
                    TaskStarted(
                        model_key=model_key,
                        task_name=task_name,
                        task_type=task_type,
                    )
                )

                task_start_time = time.time()

                try:
                    results = self._evaluate_task(
                        model=eval_model,
                        model_key=model_key,
                        model_config=model_config,
                        task_name=task_name,
                        task_type=task_type,
                        output_dir=output_dir,
                    )

                    task_duration = time.time() - task_start_time

                    for result in results:
                        # Add efficiency metrics (pass model_key for cached info lookup)
                        result["efficiency"] = self._get_efficiency_dict(
                            model_config, model_key
                        )

                        classifier = result.get("classifier")
                        result_key = self._get_result_key(
                            model_key,
                            task_name,
                            classifier=classifier,
                            multi_heads=self._has_multiple_classifiers(),
                        )
                        result_path = self._get_result_path(
                            model_key,
                            task_name,
                            classifier=classifier,
                            multi_heads=self._has_multiple_classifiers(),
                            output_dir=output_dir,
                        )

                        with open(result_path, "w") as f:
                            json.dump(result, f, indent=2, default=str)

                        all_results[result_key] = result

                        # Emit task completed
                        display_task_name = (
                            f"{task_name} [{classifier}]" if classifier else task_name
                        )
                        self.output.on_task_completed(
                            TaskCompleted(
                                model_key=model_key,
                                task_name=display_task_name,
                                task_type=task_type,
                                primary_metric=result["primary_metric"],
                                primary_score=result["primary_score"],
                                metrics=result["metrics"],
                                duration_seconds=task_duration,
                                result_path=result_path,
                            )
                        )

                        model_completed += 1
                        total_completed += 1

                except Exception as e:
                    task_duration = time.time() - task_start_time

                    # Save error result
                    error_result = {
                        "model": model_config["name"],
                        "model_key": model_key,
                        "task": task_name,
                        "task_type": task_type,
                        "error": str(e),
                    }
                    result_key = f"{model_key}_{task_name}"
                    all_results[result_key] = error_result

                    # Emit task failed
                    self.output.on_task_failed(
                        TaskFailed(
                            model_key=model_key,
                            task_name=task_name,
                            task_type=task_type,
                            error=str(e),
                            traceback=traceback.format_exc(),
                        )
                    )

                    model_failed += 1
                    total_failed += 1

                # Reset model for next task (sparse models need refitting)
                if hasattr(model, "reset"):
                    model.reset()

            model_duration = time.time() - model_start_time

            # Emit model completed
            self.output.on_model_completed(
                ModelCompleted(
                    model_key=model_key,
                    tasks_completed=model_completed,
                    tasks_failed=model_failed,
                    tasks_skipped=len(tasks_to_skip),
                    duration_seconds=model_duration,
                )
            )

        # Compute SHELF scores
        shelf_scores = self._compute_shelf_scores(all_results)

        # Compute efficiency metrics
        efficiency_metrics = self._compute_efficiency_metrics(shelf_scores, all_results)

        # Update results with SHELF scores and efficiency
        self._update_results_with_scores(
            all_results, shelf_scores, efficiency_metrics, output_dir
        )

        # Save summary
        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()

        summary_path = self._save_summary(
            output_dir=output_dir,
            all_results=all_results,
            shelf_scores=shelf_scores,
            efficiency_metrics=efficiency_metrics,
            start_time=start_time,
            end_time=end_time,
        )

        # Emit run completed
        self.output.on_run_completed(
            RunCompleted(
                run_id=run_id,
                shelf_scores=shelf_scores,
                efficiency_metrics=efficiency_metrics,
                total_tasks=total_combinations,
                completed_tasks=total_completed,
                failed_tasks=total_failed,
                skipped_tasks=total_skipped,
                duration_seconds=duration,
                summary_path=summary_path,
            )
        )

        return OrchestrationResult(
            run_id=run_id,
            shelf_scores=shelf_scores,
            efficiency_metrics=efficiency_metrics,
            results=all_results,
            completed_tasks=total_completed,
            failed_tasks=total_failed,
            skipped_tasks=total_skipped,
            duration_seconds=duration,
            summary_path=summary_path,
        )

    def _calculate_total_combinations(self) -> int:
        """Calculate total number of model-task (and classifier) combinations."""
        heads = self.config.classification_heads or ["logistic_regression"]
        total = 0
        for model_key in self.config.models:
            model_tasks = self.config.get_tasks_for_model(model_key)
            for _, task_type in model_tasks:
                total += len(heads) if task_type == "classification" else 1
        return total

    def _dry_run_result(self, run_id: str) -> OrchestrationResult:
        """Return empty result for dry run mode."""
        return OrchestrationResult(
            run_id=run_id,
            shelf_scores={},
            efficiency_metrics={},
        )

    def _setup_output_dir(self) -> Path:
        """Setup and return output directory."""
        output_dir = (
            self.config.output_dir / f"v{self.config.dataset_version}" / "baselines"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _collect_all_texts(self) -> list[str]:
        """Collect all unique texts needed for evaluation tasks."""
        from datasets import load_dataset

        texts: set[str] = set()

        # Main dataset texts (body only)
        ds = load_dataset(self.config.dataset_repo)
        for split in ["train", "validation", "test"]:
            texts.update(ds[split]["text"])

        # Pair dataset texts (title + body format)
        pair_tasks = self.config.tasks_config.get("pair_classification", [])
        for task_name in pair_tasks:
            try:
                pair_ds = load_dataset(
                    self.config.dataset_repo, task_name, split="test"
                )
                for row in pair_ds:
                    text_a = f"{row['doc_a_title']}\n\n{row['doc_a_body']}"
                    text_b = f"{row['doc_b_title']}\n\n{row['doc_b_body']}"
                    texts.add(text_a)
                    texts.add(text_b)
            except Exception:
                pass

        return list(texts)

    def _filter_tasks(
        self,
        model_key: str,
        model_tasks: list[tuple[str, str]],
        output_dir: Path,
        model_type: str | None = None,
    ) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        """Filter tasks to run vs skip.

        Returns:
            Tuple of (tasks_to_run, tasks_to_skip) where tasks_to_skip
            is list of (task_name, reason) tuples
        """
        to_run: list[tuple[str, str]] = []
        to_skip: list[tuple[str, str]] = []
        heads = self.config.classification_heads or ["logistic_regression"]
        multi_heads = len(heads) > 1
        if model_type is not None and hasattr(model_type, "lower"):
            # If the model exposes predict (classifier) we will only run one head
            if model_type == "classifier":
                multi_heads = False

        for task_name, task_type in model_tasks:
            if (
                task_type == "classification"
                and self.config.skip_existing
                and multi_heads
            ):
                # Skip only if all classifier variants already exist
                paths = [
                    self._get_result_path(
                        model_key,
                        task_name,
                        classifier=head,
                        multi_heads=True,
                        output_dir=output_dir,
                    )
                    for head in heads
                ]
                if all(p.exists() for p in paths):
                    to_skip.append((task_name, "already exists"))
                    continue

            result_path = self._get_result_path(
                model_key,
                task_name,
                classifier=heads[0] if multi_heads else None,
                multi_heads=multi_heads,
                output_dir=output_dir,
            )

            if self.config.skip_existing and result_path.exists():
                to_skip.append((task_name, "already exists"))
            else:
                to_run.append((task_name, task_type))

        return to_run, to_skip

    def _has_multiple_classifiers(self) -> bool:
        """Return True if more than one classification head is configured."""
        heads = self.config.classification_heads or ["logistic_regression"]
        return len(heads) > 1

    def _get_result_path(
        self,
        model_key: str,
        task_name: str,
        *,
        classifier: str | None = None,
        multi_heads: bool = False,
        output_dir: Path | None = None,
    ) -> Path:
        """Build result path, appending classifier suffix when running variants."""
        base_dir = output_dir or (
            self.config.output_dir / f"v{self.config.dataset_version}" / "baselines"
        )

        filename = f"{model_key}_{task_name}"
        if multi_heads and classifier:
            filename = f"{filename}_{classifier}"
        filename = f"{filename}.json"
        return Path(base_dir) / filename

    def _get_result_key(
        self,
        model_key: str,
        task_name: str,
        *,
        classifier: str | None = None,
        multi_heads: bool = False,
    ) -> str:
        """Build an in-memory result key consistent with file naming."""
        key = f"{model_key}_{task_name}"
        if multi_heads and classifier:
            key = f"{key}_{classifier}"
        return key

    def _create_model(self, model_config: dict[str, Any]) -> Any:
        """Create a model instance from configuration."""
        model_type = str(model_config["type"])

        if model_type == "tf":
            from shelf.evaluate.adapters import TfEmbedder

            params = model_config.get("params", {})
            embedding_dim = int(params.get("embedding_dim", 256))
            return TfEmbedder(embedding_dim=embedding_dim)

        elif model_type == "tfidf":
            from shelf.evaluate.adapters import TfidfEmbedder

            params = model_config.get("params", {})
            embedding_dim = int(params.get("embedding_dim", 256))
            return TfidfEmbedder(embedding_dim=embedding_dim)

        elif model_type == "bm25":
            from shelf.evaluate.adapters import BM25Retriever

            params = model_config.get("params", {})
            k1 = float(params.get("k1", 1.5))
            b = float(params.get("b", 0.75))
            return BM25Retriever(k1=k1, b=b)

        elif model_type == "sentence_transformer":
            from shelf.evaluate.adapters import SentenceTransformerEmbedder

            model_name = model_config.get("model_name")
            if not model_name:
                raise ValueError("sentence_transformer requires model_name")
            return SentenceTransformerEmbedder.from_pretrained(model_name)

        elif model_type in ("transformers_classifier", "hf_classifier"):
            from shelf.evaluate.adapters import TransformersSequenceClassifier

            model_name = model_config.get("model_name")
            if not model_name:
                raise ValueError("transformers_classifier requires model_name")
            device = model_config.get("device")
            max_length = model_config.get("max_length")
            trust_remote_code = bool(model_config.get("trust_remote_code", False))
            return TransformersSequenceClassifier.from_pretrained(
                model_name,
                device=device,
                max_length=max_length,
                trust_remote_code=trust_remote_code,
            )

        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _build_cache(
        self,
        model: Any,
        model_key: str,
        model_name: str,
        all_texts: list[str],
    ) -> Any:
        """Build embedding cache for a dense model."""
        from shelf.evaluate.adapters.cached import CachedEmbedder
        from shelf.evaluate.runner.events import (
            CacheBuilding,
            CacheBuilt,
            EmbeddingProgress,
        )

        # Extract model info if available (for efficiency metrics)
        model_info: dict[str, Any] = {}
        if hasattr(model, "get_model_info"):
            model_info = model.get_model_info()
            self._model_info[model_key] = model_info

        # Emit cache building
        self.output.on_cache_building(
            CacheBuilding(model_key=model_key, num_texts=len(all_texts))
        )

        # Calculate total bytes for throughput measurement
        total_bytes = sum(len(t.encode("utf-8")) for t in all_texts)

        cache_start = time.time()

        # Embed all texts (show_progress=True for user feedback during long cache builds)
        embeddings = model.encode(
            all_texts,
            batch_size=self.config.batch_size,
            show_progress=self.config.show_progress,
        )

        # Emit final progress
        self.output.on_embedding_progress(
            EmbeddingProgress(
                model_key=model_key,
                current=len(all_texts),
                total=len(all_texts),
            )
        )

        cache = {text: emb for text, emb in zip(all_texts, embeddings)}
        cache_duration = time.time() - cache_start
        memory_mb = embeddings.nbytes / 1024 / 1024

        # Calculate throughput in bytes per second
        throughput_bytes_sec = (
            total_bytes / cache_duration if cache_duration > 0 else 0.0
        )

        # Store throughput in model info for efficiency metrics
        self._model_info.setdefault(model_key, {})["throughput_bytes_sec"] = (
            throughput_bytes_sec
        )

        # Emit cache built with enhanced metrics
        self.output.on_cache_built(
            CacheBuilt(
                model_key=model_key,
                num_entries=len(cache),
                memory_mb=memory_mb,
                duration_seconds=cache_duration,
                total_bytes=total_bytes,
                throughput_bytes_sec=throughput_bytes_sec,
                num_params_torch=model_info.get("num_params_torch"),
                hidden_size=model_info.get("hidden_size"),
                context_window=model_info.get("context_window"),
            )
        )

        return CachedEmbedder(
            cache=cache,
            model_name=model_name,
            embedding_dim=model.embedding_dim,
            # See run_all.py: tasks that transform their inputs (instruction
            # retrieval) produce texts absent from a corpus-text cache.
            fallback=model,
        )

    def _evaluate_task(
        self,
        model: Any,
        model_key: str,
        model_config: dict[str, Any],
        task_name: str,
        task_type: str,
        output_dir: Path,
    ) -> list[dict[str, Any]]:
        """Evaluate a single model on a single task."""
        from shelf.evaluate import evaluate
        from shelf.evaluate.evaluators.pair import PairClassificationEvaluator
        from shelf.evaluate.registry import get_task

        task_spec = get_task(task_name)
        model_type = model_config["type"]
        model_name = model_config["name"]

        results: list[dict[str, Any]] = []

        # Handle pair classification separately for TF/TF-IDF/BM25
        if task_type == "pair_classification" and model_type in ("tf", "tfidf", "bm25"):
            evaluator = PairClassificationEvaluator(task_spec)
            if model_type == "bm25":
                eval_result = evaluator.evaluate_bm25(
                    model,
                    show_progress=self.config.show_progress,
                    save_samples=self.config.save_samples,
                )
            else:
                eval_result = evaluator.evaluate_tfidf(
                    model,
                    show_progress=self.config.show_progress,
                    save_samples=self.config.save_samples,
                )

            results.append(
                self._build_result_dict(
                    eval_result,
                    model_name,
                    model_key,
                    model_type,
                    task_name,
                    task_type,
                    output_dir=output_dir,
                )
            )
            return results

        # Standard evaluation
        if task_type == "classification" and not hasattr(model, "predict"):
            # Run one or more classification heads on top of embeddings
            heads = self.config.classification_heads or ["logistic_regression"]
            primary_head = heads[0]
            for head in heads:
                eval_result = evaluate(
                    task=task_name,
                    model=model,
                    max_queries=self.config.max_queries,
                    batch_size=self.config.batch_size,
                    show_progress=self.config.show_progress,
                    save_samples=self.config.save_samples,
                    model_key=model_key,
                    classifier=head,
                )
                result_dict = self._build_result_dict(
                    eval_result,
                    model_name,
                    model_key,
                    model_type,
                    task_name,
                    task_type,
                    classifier=head,
                    classifier_primary=head == primary_head,
                    output_dir=output_dir,
                )
                results.append(result_dict)
            return results

        # Default single evaluation path
        eval_result = evaluate(
            task=task_name,
            model=model,
            max_queries=self.config.max_queries,
            batch_size=self.config.batch_size,
            show_progress=self.config.show_progress,
            save_samples=self.config.save_samples,
            model_key=model_key,
        )

        results.append(
            self._build_result_dict(
                eval_result,
                model_name,
                model_key,
                model_type,
                task_name,
                task_type,
                output_dir=output_dir,
            )
        )

        return results

    def _build_result_dict(
        self,
        eval_result: Any,
        model_name: str,
        model_key: str,
        model_type: str,
        task_name: str,
        task_type: str,
        *,
        classifier: str | None = None,
        classifier_primary: bool = True,
        output_dir: Path,
    ) -> dict[str, Any]:
        """Construct result dictionary and save per-sample outputs if needed."""
        result_dict: dict[str, Any] = {
            "model": model_name,
            "model_key": model_key,
            "model_type": model_type,
            "task": task_name,
            "task_type": task_type,
            "primary_metric": eval_result.primary_metric,
            "primary_score": eval_result.primary_score,
            "metrics": eval_result.metrics,
            "num_samples": eval_result.num_samples,
            "context": eval_result.context.to_dict() if eval_result.context else None,
        }

        if classifier:
            result_dict["classifier"] = classifier
            result_dict["classifier_primary"] = classifier_primary

        # Save per-sample results if present
        if self.config.save_samples and eval_result.per_sample_results:
            samples_suffix = (
                f"_{classifier}"
                if classifier and self._has_multiple_classifiers()
                else ""
            )
            samples_path = (
                output_dir / f"{model_key}_{task_name}{samples_suffix}_samples.jsonl.gz"
            )
            eval_result.per_sample_results.save(samples_path)
            result_dict["per_sample_path"] = str(samples_path)

        return result_dict

    def _get_efficiency_dict(
        self, model_config: dict[str, Any], model_key: str | None = None
    ) -> dict[str, Any]:
        """Get efficiency metrics dict for a model.

        Args:
            model_config: Model configuration from config.yaml
            model_key: Optional model key to look up cached model info

        Returns:
            Dict of efficiency metrics for JSON serialization
        """
        from shelf.evaluate.efficiency import (
            compute_efficiency_metrics,
            get_size_category,
        )

        num_params = model_config.get("num_params")

        if num_params is None:
            # Sparse model
            return {
                "num_params": None,
                "embedding_dim": model_config.get("params", {}).get(
                    "embedding_dim", 256
                ),
                "size_category": "sparse",
                "flops_per_token": None,
                "relative_compute": None,
                "shelf_eff": None,
                "shelf_compute": None,
                "pareto_optimal": None,
                "size_rank": None,
                "num_params_torch": None,
                "hidden_size": None,
                "context_window": None,
                "throughput_bytes_sec": None,
            }

        # Get cached model info if available
        cached_info = self._model_info.get(model_key or "", {})

        # Dense model
        embedding_dim = model_config.get("embedding_dim", 0)
        size_category = model_config.get("size_category") or get_size_category(
            num_params
        )

        metrics = compute_efficiency_metrics(
            num_params=num_params,
            embedding_dim=embedding_dim,
            size_category=size_category,
            shelf_score=None,  # Computed later
            num_params_torch=cached_info.get("num_params_torch"),
            hidden_size=cached_info.get("hidden_size"),
            context_window=cached_info.get("context_window"),
            throughput_bytes_sec=cached_info.get("throughput_bytes_sec"),
        )

        return metrics.to_dict()

    def _compute_shelf_scores(
        self, results: dict[str, dict[str, Any]]
    ) -> dict[str, float]:
        """Compute aggregate SHELF Score for each model."""
        # Group results by model and task type
        model_scores: dict[str, dict[str, list[float]]] = {}

        for key, result in results.items():
            if "error" in result:
                continue

            model_key = result["model_key"]
            task_type = result["task_type"]
            if (
                task_type == "classification"
                and result.get("classifier_primary") is False
            ):
                # Skip non-primary classifier variants when aggregating SHELF score
                continue
            score = result["primary_score"]

            if model_key not in model_scores:
                model_scores[model_key] = {}
            if task_type not in model_scores[model_key]:
                model_scores[model_key][task_type] = []

            model_scores[model_key][task_type].append(score)

        # Compute weighted average
        weights = self.config.shelf_score_weights
        shelf_scores: dict[str, float] = {}

        for model_key, type_scores in model_scores.items():
            weighted_sum = 0.0
            total_weight = 0.0

            for task_type, scores in type_scores.items():
                if task_type in weights and scores:
                    avg_score = sum(scores) / len(scores)
                    weight = weights[task_type]
                    weighted_sum += avg_score * weight
                    total_weight += weight

            if total_weight > 0:
                shelf_scores[model_key] = weighted_sum / total_weight
            else:
                shelf_scores[model_key] = 0.0

        return shelf_scores

    def _compute_efficiency_metrics(
        self,
        shelf_scores: dict[str, float],
        results: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """Compute efficiency metrics with SHELF scores.

        This merges cached model info (num_params_torch, hidden_size, etc.)
        with the aggregate efficiency calculations (Pareto optimality, etc.).
        """
        from shelf.evaluate.efficiency import compute_aggregate_efficiency

        model_shelf_data = {
            model_key: {"shelf_score": score}
            for model_key, score in shelf_scores.items()
        }

        aggregate = compute_aggregate_efficiency(
            model_shelf_data, self.config.models_config
        )

        # Merge cached model info into aggregate metrics
        for model_key, metrics in aggregate.items():
            cached_info = self._model_info.get(model_key, {})
            if cached_info:
                metrics["num_params_torch"] = cached_info.get("num_params_torch")
                metrics["hidden_size"] = cached_info.get("hidden_size")
                metrics["context_window"] = cached_info.get("context_window")
                metrics["throughput_bytes_sec"] = cached_info.get(
                    "throughput_bytes_sec"
                )

        return aggregate

    def _update_results_with_scores(
        self,
        results: dict[str, dict[str, Any]],
        shelf_scores: dict[str, float],
        efficiency_metrics: dict[str, dict[str, Any]],
        output_dir: Path,
    ) -> None:
        """Update results with SHELF scores and re-save."""
        for result_key, result in results.items():
            if "error" in result:
                continue

            model_key = result["model_key"]

            if model_key in shelf_scores:
                result["shelf_score"] = round(shelf_scores[model_key], 6)

            if model_key in efficiency_metrics:
                result["efficiency"] = efficiency_metrics[model_key]

            # Re-save
            result_path = output_dir / f"{result_key}.json"
            with open(result_path, "w") as f:
                json.dump(result, f, indent=2, default=str)

    def _save_summary(
        self,
        output_dir: Path,
        all_results: dict[str, dict[str, Any]],
        shelf_scores: dict[str, float],
        efficiency_metrics: dict[str, dict[str, Any]],
        start_time: datetime,
        end_time: datetime,
    ) -> Path:
        """Save summary and manifest files."""
        # Save summary
        summary = {
            "timestamp": end_time.isoformat(),
            "dataset_version": self.config.dataset_version,
            "dataset_checksum": self._context.dataset_checksum,
            "versions": {
                "python": self._context.python_version,
                "sklearn": self._context.sklearn_version,
                "numpy": self._context.numpy_version,
                "scipy": self._context.scipy_version,
                "torch": self._context.torch_version,
                "sentence_transformers": self._context.sentence_transformers_version,
                "shelf": self._context.shelf_version,
            },
            "git": {
                "commit": self._context.git_commit,
                "branch": self._context.git_branch,
                "dirty": self._context.git_dirty,
            },
            "shelf_scores": shelf_scores,
            "efficiency": efficiency_metrics,
            "results": all_results,
        }

        summary_path = output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        # Save manifest
        manifest = {
            "run_id": start_time.strftime("%Y%m%d_%H%M%S"),
            "config_version": "1.0.0",
            "dataset_version": self.config.dataset_version,
            "dataset_checksum": self._context.dataset_checksum,
            "models_evaluated": self.config.models,
            "tasks_evaluated": [t[0] for t in self.config.tasks],
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": (end_time - start_time).total_seconds(),
            "versions": {
                "python": self._context.python_version,
                "platform": self._context.platform_info,
                "sklearn": self._context.sklearn_version,
                "numpy": self._context.numpy_version,
                "scipy": self._context.scipy_version,
                "sentence_transformers": self._context.sentence_transformers_version,
                "torch": self._context.torch_version,
                "cuda_available": str(self._context.cuda_available),
                "cuda_version": self._context.cuda_version or "N/A",
                "shelf": self._context.shelf_version,
            },
            "git": {
                "commit": self._context.git_commit,
                "branch": self._context.git_branch,
                "dirty": self._context.git_dirty,
            },
            "reproducibility": {
                "random_seed": 42,
                "num_bootstrap_samples": 1000,
                "confidence_level": 0.95,
            },
        }

        manifest_path = output_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)

        return summary_path

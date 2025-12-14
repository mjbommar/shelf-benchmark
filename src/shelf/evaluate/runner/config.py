"""Configuration for evaluation runs.

This module provides the RunConfig class for managing evaluation configurations
loaded from YAML files. The config is immutable and uses dataclass.replace() for
creating modified copies.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RunConfig:
    """Main configuration for evaluation runs.

    This dataclass is immutable (frozen=True). To create modified copies,
    use the with_* methods which use dataclass.replace() internally.

    Attributes:
        config_path: Path to the YAML config file
        output_dir: Base directory for results output
        models: List of model keys to run
        models_config: Full model configurations (model_key -> config dict)
        tasks: List of (task_name, task_type) tuples to run
        tasks_config: Task configurations (task_type -> list of task names)
        dataset_repo: HuggingFace dataset repository ID
        dataset_version: Dataset version string
        batch_size: Batch size for evaluation
        max_queries: Maximum number of queries to process (None = all)
        skip_existing: Skip tasks with existing results
        use_cache: Enable result caching
        show_progress: Show progress bars
        shelf_score_weights: Task type weights for SHELF score
        shelf_score_metrics: Metrics to use for each task type in SHELF score
    """

    config_path: Path
    output_dir: Path
    models: list[str]
    models_config: dict[str, dict[str, Any]]
    tasks: list[tuple[str, str]]
    tasks_config: dict[str, list[str]]
    dataset_repo: str
    dataset_version: str
    batch_size: int = 32
    max_queries: int | None = None
    skip_existing: bool = False
    use_cache: bool = True
    show_progress: bool = True
    classification_heads: list[str] = field(default_factory=list)
    save_samples: bool = False
    shelf_score_weights: dict[str, float] = field(default_factory=dict)
    shelf_score_metrics: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path) -> RunConfig:
        """Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            RunConfig instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML parsing fails
            KeyError: If required config keys are missing
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with path.open() as f:
            config = yaml.safe_load(f)

        # Extract models and build list
        models_config = config.get("models", {})
        models = list(models_config.keys())

        # Extract tasks and build (task_name, task_type) tuples
        tasks_config = config.get("tasks", {})
        tasks: list[tuple[str, str]] = []
        for task_type, task_names in tasks_config.items():
            for task_name in task_names:
                tasks.append((task_name, task_type))

        # Extract evaluation settings
        eval_config = config.get("evaluation", {})
        batch_size = eval_config.get("batch_size", 32)
        max_queries = eval_config.get("max_queries")
        show_progress = eval_config.get("show_progress", True)
        classification_heads = eval_config.get(
            "classification_heads", ["logistic_regression", "random_forest"]
        )

        # Extract output directory
        output_config = config.get("output", {})
        output_dir = Path(output_config.get("base_dir", "results"))

        # Extract SHELF score configuration
        shelf_score_config = config.get("shelf_score", {})
        shelf_score_weights = shelf_score_config.get("weights", {})
        shelf_score_metrics = shelf_score_config.get("metrics", {})

        # Dataset configuration
        dataset_repo = config.get("dataset_repo", "mjbommar/SHELF")
        dataset_version = config.get("dataset_version", "0.3.0")

        return cls(
            config_path=path,
            output_dir=output_dir,
            models=models,
            models_config=models_config,
            tasks=tasks,
            tasks_config=tasks_config,
            dataset_repo=dataset_repo,
            dataset_version=dataset_version,
            batch_size=batch_size,
            max_queries=max_queries,
            skip_existing=False,  # Default to False, can be changed with with_skip_existing
            use_cache=True,  # Default to True
            show_progress=show_progress,
            classification_heads=classification_heads,
            shelf_score_weights=shelf_score_weights,
            shelf_score_metrics=shelf_score_metrics,
        )

    def with_models(self, models: list[str]) -> RunConfig:
        """Return new config with filtered models.

        Args:
            models: List of model keys to include

        Returns:
            New RunConfig instance with filtered models

        Raises:
            ValueError: If any model key is not in models_config
        """
        # Validate all requested models exist
        invalid = [m for m in models if m not in self.models_config]
        if invalid:
            raise ValueError(f"Invalid model keys: {invalid}")

        return replace(self, models=models)

    def with_tasks(self, task_names: list[str]) -> RunConfig:
        """Return new config with filtered tasks.

        Args:
            task_names: List of task names to include

        Returns:
            New RunConfig instance with filtered tasks

        Raises:
            ValueError: If any task name is not in tasks
        """
        # Build set of all task names for validation
        all_task_names = {task_name for task_name, _ in self.tasks}
        invalid = [t for t in task_names if t not in all_task_names]
        if invalid:
            raise ValueError(f"Invalid task names: {invalid}")

        # Filter tasks list
        filtered_tasks = [
            (name, type_) for name, type_ in self.tasks if name in task_names
        ]

        return replace(self, tasks=filtered_tasks)

    def with_task_types(self, task_types: list[str]) -> RunConfig:
        """Return new config with filtered task types.

        Args:
            task_types: List of task types to include

        Returns:
            New RunConfig instance with filtered task types

        Raises:
            ValueError: If any task type is not in tasks_config
        """
        # Validate all requested task types exist
        invalid = [t for t in task_types if t not in self.tasks_config]
        if invalid:
            raise ValueError(f"Invalid task types: {invalid}")

        # Filter tasks list by task type
        filtered_tasks = [
            (name, type_) for name, type_ in self.tasks if type_ in task_types
        ]

        return replace(self, tasks=filtered_tasks)

    def with_classifiers(self, classifiers: list[str]) -> RunConfig:
        """Return new config with selected classification heads.

        Args:
            classifiers: List of classifier names (e.g., ["logistic_regression"])

        Returns:
            New RunConfig instance with classification heads set
        """
        return replace(self, classification_heads=classifiers)

    def with_skip_existing(self, skip: bool) -> RunConfig:
        """Return new config with skip_existing flag set.

        Args:
            skip: Whether to skip existing results

        Returns:
            New RunConfig instance with skip_existing flag set
        """
        return replace(self, skip_existing=skip)

    def with_output_dir(self, path: Path) -> RunConfig:
        """Return new config with output directory set.

        Args:
            path: Output directory path

        Returns:
            New RunConfig instance with output_dir set
        """
        return replace(self, output_dir=Path(path))

    def with_save_samples(self, save: bool) -> RunConfig:
        """Return new config with save_samples flag set.

        When save_samples is True, per-sample results are captured
        and saved to .jsonl.gz files for detailed analysis.

        Args:
            save: Whether to save per-sample results

        Returns:
            New RunConfig instance with save_samples flag set
        """
        return replace(self, save_samples=save)

    def get_model_config(self, model_key: str) -> dict[str, Any]:
        """Get configuration for a specific model.

        Args:
            model_key: Model key to retrieve config for

        Returns:
            Model configuration dictionary

        Raises:
            KeyError: If model_key not found in models_config
        """
        if model_key not in self.models_config:
            raise KeyError(f"Model '{model_key}' not found in configuration")
        return self.models_config[model_key]

    def get_tasks_for_model(self, model_key: str) -> list[tuple[str, str]]:
        """Get list of tasks supported by a specific model.

        Args:
            model_key: Model key to get tasks for

        Returns:
            List of (task_name, task_type) tuples that the model supports

        Raises:
            KeyError: If model_key not found in models_config
        """
        model_config = self.get_model_config(model_key)
        supported_types = set(model_config.get("supports", []))

        # Filter tasks to only those supported by this model
        return [(name, type_) for name, type_ in self.tasks if type_ in supported_types]

    def has_dense_models(self) -> bool:
        """Check if any dense (neural) models are in the run.

        Returns:
            True if any models have type 'sentence_transformer'
        """
        return any(
            self.models_config[model].get("type") == "sentence_transformer"
            for model in self.models
        )

    def has_sparse_models(self) -> bool:
        """Check if any sparse models are in the run.

        Sparse models include: tf, tfidf, bm25

        Returns:
            True if any models have sparse type
        """
        sparse_types = {"tf", "tfidf", "bm25"}
        return any(
            self.models_config[model].get("type") in sparse_types
            for model in self.models
        )

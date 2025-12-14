"""Clustering evaluator for SHELF tasks.

Evaluates embedding models on clustering tasks by running k-means
on embeddings and comparing against ground truth labels.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import polars as pl
from sklearn.cluster import KMeans

from shelf.evaluate.evaluators.base import TaskEvaluator
from shelf.evaluate.metrics.clustering import compute_clustering_metrics
from shelf.evaluate.results import EvaluationResult
from shelf.evaluate.schemas import (
    ValidationError,
    validate_clustering_predictions,
)
from shelf.evaluate.tasks import TaskSpec
from shelf.taxonomies.geographic import get_region_from_list

if TYPE_CHECKING:
    from shelf.evaluate.adapters.protocols import TextEmbedder

logger = logging.getLogger(__name__)


class ClusteringEvaluator(TaskEvaluator):
    """Evaluator for clustering tasks.

    Supports two modes:
    1. From predictions file: Pre-computed cluster assignments
    2. From embedder: Run k-means on embeddings

    For clustering evaluation:
    - Ground truth comes from the label_field (e.g., lcc_code)
    - Predictions are cluster assignments (integers)
    - Metrics compare cluster assignments against ground truth labels

    Example:
        from shelf.evaluate.evaluators import ClusteringEvaluator
        from shelf.evaluate.registry import get_task

        task_spec = get_task("lcc_clustering")
        evaluator = ClusteringEvaluator(task_spec)

        # From embedder (runs k-means internally)
        result = evaluator.evaluate_embedder(embedder, split="test")
        print(result.summary())
    """

    def __init__(
        self,
        task_spec: TaskSpec,
        n_clusters: int | None = None,
        random_seed: int = 42,
    ):
        """Initialize clustering evaluator.

        Args:
            task_spec: Task specification
            n_clusters: Number of clusters for k-means. If None, uses
                        the number of unique labels in the label space.
            random_seed: Random seed for reproducibility
        """
        super().__init__(task_spec, random_seed)

        # Determine number of clusters
        if n_clusters is not None:
            self.n_clusters = n_clusters
        elif task_spec.label_space:
            self.n_clusters = len(task_spec.label_space)
        else:
            self.n_clusters = None  # Will be inferred from data

    def _load_ground_truth(self, split: str) -> pl.DataFrame:
        """Load ground truth with special handling for geographic_clustering.

        For the geographic_clustering task, this method:
        1. Loads the data with the 'geographic' column (list of place names)
        2. Maps each document's geographic list to a region using get_region_from_list()
        3. Filters out documents without a valid region
        4. Adds the 'geographic_region' column expected by the task

        Args:
            split: Dataset split to load

        Returns:
            Polars DataFrame with ground truth data
        """
        df = super()._load_ground_truth(split)

        # Special preprocessing for geographic_clustering task
        if self.task_spec.name == "geographic_clustering":
            label_field = self.task_spec.label_field  # "geographic_region"

            # Check if we need to preprocess
            if label_field not in df.columns and "geographic" in df.columns:
                logger.info("Preprocessing geographic data for clustering...")

                # Map geographic lists to regions
                def map_to_region(geo_list: list[str] | None) -> str | None:
                    if geo_list is None or len(geo_list) == 0:
                        return None
                    return get_region_from_list(geo_list)

                # Add geographic_region column
                df = df.with_columns(
                    pl.col("geographic")
                    .map_elements(map_to_region, return_dtype=pl.Utf8)
                    .alias(label_field)
                )

                # Filter out documents without a valid region
                original_count = len(df)
                df = df.filter(pl.col(label_field).is_not_null())
                filtered_count = len(df)

                logger.info(
                    f"Geographic filtering: {original_count} -> {filtered_count} docs "
                    f"({filtered_count / original_count * 100:.1f}% have valid regions)"
                )

        return df

    def evaluate(
        self,
        predictions: list[dict[str, Any]],
        ground_truth: pl.DataFrame,
        compute_ci: bool = False,
    ) -> EvaluationResult:
        """Evaluate clustering predictions.

        Args:
            predictions: List of {"id": str, "cluster": int}
            ground_truth: DataFrame with ground truth labels
            compute_ci: Whether to compute confidence intervals (not yet implemented)

        Returns:
            EvaluationResult with clustering metrics
        """
        id_field = self.task_spec.id_field
        label_field = self.task_spec.label_field

        # Validate predictions against ground truth
        validation = validate_clustering_predictions(
            predictions=predictions,
            ground_truth_ids=set(ground_truth[id_field].to_list()),
            expected_clusters=self.n_clusters,
        )
        if not validation.valid:
            raise ValidationError(validation.errors)

        # Build prediction dict: id -> cluster
        pred_dict: dict[str, int] = {}
        for pred in predictions:
            doc_id = pred["id"]
            cluster = pred["cluster"]
            pred_dict[doc_id] = cluster

        labels_true: list[str] = []
        labels_pred: list[int] = []

        for row in ground_truth.iter_rows(named=True):
            doc_id = row[id_field]
            true_label = row[label_field]

            if doc_id not in pred_dict:
                logger.warning(f"Missing prediction for document: {doc_id}")
                continue

            labels_true.append(true_label)
            labels_pred.append(pred_dict[doc_id])

        if not labels_true:
            raise ValueError("No valid predictions found matching ground truth IDs")

        # Compute metrics
        metrics = compute_clustering_metrics(labels_true, labels_pred)

        return self._create_result(
            metrics=metrics,
            ground_truth=ground_truth,
            split=self.task_spec.default_split,
        )

    def evaluate_embedder(
        self,
        embedder: "TextEmbedder",
        split: str | None = None,
        n_clusters: int | None = None,
        n_init: int = 10,
        max_iter: int = 300,
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> EvaluationResult:
        """Evaluate an embedder by running k-means on embeddings.

        This method:
        1. Loads documents from the specified split
        2. Encodes all documents with the embedder
        3. Runs k-means clustering on embeddings
        4. Evaluates cluster assignments against ground truth labels

        Args:
            embedder: TextEmbedder instance
            split: Dataset split (default: task default, usually "test")
            n_clusters: Number of clusters (default: from task spec or data)
            n_init: Number of k-means initializations
            max_iter: Maximum iterations for k-means
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bars

        Returns:
            EvaluationResult with clustering metrics
        """
        split = split or self.task_spec.default_split

        logger.info(f"Loading data from split: {split}")

        # Load ground truth
        ground_truth = self._load_ground_truth(split)

        logger.info(f"Documents to cluster: {len(ground_truth)}")

        # Get field names
        text_field = self.task_spec.text_field
        label_field = self.task_spec.label_field

        # Extract texts and labels
        texts = ground_truth[text_field].to_list()
        labels_true = ground_truth[label_field].to_list()

        # Determine number of clusters
        k = n_clusters or self.n_clusters
        if k is None:
            k = len(set(labels_true))
            logger.info(f"Inferred n_clusters={k} from data")

        # Encode documents
        logger.info("Encoding documents...")
        embeddings = embedder.encode(
            texts,
            batch_size=batch_size,
            show_progress=show_progress,
        )

        # Run k-means
        logger.info(f"Running k-means with k={k}...")
        kmeans = KMeans(
            n_clusters=k,
            n_init=n_init,
            max_iter=max_iter,
            random_state=self.random_seed,
        )
        labels_pred = kmeans.fit_predict(embeddings).tolist()

        # Compute metrics
        metrics = compute_clustering_metrics(labels_true, labels_pred)

        # Get inertia, handling potential None
        inertia = kmeans.inertia_
        kmeans_inertia = float(inertia) if inertia is not None else 0.0

        return self._create_result(
            metrics=metrics,
            ground_truth=ground_truth,
            split=split,
            model_name=embedder.model_name,
            embedding_dim=embedder.embedding_dim,
            n_clusters=k,
            kmeans_inertia=kmeans_inertia,
        )

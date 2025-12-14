"""HDBSCAN clustering evaluator for SHELF discovery tasks.

This evaluator uses HDBSCAN (Hierarchical Density-Based Spatial Clustering
of Applications with Noise) for cluster discovery without requiring
a predetermined number of clusters.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import polars as pl
from sklearn.cluster import HDBSCAN

from shelf.evaluate.evaluators.base import TaskEvaluator
from shelf.evaluate.metrics.clustering import compute_discovery_metrics
from shelf.evaluate.results import (
    EvaluationResult,
    PerSampleResult,
    PerSampleResults,
)
from shelf.evaluate.tasks import TaskSpec
from shelf.evaluate.utils.normalization import ensure_normalized
from shelf.taxonomies.geographic import get_region_from_list

if TYPE_CHECKING:
    from shelf.evaluate.adapters.protocols import TextEmbedder

logger = logging.getLogger(__name__)

# Metadata fields to capture for stratification analysis
STRATIFICATION_FIELDS = [
    "form",
    "form_category",
    "register",
    "audience",
    "lcc",
    "topic",
    "region",
]


class HDBSCANClusteringEvaluator(TaskEvaluator):
    """Evaluator for cluster discovery using HDBSCAN.

    Unlike k-means clustering, HDBSCAN does not require specifying
    the number of clusters. It discovers clusters based on density
    and can identify noise points that don't belong to any cluster.

    Key hyperparameters:
    - min_cluster_size: Minimum number of samples in a cluster
    - min_samples: Number of samples in neighborhood for core points
    - cluster_selection_method: 'eom' (excess of mass) or 'leaf'

    Example:
        from shelf.evaluate.evaluators import HDBSCANClusteringEvaluator
        from shelf.evaluate.registry import get_task

        task_spec = get_task("lcc_clustering_hdbscan")
        evaluator = HDBSCANClusteringEvaluator(task_spec)
        result = evaluator.evaluate_embedder(embedder, split="test")
        print(f"Discovered {result.metrics['num_clusters_pred']} clusters")
        print(f"Noise ratio: {result.metrics['noise_ratio']:.2%}")
    """

    def __init__(
        self,
        task_spec: TaskSpec,
        min_cluster_size: int = 50,
        min_samples: int | None = None,
        cluster_selection_method: str = "eom",
        random_seed: int = 42,
    ):
        """Initialize HDBSCAN clustering evaluator.

        Args:
            task_spec: Task specification
            min_cluster_size: Minimum cluster size (default: 50, ~1% of test set)
            min_samples: Samples in neighborhood for core points (default: None = min_cluster_size)
            cluster_selection_method: 'eom' or 'leaf' (default: 'eom')
            random_seed: Random seed for reproducibility
        """
        super().__init__(task_spec, random_seed)
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_method = cluster_selection_method

    def _load_ground_truth(self, split: str) -> pl.DataFrame:
        """Load ground truth with special handling for geographic_clustering.

        Identical to ClusteringEvaluator's implementation for consistency.
        """
        df = super()._load_ground_truth(split)

        # Special preprocessing for geographic_clustering task
        if "geographic" in self.task_spec.name:
            label_field = self.task_spec.label_field

            if label_field not in df.columns and "geographic" in df.columns:
                logger.info("Preprocessing geographic data for clustering...")

                def map_to_region(geo_list: list[str] | None) -> str | None:
                    if geo_list is None or len(geo_list) == 0:
                        return None
                    return get_region_from_list(geo_list)

                df = df.with_columns(
                    pl.col("geographic")
                    .map_elements(map_to_region, return_dtype=pl.Utf8)
                    .alias(label_field)
                )

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
        """Evaluate pre-computed HDBSCAN cluster assignments.

        Args:
            predictions: List of {"id": str, "cluster": int} (-1 for noise)
            ground_truth: DataFrame with ground truth labels
            compute_ci: Whether to compute confidence intervals (not implemented)

        Returns:
            EvaluationResult with discovery metrics
        """
        id_field = self.task_spec.id_field
        label_field = self.task_spec.label_field

        # Build prediction dict
        pred_dict: dict[str, int] = {p["id"]: p["cluster"] for p in predictions}

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
            raise ValueError("No valid predictions found")

        metrics = compute_discovery_metrics(labels_true, labels_pred)

        return self._create_result(
            metrics=metrics,
            ground_truth=ground_truth,
            split=self.task_spec.default_split,
        )

    def evaluate_embedder(
        self,
        embedder: "TextEmbedder",
        split: str | None = None,
        min_cluster_size: int | None = None,
        min_samples: int | None = None,
        cluster_selection_method: str | None = None,
        batch_size: int = 32,
        show_progress: bool = True,
        save_samples: bool = False,
    ) -> EvaluationResult:
        """Evaluate an embedder using HDBSCAN clustering.

        Args:
            embedder: TextEmbedder instance
            split: Dataset split (default: task default)
            min_cluster_size: Override minimum cluster size
            min_samples: Override min samples for core points
            cluster_selection_method: Override selection method ('eom' or 'leaf')
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bars
            save_samples: Whether to capture per-sample results for detailed analysis

        Returns:
            EvaluationResult with discovery metrics
        """
        split = split or self.task_spec.default_split

        # Use instance defaults if not overridden
        min_cluster_size = min_cluster_size or self.min_cluster_size
        min_samples = min_samples or self.min_samples
        cluster_selection_method = (
            cluster_selection_method or self.cluster_selection_method
        )

        logger.info(f"Loading data from split: {split}")
        ground_truth = self._load_ground_truth(split)
        logger.info(f"Documents to cluster: {len(ground_truth)}")

        text_field = self.task_spec.text_field
        label_field = self.task_spec.label_field
        id_field = self.task_spec.id_field

        texts = ground_truth[text_field].to_list()
        labels_true = ground_truth[label_field].to_list()
        doc_ids = ground_truth[id_field].to_list()

        # Encode documents
        logger.info("Encoding documents...")
        embeddings = embedder.encode(
            texts,
            batch_size=batch_size,
            show_progress=show_progress,
        )

        # Ensure normalized for consistent behavior
        embeddings = ensure_normalized(embeddings)

        # Run HDBSCAN
        logger.info(
            f"Running HDBSCAN (min_cluster_size={min_cluster_size}, "
            f"min_samples={min_samples}, method={cluster_selection_method})..."
        )

        hdbscan = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_method=cluster_selection_method,
            metric="euclidean",  # On normalized vectors, this ≈ cosine
            n_jobs=-1,  # Use all cores
        )
        labels_pred = hdbscan.fit_predict(embeddings).tolist()

        # Compute metrics
        metrics = compute_discovery_metrics(labels_true, labels_pred)

        # Log summary
        logger.info(
            f"HDBSCAN found {metrics['num_clusters_pred']} clusters, "
            f"{metrics['noise_ratio']:.1%} noise, "
            f"k_error={metrics['cluster_k_error']:.2f}"
        )

        # Build per-sample results if requested
        per_sample_list: list[PerSampleResult] = []
        if save_samples:
            # Get available columns for metadata extraction
            available_cols = set(ground_truth.columns)
            metadata_fields = [f for f in STRATIFICATION_FIELDS if f in available_cols]

            for i, (doc_id, true_label, pred_cluster) in enumerate(
                zip(doc_ids, labels_true, labels_pred)
            ):
                # Get row from ground_truth for metadata
                row = ground_truth.row(i, named=True)
                metadata: dict[str, Any] = {}

                for field in metadata_fields:
                    if field in row and row[field] is not None:
                        metadata[field] = row[field]

                # Add text length bucket
                if text_field in row and row[text_field]:
                    text_len = len(row[text_field])
                    if text_len < 500:
                        metadata["length_bucket"] = "short"
                    elif text_len < 2000:
                        metadata["length_bucket"] = "medium"
                    else:
                        metadata["length_bucket"] = "long"
                    metadata["text_length"] = text_len

                # Mark noise points (-1 cluster)
                is_noise = pred_cluster == -1
                metadata["is_noise"] = is_noise

                per_sample_list.append(
                    PerSampleResult(
                        id=doc_id,
                        y_true=true_label,
                        y_pred=pred_cluster,
                        correct=None,  # clustering doesn't have correct/incorrect
                        metadata=metadata,
                    )
                )

        result = self._create_result(
            metrics=metrics,
            ground_truth=ground_truth,
            split=split,
            model_name=embedder.model_name,
            embedding_dim=embedder.embedding_dim,
            hdbscan_params={
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples,
                "cluster_selection_method": cluster_selection_method,
            },
        )

        # Attach per-sample results if captured
        if save_samples and per_sample_list:
            result.per_sample_results = PerSampleResults(
                task=self.task_spec.name,
                task_type="clustering",
                model_key=embedder.model_name or "unknown",
                split=split,
                samples=per_sample_list,
            )

        return result

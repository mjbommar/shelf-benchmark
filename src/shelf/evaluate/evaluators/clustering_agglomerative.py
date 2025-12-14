"""Agglomerative clustering evaluator with cosine distance for SHELF.

This evaluator uses hierarchical agglomerative clustering with cosine
distance, which directly measures angular similarity between embeddings
rather than relying on the Euclidean-cosine equivalence on normalized vectors.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

import polars as pl
from sklearn.cluster import AgglomerativeClustering

from shelf.evaluate.evaluators.base import TaskEvaluator
from shelf.evaluate.metrics.clustering import compute_clustering_metrics
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

# Type alias for linkage methods compatible with cosine distance
LinkageMethod = Literal["average", "complete", "single"]


class AgglomerativeClusteringEvaluator(TaskEvaluator):
    """Evaluator for clustering using agglomerative hierarchical clustering.

    This evaluator uses scikit-learn's AgglomerativeClustering with cosine
    distance. Unlike k-means which uses centroid-based optimization,
    agglomerative clustering builds a hierarchy by progressively merging
    clusters based on linkage criteria.

    Supported linkage methods for cosine distance:
    - 'average': UPGMA - uses average distance between all pairs
    - 'complete': Uses maximum distance between clusters
    - 'single': Uses minimum distance between clusters

    Note: Ward linkage is NOT supported with cosine distance.

    Example:
        from shelf.evaluate.evaluators import AgglomerativeClusteringEvaluator
        from shelf.evaluate.registry import get_task

        task_spec = get_task("lcc_clustering_agglomerative")
        evaluator = AgglomerativeClusteringEvaluator(task_spec, linkage="average")
        result = evaluator.evaluate_embedder(embedder, split="test")
    """

    def __init__(
        self,
        task_spec: TaskSpec,
        n_clusters: int | None = None,
        linkage: LinkageMethod = "average",
        random_seed: int = 42,
    ):
        """Initialize agglomerative clustering evaluator.

        Args:
            task_spec: Task specification
            n_clusters: Number of clusters. If None, uses label_space length.
            linkage: Linkage criterion ('average', 'complete', 'single')
            random_seed: Random seed for reproducibility (not used by algorithm
                        but kept for API consistency)
        """
        super().__init__(task_spec, random_seed)

        # Validate linkage
        valid_linkages: tuple[LinkageMethod, ...] = ("average", "complete", "single")
        if linkage not in valid_linkages:
            raise ValueError(
                f"Invalid linkage '{linkage}' for cosine distance. "
                f"Valid options: {valid_linkages}. Note: 'ward' requires Euclidean distance."
            )
        self.linkage = linkage

        # Determine number of clusters
        if n_clusters is not None:
            self.n_clusters = n_clusters
        elif task_spec.label_space:
            self.n_clusters = len(task_spec.label_space)
        else:
            self.n_clusters = None

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
        save_samples: bool = False,
        model_key: str | None = None,
    ) -> EvaluationResult:
        """Evaluate pre-computed cluster assignments.

        Args:
            predictions: List of {"id": str, "cluster": int}
            ground_truth: DataFrame with ground truth labels
            compute_ci: Whether to compute confidence intervals (not implemented)
            save_samples: Whether to capture per-sample results for detailed analysis
            model_key: Model identifier for per-sample results

        Returns:
            EvaluationResult with clustering metrics
        """
        id_field = self.task_spec.id_field
        label_field = self.task_spec.label_field

        # Build prediction dict
        pred_dict: dict[str, int] = {p["id"]: p["cluster"] for p in predictions}

        labels_true: list[str] = []
        labels_pred: list[int] = []
        per_sample_list: list[PerSampleResult] = []

        # Get available columns for metadata extraction
        available_cols = set(ground_truth.columns)
        metadata_fields = [f for f in STRATIFICATION_FIELDS if f in available_cols]

        for row in ground_truth.iter_rows(named=True):
            doc_id = row[id_field]
            true_label = row[label_field]

            if doc_id not in pred_dict:
                logger.warning(f"Missing prediction for document: {doc_id}")
                continue

            pred_cluster = pred_dict[doc_id]

            labels_true.append(true_label)
            labels_pred.append(pred_cluster)

            # Capture per-sample result with metadata for stratification
            if save_samples:
                metadata: dict[str, Any] = {}
                for field in metadata_fields:
                    if field in row and row[field] is not None:
                        metadata[field] = row[field]

                # Add text length bucket for length analysis
                text_field = self.task_spec.text_field
                if text_field in row and row[text_field]:
                    text_len = len(row[text_field])
                    # Bucket into ranges: short (<500), medium (500-2000), long (>2000)
                    if text_len < 500:
                        metadata["length_bucket"] = "short"
                    elif text_len < 2000:
                        metadata["length_bucket"] = "medium"
                    else:
                        metadata["length_bucket"] = "long"
                    metadata["text_length"] = text_len

                per_sample_list.append(
                    PerSampleResult(
                        id=doc_id,
                        y_true=true_label,
                        y_pred=pred_cluster,
                        correct=None,  # clustering doesn't have correct/incorrect
                        metadata=metadata,
                    )
                )

        if not labels_true:
            raise ValueError("No valid predictions found")

        metrics = compute_clustering_metrics(labels_true, labels_pred)

        result = self._create_result(
            metrics=metrics,
            ground_truth=ground_truth,
            split=self.task_spec.default_split,
        )

        # Attach per-sample results if captured
        if save_samples and per_sample_list:
            result.per_sample_results = PerSampleResults(
                task=self.task_spec.name,
                task_type="clustering",
                model_key=model_key or "unknown",
                split=self.task_spec.default_split,
                samples=per_sample_list,
            )

        return result

    def evaluate_embedder(
        self,
        embedder: "TextEmbedder",
        split: str | None = None,
        n_clusters: int | None = None,
        linkage: LinkageMethod | None = None,
        batch_size: int = 32,
        show_progress: bool = True,
        save_samples: bool = False,
    ) -> EvaluationResult:
        """Evaluate an embedder using agglomerative clustering with cosine distance.

        Args:
            embedder: TextEmbedder instance
            split: Dataset split (default: task default)
            n_clusters: Override number of clusters
            linkage: Override linkage method
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bars
            save_samples: Whether to capture per-sample results for detailed analysis

        Returns:
            EvaluationResult with clustering metrics
        """
        split = split or self.task_spec.default_split
        linkage = linkage or self.linkage

        logger.info(f"Loading data from split: {split}")
        ground_truth = self._load_ground_truth(split)
        logger.info(f"Documents to cluster: {len(ground_truth)}")

        text_field = self.task_spec.text_field
        label_field = self.task_spec.label_field
        id_field = self.task_spec.id_field

        texts = ground_truth[text_field].to_list()
        labels_true = ground_truth[label_field].to_list()
        doc_ids = ground_truth[id_field].to_list()

        # Determine k
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

        # Normalize embeddings (required for meaningful cosine distance)
        embeddings = ensure_normalized(embeddings)

        # Check memory requirements
        n_samples = len(embeddings)
        estimated_memory_mb = (n_samples * n_samples * 8) / (1024 * 1024)  # float64
        if estimated_memory_mb > 1000:
            logger.warning(
                f"Agglomerative clustering requires O(n²) memory. "
                f"Estimated: {estimated_memory_mb:.0f} MB for {n_samples} samples."
            )

        # Run agglomerative clustering
        logger.info(
            f"Running agglomerative clustering (k={k}, linkage={linkage}, metric=cosine)..."
        )

        agg = AgglomerativeClustering(
            n_clusters=k,
            metric="cosine",
            linkage=linkage,
        )
        labels_pred = agg.fit_predict(embeddings).tolist()

        # Compute metrics
        metrics = compute_clustering_metrics(labels_true, labels_pred)
        metrics["linkage"] = linkage

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
            n_clusters=k,
            agglomerative_params={
                "linkage": linkage,
                "metric": "cosine",
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

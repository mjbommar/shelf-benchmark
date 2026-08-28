"""Subject-conditional clustering evaluator for SHELF.

The flat clustering evaluators (:mod:`shelf.evaluate.evaluators.clustering`
and friends) run k-means over the *whole* corpus. For a low-variance
attribute like writing register or geography that is measured
``docs/data_plan_v0.4.md`` section 11.7: flat clustering asks the embedding
space to isolate register or geography while LCC subject -- which dominates
embedding variance -- is left free to vary, so subject differences swamp the
signal the task is trying to measure.

This evaluator instead clusters *within* each LCC class separately (subject
held constant) and aggregates the per-class results, so the question becomes
"among documents that are already about the same subject, does the embedding
separate them by register/geography?" rather than "does the embedding
separate documents by register/geography at all, regardless of subject?".

Two aggregates are reported, deliberately not collapsed into one:

- ``*_macro``: the unweighted mean of the per-class metric. Every LCC class
  counts equally regardless of size, so this answers "does conditioning help
  in a typical class?".
- ``*_pooled``: all per-class assignments concatenated into one clustering
  problem, where a document's predicted cluster is scoped to its class (so a
  cluster in class ``A`` is never treated as the same cluster as one in class
  ``B``). This is closer to "does conditioning help overall", weighted by
  class size, but by construction it cannot reward a register/region that
  happens to be shared *across* classes, since cross-class matches are never
  in the same predicted cluster.

Classes with too few samples or with fewer than two distinct ground-truth
labels present cannot be meaningfully clustered (k-means needs at least 2
clusters, and V-measure is undefined/trivial for a single true class) and are
skipped, with the reason recorded so a silent drop is never mistaken for a
strong score.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import polars as pl
from sklearn.cluster import KMeans

from shelf.evaluate.evaluators.base import TaskEvaluator
from shelf.evaluate.metrics.clustering import compute_clustering_metrics
from shelf.evaluate.results import EvaluationResult
from shelf.evaluate.tasks import TaskSpec
from shelf.evaluate.utils.normalization import ensure_normalized, get_norm_stats
from shelf.taxonomies.geographic import GeographicLabelPolicy, get_region_with_policy

if TYPE_CHECKING:
    from shelf.evaluate.adapters.protocols import TextEmbedder

logger = logging.getLogger(__name__)

# Metrics averaged for the macro aggregate and recomputed for the pooled one.
_AGGREGATED_METRIC_KEYS = ("v_measure", "nmi", "ari", "homogeneity", "completeness")

DEFAULT_MIN_CLASS_SIZE = 20


class SubjectConditionalClusteringEvaluator(TaskEvaluator):
    """Cluster documents within each LCC class and aggregate.

    Holds LCC subject constant (the dominant source of embedding variance)
    so that a weaker attribute named by ``task_spec.label_field`` -- register,
    geographic region, etc. -- can be measured without subject differences
    dominating the k-means solution.

    Example:
        from shelf.evaluate.evaluators.clustering_conditional import (
            SubjectConditionalClusteringEvaluator,
        )
        from shelf.evaluate.registry import get_task

        task_spec = get_task("register_clustering")  # any clustering TaskSpec
        evaluator = SubjectConditionalClusteringEvaluator(task_spec)
        result = evaluator.evaluate_embedder(embedder, split="test")
        print(result.metrics["v_measure_macro"], result.metrics["v_measure_pooled"])
    """

    def __init__(
        self,
        task_spec: TaskSpec,
        class_field: str = "lcc_code",
        min_class_size: int = DEFAULT_MIN_CLASS_SIZE,
        random_seed: int = 42,
        geographic_policy: GeographicLabelPolicy = GeographicLabelPolicy.FIRST,
    ):
        """Initialize the subject-conditional clustering evaluator.

        Args:
            task_spec: Task specification. ``label_field`` names the
                attribute clustered *within* each class (e.g. ``"register"``
                or ``"geographic_region"``); ``text_field``/``id_field`` are
                used as in the other clustering evaluators.
            class_field: Column that defines the conditioning classes
                (default: ``"lcc_code"``, SHELF's primary subject label).
            min_class_size: Minimum number of documents a class must have to
                be clustered. Classes below this are skipped rather than
                clustered unreliably.
            random_seed: Random seed for reproducibility.
            geographic_policy: Region resolution policy used only when
                ``task_spec.label_field`` requires the ``geographic`` ->
                region preprocessing (task name containing ``"geographic"``).
                Defaults to :attr:`GeographicLabelPolicy.FIRST`, matching the
                historical default elsewhere in the codebase.
        """
        super().__init__(task_spec, random_seed)
        self.class_field = class_field
        self.min_class_size = min_class_size
        self.geographic_policy = geographic_policy

    def _load_ground_truth(self, split: str) -> pl.DataFrame:
        """Load ground truth, adding ``geographic_region`` when needed.

        Mirrors :class:`~shelf.evaluate.evaluators.clustering.ClusteringEvaluator`,
        but resolves the region under ``self.geographic_policy`` instead of
        always taking the first tag.
        """
        df = super()._load_ground_truth(split)

        if "geographic" in self.task_spec.name:
            label_field = self.task_spec.label_field

            if label_field not in df.columns and "geographic" in df.columns:
                logger.info(
                    "Preprocessing geographic data for conditional clustering "
                    f"(policy={self.geographic_policy.value})..."
                )

                policy = self.geographic_policy

                def map_to_region(geo_list: list[str] | None) -> str | None:
                    if geo_list is None or len(geo_list) == 0:
                        return None
                    region = get_region_with_policy(geo_list, policy)
                    # ALL_REGIONS returns a frozenset; conditional clustering
                    # needs a single label per document, so it is not a valid
                    # choice here and callers should use FIRST or
                    # UNAMBIGUOUS_ONLY instead.
                    if isinstance(region, frozenset):
                        raise ValueError(
                            "GeographicLabelPolicy.ALL_REGIONS is not supported by "
                            "SubjectConditionalClusteringEvaluator, which needs a "
                            "single label per document. Use FIRST or "
                            "UNAMBIGUOUS_ONLY instead."
                        )
                    return region

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

    def _aggregate(
        self,
        per_class_metrics: dict[str, dict[str, Any]],
        pooled_true: list[str],
        pooled_pred: list[str],
        num_classes_total: int,
        skipped: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Combine per-class metrics into macro and pooled aggregates."""
        aggregated: dict[str, Any] = {}

        macro_accum: dict[str, list[float]] = defaultdict(list)
        for class_metrics in per_class_metrics.values():
            for key in _AGGREGATED_METRIC_KEYS:
                macro_accum[key].append(float(class_metrics[key]))

        for key in _AGGREGATED_METRIC_KEYS:
            values = macro_accum[key]
            aggregated[f"{key}_macro"] = float(np.mean(values)) if values else 0.0

        if pooled_true:
            # Pooled predicted labels are "{class}::{cluster_id}" strings, not
            # raw cluster ints, so cross-class cluster ids are never conflated
            # (see module docstring). compute_clustering_metrics accepts any
            # hashable label via sklearn's LabelEncoder under the hood.
            pooled_metrics = compute_clustering_metrics(
                pooled_true, cast("list[int]", pooled_pred)
            )
            for key in _AGGREGATED_METRIC_KEYS:
                aggregated[f"{key}_pooled"] = pooled_metrics[key]
        else:
            for key in _AGGREGATED_METRIC_KEYS:
                aggregated[f"{key}_pooled"] = 0.0

        # Headline metric is the POOLED, CHANCE-CORRECTED figure.
        #
        # Conditional clustering makes each per-class problem smaller and
        # easier, so its raw scores are not comparable to a flat task's.
        # Measured against a shuffled-label control on the real test split
        # (MiniLM, geography, unambiguous labels only):
        #
        #   V-measure macro   real 0.2584   shuffled 0.0869   <- 34% structural
        #   V-measure pooled  real 0.1455   shuffled 0.0534   <- 37% structural
        #   ARI       macro   real 0.1120   shuffled -0.0015
        #   ARI       pooled  real 0.0067   shuffled -0.0001
        #
        # V-measure is not chance-corrected and inflates badly here; ARI is and
        # shuffles to ~0. Macro additionally flatters the score because small
        # classes pose easy problems and are weighted equally. Aliasing
        # `v_measure` to `v_measure_macro` would therefore report roughly 17x
        # the defensible effect. Every component metric is still emitted so a
        # caller can choose deliberately.
        aggregated["ari"] = aggregated["ari_pooled"]
        aggregated["v_measure"] = aggregated["v_measure_pooled"]
        aggregated["n_classes_total"] = num_classes_total
        aggregated["n_classes_clustered"] = len(per_class_metrics)
        aggregated["n_classes_skipped"] = len(skipped)
        aggregated["n_samples_clustered"] = len(pooled_true)

        return aggregated

    def evaluate(
        self,
        predictions: list[dict[str, Any]],
        ground_truth: pl.DataFrame,
        compute_ci: bool = False,
    ) -> EvaluationResult:
        """Evaluate pre-computed cluster assignments, scoped by class.

        Cluster ids in ``predictions`` are local to whatever process produced
        them (e.g. a separate k-means run per class); this method groups
        predictions by ``self.class_field`` (read from ``ground_truth``)
        before scoring, so a cluster id colliding across classes is not
        misread as a shared cluster. See the pooled-aggregate note in the
        module docstring.

        Args:
            predictions: List of ``{"id": str, "cluster": int}``.
            ground_truth: DataFrame with ground truth labels and the
                conditioning class column.
            compute_ci: Not implemented for this evaluator.

        Returns:
            EvaluationResult with per-class, macro, and pooled metrics.
        """
        id_field = self.task_spec.id_field
        label_field = self.task_spec.label_field
        class_field = self.class_field

        if class_field not in ground_truth.columns:
            raise ValueError(
                f"class_field '{class_field}' not found in ground truth columns: "
                f"{ground_truth.columns}"
            )

        pred_dict: dict[str, int] = {p["id"]: p["cluster"] for p in predictions}

        rows_by_class: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for row in ground_truth.iter_rows(named=True):
            doc_id = row[id_field]
            if doc_id not in pred_dict:
                logger.warning(f"Missing prediction for document: {doc_id}")
                continue
            rows_by_class[row[class_field]].append(
                (row[label_field], pred_dict[doc_id])
            )

        if not rows_by_class:
            raise ValueError("No valid predictions found matching ground truth IDs")

        per_class_metrics, skipped, pooled_true, pooled_pred = (
            self._score_pairs_by_class(rows_by_class)
        )

        if not per_class_metrics:
            raise ValueError(
                "No class had enough samples/distinct labels to cluster "
                f"(min_class_size={self.min_class_size}); skipped: {skipped}"
            )

        metrics = self._aggregate(
            per_class_metrics, pooled_true, pooled_pred, len(rows_by_class), skipped
        )

        return self._create_result(
            metrics=metrics,
            ground_truth=ground_truth,
            split=self.task_spec.default_split,
            per_class_metrics=per_class_metrics,
            class_field=class_field,
            min_class_size=self.min_class_size,
            skipped_classes=skipped,
        )

    def _score_pairs_by_class(
        self,
        rows_by_class: dict[str, list[tuple[str, int]]],
    ) -> tuple[
        dict[str, dict[str, Any]],
        dict[str, dict[str, Any]],
        list[str],
        list[str],
    ]:
        """Score already-clustered (true_label, cluster_id) pairs per class."""
        per_class_metrics: dict[str, dict[str, Any]] = {}
        skipped: dict[str, dict[str, Any]] = {}
        pooled_true: list[str] = []
        pooled_pred: list[str] = []

        for class_label, pairs in rows_by_class.items():
            true_labels = [t for t, _ in pairs]
            n_unique = len(set(true_labels))
            if len(pairs) < self.min_class_size or n_unique < 2:
                skipped[class_label] = {
                    "n_samples": len(pairs),
                    "n_unique_labels": n_unique,
                    "reason": (
                        "too_few_samples"
                        if len(pairs) < self.min_class_size
                        else "single_label"
                    ),
                }
                continue

            pred_clusters = [c for _, c in pairs]
            per_class_metrics[class_label] = compute_clustering_metrics(
                true_labels, pred_clusters
            )
            pooled_true.extend(true_labels)
            pooled_pred.extend(f"{class_label}::{c}" for c in pred_clusters)

        return per_class_metrics, skipped, pooled_true, pooled_pred

    def evaluate_embedder(
        self,
        embedder: TextEmbedder,
        split: str | None = None,
        n_init: int = 10,
        max_iter: int = 300,
        batch_size: int = 32,
        show_progress: bool = True,
        save_samples: bool = False,
    ) -> EvaluationResult:
        """Encode documents, cluster within each class, and aggregate.

        Args:
            embedder: TextEmbedder instance.
            split: Dataset split (default: task default, usually "test").
            n_init: Number of k-means initializations per class.
            max_iter: Maximum k-means iterations per class.
            batch_size: Batch size for encoding.
            show_progress: Whether to show progress bars during encoding.

        Returns:
            EvaluationResult with per-class metrics (``per_class_metrics``)
            plus macro/pooled aggregates and ``n_classes_skipped`` /
            ``n_classes_clustered`` counts in ``metrics``.

        Raises:
            ValueError: If no class in the data has at least
                ``self.min_class_size`` documents and 2+ distinct labels.
        """
        split = split or self.task_spec.default_split

        logger.info(f"Loading data from split: {split}")
        ground_truth = self._load_ground_truth(split)
        logger.info(f"Documents to cluster: {len(ground_truth)}")

        if self.class_field not in ground_truth.columns:
            raise ValueError(
                f"class_field '{self.class_field}' not found in ground truth "
                f"columns: {ground_truth.columns}"
            )

        text_field = self.task_spec.text_field
        label_field = self.task_spec.label_field

        texts = ground_truth[text_field].to_list()
        labels_true_all = ground_truth[label_field].to_list()
        classes_all = ground_truth[self.class_field].to_list()

        logger.info("Encoding documents...")
        embeddings = embedder.encode(
            texts, batch_size=batch_size, show_progress=show_progress
        )
        embeddings = ensure_normalized(embeddings)
        norm_stats = get_norm_stats(embeddings)

        indices_by_class: dict[str, list[int]] = defaultdict(list)
        for i, class_label in enumerate(classes_all):
            indices_by_class[class_label].append(i)

        per_class_metrics: dict[str, dict[str, Any]] = {}
        skipped: dict[str, dict[str, Any]] = {}
        pooled_true: list[str] = []
        pooled_pred: list[str] = []

        for class_label, idxs in indices_by_class.items():
            true_labels = [labels_true_all[i] for i in idxs]
            n_unique = len(set(true_labels))

            if len(idxs) < self.min_class_size or n_unique < 2:
                skipped[class_label] = {
                    "n_samples": len(idxs),
                    "n_unique_labels": n_unique,
                    "reason": (
                        "too_few_samples"
                        if len(idxs) < self.min_class_size
                        else "single_label"
                    ),
                }
                continue

            k = n_unique
            sub_embeddings = embeddings[idxs]
            kmeans = KMeans(
                n_clusters=k,
                n_init=n_init,
                max_iter=max_iter,
                random_state=self.random_seed,
            )
            pred_clusters = kmeans.fit_predict(sub_embeddings).tolist()

            per_class_metrics[class_label] = compute_clustering_metrics(
                true_labels, pred_clusters
            )
            pooled_true.extend(true_labels)
            pooled_pred.extend(f"{class_label}::{c}" for c in pred_clusters)

        if not per_class_metrics:
            raise ValueError(
                "No class had enough samples/distinct labels to cluster "
                f"(min_class_size={self.min_class_size}); skipped: {skipped}"
            )

        metrics = self._aggregate(
            per_class_metrics,
            pooled_true,
            pooled_pred,
            len(indices_by_class),
            skipped,
        )

        return self._create_result(
            metrics=metrics,
            ground_truth=ground_truth,
            split=split,
            per_class_metrics=per_class_metrics,
            model_name=embedder.model_name,
            embedding_dim=embedder.embedding_dim,
            class_field=self.class_field,
            min_class_size=self.min_class_size,
            skipped_classes=skipped,
            norm_stats=norm_stats,
        )

"""Unit tests for SubjectConditionalClusteringEvaluator.

The evaluator clusters within each LCC class separately (holding subject
constant) and aggregates macro/pooled metrics, per
``docs/data_plan_v0.4.md`` section 11.7. These tests cover:

- Scoring already-clustered predictions via ``evaluate()`` (no encoding
  needed), including macro/pooled aggregation and class skip logic.
- End-to-end clustering via ``evaluate_embedder()`` on a synthetic embedder
  with clean per-class-and-label structure, to confirm the k-means/aggregate
  wiring actually recovers signal when it is present.
- The geographic preprocessing path, including that it defaults to the
  historical ``FIRST`` policy and rejects ``ALL_REGIONS`` (which cannot
  produce the single label per document conditional clustering needs).
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import polars as pl
import pytest
from shelf.evaluate.evaluators.base import TaskEvaluator
from shelf.evaluate.evaluators.clustering_conditional import (
    SubjectConditionalClusteringEvaluator,
)
from shelf.evaluate.tasks import TaskSpec, TaskType
from shelf.taxonomies.geographic import GeographicLabelPolicy

# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def register_task_spec() -> TaskSpec:
    """A clustering TaskSpec conditioning register within LCC class."""
    return TaskSpec(
        name="register_clustering",
        task_type=TaskType.CLUSTERING,
        description="Test conditional register clustering",
        text_field="text",
        label_field="register",
        id_field="id",
        label_space=("academic", "casual", "technical"),
        primary_metric="v_measure_macro",
        secondary_metrics=("v_measure_pooled", "nmi_macro", "ari_macro"),
        dataset_name="test/dataset",
        dataset_config="default",
        default_split="test",
    )


@pytest.fixture
def geographic_task_spec() -> TaskSpec:
    """A clustering TaskSpec matching the real geographic_clustering task name."""
    return TaskSpec(
        name="geographic_clustering",
        task_type=TaskType.CLUSTERING,
        description="Test conditional geographic clustering",
        text_field="text",
        label_field="geographic_region",
        id_field="id",
        label_space=None,
        primary_metric="v_measure_macro",
        secondary_metrics=("v_measure_pooled",),
        dataset_name="test/dataset",
        dataset_config="default",
        default_split="test",
    )


def _two_class_ground_truth() -> pl.DataFrame:
    """Two LCC classes, each with 2 registers x 5 docs -- clean per-class structure."""
    rows = []
    doc_id = 0
    for lcc in ("A", "B"):
        for register in ("academic", "casual"):
            for _ in range(5):
                rows.append(
                    {
                        "id": f"doc_{doc_id:03d}",
                        "text": f"{lcc}-{register}-{doc_id}",
                        "lcc_code": lcc,
                        "register": register,
                    }
                )
                doc_id += 1
    return pl.DataFrame(rows)


class FakeEmbedder:
    """Deterministic embedder placing (class, label) pairs in separated blobs.

    Cluster centers are far apart across every (lcc_code, register) pair
    encoded into the text (``f"{lcc}-{register}-{i}"``), with small noise, so
    k-means trivially recovers the register split *within* each class -- this
    exercises the evaluator's per-class k-means + aggregation wiring end to
    end without depending on a real model.
    """

    model_name = "fake-embedder"
    embedding_dim = 2

    # Clustering L2-normalizes embeddings first (Euclidean k-means on unit
    # vectors == spherical k-means), so separation must be encoded as
    # *direction*, not magnitude: an offset purely along one axis collapses
    # to the same direction after normalization and would not separate.
    _DIRECTIONS = {"academic": np.array([1.0, 0.0]), "casual": np.array([0.0, 1.0])}

    def encode(
        self, texts: list[str], batch_size: int = 32, show_progress: bool = False
    ) -> np.ndarray:
        rng = np.random.default_rng(0)
        vectors = []
        for text in texts:
            _, register, _ = text.split("-", 2)
            noise = rng.normal(scale=0.02, size=2)
            vectors.append(self._DIRECTIONS[register] + noise)
        return np.asarray(vectors, dtype=np.float64)


# ===========================================================================
# evaluate(): pre-computed cluster assignments
# ===========================================================================


@pytest.mark.unit
@pytest.mark.evaluator
class TestEvaluatePredictions:
    def test_perfect_per_class_clusters_score_1(self, register_task_spec):
        ground_truth = _two_class_ground_truth()
        # Cluster id == 0 for academic, 1 for casual, within every class.
        predictions = [
            {"id": row["id"], "cluster": 0 if row["register"] == "academic" else 1}
            for row in ground_truth.iter_rows(named=True)
        ]

        evaluator = SubjectConditionalClusteringEvaluator(
            register_task_spec, min_class_size=1
        )
        result = evaluator.evaluate(predictions, ground_truth)

        assert result.metrics["v_measure_macro"] == pytest.approx(1.0)
        # Every (class, cluster) group is register-pure, so pooled homogeneity
        # is perfect too. Pooled completeness is *not* 1.0: "academic" is
        # split across the A::0 and B::0 groups, which is expected -- see
        # the collision test below for why that is the correct behavior.
        assert result.metrics["homogeneity_pooled"] == pytest.approx(1.0)
        assert result.metrics["completeness_pooled"] < 1.0
        assert result.metrics["n_classes_clustered"] == 2
        assert result.metrics["n_classes_skipped"] == 0
        assert result.per_class_metrics is not None
        assert set(result.per_class_metrics) == {"A", "B"}
        assert result.per_class_metrics["A"]["v_measure"] == pytest.approx(1.0)

    def test_cross_class_cluster_id_collision_does_not_inflate_pooled_score(
        self, register_task_spec
    ):
        """Cluster id 0 in class A and cluster id 0 in class B must not be
        silently treated as 'the same cluster' by the pooled aggregate.

        If raw cluster ids were pooled without class-scoping, this fixture
        would score a *perfect* v_measure of 1.0 (register happens to align
        with the raw id globally too) -- that would hide the fact that
        clustering was only ever done within each class separately. The
        evaluator's actual (class, cluster)-scoped pooling must score lower,
        because it correctly refuses to credit cross-class agreement that
        the per-class k-means never had a chance to produce.
        """
        from sklearn.metrics import v_measure_score

        ground_truth = _two_class_ground_truth()
        # Perfect within each class, but literally the same raw cluster ids
        # (0/1) are reused across classes A and B.
        predictions = [
            {"id": row["id"], "cluster": 0 if row["register"] == "academic" else 1}
            for row in ground_truth.iter_rows(named=True)
        ]

        naive_true = [row["register"] for row in ground_truth.iter_rows(named=True)]
        naive_pred = [p["cluster"] for p in predictions]
        naive_v_measure = v_measure_score(naive_true, naive_pred)
        assert naive_v_measure == pytest.approx(1.0)  # the naive/unscoped score

        evaluator = SubjectConditionalClusteringEvaluator(
            register_task_spec, min_class_size=1
        )
        result = evaluator.evaluate(predictions, ground_truth)

        assert result.metrics["v_measure_pooled"] < naive_v_measure

    def test_skips_class_below_min_size(self, register_task_spec):
        ground_truth = _two_class_ground_truth()
        predictions = [
            {"id": row["id"], "cluster": 0 if row["register"] == "academic" else 1}
            for row in ground_truth.iter_rows(named=True)
        ]

        # Each class has 10 docs; min_class_size=11 skips both.
        evaluator = SubjectConditionalClusteringEvaluator(
            register_task_spec, min_class_size=11
        )
        with pytest.raises(ValueError, match="No class had enough samples"):
            evaluator.evaluate(predictions, ground_truth)

    def test_skips_single_label_class(self, register_task_spec):
        ground_truth = _two_class_ground_truth().with_columns(
            pl.when(pl.col("lcc_code") == "B")
            .then(pl.lit("academic"))
            .otherwise(pl.col("register"))
            .alias("register")
        )
        predictions = [
            {"id": row["id"], "cluster": 0 if row["register"] == "academic" else 1}
            for row in ground_truth.iter_rows(named=True)
        ]

        evaluator = SubjectConditionalClusteringEvaluator(
            register_task_spec, min_class_size=1
        )
        result = evaluator.evaluate(predictions, ground_truth)

        assert result.metrics["n_classes_clustered"] == 1
        assert result.metrics["n_classes_skipped"] == 1
        assert result.context is not None
        skipped = result.context.extra["skipped_classes"]
        assert skipped["B"]["reason"] == "single_label"

    def test_missing_class_field_raises(self, register_task_spec):
        ground_truth = _two_class_ground_truth().drop("lcc_code")
        evaluator = SubjectConditionalClusteringEvaluator(register_task_spec)
        with pytest.raises(ValueError, match="class_field"):
            evaluator.evaluate([], ground_truth)

    def test_missing_predictions_for_some_ids_are_dropped_with_warning(
        self, register_task_spec, caplog
    ):
        ground_truth = _two_class_ground_truth()
        predictions = [
            {"id": row["id"], "cluster": 0 if row["register"] == "academic" else 1}
            for row in ground_truth.iter_rows(named=True)
            if row["id"] != "doc_000"
        ]
        evaluator = SubjectConditionalClusteringEvaluator(
            register_task_spec, min_class_size=1
        )
        with caplog.at_level("WARNING"):
            result = evaluator.evaluate(predictions, ground_truth)

        assert result.num_samples == len(ground_truth)  # ground truth size, unfiltered
        assert result.metrics["n_samples_clustered"] == len(ground_truth) - 1
        assert "Missing prediction for document: doc_000" in caplog.text


# ===========================================================================
# evaluate_embedder(): end-to-end k-means over a synthetic embedder
# ===========================================================================


@pytest.mark.unit
@pytest.mark.evaluator
class TestEvaluateEmbedder:
    def test_recovers_clean_synthetic_structure(
        self, register_task_spec, tmp_path, monkeypatch
    ):
        ground_truth = _two_class_ground_truth()
        monkeypatch.chdir(tmp_path)  # avoid picking up any real data/hf_dataset

        evaluator = SubjectConditionalClusteringEvaluator(
            register_task_spec, min_class_size=1, random_seed=0
        )
        with patch.object(
            TaskEvaluator, "_load_ground_truth", return_value=ground_truth
        ):
            result = evaluator.evaluate_embedder(
                FakeEmbedder(), split="test", show_progress=False
            )

        assert result.metrics["v_measure_macro"] == pytest.approx(1.0, abs=1e-6)
        assert result.metrics["ari_macro"] == pytest.approx(1.0, abs=1e-6)
        # Pooled homogeneity is perfect (every (class, cluster) group is
        # register-pure) but pooled completeness is necessarily < 1 since
        # "academic"/"casual" each span both classes -- see the
        # `evaluate()` collision test for why that is correct, not a bug.
        assert result.metrics["homogeneity_pooled"] == pytest.approx(1.0, abs=1e-6)
        assert result.metrics["n_classes_clustered"] == 2
        assert result.metrics["n_classes_skipped"] == 0
        assert result.metrics["n_samples_clustered"] == len(ground_truth)

    def test_missing_class_field_raises(self, register_task_spec):
        ground_truth = _two_class_ground_truth().drop("lcc_code")
        evaluator = SubjectConditionalClusteringEvaluator(register_task_spec)
        with (
            patch.object(
                TaskEvaluator, "_load_ground_truth", return_value=ground_truth
            ),
            pytest.raises(ValueError, match="class_field"),
        ):
            evaluator.evaluate_embedder(FakeEmbedder(), split="test")


# ===========================================================================
# Geographic preprocessing path
# ===========================================================================


@pytest.mark.unit
@pytest.mark.evaluator
class TestGeographicLoading:
    def _raw_geo_df(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "id": ["1", "2", "3"],
                "text": ["t1", "t2", "t3"],
                "lcc_code": ["A", "A", "B"],
                "geographic": [["Paris", "Brazil"], ["Tokyo"], ["Paris", "London"]],
            }
        )

    def test_default_policy_is_first(self, geographic_task_spec):
        evaluator = SubjectConditionalClusteringEvaluator(geographic_task_spec)
        with patch.object(
            TaskEvaluator, "_load_ground_truth", return_value=self._raw_geo_df()
        ):
            df = evaluator._load_ground_truth("test")

        # FIRST-tag policy: Paris/Brazil -> Europe (first tag), nothing dropped.
        assert df["geographic_region"].to_list() == ["Europe", "East Asia", "Europe"]
        assert len(df) == 3

    def test_unambiguous_only_drops_ambiguous_docs(self, geographic_task_spec):
        evaluator = SubjectConditionalClusteringEvaluator(
            geographic_task_spec,
            geographic_policy=GeographicLabelPolicy.UNAMBIGUOUS_ONLY,
        )
        with patch.object(
            TaskEvaluator, "_load_ground_truth", return_value=self._raw_geo_df()
        ):
            df = evaluator._load_ground_truth("test")

        # Doc 1 (Paris/Brazil) is ambiguous and must be dropped.
        assert df["id"].to_list() == ["2", "3"]
        assert df["geographic_region"].to_list() == ["East Asia", "Europe"]

    def test_all_regions_policy_raises(self, geographic_task_spec):
        evaluator = SubjectConditionalClusteringEvaluator(
            geographic_task_spec,
            geographic_policy=GeographicLabelPolicy.ALL_REGIONS,
        )
        with (
            patch.object(
                TaskEvaluator, "_load_ground_truth", return_value=self._raw_geo_df()
            ),
            pytest.raises(ValueError, match="ALL_REGIONS is not supported"),
        ):
            evaluator._load_ground_truth("test")

    def test_non_geographic_task_is_unaffected(self, register_task_spec):
        """Only tasks whose name contains 'geographic' get the preprocessing."""
        ground_truth = _two_class_ground_truth()
        evaluator = SubjectConditionalClusteringEvaluator(register_task_spec)
        with patch.object(
            TaskEvaluator, "_load_ground_truth", return_value=ground_truth
        ):
            df = evaluator._load_ground_truth("test")
        assert "geographic_region" not in df.columns

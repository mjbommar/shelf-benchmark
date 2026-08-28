"""Retrieval evaluator for SHELF tasks.

Evaluates embedding models on retrieval tasks like LCC retrieval,
form retrieval, and topic retrieval.

Two things happen here beyond plain label-match retrieval, both from
``docs/data_plan_v0.4.md`` sections 11.1/11.3/11.4.

**Graded relevance.** Judging a corpus document relevant iff it carries the
query's label makes ~5% of a 34k corpus relevant to every LCC query, and NDCG@10
stops discriminating. The LC taxonomies already encode an ordinal relation
ladder, so every (query, corpus document) pair is graded through
``shelf.evaluate.strata.classify_relation`` and reported as ``graded_ndcg@k``
*alongside* the binary metrics, which are left untouched so earlier runs stay
comparable.

**Instruction-conditioned relevance.** When the task carries an instruction (see
``shelf.evaluate.instructions``), relevance comes from the instruction rather
than from ``task_spec.label_field``, the query text is prefixed with the
instruction, and two constraint diagnostics are reported next to the IR metrics.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from shelf.evaluate.evaluators.base import TaskEvaluator
from shelf.evaluate.instructions import (
    InstructionJudge,
    InstructionSpec,
    get_instruction,
)
from shelf.evaluate.metrics.retrieval import (
    compute_graded_retrieval_metrics,
    compute_retrieval_metrics,
)
from shelf.evaluate.results import (
    EvaluationResult,
    PerSampleResult,
    PerSampleResults,
)
from shelf.evaluate.schemas import (
    ValidationError,
    validate_retrieval_predictions,
)
from shelf.evaluate.strata import (
    DocumentFacets,
    FormRelation,
    SubjectRelation,
    classify_relation,
)
from shelf.evaluate.tasks import TaskSpec

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

# Facet columns the graded judgments read. Absent columns degrade the scheme
# rather than disabling it, except for the axis the task is actually about.
_LCC_FIELD = "lcc_code"
_SUBCLASS_FIELD = "lcc_subclass"
_FORM_FIELD = "lcgft_form"
_CATEGORY_FIELD = "lcgft_category"
_TOPICS_FIELD = "topics"


@dataclass(frozen=True)
class GradedScheme:
    """Ordinal gain tiers for one retrieval axis.

    Gains are linear (0-3), not exponential: the whole point of grading is to
    give partial credit for a near miss, and ``2**rel - 1`` weights the top tier
    so heavily that graded NDCG collapses back onto binary NDCG.

    The three schemes differ in how much a cross-axis match is worth, because
    that depends on what was asked for. For a *form* query a mere category match
    is a weak consolation (gain 1); for a *category* query a form match is the
    category match plus more (gain 3 over 2).

    Attributes:
        axis: The label field this scheme grades.
        tiers: ``(gain, relation name)`` pairs, best first, for reporting.
        needs: Columns without which the scheme cannot be built.
    """

    axis: str
    tiers: tuple[tuple[float, str], ...]
    needs: tuple[str, ...]


#: Gain tiers per retrieval axis. Registered by `label_field`, not by task name,
#: so a new task on an existing axis is graded without further wiring.
GRADED_SCHEMES: dict[str, GradedScheme] = {
    _LCC_FIELD: GradedScheme(
        axis=_LCC_FIELD,
        tiers=(
            (3.0, "same_lcc_subclass"),
            (2.0, "same_lcc_class"),
            (1.0, "shared_topic_different_class"),
        ),
        needs=(_LCC_FIELD,),
    ),
    _FORM_FIELD: GradedScheme(
        axis=_FORM_FIELD,
        tiers=(
            (3.0, "same_lcgft_form"),
            (1.0, "same_lcgft_category"),
        ),
        needs=(_FORM_FIELD, _CATEGORY_FIELD),
    ),
    _CATEGORY_FIELD: GradedScheme(
        axis=_CATEGORY_FIELD,
        tiers=(
            (3.0, "same_lcgft_form"),
            (2.0, "same_lcgft_category"),
        ),
        needs=(_CATEGORY_FIELD, _FORM_FIELD),
    ),
}

if TYPE_CHECKING:
    from shelf.evaluate.adapters.bm25 import BM25Retriever
    from shelf.evaluate.adapters.protocols import TextEmbedder

logger = logging.getLogger(__name__)


def _merge_per_query(
    base: dict[str, dict[str, float]] | None,
    extra: dict[str, dict[str, float]] | None,
) -> dict[str, dict[str, float]] | None:
    """Fold extra per-query metrics into the standard per-query breakdown."""
    if not extra:
        return base
    merged: dict[str, dict[str, float]] = {
        query_id: dict(metrics) for query_id, metrics in (base or {}).items()
    }
    for query_id, metrics in extra.items():
        merged.setdefault(query_id, {}).update(metrics)
    return merged


def _as_topics(value: Any) -> tuple[str, ...]:
    """Coerce a topics cell to a tuple of strings."""
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, np.ndarray):
        value = value.tolist()
    try:
        return tuple(str(item) for item in value if item is not None)
    except TypeError:
        return (str(value),)


def _facets_from_row(row: Mapping[str, Any]) -> DocumentFacets:
    """Build the strata facet record for one document row."""
    subclass = row.get(_SUBCLASS_FIELD)
    return DocumentFacets(
        lcc_code=str(row.get(_LCC_FIELD) or ""),
        lcgft_form=str(row.get(_FORM_FIELD) or ""),
        lcgft_category=str(row.get(_CATEGORY_FIELD) or ""),
        topics=_as_topics(row.get(_TOPICS_FIELD)),
        lcc_subclass=str(subclass) if subclass else None,
    )


class GradedJudge:
    """Assigns graded relevance to (query, corpus document) pairs.

    Grading delegates to ``shelf.evaluate.strata.classify_relation`` so the
    relation ladder has exactly one definition in the codebase; this class only
    maps a :class:`~shelf.evaluate.strata.PairRelation` onto the gain scale for
    the axis under evaluation.

    Two things are computed separately for good reason. Gains are needed only
    for the documents a run actually retrieved (at most ``top_k`` per query),
    while NDCG's *normalizer* needs the size of each relevance tier across the
    whole corpus. Enumerating a tier holding 1,600 documents to normalize an
    NDCG@10 would be pure waste, so tier sizes come from precomputed group
    counts and are expanded only as far as the cutoff requires.
    """

    def __init__(
        self,
        scheme: GradedScheme,
        corpus_df: pl.DataFrame,
        id_field: str = "id",
    ):
        """Index a corpus for graded judging.

        Args:
            scheme: Gain tiers for the axis under evaluation.
            corpus_df: Corpus documents.
            id_field: Column holding the document ID.
        """
        self.scheme = scheme
        self.id_field = id_field

        columns = set(corpus_df.columns)
        self.has_subclass = _SUBCLASS_FIELD in columns
        self.has_topics = _TOPICS_FIELD in columns

        self._facets: dict[str, DocumentFacets] = {}
        self._class_counts: dict[str, int] = {}
        self._subclass_counts: dict[str, int] = {}
        self._form_counts: dict[str, int] = {}
        self._category_counts: dict[str, int] = {}

        lcc_values: list[str] = []
        topic_rows: list[frozenset[str]] = []

        for row in corpus_df.iter_rows(named=True):
            doc_id = str(row[id_field])
            facets = _facets_from_row(row)
            self._facets[doc_id] = facets

            lcc_values.append(facets.lcc_code)
            topic_rows.append(facets.topic_set())

            if facets.lcc_code:
                self._class_counts[facets.lcc_code] = (
                    self._class_counts.get(facets.lcc_code, 0) + 1
                )
            if facets.lcc_subclass:
                self._subclass_counts[facets.lcc_subclass] = (
                    self._subclass_counts.get(facets.lcc_subclass, 0) + 1
                )
            if facets.lcgft_form:
                self._form_counts[facets.lcgft_form] = (
                    self._form_counts.get(facets.lcgft_form, 0) + 1
                )
            if facets.lcgft_category:
                self._category_counts[facets.lcgft_category] = (
                    self._category_counts.get(facets.lcgft_category, 0) + 1
                )

        self._lcc_array = np.array(lcc_values, dtype=object)
        self._topic_membership: dict[str, np.ndarray] = {}
        size = len(topic_rows)
        for i, topics in enumerate(topic_rows):
            for topic in topics:
                column = self._topic_membership.get(topic)
                if column is None:
                    column = np.zeros(size, dtype=bool)
                    self._topic_membership[topic] = column
                column[i] = True

    def gain(self, query: DocumentFacets, doc_id: str) -> float:
        """Graded relevance of one corpus document to one query."""
        facets = self._facets.get(doc_id)
        if facets is None:
            return 0.0
        return self.gain_for_relation(query, facets)

    def gain_for_relation(self, query: DocumentFacets, doc: DocumentFacets) -> float:
        """Map the strata relation between two documents onto the gain scale."""
        relation = classify_relation(query, doc)

        if self.scheme.axis == _LCC_FIELD:
            if relation.subject_level is SubjectRelation.SAME_SUBCLASS:
                return 3.0
            if relation.subject_level is SubjectRelation.SAME_CLASS:
                return 2.0
            # Different class: a shared LCSH topic is the only partial credit
            # the subject axis admits, and it is genuinely subject-bearing --
            # unlike a shared genre, which says nothing about subject.
            if relation.shares_any_topic:
                return 1.0
            return 0.0

        if self.scheme.axis == _FORM_FIELD:
            if relation.form_level is FormRelation.SAME_FORM:
                return 3.0
            if relation.form_level is FormRelation.SAME_CATEGORY:
                return 1.0
            return 0.0

        if self.scheme.axis == _CATEGORY_FIELD:
            if relation.form_level is FormRelation.SAME_FORM:
                return 3.0
            if relation.form_level is FormRelation.SAME_CATEGORY:
                return 2.0
            return 0.0

        return 0.0

    def ideal_tiers(self, query: DocumentFacets, k: int) -> list[tuple[float, int]]:
        """Sizes of each positive relevance tier, best first.

        Counting stops as soon as the tiers seen can fill ``k`` positions, so
        the expensive shared-topic tier is only ever counted for a query whose
        subject tiers are smaller than the cutoff.
        """
        tiers: list[tuple[float, int]] = []
        total = 0

        for gain, count in self._tier_counts(query):
            if count <= 0:
                continue
            tiers.append((gain, count))
            total += count
            if total >= k:
                break

        return tiers

    def _tier_counts(self, query: DocumentFacets):
        """Yield ``(gain, corpus count)`` per tier, best first and lazily."""
        if self.scheme.axis == _LCC_FIELD:
            subclass_count = (
                self._subclass_counts.get(query.lcc_subclass or "", 0)
                if query.lcc_subclass
                else 0
            )
            yield 3.0, subclass_count
            yield 2.0, self._class_counts.get(query.lcc_code, 0) - subclass_count
            yield 1.0, self._shared_topic_count(query)
            return

        form_count = self._form_counts.get(query.lcgft_form, 0)
        category_count = self._category_counts.get(query.lcgft_category, 0)

        if self.scheme.axis == _FORM_FIELD:
            yield 3.0, form_count
            yield 1.0, category_count - form_count
            return

        if self.scheme.axis == _CATEGORY_FIELD:
            yield 3.0, form_count
            yield 2.0, category_count - form_count

    def _shared_topic_count(self, query: DocumentFacets) -> int:
        """Corpus documents in a different LCC class sharing at least one topic."""
        if not self.has_topics or self._lcc_array.size == 0:
            return 0

        topics = query.topic_set()
        if not topics:
            return 0

        mask = np.zeros(self._lcc_array.shape[0], dtype=bool)
        for topic in topics:
            column = self._topic_membership.get(topic)
            if column is not None:
                mask |= column

        if query.lcc_code:
            mask &= self._lcc_array != query.lcc_code
        return int(mask.sum())


class RetrievalEvaluator(TaskEvaluator):
    """Evaluator for retrieval tasks.

    Supports two modes:
    1. From predictions file: Pre-computed rankings
    2. From embedder: Compute rankings via cosine similarity

    For LCC/Form retrieval:
    - Queries are documents from the test split
    - Corpus is documents from train+validation splits
    - Relevance = same label (LCC code, LCGFT form, etc.)

    Example:
        from shelf.evaluate.evaluators import RetrievalEvaluator
        from shelf.evaluate.adapters import SentenceTransformerEmbedder
        from shelf.evaluate.registry import get_task

        task_spec = get_task("lcc_retrieval")
        evaluator = RetrievalEvaluator(task_spec)

        embedder = SentenceTransformerEmbedder.from_pretrained("all-MiniLM-L6-v2")
        result = evaluator.evaluate_embedder(embedder)
        print(result.summary())
    """

    def __init__(
        self,
        task_spec: TaskSpec,
        k_values: list[int] | None = None,
        random_seed: int = 42,
    ):
        """Initialize retrieval evaluator.

        Args:
            task_spec: Task specification
            k_values: List of k values for @k metrics (default: [1, 5, 10, 50, 100])
            random_seed: Random seed for reproducibility
        """
        super().__init__(task_spec, random_seed)
        self.k_values = k_values or [1, 5, 10, 50, 100]

    def evaluate(
        self,
        predictions: list[dict[str, Any]],
        ground_truth: pl.DataFrame,
        compute_ci: bool = False,
        corpus_splits: list[str] | None = None,
        save_samples: bool = False,
        model_key: str | None = None,
    ) -> EvaluationResult:
        """Evaluate retrieval predictions.

        Args:
            predictions: List of {"query_id": str, "ranked_doc_ids": [str, ...]}
            ground_truth: DataFrame with query and corpus documents
            compute_ci: Whether to compute confidence intervals (not yet implemented)
            corpus_splits: Corpus splits to use for relevance (default: train + validation)
            save_samples: Whether to capture per-query results for detailed analysis
            model_key: Model identifier for per-sample results

        Returns:
            EvaluationResult with retrieval metrics
        """
        corpus_splits = corpus_splits or ["train", "validation"]

        # Load corpus documents (train + validation by default)
        corpus_dfs = []
        for split_name in corpus_splits:
            df = self._load_ground_truth(split_name)
            corpus_dfs.append(df)
        corpus_df = pl.concat(corpus_dfs) if corpus_dfs else pl.DataFrame()

        id_field = self.task_spec.id_field

        # Validate predictions against expected IDs
        validation = validate_retrieval_predictions(
            predictions=predictions,
            query_ids=set(ground_truth[id_field].to_list()),
            corpus_ids=set(corpus_df[id_field].to_list()),
        )
        if not validation.valid:
            raise ValidationError(validation.errors)

        # Build results dict: query_id -> ranked_doc_ids
        results: dict[str, list[str]] = {}
        for pred in predictions:
            query_id = pred["query_id"]
            ranked_ids = pred["ranked_doc_ids"]
            results[query_id] = ranked_ids

        # Build relevance judgments. For an instruction task these come from the
        # instruction, not from task_spec.label_field.
        relevance, extra_metrics, extra_per_query = self._judge(
            queries_df=ground_truth,
            corpus_df=corpus_df,
            results=results,
            compute_per_query=True,
        )

        # Compute metrics
        metrics = compute_retrieval_metrics(
            results=results,
            relevance=relevance,
            k_values=self.k_values,
            compute_per_query=True,
        )
        metrics.update(extra_metrics)

        # Extract per-query metrics
        per_query = metrics.pop("per_query", None)
        per_query = _merge_per_query(per_query, extra_per_query)
        num_queries = metrics.pop("num_queries", len(results))

        # Build per-sample results if requested
        per_sample_list: list[PerSampleResult] = []
        if save_samples and per_query:
            # Get available columns for metadata extraction
            available_cols = set(ground_truth.columns)
            metadata_fields = [f for f in STRATIFICATION_FIELDS if f in available_cols]

            # Build a lookup for query metadata
            query_metadata_map: dict[str, dict[str, Any]] = {}
            for row in ground_truth.iter_rows(named=True):
                query_id = row[id_field]
                metadata: dict[str, Any] = {}
                for field in metadata_fields:
                    if field in row and row[field] is not None:
                        metadata[field] = row[field]

                # Add text length bucket for length analysis
                text_field = self.task_spec.text_field
                if text_field in row and row[text_field]:
                    text_len = len(row[text_field])
                    if text_len < 500:
                        metadata["length_bucket"] = "short"
                    elif text_len < 2000:
                        metadata["length_bucket"] = "medium"
                    else:
                        metadata["length_bucket"] = "long"
                    metadata["text_length"] = text_len

                query_metadata_map[query_id] = metadata

            # Create per-sample results for each query
            for query_id, query_metrics in per_query.items():
                # Get relevant doc IDs (ground truth)
                relevant_ids = relevance.get(query_id, set())
                # Get retrieved doc IDs (top k from predictions)
                retrieved_ids = results.get(query_id, [])

                # Check if any relevant doc was in top 10 (simple correctness measure)
                top_k_retrieved = retrieved_ids[:10]
                has_relevant = any(doc_id in relevant_ids for doc_id in top_k_retrieved)

                # Use NDCG@10 as the per-query score
                score = query_metrics.get("ndcg@10", 0.0)

                per_sample_list.append(
                    PerSampleResult(
                        id=query_id,
                        y_true=list(relevant_ids),  # List of relevant doc IDs
                        y_pred=top_k_retrieved,  # Top 10 retrieved doc IDs
                        correct=has_relevant,
                        score=score,
                        metadata=query_metadata_map.get(query_id, {}),
                    )
                )

        result = self._create_result(
            metrics=metrics,
            ground_truth=ground_truth,
            split=self.task_spec.default_split,
            per_query_metrics=per_query,
            num_samples=num_queries,
        )

        # Attach per-sample results if captured
        if save_samples and per_sample_list:
            result.per_sample_results = PerSampleResults(
                task=self.task_spec.name,
                task_type="retrieval",
                model_key=model_key or "unknown",
                split=self.task_spec.default_split,
                samples=per_sample_list,
            )

        return result

    def evaluate_embedder(
        self,
        embedder: TextEmbedder,
        split: str | None = None,
        corpus_splits: list[str] | None = None,
        max_queries: int | None = None,
        top_k: int = 100,
        batch_size: int = 32,
        show_progress: bool = True,
        save_samples: bool = False,
    ) -> EvaluationResult:
        """Evaluate an embedder directly on retrieval task.

        This method:
        1. Loads queries from the specified split
        2. Loads corpus from corpus_splits (default: train + validation)
        3. Encodes all documents with the embedder
        4. Computes cosine similarities
        5. Ranks and evaluates

        Args:
            embedder: TextEmbedder instance
            split: Query split (default: task default, usually "test")
            corpus_splits: Corpus splits (default: ["train", "validation"])
            max_queries: Maximum number of queries to evaluate (for testing)
            top_k: Number of documents to retrieve per query (default: 100)
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bars
            save_samples: Whether to capture per-query results for detailed analysis

        Returns:
            EvaluationResult with retrieval metrics
        """
        split = split or self.task_spec.default_split
        corpus_splits = corpus_splits or ["train", "validation"]

        logger.info(f"Loading queries from split: {split}")
        logger.info(f"Loading corpus from splits: {corpus_splits}")

        # Load query documents
        queries_df = self._load_ground_truth(split)

        # Load corpus documents
        corpus_dfs = []
        for corpus_split in corpus_splits:
            df = self._load_ground_truth(corpus_split)
            corpus_dfs.append(df)
        corpus_df = pl.concat(corpus_dfs)

        # Optionally limit queries for testing
        if max_queries is not None:
            queries_df = queries_df.head(max_queries)

        logger.info(f"Queries: {len(queries_df)}, Corpus: {len(corpus_df)}")

        # Get text field
        text_field = self.task_spec.text_field
        id_field = self.task_spec.id_field

        # Extract texts
        query_texts = queries_df[text_field].to_list()
        query_ids = queries_df[id_field].to_list()

        corpus_texts = corpus_df[text_field].to_list()
        corpus_ids = corpus_df[id_field].to_list()

        # Encode corpus first (important for TF-IDF which fits on first encode)
        logger.info("Encoding corpus...")
        corpus_embeddings = embedder.encode(
            corpus_texts,
            batch_size=batch_size,
            show_progress=show_progress,
        )

        # Encode queries. An instruction task prefixes the query text with its
        # instruction and uses the query encoding role, so an instruct-embedder
        # sees the prompt shape its model card documents.
        logger.info("Encoding queries...")
        if self.instruction is not None:
            query_embeddings = self._encode_queries(
                embedder,
                self._render_queries(query_texts),
                batch_size=batch_size,
                show_progress=show_progress,
            )
        else:
            query_embeddings = embedder.encode(
                query_texts,
                batch_size=batch_size,
                show_progress=show_progress,
            )

        # Compute rankings
        logger.info("Computing rankings...")
        results = self._compute_rankings(
            query_ids=query_ids,
            query_embeddings=query_embeddings,
            corpus_ids=corpus_ids,
            corpus_embeddings=corpus_embeddings,
            top_k=top_k,
            show_progress=show_progress,
        )

        # Build relevance judgments. For an instruction task these come from the
        # instruction, not from task_spec.label_field.
        compute_per_query = save_samples
        relevance, extra_metrics, extra_per_query = self._judge(
            queries_df=queries_df,
            corpus_df=corpus_df,
            results=results,
            compute_per_query=compute_per_query,
        )

        # Compute metrics
        # Enable per-query metrics if save_samples is True
        metrics = compute_retrieval_metrics(
            results=results,
            relevance=relevance,
            k_values=self.k_values,
            compute_per_query=compute_per_query,
        )
        metrics.update(extra_metrics)

        # Extract per-query metrics if available
        per_query = metrics.pop("per_query", None)
        per_query = _merge_per_query(per_query, extra_per_query)
        num_queries = metrics.pop("num_queries", len(results))

        # Build per-sample results if requested
        per_sample_list: list[PerSampleResult] = []
        if save_samples and per_query:
            # Get available columns for metadata extraction
            available_cols = set(queries_df.columns)
            metadata_fields = [f for f in STRATIFICATION_FIELDS if f in available_cols]

            # Build a lookup for query metadata
            query_metadata_map: dict[str, dict[str, Any]] = {}
            for row in queries_df.iter_rows(named=True):
                query_id = row[id_field]
                metadata: dict[str, Any] = {}
                for field in metadata_fields:
                    if field in row and row[field] is not None:
                        metadata[field] = row[field]

                # Add text length bucket for length analysis
                text_field = self.task_spec.text_field
                if text_field in row and row[text_field]:
                    text_len = len(row[text_field])
                    if text_len < 500:
                        metadata["length_bucket"] = "short"
                    elif text_len < 2000:
                        metadata["length_bucket"] = "medium"
                    else:
                        metadata["length_bucket"] = "long"
                    metadata["text_length"] = text_len

                query_metadata_map[query_id] = metadata

            # Create per-sample results for each query
            for query_id, query_metrics in per_query.items():
                # Get relevant doc IDs (ground truth)
                relevant_ids = relevance.get(query_id, set())
                # Get retrieved doc IDs (top k from predictions)
                retrieved_ids = results.get(query_id, [])

                # Check if any relevant doc was in top 10 (simple correctness measure)
                top_k_retrieved = retrieved_ids[:10]
                has_relevant = any(doc_id in relevant_ids for doc_id in top_k_retrieved)

                # Use NDCG@10 as the per-query score
                score = query_metrics.get("ndcg@10", 0.0)

                per_sample_list.append(
                    PerSampleResult(
                        id=query_id,
                        y_true=list(relevant_ids),  # List of relevant doc IDs
                        y_pred=top_k_retrieved,  # Top 10 retrieved doc IDs
                        correct=has_relevant,
                        score=score,
                        metadata=query_metadata_map.get(query_id, {}),
                    )
                )

        result = self._create_result(
            metrics=metrics,
            ground_truth=queries_df,
            split=split,
            num_samples=num_queries,
            model_name=embedder.model_name,
            corpus_size=len(corpus_df),
            embedding_dim=embedder.embedding_dim,
        )

        # Attach per-sample results if captured
        if save_samples and per_sample_list:
            result.per_sample_results = PerSampleResults(
                task=self.task_spec.name,
                task_type="retrieval",
                model_key=embedder.model_name or "unknown",
                split=split,
                samples=per_sample_list,
            )

        return result

    def _compute_rankings(
        self,
        query_ids: list[str],
        query_embeddings: np.ndarray,
        corpus_ids: list[str],
        corpus_embeddings: np.ndarray,
        top_k: int = 100,
        show_progress: bool = True,
    ) -> dict[str, list[str]]:
        """Compute rankings for all queries.

        Uses batched cosine similarity for efficiency.

        Args:
            query_ids: List of query IDs
            query_embeddings: Query embeddings (n_queries, dim)
            corpus_ids: List of corpus document IDs
            corpus_embeddings: Corpus embeddings (n_corpus, dim)
            top_k: Number of top results to keep per query
            show_progress: Whether to show progress bar

        Returns:
            Dict mapping query_id to ranked list of corpus doc IDs
        """
        results: dict[str, list[str]] = {}
        corpus_ids_array = np.array(corpus_ids)

        # Process queries in batches for memory efficiency
        batch_size = 100
        iterator = range(0, len(query_ids), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Computing similarities")

        for start_idx in iterator:
            end_idx = min(start_idx + batch_size, len(query_ids))
            batch_query_embeddings = query_embeddings[start_idx:end_idx]
            batch_query_ids = query_ids[start_idx:end_idx]

            # Compute cosine similarities for batch
            similarities = cosine_similarity(batch_query_embeddings, corpus_embeddings)

            # Get top-k indices for each query in batch
            for i, query_id in enumerate(batch_query_ids):
                query_sims = similarities[i]

                # Get top-k indices (descending similarity)
                top_indices = np.argsort(query_sims)[::-1][:top_k]

                # Map indices to document IDs
                ranked_doc_ids = corpus_ids_array[top_indices].tolist()
                results[query_id] = ranked_doc_ids

        return results

    def evaluate_retriever(
        self,
        retriever: BM25Retriever,
        split: str | None = None,
        corpus_splits: list[str] | None = None,
        max_queries: int | None = None,
        top_k: int = 100,
        show_progress: bool = True,
        save_samples: bool = False,
    ) -> EvaluationResult:
        """Evaluate a retriever directly on retrieval task.

        This method is for retrievers like BM25 that directly produce rankings
        rather than embeddings. The retriever must implement fit() and retrieve().

        Args:
            retriever: BM25Retriever instance (or any retriever with fit/retrieve)
            split: Query split (default: task default, usually "test")
            corpus_splits: Corpus splits (default: ["train", "validation"])
            max_queries: Maximum number of queries to evaluate (for testing)
            top_k: Number of documents to retrieve per query
            show_progress: Whether to show progress bars
            save_samples: Whether to capture per-query results for detailed analysis

        Returns:
            EvaluationResult with retrieval metrics
        """
        split = split or self.task_spec.default_split
        corpus_splits = corpus_splits or ["train", "validation"]

        logger.info(f"Loading queries from split: {split}")
        logger.info(f"Loading corpus from splits: {corpus_splits}")

        # Load query documents
        queries_df = self._load_ground_truth(split)

        # Load corpus documents
        corpus_dfs = []
        for corpus_split in corpus_splits:
            df = self._load_ground_truth(corpus_split)
            corpus_dfs.append(df)
        corpus_df = pl.concat(corpus_dfs)

        # Optionally limit queries for testing
        if max_queries is not None:
            queries_df = queries_df.head(max_queries)

        logger.info(f"Queries: {len(queries_df)}, Corpus: {len(corpus_df)}")

        # Get text field
        text_field = self.task_spec.text_field
        id_field = self.task_spec.id_field

        # Extract texts and IDs
        query_texts = queries_df[text_field].to_list()
        query_ids = queries_df[id_field].to_list()

        corpus_texts = corpus_df[text_field].to_list()
        corpus_ids = corpus_df[id_field].to_list()

        # Fit retriever on corpus
        logger.info("Fitting retriever on corpus...")
        retriever.fit(corpus_texts, corpus_ids)

        # Retrieve for queries. A lexical retriever gets the same instruction
        # prefix a dense model does: it is the honest null, and a sparse model
        # that still scores well is telling us the task is lexically solvable
        # rather than instruction-sensitive.
        logger.info("Retrieving documents for queries...")
        results = retriever.retrieve(
            query_texts=self._render_queries(query_texts),
            query_ids=query_ids,
            top_k=top_k,
            show_progress=show_progress,
        )

        # Build relevance judgments. For an instruction task these come from the
        # instruction, not from task_spec.label_field.
        compute_per_query = save_samples
        relevance, extra_metrics, extra_per_query = self._judge(
            queries_df=queries_df,
            corpus_df=corpus_df,
            results=results,
            compute_per_query=compute_per_query,
        )

        # Compute metrics
        # Enable per-query metrics if save_samples is True
        metrics = compute_retrieval_metrics(
            results=results,
            relevance=relevance,
            k_values=self.k_values,
            compute_per_query=compute_per_query,
        )
        metrics.update(extra_metrics)

        # Extract per-query metrics if available
        per_query = metrics.pop("per_query", None)
        per_query = _merge_per_query(per_query, extra_per_query)
        num_queries = metrics.pop("num_queries", len(results))

        # Build per-sample results if requested
        per_sample_list: list[PerSampleResult] = []
        if save_samples and per_query:
            # Get available columns for metadata extraction
            id_field = self.task_spec.id_field
            available_cols = set(queries_df.columns)
            metadata_fields = [f for f in STRATIFICATION_FIELDS if f in available_cols]

            # Build a lookup for query metadata
            query_metadata_map: dict[str, dict[str, Any]] = {}
            for row in queries_df.iter_rows(named=True):
                query_id = row[id_field]
                metadata: dict[str, Any] = {}
                for field in metadata_fields:
                    if field in row and row[field] is not None:
                        metadata[field] = row[field]

                # Add text length bucket for length analysis
                text_field = self.task_spec.text_field
                if text_field in row and row[text_field]:
                    text_len = len(row[text_field])
                    if text_len < 500:
                        metadata["length_bucket"] = "short"
                    elif text_len < 2000:
                        metadata["length_bucket"] = "medium"
                    else:
                        metadata["length_bucket"] = "long"
                    metadata["text_length"] = text_len

                query_metadata_map[query_id] = metadata

            # Create per-sample results for each query
            for query_id, query_metrics in per_query.items():
                # Get relevant doc IDs (ground truth)
                relevant_ids = relevance.get(query_id, set())
                # Get retrieved doc IDs (top k from predictions)
                retrieved_ids = results.get(query_id, [])

                # Check if any relevant doc was in top 10 (simple correctness measure)
                top_k_retrieved = retrieved_ids[:10]
                has_relevant = any(doc_id in relevant_ids for doc_id in top_k_retrieved)

                # Use NDCG@10 as the per-query score
                score = query_metrics.get("ndcg@10", 0.0)

                per_sample_list.append(
                    PerSampleResult(
                        id=query_id,
                        y_true=list(relevant_ids),  # List of relevant doc IDs
                        y_pred=top_k_retrieved,  # Top 10 retrieved doc IDs
                        correct=has_relevant,
                        score=score,
                        metadata=query_metadata_map.get(query_id, {}),
                    )
                )

        result = self._create_result(
            metrics=metrics,
            ground_truth=queries_df,
            split=split,
            num_samples=num_queries,
            model_name=retriever.model_name,
            corpus_size=len(corpus_df),
        )

        # Attach per-sample results if captured
        if save_samples and per_sample_list:
            result.per_sample_results = PerSampleResults(
                task=self.task_spec.name,
                task_type="retrieval",
                model_key=retriever.model_name or "unknown",
                split=split,
                samples=per_sample_list,
            )

        return result

    # ------------------------------------------------------------------
    # Judging: graded relevance and instruction-conditioned relevance
    # ------------------------------------------------------------------

    @property
    def instruction(self) -> InstructionSpec | None:
        """The instruction backing this task, or None for a label-match task."""
        return get_instruction(self.task_spec.name)

    def _render_queries(self, query_texts: list[str]) -> list[str]:
        """Prefix query texts with the task instruction, if there is one.

        The prefix is applied on the query side only, which is what makes the
        task well-formed for a plain similarity model as well: it reads the
        instruction as a few extra words and is therefore the null condition
        this task is designed to separate an instruct-embedder from.
        """
        instruction = self.instruction
        if instruction is None:
            return query_texts
        return [instruction.render(text) for text in query_texts]

    @staticmethod
    def _encode_queries(
        embedder: TextEmbedder,
        texts: list[str],
        batch_size: int,
        show_progress: bool,
    ) -> np.ndarray:
        """Encode queries with the model-card query prompt when one exists.

        E5 wants ``"query: "``, BGE wants an instruction, and running them
        without it evaluates the model outside its documented usage. Adapters
        without the query role fall back to ``encode``, so nothing changes for
        the sparse baselines.
        """
        encode_queries = getattr(embedder, "encode_queries", None)
        if callable(encode_queries):
            return encode_queries(
                texts, batch_size=batch_size, show_progress=show_progress
            )
        return embedder.encode(
            texts, batch_size=batch_size, show_progress=show_progress
        )

    def _judge(
        self,
        queries_df: pl.DataFrame,
        corpus_df: pl.DataFrame,
        results: dict[str, list[str]],
        compute_per_query: bool = False,
    ) -> tuple[dict[str, set[str]], dict[str, float], dict[str, dict[str, float]]]:
        """Build relevance judgments and any extra metrics they support.

        Args:
            queries_df: Query documents.
            corpus_df: Corpus documents.
            results: query_id -> ranked corpus document IDs.
            compute_per_query: Whether to keep per-query breakdowns.

        Returns:
            ``(relevance, extra_metrics, extra_per_query)``. ``relevance`` is
            the binary qrels the standard IR metrics are computed from;
            ``extra_metrics`` holds graded NDCG for a label-match task and the
            constraint diagnostics for an instruction task.
        """
        instruction = self.instruction
        if instruction is not None:
            return self._judge_instruction(
                instruction, queries_df, corpus_df, results, compute_per_query
            )

        relevance = self._build_relevance_judgments_from_df(
            queries_df=queries_df,
            corpus_df=corpus_df,
        )
        graded_metrics, graded_per_query = self._graded_metrics(
            queries_df=queries_df,
            corpus_df=corpus_df,
            results=results,
            compute_per_query=compute_per_query,
        )
        return relevance, graded_metrics, graded_per_query

    def _judge_instruction(
        self,
        instruction: InstructionSpec,
        queries_df: pl.DataFrame,
        corpus_df: pl.DataFrame,
        results: dict[str, list[str]],
        compute_per_query: bool,
    ) -> tuple[dict[str, set[str]], dict[str, float], dict[str, dict[str, float]]]:
        """Judge an instruction task, including the constraint diagnostics."""
        missing = [
            name
            for name in instruction.required_fields
            if name not in corpus_df.columns or name not in queries_df.columns
        ]
        if missing:
            raise ValueError(
                f"task {self.task_spec.name!r} needs columns {missing}, which the "
                f"dataset does not provide"
            )

        judge = InstructionJudge(
            instruction,
            list(corpus_df.iter_rows(named=True)),
            id_field=self.task_spec.id_field,
        )
        judgments = judge.judge(
            list(queries_df.iter_rows(named=True)),
            results=results,
            k_values=self.k_values,
            compute_per_query=compute_per_query,
        )

        metrics = dict(judgments.metrics)
        # Queries the instruction has no answer for are reported, not hidden:
        # a large count means the instruction is mis-specified for this corpus.
        metrics["queries_without_answer"] = float(judgments.num_empty)

        if judgments.num_empty:
            logger.info(
                "%s: %d of %d queries have no document satisfying the instruction",
                self.task_spec.name,
                judgments.num_empty,
                judgments.num_empty + judgments.num_queries,
            )

        return judgments.relevance, metrics, judgments.per_query

    def _graded_metrics(
        self,
        queries_df: pl.DataFrame,
        corpus_df: pl.DataFrame,
        results: dict[str, list[str]],
        compute_per_query: bool,
    ) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
        """Compute graded NDCG for a label-match retrieval task.

        Returns empty dictionaries when the dataset cannot support grading --
        a task on an axis with no registered scheme, or a corpus missing the
        facet columns the scheme reads. Grading is an addition to the report,
        so its absence must never break an evaluation.
        """
        scheme = GRADED_SCHEMES.get(self.task_spec.label_field)
        if scheme is None:
            return {}, {}

        available = set(corpus_df.columns) & set(queries_df.columns)
        missing = [name for name in scheme.needs if name not in available]
        if missing:
            logger.info(
                "Skipping graded relevance for %s: missing columns %s",
                self.task_spec.name,
                missing,
            )
            return {}, {}

        if not results:
            return {}, {}

        id_field = self.task_spec.id_field
        judge = GradedJudge(scheme, corpus_df, id_field=id_field)
        max_k = max(self.k_values)

        gains: dict[str, dict[str, float]] = {}
        ideal_tiers: dict[str, list[tuple[float, int]]] = {}

        for row in queries_df.iter_rows(named=True):
            query_id = str(row[id_field])
            ranked = results.get(query_id)
            if ranked is None:
                continue

            query_facets = _facets_from_row(row)
            ideal_tiers[query_id] = judge.ideal_tiers(query_facets, max_k)
            gains[query_id] = {
                doc_id: gain
                for doc_id in ranked[:max_k]
                if (gain := judge.gain(query_facets, doc_id)) > 0.0
            }

        graded = compute_graded_retrieval_metrics(
            results=results,
            gains=gains,
            ideal_tiers=ideal_tiers,
            k_values=self.k_values,
            compute_per_query=compute_per_query,
        )
        per_query = graded.pop("per_query", {})
        graded.pop("num_queries", None)
        graded["graded_relevance_max_gain"] = float(scheme.tiers[0][0])

        return graded, per_query

    def _build_relevance_judgments(
        self,
        ground_truth: pl.DataFrame,
    ) -> dict[str, set[str]]:
        """Build relevance judgments from ground truth DataFrame.

        For retrieval tasks, a document is relevant to a query if they
        share the same label (LCC code, LCGFT form, etc.).

        Args:
            ground_truth: DataFrame with all documents

        Returns:
            Dict mapping query_id to set of relevant doc IDs
        """
        label_field = self.task_spec.label_field
        id_field = self.task_spec.id_field

        # Group documents by label
        label_to_docs: dict[str, set[str]] = {}
        for row in ground_truth.iter_rows(named=True):
            label = row[label_field]
            doc_id = row[id_field]
            if label not in label_to_docs:
                label_to_docs[label] = set()
            label_to_docs[label].add(doc_id)

        # Build relevance: for each doc, relevant docs are those with same label
        relevance: dict[str, set[str]] = {}
        for row in ground_truth.iter_rows(named=True):
            doc_id = row[id_field]
            label = row[label_field]
            # Relevant docs are all docs with same label, excluding self
            relevant = label_to_docs[label] - {doc_id}
            relevance[doc_id] = relevant

        return relevance

    def _build_relevance_judgments_from_df(
        self,
        queries_df: pl.DataFrame,
        corpus_df: pl.DataFrame,
    ) -> dict[str, set[str]]:
        """Build relevance judgments with separate query and corpus DFs.

        Args:
            queries_df: DataFrame with query documents
            corpus_df: DataFrame with corpus documents

        Returns:
            Dict mapping query_id to set of relevant corpus doc IDs
        """
        label_field = self.task_spec.label_field
        id_field = self.task_spec.id_field

        # Group corpus documents by label
        label_to_corpus_docs: dict[str, set[str]] = {}
        for row in corpus_df.iter_rows(named=True):
            label = row[label_field]
            doc_id = row[id_field]
            if label not in label_to_corpus_docs:
                label_to_corpus_docs[label] = set()
            label_to_corpus_docs[label].add(doc_id)

        # Build relevance: for each query, relevant corpus docs are those with same label
        relevance: dict[str, set[str]] = {}
        for row in queries_df.iter_rows(named=True):
            query_id = row[id_field]
            label = row[label_field]
            # Relevant docs are all corpus docs with same label
            relevant = label_to_corpus_docs.get(label, set())
            relevance[query_id] = relevant

        return relevance

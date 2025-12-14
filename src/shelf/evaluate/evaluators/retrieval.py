"""Retrieval evaluator for SHELF tasks.

Evaluates embedding models on retrieval tasks like LCC retrieval,
form retrieval, and topic retrieval.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from shelf.evaluate.evaluators.base import TaskEvaluator
from shelf.evaluate.metrics.retrieval import compute_retrieval_metrics
from shelf.evaluate.results import (
    EvaluationResult,
    PerSampleResult,
    PerSampleResults,
)
from shelf.evaluate.tasks import TaskSpec
from shelf.evaluate.schemas import (
    ValidationError,
    validate_retrieval_predictions,
)

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

if TYPE_CHECKING:
    from shelf.evaluate.adapters.bm25 import BM25Retriever
    from shelf.evaluate.adapters.protocols import TextEmbedder

logger = logging.getLogger(__name__)


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

        # Build relevance judgments from query and corpus DataFrames
        relevance = self._build_relevance_judgments_from_df(
            queries_df=ground_truth,
            corpus_df=corpus_df,
        )

        # Compute metrics
        metrics = compute_retrieval_metrics(
            results=results,
            relevance=relevance,
            k_values=self.k_values,
            compute_per_query=True,
        )

        # Extract per-query metrics
        per_query = metrics.pop("per_query", None)
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
        embedder: "TextEmbedder",
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

        # Encode queries
        logger.info("Encoding queries...")
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

        # Build relevance judgments
        relevance = self._build_relevance_judgments_from_df(
            queries_df=queries_df,
            corpus_df=corpus_df,
        )

        # Compute metrics
        # Enable per-query metrics if save_samples is True
        compute_per_query = save_samples
        metrics = compute_retrieval_metrics(
            results=results,
            relevance=relevance,
            k_values=self.k_values,
            compute_per_query=compute_per_query,
        )

        # Extract per-query metrics if available
        per_query = metrics.pop("per_query", None)
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
        retriever: "BM25Retriever",
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

        # Retrieve for queries
        logger.info("Retrieving documents for queries...")
        results = retriever.retrieve(
            query_texts=query_texts,
            query_ids=query_ids,
            top_k=top_k,
            show_progress=show_progress,
        )

        # Build relevance judgments
        relevance = self._build_relevance_judgments_from_df(
            queries_df=queries_df,
            corpus_df=corpus_df,
        )

        # Compute metrics
        # Enable per-query metrics if save_samples is True
        compute_per_query = save_samples
        metrics = compute_retrieval_metrics(
            results=results,
            relevance=relevance,
            k_values=self.k_values,
            compute_per_query=compute_per_query,
        )

        # Extract per-query metrics if available
        per_query = metrics.pop("per_query", None)
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

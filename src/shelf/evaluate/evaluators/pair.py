"""Pair classification evaluator for SHELF tasks.

Evaluates embedding and sparse models on pair classification tasks by comparing
similarity scores (cosine similarity for embeddings, BM25/TF-IDF for sparse)
against ground truth labels.

Supports:
- Embedding models: Cosine similarity between dense vectors
- Sparse models (BM25, TF-IDF): Document similarity using corpus statistics
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl
from datasets import Dataset, load_dataset
from sklearn.metrics.pairwise import cosine_similarity

from shelf.evaluate.evaluators.base import TaskEvaluator
from shelf.evaluate.metrics.pair import compute_pair_metrics
from shelf.evaluate.results import EvaluationResult
from shelf.evaluate.tasks import TaskSpec
from shelf.evaluate.schemas import (
    ValidationError,
    validate_pair_predictions,
)

if TYPE_CHECKING:
    from shelf.evaluate.adapters.bm25 import BM25Retriever
    from shelf.evaluate.adapters.protocols import TextEmbedder
    from shelf.evaluate.adapters.tfidf import TfidfEmbedder

logger = logging.getLogger(__name__)


class PairClassificationEvaluator(TaskEvaluator):
    """Evaluator for pair classification tasks.

    Supports three modes:
    1. From predictions file: Pre-computed similarity scores
    2. From embedder: Compute cosine similarity on dense embeddings
    3. From sparse model: Compute similarity using BM25 or TF-IDF scores

    For pair classification:
    - Ground truth comes from the label field (0 or 1)
    - Predictions are similarity scores between pairs
    - A threshold is found to convert scores to binary predictions
    - Primary metric is F1 score

    Example:
        from shelf.evaluate.evaluators import PairClassificationEvaluator
        from shelf.evaluate.registry import get_task

        task_spec = get_task("same_lcc_pairs")
        evaluator = PairClassificationEvaluator(task_spec)

        # From embedder (computes cosine similarity)
        result = evaluator.evaluate_embedder(embedder, split="test")
        print(result.summary())

        # From BM25 (computes document similarity using BM25 scores)
        result = evaluator.evaluate_bm25(bm25_retriever, split="test")
        print(result.summary())

        # From TF-IDF (computes cosine similarity on TF-IDF vectors)
        result = evaluator.evaluate_tfidf(tfidf_embedder, split="test")
        print(result.summary())
    """

    def __init__(
        self,
        task_spec: TaskSpec,
        random_seed: int = 42,
    ):
        """Initialize pair classification evaluator.

        Args:
            task_spec: Task specification
            random_seed: Random seed for reproducibility
        """
        super().__init__(task_spec, random_seed)

    def _load_ground_truth(self, split: str) -> pl.DataFrame:
        """Load ground truth for pair classification tasks.

        Pair data is stored in a different location than regular data.
        Tries local pairs directory first, then falls back to HuggingFace Hub.

        Args:
            split: Dataset split to load

        Returns:
            Polars DataFrame with pair ground truth data
        """
        # Map config names to local directory names
        config_to_dir = {
            "same_lcc_pairs": "same_lcc",
            "same_form_pairs": "same_lcgft",
        }

        config = self.task_spec.dataset_config
        local_dir = config_to_dir.get(config, config)

        # Try local parquet files first
        local_path = Path("data/hf_dataset/pairs") / local_dir / f"{split}.parquet"
        if local_path.exists():
            logger.info(f"Loading pairs from local file: {local_path}")
            return pl.read_parquet(local_path)

        # Fall back to HuggingFace Hub
        logger.info(f"Loading pairs from HuggingFace: {config}")
        dataset = load_dataset(
            self.task_spec.dataset_name,
            config,
            split=split,
        )

        # Ensure we have a Dataset (not DatasetDict)
        if not isinstance(dataset, Dataset):
            raise TypeError(
                f"Expected Dataset, got {type(dataset).__name__}. "
                f"Make sure to specify a split."
            )

        # Convert to Polars DataFrame via Arrow
        result = pl.from_arrow(dataset.data.table)

        # Ensure we return a DataFrame, not a Series
        if isinstance(result, pl.Series):
            raise TypeError(
                "Expected DataFrame from Arrow table conversion, got Series"
            )

        return result

    def evaluate(
        self,
        predictions: list[dict[str, Any]],
        ground_truth: pl.DataFrame,
        compute_ci: bool = False,
    ) -> EvaluationResult:
        """Evaluate pair classification predictions.

        Args:
            predictions: List of {"pair_id": str, "score": float} or {"pair_id": str, "prediction": int}
            ground_truth: DataFrame with ground truth labels
            compute_ci: Whether to compute confidence intervals (not yet implemented)

        Returns:
            EvaluationResult with pair classification metrics
        """
        id_field = self.task_spec.id_field
        label_field = self.task_spec.label_field

        # Validate predictions
        validation = validate_pair_predictions(
            predictions=predictions,
            pair_ids=set(ground_truth[id_field].to_list()),
        )
        if not validation.valid:
            raise ValidationError(validation.errors)

        # Build prediction dict: id -> score
        pred_dict: dict[str, float] = {}
        for pred in predictions:
            pair_id = pred.get("pair_id") or pred.get("id")
            if pair_id is None:
                continue  # Skip predictions without an ID
            # Prefer score; fallback to binary prediction
            score: float | None = pred.get("score")
            if score is None and pred.get("prediction") is not None:
                score = float(pred["prediction"])
            pred_dict[str(pair_id)] = float(score) if score is not None else 0.0

        # Get ground truth
        y_true: list[int] = []
        y_scores: list[float] = []

        for row in ground_truth.iter_rows(named=True):
            pair_id = row[id_field]
            true_label = int(row[label_field])

            if pair_id not in pred_dict:
                logger.warning(f"Missing prediction for pair: {pair_id}")
                continue

            y_true.append(true_label)
            y_scores.append(pred_dict[pair_id])

        if not y_true:
            raise ValueError("No valid predictions found matching ground truth IDs")

        # Compute metrics
        metrics = compute_pair_metrics(y_true, y_scores)

        return self._create_result(
            metrics=metrics,
            ground_truth=ground_truth,
            split=self.task_spec.default_split,
        )

    def evaluate_embedder(
        self,
        embedder: "TextEmbedder",
        split: str | None = None,
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> EvaluationResult:
        """Evaluate an embedder by computing cosine similarity on pairs.

        This method:
        1. Loads pairs from the specified split
        2. Encodes all documents with the embedder
        3. Computes cosine similarity between pair embeddings
        4. Finds optimal threshold and evaluates predictions

        Args:
            embedder: TextEmbedder instance
            split: Dataset split (default: task default, usually "test")
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bars

        Returns:
            EvaluationResult with pair classification metrics
        """
        split = split or self.task_spec.default_split

        logger.info(f"Loading data from split: {split}")

        # Load ground truth (pair data)
        ground_truth = self._load_ground_truth(split)

        logger.info(f"Pairs to evaluate: {len(ground_truth)}")

        # Get field names
        label_field = self.task_spec.label_field

        # Extract pair data
        # For pairs, we need doc_a and doc_b texts
        # Check for title/body fields
        if "doc_a_body" in ground_truth.columns:
            # Combine title and body for each document
            texts_a = [
                f"{row['doc_a_title']}\n\n{row['doc_a_body']}"
                for row in ground_truth.iter_rows(named=True)
            ]
            texts_b = [
                f"{row['doc_b_title']}\n\n{row['doc_b_body']}"
                for row in ground_truth.iter_rows(named=True)
            ]
        else:
            # Fallback to text_a, text_b fields
            texts_a = ground_truth["text_a"].to_list()
            texts_b = ground_truth["text_b"].to_list()

        labels = [int(x) for x in ground_truth[label_field].to_list()]

        # Get unique texts to avoid redundant encoding
        all_texts = list(set(texts_a) | set(texts_b))
        logger.info(f"Unique texts to encode: {len(all_texts)}")

        # Encode all unique texts
        logger.info("Encoding texts...")
        embeddings = embedder.encode(
            all_texts,
            batch_size=batch_size,
            show_progress=show_progress,
        )

        # Build text -> embedding index
        text_to_idx = {text: idx for idx, text in enumerate(all_texts)}

        # Compute similarities for all pairs
        logger.info("Computing pair similarities...")
        y_scores: list[float] = []
        for text_a, text_b in zip(texts_a, texts_b):
            idx_a = text_to_idx[text_a]
            idx_b = text_to_idx[text_b]
            emb_a = embeddings[idx_a].reshape(1, -1)
            emb_b = embeddings[idx_b].reshape(1, -1)
            sim = cosine_similarity(emb_a, emb_b)[0, 0]
            y_scores.append(float(sim))

        # Compute metrics
        metrics = compute_pair_metrics(labels, y_scores)

        return self._create_result(
            metrics=metrics,
            ground_truth=ground_truth,
            split=split,
            model_name=embedder.model_name,
            embedding_dim=embedder.embedding_dim,
        )

    def evaluate_bm25(
        self,
        retriever: "BM25Retriever",
        split: str | None = None,
        show_progress: bool = True,
    ) -> EvaluationResult:
        """Evaluate a BM25 retriever by computing document similarity scores.

        For pair classification with BM25, we compute similarity as:
        sim(A, B) = (BM25(A, B) + BM25(B, A)) / 2

        Where BM25(A, B) is the BM25 score of document A as a query against B.

        This implementation precomputes BM25 scores for all unique documents
        to avoid O(#pairs × #docs) complexity.

        Args:
            retriever: BM25Retriever instance
            split: Dataset split (default: task default, usually "test")
            show_progress: Whether to show progress bars

        Returns:
            EvaluationResult with pair classification metrics
        """
        split = split or self.task_spec.default_split

        logger.info(f"Loading data from split: {split}")

        # Load ground truth (pair data)
        ground_truth = self._load_ground_truth(split)

        logger.info(f"Pairs to evaluate: {len(ground_truth)}")

        # Get field names
        label_field = self.task_spec.label_field

        # Extract pair data
        if "doc_a_body" in ground_truth.columns:
            texts_a = [
                f"{row['doc_a_title']}\n\n{row['doc_a_body']}"
                for row in ground_truth.iter_rows(named=True)
            ]
            texts_b = [
                f"{row['doc_b_title']}\n\n{row['doc_b_body']}"
                for row in ground_truth.iter_rows(named=True)
            ]
        else:
            texts_a = ground_truth["text_a"].to_list()
            texts_b = ground_truth["text_b"].to_list()

        labels = [int(x) for x in ground_truth[label_field].to_list()]

        # Get unique texts
        all_texts = list(set(texts_a) | set(texts_b))
        text_ids = [f"doc_{i}" for i in range(len(all_texts))]

        logger.info(f"Unique texts: {len(all_texts)}")

        # Fit BM25 on all unique texts
        logger.info("Fitting BM25 index...")
        retriever.fit(all_texts, text_ids)

        # Build text -> index mapping
        text_to_idx = {text: idx for idx, text in enumerate(all_texts)}

        # Precompute BM25 scores for all unique documents as queries
        # This is O(#unique_docs × #docs) instead of O(#pairs × #docs)
        logger.info("Precomputing BM25 score matrix...")

        # Use encode() which returns BM25 scores for each text against corpus
        # Note: encode() logs a warning about non-standard usage, which is expected here
        score_matrix = retriever.encode(all_texts, show_progress=False)

        # score_matrix[i, j] = BM25 score of doc i as query against doc j

        # Compute symmetric BM25 similarity for all pairs
        logger.info("Computing pair similarities from precomputed scores...")
        y_scores: list[float] = []

        for text_a, text_b in zip(texts_a, texts_b):
            idx_a = text_to_idx[text_a]
            idx_b = text_to_idx[text_b]

            # BM25(A, B) from precomputed matrix
            score_ab = float(score_matrix[idx_a, idx_b])

            # BM25(B, A) from precomputed matrix
            score_ba = float(score_matrix[idx_b, idx_a])

            # Symmetric similarity
            sim = (score_ab + score_ba) / 2.0
            y_scores.append(sim)

        # Normalize scores to [0, 1] range for threshold finding
        if y_scores:
            min_score = min(y_scores)
            max_score = max(y_scores)
            if max_score > min_score:
                y_scores = [(s - min_score) / (max_score - min_score) for s in y_scores]

        # Compute metrics
        metrics = compute_pair_metrics(labels, y_scores)

        return self._create_result(
            metrics=metrics,
            ground_truth=ground_truth,
            split=split,
            model_name=retriever.model_name,
        )

    def evaluate_tfidf(
        self,
        embedder: "TfidfEmbedder",
        split: str | None = None,
        show_progress: bool = True,
    ) -> EvaluationResult:
        """Evaluate a TF-IDF embedder by computing cosine similarity.

        This method:
        1. Loads pairs from the specified split
        2. Fits TF-IDF on all unique documents
        3. Computes cosine similarity between TF-IDF vectors
        4. Finds optimal threshold and evaluates predictions

        Note: For dense TF-IDF embeddings (with SVD), use evaluate_embedder()
        instead as it handles the encoding more efficiently.

        Args:
            embedder: TfidfEmbedder instance
            split: Dataset split (default: task default, usually "test")
            show_progress: Whether to show progress bars

        Returns:
            EvaluationResult with pair classification metrics
        """
        split = split or self.task_spec.default_split

        logger.info(f"Loading data from split: {split}")

        # Load ground truth (pair data)
        ground_truth = self._load_ground_truth(split)

        logger.info(f"Pairs to evaluate: {len(ground_truth)}")

        # Get field names
        label_field = self.task_spec.label_field

        # Extract pair data
        if "doc_a_body" in ground_truth.columns:
            texts_a = [
                f"{row['doc_a_title']}\n\n{row['doc_a_body']}"
                for row in ground_truth.iter_rows(named=True)
            ]
            texts_b = [
                f"{row['doc_b_title']}\n\n{row['doc_b_body']}"
                for row in ground_truth.iter_rows(named=True)
            ]
        else:
            texts_a = ground_truth["text_a"].to_list()
            texts_b = ground_truth["text_b"].to_list()

        labels = [int(x) for x in ground_truth[label_field].to_list()]

        # Get unique texts to avoid redundant encoding
        all_texts = list(set(texts_a) | set(texts_b))
        logger.info(f"Unique texts to encode: {len(all_texts)}")

        # Fit and encode all unique texts
        logger.info("Computing TF-IDF vectors...")
        embeddings = embedder.encode(all_texts, show_progress=show_progress)

        # Build text -> embedding index
        text_to_idx = {text: idx for idx, text in enumerate(all_texts)}

        # Compute similarities for all pairs
        logger.info("Computing pair similarities...")
        y_scores: list[float] = []
        for text_a, text_b in zip(texts_a, texts_b):
            idx_a = text_to_idx[text_a]
            idx_b = text_to_idx[text_b]
            emb_a = embeddings[idx_a].reshape(1, -1)
            emb_b = embeddings[idx_b].reshape(1, -1)
            sim = cosine_similarity(emb_a, emb_b)[0, 0]
            y_scores.append(float(sim))

        # Compute metrics
        metrics = compute_pair_metrics(labels, y_scores)

        return self._create_result(
            metrics=metrics,
            ground_truth=ground_truth,
            split=split,
            model_name=embedder.model_name,
            embedding_dim=embedder.embedding_dim,
        )

    def evaluate_sparse_cosine(
        self,
        corpus_texts: list[str],
        pairs: list[tuple[int, int]],
        labels: list[int],
        method: str = "tfidf",
    ) -> dict[str, float]:
        """Compute sparse cosine similarity for pair classification.

        Low-level method for computing similarity using sparse TF-IDF vectors.
        Useful for custom evaluation workflows.

        Args:
            corpus_texts: List of all document texts
            pairs: List of (idx_a, idx_b) pairs to compare
            labels: Ground truth labels for each pair
            method: Sparse method ("tfidf" or "bm25")

        Returns:
            Dictionary of metrics
        """
        from shelf.evaluate.text import CorpusStatistics

        logger.info(f"Building corpus statistics for {len(corpus_texts)} documents...")
        corpus = CorpusStatistics()
        corpus.fit(corpus_texts)

        if method == "tfidf":
            # Get TF-IDF matrix
            tfidf_matrix = corpus.get_tfidf_matrix(sublinear_tf=True, normalize=True)

            # Compute cosine similarities
            y_scores = []
            for idx_a, idx_b in pairs:
                vec_a = tfidf_matrix[idx_a]
                vec_b = tfidf_matrix[idx_b]
                sim = float((vec_a @ vec_b.T).toarray()[0, 0])
                y_scores.append(sim)
        else:
            # BM25 similarity
            y_scores = []
            for idx_a, idx_b in pairs:
                # Get tokenized documents
                doc_a_tokens = corpus.tokenizer.tokenize(corpus_texts[idx_a])
                doc_b_tokens = corpus.tokenizer.tokenize(corpus_texts[idx_b])

                # BM25 score A as query against all, get score for B
                scores_ab = corpus.get_bm25_scores(doc_a_tokens)
                score_ab = scores_ab[idx_b]

                scores_ba = corpus.get_bm25_scores(doc_b_tokens)
                score_ba = scores_ba[idx_a]

                sim = float((score_ab + score_ba) / 2.0)
                y_scores.append(sim)

            # Normalize to [0, 1]
            if y_scores:
                min_s, max_s = min(y_scores), max(y_scores)
                if max_s > min_s:
                    y_scores = [(s - min_s) / (max_s - min_s) for s in y_scores]

        # Compute metrics
        metrics = compute_pair_metrics(labels, y_scores)
        return metrics

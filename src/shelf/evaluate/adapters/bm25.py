"""BM25 retriever adapter for SHELF evaluation.

This adapter provides BM25 retrieval with two backends:
1. rank_bm25: Uses rank-bm25's BM25Okapi (default, for compatibility)
2. shelf: Uses SHELF's CorpusStatistics with CSC optimization (faster)

The SHELF backend uses:
- CSC sparse matrices for 26x faster column access
- np.argpartition for O(n) top-k selection
- Batch query processing with term frequency caching
- Shared vocabulary and tokenizer with TF-IDF

Unlike embedding models, BM25 directly scores query-document pairs
rather than producing fixed-dimensional embeddings.

The adapter provides two interfaces:
1. Retriever interface: retrieve(queries, top_k) -> rankings
2. Embedding-compatible interface: encode() for compatibility

For best results, use the retriever interface directly.

Example:
    # Use SHELF backend (faster, recommended)
    retriever = BM25Retriever(backend="shelf")
    retriever.fit(corpus_texts, corpus_ids)
    results = retriever.retrieve(query_texts, query_ids, top_k=100)

    # Use rank_bm25 backend (for compatibility)
    retriever = BM25Retriever(backend="rank_bm25")
"""

from __future__ import annotations

import logging
import re
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
from tqdm import tqdm

if TYPE_CHECKING:
    from rank_bm25 import BM25Okapi

    from shelf.evaluate.text import CorpusStatistics, Tokenizer

logger = logging.getLogger(__name__)


class BM25Backend(Enum):
    """Backend implementations for BM25 retrieval."""

    RANK_BM25 = "rank_bm25"  # Original rank-bm25 library
    SHELF = "shelf"  # SHELF's optimized CorpusStatistics


class _FunctionTokenizerAdapter:
    """Adapter to wrap a tokenizer function as a Tokenizer protocol object.

    This allows custom tokenizer functions to be used with CorpusStatistics.
    """

    def __init__(self, tokenize_fn: Any):
        """Initialize with a tokenizer function.

        Args:
            tokenize_fn: Function that takes text and returns list of tokens
        """
        self._tokenize_fn = tokenize_fn
        self._is_fitted = True  # Function tokenizers are always "fitted"

    @property
    def is_fitted(self) -> bool:
        """Whether the tokenizer is fitted (always True for functions)."""
        return self._is_fitted

    def tokenize(self, text: str) -> list[str]:
        """Tokenize a single text."""
        return self._tokenize_fn(text)

    def tokenize_batch(self, texts: list[str]) -> list[list[str]]:
        """Tokenize multiple texts."""
        return [self._tokenize_fn(text) for text in texts]


def simple_tokenize(text: str) -> list[str]:
    """Simple whitespace tokenizer with lowercasing and punctuation removal.

    Args:
        text: Input text

    Returns:
        List of tokens
    """
    # Lowercase and split on whitespace/punctuation
    text = text.lower()
    tokens = re.findall(r"\b\w+\b", text)
    return tokens


class BM25Retriever:
    """BM25 retriever for SHELF evaluation.

    This implements BM25Okapi (the standard Okapi BM25 algorithm) for
    document retrieval. Unlike embedding models, BM25 computes relevance
    scores directly between queries and documents.

    BM25 is a bag-of-words retrieval function that ranks documents based on
    query terms appearing in each document, regardless of proximity.

    Two backends are available:
    - "rank_bm25": Uses rank-bm25 library (default, for compatibility)
    - "shelf": Uses SHELF's CorpusStatistics with CSC optimization

    The SHELF backend is faster due to:
    - CSC sparse matrices for efficient column access
    - np.argpartition for O(n) top-k selection
    - Batch query processing with term frequency caching

    The workflow for retrieval is:
    1. Fit on corpus documents (builds the BM25 index)
    2. Retrieve for queries (returns ranked document IDs)

    Example:
        # Create retriever with SHELF backend (faster)
        retriever = BM25Retriever(backend="shelf")

        # Or use rank_bm25 backend (for compatibility)
        retriever = BM25Retriever(backend="rank_bm25")

        # Fit on corpus
        retriever.fit(corpus_texts, corpus_ids)

        # Retrieve for queries
        results = retriever.retrieve(query_texts, query_ids, top_k=100)
        # results = {"query_id_1": ["doc_id_5", "doc_id_3", ...], ...}

    Args:
        k1: BM25 term frequency saturation parameter (default: 1.5)
        b: BM25 document length normalization parameter (default: 0.75)
        tokenizer: Tokenization function (default: simple whitespace tokenizer)
        backend: Backend to use ("rank_bm25" or "shelf")

    Attributes:
        bm25: The underlying BM25Okapi index (rank_bm25 backend)
        corpus_stats: CorpusStatistics instance (shelf backend)
        corpus_ids: List of document IDs in the index
        is_fitted: Whether the index has been built
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: Any | None = None,
        backend: str | BM25Backend = BM25Backend.RANK_BM25,
    ):
        """Initialize BM25 retriever.

        Args:
            k1: Term frequency saturation (higher = more weight on term frequency)
            b: Length normalization (0 = no normalization, 1 = full normalization)
            tokenizer: Custom tokenizer function (text -> list of tokens)
            backend: Backend to use ("rank_bm25" or "shelf")
        """
        self.k1 = k1
        self.b = b

        # Parse backend
        if isinstance(backend, str):
            backend = BM25Backend(backend)
        self.backend = backend

        # Set tokenizer based on backend
        if tokenizer is not None:
            # Custom tokenizer provided
            self.tokenizer = tokenizer
            if backend == BM25Backend.SHELF:
                # Wrap the custom tokenizer for use with CorpusStatistics
                self._shelf_tokenizer: Tokenizer | None = _FunctionTokenizerAdapter(
                    tokenizer
                )
            else:
                self._shelf_tokenizer = None
        elif backend == BM25Backend.SHELF:
            # Use SHELF's tokenizer for consistency with TF-IDF
            from shelf.evaluate.text import DEFAULT_TOKENIZER

            self._shelf_tokenizer = DEFAULT_TOKENIZER
            self.tokenizer = lambda text: DEFAULT_TOKENIZER.tokenize(text)
        else:
            self.tokenizer = simple_tokenize
            self._shelf_tokenizer = None

        # rank_bm25 backend state
        self.bm25: BM25Okapi | None = None

        # SHELF backend state
        self.corpus_stats: CorpusStatistics | None = None

        # Common state
        self.corpus_ids: list[str] = []
        self.is_fitted = False
        self._corpus_size = 0

    @classmethod
    def from_config(
        cls,
        preset: str = "default",
        backend: str | BM25Backend = BM25Backend.RANK_BM25,
    ) -> "BM25Retriever":
        """Create retriever with preset configuration.

        Args:
            preset: Configuration preset name
                - "default": Standard BM25 parameters (k1=1.5, b=0.75)
                - "short_docs": For shorter documents (k1=1.2, b=0.5)
                - "long_docs": For longer documents (k1=2.0, b=0.75)
            backend: Backend to use ("rank_bm25" or "shelf")

        Returns:
            BM25Retriever instance
        """
        presets = {
            "default": {"k1": 1.5, "b": 0.75},
            "short_docs": {"k1": 1.2, "b": 0.5},
            "long_docs": {"k1": 2.0, "b": 0.75},
        }

        if preset not in presets:
            raise ValueError(
                f"Unknown preset: {preset}. Available: {list(presets.keys())}"
            )

        return cls(**presets[preset], backend=backend)

    def fit(
        self,
        corpus_texts: list[str],
        corpus_ids: list[str],
    ) -> "BM25Retriever":
        """Build BM25 index from corpus.

        Args:
            corpus_texts: List of document texts
            corpus_ids: List of document IDs (same order as texts)

        Returns:
            self for chaining
        """
        if len(corpus_texts) != len(corpus_ids):
            raise ValueError(
                f"Mismatch: {len(corpus_texts)} texts vs {len(corpus_ids)} IDs"
            )

        logger.info(
            f"Building BM25 index on {len(corpus_texts)} documents "
            f"(backend={self.backend.value})..."
        )

        if self.backend == BM25Backend.SHELF:
            self._fit_shelf(corpus_texts)
        else:
            self._fit_rank_bm25(corpus_texts)

        self.corpus_ids = list(corpus_ids)
        self.is_fitted = True
        self._corpus_size = len(corpus_texts)

        logger.info(f"BM25 index built: {self._corpus_size} documents")
        return self

    def _fit_rank_bm25(self, corpus_texts: list[str]) -> None:
        """Build BM25 index using rank_bm25 library."""
        # Tokenize corpus
        tokenized_corpus = [self.tokenizer(text) for text in corpus_texts]

        # Build BM25 index
        from rank_bm25 import BM25Okapi

        self.bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)

    def _fit_shelf(self, corpus_texts: list[str]) -> None:
        """Build BM25 index using SHELF's CorpusStatistics."""
        from shelf.evaluate.text import CorpusStatistics, Vocabulary

        # Create vocabulary with default settings
        vocabulary = Vocabulary(
            min_df=1,  # Keep all terms (filtering can happen at query time)
            max_df=1.0,  # No maximum df filtering
        )

        # Create corpus statistics with shared tokenizer
        self.corpus_stats = CorpusStatistics(
            vocabulary=vocabulary,
            tokenizer=self._shelf_tokenizer,
        )

        # Fit on corpus
        self.corpus_stats.fit(corpus_texts)

    def retrieve(
        self,
        query_texts: list[str],
        query_ids: list[str],
        top_k: int = 100,
        show_progress: bool = True,
    ) -> dict[str, list[str]]:
        """Retrieve top-k documents for each query.

        Args:
            query_texts: List of query texts
            query_ids: List of query IDs
            top_k: Number of documents to retrieve per query
            show_progress: Whether to show progress bar

        Returns:
            Dict mapping query_id to ranked list of corpus doc IDs

        Raises:
            ValueError: If retriever is not fitted
        """
        if not self.is_fitted:
            raise ValueError("BM25 retriever not fitted. Call fit() first.")

        if len(query_texts) != len(query_ids):
            raise ValueError(
                f"Mismatch: {len(query_texts)} texts vs {len(query_ids)} IDs"
            )

        if not query_texts:
            return {}

        # Warn if top_k exceeds corpus size
        if top_k > self._corpus_size:
            logger.warning(
                f"top_k ({top_k}) exceeds corpus size ({self._corpus_size}), "
                f"returning all documents"
            )

        if self.backend == BM25Backend.SHELF:
            return self._retrieve_shelf(query_texts, query_ids, top_k, show_progress)
        else:
            return self._retrieve_rank_bm25(
                query_texts, query_ids, top_k, show_progress
            )

    def _retrieve_rank_bm25(
        self,
        query_texts: list[str],
        query_ids: list[str],
        top_k: int,
        show_progress: bool,
    ) -> dict[str, list[str]]:
        """Retrieve using rank_bm25 backend."""
        if self.bm25 is None:
            raise ValueError("BM25 index not built.")

        results: dict[str, list[str]] = {}
        corpus_ids_array = np.array(self.corpus_ids)

        # Create iterator with optional progress bar
        pairs = zip(query_texts, query_ids)
        if show_progress:
            pairs = tqdm(pairs, desc="BM25 retrieval", total=len(query_texts))

        empty_query_count = 0
        for query_text, query_id in pairs:
            # Tokenize query
            tokenized_query = self.tokenizer(query_text)

            # Handle empty tokenized queries
            if not tokenized_query:
                empty_query_count += 1
                results[query_id] = []
                continue

            # Get BM25 scores for all documents
            scores = self.bm25.get_scores(tokenized_query)

            # Get top-k indices (descending score, stable sort for reproducibility)
            top_indices = np.argsort(scores, kind="stable")[::-1][:top_k]

            # Map to document IDs
            ranked_doc_ids = corpus_ids_array[top_indices].tolist()
            results[query_id] = ranked_doc_ids

        if empty_query_count > 0:
            logger.warning(
                f"{empty_query_count} queries had no valid tokens after tokenization "
                f"(returned empty results)"
            )

        return results

    def _retrieve_shelf(
        self,
        query_texts: list[str],
        query_ids: list[str],
        top_k: int,
        show_progress: bool,
    ) -> dict[str, list[str]]:
        """Retrieve using SHELF backend with optimized scoring.

        Uses CSC sparse matrices and np.argpartition for efficiency.
        """
        if self.corpus_stats is None:
            raise ValueError("Corpus statistics not built.")

        results: dict[str, list[str]] = {}
        corpus_ids_array = np.array(self.corpus_ids)

        # Tokenize all queries at once using SHELF tokenizer
        if self._shelf_tokenizer is not None:
            tokenized_queries = self._shelf_tokenizer.tokenize_batch(query_texts)
        else:
            tokenized_queries = [self.tokenizer(text) for text in query_texts]

        # Use batch scoring for efficiency
        empty_query_count = 0

        # Create iterator with optional progress bar
        iterator = zip(tokenized_queries, query_ids)
        if show_progress:
            iterator = tqdm(iterator, desc="BM25 retrieval", total=len(query_ids))

        for query_tokens, query_id in iterator:
            # Handle empty tokenized queries
            if not query_tokens:
                empty_query_count += 1
                results[query_id] = []
                continue

            # Use optimized top-k retrieval
            top_indices, _ = self.corpus_stats.get_bm25_top_k(
                query_tokens, k=top_k, k1=self.k1, b=self.b
            )

            # Map to document IDs
            ranked_doc_ids = corpus_ids_array[top_indices].tolist()
            results[query_id] = ranked_doc_ids

        if empty_query_count > 0:
            logger.warning(
                f"{empty_query_count} queries had no valid tokens after tokenization "
                f"(returned empty results)"
            )

        return results

    def get_scores(
        self,
        query_text: str,
    ) -> np.ndarray:
        """Get BM25 scores for a single query against all documents.

        Args:
            query_text: Query text

        Returns:
            Array of scores (one per corpus document)
        """
        if not self.is_fitted:
            raise ValueError("BM25 retriever not fitted. Call fit() first.")

        tokenized_query = self.tokenizer(query_text)

        if self.backend == BM25Backend.SHELF:
            if self.corpus_stats is None:
                raise ValueError("Corpus statistics not built.")
            return self.corpus_stats.get_bm25_scores(
                tokenized_query, k1=self.k1, b=self.b
            )
        else:
            if self.bm25 is None:
                raise ValueError("BM25 index not built.")
            return np.array(self.bm25.get_scores(tokenized_query))

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Pseudo-embedding for compatibility with TextEmbedder protocol.

        Note: BM25 doesn't produce traditional embeddings. This method returns
        the BM25 score vectors against the corpus for each input text.
        This is NOT the standard use case - prefer retrieve() for retrieval tasks.

        For retrieval evaluation, use BM25Retriever.retrieve() directly instead.

        Args:
            texts: List of texts to "encode"
            batch_size: Not used (kept for protocol compatibility)
            show_progress: Whether to show progress

        Returns:
            np.ndarray of shape (len(texts), corpus_size) with BM25 scores

        Raises:
            ValueError: If retriever is not fitted
        """
        if not self.is_fitted:
            raise ValueError(
                "BM25 retriever not fitted. Call fit() first.\n"
                "Note: For retrieval tasks, use retrieve() instead of encode()."
            )

        logger.warning(
            "BM25.encode() returns score vectors, not embeddings. "
            "For retrieval tasks, use BM25Retriever.retrieve() directly."
        )

        if self.backend == BM25Backend.SHELF and self.corpus_stats is not None:
            # Use batch scoring for SHELF backend
            if self._shelf_tokenizer is not None:
                tokenized = self._shelf_tokenizer.tokenize_batch(texts)
            else:
                tokenized = [self.tokenizer(text) for text in texts]

            scores = self.corpus_stats.get_bm25_scores_batch(
                tokenized, k1=self.k1, b=self.b
            )
            return scores.astype(np.float32)
        else:
            # Use sequential scoring for rank_bm25 backend
            if self.bm25 is None:
                raise ValueError("BM25 index not built.")

            scores = []
            iterator = texts
            if show_progress:
                iterator = tqdm(texts, desc="Computing BM25 scores")

            for text in iterator:
                tokenized = self.tokenizer(text)
                text_scores = self.bm25.get_scores(tokenized)
                scores.append(text_scores)

            return np.array(scores, dtype=np.float32)

    def reset(self) -> None:
        """Reset the retriever to unfitted state."""
        self.bm25 = None
        self.corpus_stats = None
        self.corpus_ids = []
        self.is_fitted = False
        self._corpus_size = 0

    @property
    def embedding_dim(self) -> int:
        """Return pseudo-embedding dimension (corpus size).

        Note: This is for protocol compatibility only. BM25 doesn't have
        a fixed embedding dimension - the "dimension" is the corpus size.
        """
        return self._corpus_size or 0

    @property
    def model_name(self) -> str:
        """Return the model name/identifier."""
        return f"bm25-{self.backend.value}-k1={self.k1}-b={self.b}"

    def __repr__(self) -> str:
        """String representation."""
        status = (
            f"fitted on {self._corpus_size} docs" if self.is_fitted else "not fitted"
        )
        return (
            f"BM25Retriever(backend={self.backend.value}, "
            f"k1={self.k1}, b={self.b}, {status})"
        )

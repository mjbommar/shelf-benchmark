"""Corpus statistics and term frequency matrices for SHELF evaluation.

This module provides efficient sparse matrix representations of text corpora
used by TF-IDF and BM25 methods.

Key components:
- CorpusStatistics: Unified corpus statistics (vocabulary, TF matrix, doc lengths)
- Term frequency matrix in scipy CSR format for memory efficiency
- Document length normalization for BM25

Example:
    from shelf.evaluate.text import CorpusStatistics, DEFAULT_TOKENIZER

    # Build corpus statistics
    corpus = CorpusStatistics()
    corpus.fit(documents, tokenizer=DEFAULT_TOKENIZER)

    # Access components
    tf_matrix = corpus.term_freq_matrix  # scipy CSR sparse matrix
    doc_lengths = corpus.document_lengths
    avg_length = corpus.avg_document_length

    # Transform new documents
    new_tf = corpus.transform(new_documents)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix

if TYPE_CHECKING:
    from shelf.evaluate.text.tokenizers import Tokenizer
    from shelf.evaluate.text.vocabulary import Vocabulary

logger = logging.getLogger(__name__)


class CorpusStatistics:
    """Corpus statistics for sparse text representations.

    Computes and stores:
    - Vocabulary with document frequencies
    - Sparse term frequency matrix (CSR format)
    - Document lengths (for BM25 normalization)
    - Average document length

    This class is the foundation for both TF-IDF and BM25:
    - TF-IDF: Uses tf_matrix × idf
    - BM25: Uses tf_matrix with saturation and length normalization

    The CSR (Compressed Sparse Row) format is optimal for:
    - Row slicing (get document vectors)
    - Matrix-vector products (similarity computation)
    - Memory efficiency (only stores non-zeros)

    Example:
        from shelf.evaluate.text import (
            CorpusStatistics,
            Vocabulary,
            DEFAULT_TOKENIZER,
            IdfFormula,
        )

        # Create corpus statistics
        corpus = CorpusStatistics(
            vocabulary=Vocabulary(min_df=2, max_df=0.95),
            tokenizer=DEFAULT_TOKENIZER,
        )

        # Fit on corpus
        corpus.fit(documents)

        # Get TF-IDF matrix
        idf = corpus.vocabulary.get_idf(IdfFormula.SMOOTH)
        tfidf_matrix = corpus.term_freq_matrix.multiply(idf)

        # Get BM25 components
        tf = corpus.term_freq_matrix
        doc_lens = corpus.document_lengths
        avg_len = corpus.avg_document_length

    Args:
        vocabulary: Vocabulary instance (will be fitted if not already)
        tokenizer: Tokenizer to use for text processing
    """

    def __init__(
        self,
        vocabulary: "Vocabulary | None" = None,
        tokenizer: "Tokenizer | None" = None,
    ):
        """Initialize corpus statistics.

        Args:
            vocabulary: Vocabulary instance (created with defaults if None)
            tokenizer: Tokenizer instance (uses DEFAULT_TOKENIZER if None)
        """
        # Import here to avoid circular imports
        from shelf.evaluate.text.tokenizers import DEFAULT_TOKENIZER
        from shelf.evaluate.text.vocabulary import Vocabulary

        self.vocabulary = vocabulary or Vocabulary()
        self.tokenizer = tokenizer or DEFAULT_TOKENIZER

        # Corpus state (populated by fit())
        self._term_freq_matrix: csr_matrix | None = None
        self._term_freq_matrix_csc: csc_matrix | None = (
            None  # CSC cache for BM25 column access
        )
        self._document_lengths: np.ndarray | None = None
        self._avg_document_length: float = 0.0
        self._num_documents: int = 0
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        """Whether corpus statistics have been computed."""
        return self._is_fitted

    @property
    def term_freq_matrix(self) -> csr_matrix:
        """Sparse term frequency matrix (docs × terms) in CSR format.

        CSR (Compressed Sparse Row) is optimal for:
        - Row slicing (get document vectors)
        - Matrix-vector products

        Returns:
            scipy CSR matrix of shape (n_docs, vocab_size)
        """
        if self._term_freq_matrix is None:
            raise ValueError("Corpus not fitted. Call fit() first.")
        return self._term_freq_matrix

    @property
    def term_freq_matrix_csc(self) -> csc_matrix:
        """Sparse term frequency matrix (docs × terms) in CSC format.

        CSC (Compressed Sparse Column) is optimal for:
        - Column slicing (get term vectors across all docs)
        - BM25 scoring where we need tf values for specific terms

        The CSC matrix is lazily created on first access for efficiency.

        Returns:
            scipy CSC matrix of shape (n_docs, vocab_size)
        """
        if self._term_freq_matrix is None:
            raise ValueError("Corpus not fitted. Call fit() first.")

        # Lazy conversion to CSC (only done once)
        if self._term_freq_matrix_csc is None:
            logger.debug(
                "Converting TF matrix to CSC format for efficient column access..."
            )
            self._term_freq_matrix_csc = self._term_freq_matrix.tocsc()

        return self._term_freq_matrix_csc

    @property
    def document_lengths(self) -> np.ndarray:
        """Document lengths in tokens.

        Returns:
            Array of shape (n_docs,) with token counts
        """
        if self._document_lengths is None:
            raise ValueError("Corpus not fitted. Call fit() first.")
        return self._document_lengths

    @property
    def avg_document_length(self) -> float:
        """Average document length in tokens."""
        return self._avg_document_length

    @property
    def num_documents(self) -> int:
        """Number of documents in corpus."""
        return self._num_documents

    @property
    def vocab_size(self) -> int:
        """Vocabulary size."""
        return len(self.vocabulary)

    def fit(
        self,
        texts: list[str],
        fit_vocabulary: bool = True,
    ) -> "CorpusStatistics":
        """Fit corpus statistics on documents.

        Args:
            texts: List of document texts
            fit_vocabulary: Whether to fit vocabulary (False to use pre-fitted)

        Returns:
            self for method chaining
        """
        logger.info(f"Fitting corpus statistics on {len(texts)} documents...")

        # Tokenize all documents
        logger.info("Tokenizing documents...")
        tokenized_docs = self.tokenizer.tokenize_batch(texts)

        # Compute document lengths
        self._document_lengths = np.array(
            [len(doc) for doc in tokenized_docs], dtype=np.float64
        )
        self._avg_document_length = float(self._document_lengths.mean())
        self._num_documents = len(texts)

        logger.info(
            f"Document lengths: min={self._document_lengths.min():.0f}, "
            f"max={self._document_lengths.max():.0f}, "
            f"avg={self._avg_document_length:.1f}"
        )

        # Fit vocabulary if needed
        if fit_vocabulary or not self.vocabulary.is_fitted:
            logger.info("Fitting vocabulary...")
            self.vocabulary.fit(tokenized_docs)

        # Build term frequency matrix
        logger.info("Building term frequency matrix...")
        self._term_freq_matrix = self._build_tf_matrix(tokenized_docs)

        # Invalidate CSC cache (will be lazily rebuilt on next access)
        self._term_freq_matrix_csc = None

        self._is_fitted = True

        logger.info(
            f"Corpus fitted: {self._num_documents} docs, "
            f"{self.vocab_size} terms, "
            f"matrix shape {self._term_freq_matrix.shape}, "
            f"nnz={self._term_freq_matrix.nnz}"
        )

        return self

    def transform(
        self,
        texts: list[str],
    ) -> csr_matrix:
        """Transform new texts to term frequency matrix.

        Uses the fitted vocabulary (unknown terms are ignored).

        Args:
            texts: List of document texts

        Returns:
            Sparse term frequency matrix (n_texts × vocab_size)
        """
        if not self._is_fitted:
            raise ValueError("Corpus not fitted. Call fit() first.")

        # Tokenize
        tokenized_docs = self.tokenizer.tokenize_batch(texts)

        # Build TF matrix using fitted vocabulary
        return self._build_tf_matrix(tokenized_docs)

    def fit_transform(
        self,
        texts: list[str],
    ) -> csr_matrix:
        """Fit and transform in one step.

        Args:
            texts: List of document texts

        Returns:
            Sparse term frequency matrix
        """
        self.fit(texts)
        # Use property accessor which validates non-None
        return self.term_freq_matrix

    def _build_tf_matrix(
        self,
        tokenized_docs: list[list[str]],
    ) -> csr_matrix:
        """Build sparse term frequency matrix.

        Args:
            tokenized_docs: List of tokenized documents

        Returns:
            CSR sparse matrix of shape (n_docs, vocab_size)
        """
        n_docs = len(tokenized_docs)
        vocab_size = len(self.vocabulary)

        # Build COO format data (row, col, data) then convert to CSR
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []

        for doc_idx, doc in enumerate(tokenized_docs):
            # Count term frequencies in this document
            term_counts: dict[int, int] = {}
            for token in doc:
                term_idx = self.vocabulary.get(token, -1)
                if term_idx >= 0:
                    term_counts[term_idx] = term_counts.get(term_idx, 0) + 1

            # Add to sparse matrix data
            for term_idx, count in term_counts.items():
                rows.append(doc_idx)
                cols.append(term_idx)
                data.append(float(count))

        # Create CSR matrix
        tf_matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(n_docs, vocab_size),
            dtype=np.float64,
        )

        return tf_matrix

    def get_tfidf_matrix(
        self,
        sublinear_tf: bool = True,
        normalize: bool = True,
    ) -> csr_matrix:
        """Compute TF-IDF matrix from corpus statistics.

        Args:
            sublinear_tf: Use log(1 + tf) instead of raw tf
            normalize: L2 normalize document vectors

        Returns:
            Sparse TF-IDF matrix
        """
        from shelf.evaluate.text.vocabulary import IdfFormula

        if not self._is_fitted or self._term_freq_matrix is None:
            raise ValueError("Corpus not fitted. Call fit() first.")

        # Get term frequencies (copy to avoid modifying original)
        tf = self._term_freq_matrix.copy()

        # Apply sublinear TF scaling
        if sublinear_tf:
            tf.data = np.log1p(tf.data)

        # Get smooth IDF
        idf = self.vocabulary.get_idf(IdfFormula.SMOOTH)

        # Multiply TF by IDF
        tfidf = tf.multiply(idf)

        # L2 normalize
        if normalize:
            from sklearn.preprocessing import normalize as sklearn_normalize

            tfidf = sklearn_normalize(tfidf, norm="l2")

        return tfidf

    def get_bm25_scores(
        self,
        query_tokens: list[str],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> np.ndarray:
        """Compute BM25 scores for a query against all documents.

        BM25 formula:
            score(D, Q) = Σ IDF(qi) × (f(qi, D) × (k1 + 1)) /
                          (f(qi, D) + k1 × (1 - b + b × |D|/avgdl))

        This method uses CSC format for efficient column access (26x faster
        than CSR for column operations based on benchmarks).

        Args:
            query_tokens: Tokenized query
            k1: Term frequency saturation parameter (default: 1.5)
            b: Length normalization parameter (default: 0.75)

        Returns:
            Array of BM25 scores, one per document
        """
        from shelf.evaluate.text.vocabulary import IdfFormula

        if not self._is_fitted or self._document_lengths is None:
            raise ValueError("Corpus not fitted. Call fit() first.")

        n_docs = self._num_documents

        # Get BM25 IDF
        idf = self.vocabulary.get_idf(IdfFormula.BM25)

        # Precompute length normalization factor (shared across all queries)
        doc_lengths = self._document_lengths
        len_norm = 1 - b + b * (doc_lengths / self._avg_document_length)

        # Initialize scores
        scores = np.zeros(n_docs, dtype=np.float64)

        # Use CSC format for efficient column access (26x faster than CSR)
        tf_csc = self.term_freq_matrix_csc

        # Score each query term
        for token in query_tokens:
            term_idx = self.vocabulary.get(token, -1)
            if term_idx < 0:
                continue  # Skip OOV terms

            # Get term frequency column efficiently using CSC
            # CSC column slicing is O(nnz in column) vs O(nnz in matrix) for CSR
            tf_col = tf_csc.getcol(term_idx).toarray().ravel()

            # BM25 term score with saturation
            term_idf = idf[term_idx]
            numerator = tf_col * (k1 + 1)
            denominator = tf_col + k1 * len_norm
            term_score = term_idf * (numerator / denominator)

            scores += term_score

        return scores

    def get_bm25_scores_batch(
        self,
        queries_tokens: list[list[str]],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> np.ndarray:
        """Compute BM25 scores for multiple queries against all documents.

        This is more efficient than calling get_bm25_scores() repeatedly
        because it precomputes shared values and reuses them.

        Args:
            queries_tokens: List of tokenized queries
            k1: Term frequency saturation parameter (default: 1.5)
            b: Length normalization parameter (default: 0.75)

        Returns:
            Array of shape (n_queries, n_docs) with BM25 scores
        """
        from shelf.evaluate.text.vocabulary import IdfFormula

        if not self._is_fitted or self._document_lengths is None:
            raise ValueError("Corpus not fitted. Call fit() first.")

        n_queries = len(queries_tokens)
        n_docs = self._num_documents

        # Get BM25 IDF (computed once, cached)
        idf = self.vocabulary.get_idf(IdfFormula.BM25)

        # Precompute length normalization factor (shared across all queries)
        doc_lengths = self._document_lengths
        len_norm = 1 - b + b * (doc_lengths / self._avg_document_length)

        # Use CSC format for efficient column access
        tf_csc = self.term_freq_matrix_csc

        # Initialize score matrix
        scores = np.zeros((n_queries, n_docs), dtype=np.float64)

        # Cache term frequency columns for terms that appear in multiple queries
        tf_cache: dict[int, np.ndarray] = {}

        for q_idx, query_tokens in enumerate(queries_tokens):
            for token in query_tokens:
                term_idx = self.vocabulary.get(token, -1)
                if term_idx < 0:
                    continue

                # Get or cache TF column
                if term_idx not in tf_cache:
                    tf_cache[term_idx] = tf_csc.getcol(term_idx).toarray().ravel()

                tf_col = tf_cache[term_idx]

                # BM25 term score
                term_idf = idf[term_idx]
                numerator = tf_col * (k1 + 1)
                denominator = tf_col + k1 * len_norm
                term_score = term_idf * (numerator / denominator)

                scores[q_idx] += term_score

        return scores

    def get_bm25_top_k(
        self,
        query_tokens: list[str],
        k: int,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get top-k documents for a query using BM25 scoring.

        Uses np.argpartition for O(n) top-k selection instead of O(n log n)
        full sort, which is 3x faster for typical k values.

        Args:
            query_tokens: Tokenized query
            k: Number of top documents to return
            k1: Term frequency saturation parameter (default: 1.5)
            b: Length normalization parameter (default: 0.75)

        Returns:
            Tuple of (indices, scores) for top-k documents, sorted by score descending
        """
        scores = self.get_bm25_scores(query_tokens, k1=k1, b=b)

        # Use argpartition for efficient top-k (O(n) vs O(n log n))
        k = min(k, len(scores))
        if k <= 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

        # Get indices of top-k (unsorted)
        top_k_unsorted = np.argpartition(-scores, k - 1)[:k]

        # Sort the top-k by score
        sorted_order = np.argsort(-scores[top_k_unsorted])
        top_k_indices = top_k_unsorted[sorted_order]
        top_k_scores = scores[top_k_indices]

        return top_k_indices, top_k_scores

    def get_document_lengths_for_transform(
        self,
        texts: list[str],
    ) -> np.ndarray:
        """Get document lengths for new texts.

        Args:
            texts: List of document texts

        Returns:
            Array of document lengths (token counts)
        """
        tokenized = self.tokenizer.tokenize_batch(texts)
        return np.array([len(doc) for doc in tokenized], dtype=np.float64)

    def __repr__(self) -> str:
        """String representation."""
        if self._is_fitted:
            return (
                f"CorpusStatistics(docs={self._num_documents}, "
                f"vocab={self.vocab_size}, "
                f"avg_len={self._avg_document_length:.1f})"
            )
        else:
            return "CorpusStatistics(not fitted)"

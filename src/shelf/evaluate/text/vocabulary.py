"""Vocabulary and IDF computation for SHELF evaluation.

This module provides unified vocabulary management with document frequency
statistics and multiple IDF formula implementations used by both TF-IDF
and BM25.

Key design:
- Single source of truth for document frequencies
- Multiple IDF formulas computed from same df values
- Consistent vocabulary filtering (min_df, max_df, max_features)

IDF Formulas:
- SMOOTH: log((N+1)/(df+1)) + 1  [sklearn default, prevents zeros]
- STANDARD: log(N/df) + 1  [classic formula]
- BM25: log((N-df+0.5)/(df+0.5))  [Robertson-Spärck Jones]
- PROBABILISTIC: log((N-df)/df)  [theoretical, can be negative]

Example:
    from shelf.evaluate.text import Vocabulary, IdfFormula

    vocab = Vocabulary(min_df=2, max_df=0.95, max_features=50000)
    vocab.fit(tokenized_documents)

    # Get IDF values for different formulas
    smooth_idf = vocab.get_idf(IdfFormula.SMOOTH)
    bm25_idf = vocab.get_idf(IdfFormula.BM25)

    # Same document frequencies, different IDF values
    print(f"Term 'machine': smooth={smooth_idf[vocab['machine']]:.3f}, "
          f"bm25={bm25_idf[vocab['machine']]:.3f}")
"""

from __future__ import annotations

import logging
from collections import Counter
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from shelf.evaluate.text.tokenizers import Tokenizer

logger = logging.getLogger(__name__)


class IdfFormula(Enum):
    """IDF formula variants.

    Different formulas have different properties:
    - SMOOTH: Never zero, good for general use (sklearn default)
    - STANDARD: Classic formula, can be zero for universal terms
    - BM25: Used in BM25 ranking, handles high-df terms differently
    - PROBABILISTIC: Theoretical formula, can be negative

    Attributes:
        SMOOTH: log((N+1)/(df+1)) + 1
        STANDARD: log(N/df) + 1
        BM25: log((N-df+0.5)/(df+0.5))
        PROBABILISTIC: log((N-df)/df)
    """

    SMOOTH = "smooth"
    STANDARD = "standard"
    BM25 = "bm25"
    PROBABILISTIC = "probabilistic"


class Vocabulary:
    """Vocabulary with document frequency statistics.

    Manages a vocabulary of terms with:
    - Term to index mapping
    - Document frequency for each term
    - Multiple IDF formula computations
    - Vocabulary filtering (min_df, max_df, max_features)

    The vocabulary is built from tokenized documents and provides
    consistent statistics used by both TF-IDF and BM25 methods.

    Example:
        vocab = Vocabulary(min_df=2, max_df=0.95, max_features=50000)

        # Fit on tokenized documents
        tokenized_docs = [["hello", "world"], ["world", "python"]]
        vocab.fit(tokenized_docs)

        # Access vocabulary
        print(f"Vocab size: {len(vocab)}")
        print(f"Index of 'world': {vocab['world']}")

        # Get IDF values
        idf = vocab.get_idf(IdfFormula.SMOOTH)
        print(f"IDF of 'world': {idf[vocab['world']]:.3f}")

    Args:
        min_df: Minimum document frequency (int for absolute, float for proportion)
        max_df: Maximum document frequency (int for absolute, float for proportion)
        max_features: Maximum vocabulary size (keeps top by document frequency)
    """

    def __init__(
        self,
        min_df: int | float = 1,
        max_df: int | float = 1.0,
        max_features: int | None = None,
    ):
        """Initialize vocabulary.

        Args:
            min_df: Minimum document frequency
                - int: Absolute count (term must appear in >= min_df docs)
                - float: Proportion (term must appear in >= min_df * N docs)
            max_df: Maximum document frequency
                - int: Absolute count (term must appear in <= max_df docs)
                - float: Proportion (term must appear in <= max_df * N docs)
            max_features: Maximum vocabulary size (None = no limit)
        """
        self.min_df = min_df
        self.max_df = max_df
        self.max_features = max_features

        # Vocabulary state (populated by fit())
        self._token_to_idx: dict[str, int] = {}
        self._idx_to_token: list[str] = []
        self._document_freq: np.ndarray | None = None
        self._num_documents: int = 0
        self._is_fitted: bool = False

        # Cached IDF values
        self._idf_cache: dict[IdfFormula, np.ndarray] = {}

    @property
    def is_fitted(self) -> bool:
        """Whether vocabulary has been fitted."""
        return self._is_fitted

    @property
    def num_documents(self) -> int:
        """Number of documents used to fit vocabulary."""
        return self._num_documents

    @property
    def document_freq(self) -> np.ndarray:
        """Document frequency array (one value per term in vocabulary)."""
        if self._document_freq is None:
            raise ValueError("Vocabulary not fitted. Call fit() first.")
        return self._document_freq

    def __len__(self) -> int:
        """Vocabulary size."""
        return len(self._token_to_idx)

    def __contains__(self, token: str) -> bool:
        """Check if token is in vocabulary."""
        return token in self._token_to_idx

    def __getitem__(self, token: str) -> int:
        """Get index for token.

        Args:
            token: Token string

        Returns:
            Index in vocabulary

        Raises:
            KeyError: If token not in vocabulary
        """
        return self._token_to_idx[token]

    def get(self, token: str, default: int = -1) -> int:
        """Get index for token with default.

        Args:
            token: Token string
            default: Value to return if token not found

        Returns:
            Index in vocabulary, or default if not found
        """
        return self._token_to_idx.get(token, default)

    def get_token(self, idx: int) -> str:
        """Get token for index.

        Args:
            idx: Index in vocabulary

        Returns:
            Token string

        Raises:
            IndexError: If index out of range
        """
        return self._idx_to_token[idx]

    def tokens(self) -> list[str]:
        """Get all tokens in vocabulary order."""
        return self._idx_to_token.copy()

    def fit(
        self,
        tokenized_docs: list[list[str]],
    ) -> "Vocabulary":
        """Build vocabulary from tokenized documents.

        Args:
            tokenized_docs: List of tokenized documents (list of token lists)

        Returns:
            self for method chaining
        """
        logger.info(f"Building vocabulary from {len(tokenized_docs)} documents...")

        n_docs = len(tokenized_docs)
        self._num_documents = n_docs

        # Count document frequency for each term
        df_counter: Counter[str] = Counter()
        for doc in tokenized_docs:
            # Count each term once per document
            unique_terms = set(doc)
            df_counter.update(unique_terms)

        # Compute threshold values
        min_df_abs = self._get_absolute_df(self.min_df, n_docs)
        max_df_abs = self._get_absolute_df(self.max_df, n_docs)

        logger.info(
            f"Filtering: min_df={min_df_abs}, max_df={max_df_abs}, "
            f"max_features={self.max_features}"
        )

        # Filter terms by document frequency
        filtered_terms = [
            (term, df)
            for term, df in df_counter.items()
            if min_df_abs <= df <= max_df_abs
        ]

        logger.info(
            f"Terms after df filtering: {len(filtered_terms)} "
            f"(from {len(df_counter)} total)"
        )

        # Sort by document frequency (descending) for max_features selection
        filtered_terms.sort(key=lambda x: (-x[1], x[0]))

        # Apply max_features limit
        if self.max_features is not None and len(filtered_terms) > self.max_features:
            filtered_terms = filtered_terms[: self.max_features]
            logger.info(f"Terms after max_features: {len(filtered_terms)}")

        # Sort alphabetically for deterministic ordering
        filtered_terms.sort(key=lambda x: x[0])

        # Build vocabulary
        self._token_to_idx = {term: idx for idx, (term, _) in enumerate(filtered_terms)}
        self._idx_to_token = [term for term, _ in filtered_terms]
        self._document_freq = np.array(
            [df for _, df in filtered_terms], dtype=np.float64
        )

        # Clear IDF cache (will be recomputed on demand)
        self._idf_cache.clear()

        self._is_fitted = True
        logger.info(f"Vocabulary built: {len(self)} terms")

        return self

    def fit_from_texts(
        self,
        texts: list[str],
        tokenizer: "Tokenizer",
    ) -> "Vocabulary":
        """Build vocabulary from raw texts using tokenizer.

        Convenience method that tokenizes texts before fitting.

        Args:
            texts: List of raw text strings
            tokenizer: Tokenizer to use

        Returns:
            self for method chaining
        """
        tokenized_docs = tokenizer.tokenize_batch(texts)
        return self.fit(tokenized_docs)

    def _get_absolute_df(self, df_value: int | float, n_docs: int) -> int:
        """Convert df value to absolute count.

        Args:
            df_value: Document frequency (int or float proportion)
            n_docs: Total number of documents

        Returns:
            Absolute document frequency count
        """
        if isinstance(df_value, float):
            if df_value < 0 or df_value > 1:
                raise ValueError(f"Float df value must be in [0, 1], got {df_value}")
            return int(df_value * n_docs)
        else:
            return df_value

    def get_idf(self, formula: IdfFormula = IdfFormula.SMOOTH) -> np.ndarray:
        """Compute IDF values using specified formula.

        Results are cached for efficiency.

        Args:
            formula: IDF formula to use

        Returns:
            Array of IDF values (one per term in vocabulary)

        Raises:
            ValueError: If vocabulary not fitted
        """
        if not self._is_fitted or self._document_freq is None:
            raise ValueError("Vocabulary not fitted. Call fit() first.")

        # Check cache
        if formula in self._idf_cache:
            return self._idf_cache[formula]

        # Compute IDF
        n = self._num_documents
        df = self._document_freq  # Guaranteed non-None after check above

        if formula == IdfFormula.SMOOTH:
            # sklearn default: log((N+1)/(df+1)) + 1
            idf = np.log((n + 1) / (df + 1)) + 1

        elif formula == IdfFormula.STANDARD:
            # Classic: log(N/df) + 1
            # Avoid division by zero
            idf = np.log(n / np.maximum(df, 1)) + 1

        elif formula == IdfFormula.BM25:
            # BM25: log((N-df+0.5)/(df+0.5))
            idf = np.log((n - df + 0.5) / (df + 0.5))

        elif formula == IdfFormula.PROBABILISTIC:
            # Probabilistic: log((N-df)/df)
            # Can be negative for common terms
            idf = np.log((n - df) / np.maximum(df, 1))

        else:
            raise ValueError(f"Unknown IDF formula: {formula}")

        # Cache and return
        self._idf_cache[formula] = idf
        return idf

    def transform_to_indices(
        self,
        tokenized_docs: list[list[str]],
    ) -> list[list[int]]:
        """Convert tokenized documents to vocabulary indices.

        Unknown tokens (not in vocabulary) are skipped.

        Args:
            tokenized_docs: List of tokenized documents

        Returns:
            List of index lists (one per document)
        """
        if not self._is_fitted:
            raise ValueError("Vocabulary not fitted. Call fit() first.")

        result = []
        for doc in tokenized_docs:
            indices = [
                self._token_to_idx[token]
                for token in doc
                if token in self._token_to_idx
            ]
            result.append(indices)

        return result

    def __repr__(self) -> str:
        """String representation."""
        if self._is_fitted:
            return (
                f"Vocabulary(size={len(self)}, num_docs={self._num_documents}, "
                f"min_df={self.min_df}, max_df={self.max_df})"
            )
        else:
            return (
                f"Vocabulary(not fitted, min_df={self.min_df}, "
                f"max_df={self.max_df}, max_features={self.max_features})"
            )

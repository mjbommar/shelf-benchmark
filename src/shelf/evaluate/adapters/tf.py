"""Term Frequency (TF) adapter for SHELF evaluation.

This adapter provides TF (raw term frequency) embeddings as a baseline.
Unlike TF-IDF, this does NOT apply inverse document frequency weighting,
making it a simpler bag-of-words baseline.

Two backends are supported:
1. sklearn (default): Uses sklearn's CountVectorizer + optional SVD
2. shelf: Uses SHELF's CorpusStatistics

Example:
    # Use sklearn backend (default)
    embedder = TfEmbedder(embedding_dim=256)

    # Use SHELF backend
    embedder = TfEmbedder(backend="shelf", embedding_dim=256)

    # Encode corpus (fits the model)
    corpus_embeddings = embedder.encode(corpus_texts)

    # Encode queries (uses fitted model)
    query_embeddings = embedder.encode(query_texts)
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

if TYPE_CHECKING:
    from scipy.sparse import csr_matrix

    from shelf.evaluate.text import CorpusStatistics, Tokenizer

logger = logging.getLogger(__name__)


class TfBackend(Enum):
    """Backend implementations for TF."""

    SKLEARN = "sklearn"  # sklearn CountVectorizer (default)
    SHELF = "shelf"  # SHELF's CorpusStatistics


class TfEmbedder:
    """Adapter for Term Frequency (TF) as an embedding model.

    This implements the TextEmbedder protocol with two backends:
    - sklearn (default): Uses sklearn's CountVectorizer
    - shelf: Uses SHELF's CorpusStatistics (shared with BM25/TF-IDF)

    TF is a simpler baseline than TF-IDF - it uses raw term counts
    without IDF weighting. This can be useful for understanding
    how much IDF contributes to performance.

    By default, uses TruncatedSVD to produce dense embeddings for
    compatibility with standard cosine similarity operations.

    The adapter is stateful - it must be fit on a corpus before encoding
    queries. The typical workflow for retrieval is:
    1. Encode corpus (fits the vectorizer and SVD)
    2. Encode queries (transforms only)

    Example:
        # Create embedder with sklearn backend (default)
        embedder = TfEmbedder(embedding_dim=256)

        # Create embedder with SHELF backend
        embedder = TfEmbedder(backend="shelf", embedding_dim=256)

        # For retrieval: encode corpus first (fits the model)
        corpus_embeddings = embedder.encode(corpus_texts)

        # Then encode queries (uses fitted model)
        query_embeddings = embedder.encode(query_texts)

        # Compute cosine similarity
        similarities = cosine_similarity(query_embeddings, corpus_embeddings)

    Args:
        embedding_dim: Target embedding dimension (via SVD). None for sparse output.
        max_features: Maximum vocabulary size (default: 50000)
        ngram_range: N-gram range for features (default: (1, 2) for unigrams+bigrams)
        min_df: Minimum document frequency (default: 2)
        max_df: Maximum document frequency (default: 0.95)
        sublinear_tf: Use log(1 + tf) instead of raw tf (default: False for TF)
        normalize_output: Whether to L2-normalize output embeddings (default: True)
        backend: Backend to use ("sklearn" or "shelf")

    Attributes:
        vectorizer: The underlying CountVectorizer (sklearn backend)
        corpus_stats: CorpusStatistics instance (shelf backend)
        svd: TruncatedSVD for dimensionality reduction (if embedding_dim set)
        is_fitted: Whether the vectorizer has been fit
    """

    def __init__(
        self,
        embedding_dim: int | None = 256,
        max_features: int = 50000,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int | float = 2,
        max_df: float = 0.95,
        sublinear_tf: bool = False,
        normalize_output: bool = True,
        backend: str | TfBackend = TfBackend.SKLEARN,
    ):
        """Initialize TF embedder.

        Args:
            embedding_dim: Target dimension for dense output (None for sparse)
            max_features: Maximum vocabulary size
            ngram_range: N-gram range (min_n, max_n)
            min_df: Minimum document frequency
            max_df: Maximum document frequency
            sublinear_tf: Use log(1 + tf) instead of raw tf
            normalize_output: L2-normalize output embeddings
            backend: Backend to use ("sklearn" or "shelf")
        """
        self._embedding_dim = embedding_dim
        self.normalize_output = normalize_output
        self.sublinear_tf = sublinear_tf

        # Parse backend
        if isinstance(backend, str):
            backend = TfBackend(backend)
        self.backend = backend

        # Store configuration for reset
        self._max_features = max_features
        self._ngram_range = ngram_range
        self._min_df = min_df
        self._max_df = max_df

        # sklearn backend state
        self.vectorizer: CountVectorizer | None = None
        if backend == TfBackend.SKLEARN:
            self.vectorizer = CountVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=min_df,
                max_df=max_df,
                dtype=np.float32,
            )

        # SHELF backend state
        self.corpus_stats: CorpusStatistics | None = None
        self._shelf_tokenizer: Tokenizer | None = None
        if backend == TfBackend.SHELF:
            from shelf.evaluate.text import DEFAULT_TOKENIZER

            self._shelf_tokenizer = DEFAULT_TOKENIZER

            # Warn if n-grams requested but SHELF backend only supports unigrams
            if ngram_range != (1, 1):
                logger.warning(
                    f"SHELF backend only supports unigrams (ngram_range=(1,1)), "
                    f"but ngram_range={ngram_range} was requested. "
                    f"N-gram features will be ignored. Use backend='sklearn' for n-gram support."
                )

        # Initialize SVD if using dense embeddings
        self.svd: TruncatedSVD | None = None
        if embedding_dim is not None:
            self.svd = TruncatedSVD(n_components=embedding_dim, random_state=42)

        self.is_fitted = False
        self._actual_dim: int | None = None

    @classmethod
    def from_config(
        cls,
        preset: str = "default",
        embedding_dim: int | None = 256,
        backend: str | TfBackend = TfBackend.SKLEARN,
    ) -> "TfEmbedder":
        """Create embedder with preset configuration.

        Args:
            preset: Configuration preset name
                - "default": Balanced settings (unigrams+bigrams, 50k features)
                - "simple": Unigrams only, 10k features
                - "comprehensive": Unigrams+bigrams+trigrams, 100k features
            embedding_dim: Target embedding dimension
            backend: Backend to use ("sklearn" or "shelf")

        Returns:
            TfEmbedder instance

        Example:
            embedder = TfEmbedder.from_config("simple", embedding_dim=128)
            embedder = TfEmbedder.from_config("default", backend="shelf")
        """
        presets = {
            "default": {
                "max_features": 50000,
                "ngram_range": (1, 2),
                "min_df": 2,
                "max_df": 0.95,
            },
            "simple": {
                "max_features": 10000,
                "ngram_range": (1, 1),
                "min_df": 3,
                "max_df": 0.9,
            },
            "comprehensive": {
                "max_features": 100000,
                "ngram_range": (1, 3),
                "min_df": 2,
                "max_df": 0.95,
            },
        }

        if preset not in presets:
            raise ValueError(
                f"Unknown preset: {preset}. Available: {list(presets.keys())}"
            )

        config = presets[preset]
        return cls(embedding_dim=embedding_dim, backend=backend, **config)

    def fit(self, texts: list[str]) -> "TfEmbedder":
        """Fit the vectorizer and SVD on texts.

        Args:
            texts: List of texts to fit on

        Returns:
            self for chaining
        """
        logger.info(
            f"Fitting TF on {len(texts)} texts (backend={self.backend.value})..."
        )

        if self.backend == TfBackend.SHELF:
            tf_matrix = self._fit_shelf(texts)
        else:
            tf_matrix = self._fit_sklearn(texts)

        # Apply sublinear scaling if requested
        if self.sublinear_tf:
            tf_matrix = tf_matrix.copy()
            tf_matrix.data = np.log1p(tf_matrix.data)

        # Fit SVD if using dense embeddings
        if self.svd is not None:
            logger.info(f"Fitting SVD with {self.svd.n_components} components...")
            self.svd.fit(tf_matrix)
            explained_var = self.svd.explained_variance_ratio_.sum()
            logger.info(f"SVD explained variance: {explained_var:.3f}")
            self._actual_dim = self.svd.n_components
        else:
            self._actual_dim = tf_matrix.shape[1]

        self.is_fitted = True
        return self

    def _fit_sklearn(self, texts: list[str]) -> csr_matrix:
        """Fit using sklearn CountVectorizer."""
        if self.vectorizer is None:
            raise ValueError("sklearn vectorizer not initialized")

        tf_matrix = self.vectorizer.fit_transform(texts)
        logger.info(
            f"TF vocabulary size: {len(self.vectorizer.vocabulary_)}, "
            f"matrix shape: {tf_matrix.shape}"
        )
        return tf_matrix

    def _fit_shelf(self, texts: list[str]) -> csr_matrix:
        """Fit using SHELF's CorpusStatistics."""
        from shelf.evaluate.text import CorpusStatistics, Vocabulary

        # Create vocabulary with configured settings
        vocabulary = Vocabulary(
            min_df=self._min_df,
            max_df=self._max_df,
            max_features=self._max_features,
        )

        # Create corpus statistics with shared tokenizer
        self.corpus_stats = CorpusStatistics(
            vocabulary=vocabulary,
            tokenizer=self._shelf_tokenizer,
        )

        # Fit and get TF matrix
        self.corpus_stats.fit(texts)
        tf_matrix = self.corpus_stats.term_freq_matrix

        logger.info(
            f"TF vocabulary size: {self.corpus_stats.vocab_size}, "
            f"matrix shape: {tf_matrix.shape}"
        )
        return tf_matrix

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        """Fit the vectorizer/SVD and transform texts in one step.

        More efficient than fit() followed by encode() for corpus encoding.

        Args:
            texts: List of texts to fit on and transform

        Returns:
            np.ndarray of shape (len(texts), embedding_dim)

        Raises:
            ValueError: If texts is empty or has fewer documents than embedding_dim
        """
        if not texts:
            raise ValueError("Cannot fit TF on empty corpus")

        if len(texts) < 2:
            raise ValueError(f"TF requires at least 2 documents, got {len(texts)}")

        logger.info(
            f"Fitting TF on {len(texts)} texts (backend={self.backend.value})..."
        )

        # Fit and transform TF
        if self.backend == TfBackend.SHELF:
            tf_matrix = self._fit_shelf(texts)
            vocab_size = self.corpus_stats.vocab_size if self.corpus_stats else 0
        else:
            tf_matrix = self._fit_sklearn(texts)
            vocab_size = len(self.vectorizer.vocabulary_) if self.vectorizer else 0

        # Apply sublinear scaling if requested
        if self.sublinear_tf:
            tf_matrix = tf_matrix.copy()
            tf_matrix.data = np.log1p(tf_matrix.data)

        # Fit and transform SVD if using dense embeddings
        if self.svd is not None:
            # Clamp n_components to valid range for SVD
            max_components = min(len(texts), vocab_size) - 1
            actual_components = min(self.svd.n_components, max_components)

            # Check BEFORE creating SVD to avoid ValueError
            if actual_components < 1:
                logger.warning(
                    f"Insufficient data for SVD (vocab_size={vocab_size}, "
                    f"n_docs={len(texts)}), returning raw TF vectors"
                )
                embeddings = tf_matrix.toarray()
                self._actual_dim = vocab_size if vocab_size > 0 else tf_matrix.shape[1]
                self.svd = None
            else:
                if actual_components < self.svd.n_components:
                    logger.warning(
                        f"Reducing SVD components from {self.svd.n_components} to "
                        f"{actual_components} (vocab_size={vocab_size}, n_docs={len(texts)})"
                    )
                    self.svd = TruncatedSVD(
                        n_components=actual_components, random_state=42
                    )

                logger.info(f"Fitting SVD with {actual_components} components...")
                embeddings = self.svd.fit_transform(tf_matrix)
                explained_var = self.svd.explained_variance_ratio_.sum()
                logger.info(f"SVD explained variance: {explained_var:.3f}")
                self._actual_dim = actual_components
        else:
            embeddings = tf_matrix.toarray()
            self._actual_dim = tf_matrix.shape[1]

        self.is_fitted = True

        # Normalize if requested
        if self.normalize_output:
            embeddings = normalize(embeddings, norm="l2")

        return embeddings.astype(np.float32)

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode texts to embeddings.

        If the vectorizer is not fitted, this will fit on the provided texts
        (useful for corpus encoding in retrieval tasks).

        Args:
            texts: List of strings to encode
            batch_size: Batch size (not used for TF, kept for protocol)
            show_progress: Whether to show progress (not used for TF)

        Returns:
            np.ndarray of shape (len(texts), embedding_dim)
        """
        if not self.is_fitted:
            return self.fit_transform(texts)

        # Transform texts to TF
        if self.backend == TfBackend.SHELF:
            tf_matrix = self._transform_shelf(texts)
        else:
            tf_matrix = self._transform_sklearn(texts)

        # Apply sublinear scaling if requested
        if self.sublinear_tf:
            tf_matrix = tf_matrix.copy()
            tf_matrix.data = np.log1p(tf_matrix.data)

        # Apply SVD if using dense embeddings
        if self.svd is not None:
            embeddings = self.svd.transform(tf_matrix)
        else:
            embeddings = tf_matrix.toarray()

        # Normalize if requested
        if self.normalize_output:
            embeddings = normalize(embeddings, norm="l2")

        return embeddings.astype(np.float32)

    def _transform_sklearn(self, texts: list[str]) -> csr_matrix:
        """Transform using sklearn CountVectorizer."""
        if self.vectorizer is None:
            raise ValueError("sklearn vectorizer not initialized")
        return self.vectorizer.transform(texts)

    def _transform_shelf(self, texts: list[str]) -> csr_matrix:
        """Transform using SHELF's CorpusStatistics."""
        if self.corpus_stats is None:
            raise ValueError("SHELF corpus statistics not initialized")
        return self.corpus_stats.transform(texts)

    def reset(self) -> None:
        """Reset the embedder to unfitted state."""
        if self.backend == TfBackend.SKLEARN:
            self.vectorizer = CountVectorizer(
                max_features=self._max_features,
                ngram_range=self._ngram_range,
                min_df=self._min_df,
                max_df=self._max_df,
                dtype=np.float32,
            )
        else:
            self.corpus_stats = None

        if self._embedding_dim is not None:
            self.svd = TruncatedSVD(n_components=self._embedding_dim, random_state=42)

        self.is_fitted = False
        self._actual_dim = None

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension."""
        if self._actual_dim is not None:
            return self._actual_dim
        if self._embedding_dim is not None:
            return self._embedding_dim
        return self._max_features

    @property
    def model_name(self) -> str:
        """Return the model name/identifier."""
        ngram_str = f"{self._ngram_range[0]}-{self._ngram_range[1]}gram"
        dim_str = f"dim{self._embedding_dim}" if self._embedding_dim else "sparse"
        backend_str = self.backend.value
        sublinear_str = "-sublinear" if self.sublinear_tf else ""
        return f"tf-{backend_str}-{ngram_str}-{dim_str}{sublinear_str}"

    def __repr__(self) -> str:
        """String representation."""
        status = "fitted" if self.is_fitted else "not fitted"
        return (
            f"TfEmbedder(backend={self.backend.value}, "
            f"dim={self.embedding_dim}, {status})"
        )

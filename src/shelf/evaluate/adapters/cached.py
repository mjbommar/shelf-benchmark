"""Cached embedder adapter.

Wraps precomputed embeddings to provide a TextEmbedder-compatible interface.
Used for efficient batch evaluation where the same documents are embedded
multiple times across different tasks.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class CachedEmbedder:
    """Embedder that returns precomputed embeddings from cache.

    This adapter wraps a dictionary of text→embedding mappings and
    provides a TextEmbedder-compatible interface. It's useful for:

    1. Batch evaluation: Embed all texts once, run multiple tasks
    2. Reproducibility: Save/load embeddings for exact reproduction
    3. Debugging: Inspect cached embeddings

    Example:
        # Embed all texts once
        base_embedder = SentenceTransformerEmbedder.from_pretrained("...")
        all_texts = [...]
        all_embeddings = base_embedder.encode(all_texts)

        # Create cache
        cache = {text: emb for text, emb in zip(all_texts, all_embeddings)}

        # Create cached embedder
        cached = CachedEmbedder(
            cache=cache,
            model_name=base_embedder.model_name,
            embedding_dim=base_embedder.embedding_dim,
        )

        # Use for evaluation (no re-encoding)
        result = evaluate("lcc_retrieval", model=cached)
    """

    def __init__(
        self,
        cache: dict[str, np.ndarray],
        model_name: str,
        embedding_dim: int,
    ):
        """Initialize cached embedder.

        Args:
            cache: Dictionary mapping text strings to embedding vectors
            model_name: Name of the model that generated the embeddings
            embedding_dim: Dimension of the embeddings
        """
        self._cache = cache
        self._model_name = model_name
        self._embedding_dim = embedding_dim
        self._hits = 0
        self._misses: list[str] = []

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model_name

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension."""
        return self._embedding_dim

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Return cached embeddings for texts.

        Args:
            texts: List of text strings to encode
            batch_size: Ignored (for interface compatibility)
            show_progress: Ignored (for interface compatibility)

        Returns:
            np.ndarray of shape (len(texts), embedding_dim)

        Raises:
            KeyError: If any text is not found in the cache
        """
        embeddings = []
        missing = []

        for text in texts:
            if text in self._cache:
                embeddings.append(self._cache[text])
                self._hits += 1
            else:
                missing.append(text)
                self._misses.append(text[:100])  # Truncate for tracking

        if missing:
            # Show first few missing texts for debugging
            sample = missing[:3]
            sample_preview = [t[:80] + "..." if len(t) > 80 else t for t in sample]
            raise KeyError(
                f"Missing {len(missing)} texts from cache. "
                f"Cache has {len(self._cache)} entries. "
                f"Sample missing texts: {sample_preview}"
            )

        return np.array(embeddings)

    def get_stats(self) -> dict[str, int]:
        """Return cache hit/miss statistics."""
        return {
            "cache_size": len(self._cache),
            "hits": self._hits,
            "misses": len(self._misses),
        }

    def reset_stats(self) -> None:
        """Reset hit/miss counters."""
        self._hits = 0
        self._misses = []

    def __len__(self) -> int:
        """Return number of cached embeddings."""
        return len(self._cache)

    def __contains__(self, text: str) -> bool:
        """Check if text is in cache."""
        return text in self._cache

    def __repr__(self) -> str:
        return (
            f"CachedEmbedder(model_name={self._model_name!r}, "
            f"cache_size={len(self._cache)}, "
            f"embedding_dim={self._embedding_dim})"
        )

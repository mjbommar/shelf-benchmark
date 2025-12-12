"""Sentence Transformers adapter for SHELF evaluation.

This adapter wraps sentence-transformers models to implement
the TextEmbedder protocol.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbedder:
    """Adapter for sentence-transformers models.

    This wraps a SentenceTransformer model to implement the TextEmbedder protocol.
    Embeddings are normalized by default for cosine similarity.

    Example:
        # From existing model
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embedder = SentenceTransformerEmbedder(model)

        # Or load directly
        embedder = SentenceTransformerEmbedder.from_pretrained("all-MiniLM-L6-v2")

        # Encode texts
        embeddings = embedder.encode(["Hello world", "Another text"])
        print(embeddings.shape)  # (2, 384)

    Attributes:
        model: The underlying SentenceTransformer model
        normalize: Whether to normalize embeddings (default: True)
        _model_name: Cached model name
    """

    def __init__(
        self,
        model: "SentenceTransformer",
        normalize: bool = True,
        model_name: str | None = None,
    ):
        """Initialize adapter.

        Args:
            model: SentenceTransformer model instance
            normalize: Whether to L2-normalize embeddings (recommended for cosine sim)
            model_name: Override model name (default: infer from model)
        """
        self.model = model
        self.normalize = normalize
        self._model_name = model_name

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        normalize: bool = True,
        device: str | None = None,
        **kwargs,
    ) -> "SentenceTransformerEmbedder":
        """Load model from HuggingFace or local path.

        Args:
            model_name_or_path: Model name on HuggingFace or local path
            normalize: Whether to normalize embeddings
            device: Device to use (default: auto-detect)
            **kwargs: Additional arguments for SentenceTransformer

        Returns:
            SentenceTransformerEmbedder instance

        Example:
            embedder = SentenceTransformerEmbedder.from_pretrained("all-MiniLM-L6-v2")
            embedder = SentenceTransformerEmbedder.from_pretrained(
                "sentence-transformers/all-mpnet-base-v2",
                device="cuda"
            )
        """
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name_or_path, device=device, **kwargs)
        return cls(model, normalize=normalize, model_name=model_name_or_path)

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode texts to embeddings.

        Args:
            texts: List of strings to encode
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar

        Returns:
            np.ndarray of shape (len(texts), embedding_dim)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )

        # Ensure we return numpy array (model.encode can return tensor)
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)

        return embeddings

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension."""
        dim = self.model.get_sentence_embedding_dimension()
        if dim is None:
            raise ValueError("Could not determine embedding dimension")
        return dim

    @property
    def model_name(self) -> str:
        """Return the model name/identifier."""
        if self._model_name is not None:
            return self._model_name

        # Try to get from model config
        try:
            # sentence-transformers stores this in different places
            if hasattr(self.model, "_model_card_vars"):
                return self.model._model_card_vars.get("model_name", "unknown")
            if hasattr(self.model, "model_card_data"):
                return getattr(self.model.model_card_data, "model_name", "unknown")
        except Exception:
            pass

        return "sentence-transformer"

    def __repr__(self) -> str:
        """String representation."""
        return f"SentenceTransformerEmbedder(model={self.model_name}, dim={self.embedding_dim})"

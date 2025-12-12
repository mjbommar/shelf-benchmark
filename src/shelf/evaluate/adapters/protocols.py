"""Protocol definitions for model adapters.

These protocols define the interface that models must implement
to be used with SHELF evaluation. Implementations can wrap any
framework (sentence-transformers, OpenAI, Anthropic, etc.).

Using Python's Protocol allows for structural subtyping - any class
that implements the required methods is considered compatible,
without explicit inheritance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np


@runtime_checkable
class TextEmbedder(Protocol):
    """Protocol for text embedding models.

    Used for: retrieval, clustering, pair classification (via similarity)

    Implementations should handle batching internally for efficiency.
    Embeddings should be normalized (unit length) for cosine similarity.

    Example implementations:
        - SentenceTransformerEmbedder (sentence-transformers)
        - OpenAIEmbedder (OpenAI API)
        - HuggingFaceEmbedder (transformers)

    Example usage:
        embedder = SentenceTransformerEmbedder.from_pretrained("all-MiniLM-L6-v2")
        embeddings = embedder.encode(["Hello world", "Another text"])
        assert embeddings.shape == (2, embedder.embedding_dim)
    """

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> "np.ndarray":
        """Encode texts to embeddings.

        Args:
            texts: List of strings to encode
            batch_size: Batch size for encoding (for efficiency)
            show_progress: Whether to show progress bar

        Returns:
            np.ndarray of shape (len(texts), embedding_dim)
            Embeddings should be normalized for cosine similarity.
        """
        ...

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension.

        Returns:
            Integer dimension of embeddings
        """
        ...

    @property
    def model_name(self) -> str:
        """Return the model name/identifier.

        Returns:
            String identifier for the model
        """
        ...


@runtime_checkable
class TextClassifier(Protocol):
    """Protocol for text classification models.

    Used for: single-label classification, multi-label classification

    Implementations should handle batching internally for efficiency.

    Example implementations:
        - TransformersClassifier (HuggingFace pipeline)
        - OpenAIClassifier (GPT-4 with structured output)
        - AnthropicClassifier (Claude with tool use)

    Example usage:
        classifier = TransformersClassifier.from_pretrained("model-name")
        predictions = classifier.predict(["Text 1", "Text 2"])
        assert len(predictions) == 2
    """

    def predict(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> list[str]:
        """Predict class labels for texts.

        Args:
            texts: List of strings to classify
            batch_size: Batch size for prediction

        Returns:
            List of predicted labels (one per text)
        """
        ...

    @property
    def model_name(self) -> str:
        """Return the model name/identifier."""
        ...


@runtime_checkable
class TextClassifierWithProba(Protocol):
    """Extended classifier protocol with probability predictions.

    Extends TextClassifier with predict_proba for confidence scores.
    """

    def predict(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> list[str]:
        """Predict class labels for texts."""
        ...

    def predict_proba(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> list[dict[str, float]]:
        """Predict class probabilities.

        Args:
            texts: List of strings to classify
            batch_size: Batch size for prediction

        Returns:
            List of {label: probability} dicts, one per text
        """
        ...

    @property
    def model_name(self) -> str:
        """Return the model name/identifier."""
        ...


@runtime_checkable
class PairClassifier(Protocol):
    """Protocol for document pair classification.

    Used for: same-LCC pairs, same-form pairs

    Implementations should handle batching internally for efficiency.

    Example usage:
        classifier = PairClassifier(...)
        predictions = classifier.predict_pairs([
            ("doc1 text", "doc2 text"),
            ("doc3 text", "doc4 text"),
        ])
        assert predictions == [1, 0]  # same, different
    """

    def predict_pairs(
        self,
        pairs: list[tuple[str, str]],
        batch_size: int = 32,
    ) -> list[int]:
        """Predict whether pairs are similar (1) or different (0).

        Args:
            pairs: List of (text_a, text_b) tuples
            batch_size: Batch size for prediction

        Returns:
            List of binary predictions (0 or 1)
        """
        ...

    @property
    def model_name(self) -> str:
        """Return the model name/identifier."""
        ...


@runtime_checkable
class MultiLabelClassifier(Protocol):
    """Protocol for multi-label classification.

    Used for: topic classification (multiple topics per document)

    Example usage:
        classifier = MultiLabelClassifier(...)
        predictions = classifier.predict_multilabel(["Text 1", "Text 2"])
        assert predictions == [["topic1", "topic2"], ["topic3"]]
    """

    def predict_multilabel(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> list[list[str]]:
        """Predict multiple labels per text.

        Args:
            texts: List of strings to classify
            batch_size: Batch size for prediction

        Returns:
            List of label lists (multiple labels per text)
        """
        ...

    @property
    def model_name(self) -> str:
        """Return the model name/identifier."""
        ...

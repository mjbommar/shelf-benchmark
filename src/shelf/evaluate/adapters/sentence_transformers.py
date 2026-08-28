"""Sentence Transformers adapter for SHELF evaluation.

This adapter wraps sentence-transformers models to implement
the TextEmbedder protocol.

Two properties of this adapter matter for benchmark validity and are therefore
explicit rather than implicit:

**Context budget.** Left to itself, every model truncates at its own
``max_seq_length`` -- 256 for all-MiniLM-L6-v2, 384 for all-mpnet-base-v2, 512
for the BGE/E5/GTE families. SHELF documents average ~1,000 tokens and 46%
exceed 512, so a model-default protocol silently gives long-context models an
advantage and charges short-context models for content they never saw, while
sparse baselines read the entire document. Passing ``max_seq_length`` pins every
model to the same budget so context length becomes a reported condition instead
of an invisible confound. :attr:`truncation_stats` records how much text was
actually lost.

**Model-card prompts.** E5 requires ``"query: "`` / ``"passage: "`` prefixes,
BGE requires a query instruction, and Instructor requires instruction pairs.
Running these models without their prefixes evaluates them outside their
documented usage. ``query_prompt`` and ``document_prompt`` supply them, and
:meth:`encode_queries` / :meth:`encode_documents` select the right one.

Backwards compatibility: :meth:`encode` keeps its original signature and
defaults to the document role, so existing evaluators are unaffected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

# Cap on texts tokenized for truncation accounting. Measuring every document in
# a 34k corpus costs more than the encode itself; a sample gives a rate that is
# accurate to well within a percentage point at a fraction of the cost.
_TRUNCATION_SAMPLE_LIMIT = 2000


@dataclass
class TruncationStats:
    """How much text a context budget actually discarded."""

    budget: int | None = None
    n_measured: int = 0
    n_truncated: int = 0
    total_tokens: int = 0
    truncated_tokens: int = 0
    max_tokens_seen: int = 0
    sampled: bool = False

    @property
    def truncation_rate(self) -> float:
        """Fraction of measured texts that exceeded the budget."""
        return self.n_truncated / self.n_measured if self.n_measured else 0.0

    @property
    def token_retention_rate(self) -> float:
        """Fraction of measured tokens the model actually read."""
        if not self.total_tokens:
            return 1.0
        return 1.0 - (self.truncated_tokens / self.total_tokens)

    def as_dict(self) -> dict[str, float | int | bool | None]:
        """Serialize for inclusion in a result record."""
        return {
            "budget": self.budget,
            "n_measured": self.n_measured,
            "n_truncated": self.n_truncated,
            "truncation_rate": round(self.truncation_rate, 6),
            "token_retention_rate": round(self.token_retention_rate, 6),
            "max_tokens_seen": self.max_tokens_seen,
            "sampled": self.sampled,
        }


@dataclass
class _PromptConfig:
    """Model-card prefixes applied per encoding role."""

    query: str = ""
    document: str = ""

    def for_role(self, role: str) -> str:
        return self.query if role == "query" else self.document


class SentenceTransformerEmbedder:
    """Adapter for sentence-transformers models.

    This wraps a SentenceTransformer model to implement the TextEmbedder protocol.
    Embeddings are normalized by default for cosine similarity.

    Example:
        # From existing model
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embedder = SentenceTransformerEmbedder(model)

        # Or load directly, pinned to a shared context budget with E5 prefixes
        embedder = SentenceTransformerEmbedder.from_pretrained(
            "intfloat/e5-base-v2",
            max_seq_length=512,
            query_prompt="query: ",
            document_prompt="passage: ",
        )

        embeddings = embedder.encode(["Hello world", "Another text"])
        embedder.truncation_stats.truncation_rate  # how much was cut

    Attributes:
        model: The underlying SentenceTransformer model
        normalize: Whether to normalize embeddings (default: True)
    """

    def __init__(
        self,
        model: SentenceTransformer,
        normalize: bool = True,
        model_name: str | None = None,
        max_seq_length: int | None = None,
        query_prompt: str = "",
        document_prompt: str = "",
        measure_truncation: bool = True,
    ):
        """Initialize adapter.

        Args:
            model: SentenceTransformer model instance
            normalize: Whether to L2-normalize embeddings (recommended for cosine sim)
            model_name: Override model name (default: infer from model)
            max_seq_length: Pin the context budget in tokens. None keeps the
                model's own default, which differs per model and makes
                cross-model comparison unsound on long documents.
            query_prompt: Prefix applied when encoding queries (model-card value).
            document_prompt: Prefix applied when encoding documents.
            measure_truncation: Whether to record truncation statistics.

        Raises:
            ValueError: If max_seq_length is not positive.
        """
        if max_seq_length is not None and max_seq_length <= 0:
            raise ValueError(f"max_seq_length must be positive, got {max_seq_length}")

        self.model = model
        self.normalize = normalize
        self._model_name = model_name
        self._prompts = _PromptConfig(query=query_prompt, document=document_prompt)
        self._measure_truncation = measure_truncation

        self._default_max_seq_length = getattr(model, "max_seq_length", None)
        self._requested_max_seq_length = max_seq_length
        if max_seq_length is not None:
            self.model.max_seq_length = self._clamp_to_architecture(max_seq_length)

        self.truncation_stats = TruncationStats(budget=self.max_seq_length)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        normalize: bool = True,
        device: str | None = None,
        max_seq_length: int | None = None,
        query_prompt: str = "",
        document_prompt: str = "",
        measure_truncation: bool = True,
        **kwargs,
    ) -> SentenceTransformerEmbedder:
        """Load model from HuggingFace or local path.

        Args:
            model_name_or_path: Model name on HuggingFace or local path
            normalize: Whether to normalize embeddings
            device: Device to use (default: auto-detect)
            max_seq_length: Pin the context budget in tokens
            query_prompt: Model-card query prefix
            document_prompt: Model-card document prefix
            measure_truncation: Whether to record truncation statistics
            **kwargs: Additional arguments for SentenceTransformer

        Returns:
            SentenceTransformerEmbedder instance
        """
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name_or_path, device=device, **kwargs)
        return cls(
            model,
            normalize=normalize,
            model_name=model_name_or_path,
            max_seq_length=max_seq_length,
            query_prompt=query_prompt,
            document_prompt=document_prompt,
            measure_truncation=measure_truncation,
        )

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode texts to embeddings using the document role.

        Signature is unchanged from the original adapter so existing evaluators
        continue to work. Use :meth:`encode_queries` where the query role is
        meant.

        Args:
            texts: List of strings to encode
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar

        Returns:
            np.ndarray of shape (len(texts), embedding_dim)
        """
        return self._encode_with_role(
            texts, role="document", batch_size=batch_size, show_progress=show_progress
        )

    def encode_documents(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode texts with the model-card document prefix."""
        return self._encode_with_role(
            texts, role="document", batch_size=batch_size, show_progress=show_progress
        )

    def encode_queries(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode texts with the model-card query prefix."""
        return self._encode_with_role(
            texts, role="query", batch_size=batch_size, show_progress=show_progress
        )

    def _encode_with_role(
        self,
        texts: list[str],
        role: str,
        batch_size: int,
        show_progress: bool,
    ) -> np.ndarray:
        """Apply the role prefix, record truncation, and encode."""
        prefix = self._prompts.for_role(role)
        prepared = [prefix + text for text in texts] if prefix else list(texts)

        if self._measure_truncation:
            self._record_truncation(prepared)

        embeddings = self.model.encode(
            prepared,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )

        # Ensure we return numpy array (model.encode can return tensor)
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)

        return embeddings

    def _record_truncation(self, texts: list[str]) -> None:
        """Accumulate truncation statistics over a sample of ``texts``."""
        budget = self.max_seq_length
        if budget is None or not texts:
            return

        tokenizer = getattr(self.model, "tokenizer", None)
        if tokenizer is None:
            return

        sampled = len(texts) > _TRUNCATION_SAMPLE_LIMIT
        if sampled:
            step = len(texts) / _TRUNCATION_SAMPLE_LIMIT
            sample = [texts[int(i * step)] for i in range(_TRUNCATION_SAMPLE_LIMIT)]
        else:
            sample = texts

        try:
            encoded = tokenizer(
                sample, add_special_tokens=True, truncation=False, padding=False
            )["input_ids"]
        except Exception:
            # Truncation accounting must never break an evaluation run.
            return

        stats = self.truncation_stats
        stats.budget = budget
        stats.sampled = stats.sampled or sampled
        for ids in encoded:
            length = len(ids)
            stats.n_measured += 1
            stats.total_tokens += length
            stats.max_tokens_seen = max(stats.max_tokens_seen, length)
            if length > budget:
                stats.n_truncated += 1
                stats.truncated_tokens += length - budget

    def reset_truncation_stats(self) -> None:
        """Clear accumulated truncation statistics."""
        self.truncation_stats = TruncationStats(budget=self.max_seq_length)

    # ------------------------------------------------------------------
    # Model characteristics
    # ------------------------------------------------------------------

    def _clamp_to_architecture(self, requested: int) -> int:
        """Clamp a requested budget to what the architecture can actually run.

        Setting ``max_seq_length`` above a model's positional-embedding table
        does not fail at configuration time -- it fails in the forward pass with
        a shape mismatch. all-MiniLM-L6-v2 caps at 512 positions, so asking it
        for a 2048-token budget crashes mid-encode.

        A shared-budget sweep (512 / 2048 / 8192) will always request more than
        some model can serve, so the budget is clamped and the shortfall is
        recorded via :attr:`budget_honored` rather than raising. Silently
        dropping short-context models from a long-context condition would be
        worse: it would report a sweep that never ran.
        """
        limit = self.architecture_limit
        if limit is None:
            return requested
        return min(requested, limit)

    @property
    def architecture_limit(self) -> int | None:
        """Maximum sequence length the architecture supports, if discoverable.

        Returns ``None`` for models with no fixed positional limit (e.g. the
        relative-position T5 encoders behind GTR), where no clamp applies.
        """
        window = self.context_window
        if window is None or window <= 0:
            return None
        return window

    @property
    def requested_max_seq_length(self) -> int | None:
        """The budget that was asked for, before architectural clamping."""
        return self._requested_max_seq_length

    @property
    def budget_honored(self) -> bool:
        """Whether the active budget matches what was requested.

        False means the model could not serve the requested context budget and
        was clamped. Results from that model are not comparable to models that
        honored the budget and must be reported as such.
        """
        if self._requested_max_seq_length is None:
            return True
        return self.max_seq_length == self._requested_max_seq_length

    @property
    def max_seq_length(self) -> int | None:
        """Return the active context budget in tokens."""
        return getattr(self.model, "max_seq_length", None)

    @property
    def default_max_seq_length(self) -> int | None:
        """Return the model's own default budget, before any override."""
        return self._default_max_seq_length

    @property
    def query_prompt(self) -> str:
        """Return the configured query prefix."""
        return self._prompts.query

    @property
    def document_prompt(self) -> str:
        """Return the configured document prefix."""
        return self._prompts.document

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

        # sentence-transformers stores this in different places
        try:
            if hasattr(self.model, "_model_card_vars"):
                return self.model._model_card_vars.get("model_name", "unknown")
            if hasattr(self.model, "model_card_data"):
                return getattr(self.model.model_card_data, "model_name", "unknown")
        except Exception:
            pass

        return "sentence-transformer"

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SentenceTransformerEmbedder(model={self.model_name}, "
            f"dim={self.embedding_dim}, max_seq_length={self.max_seq_length})"
        )

    @property
    def num_params_torch(self) -> int:
        """Count actual parameters via torch numel().

        Returns:
            Total number of trainable parameters in the model.
        """
        return sum(p.numel() for p in self.model.parameters())

    @property
    def hidden_size(self) -> int | None:
        """Get hidden size from model config.

        For sentence-transformers, the first module [0] is typically
        the Transformer module with an auto_model attribute.

        Returns:
            Hidden size dimension, or None if not accessible.
        """
        try:
            auto_model = self.model[0].auto_model
            return getattr(auto_model.config, "hidden_size", None)
        except (IndexError, AttributeError):
            return None

    @property
    def context_window(self) -> int | None:
        """Get max position embeddings (context window) from model config.

        Returns:
            Maximum sequence length the model can handle, or None if not accessible.
        """
        try:
            auto_model = self.model[0].auto_model
            return getattr(auto_model.config, "max_position_embeddings", None)
        except (IndexError, AttributeError):
            return None

    def get_model_info(self) -> dict[str, int | bool | None]:
        """Return all model info for efficiency metrics.

        Returns:
            Dict with parameter count, hidden size, context window, embedding
            dimension, and the full context-budget picture: the active budget,
            the model default, what was requested, and the architectural limit.
        """
        return {
            "num_params_torch": self.num_params_torch,
            "hidden_size": self.hidden_size,
            "context_window": self.context_window,
            "embedding_dim": self.embedding_dim,
            "max_seq_length": self.max_seq_length,
            "default_max_seq_length": self.default_max_seq_length,
            "requested_max_seq_length": self.requested_max_seq_length,
            "architecture_limit": self.architecture_limit,
        }

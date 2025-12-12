"""Tokenizer implementations for SHELF evaluation.

This module provides tokenization using the HuggingFace tokenizers library
for high-performance, consistent text processing across BM25 and TF-IDF.

The tokenizers library is written in Rust and provides:
- 10-100x faster tokenization than pure Python
- Batch processing with parallelization
- Serialization/deserialization for reproducibility

Example:
    from shelf.evaluate.text import WordTokenizer, DEFAULT_TOKENIZER

    # Use default tokenizer (recommended for consistency)
    tokens = DEFAULT_TOKENIZER.tokenize("Hello, world!")
    # ['hello', 'world']

    # Batch tokenization (much faster for many documents)
    all_tokens = DEFAULT_TOKENIZER.tokenize_batch(documents)

    # Create custom tokenizer
    tokenizer = WordTokenizer(min_frequency=5, lowercase=True)
    tokenizer.fit(corpus)
    tokens = tokenizer.tokenize("New document text")
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@runtime_checkable
class Tokenizer(Protocol):
    """Protocol for tokenizers.

    All tokenizers must implement:
    - tokenize(): Single text tokenization
    - tokenize_batch(): Batch tokenization for efficiency
    - is_fitted: Property indicating if tokenizer is ready
    """

    @property
    def is_fitted(self) -> bool:
        """Whether the tokenizer has been fitted/trained."""
        ...

    def tokenize(self, text: str) -> list[str]:
        """Tokenize a single text into tokens.

        Args:
            text: Input text string

        Returns:
            List of token strings
        """
        ...

    def tokenize_batch(self, texts: list[str]) -> list[list[str]]:
        """Tokenize multiple texts efficiently.

        Args:
            texts: List of input text strings

        Returns:
            List of token lists, one per input text
        """
        ...


class WordTokenizer:
    """Word-level tokenizer using HuggingFace tokenizers library.

    This tokenizer:
    - Lowercases text (configurable)
    - Splits on whitespace and punctuation
    - Removes accents/diacritics
    - Filters by minimum token length
    - Builds vocabulary from corpus (optional)

    The tokenizer can operate in two modes:
    1. Stateless mode (default): Tokenizes without vocabulary lookup
    2. Fitted mode: After fit(), tracks vocabulary and can filter rare terms

    For BM25/TF-IDF evaluation, stateless mode is typically sufficient
    since vocabulary filtering happens in the Vocabulary class.

    Example:
        # Stateless tokenization
        tokenizer = WordTokenizer()
        tokens = tokenizer.tokenize("Hello, world!")
        # ['hello', 'world']

        # With vocabulary fitting
        tokenizer = WordTokenizer(min_frequency=2)
        tokenizer.fit(corpus)
        tokens = tokenizer.tokenize("New text")  # OOV tokens → [UNK]

    Args:
        lowercase: Convert text to lowercase (default: True)
        min_token_length: Minimum token length to keep (default: 1)
        strip_accents: Remove accents/diacritics (default: True)
        min_frequency: Minimum frequency for vocabulary (only used if fit)
        max_vocab_size: Maximum vocabulary size (only used if fit)
    """

    # Regex pattern for word tokenization
    # Matches: word characters (letters, digits, underscore)
    _TOKEN_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)

    def __init__(
        self,
        lowercase: bool = True,
        min_token_length: int = 1,
        strip_accents: bool = True,
        min_frequency: int = 1,
        max_vocab_size: int | None = None,
    ):
        """Initialize word tokenizer.

        Args:
            lowercase: Convert to lowercase
            min_token_length: Minimum token length (shorter tokens dropped)
            strip_accents: Remove accents/diacritics
            min_frequency: Minimum term frequency for vocabulary (if fitting)
            max_vocab_size: Maximum vocabulary size (if fitting)
        """
        self.lowercase = lowercase
        self.min_token_length = min_token_length
        self.strip_accents = strip_accents
        self.min_frequency = min_frequency
        self.max_vocab_size = max_vocab_size

        # Vocabulary (populated by fit())
        self._vocab: dict[str, int] | None = None
        self._is_fitted = False

        # HuggingFace tokenizer (created lazily)
        self._hf_tokenizer = None

    @property
    def is_fitted(self) -> bool:
        """Whether the tokenizer has been fitted with vocabulary."""
        return self._is_fitted

    @property
    def vocab(self) -> dict[str, int] | None:
        """Vocabulary mapping token -> index (None if not fitted)."""
        return self._vocab

    @property
    def vocab_size(self) -> int:
        """Vocabulary size (0 if not fitted)."""
        return len(self._vocab) if self._vocab else 0

    def _normalize(self, text: str) -> str:
        """Apply text normalization.

        Args:
            text: Input text

        Returns:
            Normalized text
        """
        if self.lowercase:
            text = text.lower()

        if self.strip_accents:
            # Unicode normalization to decompose accents
            import unicodedata

            text = unicodedata.normalize("NFD", text)
            # Remove combining diacritical marks
            text = "".join(char for char in text if unicodedata.category(char) != "Mn")

        return text

    def _extract_tokens(self, text: str) -> list[str]:
        """Extract tokens from normalized text.

        Args:
            text: Normalized text

        Returns:
            List of tokens
        """
        # Find all word tokens
        tokens = self._TOKEN_PATTERN.findall(text)

        # Filter by minimum length
        if self.min_token_length > 1:
            tokens = [t for t in tokens if len(t) >= self.min_token_length]

        return tokens

    def tokenize(self, text: str) -> list[str]:
        """Tokenize a single text.

        Args:
            text: Input text string

        Returns:
            List of token strings
        """
        # Normalize
        normalized = self._normalize(text)

        # Extract tokens
        tokens = self._extract_tokens(normalized)

        return tokens

    def tokenize_batch(self, texts: list[str]) -> list[list[str]]:
        """Tokenize multiple texts efficiently.

        Uses HuggingFace tokenizers for batch processing when fitted,
        otherwise falls back to sequential processing.

        Args:
            texts: List of input text strings

        Returns:
            List of token lists
        """
        # If we have a fitted HF tokenizer, use it for batch processing
        if self._hf_tokenizer is not None:
            encodings = self._hf_tokenizer.encode_batch(texts)
            return [enc.tokens for enc in encodings]

        # Otherwise, process sequentially (still fast for reasonable corpus sizes)
        return [self.tokenize(text) for text in texts]

    def fit(self, texts: list[str]) -> "WordTokenizer":
        """Fit tokenizer on corpus to build vocabulary.

        This creates a HuggingFace tokenizer with vocabulary from the corpus.
        After fitting, tokenize_batch() uses the fast HF tokenizer.

        Args:
            texts: Corpus of texts to build vocabulary from

        Returns:
            self for method chaining
        """
        try:
            from tokenizers import Tokenizer as HFTokenizer
            from tokenizers import normalizers, pre_tokenizers
            from tokenizers.models import WordLevel
            from tokenizers.trainers import WordLevelTrainer
        except ImportError as e:
            raise ImportError(
                "HuggingFace tokenizers library required for vocabulary fitting. "
                "Install with: pip install tokenizers"
            ) from e

        logger.info(f"Fitting tokenizer on {len(texts)} documents...")

        # Create HF tokenizer with empty vocab (will be populated by trainer)
        hf_tokenizer = HFTokenizer(WordLevel(vocab={}, unk_token="[UNK]"))

        # Set up normalization
        normalizer_list = []
        if self.strip_accents:
            normalizer_list.append(normalizers.NFD())
            normalizer_list.append(normalizers.StripAccents())
        if self.lowercase:
            normalizer_list.append(normalizers.Lowercase())

        if normalizer_list:
            # Type stubs are incomplete - normalizer is settable at runtime
            hf_tokenizer.normalizer = normalizers.Sequence(normalizer_list)  # type: ignore[misc]

        # Set up pre-tokenization (whitespace splitting)
        # Type stubs are incomplete - pre_tokenizer is settable at runtime
        hf_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()  # type: ignore[misc]

        # Create trainer
        trainer_kwargs = {
            "special_tokens": ["[UNK]"],
            "min_frequency": self.min_frequency,
            "show_progress": True,
        }
        if self.max_vocab_size:
            trainer_kwargs["vocab_size"] = self.max_vocab_size

        trainer = WordLevelTrainer(**trainer_kwargs)

        # Train on corpus
        hf_tokenizer.train_from_iterator(texts, trainer=trainer)

        # Store tokenizer and vocabulary
        self._hf_tokenizer = hf_tokenizer
        self._vocab = hf_tokenizer.get_vocab()
        self._is_fitted = True

        logger.info(f"Vocabulary size: {len(self._vocab)}")

        return self

    def save(self, path: Path | str) -> None:
        """Save tokenizer to file.

        Args:
            path: Path to save tokenizer JSON

        Raises:
            ValueError: If tokenizer is not fitted
        """
        if self._hf_tokenizer is None:
            raise ValueError("Tokenizer must be fitted before saving")

        path = Path(path)
        self._hf_tokenizer.save(str(path))
        logger.info(f"Tokenizer saved to: {path}")

    @classmethod
    def load(cls, path: Path | str) -> "WordTokenizer":
        """Load tokenizer from file.

        Args:
            path: Path to tokenizer JSON

        Returns:
            Loaded WordTokenizer instance
        """
        from tokenizers import Tokenizer as HFTokenizer

        path = Path(path)
        hf_tokenizer = HFTokenizer.from_file(str(path))

        # Create instance and populate from loaded tokenizer
        instance = cls()
        instance._hf_tokenizer = hf_tokenizer
        instance._vocab = hf_tokenizer.get_vocab()
        instance._is_fitted = True

        logger.info(
            f"Tokenizer loaded from: {path} (vocab size: {len(instance._vocab)})"
        )

        return instance

    def __repr__(self) -> str:
        """String representation."""
        status = f"fitted, vocab={self.vocab_size}" if self._is_fitted else "not fitted"
        return (
            f"WordTokenizer(lowercase={self.lowercase}, "
            f"min_token_length={self.min_token_length}, {status})"
        )


# Default tokenizer instance for consistent tokenization across methods
# This uses stateless tokenization (no vocabulary fitting required)
DEFAULT_TOKENIZER = WordTokenizer(
    lowercase=True,
    min_token_length=1,
    strip_accents=True,
)

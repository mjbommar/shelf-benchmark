"""Text segmentation for SHELF evaluation.

This module provides sentence and paragraph segmentation using the nupunkt
library, enabling evaluation at different granularities:

- Document level: Entire document as a unit (default)
- Paragraph level: Split documents into paragraphs
- Sentence level: Split documents into sentences

Different granularities test different aspects of model capabilities:
- Document: Overall topic/theme understanding
- Paragraph: Coherent idea chunks, discourse structure
- Sentence: Fine-grained semantic similarity

Example:
    from shelf.evaluate.text import Segmenter, Granularity

    segmenter = Segmenter()

    # Split into sentences
    sentences = segmenter.segment(document, Granularity.SENTENCE)

    # Split into paragraphs
    paragraphs = segmenter.segment(document, Granularity.PARAGRAPH)

    # Batch processing
    all_segments = segmenter.segment_batch(documents, Granularity.SENTENCE)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Iterator

logger = logging.getLogger(__name__)


class Granularity(Enum):
    """Text granularity levels for segmentation.

    Attributes:
        DOCUMENT: No segmentation, treat entire text as one unit
        PARAGRAPH: Split on blank lines (double newlines)
        SENTENCE: Split into individual sentences using nupunkt
    """

    DOCUMENT = "document"
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"


@dataclass
class Segment:
    """A segment of text with metadata.

    Attributes:
        text: The segment text content
        source_id: ID of the source document
        segment_index: Index of this segment within the source
        start_char: Starting character position in source
        end_char: Ending character position in source
        granularity: The granularity level of this segment
    """

    text: str
    source_id: str
    segment_index: int
    start_char: int
    end_char: int
    granularity: Granularity

    @property
    def segment_id(self) -> str:
        """Unique identifier for this segment."""
        return f"{self.source_id}_{self.granularity.value}_{self.segment_index}"


class Segmenter:
    """Text segmenter supporting multiple granularities.

    Uses nupunkt for high-precision sentence boundary detection,
    optimized for professional/academic text.

    Example:
        segmenter = Segmenter()

        # Segment single document
        sentences = segmenter.segment_sentences("First sentence. Second sentence.")
        # ["First sentence.", "Second sentence."]

        paragraphs = segmenter.segment_paragraphs("Para 1.\\n\\nPara 2.")
        # ["Para 1.", "Para 2."]

        # Segment with metadata
        segments = segmenter.segment("Text here.", Granularity.SENTENCE, doc_id="doc1")
        # [Segment(text="Text here.", source_id="doc1", ...)]

    Args:
        min_sentence_length: Minimum characters for a valid sentence (default: 3)
        min_paragraph_length: Minimum characters for a valid paragraph (default: 10)
    """

    # Pattern for paragraph splitting (2+ newlines, optionally with whitespace)
    _PARAGRAPH_PATTERN = re.compile(r"\n\s*\n+")

    def __init__(
        self,
        min_sentence_length: int = 3,
        min_paragraph_length: int = 10,
    ):
        """Initialize segmenter.

        Args:
            min_sentence_length: Minimum sentence length in characters
            min_paragraph_length: Minimum paragraph length in characters
        """
        self.min_sentence_length = min_sentence_length
        self.min_paragraph_length = min_paragraph_length

    def segment_sentences(self, text: str) -> list[str]:
        """Segment text into sentences.

        Uses nupunkt for high-precision sentence boundary detection.

        Args:
            text: Input text

        Returns:
            List of sentence strings
        """
        try:
            from nupunkt import sent_tokenize
        except ImportError as e:
            raise ImportError(
                "nupunkt library required for sentence segmentation. "
                "Install with: pip install nupunkt"
            ) from e

        if not text or not text.strip():
            return []

        # Use nupunkt for sentence tokenization
        # sent_tokenize returns list[str] when return_confidence is False (default)
        raw_sentences: list[str] = sent_tokenize(text)  # type: ignore[assignment]

        # Filter by minimum length and strip whitespace
        sentences = [
            s.strip()
            for s in raw_sentences
            if s.strip() and len(s.strip()) >= self.min_sentence_length
        ]

        return sentences

    def segment_paragraphs(self, text: str) -> list[str]:
        """Segment text into paragraphs.

        Splits on blank lines (two or more newlines).

        Args:
            text: Input text

        Returns:
            List of paragraph strings
        """
        if not text or not text.strip():
            return []

        # Split on blank lines
        paragraphs = self._PARAGRAPH_PATTERN.split(text)

        # Filter by minimum length and strip whitespace
        paragraphs = [
            p.strip()
            for p in paragraphs
            if p.strip() and len(p.strip()) >= self.min_paragraph_length
        ]

        return paragraphs

    def segment(
        self,
        text: str,
        granularity: Granularity,
        source_id: str = "",
    ) -> list[Segment]:
        """Segment text at specified granularity with metadata.

        Args:
            text: Input text
            granularity: Segmentation granularity
            source_id: ID of the source document

        Returns:
            List of Segment objects with metadata
        """
        if granularity == Granularity.DOCUMENT:
            # No segmentation - return entire document
            return [
                Segment(
                    text=text.strip(),
                    source_id=source_id,
                    segment_index=0,
                    start_char=0,
                    end_char=len(text),
                    granularity=granularity,
                )
            ]

        elif granularity == Granularity.PARAGRAPH:
            paragraphs = self.segment_paragraphs(text)
            segments = []
            current_pos = 0

            for i, para in enumerate(paragraphs):
                # Find position in original text
                start = text.find(para, current_pos)
                if start == -1:
                    start = current_pos
                end = start + len(para)
                current_pos = end

                segments.append(
                    Segment(
                        text=para,
                        source_id=source_id,
                        segment_index=i,
                        start_char=start,
                        end_char=end,
                        granularity=granularity,
                    )
                )

            return segments

        elif granularity == Granularity.SENTENCE:
            sentences = self.segment_sentences(text)
            segments = []
            current_pos = 0

            for i, sent in enumerate(sentences):
                # Find position in original text
                start = text.find(sent, current_pos)
                if start == -1:
                    start = current_pos
                end = start + len(sent)
                current_pos = end

                segments.append(
                    Segment(
                        text=sent,
                        source_id=source_id,
                        segment_index=i,
                        start_char=start,
                        end_char=end,
                        granularity=granularity,
                    )
                )

            return segments

        else:
            raise ValueError(f"Unknown granularity: {granularity}")

    def segment_batch(
        self,
        texts: list[str],
        granularity: Granularity,
        source_ids: list[str] | None = None,
    ) -> list[list[Segment]]:
        """Segment multiple texts.

        Args:
            texts: List of input texts
            granularity: Segmentation granularity
            source_ids: Optional list of source document IDs

        Returns:
            List of segment lists, one per input text
        """
        if source_ids is None:
            source_ids = [f"doc_{i}" for i in range(len(texts))]

        if len(source_ids) != len(texts):
            raise ValueError(
                f"Length mismatch: {len(texts)} texts vs {len(source_ids)} source_ids"
            )

        return [
            self.segment(text, granularity, source_id)
            for text, source_id in zip(texts, source_ids)
        ]

    def iter_segments(
        self,
        texts: list[str],
        granularity: Granularity,
        source_ids: list[str] | None = None,
    ) -> Iterator[Segment]:
        """Iterate over segments from multiple texts.

        Memory-efficient alternative to segment_batch() for large corpora.

        Args:
            texts: List of input texts
            granularity: Segmentation granularity
            source_ids: Optional list of source document IDs

        Yields:
            Segment objects
        """
        if source_ids is None:
            source_ids = [f"doc_{i}" for i in range(len(texts))]

        for text, source_id in zip(texts, source_ids):
            for segment in self.segment(text, granularity, source_id):
                yield segment

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Segmenter(min_sentence_length={self.min_sentence_length}, "
            f"min_paragraph_length={self.min_paragraph_length})"
        )


# Default segmenter instance
DEFAULT_SEGMENTER = Segmenter()

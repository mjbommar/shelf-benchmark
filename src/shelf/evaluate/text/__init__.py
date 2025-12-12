"""Shared text processing components for SHELF evaluation.

This module provides consistent tokenization, vocabulary management,
corpus statistics, and text segmentation used by all sparse methods.

Design principles:
- Single tokenizer shared across BM25 and TF-IDF
- Unified vocabulary with document frequency statistics
- Multiple IDF formulas (smooth, BM25) from same df source
- Efficient batch processing using HuggingFace tokenizers
- Sentence/paragraph segmentation for multi-granularity evaluation

Components:
- Tokenizers: WordTokenizer for consistent word-level tokenization
- Vocabulary: Document frequencies, IDF computation with multiple formulas
- Segmentation: Sentence/paragraph splitting using nupunkt
- Corpus: Term frequency matrices, corpus statistics
"""

from shelf.evaluate.text.tokenizers import (
    Tokenizer,
    WordTokenizer,
    DEFAULT_TOKENIZER,
)
from shelf.evaluate.text.segmentation import (
    Granularity,
    Segment,
    Segmenter,
    DEFAULT_SEGMENTER,
)
from shelf.evaluate.text.vocabulary import (
    Vocabulary,
    IdfFormula,
)
from shelf.evaluate.text.corpus import (
    CorpusStatistics,
)

__all__ = [
    # Tokenizers
    "Tokenizer",
    "WordTokenizer",
    "DEFAULT_TOKENIZER",
    # Segmentation
    "Granularity",
    "Segment",
    "Segmenter",
    "DEFAULT_SEGMENTER",
    # Vocabulary
    "Vocabulary",
    "IdfFormula",
    # Corpus
    "CorpusStatistics",
]

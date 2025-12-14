"""Model adapters implementing SHELF protocols.

Adapters wrap common frameworks (sentence-transformers, OpenAI, sklearn, etc.)
to implement the TextEmbedder, TextClassifier, and PairClassifier protocols.

Available adapters:
- SentenceTransformerEmbedder: Wraps sentence-transformers models
- TfEmbedder: Term frequency with optional SVD (supports sklearn and shelf backends)
- TfidfEmbedder: TF-IDF with optional SVD (supports sklearn and shelf backends)
- BM25Retriever: BM25 for sparse retrieval (supports rank_bm25 and shelf backends)

Backend enums:
- BM25Backend: Backend options for BM25 retriever
- TfBackend: Backend options for TF embedder
- TfidfBackend: Backend options for TF-IDF embedder
"""

from shelf.evaluate.adapters.bm25 import BM25Backend, BM25Retriever
from shelf.evaluate.adapters.cached import CachedEmbedder
from shelf.evaluate.adapters.protocols import (
    PairClassifier,
    TextClassifier,
    TextEmbedder,
)
from shelf.evaluate.adapters.sentence_transformers import SentenceTransformerEmbedder
from shelf.evaluate.adapters.transformers_classifier import (
    TransformersSequenceClassifier,
)
from shelf.evaluate.adapters.tf import TfBackend, TfEmbedder
from shelf.evaluate.adapters.tfidf import TfidfBackend, TfidfEmbedder

__all__ = [
    # Protocols
    "TextEmbedder",
    "TextClassifier",
    "PairClassifier",
    # Embedding Adapters
    "SentenceTransformerEmbedder",
    "TransformersSequenceClassifier",
    "CachedEmbedder",
    "TfEmbedder",
    "TfBackend",
    "TfidfEmbedder",
    "TfidfBackend",
    # Retrieval Adapters
    "BM25Retriever",
    "BM25Backend",
]

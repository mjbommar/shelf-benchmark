# SHELF Evaluation Harness - Implementation Plan

> **Goal**: Build a clean, modular evaluation framework for RAG/search tasks
> **Primary users**: Practitioners using transformers, sentence-transformers, OpenAI, Anthropic
> **Design principle**: Prediction-file-first with optional protocol-based convenience adapters

---

## Progress Tracker

### Completed ✅

**Sprint 1: Foundation + Retrieval** (December 11, 2025)
- [x] `tasks.py` - TaskType, TaskSpec
- [x] `results.py` - EvaluationContext, EvaluationResult
- [x] `schemas.py` - Pydantic models for prediction validation
- [x] `metrics/retrieval.py` - NDCG, MRR, Recall@k, Precision@k, MAP@k
- [x] `evaluators/base.py` - TaskEvaluator ABC
- [x] `evaluators/retrieval.py` - RetrievalEvaluator
- [x] `adapters/protocols.py` - TextEmbedder, TextClassifier, PairClassifier protocols
- [x] `adapters/sentence_transformers.py` - SentenceTransformerEmbedder
- [x] `registry.py` - TASK_REGISTRY with retrieval tasks
- [x] `runner.py` - Main evaluate() function
- [x] Dataset uploaded to HuggingFace Hub (mjbommar/SHELF)

**First Baselines** (December 11, 2025)
- [x] all-MiniLM-L6-v2 on lcc_retrieval: NDCG@10=0.4551, MRR=0.6425
- [x] bert-base-uncased on lcc_retrieval: NDCG@10=0.3271, MRR=0.5530
- [x] Results documented in docs/tasks/retrieval.md and HF README

**Traditional Baselines** (December 11, 2025)
- [x] `adapters/tfidf.py` - TfidfEmbedder adapter (sklearn TfidfVectorizer + SVD)
- [x] `adapters/bm25.py` - BM25Retriever (rank-bm25)
- [x] TF-IDF on lcc_retrieval: NDCG@10=0.4566, MRR=0.6630
- [x] BM25 on lcc_retrieval: NDCG@10=0.4468, MRR=0.6778
- [x] Results documented in docs/ and HF README

**Classification Evaluator** (December 11, 2025)
- [x] `metrics/classification.py` - macro/micro F1, accuracy, per-class metrics
- [x] `evaluators/classification.py` - ClassificationEvaluator
- [x] Classification tasks in registry (lcc, category, register)
- [x] TF-IDF + LogisticRegression baselines:
  - lcc_classification: Accuracy=0.7770, Macro-F1=0.7781
  - lcgft_category_classification: Accuracy=0.7505, Macro-F1=0.7479
  - register_classification: Accuracy=0.6440, Macro-F1=0.5764
- [x] Sentence-Transformer + LogisticRegression baselines:
  - lcc_classification: Accuracy=0.7145, Macro-F1=0.7149
  - lcgft_category_classification: Accuracy=0.6215, Macro-F1=0.6147
  - register_classification: Accuracy=0.4295, Macro-F1=0.3654
- [x] Key finding: TF-IDF outperforms neural embeddings on classification!

**Clustering Evaluator** (December 12, 2025)
- [x] `metrics/clustering.py` - V-measure, NMI, ARI, homogeneity, completeness
- [x] `evaluators/clustering.py` - ClusteringEvaluator with k-means
- [x] Clustering tasks in registry (lcc_clustering, lcgft_clustering)
- [x] TF-IDF baselines:
  - lcc_clustering: V-measure=0.3774, NMI=0.3774, ARI=0.1810
  - lcgft_clustering: V-measure=0.0527, NMI=0.0527, ARI=0.0168
- [x] Sentence-Transformer baselines:
  - lcc_clustering: V-measure=0.4835, NMI=0.4835, ARI=0.3095
  - lcgft_clustering: V-measure=0.0571, NMI=0.0571, ARI=0.0222
- [x] Key finding: MiniLM outperforms TF-IDF on subject clustering, both struggle on genre/form

**Pair Classification Evaluator** (December 12, 2025)
- [x] `metrics/pair.py` - F1, accuracy, precision, recall, AUC-ROC, AP
- [x] `evaluators/pair.py` - PairClassificationEvaluator with cosine similarity
- [x] Pair tasks in registry (same_lcc_pairs, same_form_pairs)
- [x] TF-IDF baselines:
  - same_lcc_pairs: F1=0.667, Accuracy=0.500, AUC-ROC=0.660
  - same_form_pairs: F1=0.698, Accuracy=0.653, AUC-ROC=0.732
- [x] Sentence-Transformer baselines:
  - same_lcc_pairs: F1=0.728, Accuracy=0.703, AUC-ROC=0.776
  - same_form_pairs: F1=0.668, Accuracy=0.519, AUC-ROC=0.638
- [x] Key finding: MiniLM beats TF-IDF on subject pairs, TF-IDF beats MiniLM on form pairs!

### Next Up 🎯

**Priority 5: API Adapters**
- [ ] `adapters/openai.py` - OpenAIEmbedder (text-embedding-3-small/large)
- [ ] `adapters/anthropic.py` - AnthropicClassifier (for classification tasks)

**Priority 6: CLI Integration**
- [ ] Add `shelf evaluate` command to CLI
- [ ] Add `shelf list-tasks` command

---

## Design Philosophy

### Core Principles

1. **Decouple evaluation from inference**: Evaluators consume prediction files, not models
2. **Protocol-based adapters**: Optional layer for common frameworks (sentence-transformers, OpenAI)
3. **Rich results**: Per-class breakdowns, confidence intervals, confusion matrices—not just scores
4. **Reproducibility**: Every result includes full context (versions, seeds, checksums)
5. **Fail fast**: Validate predictions before evaluation with clear error messages

### Two Paths for Users

```
Path 1 (Quick): Model → Adapter → evaluate() → Results
Path 2 (Flexible): Model → [user code] → predictions.jsonl → evaluate() → Results
```

### Primary Use Cases (in priority order)

1. **Retrieval with sentence-transformers**: `model.encode()` → cosine similarity → NDCG@10
2. **Retrieval with OpenAI embeddings**: `client.embeddings.create()` → cosine similarity → NDCG@10
3. **Classification with transformers**: `pipeline("text-classification")` → Macro-F1
4. **Classification with LLM APIs**: Structured output → Macro-F1
5. **Clustering with any embedder**: embeddings → k-means → V-measure

---

## Phase 1: Foundation (Core Types)

### 1.1 Task Types and Specifications

**File**: `src/shelf/evaluate/tasks.py`

```python
from enum import Enum
from dataclasses import dataclass

class TaskType(Enum):
    CLASSIFICATION = "classification"
    MULTILABEL = "multilabel"
    RETRIEVAL = "retrieval"
    CLUSTERING = "clustering"
    PAIR_CLASSIFICATION = "pair_classification"

@dataclass(frozen=True)
class TaskSpec:
    """Immutable specification for an evaluation task."""
    name: str                           # "lcc_classification"
    task_type: TaskType                 # TaskType.CLASSIFICATION
    description: str                    # Human-readable description

    # Data fields
    text_field: str                     # "text" (concatenated title + body)
    label_field: str                    # "lcc_code"
    id_field: str                       # "id"

    # Label space (None = open vocabulary)
    label_space: tuple[str, ...] | None # ("A", "B", "C", ...) or None

    # Metrics
    primary_metric: str                 # "macro_f1"
    secondary_metrics: tuple[str, ...]  # ("micro_f1", "accuracy", ...)

    # Dataset config
    dataset_name: str                   # "mjbommar/SHELF"
    dataset_config: str | None          # "default" or "same_lcc_pairs"
    split: str                          # "test"
```

**Tasks to define**:
- `lcc_classification`: 21-class subject classification
- `lcgft_form_classification`: 133-class genre/form classification
- `lcgft_category_classification`: 14-class category classification
- `topic_classification`: Multi-label topic classification
- `audience_classification`: Target audience prediction
- `register_classification`: Writing style classification
- `lcc_retrieval`: Find documents with same LCC code
- `form_retrieval`: Find documents with same LCGFT form
- `topic_retrieval`: Find documents matching a topic query
- `lcc_clustering`: Cluster into 21 subject groups
- `lcgft_clustering`: Cluster into 14 genre categories
- `same_lcc_pairs`: Binary pair classification (same subject?)
- `same_form_pairs`: Binary pair classification (same genre?)

### 1.2 Evaluation Context (Reproducibility)

**File**: `src/shelf/evaluate/results.py`

```python
@dataclass
class EvaluationContext:
    """Full context for reproducible evaluation."""
    # Versions
    shelf_version: str
    python_version: str
    sklearn_version: str
    numpy_version: str

    # Data
    dataset_checksum: str               # MD5 of the split used
    prediction_file_checksum: str | None

    # Reproducibility
    random_seed: int

    # Environment
    platform: str                       # "Linux-6.5.0-x86_64"
    timestamp: str                      # ISO8601

    @classmethod
    def capture(cls) -> "EvaluationContext":
        """Capture current environment context."""
        ...
```

### 1.3 Evaluation Results

**File**: `src/shelf/evaluate/results.py`

```python
@dataclass
class EvaluationResult:
    """Complete evaluation results with context."""
    # Task info
    task: str
    task_type: str
    split: str

    # Primary result
    primary_metric: str
    primary_score: float

    # All metrics
    metrics: dict[str, float]

    # Detailed breakdowns (task-specific)
    per_class_metrics: dict[str, dict[str, float]] | None  # Classification
    confusion_matrix: list[list[int]] | None               # Classification
    per_query_metrics: dict[str, dict[str, float]] | None  # Retrieval

    # For debugging/analysis
    num_samples: int
    num_correct: int | None
    misclassified_ids: list[str] | None

    # Confidence intervals (bootstrap)
    confidence_intervals: dict[str, tuple[float, float]] | None

    # Full context
    context: EvaluationContext

    def to_dict(self) -> dict: ...
    def to_json(self, path: Path) -> None: ...
    @classmethod
    def from_json(cls, path: Path) -> "EvaluationResult": ...
```

### 1.4 Prediction Schemas (Pydantic)

**File**: `src/shelf/evaluate/schemas.py`

```python
from pydantic import BaseModel, field_validator

class ClassificationPrediction(BaseModel):
    """Single-label classification prediction."""
    id: str
    prediction: str
    confidence: float | None = None

class MultiLabelPrediction(BaseModel):
    """Multi-label classification prediction."""
    id: str
    predictions: list[str]
    confidences: list[float] | None = None

class RetrievalPrediction(BaseModel):
    """Retrieval results for a single query."""
    query_id: str
    ranked_doc_ids: list[str]
    scores: list[float] | None = None

class ClusteringPrediction(BaseModel):
    """Clustering assignment."""
    id: str
    cluster: int

class PairPrediction(BaseModel):
    """Pair classification prediction."""
    pair_id: str
    prediction: int  # 0 or 1
    confidence: float | None = None
```

**Validation functions**:
```python
def validate_predictions(
    predictions: list[dict],
    task_spec: TaskSpec,
    ground_truth_ids: set[str],
) -> ValidationResult:
    """Validate predictions against schema and ground truth.

    Checks:
    - All required fields present
    - All IDs exist in ground truth
    - No duplicate IDs
    - Predictions in valid label space (if defined)
    - No missing predictions

    Returns ValidationResult with errors and warnings.
    Raises ValidationError if critical errors found.
    """
```

---

## Phase 2: Metrics (Pure Functions)

### 2.1 Classification Metrics

**File**: `src/shelf/evaluate/metrics/classification.py`

```python
def macro_f1(y_true: list[str], y_pred: list[str], labels: list[str] | None = None) -> float:
    """Macro-averaged F1 score."""
    # Use sklearn with zero_division=0.0 explicitly

def micro_f1(y_true: list[str], y_pred: list[str]) -> float:
    """Micro-averaged F1 score."""

def weighted_f1(y_true: list[str], y_pred: list[str]) -> float:
    """Weighted F1 score."""

def accuracy(y_true: list[str], y_pred: list[str]) -> float:
    """Simple accuracy."""

def per_class_f1(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict[str, float]:
    """F1 score for each class."""

def per_class_metrics(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str]
) -> dict[str, dict[str, float]]:
    """Full metrics (precision, recall, f1, support) per class."""

def confusion_matrix(y_true: list[str], y_pred: list[str], labels: list[str]) -> list[list[int]]:
    """Confusion matrix as nested list."""

def compute_classification_metrics(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str] | None = None,
    compute_ci: bool = False,
    n_bootstrap: int = 1000,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Compute all classification metrics at once."""
```

### 2.2 Retrieval Metrics

**File**: `src/shelf/evaluate/metrics/retrieval.py`

```python
def ndcg_at_k(
    ranked_ids: list[str],
    relevant_ids: set[str],
    k: int = 10,
) -> float:
    """Normalized Discounted Cumulative Gain at k."""

def mrr(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    """Mean Reciprocal Rank (for single query)."""

def recall_at_k(
    ranked_ids: list[str],
    relevant_ids: set[str],
    k: int = 10,
) -> float:
    """Recall at k."""

def precision_at_k(
    ranked_ids: list[str],
    relevant_ids: set[str],
    k: int = 10,
) -> float:
    """Precision at k."""

def map_at_k(
    ranked_ids: list[str],
    relevant_ids: set[str],
    k: int = 10,
) -> float:
    """Mean Average Precision at k (for single query)."""

def compute_retrieval_metrics(
    results: dict[str, list[str]],           # query_id -> ranked_doc_ids
    relevance: dict[str, set[str]],          # query_id -> relevant_doc_ids
    k_values: list[int] = [1, 5, 10, 50, 100],
    compute_ci: bool = False,
    n_bootstrap: int = 1000,
) -> dict[str, float]:
    """Compute all retrieval metrics aggregated over queries.

    Returns: {
        "ndcg@10": 0.72,
        "mrr": 0.81,
        "recall@10": 0.65,
        "map@10": 0.68,
        ...
    }
    """
```

### 2.3 Clustering Metrics

**File**: `src/shelf/evaluate/metrics/clustering.py`

```python
def v_measure(labels_true: list[int], labels_pred: list[int]) -> float:
    """V-measure (harmonic mean of homogeneity and completeness)."""

def normalized_mutual_info(labels_true: list[int], labels_pred: list[int]) -> float:
    """Normalized Mutual Information."""

def adjusted_rand_index(labels_true: list[int], labels_pred: list[int]) -> float:
    """Adjusted Rand Index."""

def homogeneity(labels_true: list[int], labels_pred: list[int]) -> float:
    """Homogeneity score."""

def completeness(labels_true: list[int], labels_pred: list[int]) -> float:
    """Completeness score."""

def compute_clustering_metrics(
    labels_true: list[int] | list[str],
    labels_pred: list[int],
    compute_ci: bool = False,
) -> dict[str, float]:
    """Compute all clustering metrics."""
```

### 2.4 Bootstrap Confidence Intervals

**File**: `src/shelf/evaluate/metrics/bootstrap.py`

```python
def bootstrap_ci(
    metric_fn: Callable,
    *args,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_seed: int = 42,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for any metric."""
```

---

## Phase 3: Protocols & Adapters

### 3.1 Protocol Definitions

**File**: `src/shelf/evaluate/adapters/protocols.py`

```python
from typing import Protocol, runtime_checkable
import numpy as np

@runtime_checkable
class TextEmbedder(Protocol):
    """Protocol for text embedding models.

    Used for: retrieval, clustering, pair classification (via similarity)

    Implementations should handle batching internally.
    """

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
        ...

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension."""
        ...

@runtime_checkable
class TextClassifier(Protocol):
    """Protocol for text classification models.

    Used for: single-label classification, multi-label classification
    """

    def predict(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> list[str]:
        """Predict class labels for texts.

        Returns: List of predicted labels (one per text)
        """
        ...

    def predict_proba(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> list[dict[str, float]]:
        """Predict class probabilities.

        Returns: List of {label: probability} dicts
        """
        ...

@runtime_checkable
class PairClassifier(Protocol):
    """Protocol for document pair classification.

    Used for: same-LCC pairs, same-form pairs
    """

    def predict_pairs(
        self,
        pairs: list[tuple[str, str]],
        batch_size: int = 32,
    ) -> list[int]:
        """Predict whether pairs are similar (1) or different (0)."""
        ...
```

### 3.2 Sentence Transformers Adapter

**File**: `src/shelf/evaluate/adapters/sentence_transformers.py`

```python
class SentenceTransformerEmbedder:
    """Adapter for sentence-transformers models.

    Usage:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
        embedder = SentenceTransformerEmbedder(model)

        # Or directly:
        embedder = SentenceTransformerEmbedder.from_pretrained("all-MiniLM-L6-v2")
    """

    def __init__(
        self,
        model: "SentenceTransformer",
        normalize: bool = True,
    ):
        self.model = model
        self.normalize = normalize

    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs) -> "SentenceTransformerEmbedder":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name, **kwargs)
        return cls(model)

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize,
        )
        return embeddings

    @property
    def embedding_dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()
```

### 3.3 OpenAI Adapter

**File**: `src/shelf/evaluate/adapters/openai.py`

```python
class OpenAIEmbedder:
    """Adapter for OpenAI embedding API.

    Usage:
        embedder = OpenAIEmbedder(model="text-embedding-3-small")
        embeddings = embedder.encode(texts)

    Handles batching and rate limiting automatically.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        max_batch_size: int = 100,
        retry_on_rate_limit: bool = True,
    ):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_batch_size = max_batch_size
        self.retry_on_rate_limit = retry_on_rate_limit
        self._embedding_dim: int | None = None

    def encode(
        self,
        texts: list[str],
        batch_size: int = 100,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode texts using OpenAI API."""
        # Implement batching with rate limit handling
        ...

    @property
    def embedding_dim(self) -> int:
        if self._embedding_dim is None:
            # Compute from a sample embedding
            sample = self.encode(["test"])
            self._embedding_dim = sample.shape[1]
        return self._embedding_dim


class OpenAIClassifier:
    """Adapter for OpenAI chat models used for classification.

    Usage:
        classifier = OpenAIClassifier(
            model="gpt-4o-mini",
            label_space=["A", "B", "C", ...],
            system_prompt="Classify this document...",
        )
        predictions = classifier.predict(texts)

    Uses structured output for reliable predictions.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        label_space: list[str],
        system_prompt: str | None = None,
        api_key: str | None = None,
    ):
        ...

    def predict(self, texts: list[str], batch_size: int = 10) -> list[str]:
        """Predict labels using chat completions with structured output."""
        ...
```

### 3.4 Anthropic Adapter

**File**: `src/shelf/evaluate/adapters/anthropic.py`

```python
class AnthropicClassifier:
    """Adapter for Anthropic Claude models used for classification.

    Usage:
        classifier = AnthropicClassifier(
            model="claude-3-5-haiku-20241022",
            label_space=["A", "B", "C", ...],
        )
        predictions = classifier.predict(texts)
    """

    def __init__(
        self,
        model: str = "claude-3-5-haiku-20241022",
        label_space: list[str],
        system_prompt: str | None = None,
        api_key: str | None = None,
    ):
        from anthropic import Anthropic
        self.client = Anthropic(api_key=api_key)
        ...

    def predict(self, texts: list[str], batch_size: int = 10) -> list[str]:
        """Predict labels using Claude with tool use for structured output."""
        ...
```

### 3.5 Transformers Pipeline Adapter

**File**: `src/shelf/evaluate/adapters/transformers.py`

```python
class TransformersClassifier:
    """Adapter for HuggingFace transformers classification pipelines.

    Usage:
        # From pipeline
        pipe = pipeline("text-classification", model="...")
        classifier = TransformersClassifier(pipe, label_map={0: "A", 1: "B", ...})

        # From model name
        classifier = TransformersClassifier.from_pretrained("model-name")
    """

    def __init__(
        self,
        pipeline: "Pipeline",
        label_map: dict[int | str, str] | None = None,
    ):
        ...

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        label_map: dict | None = None,
        **kwargs
    ) -> "TransformersClassifier":
        from transformers import pipeline
        pipe = pipeline("text-classification", model=model_name, **kwargs)
        return cls(pipe, label_map)

    def predict(self, texts: list[str], batch_size: int = 32) -> list[str]:
        ...
```

---

## Phase 4: Evaluators

### 4.1 Base Evaluator

**File**: `src/shelf/evaluate/evaluators/base.py`

```python
from abc import ABC, abstractmethod

class TaskEvaluator(ABC):
    """Base class for all task evaluators.

    Evaluators are stateless - they take predictions and ground truth,
    and return results. They don't hold data.
    """

    def __init__(self, task_spec: TaskSpec):
        self.task_spec = task_spec

    @abstractmethod
    def evaluate(
        self,
        predictions: list[dict],
        ground_truth: list[dict],
        compute_ci: bool = False,
    ) -> EvaluationResult:
        """Evaluate predictions against ground truth.

        Args:
            predictions: List of prediction dicts (validated against schema)
            ground_truth: List of ground truth dicts from dataset
            compute_ci: Whether to compute bootstrap confidence intervals

        Returns:
            EvaluationResult with all metrics and context
        """
        ...

    def evaluate_from_file(
        self,
        predictions_path: Path,
        split: str = "test",
        compute_ci: bool = False,
    ) -> EvaluationResult:
        """Load predictions from file and evaluate."""
        predictions = self._load_predictions(predictions_path)
        ground_truth = self._load_ground_truth(split)
        return self.evaluate(predictions, ground_truth, compute_ci)

    def _load_predictions(self, path: Path) -> list[dict]:
        """Load and validate predictions from JSONL file."""
        ...

    def _load_ground_truth(self, split: str) -> list[dict]:
        """Load ground truth from HuggingFace dataset."""
        ...
```

### 4.2 Retrieval Evaluator (Priority: Highest)

**File**: `src/shelf/evaluate/evaluators/retrieval.py`

```python
class RetrievalEvaluator(TaskEvaluator):
    """Evaluator for retrieval tasks.

    Supports two modes:
    1. From predictions file (ranked doc IDs per query)
    2. From embedder (compute rankings via cosine similarity)
    """

    def __init__(
        self,
        task_spec: TaskSpec,
        k_values: list[int] = [1, 5, 10, 50, 100],
    ):
        super().__init__(task_spec)
        self.k_values = k_values

    def evaluate(
        self,
        predictions: list[dict],  # [{"query_id": ..., "ranked_doc_ids": [...]}]
        ground_truth: list[dict],
        compute_ci: bool = False,
    ) -> EvaluationResult:
        """Evaluate retrieval predictions."""
        ...

    def evaluate_embedder(
        self,
        embedder: TextEmbedder,
        split: str = "test",
        corpus_split: str | None = None,  # If None, use train+val as corpus
        compute_ci: bool = False,
        show_progress: bool = True,
    ) -> EvaluationResult:
        """Evaluate an embedder directly on retrieval task.

        1. Load queries from split
        2. Load corpus documents
        3. Encode all with embedder
        4. Compute cosine similarities
        5. Rank and evaluate
        """
        ...

    def _build_relevance_judgments(
        self,
        queries: list[dict],
        corpus: list[dict],
    ) -> dict[str, set[str]]:
        """Build query_id -> set of relevant doc_ids mapping."""
        # For LCC retrieval: relevant = same lcc_code
        # For form retrieval: relevant = same lcgft_form
        # For topic retrieval: relevant = contains query topic
        ...
```

### 4.3 Classification Evaluator

**File**: `src/shelf/evaluate/evaluators/classification.py`

```python
class ClassificationEvaluator(TaskEvaluator):
    """Evaluator for single-label classification tasks."""

    def evaluate(
        self,
        predictions: list[dict],  # [{"id": ..., "prediction": ...}]
        ground_truth: list[dict],
        compute_ci: bool = False,
    ) -> EvaluationResult:
        ...

    def evaluate_classifier(
        self,
        classifier: TextClassifier,
        split: str = "test",
        compute_ci: bool = False,
    ) -> EvaluationResult:
        """Evaluate a classifier directly."""
        ...


class MultiLabelEvaluator(TaskEvaluator):
    """Evaluator for multi-label classification (e.g., topics)."""

    def evaluate(
        self,
        predictions: list[dict],  # [{"id": ..., "predictions": [...]}]
        ground_truth: list[dict],
        compute_ci: bool = False,
    ) -> EvaluationResult:
        ...
```

### 4.4 Clustering Evaluator

**File**: `src/shelf/evaluate/evaluators/clustering.py`

```python
class ClusteringEvaluator(TaskEvaluator):
    """Evaluator for clustering tasks."""

    def __init__(
        self,
        task_spec: TaskSpec,
        n_clusters: int | None = None,  # If None, infer from label space
    ):
        super().__init__(task_spec)
        self.n_clusters = n_clusters or len(task_spec.label_space or [])

    def evaluate(
        self,
        predictions: list[dict],  # [{"id": ..., "cluster": ...}]
        ground_truth: list[dict],
        compute_ci: bool = False,
    ) -> EvaluationResult:
        ...

    def evaluate_embedder(
        self,
        embedder: TextEmbedder,
        split: str = "test",
        compute_ci: bool = False,
        random_seed: int = 42,
    ) -> EvaluationResult:
        """Evaluate embedder by running k-means on embeddings.

        1. Load documents from split
        2. Encode with embedder
        3. Run k-means clustering
        4. Evaluate against ground truth labels
        """
        ...
```

### 4.5 Pair Classification Evaluator

**File**: `src/shelf/evaluate/evaluators/pair.py`

```python
class PairClassificationEvaluator(TaskEvaluator):
    """Evaluator for pair classification tasks."""

    def evaluate(
        self,
        predictions: list[dict],  # [{"pair_id": ..., "prediction": 0|1}]
        ground_truth: list[dict],
        compute_ci: bool = False,
    ) -> EvaluationResult:
        ...

    def evaluate_embedder(
        self,
        embedder: TextEmbedder,
        split: str = "test",
        threshold: float = 0.5,  # Cosine similarity threshold
        compute_ci: bool = False,
    ) -> EvaluationResult:
        """Evaluate embedder on pair task via similarity threshold."""
        ...

    def evaluate_classifier(
        self,
        classifier: PairClassifier,
        split: str = "test",
        compute_ci: bool = False,
    ) -> EvaluationResult:
        """Evaluate a pair classifier directly."""
        ...
```

---

## Phase 5: Task Registry & Runner

### 5.1 Task Registry

**File**: `src/shelf/evaluate/registry.py`

```python
# All SHELF tasks with full specifications
TASK_REGISTRY: dict[str, TaskSpec] = {
    # Classification tasks
    "lcc_classification": TaskSpec(
        name="lcc_classification",
        task_type=TaskType.CLASSIFICATION,
        description="Classify documents into 21 Library of Congress subject classes",
        text_field="text",
        label_field="lcc_code",
        id_field="id",
        label_space=("A", "B", "C", "D", "E", "F", "G", "H", "J", "K",
                     "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "Z"),
        primary_metric="macro_f1",
        secondary_metrics=("micro_f1", "accuracy", "weighted_f1"),
        dataset_name="mjbommar/SHELF",
        dataset_config="default",
        split="test",
    ),

    # ... other tasks ...

    # Retrieval tasks
    "lcc_retrieval": TaskSpec(
        name="lcc_retrieval",
        task_type=TaskType.RETRIEVAL,
        description="Retrieve documents with the same LCC subject class",
        # ...
        primary_metric="ndcg@10",
        secondary_metrics=("mrr", "recall@10", "recall@100", "map@10"),
    ),

    # Clustering tasks
    "lcc_clustering": TaskSpec(
        name="lcc_clustering",
        task_type=TaskType.CLUSTERING,
        description="Cluster documents into 21 subject groups",
        # ...
        primary_metric="v_measure",
        secondary_metrics=("nmi", "ari"),
    ),

    # Pair classification tasks
    "same_lcc_pairs": TaskSpec(
        name="same_lcc_pairs",
        task_type=TaskType.PAIR_CLASSIFICATION,
        description="Predict whether two documents share the same LCC class",
        # ...
        primary_metric="f1",
        secondary_metrics=("accuracy", "mcc"),
        dataset_config="same_lcc_pairs",
    ),
}

def get_task(name: str) -> TaskSpec:
    """Get task specification by name."""
    if name not in TASK_REGISTRY:
        raise ValueError(f"Unknown task: {name}. Available: {list(TASK_REGISTRY.keys())}")
    return TASK_REGISTRY[name]

def list_tasks(task_type: TaskType | None = None) -> list[str]:
    """List available tasks, optionally filtered by type."""
    ...
```

### 5.2 Main Runner

**File**: `src/shelf/evaluate/runner.py`

```python
def evaluate(
    task: str,
    predictions: Path | list[dict] | None = None,
    model: TextEmbedder | TextClassifier | PairClassifier | None = None,
    split: str = "test",
    compute_ci: bool = False,
    output_path: Path | None = None,
) -> EvaluationResult:
    """Main evaluation entry point.

    Usage:
        # From predictions file
        result = evaluate("lcc_classification", predictions="preds.jsonl")

        # From model directly
        result = evaluate("lcc_retrieval", model=my_embedder)

        # Save results
        result = evaluate("lcc_clustering", model=embedder, output_path="results.json")

    Args:
        task: Task name from registry
        predictions: Path to predictions file or list of prediction dicts
        model: Model implementing appropriate protocol (alternative to predictions)
        split: Dataset split to evaluate on
        compute_ci: Whether to compute bootstrap confidence intervals
        output_path: Path to save results JSON

    Returns:
        EvaluationResult with all metrics and context
    """
    task_spec = get_task(task)
    evaluator = _get_evaluator(task_spec)

    if predictions is not None:
        result = evaluator.evaluate_from_file(predictions, split, compute_ci)
    elif model is not None:
        result = _evaluate_model(evaluator, model, split, compute_ci)
    else:
        raise ValueError("Must provide either predictions or model")

    if output_path:
        result.to_json(output_path)

    return result


def evaluate_all(
    model: TextEmbedder,
    tasks: list[str] | None = None,  # None = all compatible tasks
    split: str = "test",
    output_dir: Path | None = None,
) -> dict[str, EvaluationResult]:
    """Evaluate a model on multiple tasks.

    For embedders, runs all retrieval + clustering tasks.
    Returns dict of task_name -> EvaluationResult.
    """
    ...
```

### 5.3 CLI Integration

**File**: `src/shelf/cli.py` (extend existing)

```python
@app.command()
def evaluate(
    task: str = typer.Argument(..., help="Task name"),
    predictions: Path = typer.Option(None, "--predictions", "-p", help="Predictions file"),
    model: str = typer.Option(None, "--model", "-m", help="Model name (sentence-transformers)"),
    split: str = typer.Option("test", "--split", "-s", help="Dataset split"),
    output: Path = typer.Option(None, "--output", "-o", help="Output file for results"),
    detailed: bool = typer.Option(False, "--detailed", help="Show detailed metrics"),
    ci: bool = typer.Option(False, "--ci", help="Compute confidence intervals"),
):
    """Evaluate predictions or a model on a SHELF task.

    Examples:
        shelf evaluate lcc_classification -p predictions.jsonl
        shelf evaluate lcc_retrieval -m all-MiniLM-L6-v2
        shelf evaluate --list-tasks
    """
    ...

@app.command("list-tasks")
def list_tasks_cmd(
    task_type: str = typer.Option(None, "--type", "-t", help="Filter by task type"),
):
    """List available evaluation tasks."""
    ...
```

---

## Phase 6: Polish & Documentation

### 6.1 Error Handling

- Clear error messages with suggestions
- Validation errors show exactly what's wrong
- Network errors handled gracefully (API adapters)
- Version mismatch warnings

### 6.2 Logging

```python
# Structured logging throughout
import logging
logger = logging.getLogger("shelf.evaluate")

# Progress tracking for long operations
from tqdm import tqdm
```

### 6.3 Documentation

- Docstrings on all public functions
- Usage examples in docstrings
- README.md in src/shelf/evaluate/
- Examples directory with notebooks

### 6.4 Examples

**File**: `examples/evaluate_retrieval.py`

```python
"""Example: Evaluate sentence-transformers on LCC retrieval."""
from shelf.evaluate import evaluate
from shelf.evaluate.adapters import SentenceTransformerEmbedder

# Load model
embedder = SentenceTransformerEmbedder.from_pretrained("all-MiniLM-L6-v2")

# Evaluate
result = evaluate("lcc_retrieval", model=embedder)

# Print results
print(f"NDCG@10: {result.metrics['ndcg@10']:.4f}")
print(f"MRR: {result.metrics['mrr']:.4f}")
```

**File**: `examples/evaluate_openai.py`

```python
"""Example: Evaluate OpenAI embeddings on retrieval."""
from shelf.evaluate import evaluate
from shelf.evaluate.adapters import OpenAIEmbedder

embedder = OpenAIEmbedder(model="text-embedding-3-small")
result = evaluate("lcc_retrieval", model=embedder)
print(f"NDCG@10: {result.metrics['ndcg@10']:.4f}")
```

---

## Implementation Order

### Sprint 1: Foundation + Retrieval (Days 1-3)
1. `tasks.py` - TaskType, TaskSpec
2. `results.py` - EvaluationContext, EvaluationResult
3. `schemas.py` - Pydantic models for prediction validation
4. `metrics/retrieval.py` - NDCG, MRR, Recall@k
5. `evaluators/base.py` - TaskEvaluator ABC
6. `evaluators/retrieval.py` - RetrievalEvaluator
7. `adapters/protocols.py` - TextEmbedder protocol
8. `adapters/sentence_transformers.py` - First adapter

**Milestone**: Can evaluate sentence-transformers on LCC retrieval

### Sprint 2: Classification + Clustering (Days 4-5)
1. `metrics/classification.py` - F1, accuracy, per-class
2. `metrics/clustering.py` - V-measure, NMI, ARI
3. `evaluators/classification.py` - ClassificationEvaluator
4. `evaluators/clustering.py` - ClusteringEvaluator
5. `registry.py` - Full task registry

**Milestone**: Can evaluate embedders on classification + clustering

### Sprint 3: API Adapters (Days 6-7)
1. `adapters/openai.py` - OpenAIEmbedder, OpenAIClassifier
2. `adapters/anthropic.py` - AnthropicClassifier
3. `adapters/transformers.py` - TransformersClassifier

**Milestone**: Can evaluate OpenAI/Anthropic models

### Sprint 4: Runner + CLI (Days 8-9)
1. `runner.py` - Main evaluate() function
2. CLI integration in `cli.py`
3. `evaluators/pair.py` - PairClassificationEvaluator
4. `evaluators/multilabel.py` - MultiLabelEvaluator

**Milestone**: Full CLI working

### Sprint 5: Polish (Days 10+)
1. Error handling improvements
2. Logging and progress bars
3. Documentation
4. Examples
5. Tests

---

## File Checklist

```
src/shelf/evaluate/
├── __init__.py                 # [x] Public exports
├── tasks.py                    # [x] TaskType, TaskSpec
├── results.py                  # [x] EvaluationContext, EvaluationResult
├── schemas.py                  # [x] Pydantic validation models
├── registry.py                 # [x] TASK_REGISTRY (retrieval tasks)
├── runner.py                   # [x] Main evaluate() function
├── evaluators/
│   ├── __init__.py             # [x] Evaluator exports
│   ├── base.py                 # [x] TaskEvaluator ABC
│   ├── retrieval.py            # [x] RetrievalEvaluator
│   ├── classification.py       # [x] ClassificationEvaluator
│   ├── multilabel.py           # [ ] MultiLabelEvaluator
│   ├── clustering.py           # [x] ClusteringEvaluator
│   └── pair.py                 # [x] PairClassificationEvaluator
├── metrics/
│   ├── __init__.py             # [x] Metric exports
│   ├── classification.py       # [x] F1, accuracy, per-class
│   ├── retrieval.py            # [x] NDCG, MRR, Recall, Precision, MAP
│   ├── clustering.py           # [x] V-measure, NMI, ARI
│   ├── pair.py                 # [x] F1, Accuracy, AUC-ROC, AP
│   └── bootstrap.py            # [ ] Confidence intervals
└── adapters/
    ├── __init__.py             # [x] Adapter exports
    ├── protocols.py            # [x] TextEmbedder, TextClassifier, PairClassifier
    ├── sentence_transformers.py # [x] SentenceTransformerEmbedder
    ├── tfidf.py                # [x] TfidfEmbedder (SVD for dense output)
    ├── bm25.py                 # [x] BM25Retriever (rank-bm25)
    ├── openai.py               # [ ] OpenAIEmbedder, OpenAIClassifier
    ├── anthropic.py            # [ ] AnthropicClassifier
    └── transformers.py         # [ ] TransformersClassifier
```

---

## Success Criteria

1. **Retrieval evaluation works**: `evaluate("lcc_retrieval", model=embedder)` returns valid NDCG@10
2. **Prediction file flow works**: `evaluate("lcc_classification", predictions="preds.jsonl")`
3. **Multiple adapters work**: sentence-transformers, OpenAI, Anthropic
4. **CLI works**: `shelf evaluate lcc_retrieval -m all-MiniLM-L6-v2`
5. **Rich results**: Per-class metrics, confusion matrices, confidence intervals
6. **Full reproducibility**: Same inputs → identical results (with fixed seeds)

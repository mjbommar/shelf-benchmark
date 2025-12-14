# TF-IDF/TF+SVD Data Flow Diagrams

Visual representation of data flow for each task type in SHELF.

## Classification (Inductive Learning)

```
┌─────────────────────────────────────────────────────────────────┐
│ TRAINING PHASE                                                  │
└─────────────────────────────────────────────────────────────────┘

Train Documents (25,569 docs)
         ↓
    encode() [FIRST CALL - FITS]
         ↓
┌────────────────────────┐
│ TfidfVectorizer        │
│ .fit_transform()       │  ← Learns vocabulary from train
│                        │  ← Computes IDF from train
│ Vocabulary: 50,000     │
│ Features: TF-IDF       │
└────────────────────────┘
         ↓
    TF-IDF Matrix (25,569 × 50,000)
         ↓
┌────────────────────────┐
│ TruncatedSVD           │
│ .fit_transform()       │  ← Learns components from train
│                        │
│ Components: 256        │
└────────────────────────┘
         ↓
    Train Embeddings (25,569 × 256)
         ↓
┌────────────────────────┐
│ LogisticRegression     │
│ .fit()                 │  ← Trains on train embeddings
└────────────────────────┘


┌─────────────────────────────────────────────────────────────────┐
│ TEST PHASE                                                      │
└─────────────────────────────────────────────────────────────────┘

Test Documents (8,524 docs)
         ↓
    encode() [SUBSEQUENT CALL - TRANSFORMS ONLY]
         ↓
┌────────────────────────┐
│ TfidfVectorizer        │
│ .transform()           │  ← Uses FROZEN vocabulary from train
│                        │  ← OOV words ignored
│ Vocabulary: 50,000     │  ← SAME as train
└────────────────────────┘
         ↓
    TF-IDF Matrix (8,524 × 50,000)
         ↓
┌────────────────────────┐
│ TruncatedSVD           │
│ .transform()           │  ← Uses FROZEN components from train
│                        │
│ Components: 256        │  ← SAME as train
└────────────────────────┘
         ↓
    Test Embeddings (8,524 × 256)
         ↓
┌────────────────────────┐
│ LogisticRegression     │
│ .predict()             │  ← Predicts using trained model
└────────────────────────┘

✅ NO LEAKAGE: Vocabulary and SVD components learned from train only
```

## Retrieval (Corpus-Fitted, Query-Transformed)

```
┌─────────────────────────────────────────────────────────────────┐
│ CORPUS INDEXING (train + validation)                           │
└─────────────────────────────────────────────────────────────────┘

Corpus Documents (34,092 docs)
         ↓
    encode() [FIRST CALL - FITS]
         ↓
┌────────────────────────┐
│ TfidfVectorizer        │
│ .fit_transform()       │  ← Learns vocabulary from corpus
│                        │  ← Computes IDF from corpus
│ Vocabulary: 50,000     │
└────────────────────────┘
         ↓
    TF-IDF Matrix (34,092 × 50,000)
         ↓
┌────────────────────────┐
│ TruncatedSVD           │
│ .fit_transform()       │  ← Learns components from corpus
│                        │
│ Components: 256        │
└────────────────────────┘
         ↓
    Corpus Embeddings (34,092 × 256)  [STORED FOR RETRIEVAL]


┌─────────────────────────────────────────────────────────────────┐
│ QUERY MATCHING (test)                                          │
└─────────────────────────────────────────────────────────────────┘

Query Documents (8,524 docs)
         ↓
    encode() [SUBSEQUENT CALL - TRANSFORMS ONLY]
         ↓
┌────────────────────────┐
│ TfidfVectorizer        │
│ .transform()           │  ← Uses FROZEN corpus vocabulary
│                        │  ← Query-only words ignored
│ Vocabulary: 50,000     │  ← SAME as corpus
└────────────────────────┘
         ↓
    TF-IDF Matrix (8,524 × 50,000)
         ↓
┌────────────────────────┐
│ TruncatedSVD           │
│ .transform()           │  ← Uses FROZEN corpus components
│                        │
│ Components: 256        │  ← SAME as corpus
└────────────────────────┘
         ↓
    Query Embeddings (8,524 × 256)
         ↓
┌────────────────────────────────────────────┐
│ cosine_similarity(query, corpus)          │
│                                            │
│ Returns: (8,524 × 34,092) similarity       │
└────────────────────────────────────────────┘
         ↓
    Ranked Lists → Metrics (NDCG@10, Recall@100, etc.)

✅ NO LEAKAGE: Standard IR protocol (corpus-fitted, query-transformed)
```

## Clustering (Transductive Learning)

```
┌─────────────────────────────────────────────────────────────────┐
│ CLUSTERING (test split only)                                   │
└─────────────────────────────────────────────────────────────────┘

Test Documents (8,524 docs)
         ↓
    encode() [FIRST CALL - FITS]
         ↓
┌────────────────────────┐
│ TfidfVectorizer        │
│ .fit_transform()       │  ← Learns vocabulary from test split
│                        │  ← Computes IDF from test split
│ Vocabulary: 50,000     │
└────────────────────────┘
         ↓
    TF-IDF Matrix (8,524 × 50,000)
         ↓
┌────────────────────────┐
│ TruncatedSVD           │
│ .fit_transform()       │  ← Learns components from test split
│                        │
│ Components: 256        │
└────────────────────────┘
         ↓
    Embeddings (8,524 × 256)
         ↓
┌────────────────────────┐
│ KMeans                 │
│ .fit_predict()         │  ← Clusters embeddings (unsupervised)
│                        │  ← Ground truth labels NOT used
│ n_clusters: 21         │
└────────────────────────┘
         ↓
    Predicted Clusters (8,524 labels)
         ↓
┌────────────────────────────────────────────┐
│ Compare with Ground Truth                  │
│                                            │
│ Metrics: AMI, NMI, ARI                     │
│ (labels used only for evaluation)          │
└────────────────────────────────────────────┘

⚠️ TRANSDUCTIVE: Embeddings fitted on test split
✅ ACCEPTABLE: Standard protocol for clustering evaluation
✅ NO LABEL LEAKAGE: Labels used only for metrics, not clustering
```

## Key Differences

| Task           | Vocabulary Source      | SVD Source             | Protocol      | Leakage? |
|----------------|------------------------|------------------------|---------------|----------|
| Classification | Train split            | Train embeddings       | Inductive     | ✅ No    |
| Retrieval      | Corpus (train+val)     | Corpus embeddings      | Corpus-fitted | ✅ No    |
| Clustering     | Test split             | Test embeddings        | Transductive  | ⚠️ N/A   |

## Why Clustering is Different

**Inductive Learning** (Classification, Retrieval):
- Learn from labeled/indexed data
- Generalize to unseen data
- Test data must be held out during training

**Transductive Learning** (Clustering):
- Algorithm sees all data points (unlabeled)
- No generalization to unseen data
- Evaluates quality of grouping, not prediction

**Analogy**:
- **Classification**: "Train a model to recognize cats, then test on new images"
- **Clustering**: "Here are 100 animals, group them by species" (must see all animals to group them)

## Implementation Detail: State Management

The `TfidfEmbedder.is_fitted` flag ensures proper behavior:

```python
class TfidfEmbedder:
    def encode(self, texts):
        if not self.is_fitted:
            # First call: fit + transform
            embeddings = self.fit_transform(texts)
            self.is_fitted = True
            return embeddings
        else:
            # Subsequent calls: transform only
            return self.transform(texts)
```

**For Classification**:
1. `encode(train_texts)` → fits, returns train embeddings
2. `encode(test_texts)` → transforms, returns test embeddings

**For Retrieval**:
1. `encode(corpus_texts)` → fits, returns corpus embeddings
2. `encode(query_texts)` → transforms, returns query embeddings

**For Clustering**:
1. `encode(test_texts)` → fits, returns embeddings (all in one call)

## Conclusion

The data flow diagrams clearly show:
- **Classification and Retrieval**: Proper train/test separation maintained
- **Clustering**: Transductive by design, following standard protocols
- **No train/test leakage** that would artificially inflate performance

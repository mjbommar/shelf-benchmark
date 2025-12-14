# Ablation Study Suggestions: Strengthening Fairness Claims

## Overview

While our baseline comparison is fair (see `analysis.md`), ablation studies can strengthen the paper by isolating the contribution of specific design choices. This document proposes targeted ablations to address potential reviewer concerns.

## Priority 1: Critical Ablations

These ablations directly address fairness concerns and should be run if possible.

### 1.1. TF-IDF Without Bigrams

**Concern**: TF-IDF uses bigrams `(1,2)` while BM25 uses unigrams only. Does the bigram advantage explain TF-IDF performance?

**Experiment**:
```python
# Current: TF-IDF with bigrams
tfidf_bigram = TfidfEmbedder(
    embedding_dim=256,
    ngram_range=(1, 2),  # Current
)

# Ablation: TF-IDF unigrams only
tfidf_unigram = TfidfEmbedder(
    embedding_dim=256,
    ngram_range=(1, 1),  # Ablation
)
```

**Expected outcome**:
- If performance drops significantly: Bigrams matter for SHELF
- If performance similar: Unigrams sufficient, bigrams not unfair advantage

**Value**: Isolates the contribution of explicit phrase features.

**Effort**: Low (single config change)

---

### 1.2. TF-IDF Dimensionality Sweep

**Concern**: Is 256 dimensions optimal for TF-IDF? Would other dimensions change the ranking?

**Experiment**:
```python
dimensions = [64, 128, 256, 512, 768, 1024]

for dim in dimensions:
    tfidf = TfidfEmbedder(embedding_dim=dim, ngram_range=(1, 2))
    results[f"tfidf-{dim}"] = evaluate(task, model=tfidf)
```

**Expected outcome**:
- Performance curve shows optimal dimension range
- 256 is reasonable choice, not cherry-picked

**Value**: Demonstrates robustness to dimension choice.

**Effort**: Medium (6 runs per task)

**Visualization**:
```
NDCG@10 vs. Embedding Dimension (TF-IDF+SVD)

NDCG@10
  ^
  |                 *---*
  |             *---*
  |         *---*
  |     *---*
  | *---*
  +----------------------------> Dimension
    64  128  256  512  768 1024
```

---

### 1.3. BM25 Parameter Sensitivity

**Concern**: Are default BM25 parameters (k1=1.5, b=0.75) optimal for SHELF?

**Experiment**:
```python
# Test standard parameter ranges from literature
k1_values = [0.5, 1.0, 1.5, 2.0]  # Term saturation
b_values = [0.5, 0.75, 0.9]       # Length normalization

for k1 in k1_values:
    for b in b_values:
        bm25 = BM25Embedder(k1=k1, b=b)
        results[f"bm25-k1{k1}-b{b}"] = evaluate(task, model=bm25)
```

**Expected outcome**:
- Default parameters (1.5, 0.75) are reasonable
- Shows we didn't cherry-pick BM25 parameters

**Value**: Demonstrates parameter robustness.

**Effort**: Medium (12 combinations per task)

**Visualization** (heatmap):
```
BM25 NDCG@10 by (k1, b)

b    \ k1   0.5   1.0   1.5   2.0
0.5        0.45  0.52  0.54  0.53
0.75       0.47  0.54  0.56  0.55  <- Default
0.9        0.44  0.51  0.53  0.52
```

---

### 1.4. Dense Model Without Normalization

**Concern**: Does L2 normalization help dense models more than sparse?

**Experiment**:
```python
# Current: With normalization
embedder_norm = SentenceTransformerEmbedder(
    model_name="all-MiniLM-L6-v2",
    normalize=True  # Current
)

# Ablation: Without normalization
embedder_no_norm = SentenceTransformerEmbedder(
    model_name="all-MiniLM-L6-v2",
    normalize=False  # Ablation
)
```

**Expected outcome**:
- Normalization helps both sparse and dense
- Similar relative ranking with/without

**Value**: Shows normalization isn't favoring one paradigm.

**Effort**: Low (run subset of models)

---

## Priority 2: Insightful Ablations

These provide scientific insight but aren't critical for fairness.

### 2.1. Vocabulary Size Sweep (TF-IDF)

**Question**: How does vocabulary size affect performance?

**Experiment**:
```python
vocab_sizes = [10000, 25000, 50000, 100000]

for size in vocab_sizes:
    tfidf = TfidfEmbedder(
        embedding_dim=256,
        max_features=size,
        ngram_range=(1, 2)
    )
    results[f"tfidf-vocab{size}"] = evaluate(task, model=tfidf)
```

**Value**: Shows whether larger vocabulary helps (domain-specific terms).

**Effort**: Medium (4 runs per task)

---

### 2.2. Min/Max DF Sensitivity

**Question**: Do document frequency filters significantly impact results?

**Experiment**:
```python
configs = [
    {"min_df": 1, "max_df": 1.0},     # No filtering
    {"min_df": 2, "max_df": 0.95},    # Default
    {"min_df": 5, "max_df": 0.90},    # Aggressive
]

for config in configs:
    tfidf = TfidfEmbedder(embedding_dim=256, **config)
    results[f"tfidf-{config}"] = evaluate(task, model=tfidf)
```

**Value**: Shows robustness to preprocessing choices.

**Effort**: Low (3 runs per task)

---

### 2.3. Sublinear TF Ablation

**Question**: Does sublinear TF (1 + log(tf)) help?

**Experiment**:
```python
# Current: Sublinear TF
tfidf_sublinear = TfidfEmbedder(
    embedding_dim=256,
    sublinear_tf=True  # Current
)

# Ablation: Raw TF
tfidf_raw = TfidfEmbedder(
    embedding_dim=256,
    sublinear_tf=False  # Ablation
)
```

**Value**: Shows we follow sklearn best practices.

**Effort**: Low (single toggle)

---

### 2.4. Hybrid Sparse + Dense

**Question**: Can combining sparse and dense beat either alone?

**Experiment**:
```python
# Hybrid retrieval: average BM25 and MiniLM scores
for alpha in [0.0, 0.2, 0.5, 0.8, 1.0]:
    hybrid_score = alpha * bm25_score + (1 - alpha) * dense_score
    results[f"hybrid-{alpha}"] = evaluate_from_scores(hybrid_score)
```

**Value**:
- Follows industry best practice (sparse + dense hybrid)
- May improve over either baseline
- Cited in recent literature as optimal approach

**Effort**: Medium (requires score-level fusion)

**Expected result** (from literature):
- Hybrid often beats both baselines
- Optimal alpha typically 0.5-0.8 (favoring dense slightly)

---

## Priority 3: Advanced Ablations

These are interesting but not essential for publication.

### 3.1. Dense Models Fine-tuned on SHELF

**Question**: Would fine-tuning dense models on SHELF close the gap?

**Experiment**:
```python
# Fine-tune small dense model on SHELF train split
model = SentenceTransformer("all-MiniLM-L6-v2")
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100,
)
```

**Value**:
- Shows whether domain adaptation helps
- Demonstrates that off-the-shelf models are fair baseline

**Effort**: High (requires training infrastructure)

**Note**: This would make sparse vs. dense unfair (sparse not fine-tuned), but shows potential.

---

### 3.2. Different SVD Algorithms

**Question**: Does SVD algorithm choice matter?

**Experiment**:
```python
# Current: TruncatedSVD (randomized)
from sklearn.decomposition import TruncatedSVD

# Ablation: Full SVD via PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=256)
```

**Value**: Shows robustness to decomposition method.

**Effort**: Low (swap decomposition)

---

### 3.3. Sparse Models with Learned Term Weighting

**Question**: Can learned term weighting improve sparse models?

**Experiment**:
```python
# SPLADE-style learned sparse retrieval
# (requires training on SHELF)
```

**Value**: Bridges gap between sparse and dense.

**Effort**: Very high (new model architecture)

---

## Priority 4: Analysis, Not Experiments

These don't require new runs, just analysis of existing results.

### 4.1. Performance by Document Length

**Analysis**: Stratify results by document length.

**Hypothesis**:
- Sparse may excel on short documents (less context needed)
- Dense may excel on long documents (more context available)

**Implementation**:
```python
for length_bin in ["short (<500)", "medium (500-1500)", "long (>1500)"]:
    docs = filter_by_length(corpus, length_bin)
    metrics = compute_metrics(predictions, docs)
```

**Value**: Explains when each paradigm excels.

**Effort**: Low (post-hoc analysis)

---

### 4.2. Error Analysis: Where Sparse Wins

**Analysis**: Identify queries where TF-IDF/BM25 outperform dense models.

**Questions**:
- Do they involve exact terminology? (e.g., "Dewey Decimal")
- Are they short queries? (less context for dense)
- Do they rely on rare terms?

**Implementation**:
```python
sparse_wins = queries where (tfidf_score > dense_score)
analyze_characteristics(sparse_wins)
```

**Value**: Provides scientific insight into sparse advantages.

**Effort**: Low (post-hoc analysis)

---

### 4.3. Error Analysis: Where Dense Wins

**Analysis**: Identify queries where dense models outperform sparse.

**Questions**:
- Do they involve synonyms? (e.g., "car" vs. "automobile")
- Are they long, descriptive queries?
- Do they have polysemous terms?

**Value**: Provides scientific insight into dense advantages.

**Effort**: Low (post-hoc analysis)

---

## Recommended Ablation Set (Minimal)

If time/resources are limited, run these:

1. **TF-IDF without bigrams** (1.1) - Critical fairness concern
2. **TF-IDF dimension sweep** (1.2) - Shows 256 is reasonable
3. **BM25 parameter grid** (1.3) - Shows defaults are reasonable
4. **Error analysis** (4.2, 4.3) - Free, high-value insights

**Estimated effort**: ~40 model runs + analysis (1-2 days compute)

---

## Recommended Ablation Set (Comprehensive)

For a thorough fairness analysis:

1. **All Priority 1 ablations** (1.1-1.4)
2. **Vocabulary size sweep** (2.1)
3. **Sublinear TF ablation** (2.3)
4. **Hybrid sparse+dense** (2.4)
5. **All Priority 4 analyses** (4.1-4.3)

**Estimated effort**: ~100 model runs + analysis (3-5 days compute)

---

## How to Present Ablations in Paper

### In Main Paper

**Section: Experimental Design**
> To ensure fair comparison, we use standard hyperparameters for all models:
> - TF-IDF: sklearn defaults with bigrams (1,2), 256-dim SVD
> - BM25: Okapi defaults (k1=1.5, b=0.75)
> - Dense: HuggingFace pretrained, normalized
>
> Ablation studies (Appendix A) show these choices are robust: TF-IDF performance is stable across dimensions (128-512), BM25 results vary <5% across standard parameter ranges, and bigrams provide consistent but modest improvement.

### In Appendix

**Appendix A: Ablation Studies**

**A.1 TF-IDF Configuration Robustness**
- Figure: NDCG@10 vs. embedding dimension
- Table: Performance with/without bigrams
- Finding: 256 dimensions is reasonable, bigrams add ~3% improvement

**A.2 BM25 Parameter Sensitivity**
- Heatmap: NDCG@10 across (k1, b) grid
- Finding: Default parameters (1.5, 0.75) within 2% of optimal

**A.3 Error Analysis**
- Table: Characteristics of queries where sparse > dense
- Table: Characteristics of queries where dense > sparse
- Finding: Sparse excels on exact terminology, dense on semantic similarity

---

## Experimental Design Template

For reproducibility, use this template:

```python
# ablation_runner.py
from shelf.evaluate import evaluate
from shelf.evaluate.adapters import TfidfEmbedder, BM25Embedder

# Define ablation configs
ablations = {
    "tfidf-unigram": {
        "model": TfidfEmbedder(embedding_dim=256, ngram_range=(1, 1)),
        "priority": 1,
    },
    "tfidf-bigram": {
        "model": TfidfEmbedder(embedding_dim=256, ngram_range=(1, 2)),
        "priority": 1,
    },
    "tfidf-dim128": {
        "model": TfidfEmbedder(embedding_dim=128, ngram_range=(1, 2)),
        "priority": 1,
    },
    # ... more configs
}

# Run ablations
for name, config in ablations.items():
    if config["priority"] <= MAX_PRIORITY:
        result = evaluate(task="lcc_retrieval", model=config["model"])
        save_result(name, result)
```

---

## Success Criteria

**Ablations successfully strengthen fairness claims if:**

1. ✓ TF-IDF performance robust to dimension choice (±5%)
2. ✓ BM25 performance robust to parameter choice (±5%)
3. ✓ Bigrams provide consistent but modest advantage (~3-5%)
4. ✓ Normalization helps both sparse and dense similarly
5. ✓ Error analysis explains when each paradigm excels

**Red flags** (would require investigation):
- ⚠ Huge performance swings with small parameter changes
- ⚠ Cherry-picked hyperparameters far from standard values
- ⚠ Bigrams provide >20% improvement (suggests unfair advantage)

---

## Conclusion

Running Priority 1 ablations (1.1-1.4) + analyses (4.1-4.3) will:
- **Demonstrate fairness**: No cherry-picking of hyperparameters
- **Provide scientific insight**: When sparse vs. dense excels
- **Strengthen paper**: Addresses reviewer concerns proactively
- **Modest cost**: ~40 runs + analysis (1-2 days)

These ablations transform a potential weakness (sparse > dense surprising) into a strength (robust, well-analyzed finding).

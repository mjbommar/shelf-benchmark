# Hyperparameter Comparison: Sparse vs. Dense Models

## Overview

This document provides a comprehensive comparison of hyperparameters and configuration settings for sparse (TF-IDF, BM25) and dense (neural embedding) models in the SHELF benchmark.

## TF-IDF Configuration

### Default Hyperparameters
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `embedding_dim` | 256 | SVD dimensionality reduction to match typical dense model ranges |
| `max_features` | 50,000 | Large vocabulary to capture domain-specific terms |
| `ngram_range` | (1, 2) | Unigrams + bigrams for phrase capture |
| `min_df` | 2 | Minimum document frequency (filter very rare terms) |
| `max_df` | 0.95 | Maximum document frequency (filter stop words) |
| `sublinear_tf` | True | Use 1 + log(tf) instead of raw tf (recommended best practice) |
| `normalize_output` | True | L2-normalize embeddings for cosine similarity |
| `backend` | sklearn | Use sklearn's TfidfVectorizer (industry standard) |

### SVD Configuration
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `n_components` | 256 | Target dimensionality (between small/base dense models) |
| `random_state` | 42 | Reproducibility |

### Preprocessing (via sklearn TfidfVectorizer)
- **Tokenization**: Whitespace and punctuation splitting (sklearn default)
- **Lowercasing**: Yes (sklearn default)
- **Stop word removal**: Via max_df=0.95 (frequency-based)
- **Accent removal**: No (sklearn default)
- **Vocabulary**: Fitted on corpus (up to 50k features)

## BM25 Configuration

### Default Hyperparameters
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `k1` | 1.5 | Standard Okapi BM25 default (term saturation) |
| `b` | 0.75 | Standard Okapi BM25 default (length normalization) |

### Preprocessing (via SHELF WordTokenizer)
- **Tokenization**: Regex-based word boundary detection (`\b\w+\b`)
- **Lowercasing**: Yes
- **Stop word removal**: No (BM25 IDF handles this naturally)
- **Accent removal**: Yes (Unicode NFD normalization + strip)
- **Min token length**: 1
- **Vocabulary**: Fitted on corpus with min_df and max_df filters

## Dense Model Configuration (SentenceTransformers)

### Default Hyperparameters
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `normalize` | True | L2-normalize for cosine similarity |
| `batch_size` | 32 | Standard batch size for GPU inference |
| `show_progress` | False | Default (enabled in scripts) |

### Model-Specific Configurations

#### Small Models (22M-33M params, 384 dims)
| Model | Params | Embedding Dim | Context Window | Architecture |
|-------|--------|---------------|----------------|--------------|
| all-MiniLM-L6-v2 | 22.7M | 384 | 256 | BERT-like, 6 layers |
| BAAI/bge-small-en-v1.5 | 33.4M | 384 | 512 | BERT-like, optimized for retrieval |
| intfloat/e5-small-v2 | 33.4M | 384 | 512 | BERT-like, contrastive learning |
| thenlper/gte-small | 33.4M | 384 | 512 | BERT-like, trained on diverse corpus |

#### Base Models (~110M params, 768 dims)
| Model | Params | Embedding Dim | Context Window | Architecture |
|-------|--------|---------------|----------------|--------------|
| all-mpnet-base-v2 | 109.5M | 768 | 514 | MPNet (masked + permuted) |
| BAAI/bge-base-en-v1.5 | 109.5M | 768 | 512 | BERT-base, optimized for retrieval |
| intfloat/e5-base-v2 | 109.5M | 768 | 512 | BERT-base, contrastive learning |
| thenlper/gte-base | 109.5M | 768 | 512 | BERT-base |
| gtr-t5-base | 109.6M | 768 | 512 | T5-based encoder |
| hkunlp/instructor-base | 109.5M | 768 | 512 | BERT-base with instruction tuning |

#### Large Models (~335M params, 768-1024 dims)
| Model | Params | Embedding Dim | Context Window | Architecture |
|-------|--------|---------------|----------------|--------------|
| BAAI/bge-large-en-v1.5 | 335.1M | 1024 | 512 | BERT-large, optimized for retrieval |
| intfloat/e5-large-v2 | 335.1M | 1024 | 512 | BERT-large, contrastive learning |
| gtr-t5-large | 335.9M | 768 | 512 | T5-large encoder |

### Preprocessing (Model-Specific)

All dense models use learned tokenization:

- **Tokenization**: WordPiece (BERT-based) or SentencePiece (T5-based)
- **Lowercasing**: Model-dependent (uncased vs. cased)
- **Stop word removal**: No (learned in pretraining)
- **Accent handling**: Learned in pretraining
- **Vocabulary**: Fixed pretrained vocabulary (typically 30k tokens)
- **Subword splitting**: Yes (handles OOV words)
- **Special tokens**: [CLS], [SEP], [PAD] (model-specific)

## Key Differences: Sparse vs. Dense

### Tokenization
| Aspect | TF-IDF/BM25 | Dense Models |
|--------|-------------|--------------|
| Method | Regex word boundaries or sklearn default | WordPiece/SentencePiece |
| OOV handling | Ignored | Subword decomposition |
| Vocabulary size | 50k (fitted) | 30k (pretrained) |
| Context awareness | No | Yes (via pretraining) |

### Normalization
| Aspect | TF-IDF/BM25 | Dense Models |
|--------|-------------|--------------|
| Output normalization | L2-norm (explicit) | L2-norm (explicit) |
| Term weighting | TF-IDF or BM25 formula | Learned attention weights |
| Length normalization | BM25 b parameter | Handled by pooling strategy |

### Embedding Dimensions
| Model Type | Dimensions | Notes |
|------------|------------|-------|
| TF-IDF | 256 (via SVD) | Reduced from 50k vocabulary |
| BM25 | 50k (sparse) | No dimensionality reduction |
| Small Dense | 384 | Fixed model architecture |
| Base Dense | 768 | Fixed model architecture |
| Large Dense | 768-1024 | Fixed model architecture |

### Feature Representation
| Aspect | TF-IDF | BM25 | Dense |
|--------|--------|------|-------|
| N-grams | Unigrams + bigrams | Unigrams only | Implicit via context |
| Phrase capture | Explicit bigrams | No | Learned representations |
| Semantic similarity | No | No | Yes |
| Exact match | Strong | Strong | Weaker |

## Potential Advantages Analysis

### TF-IDF Advantages
1. **Bigram features**: Captures explicit phrases (e.g., "machine learning", "New York")
2. **Large vocabulary**: 50k features can capture rare domain-specific terms
3. **SVD dimension**: 256 is smaller than base models (768) but larger than small models (384)
4. **Sublinear TF**: Prevents common terms from dominating (standard best practice)
5. **Tuned for retrieval**: IDF naturally downweights common terms

### TF-IDF Disadvantages
1. **No semantic understanding**: "car" and "automobile" are unrelated
2. **Dimensionality reduction loss**: SVD to 256 loses information
3. **No context**: "bank" (financial) vs. "bank" (river) indistinguishable
4. **Sparse vocabulary**: Only 50k terms, may miss rare words
5. **No pretraining**: Must learn from SHELF corpus only

### BM25 Advantages
1. **No dimensionality reduction**: Full 50k sparse representation
2. **Proven retrieval formula**: Industry standard for lexical search
3. **Length normalization**: b parameter handles document length bias
4. **IDF weighting**: Naturally handles term importance
5. **No training needed**: Parameter-free on new corpora

### BM25 Disadvantages
1. **Unigrams only**: No bigram/phrase features (unlike TF-IDF)
2. **No semantic understanding**: Same as TF-IDF
3. **Sparse representation**: Inefficient for similarity computation
4. **No context**: Same as TF-IDF
5. **Fixed parameters**: k1=1.5, b=0.75 not tuned for SHELF

### Dense Model Advantages
1. **Semantic understanding**: Trained on billions of tokens
2. **Context-aware**: "bank" meaning depends on context
3. **OOV handling**: Subword tokenization handles rare words
4. **Transfer learning**: Pretrained on diverse corpora
5. **Dense representation**: Efficient cosine similarity

### Dense Model Disadvantages
1. **Smaller vocabulary**: 30k tokens vs. 50k for sparse
2. **Fixed dimensions**: 384/768/1024, not tunable
3. **No explicit bigrams**: Must learn phrase patterns
4. **Computational cost**: Much slower than sparse methods
5. **Potential domain mismatch**: Pretrained on web text, not library science

## Fairness Assessment

### Are Hyperparameters Reasonable?

**TF-IDF:**
- `ngram_range=(1,2)`: Standard practice for retrieval tasks
- `max_features=50000`: Standard for large corpora (BEIR uses similar)
- `sublinear_tf=True`: Recommended in sklearn documentation
- `embedding_dim=256`: Conservative choice between small (384) and base (768) models
- **Verdict**: Hyperparameters follow standard best practices, not tuned to favor sparse

**BM25:**
- `k1=1.5, b=0.75`: Standard Okapi BM25 defaults used in Elasticsearch, Lucene, etc.
- No tuning on SHELF corpus
- **Verdict**: Industry-standard defaults, maximally fair

**Dense Models:**
- All models use default configurations from HuggingFace
- No hyperparameter tuning on SHELF
- Normalization enabled (standard for cosine similarity)
- **Verdict**: Default pretrained models, maximally fair

### Are Preprocessing Steps Comparable?

| Preprocessing Step | TF-IDF | BM25 | Dense |
|--------------------|--------|------|-------|
| Lowercasing | Yes | Yes | Model-dependent |
| Accent removal | No | Yes | Learned |
| Stop word removal | Frequency-based | No | Learned |
| Tokenization | Whitespace/punct | Regex word boundary | WordPiece/SentencePiece |
| OOV handling | Drop | Drop | Subword split |
| Normalization | L2 (post-hoc) | No | L2 (post-hoc) |

**Assessment:**
- Sparse models use simpler, more transparent preprocessing
- Dense models rely on learned preprocessing from pretraining
- Both use L2 normalization for cosine similarity (when applicable)
- **Verdict**: Preprocessing differs by design, but neither is unfairly advantaged

### Potential Unfair Advantages?

**TF-IDF Bigram Advantage:**
- TF-IDF uses bigrams, BM25 does not
- Dense models must learn phrase patterns
- **Impact**: TF-IDF may have advantage on phrase-heavy queries
- **Mitigation**: This is a standard feature of TF-IDF, not tuning

**Dimension Mismatch:**
- TF-IDF: 256 dims (via SVD)
- Small dense: 384 dims
- Base dense: 768 dims
- **Impact**: TF-IDF is smaller than base models
- **Mitigation**: 256 is reasonable compression, larger would exceed small models

**Vocabulary Size:**
- TF-IDF/BM25: 50k terms (fitted on SHELF)
- Dense: 30k tokens (pretrained)
- **Impact**: Sparse has more terms but less context
- **Mitigation**: Trade-off between vocabulary size and semantic understanding

### Overall Fairness Conclusion

**The comparison is fair because:**

1. **No corpus-specific tuning**: All hyperparameters use standard defaults
2. **Consistent normalization**: L2-norm applied to all models for cosine similarity
3. **Standard best practices**: TF-IDF settings from sklearn docs, BM25 from Elasticsearch defaults
4. **No cherry-picking**: All major embedding models from MTEB/BEIR benchmarks included
5. **Transparent preprocessing**: All tokenization and normalization steps documented

**Potential concerns:**
1. **Bigrams**: TF-IDF has explicit bigrams, but this is standard for TF-IDF
2. **Dimensions**: TF-IDF uses 256 (conservative choice), could test other dimensions
3. **Preprocessing**: Different tokenization schemes, but inherent to model types

**Recommendation:**
- The comparison is fair and follows best practices
- Ablation studies (see `ablation_suggestions.md`) can strengthen claims
- Surprising sparse > dense results warrant investigation but not unfair advantage

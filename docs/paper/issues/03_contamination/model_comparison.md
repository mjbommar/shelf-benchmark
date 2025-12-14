# Model Architecture Comparison: Generation vs. Evaluation

## Executive Summary

**Key Finding**: Generation models (decoder-only) and evaluation models (encoder-only) have fundamentally different architectures, training objectives, and capabilities. This architectural separation prevents circular feedback loops and ensures independent evaluation.

## Architecture Taxonomy

### Decoder-Only Models (Generation)

**Used in SHELF for**: Document generation

**Models**: GPT-5.1, GPT-5.2, Gemini 2.5/3, Claude 4.5

**Architecture Characteristics**:
- **Attention**: Unidirectional (causal/left-to-right)
- **Training Objective**: Autoregressive next-token prediction
- **Primary Capability**: Text generation
- **Context Processing**: Sequential (can only attend to previous tokens)
- **Output**: Token probabilities for generation
- **Typical Use Cases**:
  - Text completion
  - Story writing
  - Code generation
  - Conversational AI

### Encoder-Only Models (Evaluation)

**Used in SHELF for**: Document embedding and retrieval

**Models**: BGE, E5, GTE, BERT variants

**Architecture Characteristics**:
- **Attention**: Bidirectional (can attend to all tokens)
- **Training Objective**: Masked Language Modeling (MLM) + Contrastive Learning
- **Primary Capability**: Representation learning
- **Context Processing**: Parallel (attends to entire sequence simultaneously)
- **Output**: Dense vector embeddings
- **Typical Use Cases**:
  - Semantic search
  - Document similarity
  - Classification
  - Information retrieval

## Detailed Model Comparison Table

| Dimension | Decoder-Only (Generation) | Encoder-Only (Evaluation) |
|-----------|---------------------------|---------------------------|
| **Architecture** | Causal self-attention layers | Bidirectional self-attention layers |
| **Mask Pattern** | Lower triangular (future masking) | Full attention matrix |
| **Training Loss** | Cross-entropy (next token) | MLM + Contrastive (SimCSE/InfoNCE) |
| **Output Space** | Vocabulary distribution (50k+ dims) | Fixed embedding (768-1024 dims) |
| **Inference Mode** | Iterative (token-by-token) | Single-pass (parallel) |
| **Token Processing** | Sequential dependencies | Independent representations |
| **Objective** | P(x_t \| x_<t) | Learn f: text → vector |
| **Pretraining** | Web text, books, code | Contrastive pairs, retrieval tasks |
| **Fine-tuning** | Instruction tuning, RLHF | Retrieval datasets, similarity labels |

## Training Data Comparison

### Generation Models (Decoder-Only)

**GPT-5.x Training Data**:
- Web text (Common Crawl, filtered)
- Books corpus
- GitHub code
- Academic papers
- News articles
- **Size**: Multi-trillion tokens
- **Cutoff**: September 2024 (GPT-5.1), August 2025 (GPT-5.2)

**Gemini 2.5/3 Training Data**:
- Multimodal (text, images, video)
- Web documents
- Code repositories
- Scientific literature
- **Size**: Estimated multi-trillion tokens
- **Cutoff**: January 2025 (Gemini 2.5)

**Claude 4.5 Training Data**:
- Web text (curated)
- Books and articles
- Code
- Constitutional AI feedback data
- **Size**: Estimated trillions of tokens
- **Cutoff**: March-August 2025

### Evaluation Models (Encoder-Only)

**BGE Training Data**:
- Pre-training: RetroMAE (unsupervised reconstruction)
- Fine-tuning: Large-scale contrastive pairs
  - MS MARCO (passage ranking)
  - Natural Questions
  - HotpotQA
  - BEIR datasets
- **Size**: ~1 billion text pairs
- **Cutoff**: Pre-2024 (most models)

**E5 Training Data**:
- Multilingual text pairs: ~1 billion
- Sources:
  - Wikipedia inter-language links
  - CCMatrix parallel corpus
  - MS MARCO
  - NLI datasets (SNLI, MultiNLI)
- **Size**: ~1 billion pairs
- **Cutoff**: Pre-2024

**GTE Training Data**:
- Diverse retrieval tasks
- Code and natural language pairs
- Web text with hyperlinks
- Contrastive learning pairs
- **Size**: ~1 billion pairs
- **Cutoff**: Pre-2024

## Key Architectural Differences

### 1. Attention Mechanism

**Decoder-Only (Causal Attention)**:
```
Query at position i can only attend to positions ≤ i

Attention Matrix (4 tokens):
[1, 0, 0, 0]
[1, 1, 0, 0]
[1, 1, 1, 0]
[1, 1, 1, 1]
```

**Encoder-Only (Full Attention)**:
```
Query at position i can attend to all positions

Attention Matrix (4 tokens):
[1, 1, 1, 1]
[1, 1, 1, 1]
[1, 1, 1, 1]
[1, 1, 1, 1]
```

### 2. Training Objectives

**Decoder-Only**:
```python
# Next token prediction
loss = CrossEntropy(model(x[:-1]), x[1:])
# Maximize: P(x_t | x_<t)
```

**Encoder-Only**:
```python
# Contrastive learning (e.g., SimCSE, InfoNCE)
embeddings = encoder(texts)
loss = InfoNCE(embeddings[positives], embeddings[negatives])
# Maximize: similarity(related texts), minimize: similarity(unrelated texts)
```

### 3. Output Representations

**Decoder-Only**:
- Output: Probability distribution over vocabulary (dim: 50k-100k)
- Usage: Sample next token, continue generation
- Not optimized for similarity comparison

**Encoder-Only**:
- Output: Dense vector embedding (dim: 768-1024)
- Usage: Cosine similarity, dot product, clustering
- Explicitly optimized for semantic similarity

## Why Architectural Separation Prevents Circularity

### 1. Different Optimization Targets

- **Decoders**: Optimize for fluent text generation (perplexity, likelihood)
- **Encoders**: Optimize for semantic clustering (contrastive loss, triplet margin)

These objectives are **orthogonal** - excelling at generation does not guarantee good embeddings.

### 2. No Shared Parameters or Training Pipeline

- Generation models: Trained by OpenAI, Google, Anthropic (proprietary)
- Embedding models: Trained by BAAI, Microsoft, Alibaba (open source)
- **Zero overlap** in training infrastructure, data pipelines, or model weights

### 3. Evaluation Independence

SHELF evaluates:
- **What**: Embedding quality (similarity, retrieval, clustering)
- **How**: Cosine similarity, kNN, classification metrics

SHELF does **NOT** evaluate:
- Generation quality
- Perplexity
- Token prediction accuracy

### 4. Temporal and Architectural Firewall

```
Generation Models → SHELF Documents → Embedding Models
(Dec 2025)          (Dec 2025)        (Pre-2024 training)
    ↑                                      ↑
    |                                      |
    Decoder-only                    Encoder-only
    Causal attention                Full attention
    Next token pred.                Contrastive learning
```

**No feedback loop possible**:
1. Embedding models trained before SHELF documents existed
2. Different architectures prevent direct knowledge transfer
3. Generation models don't use embedding model outputs

## Comparison to Problematic Circular Scenarios

### Problematic: LLM-as-a-Judge for LLM Outputs

**Scenario**: Use GPT-4 to judge GPT-4 generations
- **Issue**: Same model family, shared biases
- **Risk**: Circular preference amplification
- **Evidence**: Models favor outputs with their own writing style

### Problematic: Training on Model-Generated Data (Self-Distillation)

**Scenario**: Train GPT-N+1 on GPT-N outputs
- **Issue**: Error accumulation, diversity collapse
- **Risk**: "Model collapse" - representations converge to artifacts
- **Evidence**: Recursively generated data shows pattern degradation

### Non-Problematic: SHELF's Approach

**Scenario**: Use decoder-only to generate text, evaluate with encoder-only
- **No shared architecture**: Different attention mechanisms, objectives
- **No shared training**: Independent training pipelines, organizations
- **No temporal feedback**: Encoders frozen before generation occurred
- **No preferential bias**: Encoders don't favor specific generation model styles

## Empirical Evidence of Independence

### From Research Literature

1. **"Improving Text Embeddings with Large Language Models" (2024)**:
   - Shows decoder-only models (Llama, Mistral) can improve when fine-tuned for embeddings
   - **Key insight**: Requires explicit fine-tuning with contrastive loss - generative pretraining alone doesn't produce good embeddings

2. **"LLM2Vec" (2024)**:
   - Transforms decoder-only LLMs into encoder-style models
   - **Key insight**: Architectural modifications + embedding-specific training required
   - Confirms generation capability ≠ embedding quality

3. **MTEB Benchmark Results**:
   - Top embedding models: BGE, E5, GTE (encoder-only)
   - Decoder-only models (GPT-3.5, GPT-4) without fine-tuning: mediocre embedding performance
   - **Key insight**: Architecture specialization matters

## Conclusion

The architectural comparison demonstrates:

1. **Fundamental Differences**: Generation models (decoder-only) and evaluation models (encoder-only) have orthogonal capabilities and training objectives

2. **No Circularity Risk**: Different architectures, organizations, training pipelines, and temporal separation prevent feedback loops

3. **Independent Evaluation**: SHELF evaluates embedding quality (encoder domain) using documents generated by text generators (decoder domain) - these are separate capabilities

4. **Stronger Than Alternatives**: Unlike LLM-as-a-judge scenarios (same architecture evaluating same architecture), SHELF uses cross-architecture evaluation

**The architectural separation is a feature, not a bug - it ensures independent, unbiased evaluation of embedding model capabilities.**

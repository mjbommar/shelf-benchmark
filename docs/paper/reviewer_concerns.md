> **SUPERSEDED, 2026-08-27.** Written before the literature check and
> before the v0.4 corpus. It contains claims that do not survive review
> (see [contributions.md](contributions.md) for the retired list) and
> numbers that do not reproduce. Kept for history.
>
> Current framing: [outline_v2.md](outline_v2.md).
> Current claims: [contributions.md](contributions.md).
> Work queue: [TODO.md](TODO.md).

# Anticipated Reviewer Concerns

This document anticipates likely peer review criticisms and prepares responses.

---

## 1. TF-IDF Train/Test Leakage (CRITICAL)

**Concern**: "Does TF-IDF cheat by seeing test data during IDF computation or SVD fitting?"

**Analysis of Our Implementation**:

| Task Type | Data Flow | Leakage Risk |
|-----------|-----------|--------------|
| **Classification** | encode(train) → fit; encode(test) → transform | **None** - proper separation |
| **Retrieval** | encode(corpus) → fit; encode(queries) → transform | **None** - proper separation |
| **Clustering** | encode(test_split) → fit+transform | **Transductive** - fit on eval data only |
| **Pair Classification** | encode(all_pairs) → fit+transform | **Transductive** - fit on eval data only |

**Classification Code** (`classification.py:309-317`):
```python
train_embeddings = embedder.encode(train_texts)  # Fits TF-IDF + SVD on train
test_embeddings = embedder.encode(test_texts)    # Transforms test using train-fitted model
```
This is **correct** - no leakage.

**Clustering/Pairs**: Uses transductive approach (common in unsupervised tasks). The TF-IDF vocabulary and SVD are fit on the same data being evaluated. This is standard practice for clustering benchmarks but should be documented.

**Mitigation for Paper**:
- Clearly document the evaluation protocol for each task type
- Note that neural embeddings are fixed/frozen (no fitting on eval data)
- For clustering, emphasize this is transductive (same as MTEB clustering tasks)

---

## 2. Synthetic Data Quality

**Concern**: "Are LLM-generated documents realistic enough to evaluate document understanding?"

**Response**:
1. **Quality filtering**: 99.7% pass rate after filtering empty/non-English
2. **Multi-model generation**: 9 different frontier LLMs reduce systematic biases
3. **Human-designed taxonomies**: LC Classification is the most rigorous bibliographic system ever developed
4. **Controlled diversity**: Factorial design ensures coverage that exceeds natural corpora
5. **Actually harder**: Cross-dimensional combinations (jokes about philosophy) are MORE challenging than natural corpora where genres correlate with subjects

**Evidence to Include**:
- Sample documents in appendix
- Inter-annotator agreement on a subset
- Comparison of document statistics to real corpora (length distribution, vocabulary diversity)

---

## 3. Circularity/Contamination

**Concern**: "Doesn't testing embedding models on LLM-generated content create circularity?"

**Response**:
1. **Different models**: Generation models (GPT-5.x, Gemini, Claude) are NOT the embedding models being evaluated (BGE, E5, GTE, etc.)
2. **Text encoders vs. decoders**: Embedding models are encoder-only; generation models are decoder-only or encoder-decoder
3. **Training data separation**: SHELF documents are new (generated Dec 2025), cannot be in embedding model pretraining data
4. **Actually tests generalization**: If embeddings truly capture semantic meaning, they should work on any coherent text regardless of source

---

## 4. LC Taxonomy Bias

**Concern**: "The Library of Congress Classification has Western/American bias."

**Response**:
1. **Acknowledged limitation**: LC system was developed for US Congress library
2. **Mitigation**:
   - 44 geographic regions include global coverage (Asia, South America, Africa, etc.)
   - Topics include non-Western subjects (Buddhism, Arabic literature, etc.)
3. **Still universal**: LC is used by libraries worldwide; it's the most comprehensive taxonomy available
4. **Future work**: Could extend with non-Western classification systems (e.g., Chinese Library Classification)

---

## 5. Statistical Significance

**Concern**: "With high variance across tasks (std ~0.28), are model differences significant?"

**Response**:
1. **95% CIs reported**: All results include bootstrap confidence intervals
2. **Top neural models NOT significantly different**: BGE-large (0.52 ± 0.14) vs GTE-base (0.51 ± 0.14) overlap
3. **Sparse vs Dense IS significant**: TF+SVD (0.68) vs BGE-large (0.51) - CIs don't overlap
4. **Task-level analysis**: Per-task breakdowns reveal where differences are significant

**Mitigation**:
- Report effect sizes (Cohen's d)
- Use paired comparisons (same documents across models)
- Bonferroni correction for multiple comparisons

---

## 6. Task Independence

**Concern**: "Are the tasks truly independent, or do they share structure that inflates aggregate scores?"

**Response**:
1. **Same documents, different labels**: Tasks use different ground truth columns from same documents
2. **Different evaluation protocols**: Classification (supervised), Retrieval (ranking), Clustering (unsupervised)
3. **Correlation analysis**: Can report task score correlations to show independence
4. **Weighted aggregation**: SHELF score uses predefined weights, not learned

**Evidence to Include**:
- Task correlation matrix
- Per-task model rankings (show they differ)

---

## 7. Baseline Fairness

**Concern**: "Are sparse and dense models given equal treatment?"

**Response**:
1. **Same evaluation protocol**: All models get same texts, same metrics
2. **Preprocessing**:
   - Sparse: Tokenization + n-grams + SVD
   - Dense: Pre-trained tokenizer + neural encoding
3. **Hyperparameters**: Sparse uses reasonable defaults (50k features, 256 SVD dims); dense uses default model configs
4. **No cherry-picking**: Report all models evaluated, not just best

**Potential Issue**:
- Sparse models may benefit from longer documents (more terms)
- Could add length-stratified analysis

---

## 8. Document Length Effects

**Concern**: "Do sparse methods benefit unfairly from longer documents?"

**Analysis**:
- SHELF corpus: Mean 668 words, Median 326 words
- Sparse methods: More words → more features → potentially better discrimination
- Dense methods: Typically truncate to 512 tokens

**Mitigation**:
- Report length-stratified results
- Truncate all documents to 512 tokens for fairness analysis
- Note that real-world documents also vary in length

---

## 9. Reproducibility

**Concern**: "Can results be reproduced?"

**Response**:
1. **Strict versioning**: Every result includes:
   - Dataset checksum
   - sklearn/numpy/torch versions
   - Random seed (42 throughout)
   - Git commit hash
2. **Open source**: Full code + data on HuggingFace + GitHub
3. **Prediction-file-first**: Primary interface is JSONL predictions, enabling framework-agnostic evaluation
4. **Manifest files**: Every run produces `manifest.json` with full environment

---

## 10. Single Dataset Evaluation

**Concern**: "Results on one benchmark may not generalize."

**Response**:
1. **SHELF is intentionally diverse**: 21 subjects × 133 forms × 112 topics × 44 regions
2. **Complements existing benchmarks**: Not replacing MTEB, but addressing its gaps
3. **Contamination-resistant**: A key advantage over web-scraped benchmarks
4. **Future work**: Could create SHELF-v2 with different generation models

---

## Summary: Top 5 Concerns to Address Prominently

| Priority | Concern | Section to Address |
|----------|---------|-------------------|
| 1 | TF-IDF train/test separation | Methods: Evaluation Protocol |
| 2 | Synthetic data quality | Methods: Data Generation + Appendix |
| 3 | Statistical significance | Results: Confidence Intervals |
| 4 | Sparse vs Dense fairness | Results: Ablations |
| 5 | Reproducibility | Methods: Reproducibility Statement |

---

## Rebuttal Templates

### For Leakage Concern:
> "We appreciate this careful review. Our classification evaluation strictly separates train and test: TF-IDF vocabulary and SVD are fit on training data only (lines 309-317 of classification.py). Test documents are transformed using the train-fitted model. For clustering tasks, we follow the transductive protocol standard in the literature (same as MTEB clustering), where models process only the evaluation split. Neural embedding models are frozen and require no fitting on evaluation data."

### For Synthetic Data Concern:
> "We acknowledge the novelty of evaluating on synthetic documents. However, we argue this is a feature, not a bug: (1) Synthetic data is contamination-resistant—it cannot be in any model's pretraining corpus. (2) Our factorial design produces combinations (e.g., 'jokes about military science') that stress-test semantic understanding beyond surface correlations in natural corpora. (3) Quality filtering (99.7% pass rate) and multi-model generation (9 LLMs) ensure document quality and diversity. We include sample documents in Appendix A for qualitative assessment."

### For Statistical Significance:
> "We agree that reporting significance is essential given high task variance. Our results include 95% bootstrap confidence intervals for all scores. While differences between top neural models (e.g., BGE-large vs. GTE-base) are not significant, the gap between sparse (TF+SVD: 0.68) and dense (BGE-large: 0.51) methods is significant (non-overlapping CIs). We have added Cohen's d effect sizes and Bonferroni-corrected p-values in the revised manuscript."

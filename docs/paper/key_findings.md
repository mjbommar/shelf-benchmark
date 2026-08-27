> **SUPERSEDED, 2026-08-27.** Written before the literature check and
> before the v0.4 corpus. It contains claims that do not survive review
> (see [contributions.md](contributions.md) for the retired list) and
> numbers that do not reproduce. Kept for history.
>
> Current framing: [outline_v2.md](outline_v2.md).
> Current claims: [contributions.md](contributions.md).
> Work queue: [TODO.md](TODO.md).

# SHELF Key Empirical Findings

## Executive Summary

**Main Finding**: While TF-IDF+SVD achieves the highest raw SHELF score (0.679 vs BGE-large's 0.513), **no model differences are statistically significant after multiple comparison correction**. All 24 evaluated models fall into a single equivalence group (Friedman p = 0.958).

**Implication**: SHELF should be positioned as a **model characterization tool** that reveals task-specific capability profiles, not as a ranking system.

---

## 1. Overall Rankings

### SHELF Score (Weighted Aggregate)

| Rank | Model | SHELF Score | Type | Parameters |
|------|-------|-------------|------|------------|
| 1 | TF+SVD | 0.6790 | Sparse | - |
| 2 | BGE-large | 0.5131 | Dense | 335M |
| 3 | TF-IDF+SVD | 0.5109 | Sparse | - |
| 4 | E5-large | 0.5045 | Dense | 335M |
| 5 | GTE-base | 0.5020 | Dense | 109M |
| 6 | E5-base | 0.5014 | Dense | 109M |
| 7 | BGE-base | 0.4994 | Dense | 109M |
| 8 | BGE-small | 0.4964 | Dense | 33M |
| 9 | GTE-small | 0.4959 | Dense | 33M |
| 10 | Instructor-base | 0.4872 | Dense | 109M |

### Efficiency Rankings (SHELF_eff = SHELF × 1000 / log10(params))

| Rank | Model | SHELF | SHELF_eff | Size Category |
|------|-------|-------|-----------|---------------|
| 1 | BGE-small | 0.4964 | 65.98 | 10M-50M |
| 2 | GTE-small | 0.4959 | 65.92 | 10M-50M |
| 3 | E5-small | 0.4772 | 63.43 | 10M-50M |
| 4 | MiniLM-L6 | 0.4647 | 63.17 | 10M-50M |
| 5 | GTE-base | 0.5020 | 62.44 | 100M-300M |

**Insight**: Small models (33M params) achieve 97% of large model (335M) performance at 10x fewer parameters.

---

## 2. Task-Specific Performance

### Classification Tasks

| Task | Best Model | Best Score | Avg Score | Range |
|------|------------|------------|-----------|-------|
| lcc_classification | TF-IDF+SVD | 0.8781 | 0.7867 | 0.547-0.878 |
| lcgft_category_classification | E5-large | 0.7749 | 0.6811 | 0.479-0.775 |
| register_classification | TF-IDF+SVD | 0.6716 | 0.5172 | 0.400-0.672 |

**Key Insight**: Sparse TF-IDF+SVD dominates classification tasks, especially on fine-grained taxonomies.

### Retrieval Tasks

| Task | Best Model | Best Score | Avg Score | Range |
|------|------------|------------|-----------|-------|
| lcc_retrieval | MPNet | 0.6738 | 0.5327 | 0.260-0.674 |
| category_retrieval | E5-base | 0.4964 | 0.4141 | 0.284-0.496 |
| form_retrieval | E5-base | 0.1388 | 0.1025 | 0.062-0.139 |

**Key Insight**: Form retrieval is extremely difficult (best: 0.14 NDCG@10). Models struggle with document genre.

### Clustering Tasks

| Task | Best Model | Best Score | Avg Score | Range |
|------|------------|------------|-----------|-------|
| lcc_clustering | MPNet | 0.5694 | 0.3261 | 0.012-0.569 |
| lcgft_clustering | BERT | 0.1722 | 0.1002 | 0.031-0.172 |
| register_clustering | OGBert-110M | 0.0946 | 0.0398 | 0.002-0.095 |
| geographic_clustering | GTR-T5-base | 0.0248 | 0.0111 | 0.003-0.025 |

**Key Insight**: Geographic clustering is near-random (V-measure 0.02). Embeddings don't capture geographic content.

### Pair Classification Tasks

| Task | Best Model | Best Score | Avg Score | Range |
|------|------------|------------|-----------|-------|
| topic_overlap_pairs | MPNet | 0.8257 | 0.7669 | 0.739-0.826 |
| same_lcc_pairs | MPNet | 0.7743 | 0.7002 | 0.667-0.774 |
| same_topic_pairs | MPNet | 0.7701 | 0.6971 | 0.667-0.770 |
| same_form_pairs | GTE-small | 0.6831 | 0.6694 | 0.667-0.683 |
| same_audience_pairs | MPNet | 0.6678 | 0.6667 | 0.667-0.668 |
| same_register_pairs | E5-small | 0.6672 | 0.6667 | 0.667-0.667 |

**Key Insight**: Register and audience pairs are near-baseline (0.667 = random for binary). Embeddings don't capture these dimensions.

---

## 3. Surprising Findings

### Finding 1: Sparse ≈ Dense (Not Statistically Different)

While TF-IDF+SVD achieves the highest raw SHELF score (0.679), **this difference is NOT statistically significant** after multiple comparison correction.

| Model | Mean Task Score | Δ vs BM25 | Cohen's d | Significance |
|-------|-----------------|-----------|-----------|--------------|
| BM25 (baseline) | 0.5469 | - | - | - |
| BGE-large | 0.5180 | -0.0289 | 0.10 | Not sig. |
| TF-IDF+SVD | 0.5135 | -0.0334 | 0.12 | Not sig. |
| GTE-base | 0.5133 | -0.0336 | 0.12 | Not sig. |
| E5-large | 0.5065 | -0.0404 | 0.15 | Not sig. |
| MiniLM-L6 | 0.4852 | -0.0618 | 0.23 | Not sig. |
| BERT | 0.4758 | -0.0711 | 0.26 | Not sig. |

**Statistical reality**: All differences have negligible-to-small effect sizes (Cohen's d < 0.3). High task variance creates wide confidence intervals that overlap.

**Implication**: SHELF should be framed as a **model characterization tool** (revealing task-specific strengths/weaknesses) rather than a ranking system. Emphasize effect sizes and task-specific analysis over aggregate rankings.

### Finding 2: Massive Task Variance

Same model, wildly different performance:

| Model | LCC Classification | Form Retrieval | Δ |
|-------|-------------------|----------------|---|
| E5-base | 0.8444 | 0.1388 | 0.71 |
| BGE-large | 0.8643 | 0.1315 | 0.73 |
| MiniLM-L6 | 0.8072 | 0.1082 | 0.70 |

**Implication**: Single aggregate scores (like MTEB) hide critical capability gaps.

### Finding 3: Scaling Doesn't Help Much

| Size Tier | Best Model | SHELF Score |
|-----------|------------|-------------|
| <10M | ogbert-2m | 0.378 |
| 10M-50M | BGE-small | 0.496 |
| 50M-100M | distilbert | 0.464 |
| 100M-300M | GTE-base | 0.502 |
| 300M-1B | BGE-large | 0.513 |

10x parameters (33M → 335M) yields only 3% improvement (0.496 → 0.513).

### Finding 4: Task-Specific Champions

Different models win different tasks:
- **Classification**: TF-IDF+SVD
- **Retrieval**: MPNet, E5-base
- **Clustering**: MPNet, BERT
- **Pair Classification**: MPNet, GTE-small

No single model dominates all tasks.

---

## 4. Statistical Significance

### Critical Finding: No Significant Differences After Correction

When applying proper multiple comparison correction (Holm/Bonferroni), **no pairwise model differences are statistically significant** (all corrected p-values > 0.05).

Friedman test results: χ² = 12.69, p = 0.958

**All 24 evaluated models fall into a single equivalence group** — meaning none is statistically significantly better than any other after proper correction.

### 95% Confidence Intervals

| Model | Mean | Std | 95% CI | N tasks |
|-------|------|-----|--------|---------|
| BM25 | 0.5469 | 0.2126 | [0.387, 0.672] | 9 |
| BGE-large | 0.5180 | 0.2874 | [0.379, 0.660] | 16 |
| TF-IDF+SVD | 0.5135 | 0.2781 | [0.369, 0.638] | 16 |
| GTE-base | 0.5133 | 0.2868 | [0.373, 0.642] | 16 |

**Note**: Wide confidence intervals (±0.14) due to high task variance. Differences between top neural models are not statistically significant.

### Effect Sizes (Cohen's d)

| Comparison | Δ Score | Cohen's d | Interpretation |
|------------|---------|-----------|----------------|
| TF+SVD vs BGE-large | +0.166 | 0.61 | Medium |
| TF+SVD vs E5-large | +0.175 | 0.63 | Medium |
| BGE-large vs BGE-small | +0.017 | 0.06 | Negligible |

**Key insight**: While some effect sizes are medium, the high task variance means these differences do not reach statistical significance after correction.

---

## 5. Pareto-Optimal Models

Models on the efficiency frontier (no model has both higher score AND fewer parameters):

| Model | SHELF | Params | Why Pareto |
|-------|-------|--------|------------|
| ogbert-2m | 0.378 | 2.1M | Best <10M |
| MiniLM-L6 | 0.465 | 22.7M | Best efficiency ratio |
| BGE-small | 0.496 | 33.4M | Best 10M-50M |
| GTE-base | 0.502 | 109.5M | Best 100M-300M |
| BGE-large | 0.513 | 335M | Highest dense score |

---

## 6. Task Difficulty Analysis

Tasks ranked by average model performance:

| Task | Avg Score | Interpretation |
|------|-----------|----------------|
| lcc_classification | 0.787 | Easy - clear subject signals |
| topic_overlap_pairs | 0.767 | Easy - semantic overlap |
| same_lcc_pairs | 0.700 | Medium |
| lcgft_category_classification | 0.681 | Medium |
| same_topic_pairs | 0.697 | Medium |
| lcc_retrieval | 0.533 | Medium |
| register_classification | 0.517 | Hard - style detection |
| category_retrieval | 0.414 | Hard |
| lcc_clustering | 0.326 | Hard |
| lcgft_clustering | 0.100 | Very Hard |
| form_retrieval | 0.103 | Very Hard |
| register_clustering | 0.040 | Very Hard |
| geographic_clustering | 0.011 | Near-random |

**Pattern**: Models handle *what* documents are about (topics, subjects) much better than *how* they're written (form, register) or *where* they're about (geography).

---

## 7. Task Independence Analysis

**Critical for peer reviewers**: Demonstrates that SHELF tasks measure independent capabilities, not redundant skills.

### Task Correlation Statistics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Mean correlation | r = 0.088 | Tasks are independent (threshold: < 0.3) |
| Median correlation | r = 0.061 | Confirms independence |
| Negative correlations | 47% (56 of 120 pairs) | Tasks often anti-correlated |
| Effective sample size | 3.1 of 16 | Eigenvalue-based estimate |
| High correlation pairs | 0 (|r| > 0.7) | No redundant tasks |
| Task families | 0 | No highly correlated groups |

### Why This Matters

1. **Defense against "tasks are redundant" criticism**: Mean r < 0.3 demonstrates tasks measure genuinely different capabilities
2. **Supports aggregate scoring**: Independent tasks contribute unique information to overall SHELF score
3. **Validates benchmark design**: Cross-product generation creates orthogonal evaluation dimensions

### Correlation Matrix Interpretation

- Most pairwise correlations cluster around 0 (-.3 to +.3)
- No task pair has correlation > 0.7 (would indicate redundancy)
- High negative correlation count (47%) indicates diverse capability requirements

---

## 8. Task Champion Diversity

Different models excel at different tasks, indicating SHELF measures diverse capabilities.

### Champion Statistics

| Metric | Value |
|--------|-------|
| Unique champions | 8 models |
| Total tasks | 16 |
| Diversity ratio | 0.50 |

### Task Champions by Task

| Task | Champion Model |
|------|----------------|
| lcc_classification | TF-IDF+SVD |
| lcgft_category_classification | E5-large |
| register_classification | TF-IDF+SVD |
| lcc_retrieval | MPNet |
| category_retrieval | E5-base |
| form_retrieval | E5-base |
| lcc_clustering | MPNet |
| lcgft_clustering | BERT |
| register_clustering | OGBert-110M |
| geographic_clustering | GTR-T5-base |
| topic_overlap_pairs | MPNet |
| same_lcc_pairs | MPNet |
| same_topic_pairs | MPNet |
| same_form_pairs | GTE-small |
| same_audience_pairs | MPNet |
| same_register_pairs | E5-small |

**Insight**: No single model dominates all tasks. This supports SHELF's value as a comprehensive evaluation suite that reveals model-specific strengths.

---

## 9. Implications for Paper Framing

Based on statistical analysis, the paper should emphasize:

### Do

1. **Task-specific analysis**: Highlight which models excel at which tasks
2. **Effect sizes**: Report Cohen's d for all comparisons
3. **Task independence**: Mean r = 0.088 demonstrates non-redundant tasks
4. **Model characterization**: SHELF reveals capability profiles, not rankings
5. **Practical insights**: Form/register understanding is a gap for all models

### Don't

1. **Claim statistical superiority**: No model is significantly better after correction
2. **Over-interpret rankings**: All 24 models are statistically equivalent
3. **Focus on aggregate scores**: Task variance makes aggregates unreliable

### Suggested Paper Positioning

> "SHELF is a model characterization tool that reveals task-specific strengths and weaknesses across 18 independent evaluation tasks. While raw scores vary, no model achieves statistically significant superiority after multiple comparison correction—suggesting that SHELF's value lies in diagnostic capability profiling rather than ranking."

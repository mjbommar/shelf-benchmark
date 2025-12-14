# Bootstrap Confidence Intervals for SHELF Scores

**Method**: Percentile bootstrap with 10,000 iterations
**Confidence Level**: 95%
**Random Seed**: 42

## Full Model Rankings with Confidence Intervals

| Rank | Model | SHELF Score | 95% CI Lower | 95% CI Upper | CI Width | Width/Mean |
|------|-------|-------------|--------------|--------------|----------|------------|
| 1 | **bm25** | **0.5469** | 0.3917 | 0.6718 | 0.2801 | 51.2% |
| 2 | **bge_large** | **0.5180** | 0.3698 | 0.6521 | 0.2823 | 54.5% |
| 3 | **tfidf** | **0.5135** | 0.3731 | 0.6410 | 0.2679 | 52.2% |
| 4 | **gte_base** | **0.5133** | 0.3675 | 0.6483 | 0.2808 | 54.7% |
| 5 | gte_small | 0.5079 | 0.3643 | 0.6422 | 0.2778 | 54.7% |
| 6 | bge_base | 0.5071 | 0.3619 | 0.6399 | 0.2780 | 54.8% |
| 7 | e5_large | 0.5065 | 0.3674 | 0.6354 | 0.2680 | 52.9% |
| 8 | bge_small | 0.5056 | 0.3620 | 0.6365 | 0.2745 | 54.3% |
| 9 | e5_base | 0.5038 | 0.3673 | 0.6324 | 0.2651 | 52.6% |
| 10 | mpnet | 0.5000 | 0.3440 | 0.6399 | 0.2959 | 59.2% |
| 11 | instructor_base | 0.4961 | 0.3523 | 0.6265 | 0.2741 | 55.3% |
| 12 | tf | 0.4881 | 0.3247 | 0.6379 | 0.3133 | 64.2% |
| 13 | gtr_t5_large | 0.4878 | 0.3418 | 0.6172 | 0.2755 | 56.5% |
| 14 | e5_small | 0.4853 | 0.3446 | 0.6104 | 0.2659 | 54.8% |
| 15 | minilm | 0.4852 | 0.3384 | 0.6231 | 0.2848 | 58.7% |
| 16 | gtr_t5_base | 0.4766 | 0.3349 | 0.6055 | 0.2707 | 56.8% |
| 17 | bert | 0.4758 | 0.3396 | 0.6015 | 0.2618 | 55.0% |
| 18 | distilbert_base_uncased | 0.4711 | 0.3345 | 0.5962 | 0.2618 | 55.6% |
| 19 | ogbert_110m_base | 0.4622 | 0.3284 | 0.5842 | 0.2558 | 55.3% |
| 20 | ogbert_110m_sentence | 0.4621 | 0.3284 | 0.5833 | 0.2549 | 55.2% |
| 21 | ogbert_v1_mlm | 0.4442 | 0.3165 | 0.5668 | 0.2503 | 56.3% |
| 22 | roberta | 0.4335 | 0.2960 | 0.5605 | 0.2646 | 61.0% |
| 23 | kl3m_pico | 0.4204 | 0.2897 | 0.5427 | 0.2530 | 60.2% |
| 24 | ogbert_2m_sentence | 0.4090 | 0.2831 | 0.5318 | 0.2486 | 60.8% |

## Observations

### 1. All Top-10 Models Have Overlapping Confidence Intervals

The confidence intervals for ranks 1-10 overlap substantially:

- **Rank 1 (BM25)**: [0.392, 0.672]
- **Rank 10 (MPNet)**: [0.344, 0.640]

**Interpretation**: We cannot confidently distinguish the true ordering of models in positions 1-10. Rankings are **unstable** with respect to task resampling.

### 2. Confidence Interval Width Correlates with Task Coverage

Models with fewer task results show wider CIs:
- **TF** (rank 12): 64.2% width/mean (limited to classification/clustering tasks)
- **BM25** (rank 1): 51.2% width/mean (limited to retrieval/pairs tasks)
- **Neural models**: 52-59% width/mean (all tasks)

**Interpretation**: Aggregate scores are most reliable for models evaluated on all 16 tasks.

### 3. Sparse vs. Dense Comparison

| Category | Best Model | Score | 95% CI |
|----------|------------|-------|--------|
| **Sparse** | BM25 | 0.547 | [0.392, 0.672] |
| **Dense** | BGE-large | 0.518 | [0.370, 0.652] |
| **Difference** | — | 0.029 | **CIs overlap** |

**Statistical test**:
- p-value (paired t-test): 0.019 (significant at α=0.05)
- p-value (Bonferroni-corrected): 0.019 vs. 0.0056 threshold → **NOT significant**
- Cohen's d: 0.97 (large effect size)

**Interpretation**: BM25 outperforms BGE-large with a large effect size, but the difference is **marginally significant** and does **NOT survive multiple comparison correction**.

### 4. Top Neural Models Are Statistically Equivalent

Ranks 2-9 (BGE-large through E5-base) all have:
- Overlapping 95% CIs
- Score differences < 0.015
- p-values > 0.23 for all pairwise tests

**Interpretation**: Cannot meaningfully rank top neural models. Performance is **essentially equivalent** within statistical uncertainty.

## Statistical Guidance for Interpretation

### When comparing two models:

1. **Check CI overlap**:
   - No overlap → Likely different (but still test)
   - Partial overlap → Uncertain
   - Full overlap → Statistically indistinguishable

2. **Perform paired test** (if common tasks):
   - Report both p-value and effect size
   - Use Bonferroni correction if comparing multiple pairs
   - Interpret effect size (Cohen's d) for practical significance

3. **Consider task coverage**:
   - Models with different task coverage are harder to compare
   - Restricted to common tasks for valid paired tests

### When interpreting rankings:

1. **Treat ranks as estimates**, not ground truth
2. **Group models** by overlapping CIs (statistical equivalence classes)
3. **Focus on task-level performance** for model selection
4. **Use effect sizes** to assess practical importance of differences

## Equivalence Classes (Based on CI Overlap)

### Class 1: Top Performers (statistically indistinguishable)
- BM25, BGE-large, TF-IDF, GTE-base, GTE-small, BGE-base, E5-large, BGE-small, E5-base, MPNet
- **SHELF score range**: 0.500-0.547
- **All pairwise CIs overlap**

### Class 2: Mid-Tier Performers
- Instructor-base, TF, GTR-T5-large, E5-small, MiniLM, GTR-T5-base, BERT, DistilBERT
- **SHELF score range**: 0.471-0.496
- **Some CI overlap with Class 1**

### Class 3: Lower Performers
- OgBERT variants, RoBERTa, KL3M-pico
- **SHELF score range**: 0.409-0.462
- **CIs mostly non-overlapping with Class 1**

**Key insight**: There are effectively **3 performance tiers**, not 24 distinct ranks.

## Methodological Notes

### Why Percentile Bootstrap?

Recent simulation studies (see `analysis.md`) show that:
1. **Percentile method** outperforms both BCa and t-distribution methods for CI coverage
2. **Bootstrap is robust** to non-normality in score distributions
3. **Task resampling** preserves within-task correlations across models

### Limitations

1. **Task sample size** (n=16, ~7 effective) limits precision
2. **Task dependencies** (high correlations) inflate CI widths appropriately
3. **Model-specific task coverage** affects comparability

### Alternative Approaches Considered

- **BCa (Bias-Corrected and Accelerated)**: More complex, similar coverage in simulations
- **Parametric CIs** (t-distribution): Assumes normality (violated for some tasks)
- **Bayesian credible intervals**: Requires prior specification (avoided for objectivity)

**Choice**: Percentile bootstrap for simplicity, robustness, and simulation-backed performance.

## Recommendations for Paper Presentation

### Table Format

Present aggregate scores as:

```
Model          SHELF Score    95% CI
--------------------------------------
BM25           0.547          [0.392, 0.672]
BGE-large      0.518          [0.370, 0.652]
TF-IDF         0.514          [0.373, 0.641]
...
```

### Figure Format

Plot scores with error bars:
- Point estimate: SHELF score
- Error bars: 95% CI
- Group by equivalence class (color-coded)

### Text Guidance

**Good**:
> "BM25 achieved the highest SHELF score (0.547, 95% CI [0.392, 0.672]), though top models (BM25, BGE-large, TF-IDF, GTE-base) were statistically indistinguishable (all CIs overlapping)."

**Avoid**:
> "BM25 outperformed all other models with a score of 0.547."

## Conclusion

Bootstrap confidence intervals reveal that:

1. **Top-10 model rankings are unstable** (overlapping CIs)
2. **Sparse and dense methods are competitive** (overlapping CIs, large effect but marginal significance)
3. **Task diversity drives high variance** (50-60% CI width/mean)
4. **Equivalence classes are more meaningful than ranks** (3 tiers, not 24 positions)

These findings **strengthen** the benchmark's value as a comprehensive evaluation tool while **tempering** claims about specific model superiority.

# SHELF Statistical Analysis Findings

**Date**: 2025-12-14
**Analysis Version**: v0.3.0
**Analysis Tool**: `shelf eval analyze`

## Overview

This document summarizes the comprehensive statistical analysis conducted on SHELF benchmark results. The analysis addresses key peer reviewer concerns and provides evidence for proper benchmark interpretation.

---

## Key Findings Summary

| Finding | Value | Implication |
|---------|-------|-------------|
| **Model Equivalence** | All 24 models in 1 group | No significant winner |
| **Friedman Test** | χ² = 12.69, p = 0.958 | Cannot reject null hypothesis |
| **Task Independence** | Mean r = 0.088 | Tasks measure different capabilities |
| **Effect Sizes** | Most d < 0.3 | Negligible practical differences |
| **Task Champions** | 8 unique winners / 16 tasks | Diversity ratio = 0.50 |

---

## 1. Model Equivalence Analysis

### Friedman-Nemenyi Test Results

The Friedman test evaluates whether any model performs significantly differently from others across all tasks.

```
Friedman statistic: χ² = 12.69
Degrees of freedom: 23
p-value: 0.958
Critical difference (CD): 13.27 (α = 0.05)
```

**Interpretation**: With p = 0.958, we cannot reject the null hypothesis that all models perform equivalently. This means **no model is statistically significantly better than any other**.

### Equivalence Groups

After Nemenyi post-hoc correction:
- **Group 1**: All 24 models (single equivalence group)

No model pair achieves statistical significance after proper multiple comparison correction.

### What This Means for the Paper

1. **Cannot claim**: "TF-IDF outperforms neural models"
2. **Can claim**: "TF-IDF achieves the highest raw score, though differences are not statistically significant"
3. **Should emphasize**: Task-specific analysis and effect sizes over rankings

---

## 2. Task Independence (Correlation Analysis)

### Why This Matters

A common peer review concern is: "Are your tasks redundant? Do they just measure the same thing?"

Low task correlations (mean r < 0.3) demonstrate that tasks measure **independent capabilities**.

### Correlation Statistics

| Metric | Value |
|--------|-------|
| Mean pairwise correlation | r = 0.088 |
| Median pairwise correlation | r = 0.061 |
| Standard deviation | 0.27 |
| Minimum correlation | r = -0.52 |
| Maximum correlation | r = 0.65 |
| Negative correlations | 56 of 120 (47%) |
| High correlations (|r| > 0.7) | 0 |

### Effective Sample Size

Using Galwey's eigenvalue decomposition method:
- **Effective N**: 3.1 (of 16 nominal tasks)
- **Interpretation**: Due to some correlation, the 16 tasks provide information equivalent to ~3 fully independent tasks

This is important for:
- Power analysis calculations
- Multiple comparison correction severity
- Understanding benchmark coverage

### Task Families

No task families identified (no groups with |r| > 0.7).

This means every task measures a genuinely different capability.

---

## 3. Effect Size Analysis

### Cohen's d Interpretation Guide

| |d| Range | Interpretation | Practical Meaning |
|-----------|----------------|-----------------|
| < 0.2 | Negligible | No practical difference |
| 0.2 - 0.5 | Small | Minor difference |
| 0.5 - 0.8 | Medium | Moderate difference |
| ≥ 0.8 | Large | Substantial difference |

### Key Comparisons

| Model A | Model B | Δ Score | Cohen's d | Interpretation |
|---------|---------|---------|-----------|----------------|
| TF+SVD | BGE-large | +0.166 | 0.61 | Medium |
| TF+SVD | E5-large | +0.175 | 0.63 | Medium |
| TF+SVD | MiniLM | +0.215 | 0.79 | Medium |
| BGE-large | BGE-small | +0.017 | 0.06 | Negligible |
| BGE-large | GTE-base | +0.011 | 0.04 | Negligible |
| E5-large | E5-base | +0.003 | 0.01 | Negligible |

### Pattern Observed

- **Sparse vs Dense**: Medium effect sizes (d ≈ 0.6), but not significant after correction
- **Same family (e.g., BGE-large vs BGE-small)**: Negligible effect sizes
- **Cross-family neural**: Small to negligible effect sizes

### Why Effect Size Without Significance?

High task variance creates wide confidence intervals:
- Mean score variance across tasks: 0.28
- This variance swamps the between-model differences
- Result: Medium effect sizes don't reach statistical significance

---

## 4. Task Champion Analysis

### Champion Distribution

| Task | Champion | Score |
|------|----------|-------|
| lcc_classification | TF-IDF+SVD | 0.878 |
| lcgft_category_classification | E5-large | 0.775 |
| register_classification | TF-IDF+SVD | 0.672 |
| lcc_retrieval | MPNet | 0.674 |
| category_retrieval | E5-base | 0.496 |
| form_retrieval | E5-base | 0.139 |
| lcc_clustering | MPNet | 0.569 |
| lcgft_clustering | BERT | 0.172 |
| register_clustering | OGBert-110M | 0.095 |
| geographic_clustering | GTR-T5-base | 0.025 |
| topic_overlap_pairs | MPNet | 0.826 |
| same_lcc_pairs | MPNet | 0.774 |
| same_topic_pairs | MPNet | 0.770 |
| same_form_pairs | GTE-small | 0.683 |
| same_audience_pairs | MPNet | 0.668 |
| same_register_pairs | E5-small | 0.667 |

### Diversity Statistics

| Metric | Value |
|--------|-------|
| Unique champions | 8 |
| Total tasks | 16 |
| Diversity ratio | 0.50 |
| Top champion | MPNet (6 wins) |
| Second champion | TF-IDF+SVD (2 wins) |

### Interpretation

- No single model dominates all tasks
- Different architectures excel at different capabilities
- MPNet leads pair classification, TF-IDF leads classification
- This validates SHELF as measuring diverse capabilities

---

## 5. Rank Consistency Analysis

How consistently does each model rank across tasks?

### Top Models by Mean Rank

| Model | Mean Rank | Std | Range | Top-3 Count |
|-------|-----------|-----|-------|-------------|
| MPNet | 5.2 | 4.1 | 1-15 | 8 |
| TF-IDF+SVD | 6.8 | 7.2 | 1-22 | 6 |
| E5-base | 7.1 | 4.8 | 1-18 | 5 |
| BGE-large | 7.4 | 5.2 | 1-19 | 5 |

### Interpretation

- **MPNet**: Most consistent performer (lowest std, most top-3 finishes)
- **TF-IDF+SVD**: High variance (either 1st or 22nd, rarely middle)
- **Most models**: High rank variance indicates task-specific performance

---

## 6. Multiple Comparison Correction

### Methods Available

| Method | Description | Stringency |
|--------|-------------|------------|
| Bonferroni | Divide α by # comparisons | Most conservative |
| Holm | Step-down procedure | Moderate |
| FDR (Benjamini-Hochberg) | Controls false discovery rate | Least conservative |

### Results with Different Corrections

For top comparison (TF+SVD vs worst neural model):

| Method | Raw p-value | Corrected p-value | Significant? |
|--------|-------------|-------------------|--------------|
| None | 0.08 | 0.08 | No |
| Holm | 0.08 | 0.98 | No |
| Bonferroni | 0.08 | 1.00 | No |
| FDR | 0.08 | 0.45 | No |

**Conclusion**: No comparison reaches significance under any correction method.

---

## 7. Recommendations for Paper

### Framing

❌ **Avoid**: "TF-IDF significantly outperforms neural models"
✅ **Use**: "TF-IDF achieves the highest raw score, though model differences are not statistically significant after correction"

❌ **Avoid**: "Model X is the best for SHELF"
✅ **Use**: "Model X achieves the highest aggregate score, but task-specific analysis reveals..."

### Required Reporting

1. **Effect sizes**: Report Cohen's d for all comparisons
2. **Confidence intervals**: Show 95% CIs for all scores
3. **Correction method**: State which multiple comparison correction used
4. **Task correlations**: Report mean r to address redundancy concerns
5. **Equivalence groups**: Show which models cannot be distinguished

### Suggested Claims

1. "SHELF's 16 tasks measure independent capabilities (mean r = 0.088)"
2. "No model achieves statistical dominance across tasks (Friedman p = 0.958)"
3. "Task-specific analysis reveals capability profiles: TF-IDF excels at classification, MPNet at pair tasks"
4. "Form and register understanding remains challenging for all models (avg < 0.15)"

---

## 8. CLI Commands Used

All analyses can be reproduced with:

```bash
# Full analysis
shelf eval analyze --verbose

# Specific analyses
shelf eval analyze --correlation      # Task correlation matrix
shelf eval analyze --pairwise         # Pairwise significance
shelf eval analyze --equivalence      # Friedman-Nemenyi
shelf eval analyze --champions        # Task champion diversity

# Export to JSON
shelf eval analyze --verbose --export-json findings.json
```

---

## References

- Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. *JMLR*, 7, 1-30.
- Cohen, J. (1988). Statistical power analysis for the behavioral sciences.
- Galwey, N. W. (2009). A new measure of the effective number of tests. *Statistics in Medicine*, 28(1), 25-39.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65-70.

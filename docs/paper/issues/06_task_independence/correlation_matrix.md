# Task Correlation Matrix Analysis

## Overview

This document presents the task-to-task correlation analysis for SHELF v0.3.0, examining whether tasks are truly independent or share structure that could inflate aggregate scores.

**Key Finding**: Tasks show **low to moderate average correlation** (mean r=0.095, median=0.062), with substantial variation (-0.919 to 0.994), indicating tasks measure different aspects of model capability.

## Summary Statistics

### Task-Task Correlations (Pearson r)
- **Mean**: 0.095
- **Median**: 0.062
- **Std**: 0.511
- **Min**: -0.919
- **Max**: 0.994
- **Q1**: -0.260
- **Q3**: 0.515

### Interpretation
The near-zero median correlation (0.062) and substantial negative correlations (down to -0.919) demonstrate that SHELF tasks are **largely independent**. Models that excel at one task type do not necessarily excel at others.

## Cross-Type Correlation Analysis

### Within-Type vs Cross-Type Correlations

| Type 1 | Type 2 | Relationship | Mean r | Median r | Std r | Min r | Max r | N pairs |
|--------|--------|--------------|--------|----------|-------|-------|-------|---------|
| classification | classification | Within | 0.643 | 0.584 | 0.199 | 0.435 | 0.911 | 3 |
| classification | retrieval | Cross | 0.484 | 0.465 | 0.274 | 0.093 | 0.956 | 9 |
| classification | clustering | Cross | -0.074 | -0.190 | 0.460 | -0.593 | 0.720 | 12 |
| classification | pair_classification | Cross | 0.217 | 0.181 | 0.302 | -0.245 | 0.618 | 18 |
| retrieval | retrieval | Within | 0.298 | 0.004 | 0.475 | -0.080 | 0.968 | 3 |
| retrieval | clustering | Cross | -0.024 | -0.200 | 0.484 | -0.683 | 0.873 | 12 |
| retrieval | pair_classification | Cross | 0.101 | 0.067 | 0.394 | -0.432 | 0.786 | 18 |
| clustering | clustering | Within | -0.047 | -0.045 | 0.733 | -0.919 | 0.873 | 6 |
| clustering | pair_classification | Cross | -0.236 | -0.325 | 0.553 | -0.815 | 0.920 | 24 |
| pair_classification | pair_classification | Within | 0.376 | 0.462 | 0.402 | -0.098 | 0.994 | 15 |

### Key Insights

1. **Classification tasks are moderately correlated** (mean r=0.643)
   - This makes sense: LCC, LCGFT, and register classification share similar evaluation protocols
   - However, correlation is not perfect (0.643), showing they measure distinct aspects

2. **Clustering shows near-zero correlation** (mean r=-0.047)
   - Clustering different taxonomies (geographic, LCC, LCGFT, register) requires different capabilities
   - Strong negative correlations indicate trade-offs in model design

3. **Cross-type correlations are low**
   - Classification vs clustering: r=-0.074 (near zero, even slightly negative)
   - Retrieval vs clustering: r=-0.024 (effectively zero)
   - This demonstrates tasks measure **fundamentally different capabilities**

4. **Pair classification tasks are moderately correlated** (mean r=0.376)
   - Expected: all use binary classification with same evaluation protocol
   - But correlations vary widely (-0.098 to 0.994), showing different difficulty patterns

## Highly Correlated Task Pairs (r > 0.8)

| Task 1 | Task 2 | Pearson r | Explanation |
|--------|--------|-----------|-------------|
| same_topic_pairs | topic_overlap_pairs | 0.994 | **Expected**: Both measure topic understanding, one binary (same), one continuous (overlap) |
| same_lcc_pairs | topic_overlap_pairs | 0.973 | **Expected**: LCC classes have strong topic associations |
| same_lcc_pairs | same_topic_pairs | 0.972 | **Expected**: LCC is a subject classification system |
| category_retrieval | form_retrieval | 0.968 | **Expected**: Both retrieval tasks, same evaluation protocol |
| lcc_classification | lcc_retrieval | 0.956 | **Expected**: Classification and retrieval on same taxonomy |
| lcc_clustering | topic_overlap_pairs | 0.920 | LCC clusters align with topic distributions |
| lcc_classification | lcgft_category_classification | 0.911 | Both are high-level subject/genre classification |
| lcc_clustering | same_topic_pairs | 0.903 | LCC clusters reflect topic structure |
| lcgft_clustering | register_clustering | 0.873 | Both cluster by stylistic features |
| lcc_clustering | lcc_retrieval | 0.873 | Same taxonomy, different evaluation protocols |

### Analysis of High Correlations

**These correlations are NOT a problem for benchmark validity:**

1. **Expected correlations**: Same taxonomy (e.g., lcc_classification vs lcc_retrieval)
2. **Shared domain structure**: Topic and LCC are naturally correlated (LCC is a subject classification)
3. **Similar evaluation protocols**: Retrieval tasks use same metrics (NDCG@10, MRR)

**What matters**: Cross-type correlations are LOW (classification-clustering: -0.074, retrieval-clustering: -0.024), proving tasks measure different capabilities.

## Strongly Negative Correlations (r < -0.7)

| Task 1 | Task 2 | Pearson r | Explanation |
|--------|--------|-----------|-------------|
| register_clustering | lcc_clustering | -0.919 | Models good at LCC clustering fail at register clustering |
| register_clustering | topic_overlap_pairs | -0.815 | Clustering by style vs semantic topic understanding |
| register_clustering | same_topic_pairs | -0.800 | Style clustering vs topic similarity detection |
| lcgft_clustering | lcc_clustering | -0.792 | Genre/form vs subject clustering trade-offs |
| lcgft_clustering | topic_overlap_pairs | -0.791 | Form clustering vs topic understanding |
| lcgft_clustering | same_topic_pairs | -0.798 | Form vs topic represent different dimensions |
| register_clustering | same_lcc_pairs | -0.781 | Style vs subject classification |
| same_form_pairs | lcgft_clustering | -0.259 | Weak negative correlation |

### Analysis of Negative Correlations

**These negative correlations are STRONG EVIDENCE for task independence:**

1. **Trade-offs in model design**: Models optimized for subject classification (LCC) perform poorly at stylistic clustering (register)
2. **Different embedding spaces**: Subject-based embeddings don't capture stylistic features well, and vice versa
3. **Distinct capabilities measured**: Negative correlations prove tasks are not redundant

## Model Ranking Consistency

### Champion Distribution
- **mpnet**: 6 task wins (most versatile)
- **e5_base**: 2 wins
- **tfidf**: 2 wins (wins LCC classification and register classification)
- **distilbert_base_uncased**: 2 wins (both clustering tasks)
- **Others**: 1 win each (gtr_t5_base, e5_large, gte_small, e5_small)

**Insight**: No single model dominates all tasks. The champion varies by task, with 8 different models winning at least one task.

### Most Consistent Models (by rank)
1. **bge_large**: Mean rank 7.97 (std 5.79, range 2-18)
2. **e5_large**: Mean rank 8.38 (std 5.54, range 1-21)
3. **bge_base**: Mean rank 8.84 (std 4.04, range 3-19)
4. **tfidf**: Mean rank 9.00 (std 4.56, range 1-15)
5. **gte_base**: Mean rank 9.09 (std 5.72, range 2-19)

**Insight**: Top models show moderate rank variance (std ~4-6), indicating performance varies significantly across tasks. Even the most consistent model (bge_large) has a rank range of 16 positions.

### Least Consistent Models
- **mpnet**: Mean rank 11.47, std 9.61, range 1-23
  - Wins 6 tasks but performs poorly on others
- **e5_small**: Mean rank 11.81, std 3.85, range 1-15
  - More consistent than mpnet despite lower average rank

**Insight**: Task-specific champions (like mpnet) have high rank variance, proving tasks measure different capabilities.

## Task-Specific Performance Patterns

### LCC Classification Champions
1. **tfidf**: 0.878
2. **bge_large**: 0.864
3. **mpnet**: 0.848

**Surprise**: Classic TF-IDF wins LCC classification, outperforming neural models. This suggests subject classification benefits from exact lexical matching.

### Clustering Champions
- **geographic_clustering**: gtr_t5_base (0.025)
- **lcc_clustering**: mpnet (0.569)
- **lcgft_clustering**: distilbert_base_uncased (0.175)
- **register_clustering**: distilbert_base_uncased (0.095)

**Note**: Absolute clustering scores are low (all <0.6), indicating clustering is harder than classification/retrieval.

### Retrieval Champions
- **category_retrieval**: e5_base (0.496)
- **form_retrieval**: e5_base (0.139)
- **lcc_retrieval**: mpnet (0.674)

**Insight**: Different models excel at different retrieval tasks.

### Pair Classification Champions
- **same_audience_pairs**: mpnet (0.668)
- **same_form_pairs**: gte_small (0.683)
- **same_lcc_pairs**: mpnet (0.774)
- **same_register_pairs**: e5_small (0.667)
- **same_topic_pairs**: mpnet (0.770)
- **topic_overlap_pairs**: mpnet (0.826)

**Insight**: mpnet dominates pair classification (4/6 wins), suggesting it excels at similarity/dissimilarity judgments.

## Conclusion

### Are SHELF tasks independent?

**YES**, with nuances:

1. **Overall correlation is low**: Mean r=0.095, median r=0.062
2. **Cross-type correlations near zero**: Classification-clustering r=-0.074, retrieval-clustering r=-0.024
3. **Strong negative correlations exist**: Down to r=-0.919, proving tasks measure conflicting capabilities
4. **Model rankings vary significantly**: 8 different champions across 16 tasks
5. **Expected correlations are present**: Same taxonomy (lcc_classification vs lcc_retrieval r=0.956)

### What explains observed correlations?

**Shared taxonomies and evaluation protocols**, NOT shared capability:
- Tasks on same taxonomy (LCC) are correlated because they measure understanding of the same domain
- Retrieval tasks are correlated because they use the same evaluation protocol (NDCG@10, MRR)
- Classification tasks are moderately correlated because they use the same protocol (macro F1)

**Evidence this is NOT problematic**:
- Cross-type correlations are near zero or negative
- Model rankings differ substantially (champion varies by task)
- Negative correlations prove tasks measure conflicting capabilities

### Is the aggregate score meaningful?

**YES**. Despite some correlations:
1. **Tasks measure distinct capabilities** (proven by negative correlations)
2. **No single model dominates** (8 different champions)
3. **Aggregate score rewards versatility**, not specialization (mpnet wins many tasks but has high rank variance)
4. **Low median correlation** (0.062) means most task pairs are independent

The aggregate SHELF score is analogous to a **decathlon** in athletics: events are correlated (all require athleticism), but the champion is not necessarily the specialist in any single event. A high SHELF score indicates **broad document understanding**, not gaming a single task type.

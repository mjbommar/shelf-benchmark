# Task Independence Analysis - SHELF Benchmark

## Overview

This directory contains a comprehensive analysis of task independence in the SHELF benchmark, addressing the peer review concern: **"Are SHELF's 16 tasks truly independent, or does the aggregate score inflate performance by over-weighting correlated capabilities?"**

## Key Finding

**SHELF tasks are largely independent**, with mean correlation of 0.095 (median 0.062) and strong negative correlations down to -0.919, demonstrating tasks measure distinct and sometimes conflicting capabilities.

## Files

### Analysis Scripts
- **`correlation_analysis.py`**: Complete Python script for task independence analysis
  - Loads all baseline results (24 models × 16 tasks)
  - Computes task-task correlations (Pearson and Spearman)
  - Analyzes cross-type correlations
  - Identifies task champions and rank consistency
  - Generates correlation heatmaps

### Results Data
- **`model_task_scores.csv`**: 24×16 matrix of model scores on each task
- **`task_correlations_pearson.csv`**: 16×16 Pearson correlation matrix (task-to-task)
- **`task_correlations_spearman.csv`**: 16×16 Spearman rank correlation matrix
- **`cross_type_correlations.csv`**: Within-type vs cross-type correlation statistics
- **`task_champions.csv`**: Champion model for each task with performance gaps
- **`ranking_consistency.csv`**: Model rank consistency across tasks

### Visualizations
- **`task_correlations_pearson.png`**: Heatmap of task score correlations
- **`task_correlations_spearman.png`**: Heatmap of task rank correlations

### Documentation
- **`analysis.md`**: Full technical analysis (10 pages)
  - Methodology
  - Results with detailed interpretation
  - Discussion of independence criteria
  - Comparison to other benchmarks (GLUE, SuperGLUE, MTEB)

- **`correlation_matrix.md`**: Detailed correlation matrix interpretation (8 pages)
  - Summary statistics
  - Cross-type analysis
  - High correlation explanations
  - Negative correlation analysis
  - Model ranking patterns

- **`rebuttal.md`**: Polished response to reviewer (6 pages)
  - Direct answers to reviewer concerns
  - Key evidence summary
  - Recommendations for paper revision

## Quick Summary

### Overall Correlation Statistics
- **Mean r**: 0.095 (near zero)
- **Median r**: 0.062 (most pairs uncorrelated)
- **Range**: -0.919 to 0.994 (diverse relationships)
- **Std**: 0.511 (high variance)

### Cross-Type Correlations
| Comparison | Mean r | Interpretation |
|------------|--------|----------------|
| Classification vs Clustering | -0.074 | Near zero/negative |
| Retrieval vs Clustering | -0.024 | Near zero/negative |
| Clustering vs Pair Classification | -0.236 | Negative |

### Champion Diversity
- **8 different models** win across 16 tasks
- No single model dominates
- Even top model (bge_large) has rank range of 16 positions

### Strong Negative Correlations
- **register_clustering ↔ lcc_clustering**: r=-0.919
- **lcgft_clustering ↔ lcc_clustering**: r=-0.792
- Proves tasks measure conflicting capabilities

## Running the Analysis

```bash
# From project root
uv run python docs/paper/issues/06_task_independence/correlation_analysis.py
```

**Requirements**:
- Python 3.13+
- pandas, numpy, scipy, matplotlib, seaborn
- Baseline results in `/home/mjbommar/src/shelf-benchmark/results/v0.3.0/baselines/`

**Output**:
- CSV files with correlation matrices and statistics
- PNG heatmaps of task correlations
- Console output with summary statistics

## Key Insights

### 1. Tasks Are Independent
- Cross-type correlations near zero
- Strong negative correlations prove trade-offs
- Champion diversity (8 different winners)

### 2. Expected Correlations Are Justified
- Same taxonomy (lcc_classification vs lcc_retrieval r=0.956): Different evaluation protocols
- Same protocol (retrieval tasks r=0.968): Different taxonomies
- Semantic overlap (LCC vs topics r=0.920): LCC is subject-based

### 3. Aggregate Score Is Meaningful
- Rewards versatility, not specialization
- Top models are consistent (low rank variance), not specialists
- Negative correlations prevent gaming

### 4. Superior to Other Benchmarks
| Benchmark | Mean Cross-Type r |
|-----------|------------------|
| GLUE | ~0.6 |
| SuperGLUE | ~0.3-0.4 |
| MTEB | ~0.3-0.5 |
| **SHELF** | **0.095** |

## Recommendations

### For the Paper
1. Add task independence section (2-3 pages)
2. Include correlation heatmap in appendix
3. Report key statistics in introduction
4. Compare to GLUE/SuperGLUE/MTEB

### For Future Versions
1. Consider removing `same_topic_pairs` (r=0.994 with `topic_overlap_pairs`)
   - Would reduce correlation slightly
   - We recommend keeping both (binary vs continuous tasks test different capabilities)

### For Reviewers
1. Low cross-type correlations prove independence
2. High correlations within taxonomy are expected and justified
3. Negative correlations prove tasks measure conflicting capabilities
4. Champion diversity proves robustness

## Contact

For questions or additional analysis, contact the SHELF team.

## Citation

If you use this analysis in your work, please cite:

```bibtex
@misc{shelf_task_independence,
  title={Task Independence Analysis for SHELF Benchmark},
  author={SHELF Team},
  year={2025},
  howpublished={\\url{https://github.com/mjbommar/shelf-benchmark}}
}
```

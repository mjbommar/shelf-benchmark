# Task Independence Analysis - Complete Index

## Quick Navigation

### Start Here
1. **[EXECUTIVE_SUMMARY.md](EXECUTIVE_SUMMARY.md)** - 2-page bottom line (START HERE!)
2. **[README.md](README.md)** - Overview and file guide
3. **[rebuttal.md](rebuttal.md)** - Polished response to reviewer

### Deep Dives
4. **[analysis.md](analysis.md)** - Full 10-page technical analysis
5. **[correlation_matrix.md](correlation_matrix.md)** - Detailed 8-page interpretation

### Code
6. **[correlation_analysis.py](correlation_analysis.py)** - Main analysis script
7. **[create_summary_plots.py](create_summary_plots.py)** - Visualization script

### Data Files
8. **[model_task_scores.csv](model_task_scores.csv)** - 24×16 score matrix
9. **[task_correlations_pearson.csv](task_correlations_pearson.csv)** - Task correlation matrix
10. **[task_correlations_spearman.csv](task_correlations_spearman.csv)** - Rank correlation matrix
11. **[cross_type_correlations.csv](cross_type_correlations.csv)** - Within/cross-type stats
12. **[task_champions.csv](task_champions.csv)** - Champion per task
13. **[ranking_consistency.csv](ranking_consistency.csv)** - Model rank variance

### Visualizations
14. **[task_correlations_pearson.png](task_correlations_pearson.png)** - Main heatmap (1.1MB)
15. **[task_correlations_spearman.png](task_correlations_spearman.png)** - Rank heatmap (1.1MB)
16. **[correlation_summary.png](correlation_summary.png)** - 4-panel summary (541KB)
17. **[champion_analysis.png](champion_analysis.png)** - Champion distribution (272KB)

## Reading Paths

### For Busy Reviewers (10 minutes)
1. EXECUTIVE_SUMMARY.md
2. correlation_summary.png
3. task_correlations_pearson.png

### For Paper Authors (30 minutes)
1. EXECUTIVE_SUMMARY.md
2. rebuttal.md
3. correlation_matrix.md (key sections)
4. All visualizations

### For Deep Technical Review (2 hours)
1. README.md
2. analysis.md
3. correlation_matrix.md
4. correlation_analysis.py (review methodology)
5. All data files and visualizations

## Key Statistics (Quick Reference)

### Overall Correlation
- **Mean**: 0.095
- **Median**: 0.062
- **Range**: [-0.919, 0.994]
- **Std**: 0.511

### Cross-Type Correlations
- Classification vs Clustering: r = -0.074
- Retrieval vs Clustering: r = -0.024
- Clustering vs Pair: r = -0.236

### Champion Diversity
- 8 different champions across 16 tasks
- Top model rank range: 16 positions
- No single model dominates

### Negative Correlations
- 55 task pairs (45.8%) negatively correlated
- Strongest: r = -0.919 (register vs LCC clustering)
- 7 pairs with r < -0.7

## File Sizes

**Total**: ~2.3 MB

**Large files**:
- Heatmaps: ~1.1 MB each (high-res PNG)
- Summary plots: 541 KB, 272 KB
- Documentation: ~50 KB total
- Data: ~20 KB total

## Reproducibility

### Running the Analysis
```bash
# From project root
uv run python docs/paper/issues/06_task_independence/correlation_analysis.py
uv run python docs/paper/issues/06_task_independence/create_summary_plots.py
```

### Requirements
- Python 3.13+
- pandas, numpy, scipy, matplotlib, seaborn
- Baseline results in `/home/mjbommar/src/shelf-benchmark/results/v0.3.0/baselines/`

### Output
- All CSV files regenerated
- All PNG visualizations regenerated
- Console output with summary statistics

## Citation

If you use this analysis:

```bibtex
@misc{shelf_task_independence_2025,
  title={Task Independence Analysis for SHELF Benchmark},
  author={SHELF Team},
  year={2025},
  howpublished={\url{https://github.com/mjbommar/shelf-benchmark}}
}
```

## Questions?

See individual file headers or contact the SHELF team.

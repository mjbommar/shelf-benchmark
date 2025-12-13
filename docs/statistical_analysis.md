# Statistical Analysis for SHELF Benchmark

This document describes the statistical analysis framework for comparing model performance on SHELF benchmark tasks.

## Overview

The `shelf.evaluate.analysis` module provides tools for:
1. **Model comparison**: Compare which models are statistically significantly better
2. **Commit comparison**: Compare performance across different data generation commits
3. **Generation model comparison**: Compare performance by generating model (gpt-5.1 vs gpt-5.2)
4. **Facet analysis**: Analyze how performance varies by dataset facets (form, LCC, register, etc.)

Key insight from MTEB research: Many top models on leaderboards are statistically equivalent despite different average scores. This framework helps identify when differences are truly meaningful.

## Use Cases

### 1. Compare Models on a Benchmark

```python
from shelf.evaluate.analysis import compare_multiple, load_results_from_dir

# Load results from directory
results = load_results_from_dir("results/lcc_classification/")

# Extract scores by model
from shelf.evaluate.analysis.comparison import extract_scores
scores = extract_scores(results, metric="macro_f1", group_by="model")

# Compare all models
comparison = compare_multiple(scores, metric="macro_f1")
print(comparison.summary())

# Key outputs:
# - ranking: Models ranked by mean score
# - friedman_nemenyi: Friedman test + Nemenyi post-hoc results
# - equivalence_groups: Groups of models that are NOT significantly different
```

### 2. Compare Two Specific Models

```python
from shelf.evaluate.analysis import compare_two

# Direct comparison
result = compare_two(
    scores_a=model_a_scores,
    scores_b=model_b_scores,
    name_a="BERT",
    name_b="RoBERTa",
    metric="macro_f1",
    test="bootstrap",  # or "wilcoxon", "t-test", "permutation"
    alpha=0.05,
)

print(result.summary())
# Shows:
# - Mean scores for both models
# - Difference and relative difference %
# - Statistical significance (p-value, effect size)
# - Bootstrap confidence interval
# - Winner (if significant)
```

### 3. Compare Performance by Commit ID

```python
from shelf.evaluate.analysis import compare_commits

# Compare two data generation commits
result = compare_commits(
    results=all_results,
    metric="macro_f1",
    commit_a="abc123",
    commit_b="def456",
)

print(result.summary())
```

### 4. Compare by Generating Model

```python
from shelf.evaluate.analysis import compare_generation_models

# Check if generating model affects performance
result = compare_generation_models(
    results=all_results,
    metric="macro_f1",
)

print(result.summary())
```

### 5. Analyze Performance by Facet

```python
from shelf.evaluate.analysis import compare_by_facet

# Analyze performance by document form
analysis = compare_by_facet(
    results=all_results,
    metric="macro_f1",
    facet="form",  # or "lcc", "register", "audience", etc.
    model="BERT",  # optional: filter to specific model
)

print(analysis.summary())
# Shows:
# - Best/worst performing facet values
# - Score range across facets
# - Kruskal-Wallis test for variance
# - Ranking of all facet values
```

## Evaluator Configuration

Evaluators support filtering and stratification:

```python
from shelf.evaluate.evaluators import ClassificationEvaluator
from shelf.evaluate.registry import get_task

task_spec = get_task("lcc_classification")

# Evaluate with filtering (only specific commit)
evaluator = ClassificationEvaluator(
    task_spec,
    filter_by={"git_commit": "abc123"},
)

# Evaluate with stratification (compute metrics per form)
evaluator = ClassificationEvaluator(
    task_spec,
    stratify_by="form",  # or ["form", "register"] for multiple
)

result = evaluator.evaluate_classifier(classifier, split="test")

# Access stratified metrics
print(result.stratified_metrics)
# {"form=lecture": {"macro_f1": 0.85, ...}, "form=map": {...}, ...}
```

## Statistical Tests

### Paired Tests (Same Samples)

For comparing models evaluated on the same data:

| Test | Use Case | Assumptions |
|------|----------|-------------|
| **Wilcoxon signed-rank** | Non-parametric, robust | None (distribution-free) |
| **Paired t-test** | Parametric | Differences normally distributed |
| **Paired permutation** | Assumption-free | None |
| **Bootstrap** | Most flexible | None |

```python
from shelf.evaluate.analysis import wilcoxon_test, bootstrap_paired_difference

# Wilcoxon (recommended default)
result = wilcoxon_test(scores_a, scores_b, alpha=0.05)

# Bootstrap (most detailed)
result = bootstrap_paired_difference(scores_a, scores_b, n_bootstrap=10000)
```

### Multiple Comparison (Many Models)

For comparing 3+ models:

```python
from shelf.evaluate.analysis import friedman_nemenyi_test

# Standard approach for ML benchmarks
result = friedman_nemenyi_test(
    data={"BERT": [...], "RoBERTa": [...], "MiniLM": [...]},
    alpha=0.05,
)

# Key outputs:
# - friedman_p_value: Overall significance
# - critical_difference: CD for Nemenyi test
# - mean_ranks: Rank for each model
# - significant_pairs: Pairs that differ significantly
```

### McNemar's Test (Binary Outcomes)

For comparing classification accuracy:

```python
from shelf.evaluate.analysis import mcnemar_test

# Compare correct/incorrect predictions
result = mcnemar_test(
    correct_a=[1, 0, 1, 1, 0, ...],  # 1=correct, 0=incorrect
    correct_b=[1, 1, 1, 0, 0, ...],
)
```

## Effect Size Interpretation

Cohen's d effect sizes are computed automatically:

| |d| | Interpretation |
|-----|----------------|
| < 0.2 | Negligible |
| 0.2 - 0.5 | Small |
| 0.5 - 0.8 | Medium |
| >= 0.8 | Large |

## Visualization

```python
from shelf.evaluate.analysis import (
    plot_critical_difference,
    plot_pairwise_significance_heatmap,
    plot_score_distribution,
    plot_comparison_summary,
)

# Critical difference diagram
fig = plot_critical_difference(multiple_comparison_result)
fig.savefig("cd_diagram.png")

# Generate all comparison plots
figs = plot_comparison_summary(
    result=multiple_comparison_result,
    save_dir="plots/",
)
```

## Data Provenance

Results automatically track provenance for reproducibility:

```python
# Access provenance in results
result = evaluator.evaluate_classifier(classifier)

# Data provenance
print(result.data_provenance.unique_commits)
print(result.data_provenance.unique_models)
print(result.data_provenance.primary_commit)

# Context
print(result.context.shelf_version)
print(result.context.sklearn_version)
print(result.context.dataset_checksum)
print(result.context.data_commit)
```

## Report Generation

```python
from shelf.evaluate.analysis import generate_comparison_report, print_comparison_summary

# Generate full report
report = generate_comparison_report(
    comparison_result,
    format="markdown",  # or "json", "text"
)

# Print summary to console
print_comparison_summary(comparison_result)
```

## Available Facets for Stratification

The following fields can be used with `filter_by` and `stratify_by`:

| Field | Description |
|-------|-------------|
| `lcc` | Library of Congress Classification (A-Z) |
| `form` | Document form (lecture, map, etc.) |
| `form_category` | Broader form category |
| `topic` | Topic |
| `region` | Geographic region |
| `audience` | Target audience |
| `register` | Writing register |
| `model` | Generating model (gpt-5.1, etc.) |
| `git_commit` | Data generation commit |

## Best Practices

1. **Always report confidence intervals**: Point estimates alone can be misleading.

2. **Use appropriate tests**: Bootstrap is most flexible; Wilcoxon for paired non-parametric.

3. **Report multiple metrics**: Don't cherry-pick the best-looking metric.

4. **Check for equivalence groups**: Models in the same group are statistically indistinguishable.

5. **Consider effect sizes**: Statistical significance doesn't mean practical importance.

6. **Track provenance**: Always save the full context for reproducibility.

## References

- [MTEB Leaderboard Best Practices](https://huggingface.co/blog/lyon-nlp-group/mteb-leaderboard-best-practices)
- Efron & Tibshirani (1993). An Introduction to the Bootstrap.
- Berg-Kirkpatrick et al. (2012). An Empirical Investigation of Statistical Significance in NLP.
- Demšar (2006). Statistical Comparisons of Classifiers over Multiple Data Sets.

"""Statistical analysis for SHELF benchmark comparisons.

This module provides tools for:
- Comparing model performance with statistical significance tests
- Comparing results across different data generations (commit IDs)
- Comparing performance by facet (form, LCC, register, etc.)
- Multiple comparison correction for comparing many models
- Bootstrap confidence intervals for metrics
- Critical difference diagrams for visualization

Key use cases:
1. Compare which evaluated models are statistically better on a benchmark
2. Compare whether performance varies by dataset commit ID
3. Compare whether performance varies by generating model (gpt-5.1 vs gpt-5.2)
4. Analyze how performance varies by dataset facets (form, LCC, register)

Key insight from MTEB research: Many top models on leaderboards are
statistically equivalent despite different average scores. This module
helps identify when differences are truly meaningful.

References:
- MTEB Leaderboard Best Practices: https://huggingface.co/blog/lyon-nlp-group/mteb-leaderboard-best-practices
- scikit-posthocs: https://scikit-posthocs.readthedocs.io/
- Friedman test + Nemenyi post-hoc: standard for ML benchmark comparison
"""

from shelf.evaluate.analysis.comparison import (
    ComparisonResult,
    FacetAnalysisResult,
    MultipleComparisonResult,
    compare_by_facet,
    compare_by_group,
    compare_commits,
    compare_generation_models,
    compare_multiple,
    compare_two,
    load_results_from_dir,
)
from shelf.evaluate.analysis.bootstrap import (
    bootstrap_ci,
    bootstrap_paired_difference,
)
from shelf.evaluate.analysis.significance import (
    paired_permutation_test,
    mcnemar_test,
    wilcoxon_test,
    friedman_nemenyi_test,
)
from shelf.evaluate.analysis.report import (
    generate_comparison_report,
    print_comparison_summary,
)
from shelf.evaluate.analysis.plots import (
    plot_critical_difference,
    plot_confidence_intervals,
    plot_pairwise_significance_heatmap,
    plot_score_distribution,
    plot_comparison_summary,
)

__all__ = [
    # Comparison functions
    "ComparisonResult",
    "FacetAnalysisResult",
    "MultipleComparisonResult",
    "compare_by_facet",
    "compare_by_group",
    "compare_commits",
    "compare_generation_models",
    "compare_multiple",
    "compare_two",
    "load_results_from_dir",
    # Bootstrap
    "bootstrap_ci",
    "bootstrap_paired_difference",
    # Significance tests
    "paired_permutation_test",
    "mcnemar_test",
    "wilcoxon_test",
    "friedman_nemenyi_test",
    # Reports
    "generate_comparison_report",
    "print_comparison_summary",
    # Plots
    "plot_critical_difference",
    "plot_confidence_intervals",
    "plot_pairwise_significance_heatmap",
    "plot_score_distribution",
    "plot_comparison_summary",
]

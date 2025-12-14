#!/usr/bin/env python3
"""
Statistical Significance Testing for SHELF Benchmark Results

This script performs comprehensive statistical significance testing on SHELF benchmark
results to validate the reported performance differences between models.

Key analyses:
1. Bootstrap confidence intervals for aggregate SHELF scores
2. Paired statistical tests (t-test, Wilcoxon) for model comparisons
3. Effect sizes (Cohen's d) for sparse vs. dense model comparison
4. Multiple comparison corrections (Bonferroni, Holm)
5. Task correlation analysis
6. Per-task variance analysis

References:
- deep-significance: https://github.com/Kaleidophon/deep-significance
- Bootstrap CI best practices: https://arxiv.org/abs/2205.11134
- NLP significance testing: https://cs.stanford.edu/people/wmorgan/sigtest.pdf
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from scipy import stats
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore")

# Constants
RESULTS_DIR = Path("/home/mjbommar/src/shelf-benchmark/results/v0.3.0/baselines")
OUTPUT_DIR = Path("/home/mjbommar/src/shelf-benchmark/docs/paper/issues/05_statistical_significance")
RANDOM_SEED = 42
N_BOOTSTRAP = 10000
ALPHA = 0.05

# Task types and their primary metrics
TASK_METRICS = {
    "classification": "macro_f1",
    "retrieval": "ndcg@10",
    "clustering": "v_measure",
    "pair_classification": "f1"
}

# Models of interest for key comparisons
SPARSE_MODELS = ["tf", "tfidf", "bm25"]
DENSE_MODELS = ["bge_large", "bge_base", "gte_base", "e5_large", "mpnet"]
SMALL_MODELS = ["bge_small", "e5_small", "gte_small", "minilm"]


def load_all_results() -> Dict[str, Dict[str, float]]:
    """Load all model results and extract per-task scores."""
    model_scores = defaultdict(dict)

    # Load individual result files
    for result_file in RESULTS_DIR.glob("*.json"):
        if result_file.name in ["summary.json", "manifest.json"]:
            continue

        with open(result_file) as f:
            data = json.load(f)

        model = data.get("model_key")
        task = data.get("task")
        task_type = data.get("task_type")

        if not all([model, task, task_type]):
            continue

        # Get primary metric for this task type
        primary_metric = TASK_METRICS.get(task_type)
        if not primary_metric:
            continue

        # Extract primary score
        metrics = data.get("metrics", {})
        if primary_metric in metrics:
            score = metrics[primary_metric]
            model_scores[model][task] = score

    return dict(model_scores)


def compute_shelf_score(task_scores: Dict[str, float]) -> float:
    """Compute aggregate SHELF score from task scores."""
    if not task_scores:
        return 0.0
    return np.mean(list(task_scores.values()))


def bootstrap_confidence_interval(
    task_scores: Dict[str, float],
    n_bootstrap: int = N_BOOTSTRAP,
    confidence_level: float = 0.95
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for SHELF score.

    Uses percentile method as recommended by recent simulation studies.

    Returns:
        (mean, lower_ci, upper_ci)
    """
    scores = list(task_scores.values())
    n_tasks = len(scores)

    if n_tasks == 0:
        return (0.0, 0.0, 0.0)

    np.random.seed(RANDOM_SEED)
    bootstrap_samples = []

    for _ in range(n_bootstrap):
        # Resample tasks with replacement
        resampled = np.random.choice(scores, size=n_tasks, replace=True)
        bootstrap_samples.append(np.mean(resampled))

    bootstrap_samples = np.array(bootstrap_samples)

    # Percentile method
    alpha = 1 - confidence_level
    lower_ci = np.percentile(bootstrap_samples, 100 * alpha / 2)
    upper_ci = np.percentile(bootstrap_samples, 100 * (1 - alpha / 2))
    mean = np.mean(scores)

    return (mean, lower_ci, upper_ci)


def paired_test(
    scores1: Dict[str, float],
    scores2: Dict[str, float],
    test_type: str = "both"
) -> Dict[str, float]:
    """
    Perform paired statistical tests between two models.

    Args:
        scores1: Task scores for model 1
        scores2: Task scores for model 2
        test_type: "t-test", "wilcoxon", or "both"

    Returns:
        Dictionary with test statistics and p-values
    """
    # Get common tasks
    common_tasks = set(scores1.keys()) & set(scores2.keys())
    if len(common_tasks) < 2:
        return {"error": "Insufficient common tasks"}

    # Extract paired scores
    pairs1 = [scores1[task] for task in sorted(common_tasks)]
    pairs2 = [scores2[task] for task in sorted(common_tasks)]

    results = {
        "n_tasks": len(common_tasks),
        "mean_diff": np.mean(pairs1) - np.mean(pairs2),
        "std_diff": np.std([p1 - p2 for p1, p2 in zip(pairs1, pairs2)])
    }

    # Paired t-test (assumes normality)
    if test_type in ["t-test", "both"]:
        t_stat, t_pval = stats.ttest_rel(pairs1, pairs2)
        results["t_statistic"] = float(t_stat)
        results["t_pvalue"] = float(t_pval)

    # Wilcoxon signed-rank test (non-parametric)
    if test_type in ["wilcoxon", "both"]:
        try:
            w_stat, w_pval = stats.wilcoxon(pairs1, pairs2, alternative='two-sided')
            results["wilcoxon_statistic"] = float(w_stat)
            results["wilcoxon_pvalue"] = float(w_pval)
        except ValueError as e:
            results["wilcoxon_error"] = str(e)

    return results


def cohens_d(scores1: Dict[str, float], scores2: Dict[str, float]) -> float:
    """
    Compute Cohen's d effect size for paired samples.

    Interpretation:
        |d| < 0.2: negligible
        0.2 <= |d| < 0.5: small
        0.5 <= |d| < 0.8: medium
        |d| >= 0.8: large
    """
    common_tasks = set(scores1.keys()) & set(scores2.keys())
    if len(common_tasks) < 2:
        return 0.0

    pairs1 = np.array([scores1[task] for task in sorted(common_tasks)])
    pairs2 = np.array([scores2[task] for task in sorted(common_tasks)])

    # Paired Cohen's d
    diff = pairs1 - pairs2
    d = np.mean(diff) / np.std(diff, ddof=1)

    return float(d)


def bonferroni_correction(pvalues: List[float], alpha: float = ALPHA) -> List[bool]:
    """Apply Bonferroni correction for multiple comparisons."""
    n_tests = len(pvalues)
    corrected_alpha = alpha / n_tests
    return [p < corrected_alpha for p in pvalues]


def holm_correction(pvalues: List[float], alpha: float = ALPHA) -> List[bool]:
    """
    Apply Holm-Bonferroni correction (less conservative than Bonferroni).
    """
    n_tests = len(pvalues)

    # Sort p-values with indices
    sorted_indices = np.argsort(pvalues)
    sorted_pvalues = np.array(pvalues)[sorted_indices]

    # Apply Holm procedure
    rejected = np.zeros(n_tests, dtype=bool)
    for i, p in enumerate(sorted_pvalues):
        if p < alpha / (n_tests - i):
            rejected[sorted_indices[i]] = True
        else:
            break

    return list(rejected)


def compute_task_correlations(model_scores: Dict[str, Dict[str, float]]) -> np.ndarray:
    """Compute correlation matrix between task scores across models."""
    # Get all tasks
    all_tasks = set()
    for scores in model_scores.values():
        all_tasks.update(scores.keys())
    all_tasks = sorted(all_tasks)

    # Build task score matrix (tasks x models)
    task_matrix = []
    for task in all_tasks:
        task_scores = []
        for model in sorted(model_scores.keys()):
            if task in model_scores[model]:
                task_scores.append(model_scores[model][task])
            else:
                task_scores.append(np.nan)
        task_matrix.append(task_scores)

    task_matrix = np.array(task_matrix)

    # Compute correlation matrix (with NaN handling)
    n_tasks = len(all_tasks)
    corr_matrix = np.zeros((n_tasks, n_tasks))

    for i in range(n_tasks):
        for j in range(n_tasks):
            # Get valid (non-NaN) pairs
            valid = ~(np.isnan(task_matrix[i]) | np.isnan(task_matrix[j]))
            if np.sum(valid) > 1:
                corr_matrix[i, j] = np.corrcoef(
                    task_matrix[i][valid],
                    task_matrix[j][valid]
                )[0, 1]
            else:
                corr_matrix[i, j] = np.nan

    return corr_matrix, all_tasks


def main():
    """Run all statistical analyses."""
    print("=" * 80)
    print("SHELF Benchmark: Statistical Significance Analysis")
    print("=" * 80)
    print()

    # Load results
    print("Loading results...")
    model_scores = load_all_results()
    print(f"Loaded {len(model_scores)} models")

    # Compute SHELF scores with confidence intervals
    print("\n" + "=" * 80)
    print("1. Bootstrap Confidence Intervals (95%)")
    print("=" * 80)

    shelf_scores = {}
    for model, task_scores in sorted(model_scores.items()):
        mean, lower, upper = bootstrap_confidence_interval(task_scores)
        shelf_scores[model] = mean
        width = upper - lower
        print(f"{model:30s}: {mean:.4f} [{lower:.4f}, {upper:.4f}] (width: {width:.4f})")

    # Key comparisons
    print("\n" + "=" * 80)
    print("2. Key Model Comparisons (Paired Tests)")
    print("=" * 80)

    comparisons = [
        # Main claim: sparse > dense
        ("tf", "bge_large", "Sparse (TF) vs. Dense (BGE-large)"),
        ("tfidf", "bge_large", "Sparse (TF-IDF) vs. Dense (BGE-large)"),

        # Top neural models
        ("bge_large", "gte_base", "BGE-large vs. GTE-base"),
        ("bge_large", "e5_large", "BGE-large vs. E5-large"),
        ("bge_large", "mpnet", "BGE-large vs. MPNet"),

        # Model size effects
        ("bge_large", "bge_base", "BGE-large vs. BGE-base"),
        ("bge_base", "bge_small", "BGE-base vs. BGE-small"),

        # Sparse methods
        ("tfidf", "bm25", "TF-IDF vs. BM25"),
        ("tf", "tfidf", "TF vs. TF-IDF"),
    ]

    comparison_results = []
    pvalues_ttest = []
    pvalues_wilcoxon = []

    for model1, model2, description in comparisons:
        if model1 not in model_scores or model2 not in model_scores:
            print(f"\n{description}: SKIPPED (model not found)")
            continue

        print(f"\n{description}:")
        print(f"  {model1}: {shelf_scores.get(model1, 0):.4f}")
        print(f"  {model2}: {shelf_scores.get(model2, 0):.4f}")

        result = paired_test(model_scores[model1], model_scores[model2])
        d = cohens_d(model_scores[model1], model_scores[model2])

        print(f"  Difference: {result.get('mean_diff', 0):.4f} ± {result.get('std_diff', 0):.4f}")
        print(f"  Cohen's d: {d:.4f}", end="")

        # Interpret effect size
        if abs(d) < 0.2:
            interpretation = "(negligible)"
        elif abs(d) < 0.5:
            interpretation = "(small)"
        elif abs(d) < 0.8:
            interpretation = "(medium)"
        else:
            interpretation = "(large)"
        print(f" {interpretation}")

        if "t_pvalue" in result:
            print(f"  Paired t-test: t={result['t_statistic']:.3f}, p={result['t_pvalue']:.4f}", end="")
            if result['t_pvalue'] < ALPHA:
                print(" ***")
            elif result['t_pvalue'] < 0.1:
                print(" *")
            else:
                print()
            pvalues_ttest.append(result['t_pvalue'])

        if "wilcoxon_pvalue" in result:
            print(f"  Wilcoxon test: W={result['wilcoxon_statistic']:.1f}, p={result['wilcoxon_pvalue']:.4f}", end="")
            if result['wilcoxon_pvalue'] < ALPHA:
                print(" ***")
            elif result['wilcoxon_pvalue'] < 0.1:
                print(" *")
            else:
                print()
            pvalues_wilcoxon.append(result['wilcoxon_pvalue'])

        comparison_results.append({
            "comparison": description,
            "model1": model1,
            "model2": model2,
            "result": result,
            "cohens_d": d
        })

    # Multiple comparison corrections
    print("\n" + "=" * 80)
    print("3. Multiple Comparison Corrections")
    print("=" * 80)

    print(f"\nNumber of comparisons: {len(pvalues_ttest)}")
    print(f"Uncorrected alpha: {ALPHA}")
    print(f"Bonferroni-corrected alpha: {ALPHA / len(pvalues_ttest):.6f}")

    bonferroni_rejected = bonferroni_correction(pvalues_ttest)
    holm_rejected = holm_correction(pvalues_ttest)

    print("\nSignificant after correction (t-test):")
    print(f"{'Comparison':40s} {'p-value':>10s} {'Bonf.':>8s} {'Holm':>8s}")
    print("-" * 70)
    for i, (model1, model2, desc) in enumerate(comparisons):
        if i < len(pvalues_ttest):
            p = pvalues_ttest[i]
            bonf = "Yes" if bonferroni_rejected[i] else "No"
            holm = "Yes" if holm_rejected[i] else "No"
            print(f"{desc:40s} {p:10.6f} {bonf:>8s} {holm:>8s}")

    # Task variance analysis
    print("\n" + "=" * 80)
    print("4. Task Variance Analysis")
    print("=" * 80)

    # Compute variance of each task across models
    all_tasks = set()
    for scores in model_scores.values():
        all_tasks.update(scores.keys())

    task_variances = {}
    for task in sorted(all_tasks):
        task_scores_list = []
        for model, scores in model_scores.items():
            if task in scores:
                task_scores_list.append(scores[task])

        if len(task_scores_list) > 1:
            task_variances[task] = {
                "mean": np.mean(task_scores_list),
                "std": np.std(task_scores_list, ddof=1),
                "min": np.min(task_scores_list),
                "max": np.max(task_scores_list),
                "n_models": len(task_scores_list)
            }

    # Sort by variance (descending)
    sorted_tasks = sorted(task_variances.items(), key=lambda x: x[1]["std"], reverse=True)

    print(f"\n{'Task':40s} {'Mean':>8s} {'Std':>8s} {'Range':>12s} {'CV':>8s}")
    print("-" * 80)
    for task, stats in sorted_tasks[:10]:  # Top 10 highest variance
        cv = stats["std"] / stats["mean"] if stats["mean"] > 0 else 0
        range_str = f"{stats['min']:.3f}-{stats['max']:.3f}"
        print(f"{task:40s} {stats['mean']:8.4f} {stats['std']:8.4f} {range_str:>12s} {cv:8.4f}")

    # Correlation analysis
    print("\n" + "=" * 80)
    print("5. Task Correlation Analysis")
    print("=" * 80)

    corr_matrix, task_names = compute_task_correlations(model_scores)

    # Find highly correlated task pairs
    high_corr_pairs = []
    n_tasks = len(task_names)
    for i in range(n_tasks):
        for j in range(i + 1, n_tasks):
            corr = corr_matrix[i, j]
            if not np.isnan(corr) and abs(corr) > 0.7:
                high_corr_pairs.append((task_names[i], task_names[j], corr))

    high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    print("\nHighly correlated task pairs (|r| > 0.7):")
    if high_corr_pairs:
        print(f"{'Task 1':30s} {'Task 2':30s} {'Correlation':>12s}")
        print("-" * 75)
        for task1, task2, corr in high_corr_pairs[:10]:
            print(f"{task1:30s} {task2:30s} {corr:12.4f}")
    else:
        print("No highly correlated task pairs found (tasks are largely independent)")

    # Summary statistics
    print("\n" + "=" * 80)
    print("6. Summary Statistics")
    print("=" * 80)

    # Best sparse model
    best_sparse = max(
        [(m, s) for m, s in shelf_scores.items() if m in SPARSE_MODELS],
        key=lambda x: x[1],
        default=(None, 0)
    )

    # Best dense model
    best_dense = max(
        [(m, s) for m, s in shelf_scores.items() if m in DENSE_MODELS],
        key=lambda x: x[1],
        default=(None, 0)
    )

    print(f"\nBest sparse model: {best_sparse[0]} ({best_sparse[1]:.4f})")
    print(f"Best dense model: {best_dense[0]} ({best_dense[1]:.4f})")

    if best_sparse[0] and best_dense[0]:
        if best_sparse[0] in model_scores and best_dense[0] in model_scores:
            sparse_vs_dense = paired_test(
                model_scores[best_sparse[0]],
                model_scores[best_dense[0]]
            )
            d_sparse_dense = cohens_d(
                model_scores[best_sparse[0]],
                model_scores[best_dense[0]]
            )

            print(f"\nSparse vs. Dense (best models):")
            print(f"  Difference: {sparse_vs_dense.get('mean_diff', 0):.4f}")
            print(f"  Cohen's d: {d_sparse_dense:.4f}")
            print(f"  p-value (t-test): {sparse_vs_dense.get('t_pvalue', 1):.6f}")
            print(f"  p-value (Wilcoxon): {sparse_vs_dense.get('wilcoxon_pvalue', 1):.6f}")

            if sparse_vs_dense.get('t_pvalue', 1) < ALPHA:
                print(f"  *** STATISTICALLY SIGNIFICANT (uncorrected)")

            # Check with Bonferroni correction (assume 9 main comparisons)
            if sparse_vs_dense.get('t_pvalue', 1) < ALPHA / 9:
                print(f"  *** STATISTICALLY SIGNIFICANT (Bonferroni-corrected)")

    # Save detailed results
    output_file = OUTPUT_DIR / "statistical_tests_results.json"
    with open(output_file, "w") as f:
        json.dump({
            "metadata": {
                "n_bootstrap": N_BOOTSTRAP,
                "alpha": ALPHA,
                "random_seed": RANDOM_SEED
            },
            "shelf_scores": shelf_scores,
            "comparisons": [
                {
                    "description": cr["comparison"],
                    "model1": cr["model1"],
                    "model2": cr["model2"],
                    "cohens_d": cr["cohens_d"],
                    **cr["result"]
                }
                for cr in comparison_results
            ],
            "task_variances": task_variances
        }, f, indent=2)

    print(f"\n\nDetailed results saved to: {output_file}")
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()

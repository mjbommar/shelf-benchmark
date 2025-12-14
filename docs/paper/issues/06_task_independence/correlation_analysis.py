#!/usr/bin/env python3
"""
Task Independence Analysis for SHELF Benchmark

Investigates whether SHELF tasks are truly independent or share structure
that inflates aggregate scores.

Key questions:
1. How correlated are task scores across models?
2. Do model rankings differ by task?
3. What explains observed correlations?
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 12)


def load_all_results(results_dir: Path) -> Dict[str, Dict[str, float]]:
    """
    Load all result JSON files and extract primary scores.

    Returns:
        Dict mapping model_key to dict of task->score
    """
    results = {}

    for json_file in results_dir.glob("*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)

            model_key = data.get("model_key")
            task = data.get("task")
            primary_score = data.get("primary_score")

            if model_key and task and primary_score is not None:
                if model_key not in results:
                    results[model_key] = {}
                results[model_key][task] = primary_score
        except Exception as e:
            print(f"Error loading {json_file}: {e}")

    return results


def build_score_matrix(results: Dict[str, Dict[str, float]]) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Build model x task score matrix.

    Returns:
        DataFrame with models as rows, tasks as columns
    """
    # Get all unique tasks
    all_tasks = sorted(set(
        task for model_tasks in results.values() for task in model_tasks.keys()
    ))

    # Get all models
    all_models = sorted(results.keys())

    # Build matrix
    matrix_data = []
    for model in all_models:
        row = [results[model].get(task, np.nan) for task in all_tasks]
        matrix_data.append(row)

    df = pd.DataFrame(matrix_data, index=all_models, columns=all_tasks)

    return df, all_models, all_tasks


def compute_task_correlations(score_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Pearson correlation between all task pairs.

    Args:
        score_matrix: Model x Task score matrix (rows=models, cols=tasks)

    Returns:
        Task x Task correlation matrix
    """
    # Correlation of columns gives task-to-task correlation
    # (how similar are task score profiles across models)
    corr_matrix = score_matrix.corr(method='pearson')

    return corr_matrix


def compute_rank_correlations(score_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Spearman rank correlation between all task pairs.

    This shows whether model rankings are consistent across tasks.
    """
    rank_corr = score_matrix.corr(method='spearman')

    return rank_corr


def group_tasks_by_type(tasks: List[str]) -> Dict[str, List[str]]:
    """Group tasks by their evaluation type."""
    task_groups = {
        'classification': [],
        'retrieval': [],
        'clustering': [],
        'pair_classification': []
    }

    for task in tasks:
        if 'classification' in task and 'pairs' not in task:
            task_groups['classification'].append(task)
        elif 'retrieval' in task:
            task_groups['retrieval'].append(task)
        elif 'clustering' in task:
            task_groups['clustering'].append(task)
        elif 'pairs' in task:
            task_groups['pair_classification'].append(task)

    return task_groups


def plot_correlation_heatmap(corr_matrix: pd.DataFrame, title: str, output_file: Path):
    """Plot and save correlation heatmap."""
    plt.figure(figsize=(16, 14))

    # Create heatmap
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Correlation Coefficient'}
    )

    plt.title(title, fontsize=16, pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved heatmap to {output_file}")


def analyze_cross_type_correlations(corr_matrix: pd.DataFrame, task_groups: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Analyze correlations between different task types.

    Returns summary statistics for within-type vs cross-type correlations.
    """
    stats_list = []

    type_names = list(task_groups.keys())

    for i, type1 in enumerate(type_names):
        for j, type2 in enumerate(type_names):
            if j < i:  # Skip lower triangle to avoid duplicates
                continue

            tasks1 = task_groups[type1]
            tasks2 = task_groups[type2]

            if not tasks1 or not tasks2:
                continue

            # Extract correlations between these two types
            corrs = []
            for t1 in tasks1:
                for t2 in tasks2:
                    if t1 not in corr_matrix.index or t2 not in corr_matrix.columns:
                        continue

                    # Skip diagonal (self-correlation = 1.0)
                    if t1 == t2:
                        continue

                    # For same type, skip duplicates (t1 vs t2 same as t2 vs t1)
                    if type1 == type2 and t1 > t2:
                        continue

                    corr = corr_matrix.loc[t1, t2]
                    if not np.isnan(corr):
                        corrs.append(corr)

            if corrs:
                stats_list.append({
                    'Type 1': type1,
                    'Type 2': type2,
                    'Relationship': 'Within' if type1 == type2 else 'Cross',
                    'Mean r': np.mean(corrs),
                    'Median r': np.median(corrs),
                    'Std r': np.std(corrs),
                    'Min r': np.min(corrs),
                    'Max r': np.max(corrs),
                    'N pairs': len(corrs)
                })

    return pd.DataFrame(stats_list)


def find_model_champions(score_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Identify which model performs best on each task.

    Returns DataFrame showing task champions and score differences.
    """
    champions = []

    for task in score_matrix.columns:
        task_scores = score_matrix[task].dropna()
        if len(task_scores) == 0:
            continue

        # Sort models by score
        ranked = task_scores.sort_values(ascending=False)

        champions.append({
            'Task': task,
            'Champion': ranked.index[0],
            'Champion Score': ranked.iloc[0],
            'Runner-up': ranked.index[1] if len(ranked) > 1 else None,
            'Runner-up Score': ranked.iloc[1] if len(ranked) > 1 else None,
            'Gap': ranked.iloc[0] - ranked.iloc[1] if len(ranked) > 1 else None,
            'Last Place': ranked.index[-1],
            'Last Score': ranked.iloc[-1],
            'Range': ranked.iloc[0] - ranked.iloc[-1]
        })

    return pd.DataFrame(champions)


def compute_ranking_consistency(score_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Compute how consistent model rankings are across tasks.

    Uses Kendall's W (coefficient of concordance).
    """
    # Convert scores to ranks for each task
    rank_matrix = score_matrix.rank(ascending=False, method='average')

    # Compute mean rank for each model
    mean_ranks = rank_matrix.mean(axis=1)

    # Compute variance in ranks for each model
    rank_variance = rank_matrix.var(axis=1)

    # Overall consistency
    results = pd.DataFrame({
        'Model': rank_matrix.index,
        'Mean Rank': mean_ranks,
        'Rank Std': np.sqrt(rank_variance),
        'Best Rank': rank_matrix.min(axis=1),
        'Worst Rank': rank_matrix.max(axis=1),
        'Rank Range': rank_matrix.max(axis=1) - rank_matrix.min(axis=1)
    })

    return results.sort_values('Mean Rank')


def main():
    """Run complete task independence analysis."""
    # Paths
    results_dir = Path("/home/mjbommar/src/shelf-benchmark/results/v0.3.0/baselines")
    output_dir = Path("/home/mjbommar/src/shelf-benchmark/docs/paper/issues/06_task_independence")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("SHELF Task Independence Analysis")
    print("="*80)

    # Load all results
    print("\n1. Loading all baseline results...")
    results = load_all_results(results_dir)
    print(f"   Loaded results for {len(results)} models")

    # Build score matrix
    print("\n2. Building model x task score matrix...")
    score_matrix, models, tasks = build_score_matrix(results)
    print(f"   Matrix shape: {len(models)} models x {len(tasks)} tasks")
    print(f"   Coverage: {(~score_matrix.isna()).sum().sum()} / {len(models) * len(tasks)} cells")

    # Save raw matrix
    score_matrix.to_csv(output_dir / "model_task_scores.csv")
    print(f"   Saved score matrix to model_task_scores.csv")

    # Compute task correlations
    print("\n3. Computing task correlations...")
    pearson_corr = compute_task_correlations(score_matrix)
    spearman_corr = compute_rank_correlations(score_matrix)

    # Save correlation matrices
    pearson_corr.to_csv(output_dir / "task_correlations_pearson.csv")
    spearman_corr.to_csv(output_dir / "task_correlations_spearman.csv")

    # Plot heatmaps
    print("\n4. Generating correlation heatmaps...")
    plot_correlation_heatmap(
        pearson_corr,
        "SHELF Task Score Correlations (Pearson r)",
        output_dir / "task_correlations_pearson.png"
    )
    plot_correlation_heatmap(
        spearman_corr,
        "SHELF Task Rank Correlations (Spearman ρ)",
        output_dir / "task_correlations_spearman.png"
    )

    # Group tasks by type
    print("\n5. Analyzing cross-type correlations...")
    task_groups = group_tasks_by_type(tasks)
    for task_type, task_list in task_groups.items():
        print(f"   {task_type}: {len(task_list)} tasks")

    cross_type_stats = analyze_cross_type_correlations(pearson_corr, task_groups)
    cross_type_stats.to_csv(output_dir / "cross_type_correlations.csv", index=False)

    print("\n   Cross-type correlation summary:")
    print(cross_type_stats.to_string(index=False))

    # Find task champions
    print("\n6. Identifying task champions...")
    champions = find_model_champions(score_matrix)
    champions.to_csv(output_dir / "task_champions.csv", index=False)

    # Count unique champions
    unique_champions = champions['Champion'].value_counts()
    print(f"\n   Champion distribution:")
    print(unique_champions.head(10).to_string())

    # Ranking consistency
    print("\n7. Computing ranking consistency...")
    ranking_consistency = compute_ranking_consistency(score_matrix)
    ranking_consistency.to_csv(output_dir / "ranking_consistency.csv", index=False)

    print("\n   Top 10 most consistent models:")
    print(ranking_consistency.head(10).to_string(index=False))

    # Summary statistics
    print("\n8. Summary Statistics:")
    print("="*80)

    # Overall correlation statistics
    # Get upper triangle (excluding diagonal)
    mask = np.triu(np.ones_like(pearson_corr, dtype=bool), k=1)
    upper_tri_corrs = pearson_corr.where(mask).stack().values

    print(f"\n   Task-Task Correlations (Pearson r):")
    print(f"   Mean: {np.mean(upper_tri_corrs):.3f}")
    print(f"   Median: {np.median(upper_tri_corrs):.3f}")
    print(f"   Std: {np.std(upper_tri_corrs):.3f}")
    print(f"   Min: {np.min(upper_tri_corrs):.3f}")
    print(f"   Max: {np.max(upper_tri_corrs):.3f}")
    print(f"   Q1: {np.percentile(upper_tri_corrs, 25):.3f}")
    print(f"   Q3: {np.percentile(upper_tri_corrs, 75):.3f}")

    # Highly correlated pairs
    print(f"\n   Highly correlated task pairs (r > 0.8):")
    high_corr_pairs = []
    for i in range(len(pearson_corr)):
        for j in range(i+1, len(pearson_corr)):
            r = pearson_corr.iloc[i, j]
            if r > 0.8:
                high_corr_pairs.append((
                    pearson_corr.index[i],
                    pearson_corr.columns[j],
                    r
                ))

    high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
    for task1, task2, r in high_corr_pairs[:10]:
        print(f"   {task1} <-> {task2}: r={r:.3f}")

    print("\n" + "="*80)
    print("Analysis complete! Results saved to:")
    print(f"   {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()

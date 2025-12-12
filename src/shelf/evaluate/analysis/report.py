"""Report generation for benchmark comparisons.

Provides human-readable reports and machine-readable summaries
for model and commit comparisons.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from shelf.evaluate.analysis.comparison import (
    ComparisonResult,
    MultipleComparisonResult,
)


def generate_comparison_report(
    comparison: ComparisonResult | MultipleComparisonResult,
    output_path: str | Path | None = None,
    format: str = "text",
) -> str:
    """Generate a comparison report.

    Args:
        comparison: Comparison result to report on
        output_path: Optional path to save report
        format: Output format ("text", "json", "markdown")

    Returns:
        Report string
    """
    if format == "text":
        report = _generate_text_report(comparison)
    elif format == "markdown":
        report = _generate_markdown_report(comparison)
    elif format == "json":
        report = _generate_json_report(comparison)
    else:
        raise ValueError(f"Unknown format: {format}")

    if output_path:
        Path(output_path).write_text(report)

    return report


def _generate_text_report(
    comparison: ComparisonResult | MultipleComparisonResult,
) -> str:
    """Generate plain text report."""
    lines = [
        "=" * 70,
        "SHELF Benchmark Comparison Report",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "=" * 70,
        "",
    ]

    if isinstance(comparison, ComparisonResult):
        lines.extend(_pairwise_text_report(comparison))
    else:
        lines.extend(_multiple_text_report(comparison))

    return "\n".join(lines)


def _pairwise_text_report(comp: ComparisonResult) -> list[str]:
    """Generate text report for pairwise comparison."""
    lines = [
        f"Metric: {comp.metric}",
        f"Samples: {comp.n_samples} ({'paired' if comp.is_paired else 'unpaired'})",
        "",
        "Results:",
        f"  {comp.name_a}: {comp.mean_a:.4f}",
        f"  {comp.name_b}: {comp.mean_b:.4f}",
        "",
        f"Difference: {comp.difference:+.4f} ({comp.relative_diff_pct:+.1f}%)",
        "",
        "Statistical Test:",
        f"  {comp.significance}",
    ]

    if comp.bootstrap:
        lines.extend(
            [
                "",
                "Bootstrap Analysis:",
                f"  {comp.bootstrap}",
            ]
        )

    lines.extend(
        [
            "",
            "-" * 40,
            "Conclusion: ",
        ]
    )

    if comp.winner:
        lines.append(
            f"  {comp.winner} is significantly better (p < {comp.significance.alpha})"
        )
    else:
        lines.append(
            f"  No statistically significant difference (p >= {comp.significance.alpha})"
        )

    if comp.significance.effect_size is not None:
        lines.append(f"  Effect size: {comp.significance.effect_size_interpretation}")

    return lines


def _multiple_text_report(comp: MultipleComparisonResult) -> list[str]:
    """Generate text report for multiple comparison."""
    lines = [
        f"Metric: {comp.metric}",
        f"Models: {len(comp.models)}",
        f"Tasks/Datasets: {comp.friedman_nemenyi.n_blocks}",
        "",
        "Ranking (by mean score):",
        "-" * 40,
    ]

    for i, model in enumerate(comp.ranking, 1):
        mean_rank = comp.friedman_nemenyi.mean_ranks[model]
        lines.append(
            f"  {i}. {model:20s} {comp.mean_scores[model]:.4f}  (rank: {mean_rank:.2f})"
        )

    lines.extend(
        [
            "",
            "Friedman Test:",
            f"  χ² = {comp.friedman_nemenyi.friedman_statistic:.2f}",
            f"  p-value = {comp.friedman_nemenyi.friedman_p_value:.4f}",
            f"  Significant: {'Yes' if comp.friedman_nemenyi.significant_overall else 'No'}",
            "",
            "Nemenyi Post-hoc Test:",
            f"  Critical difference: {comp.friedman_nemenyi.critical_difference:.3f}",
            f"  Significant pairs: {len(comp.friedman_nemenyi.significant_pairs)}",
        ]
    )

    if comp.friedman_nemenyi.significant_pairs:
        lines.append("")
        for a, b in comp.friedman_nemenyi.significant_pairs:
            p = comp.friedman_nemenyi.pairwise_p_values[(a, b)]
            diff = comp.mean_scores[a] - comp.mean_scores[b]
            lines.append(f"    {a} vs {b}: p={p:.4f}, Δ={diff:+.4f}")

    if comp.equivalence_groups:
        lines.extend(
            [
                "",
                "Equivalence Groups (not significantly different):",
            ]
        )
        for group in comp.equivalence_groups:
            lines.append(f"  • {', '.join(group)}")

    return lines


def _generate_markdown_report(
    comparison: ComparisonResult | MultipleComparisonResult,
) -> str:
    """Generate Markdown report."""
    lines = [
        "# SHELF Benchmark Comparison Report",
        "",
        f"*Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*",
        "",
    ]

    if isinstance(comparison, MultipleComparisonResult):
        lines.extend(_multiple_markdown_report(comparison))
    else:
        lines.extend(_pairwise_markdown_report(comparison))

    return "\n".join(lines)


def _pairwise_markdown_report(comp: ComparisonResult) -> list[str]:
    """Generate Markdown for pairwise comparison."""
    sig_emoji = "✅" if comp.significance.significant else "❌"

    lines = [
        f"## {comp.name_a} vs {comp.name_b}",
        "",
        f"**Metric:** {comp.metric}  ",
        f"**Samples:** {comp.n_samples} ({'paired' if comp.is_paired else 'unpaired'})",
        "",
        "### Results",
        "",
        "| Model | Score |",
        "|-------|-------|",
        f"| {comp.name_a} | {comp.mean_a:.4f} |",
        f"| {comp.name_b} | {comp.mean_b:.4f} |",
        "",
        f"**Difference:** {comp.difference:+.4f} ({comp.relative_diff_pct:+.1f}%)",
        "",
        "### Statistical Analysis",
        "",
        f"- **Test:** {comp.significance.test_name}",
        f"- **p-value:** {comp.significance.p_value:.4f}",
        f"- **Significant:** {sig_emoji} {'Yes' if comp.significance.significant else 'No'} (α={comp.significance.alpha})",
    ]

    if comp.significance.effect_size is not None:
        lines.append(
            f"- **Effect size (Cohen's d):** {comp.significance.effect_size:.3f} ({comp.significance.effect_size_interpretation})"
        )

    if comp.bootstrap:
        ci = f"[{comp.bootstrap.ci_lower:+.4f}, {comp.bootstrap.ci_upper:+.4f}]"
        lines.extend(
            [
                "",
                "### Bootstrap Confidence Interval",
                "",
                f"- **95% CI for difference:** {ci}",
                f"- **Standard error:** {comp.bootstrap.se:.4f}",
            ]
        )

    lines.extend(
        [
            "",
            "### Conclusion",
            "",
        ]
    )

    if comp.winner:
        lines.append(f"**{comp.winner}** performs significantly better.")
    else:
        lines.append("No statistically significant difference detected.")

    return lines


def _multiple_markdown_report(comp: MultipleComparisonResult) -> list[str]:
    """Generate Markdown for multiple comparison."""
    lines = [
        f"## Multiple Model Comparison: {comp.metric}",
        "",
        f"- **Models:** {len(comp.models)}",
        f"- **Tasks/Datasets:** {comp.friedman_nemenyi.n_blocks}",
        "",
        "### Ranking",
        "",
        "| Rank | Model | Mean Score | Mean Rank |",
        "|------|-------|------------|-----------|",
    ]

    for i, model in enumerate(comp.ranking, 1):
        mean_rank = comp.friedman_nemenyi.mean_ranks[model]
        lines.append(
            f"| {i} | {model} | {comp.mean_scores[model]:.4f} | {mean_rank:.2f} |"
        )

    sig_emoji = "✅" if comp.friedman_nemenyi.significant_overall else "❌"
    lines.extend(
        [
            "",
            "### Friedman Test",
            "",
            f"- **χ²:** {comp.friedman_nemenyi.friedman_statistic:.2f}",
            f"- **p-value:** {comp.friedman_nemenyi.friedman_p_value:.4f}",
            f"- **Significant:** {sig_emoji}",
            "",
            "### Nemenyi Post-hoc Test",
            "",
            f"- **Critical difference:** {comp.friedman_nemenyi.critical_difference:.3f}",
            f"- **Significant pairs:** {len(comp.friedman_nemenyi.significant_pairs)}",
        ]
    )

    if comp.friedman_nemenyi.significant_pairs:
        lines.extend(
            [
                "",
                "| Pair | p-value | Difference |",
                "|------|---------|------------|",
            ]
        )
        for a, b in comp.friedman_nemenyi.significant_pairs:
            p = comp.friedman_nemenyi.pairwise_p_values[(a, b)]
            diff = comp.mean_scores[a] - comp.mean_scores[b]
            lines.append(f"| {a} vs {b} | {p:.4f} | {diff:+.4f} |")

    if comp.equivalence_groups:
        lines.extend(
            [
                "",
                "### Equivalence Groups",
                "",
                "Models within each group are **not** significantly different:",
                "",
            ]
        )
        for group in comp.equivalence_groups:
            lines.append(f"- {', '.join(group)}")

    return lines


def _generate_json_report(
    comparison: ComparisonResult | MultipleComparisonResult,
) -> str:
    """Generate JSON report."""

    def make_serializable(obj: Any) -> Any:
        """Convert objects to JSON-serializable form."""
        if hasattr(obj, "__dict__"):
            return {
                k: make_serializable(v)
                for k, v in obj.__dict__.items()
                if not k.startswith("_")
            }
        elif isinstance(obj, dict):
            return {str(k): make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif hasattr(obj, "tolist"):  # numpy arrays
            return obj.tolist()
        else:
            return str(obj)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "comparison_type": type(comparison).__name__,
        "data": make_serializable(comparison),
    }

    return json.dumps(report, indent=2)


def print_comparison_summary(
    comparison: ComparisonResult | MultipleComparisonResult,
) -> None:
    """Print a concise comparison summary to stdout."""
    print(comparison.summary())


def print_significance_table(
    comparisons: list[ComparisonResult],
    show_effect_size: bool = True,
) -> None:
    """Print a summary table of multiple pairwise comparisons.

    Args:
        comparisons: List of pairwise comparison results
        show_effect_size: Whether to show effect sizes
    """
    # Header
    header = f"{'Comparison':<30} {'Diff':>8} {'p-value':>10} {'Sig':>5}"
    if show_effect_size:
        header += f" {'Effect':>8}"
    print(header)
    print("-" * len(header))

    # Rows
    for comp in comparisons:
        name = f"{comp.name_a} vs {comp.name_b}"[:30]
        sig = (
            "***"
            if comp.significance.p_value < 0.001
            else (
                "**"
                if comp.significance.p_value < 0.01
                else ("*" if comp.significance.p_value < 0.05 else "")
            )
        )
        row = f"{name:<30} {comp.difference:>+8.4f} {comp.significance.p_value:>10.4f} {sig:>5}"
        if show_effect_size and comp.significance.effect_size is not None:
            row += f" {comp.significance.effect_size:>+8.3f}"
        print(row)

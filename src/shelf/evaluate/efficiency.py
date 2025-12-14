"""Efficiency-adjusted metrics for fair model comparison.

This module provides functions for computing efficiency-normalized scores
that account for model size when comparing benchmark performance.

Metrics implemented:
- SHELF_eff: Score normalized by log(params)
- SHELF_compute: Score normalized by compute cost with sub-linear scaling
- Pareto efficiency: Binary flag for Pareto-optimal models
- Size-stratified rank: Rank within size category

See docs/efficiency_metrics.md for detailed methodology and references.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


# Baseline model for relative efficiency calculations
BASELINE_PARAMS = 22_713_216  # MiniLM-L6 (smallest model)

# Size category thresholds
SIZE_THRESHOLDS = {
    "small": 50_000_000,  # < 50M params
    "base": 300_000_000,  # < 300M params
    "large": float("inf"),  # >= 300M params
}


@dataclass
class EfficiencyMetrics:
    """Container for efficiency-adjusted metrics.

    Attributes:
        num_params: Number of model parameters (from config/metadata)
        embedding_dim: Output embedding dimension
        size_category: Size category (e.g., '<10M', '10M-50M', '100M-300M')
        flops_per_token: Estimated FLOPs per token (2 * num_params)
        relative_compute: Compute cost relative to baseline (MiniLM)
        shelf_eff: SHELF score * 1000 / log10(params)
        shelf_compute: SHELF score / (params/baseline)^0.1
        pareto_optimal: Whether model is on Pareto frontier
        size_rank: Rank within size category (1 = best)
        num_params_torch: Actual parameter count via torch.numel()
        hidden_size: Model hidden dimension from config
        context_window: Max position embeddings from config
        throughput_bytes_sec: Bytes processed per second during embedding
    """

    num_params: int
    embedding_dim: int
    size_category: str
    flops_per_token: int
    relative_compute: float
    shelf_eff: float | None = None
    shelf_compute: float | None = None
    pareto_optimal: bool | None = None
    size_rank: int | None = None
    # New fields for enhanced model metrics
    num_params_torch: int | None = None
    hidden_size: int | None = None
    context_window: int | None = None
    throughput_bytes_sec: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "num_params": self.num_params,
            "embedding_dim": self.embedding_dim,
            "size_category": self.size_category,
            "flops_per_token": self.flops_per_token,
            "relative_compute": round(self.relative_compute, 4),
            "shelf_eff": round(self.shelf_eff, 4)
            if self.shelf_eff is not None
            else None,
            "shelf_compute": round(self.shelf_compute, 4)
            if self.shelf_compute is not None
            else None,
            "pareto_optimal": self.pareto_optimal,
            "size_rank": self.size_rank,
            "num_params_torch": self.num_params_torch,
            "hidden_size": self.hidden_size,
            "context_window": self.context_window,
            "throughput_bytes_sec": round(self.throughput_bytes_sec, 2)
            if self.throughput_bytes_sec is not None
            else None,
        }


def get_size_category(num_params: int) -> str:
    """Determine size category from parameter count.

    Categories:
    - small: < 50M params
    - base: 50M - 300M params
    - large: >= 300M params

    Args:
        num_params: Number of model parameters

    Returns:
        Size category string
    """
    if num_params < SIZE_THRESHOLDS["small"]:
        return "small"
    elif num_params < SIZE_THRESHOLDS["base"]:
        return "base"
    else:
        return "large"


def compute_shelf_eff(shelf_score: float, num_params: int) -> float:
    """Compute SHELF efficiency score.

    Formula: SHELF_eff = SHELF_score * 1000 / log10(params)

    Uses logarithmic scaling because performance doesn't scale linearly
    with parameters. A 10x increase in params adds 1.0 to the denominator.

    Args:
        shelf_score: Raw SHELF score (0-1 scale)
        num_params: Number of model parameters

    Returns:
        SHELF efficiency score (higher = more efficient)

    Example:
        >>> compute_shelf_eff(0.465, 22_713_216)  # MiniLM
        63.35
        >>> compute_shelf_eff(0.513, 335_141_888)  # BGE-large
        60.18
    """
    if num_params <= 0:
        raise ValueError("num_params must be positive")

    log_params = math.log10(num_params)
    return shelf_score * 1000 / log_params


def compute_shelf_compute(
    shelf_score: float,
    num_params: int,
    baseline_params: int = BASELINE_PARAMS,
    alpha: float = 0.1,
) -> float:
    """Compute compute-adjusted SHELF score.

    Formula: SHELF_compute = SHELF_score / (params / baseline)^alpha

    Uses sub-linear scaling (alpha < 1) because performance scales
    sub-linearly with parameters. Default alpha=0.1 based on scaling laws.

    Args:
        shelf_score: Raw SHELF score (0-1 scale)
        num_params: Number of model parameters
        baseline_params: Baseline model params (default: MiniLM)
        alpha: Scaling exponent (default: 0.1)

    Returns:
        Compute-adjusted SHELF score

    Example:
        >>> compute_shelf_compute(0.465, 22_713_216)  # MiniLM
        0.465  # No penalty for baseline
        >>> compute_shelf_compute(0.513, 335_141_888)  # BGE-large
        0.388  # Penalized for 15x more params
    """
    if num_params <= 0 or baseline_params <= 0:
        raise ValueError("params must be positive")

    relative_size = num_params / baseline_params
    return shelf_score / (relative_size**alpha)


def compute_relative_efficiency(
    shelf_score: float,
    num_params: int,
    baseline_score: float,
    baseline_params: int = BASELINE_PARAMS,
) -> float:
    """Compute relative efficiency compared to baseline.

    Formula: (score / baseline_score) / (params / baseline_params)

    Values > 1.0 mean better efficiency than baseline (more performance
    per parameter).

    Args:
        shelf_score: Model's SHELF score
        num_params: Model's parameter count
        baseline_score: Baseline model's SHELF score
        baseline_params: Baseline model's parameter count

    Returns:
        Relative efficiency ratio
    """
    if baseline_score <= 0 or baseline_params <= 0:
        raise ValueError("baseline values must be positive")

    score_ratio = shelf_score / baseline_score
    param_ratio = num_params / baseline_params

    return score_ratio / param_ratio


def find_pareto_optimal(
    models: dict[str, tuple[float, int]],
) -> set[str]:
    """Find Pareto-optimal models on score vs params frontier.

    A model is Pareto-optimal if no other model has both:
    - Higher SHELF score AND
    - Fewer parameters

    Args:
        models: Dict mapping model_key to (shelf_score, num_params)

    Returns:
        Set of model keys that are Pareto-optimal

    Example:
        >>> models = {
        ...     "minilm": (0.465, 22_713_216),
        ...     "bge_small": (0.479, 33_360_512),
        ...     "bge_large": (0.513, 335_141_888),
        ... }
        >>> find_pareto_optimal(models)
        {'minilm', 'bge_small', 'bge_large'}
    """
    pareto_set = set()

    for model_key, (score, params) in models.items():
        is_dominated = False

        for other_key, (other_score, other_params) in models.items():
            if model_key == other_key:
                continue

            # Check if other model dominates this one
            # (higher or equal score AND fewer or equal params, with at least one strict)
            if (
                other_score >= score
                and other_params <= params
                and (other_score > score or other_params < params)
            ):
                is_dominated = True
                break

        if not is_dominated:
            pareto_set.add(model_key)

    return pareto_set


def compute_size_ranks(
    models: dict[str, tuple[float, int, str]],
) -> dict[str, int]:
    """Compute rank within size category for each model.

    Args:
        models: Dict mapping model_key to (shelf_score, num_params, size_category)

    Returns:
        Dict mapping model_key to rank within its size category (1 = best)
    """
    # Group by category
    by_category: dict[str, list[tuple[str, float]]] = {}
    for model_key, (score, _params, category) in models.items():
        if category not in by_category:
            by_category[category] = []
        by_category[category].append((model_key, score))

    # Rank within each category (higher score = better rank)
    ranks = {}
    for category, model_list in by_category.items():
        sorted_models = sorted(model_list, key=lambda x: x[1], reverse=True)
        for rank, (model_key, _score) in enumerate(sorted_models, start=1):
            ranks[model_key] = rank

    return ranks


def compute_efficiency_metrics(
    num_params: int,
    embedding_dim: int,
    size_category: str | None = None,
    shelf_score: float | None = None,
    num_params_torch: int | None = None,
    hidden_size: int | None = None,
    context_window: int | None = None,
    throughput_bytes_sec: float | None = None,
) -> EfficiencyMetrics:
    """Compute all efficiency metrics for a model.

    Args:
        num_params: Number of model parameters (from config/metadata)
        embedding_dim: Output embedding dimension
        size_category: Optional size category (computed if not provided)
        shelf_score: Optional SHELF score for efficiency calculations
        num_params_torch: Actual parameter count via torch.numel()
        hidden_size: Model hidden dimension from config
        context_window: Max position embeddings from config
        throughput_bytes_sec: Bytes processed per second during embedding

    Returns:
        EfficiencyMetrics dataclass with all computed values
    """
    if size_category is None:
        size_category = get_size_category(num_params)

    # FLOPs per token ≈ 2 * params (forward pass)
    flops_per_token = 2 * num_params

    # Relative compute vs baseline
    relative_compute = num_params / BASELINE_PARAMS

    # Score-dependent metrics
    shelf_eff = None
    shelf_compute = None

    if shelf_score is not None:
        shelf_eff = compute_shelf_eff(shelf_score, num_params)
        shelf_compute = compute_shelf_compute(shelf_score, num_params)

    return EfficiencyMetrics(
        num_params=num_params,
        embedding_dim=embedding_dim,
        size_category=size_category,
        flops_per_token=flops_per_token,
        relative_compute=relative_compute,
        shelf_eff=shelf_eff,
        shelf_compute=shelf_compute,
        num_params_torch=num_params_torch,
        hidden_size=hidden_size,
        context_window=context_window,
        throughput_bytes_sec=throughput_bytes_sec,
    )


def compute_aggregate_efficiency(
    model_results: dict[str, dict[str, Any]],
    model_configs: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Compute aggregate efficiency metrics for all models.

    Takes individual task results and model configs, computes SHELF scores
    and all efficiency metrics including Pareto optimality and size ranks.

    Args:
        model_results: Dict mapping model_key to aggregated results
            Expected format: {model_key: {"shelf_score": float, ...}}
        model_configs: Dict mapping model_key to config from YAML
            Expected format: {model_key: {"num_params": int, "embedding_dim": int, ...}}

    Returns:
        Dict mapping model_key to efficiency metrics dict
    """
    # First pass: compute individual metrics
    efficiency_data = {}
    model_tuples = {}  # For Pareto computation
    model_tuples_with_category = {}  # For size rank computation

    for model_key, results in model_results.items():
        config = model_configs.get(model_key, {})
        num_params = config.get("num_params")

        # Skip sparse models (no param count)
        if num_params is None:
            continue

        embedding_dim = config.get("embedding_dim", 0)
        size_category = config.get("size_category") or get_size_category(num_params)
        shelf_score = results.get("shelf_score")

        metrics = compute_efficiency_metrics(
            num_params=num_params,
            embedding_dim=embedding_dim,
            size_category=size_category,
            shelf_score=shelf_score,
        )

        efficiency_data[model_key] = metrics

        if shelf_score is not None:
            model_tuples[model_key] = (shelf_score, num_params)
            model_tuples_with_category[model_key] = (
                shelf_score,
                num_params,
                size_category,
            )

    # Second pass: compute Pareto and size ranks
    if model_tuples:
        pareto_set = find_pareto_optimal(model_tuples)
        size_ranks = compute_size_ranks(model_tuples_with_category)

        for model_key, metrics in efficiency_data.items():
            metrics.pareto_optimal = model_key in pareto_set
            metrics.size_rank = size_ranks.get(model_key)

    # Convert to dicts
    return {key: metrics.to_dict() for key, metrics in efficiency_data.items()}

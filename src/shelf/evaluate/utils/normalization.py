"""Embedding normalization utilities for SHELF evaluation.

This module provides functions to verify and enforce L2 normalization
of embeddings, which is critical for clustering evaluation where
Euclidean distance on normalized vectors equals cosine distance.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def is_normalized(embeddings: np.ndarray, rtol: float = 1e-4) -> bool:
    """Check if embeddings are L2-normalized (unit length).

    Args:
        embeddings: Array of shape (n_samples, embedding_dim)
        rtol: Relative tolerance for checking norm ≈ 1.0

    Returns:
        True if all embeddings have unit length within tolerance
    """
    norms = np.linalg.norm(embeddings, axis=1)
    return bool(np.allclose(norms, 1.0, rtol=rtol))


def get_norm_stats(embeddings: np.ndarray) -> dict[str, float]:
    """Get statistics about embedding norms.

    Args:
        embeddings: Array of shape (n_samples, embedding_dim)

    Returns:
        Dict with min, max, mean, std of norms
    """
    norms = np.linalg.norm(embeddings, axis=1)
    return {
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
    }


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize embeddings to unit length.

    Args:
        embeddings: Array of shape (n_samples, embedding_dim)

    Returns:
        Normalized embeddings with unit L2 norm
    """
    from sklearn.preprocessing import normalize

    return normalize(embeddings, norm="l2")


def ensure_normalized(
    embeddings: np.ndarray,
    rtol: float = 1e-4,
    warn_if_normalizing: bool = True,
) -> np.ndarray:
    """Ensure embeddings are L2-normalized, normalizing if needed.

    This function checks if embeddings are already normalized and only
    performs normalization if necessary. Logs a warning when normalization
    is applied to help identify embedders that should be normalizing.

    Args:
        embeddings: Array of shape (n_samples, embedding_dim)
        rtol: Relative tolerance for checking norm ≈ 1.0
        warn_if_normalizing: Whether to log warning if normalization needed

    Returns:
        Normalized embeddings (original if already normalized)
    """
    if is_normalized(embeddings, rtol=rtol):
        return embeddings

    if warn_if_normalizing:
        stats = get_norm_stats(embeddings)
        logger.warning(
            f"Embeddings not L2-normalized (norms: {stats['norm_min']:.4f}-{stats['norm_max']:.4f}, "
            f"mean={stats['norm_mean']:.4f}). Normalizing before clustering."
        )

    return normalize_embeddings(embeddings)

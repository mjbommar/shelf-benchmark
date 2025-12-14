"""Run context for capturing environment and version information.

This module provides RunContext for tracking all reproducibility-critical
information about an evaluation run, including package versions, git state,
dataset checksums, and platform details.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def get_git_info() -> dict[str, str | bool]:
    """Get current git commit, branch, and dirty status.

    Returns:
        Dict with 'commit', 'branch', and 'dirty' keys. If git operations fail,
        returns default values ('unknown', 'unknown', False).
    """
    info: dict[str, str | bool] = {
        "commit": "unknown",
        "branch": "unknown",
        "dirty": False,
    }

    try:
        # Get short commit hash
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            info["commit"] = result.stdout.strip()

        # Get branch name
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            info["branch"] = result.stdout.strip()

        # Check if working tree is dirty
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            info["dirty"] = bool(result.stdout.strip())

    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        # Git not available or timeout
        pass

    return info


def get_version_info() -> dict[str, str]:
    """Get version information for all relevant packages.

    Returns:
        Dict mapping package names to version strings. For packages that are
        not installed, the value will be 'not installed'.
    """
    versions: dict[str, str] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }

    # Core scientific stack (always required)
    try:
        import numpy as np

        versions["numpy"] = np.__version__
    except ImportError:
        versions["numpy"] = "not installed"

    try:
        import scipy

        versions["scipy"] = scipy.__version__
    except ImportError:
        versions["scipy"] = "not installed"

    try:
        import sklearn

        versions["sklearn"] = sklearn.__version__
    except ImportError:
        versions["sklearn"] = "not installed"

    # Optional ML packages
    try:
        import sentence_transformers

        versions["sentence_transformers"] = sentence_transformers.__version__
    except ImportError:
        versions["sentence_transformers"] = "not installed"

    try:
        import torch

        versions["torch"] = torch.__version__
        versions["cuda_available"] = str(torch.cuda.is_available())
        if torch.cuda.is_available():
            versions["cuda_version"] = torch.version.cuda or "unknown"
        else:
            versions["cuda_version"] = "N/A"
    except ImportError:
        versions["torch"] = "not installed"
        versions["cuda_available"] = "False"
        versions["cuda_version"] = "N/A"

    # SHELF package version
    try:
        from importlib.metadata import version

        versions["shelf"] = version("shelf")
    except Exception:
        versions["shelf"] = "unknown"

    return versions


def compute_dataset_checksum(dataset_version: str) -> str:
    """Compute checksum of the dataset metadata.

    Args:
        dataset_version: Dataset version string (e.g., "0.3.0")

    Returns:
        MD5 checksum of metadata.json if it exists, otherwise returns
        "v{dataset_version}" as a fallback.
    """
    metadata_path = Path("data/hf_dataset/metadata.json")
    if metadata_path.exists():
        with open(metadata_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    return f"v{dataset_version}"


@dataclass
class RunContext:
    """Captures environment and version information for reproducibility.

    This class records all information needed to reproduce an evaluation run,
    including package versions, git state, dataset checksums, and platform details.

    Attributes:
        shelf_version: SHELF package version
        python_version: Python version (e.g., "3.13.1")
        platform_info: Platform string (OS, version, architecture)
        sklearn_version: scikit-learn version
        numpy_version: NumPy version
        scipy_version: SciPy version
        torch_version: PyTorch version (None if not installed)
        sentence_transformers_version: sentence-transformers version (None if not installed)
        cuda_available: Whether CUDA is available
        cuda_version: CUDA version (None if CUDA not available)
        git_commit: Git commit hash (short)
        git_branch: Git branch name
        git_dirty: Whether working tree has uncommitted changes
        dataset_version: Dataset version string (e.g., "0.3.0")
        dataset_checksum: MD5 checksum of dataset metadata
        timestamp: UTC timestamp when context was captured
    """

    shelf_version: str
    python_version: str
    platform_info: str
    sklearn_version: str
    numpy_version: str
    scipy_version: str
    torch_version: str | None
    sentence_transformers_version: str | None
    cuda_available: bool
    cuda_version: str | None
    git_commit: str
    git_branch: str
    git_dirty: bool
    dataset_version: str
    dataset_checksum: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def capture(cls, dataset_version: str = "0.3.0") -> RunContext:
        """Capture current environment and version information.

        Args:
            dataset_version: Dataset version to use (default: "0.3.0")

        Returns:
            RunContext instance with captured information
        """
        version_info = get_version_info()
        git_info = get_git_info()
        dataset_checksum = compute_dataset_checksum(dataset_version)

        # Parse torch and sentence_transformers versions
        torch_version = version_info.get("torch")
        if torch_version == "not installed":
            torch_version = None

        st_version = version_info.get("sentence_transformers")
        if st_version == "not installed":
            st_version = None

        # Parse CUDA info
        cuda_available = version_info.get("cuda_available", "False") == "True"
        cuda_version = version_info.get("cuda_version")
        if cuda_version == "N/A":
            cuda_version = None

        return cls(
            shelf_version=version_info.get("shelf", "unknown"),
            python_version=version_info["python"],
            platform_info=version_info["platform"],
            sklearn_version=version_info.get("sklearn", "unknown"),
            numpy_version=version_info.get("numpy", "unknown"),
            scipy_version=version_info.get("scipy", "unknown"),
            torch_version=torch_version,
            sentence_transformers_version=st_version,
            cuda_available=cuda_available,
            cuda_version=cuda_version,
            git_commit=str(git_info["commit"]),
            git_branch=str(git_info["branch"]),
            git_dirty=bool(git_info["dirty"]),
            dataset_version=dataset_version,
            dataset_checksum=dataset_checksum,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON output.

        Returns:
            Dict with all context fields, suitable for JSON serialization
        """
        return {
            "shelf_version": self.shelf_version,
            "python_version": self.python_version,
            "platform_info": self.platform_info,
            "sklearn_version": self.sklearn_version,
            "numpy_version": self.numpy_version,
            "scipy_version": self.scipy_version,
            "torch_version": self.torch_version,
            "sentence_transformers_version": self.sentence_transformers_version,
            "cuda_available": self.cuda_available,
            "cuda_version": self.cuda_version,
            "git_commit": self.git_commit,
            "git_branch": self.git_branch,
            "git_dirty": self.git_dirty,
            "dataset_version": self.dataset_version,
            "dataset_checksum": self.dataset_checksum,
            "timestamp": self.timestamp.isoformat(),
        }

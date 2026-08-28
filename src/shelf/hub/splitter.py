"""
Stratified Train/Dev/Test Splitting for SHELF

This module implements multi-dimensional stratified splitting to ensure
balanced label distributions across train, dev, and test sets.

Stratification Strategy:
1. Primary: LCC code (21 classes) - ensures subject balance
2. Secondary: LCGFT category (14 classes) - ensures genre balance
3. Tertiary: Register (8 classes) - ensures style balance

The splitting uses scikit-learn's StratifiedShuffleSplit with a composite
stratification key combining LCC code and LCGFT category.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

logger = logging.getLogger(__name__)


@dataclass
class SplitConfig:
    """Configuration for dataset splitting.

    Attributes:
        train_ratio: Proportion of data for training (default: 0.6)
        dev_ratio: Proportion of data for development/validation (default: 0.2)
        test_ratio: Proportion of data for testing (default: 0.2)
        random_seed: Random seed for reproducibility (default: 42)
        min_per_class: Minimum samples per stratification class per split (default: 3)
        stratify_by: List of fields to use for stratification (default: ["lcc_code", "lcgft_category"])
        group_by: Field whose value must never straddle two splits (default: None).

            Set this to ``"spec_id"`` for any v0.4 corpus. Phase 1 gives the
            same spec to every generator, so a spec's realizations are
            near-duplicates by construction; splitting them independently
            manufactures exactly the train/test leakage v0.3.1 avoids. With
            ``group_by`` set, the split is performed over groups and expanded
            back to documents, and a post-condition verifies no group straddles.

            None reproduces v0.3.1 document-level splitting exactly.
    """

    train_ratio: float = 0.6
    dev_ratio: float = 0.2
    test_ratio: float = 0.2
    random_seed: int = 42
    min_per_class: int = 3
    stratify_by: list[str] = field(
        default_factory=lambda: ["lcc_code", "lcgft_category"]
    )
    group_by: str | None = None

    def __post_init__(self) -> None:
        """Validate split ratios sum to 1.0."""
        total = self.train_ratio + self.dev_ratio + self.test_ratio
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(
                f"Split ratios must sum to 1.0, got {total:.4f} "
                f"(train={self.train_ratio}, dev={self.dev_ratio}, test={self.test_ratio})"
            )
        if any(r <= 0 for r in [self.train_ratio, self.dev_ratio, self.test_ratio]):
            raise ValueError("All split ratios must be positive")

    @classmethod
    def from_ratios(
        cls,
        train: float = 0.6,
        dev: float = 0.2,
        test: float = 0.2,
        **kwargs: Any,
    ) -> SplitConfig:
        """Create config from explicit ratios."""
        return cls(train_ratio=train, dev_ratio=dev, test_ratio=test, **kwargs)

    @classmethod
    def standard(cls) -> SplitConfig:
        """Standard 60/20/20 split."""
        return cls()

    @classmethod
    def large_train(cls) -> SplitConfig:
        """80/10/10 split for more training data."""
        return cls(train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1)

    @classmethod
    def balanced(cls) -> SplitConfig:
        """Balanced 70/15/15 split."""
        return cls(train_ratio=0.7, dev_ratio=0.15, test_ratio=0.15)


@dataclass
class SplitResult:
    """Result of a stratified split operation.

    Attributes:
        train: List of documents in training set
        dev: List of documents in development set
        test: List of documents in test set
        config: Configuration used for splitting
        statistics: Dictionary of split statistics
        checksum: SHA256 checksum of the split for reproducibility verification
    """

    train: list[dict[str, Any]]
    dev: list[dict[str, Any]]
    test: list[dict[str, Any]]
    config: SplitConfig
    statistics: dict[str, Any]
    checksum: str

    @property
    def total_documents(self) -> int:
        return len(self.train) + len(self.dev) + len(self.test)

    def get_split(self, name: str) -> list[dict[str, Any]]:
        """Get split by name."""
        splits = {"train": self.train, "dev": self.dev, "test": self.test}
        if name not in splits:
            raise ValueError(
                f"Unknown split: {name}. Must be one of {list(splits.keys())}"
            )
        return splits[name]


class StratifiedSplitter:
    """Performs stratified splitting of SHELF documents.

    This splitter ensures that the distribution of labels (LCC codes, LCGFT categories,
    registers, etc.) is preserved across all splits, which is critical for fair
    evaluation of classification models.

    Example:
        >>> config = SplitConfig(train_ratio=0.6, dev_ratio=0.2, test_ratio=0.2)
        >>> splitter = StratifiedSplitter(config)
        >>> result = splitter.split(documents)
        >>> print(f"Train: {len(result.train)}, Dev: {len(result.dev)}, Test: {len(result.test)}")
    """

    def __init__(self, config: SplitConfig | None = None) -> None:
        """Initialize splitter with configuration.

        Args:
            config: Split configuration. Defaults to standard 60/20/20 split.
        """
        self.config = config or SplitConfig.standard()

    def _create_stratification_key(self, doc: dict[str, Any]) -> str:
        """Create composite stratification key from document fields.

        Combines multiple fields into a single string key for stratification.
        """
        parts = []
        for field_name in self.config.stratify_by:
            value = doc.get(field_name, "unknown")
            if value is None:
                value = "none"
            parts.append(f"{field_name}={value}")
        return "|".join(parts)

    def _validate_stratification(
        self,
        documents: list[dict[str, Any]],
        stratification_keys: list[str],
    ) -> None:
        """Validate that stratification is feasible.

        Checks that each stratification class has enough samples to appear
        in all splits with the minimum required count.
        """
        key_counts = Counter(stratification_keys)
        n_splits = 3  # train, dev, test
        min_required = self.config.min_per_class * n_splits

        small_classes = {k: v for k, v in key_counts.items() if v < min_required}
        if small_classes:
            # Log warning but don't fail - we'll handle this by combining small classes
            print(
                f"Warning: {len(small_classes)} stratification classes have fewer than "
                f"{min_required} samples and may not appear in all splits"
            )

    def _combine_small_classes(
        self,
        stratification_keys: list[str],
        min_size: int = 10,
    ) -> list[str]:
        """Combine small classes into an 'other' category for more stable stratification.

        Args:
            stratification_keys: Original stratification keys
            min_size: Minimum class size to keep separate

        Returns:
            Modified stratification keys with small classes combined
        """
        key_counts = Counter(stratification_keys)

        # Identify small classes
        small_classes = {k for k, v in key_counts.items() if v < min_size}

        if not small_classes:
            return stratification_keys

        # Replace small class keys with combined key based on first stratification field only
        modified_keys = []
        for key in stratification_keys:
            if key in small_classes:
                # Extract first field (usually lcc_code) to maintain some stratification
                first_field = key.split("|")[0]
                modified_keys.append(f"{first_field}|_combined")
            else:
                modified_keys.append(key)

        return modified_keys

    def _compute_checksum(self, result: dict[str, list[str]]) -> str:
        """Compute SHA256 checksum of split document IDs for reproducibility."""
        checksums = {}
        for split_name, doc_ids in result.items():
            id_string = ",".join(sorted(doc_ids))
            checksums[split_name] = hashlib.sha256(id_string.encode()).hexdigest()[:16]

        combined = "|".join(f"{k}:{v}" for k, v in sorted(checksums.items()))
        return hashlib.sha256(combined.encode()).hexdigest()[:32]

    def _compute_statistics(
        self,
        train: list[dict[str, Any]],
        dev: list[dict[str, Any]],
        test: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Compute statistics about the split for validation and documentation."""
        stats: dict[str, Any] = {
            "counts": {
                "train": len(train),
                "dev": len(dev),
                "test": len(test),
                "total": len(train) + len(dev) + len(test),
            },
            "ratios": {
                "train": len(train) / (len(train) + len(dev) + len(test)),
                "dev": len(dev) / (len(train) + len(dev) + len(test)),
                "test": len(test) / (len(train) + len(dev) + len(test)),
            },
            "distributions": {},
        }

        # Compute distribution stats for key fields
        for field_name in ["lcc_code", "lcgft_category", "register", "target_length"]:
            stats["distributions"][field_name] = {}
            for split_name, split_docs in [
                ("train", train),
                ("dev", dev),
                ("test", test),
            ]:
                field_values = [doc.get(field_name, "unknown") for doc in split_docs]
                stats["distributions"][field_name][split_name] = dict(
                    Counter(field_values)
                )

        return stats

    def _split_grouped(self, documents: list[dict[str, Any]]) -> SplitResult:
        """Split so that a whole group lands in exactly one split.

        Groups are split rather than documents. Each group contributes one
        representative to stratification, since every member of a spec group
        shares that spec's labels by construction.

        Raises:
            ValueError: If any document lacks the group field, or if the
                resulting split somehow straddles a group. The second check is
                a post-condition, not a guard against bad input -- if it ever
                fires, the split is silently leaking and must not be published.
        """
        group_field = self.config.group_by
        assert group_field is not None

        missing = sum(1 for doc in documents if not doc.get(group_field))
        if missing:
            raise ValueError(
                f"{missing} of {len(documents)} documents have no '{group_field}'; "
                "cannot group-split without it"
            )

        groups: dict[str, list[dict[str, Any]]] = {}
        for doc in documents:
            groups.setdefault(str(doc[group_field]), []).append(doc)

        if len(groups) < 100:
            raise ValueError(
                f"Group-splitting needs at least 100 distinct '{group_field}' values, "
                f"got {len(groups)} across {len(documents)} documents. The split is "
                f"performed over groups, so a coarse grouping field leaves too few "
                f"units to stratify."
            )

        logger.info(
            "Group-splitting %d documents into %d '%s' groups (mean %.1f docs/group)",
            len(documents),
            len(groups),
            group_field,
            len(documents) / len(groups),
        )

        # One representative per group carries the stratification labels.
        representatives = [members[0] for members in groups.values()]
        group_keys = list(groups)

        sub_config = replace(self.config, group_by=None)
        group_result = StratifiedSplitter(sub_config).split(
            [
                {**rep, "__group_key__": key}
                for rep, key in zip(representatives, group_keys, strict=True)
            ]
        )

        expanded: dict[str, list[dict[str, Any]]] = {}
        assigned: dict[str, str] = {}
        for split_name in ("train", "dev", "test"):
            docs: list[dict[str, Any]] = []
            for rep in group_result.get_split(split_name):
                key = rep["__group_key__"]
                assigned[key] = split_name
                docs.extend(groups[key])
            expanded[split_name] = docs

        straddling = self._find_straddling_groups(expanded, group_field)
        if straddling:
            raise ValueError(
                f"{len(straddling)} '{group_field}' groups straddle splits after "
                f"grouping (e.g. {sorted(straddling)[:3]}); this would leak"
            )

        statistics = dict(group_result.statistics or {})
        statistics["grouping"] = {
            "group_by": group_field,
            "n_groups": len(groups),
            "n_documents": len(documents),
            "mean_docs_per_group": round(len(documents) / len(groups), 4),
            "groups_per_split": {
                name: sum(1 for v in assigned.values() if v == name)
                for name in ("train", "dev", "test")
            },
        }

        return SplitResult(
            train=expanded["train"],
            dev=expanded["dev"],
            test=expanded["test"],
            config=self.config,
            checksum=self._compute_checksum(
                {
                    name: [str(d.get("id", "")) for d in expanded[name]]
                    for name in ("train", "dev", "test")
                }
            ),
            statistics=statistics,
        )

    @staticmethod
    def _find_straddling_groups(
        splits: dict[str, list[dict[str, Any]]], group_field: str
    ) -> set[str]:
        """Return group values that appear in more than one split."""
        seen: dict[str, set[str]] = {}
        for split_name, docs in splits.items():
            for doc in docs:
                seen.setdefault(str(doc.get(group_field, "")), set()).add(split_name)
        return {key for key, names in seen.items() if len(names) > 1}

    def split(self, documents: list[dict[str, Any]]) -> SplitResult:
        """Perform stratified split on documents.

        Args:
            documents: List of document dictionaries to split

        Returns:
            SplitResult containing train, dev, test sets with statistics
        """
        if len(documents) < 100:
            raise ValueError(
                f"Need at least 100 documents for splitting, got {len(documents)}"
            )

        if self.config.group_by:
            return self._split_grouped(documents)

        # Set random seeds for reproducibility
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)

        # Create stratification keys
        stratification_keys = [
            self._create_stratification_key(doc) for doc in documents
        ]

        # Validate and potentially combine small classes
        self._validate_stratification(documents, stratification_keys)
        stratification_keys = self._combine_small_classes(stratification_keys)

        # Convert to numpy arrays for sklearn
        indices = np.arange(len(documents))
        labels = np.array(stratification_keys)

        # First split: separate test set
        test_size = self.config.test_ratio
        splitter1 = StratifiedShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=self.config.random_seed,
        )

        train_dev_idx, test_idx = next(splitter1.split(indices, labels))

        # Second split: separate train and dev from remaining
        # Adjust dev ratio for remaining data
        remaining_ratio = 1.0 - self.config.test_ratio
        dev_ratio_adjusted = self.config.dev_ratio / remaining_ratio

        train_dev_labels = labels[train_dev_idx]
        splitter2 = StratifiedShuffleSplit(
            n_splits=1,
            test_size=dev_ratio_adjusted,
            random_state=self.config.random_seed + 1,  # Different seed for second split
        )

        train_idx_rel, dev_idx_rel = next(
            splitter2.split(train_dev_idx, train_dev_labels)
        )

        # Map relative indices back to original
        train_idx = train_dev_idx[train_idx_rel]
        dev_idx = train_dev_idx[dev_idx_rel]

        # Create split document lists
        train = [documents[i] for i in train_idx]
        dev = [documents[i] for i in dev_idx]
        test = [documents[i] for i in test_idx]

        # Compute statistics and checksum
        statistics = self._compute_statistics(train, dev, test)
        checksum = self._compute_checksum(
            {
                "train": [doc["id"] for doc in train],
                "dev": [doc["id"] for doc in dev],
                "test": [doc["id"] for doc in test],
            }
        )

        return SplitResult(
            train=train,
            dev=dev,
            test=test,
            config=self.config,
            statistics=statistics,
            checksum=checksum,
        )

    def verify_stratification(self, result: SplitResult) -> dict[str, Any]:
        """Verify that stratification was successful.

        Computes distribution divergence metrics to check that splits
        are properly stratified.

        Args:
            result: Split result to verify

        Returns:
            Dictionary with verification metrics
        """
        verification: dict[str, Any] = {"passed": True, "issues": [], "metrics": {}}

        for field_name in self.config.stratify_by:
            distributions = result.statistics["distributions"].get(field_name, {})
            if not distributions:
                continue

            # Get all unique values across splits
            all_values = set()
            for split_dist in distributions.values():
                all_values.update(split_dist.keys())

            # Check coverage in each split
            for split_name, split_dist in distributions.items():
                missing = all_values - set(split_dist.keys())
                if missing:
                    verification["issues"].append(
                        f"{field_name}: {len(missing)} values missing from {split_name} split"
                    )

            # Compute max ratio divergence
            train_dist = distributions.get("train", {})
            for split_name in ["dev", "test"]:
                split_dist = distributions.get(split_name, {})
                max_divergence = 0.0
                for value in all_values:
                    train_ratio = train_dist.get(value, 0) / max(
                        sum(train_dist.values()), 1
                    )
                    split_ratio = split_dist.get(value, 0) / max(
                        sum(split_dist.values()), 1
                    )
                    divergence = abs(train_ratio - split_ratio)
                    max_divergence = max(max_divergence, divergence)

                verification["metrics"][f"{field_name}_{split_name}_max_divergence"] = (
                    max_divergence
                )

                if max_divergence > 0.1:  # More than 10% divergence is concerning
                    verification["issues"].append(
                        f"{field_name}: High divergence ({max_divergence:.2%}) between train and {split_name}"
                    )

        verification["passed"] = len(verification["issues"]) == 0
        return verification


def create_splits(
    artifacts_dir: str | Path,
    config: SplitConfig | None = None,
    output_dir: str | Path | None = None,
    filter_commit: str | None = None,
    filter_code_version: str | None = None,
) -> SplitResult:
    """Convenience function to create splits from artifact files.

    Args:
        artifacts_dir: Directory containing JSON artifact files
        config: Split configuration (defaults to standard 60/20/20)
        output_dir: Optional directory to save split files
        filter_commit: Only include artifacts from this git commit (e.g., "ee52a05")
        filter_code_version: Only include artifacts from this code version (e.g., "ee52a05*")

    Returns:
        SplitResult with train, dev, test splits

    Example:
        >>> # Create splits from all artifacts
        >>> result = create_splits("data/artifacts/", output_dir="data/splits/")
        >>> print(f"Created splits with checksum: {result.checksum}")

        >>> # Create splits from a specific code version only
        >>> result = create_splits("data/artifacts/", filter_commit="ee52a05")
        >>> print(f"Created splits from commit ee52a05: {len(result.train)} train docs")
    """
    artifacts_path = Path(artifacts_dir)
    if not artifacts_path.exists():
        raise FileNotFoundError(f"Artifacts directory not found: {artifacts_path}")

    # Load all documents (with optional filtering)
    documents = []
    filtered_count = 0
    empty_body_count = 0
    for json_file in sorted(artifacts_path.glob("*.json")):
        with open(json_file, encoding="utf-8") as f:
            doc = json.load(f)

        # Filter out documents with empty body (failed generations)
        body = doc.get("body", "")
        if not body or not body.strip():
            empty_body_count += 1
            continue

        # Apply commit filtering if requested
        if filter_commit:
            doc_commit = doc.get("git_commit", "")
            if doc_commit != filter_commit:
                filtered_count += 1
                continue

        # Apply code_version filtering if requested
        if filter_code_version:
            doc_version = doc.get("code_version", "")
            if doc_version != filter_code_version:
                filtered_count += 1
                continue

        documents.append(doc)

    if empty_body_count > 0:
        print(
            f"Filtered out {empty_body_count} documents with empty body (failed generations)"
        )

    if filtered_count > 0:
        print(f"Filtered out {filtered_count} documents (commit/version mismatch)")

    if not documents:
        raise ValueError(f"No JSON files found in {artifacts_path}")

    print(f"Loaded {len(documents)} documents from {artifacts_path}")

    # Create splits
    splitter = StratifiedSplitter(config)
    result = splitter.split(documents)

    # Verify stratification
    verification = splitter.verify_stratification(result)
    if not verification["passed"]:
        print("Warning: Stratification verification found issues:")
        for issue in verification["issues"]:
            print(f"  - {issue}")

    # Save splits if output directory specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for split_name in ["train", "dev", "test"]:
            split_docs = result.get_split(split_name)
            output_file = output_path / f"{split_name}.jsonl"

            with open(output_file, "w", encoding="utf-8") as f:
                for doc in split_docs:
                    f.write(json.dumps(doc, ensure_ascii=False) + "\n")

            print(f"Saved {len(split_docs)} documents to {output_file}")

        # Save split metadata
        metadata = {
            "config": {
                "train_ratio": result.config.train_ratio,
                "dev_ratio": result.config.dev_ratio,
                "test_ratio": result.config.test_ratio,
                "random_seed": result.config.random_seed,
                "stratify_by": result.config.stratify_by,
            },
            "statistics": result.statistics,
            "checksum": result.checksum,
        }

        metadata_file = output_path / "split_metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"Saved split metadata to {metadata_file}")

    return result

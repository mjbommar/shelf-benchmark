"""
HuggingFace Dataset Preparation and Upload for SHELF

This module handles the conversion of SHELF documents to HuggingFace Dataset
format and upload to the HuggingFace Hub.

Key Features:
- Converts JSON documents to HuggingFace Dataset format
- Handles multi-label fields (topics, geographic) properly
- Uploads in Parquet format for efficient storage and loading
- Supports both public and private dataset repositories

References:
- https://huggingface.co/docs/datasets/en/upload_dataset
- https://huggingface.co/docs/hub/en/datasets-adding
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from shelf.hub.splitter import SplitConfig, SplitResult, create_splits


# Feature schema for the dataset
# NOTE: "text" is body-only (no title) to avoid label leakage from titles.
# Title is available as a separate field for users who want it.
SHELF_FEATURES = {
    # Core identifiers
    "id": "string",
    "text": "string",  # Body-only (default for evaluation)
    "title": "string",  # Separate field (may contain label leakage)
    "body": "string",  # Raw body text
    "word_count": "int32",
    # Library of Congress Classification
    "lcc_code": "string",  # Single label: A, B, C, ..., Z
    "lcc_name": "string",
    "lcc_uri": "string",
    # Library of Congress Genre/Form Terms
    "lcgft_category": "string",  # 14 categories
    "lcgft_form": "string",  # 133 forms
    # Multi-label fields
    "topics": "list<string>",  # Multi-label: 112 unique topics
    "geographic": "list<string>",  # Multi-label: 44 locations
    # Audience and Register
    "audience": "string",  # 25 types (nullable)
    "register": "string",  # 8 types
    "register_description": "string",
    # Generation metadata
    "target_length": "string",  # 8 categories
    "target_word_range": "list<int32>",  # [min, max]
    "temperature": "float32",
    "top_p": "float32",
    "model": "string",
    # Git versioning for reproducibility (allows filtering by generation code)
    "git_commit": "string",  # Short commit hash, e.g., "ee52a05"
    "code_version": "string",  # commit + "*" if dirty, e.g., "ee52a05*"
    # Gemini thinking configuration (optional)
    "thinking_budget": "int32",  # Thinking token budget (null if not applicable)
    "token_multiplier": "float32",  # Output token multiplier for thinking overhead
}


@dataclass
class DatasetConfig:
    """Configuration for HuggingFace dataset creation and upload.

    Attributes:
        repo_id: HuggingFace repository ID (e.g., "username/shelf")
        private: Whether to create a private repository
        commit_message: Commit message for the upload
        revision: Branch/revision to push to
        max_shard_size: Maximum size of data shards
        include_generation_metadata: Whether to include generation fields (model, temperature, etc.)
    """

    repo_id: str = "mjbommar/SHELF"
    private: bool = False
    commit_message: str = "Upload SHELF dataset"
    revision: str = "main"
    max_shard_size: str = "500MB"
    include_generation_metadata: bool = True
    token: str | None = None


@dataclass
class SHELFDataset:
    """Container for SHELF dataset with train/dev/test splits.

    Attributes:
        train: Training set documents
        dev: Development/validation set documents
        test: Test set documents
        config: Dataset configuration
        split_config: Split configuration used
        metadata: Additional metadata about the dataset
    """

    train: list[dict[str, Any]]
    dev: list[dict[str, Any]]
    test: list[dict[str, Any]]
    config: DatasetConfig
    split_config: SplitConfig
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_split_result(
        cls,
        result: SplitResult,
        config: DatasetConfig | None = None,
    ) -> SHELFDataset:
        """Create dataset from a SplitResult."""
        return cls(
            train=result.train,
            dev=result.dev,
            test=result.test,
            config=config or DatasetConfig(),
            split_config=result.config,
            metadata={
                "checksum": result.checksum,
                "statistics": result.statistics,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    @classmethod
    def from_artifacts(
        cls,
        artifacts_dir: str | Path,
        split_config: SplitConfig | None = None,
        dataset_config: DatasetConfig | None = None,
    ) -> SHELFDataset:
        """Create dataset directly from artifact files."""
        result = create_splits(artifacts_dir, config=split_config)
        return cls.from_split_result(result, config=dataset_config)

    @property
    def total_documents(self) -> int:
        return len(self.train) + len(self.dev) + len(self.test)

    def get_split_documents(self, split: str) -> list[dict[str, Any]]:
        """Get documents for a specific split."""
        splits = {"train": self.train, "dev": self.dev, "test": self.test}
        if split not in splits:
            raise ValueError(f"Unknown split: {split}")
        return splits[split]


def _normalize_document(
    doc: dict[str, Any],
    include_generation_metadata: bool = True,
) -> dict[str, Any]:
    """Normalize a document for HuggingFace Dataset format.

    Handles:
    - Null values -> appropriate defaults
    - List fields that might be None -> empty lists
    - Type conversions for proper schema compliance

    NOTE: The "text" field is set to body-only (no title) to avoid label
    leakage from titles. Title is available as a separate field for users
    who want to include it in their evaluation.
    """
    body = str(doc.get("body", ""))
    normalized = {
        # Core fields
        "id": str(doc.get("id", "")),
        "text": body,  # Body-only (no title) to avoid label leakage
        "title": str(doc.get("title", "")),  # Separate field for optional use
        "body": body,  # Raw body preserved for reference
        "word_count": int(doc.get("word_count", 0)),
        # LCC fields
        "lcc_code": str(doc.get("lcc_code", "")),
        "lcc_name": str(doc.get("lcc_name", "")),
        "lcc_uri": str(doc.get("lcc_uri", "")),
        # LCGFT fields
        "lcgft_category": str(doc.get("lcgft_category", "")),
        "lcgft_form": str(doc.get("lcgft_form", "")),
        # Multi-label fields (ensure lists)
        "topics": list(doc.get("topics") or []),
        "geographic": list(doc.get("geographic") or []),
        # Audience and register
        "audience": str(doc.get("audience") or ""),
        "register": str(doc.get("register", "")),
        "register_description": str(doc.get("register_description", "")),
        # Length metadata
        "target_length": str(doc.get("target_length", "")),
        "target_word_range": list(doc.get("target_word_range") or [0, 0]),
    }

    if include_generation_metadata:
        # Handle thinking_budget: can be None, 0, or positive int
        thinking_budget = doc.get("thinking_budget")
        if thinking_budget is None:
            thinking_budget = -1  # Use -1 to represent "not applicable" (null)
        else:
            thinking_budget = int(thinking_budget)

        # Handle token_multiplier: default to 1.0 if not present
        token_multiplier = doc.get("token_multiplier")
        if token_multiplier is None:
            token_multiplier = 1.0
        else:
            token_multiplier = float(token_multiplier)

        normalized.update(
            {
                "temperature": float(doc.get("temperature", 0.0)),
                "top_p": float(doc.get("top_p", 0.0)),
                "model": str(doc.get("model", "")),
                # Git versioning for reproducibility
                "git_commit": str(doc.get("git_commit") or ""),
                "code_version": str(doc.get("code_version") or ""),
                # Gemini thinking configuration
                "thinking_budget": thinking_budget,
                "token_multiplier": token_multiplier,
            }
        )

    return normalized


def _create_hf_features(include_generation_metadata: bool = True) -> Any:
    """Create HuggingFace Features schema.

    NOTE: The "text" field is body-only (no title) to avoid label leakage.
    Title is available as a separate field.

    Returns:
        datasets.Features object
    """
    try:
        from datasets import Features, Sequence, Value
    except ImportError:
        raise ImportError(
            "The 'datasets' library is required. Install with: pip install datasets"
        )

    features = {
        "id": Value("string"),
        "text": Value("string"),  # Body-only (default for evaluation)
        "title": Value("string"),  # Separate field (may contain label leakage)
        "body": Value("string"),  # Raw body preserved
        "word_count": Value("int32"),
        "lcc_code": Value("string"),
        "lcc_name": Value("string"),
        "lcc_uri": Value("string"),
        "lcgft_category": Value("string"),
        "lcgft_form": Value("string"),
        "topics": Sequence(Value("string")),
        "geographic": Sequence(Value("string")),
        "audience": Value("string"),
        "register": Value("string"),
        "register_description": Value("string"),
        "target_length": Value("string"),
        "target_word_range": Sequence(Value("int32")),
    }

    if include_generation_metadata:
        features.update(
            {
                "temperature": Value("float32"),
                "top_p": Value("float32"),
                "model": Value("string"),
                # Git versioning for reproducibility
                "git_commit": Value("string"),
                "code_version": Value("string"),
                # Gemini thinking configuration
                "thinking_budget": Value("int32"),  # -1 means not applicable
                "token_multiplier": Value("float32"),
            }
        )

    return Features(features)


def prepare_dataset(
    dataset: SHELFDataset,
) -> Any:  # Returns datasets.DatasetDict
    """Convert SHELFDataset to HuggingFace DatasetDict.

    Args:
        dataset: SHELFDataset with train/dev/test splits

    Returns:
        HuggingFace DatasetDict with train, validation, and test splits

    Example:
        >>> shelf_data = SHELFDataset.from_artifacts("data/artifacts/")
        >>> hf_dataset = prepare_dataset(shelf_data)
        >>> print(hf_dataset)
        DatasetDict({
            train: Dataset({features: [...], num_rows: 6000})
            validation: Dataset({features: [...], num_rows: 2000})
            test: Dataset({features: [...], num_rows: 2000})
        })
    """
    try:
        from datasets import Dataset, DatasetDict
    except ImportError:
        raise ImportError(
            "The 'datasets' library is required. Install with: pip install datasets"
        )

    include_gen = dataset.config.include_generation_metadata
    features = _create_hf_features(include_gen)

    def make_dataset(docs: list[dict]) -> Dataset:
        normalized = [_normalize_document(d, include_gen) for d in docs]

        # Convert to columnar format
        columns: dict[str, list] = {k: [] for k in normalized[0].keys()}
        for doc in normalized:
            for k, v in doc.items():
                columns[k].append(v)

        return Dataset.from_dict(columns, features=features)

    # Note: HuggingFace uses "validation" not "dev"
    return DatasetDict(
        {
            "train": make_dataset(dataset.train),
            "validation": make_dataset(dataset.dev),
            "test": make_dataset(dataset.test),
        }
    )


def push_to_hub(
    dataset: SHELFDataset,
    repo_id: str | None = None,
    private: bool | None = None,
    token: str | None = None,
    commit_message: str | None = None,
) -> str:
    """Upload SHELF dataset to HuggingFace Hub.

    Args:
        dataset: SHELFDataset to upload
        repo_id: Repository ID (overrides dataset.config.repo_id)
        private: Whether repository is private (overrides dataset.config.private)
        token: HuggingFace token (overrides dataset.config.token)
        commit_message: Commit message (overrides dataset.config.commit_message)

    Returns:
        URL of the uploaded dataset

    Example:
        >>> dataset = SHELFDataset.from_artifacts("data/artifacts/")
        >>> url = push_to_hub(dataset, repo_id="myorg/shelf", private=False)
        >>> print(f"Dataset uploaded to: {url}")
    """
    # Resolve configuration
    repo_id = repo_id or dataset.config.repo_id
    private = private if private is not None else dataset.config.private
    token = token or dataset.config.token
    commit_message = commit_message or dataset.config.commit_message

    # Prepare HuggingFace dataset
    hf_dataset = prepare_dataset(dataset)

    print(f"Uploading dataset to {repo_id}...")
    print(f"  Train: {len(hf_dataset['train'])} documents")
    print(f"  Validation: {len(hf_dataset['validation'])} documents")
    print(f"  Test: {len(hf_dataset['test'])} documents")

    # Push to hub
    hf_dataset.push_to_hub(
        repo_id=repo_id,
        private=private,
        token=token,
        commit_message=commit_message,
        max_shard_size=dataset.config.max_shard_size,
    )

    url = f"https://huggingface.co/datasets/{repo_id}"
    print(f"Dataset uploaded successfully: {url}")

    return url


def push_folder_to_hub(
    folder_path: str | Path,
    repo_id: str,
    private: bool = False,
    token: str | None = None,
    commit_message: str = "Upload SHELF dataset",
) -> str:
    """Upload entire dataset folder to HuggingFace Hub including all configurations.

    This uploads all files in the folder (parquet files, README, metadata, pairs)
    to the HuggingFace Hub repository. This is the recommended way to upload
    SHELF datasets as it preserves all configurations including pair datasets.

    Args:
        folder_path: Path to the dataset folder (e.g., "data/hf_dataset")
        repo_id: HuggingFace repository ID (e.g., "username/SHELF")
        private: Whether repository is private
        token: HuggingFace token (or use HF_TOKEN env var)
        commit_message: Commit message for the upload

    Returns:
        URL of the uploaded dataset

    Example:
        >>> url = push_folder_to_hub("data/hf_dataset", "myorg/SHELF")
        >>> print(f"Dataset uploaded to: {url}")
    """
    from huggingface_hub import HfApi

    folder_path = Path(folder_path)
    if not folder_path.exists():
        raise FileNotFoundError(f"Dataset folder not found: {folder_path}")

    api = HfApi(token=token)

    # Create the repository if it doesn't exist
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )

    print(f"Uploading folder {folder_path} to {repo_id}...")

    # Upload the entire folder
    api.upload_folder(
        folder_path=str(folder_path),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_message,
    )

    url = f"https://huggingface.co/datasets/{repo_id}"
    print(f"Dataset folder uploaded successfully: {url}")

    return url


def generate_pairs(
    documents: list[dict[str, Any]],
    label_field: str,
    num_pairs: int = 5000,
    positive_ratio: float = 0.5,
    seed: int = 42,
    treat_null_as_class: bool = False,
) -> list[dict[str, Any]]:
    """Generate document pairs for pair classification tasks.

    Args:
        documents: List of document dictionaries
        label_field: Field to use for positive/negative determination (e.g., 'lcc_code', 'lcgft_form', 'audience')
        num_pairs: Total number of pairs to generate
        positive_ratio: Ratio of positive pairs (same label)
        seed: Random seed for reproducibility
        treat_null_as_class: If True, treat None/null as a valid class (e.g., for audience field)

    Returns:
        List of pair dictionaries with doc_a, doc_b, and label
    """
    import random
    from collections import defaultdict

    random.seed(seed)

    # Group documents by label
    label_to_docs: dict[str | None, list[dict]] = defaultdict(list)
    for doc in documents:
        label = doc.get(label_field)
        # Handle null values based on treat_null_as_class flag
        if treat_null_as_class:
            # Normalize empty string and None to None
            if label == "" or label is None:
                label = None
            label_to_docs[label].append(doc)
        else:
            # Only include documents with non-null labels
            if label:
                label_to_docs[label].append(doc)

    labels_with_multiple = [
        label for label, docs in label_to_docs.items() if len(docs) >= 2
    ]

    pairs = []
    num_positive = int(num_pairs * positive_ratio)
    num_negative = num_pairs - num_positive

    # Generate positive pairs (same label)
    for _ in range(num_positive):
        label = random.choice(labels_with_multiple)
        doc_a, doc_b = random.sample(label_to_docs[label], 2)
        pairs.append(
            {
                "id": f"pair_{len(pairs):06d}",
                "doc_a_id": doc_a["id"],
                "doc_a_title": doc_a.get("title", ""),
                "doc_a_body": doc_a.get("body", ""),
                "doc_b_id": doc_b["id"],
                "doc_b_title": doc_b.get("title", ""),
                "doc_b_body": doc_b.get("body", ""),
                "label": 1,
                "label_field": label_field,
            }
        )

    # Generate negative pairs (different labels)
    all_labels = list(label_to_docs.keys())
    for _ in range(num_negative):
        label_a, label_b = random.sample(all_labels, 2)
        doc_a = random.choice(label_to_docs[label_a])
        doc_b = random.choice(label_to_docs[label_b])
        pairs.append(
            {
                "id": f"pair_{len(pairs):06d}",
                "doc_a_id": doc_a["id"],
                "doc_a_title": doc_a.get("title", ""),
                "doc_a_body": doc_a.get("body", ""),
                "doc_b_id": doc_b["id"],
                "doc_b_title": doc_b.get("title", ""),
                "doc_b_body": doc_b.get("body", ""),
                "label": 0,
                "label_field": label_field,
            }
        )

    random.shuffle(pairs)
    return pairs


def prepare_pair_dataset(
    dataset: SHELFDataset,
    label_field: str = "lcc_code",
    pairs_per_split: dict[str, int] | None = None,
    treat_null_as_class: bool = False,
) -> Any:  # Returns datasets.DatasetDict
    """Prepare pair classification dataset.

    Args:
        dataset: SHELFDataset with train/dev/test splits
        label_field: Field to use for pair labels ('lcc_code', 'lcgft_form', or 'audience')
        pairs_per_split: Number of pairs per split (default: train=20000, validation=4000, test=4000)
        treat_null_as_class: If True, treat None/null as a valid class (e.g., for audience field)

    Returns:
        HuggingFace DatasetDict with pair data
    """
    try:
        from datasets import Dataset, DatasetDict, Features, Value
    except ImportError:
        raise ImportError(
            "The 'datasets' library is required. Install with: pip install datasets"
        )

    pairs_per_split = pairs_per_split or {
        "train": 20000,
        "validation": 4000,
        "test": 4000,
    }

    features = Features(
        {
            "id": Value("string"),
            "doc_a_id": Value("string"),
            "doc_a_title": Value("string"),
            "doc_a_body": Value("string"),
            "doc_b_id": Value("string"),
            "doc_b_title": Value("string"),
            "doc_b_body": Value("string"),
            "label": Value("int32"),
            "label_field": Value("string"),
        }
    )

    splits = {}
    for split_name, docs in [
        ("train", dataset.train),
        ("validation", dataset.dev),
        ("test", dataset.test),
    ]:
        num_pairs = pairs_per_split.get(split_name, 4000)
        pairs = generate_pairs(
            docs,
            label_field,
            num_pairs=num_pairs,
            seed=42,
            treat_null_as_class=treat_null_as_class,
        )

        # Convert to columnar format
        columns: dict[str, list] = {k: [] for k in pairs[0].keys()}
        for pair in pairs:
            for k, v in pair.items():
                columns[k].append(v)

        splits[split_name] = Dataset.from_dict(columns, features=features)

    return DatasetDict(splits)


def generate_topic_pairs(
    documents: list[dict[str, Any]],
    num_pairs: int = 5000,
    mode: str = "binary",
    target_distribution: dict[int, float] | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Generate document pairs for topic overlap classification tasks.

    Args:
        documents: List of document dictionaries
        num_pairs: Total number of pairs to generate
        mode: "binary" (any overlap) or "graded" (count overlaps)
        target_distribution: Distribution of overlap counts {0: 0.4, 1: 0.3, 2: 0.2, 3: 0.1}
        seed: Random seed for reproducibility

    Returns:
        List of pair dictionaries with doc_a, doc_b, label, and overlap_count

    Example:
        >>> # Binary: label=1 if any topic overlap, label=0 otherwise
        >>> pairs = generate_topic_pairs(docs, num_pairs=1000, mode="binary")
        >>> # Graded: label=number of shared topics (0, 1, 2, 3+)
        >>> pairs = generate_topic_pairs(docs, num_pairs=1000, mode="graded")
    """
    import random

    random.seed(seed)

    # Default distributions
    if target_distribution is None:
        if mode == "binary":
            # 50% no overlap, 50% any overlap
            target_distribution = {0: 0.5, 1: 0.5}
        else:  # graded
            # 40% no overlap, 30% one topic, 20% two topics, 10% three+ topics
            target_distribution = {0: 0.4, 1: 0.3, 2: 0.2, 3: 0.1}

    # Build topic indices for efficient sampling
    topic_to_docs: dict[str, list[dict]] = {}
    for doc in documents:
        topics = doc.get("topics", [])
        for topic in topics:
            if topic not in topic_to_docs:
                topic_to_docs[topic] = []
            topic_to_docs[topic].append(doc)

    # Filter out documents with no topics
    docs_with_topics = [doc for doc in documents if doc.get("topics")]

    pairs = []
    attempts_per_target = num_pairs * 3  # Allow multiple attempts

    # Generate pairs for each target overlap count
    for overlap_target, ratio in sorted(target_distribution.items()):
        num_needed = int(num_pairs * ratio)
        attempts = 0
        found = 0

        while found < num_needed and attempts < attempts_per_target:
            attempts += 1

            if overlap_target == 0:
                # Sample two random documents, check for no overlap
                doc_a, doc_b = random.sample(docs_with_topics, 2)
                topics_a = set(doc_a.get("topics", []))
                topics_b = set(doc_b.get("topics", []))
                overlap = len(topics_a & topics_b)

                if overlap == 0:
                    found += 1
                else:
                    continue
            else:
                # Sample a document with topics
                doc_a = random.choice(docs_with_topics)
                topics_a = set(doc_a.get("topics", []))

                # Sample a topic from doc_a
                if not topics_a:
                    continue

                # For graded mode with specific counts
                if mode == "graded" and overlap_target < 3:
                    # Try to find doc with exactly overlap_target shared topics
                    # Sample documents that share at least one topic
                    shared_topic = random.choice(list(topics_a))
                    candidates = [
                        d
                        for d in topic_to_docs.get(shared_topic, [])
                        if d["id"] != doc_a["id"]
                    ]

                    if not candidates:
                        continue

                    # Check candidates for exact overlap count
                    random.shuffle(candidates)
                    doc_b = None
                    for candidate in candidates[:50]:  # Check up to 50 candidates
                        topics_b = set(candidate.get("topics", []))
                        if len(topics_a & topics_b) == overlap_target:
                            doc_b = candidate
                            break

                    if doc_b is None:
                        continue
                    overlap = overlap_target
                else:
                    # For binary mode or 3+ overlap in graded mode
                    # Just need any overlap
                    shared_topic = random.choice(list(topics_a))
                    candidates = [
                        d
                        for d in topic_to_docs.get(shared_topic, [])
                        if d["id"] != doc_a["id"]
                    ]

                    if not candidates:
                        continue

                    doc_b = random.choice(candidates)
                    topics_b = set(doc_b.get("topics", []))
                    overlap = len(topics_a & topics_b)

                    # For 3+ target, accept any overlap >= 3
                    if mode == "graded" and overlap_target >= 3:
                        if overlap < 3:
                            continue
                    elif overlap < 1:
                        continue

                found += 1

            # Determine label based on mode
            if mode == "binary":
                label = 1 if overlap > 0 else 0
            else:  # graded
                # Cap at 3+ for graded mode
                label = min(overlap, 3)

            pairs.append(
                {
                    "id": f"pair_{len(pairs):06d}",
                    "doc_a_id": doc_a["id"],
                    "doc_a_title": doc_a.get("title", ""),
                    "doc_a_body": doc_a.get("body", ""),
                    "doc_b_id": doc_b["id"],
                    "doc_b_title": doc_b.get("title", ""),
                    "doc_b_body": doc_b.get("body", ""),
                    "label": label,
                    "overlap_count": overlap,
                    "shared_topics": sorted(
                        list(
                            set(doc_a.get("topics", [])) & set(doc_b.get("topics", []))
                        )
                    ),
                }
            )

    # Shuffle all pairs
    random.shuffle(pairs)

    # Trim to exact count
    pairs = pairs[:num_pairs]

    return pairs


def prepare_topic_pair_dataset(
    dataset: SHELFDataset,
    mode: str = "binary",
    pairs_per_split: dict[str, int] | None = None,
) -> Any:  # Returns datasets.DatasetDict
    """Prepare topic overlap pair classification dataset.

    Args:
        dataset: SHELFDataset with train/dev/test splits
        mode: "binary" (any overlap) or "graded" (count overlaps: 0, 1, 2, 3+)
        pairs_per_split: Number of pairs per split (default: train=20000, validation=4000, test=4000)

    Returns:
        HuggingFace DatasetDict with pair data

    Example:
        >>> # Binary classification: Do documents share ANY topic?
        >>> binary_pairs = prepare_topic_pair_dataset(dataset, mode="binary")
        >>> # Graded classification: How many topics do they share?
        >>> graded_pairs = prepare_topic_pair_dataset(dataset, mode="graded")
    """
    try:
        from datasets import Dataset, DatasetDict, Features, Sequence, Value
    except ImportError:
        raise ImportError(
            "The 'datasets' library is required. Install with: pip install datasets"
        )

    pairs_per_split = pairs_per_split or {
        "train": 20000,
        "validation": 4000,
        "test": 4000,
    }

    features = Features(
        {
            "id": Value("string"),
            "doc_a_id": Value("string"),
            "doc_a_title": Value("string"),
            "doc_a_body": Value("string"),
            "doc_b_id": Value("string"),
            "doc_b_title": Value("string"),
            "doc_b_body": Value("string"),
            "label": Value("int32"),
            "overlap_count": Value("int32"),
            "shared_topics": Sequence(Value("string")),
        }
    )

    splits = {}
    for split_name, docs in [
        ("train", dataset.train),
        ("validation", dataset.dev),
        ("test", dataset.test),
    ]:
        num_pairs = pairs_per_split.get(split_name, 4000)
        pairs = generate_topic_pairs(docs, num_pairs=num_pairs, mode=mode, seed=42)

        # Convert to columnar format
        columns: dict[str, list] = {k: [] for k in pairs[0].keys()}
        for pair in pairs:
            for k, v in pair.items():
                columns[k].append(v)

        splits[split_name] = Dataset.from_dict(columns, features=features)

    return DatasetDict(splits)


def save_locally(
    dataset: SHELFDataset,
    output_dir: str | Path,
    format: str = "parquet",
    include_pairs: bool = True,
) -> Path:
    """Save dataset locally in specified format.

    Args:
        dataset: SHELFDataset to save
        output_dir: Output directory
        format: Output format ('parquet', 'jsonl', or 'arrow')
        include_pairs: Whether to generate and save pair classification subsets

    Returns:
        Path to output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    hf_dataset = prepare_dataset(dataset)

    if format == "parquet":
        for split_name, split_data in hf_dataset.items():
            split_file = output_path / f"{split_name}.parquet"
            split_data.to_parquet(split_file)
            print(f"Saved {split_name} to {split_file}")

    elif format == "jsonl":
        for split_name, split_data in hf_dataset.items():
            split_file = output_path / f"{split_name}.jsonl"
            split_data.to_json(split_file, orient="records", lines=True)
            print(f"Saved {split_name} to {split_file}")

    elif format == "arrow":
        hf_dataset.save_to_disk(output_path)
        print(f"Saved dataset to {output_path}")

    else:
        raise ValueError(
            f"Unknown format: {format}. Use 'parquet', 'jsonl', or 'arrow'"
        )

    # Generate pair classification subsets
    if include_pairs:
        pairs_dir = output_path / "pairs"
        pairs_dir.mkdir(exist_ok=True)

        # Configure pair generation for each field
        pair_configs = [
            ("lcc_code", False),  # No null handling for LCC
            ("lcgft_form", False),  # No null handling for form
            ("register", False),  # No null handling for register
            ("audience", True),  # Treat null as valid class for audience
        ]

        for label_field, treat_null in pair_configs:
            pair_dataset = prepare_pair_dataset(
                dataset, label_field=label_field, treat_null_as_class=treat_null
            )
            subset_dir = (
                pairs_dir
                / f"same_{label_field.replace('_code', '').replace('_form', '')}"
            )
            subset_dir.mkdir(exist_ok=True)

            if format == "parquet":
                for split_name, split_data in pair_dataset.items():
                    split_file = subset_dir / f"{split_name}.parquet"
                    split_data.to_parquet(split_file)
                print(f"Saved {label_field} pairs to {subset_dir}")
            elif format == "jsonl":
                for split_name, split_data in pair_dataset.items():
                    split_file = subset_dir / f"{split_name}.jsonl"
                    split_data.to_json(split_file, orient="records", lines=True)
                print(f"Saved {label_field} pairs to {subset_dir}")

        # Generate topic overlap pair datasets
        topic_pair_configs = [
            ("same_topic", "binary"),  # Binary: any topic overlap
            ("topic_overlap", "graded"),  # Graded: count of overlapping topics
        ]

        for subset_name, mode in topic_pair_configs:
            topic_pair_dataset = prepare_topic_pair_dataset(dataset, mode=mode)
            subset_dir = pairs_dir / subset_name
            subset_dir.mkdir(exist_ok=True)

            if format == "parquet":
                for split_name, split_data in topic_pair_dataset.items():
                    split_file = subset_dir / f"{split_name}.parquet"
                    split_data.to_parquet(split_file)
                print(f"Saved {mode} topic pairs to {subset_dir}")
            elif format == "jsonl":
                for split_name, split_data in topic_pair_dataset.items():
                    split_file = subset_dir / f"{split_name}.jsonl"
                    split_data.to_json(split_file, orient="records", lines=True)
                print(f"Saved {mode} topic pairs to {subset_dir}")

    # Save metadata
    metadata = {
        "format": format,
        "split_config": {
            "train_ratio": dataset.split_config.train_ratio,
            "dev_ratio": dataset.split_config.dev_ratio,
            "test_ratio": dataset.split_config.test_ratio,
            "random_seed": dataset.split_config.random_seed,
            "stratify_by": dataset.split_config.stratify_by,
        },
        "counts": {
            "train": len(dataset.train),
            "validation": len(dataset.dev),
            "test": len(dataset.test),
        },
        "checksum": dataset.metadata.get("checksum", ""),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    metadata_file = output_path / "metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved metadata to {metadata_file}")

    return output_path

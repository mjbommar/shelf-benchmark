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
from datetime import datetime
from pathlib import Path
from typing import Any

from shelf.hub.splitter import SplitConfig, SplitResult, create_splits


# Feature schema for the dataset
SHELF_FEATURES = {
    # Core identifiers
    "id": "string",
    "title": "string",
    "body": "string",
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
                "created_at": datetime.utcnow().isoformat(),
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
    """
    normalized = {
        # Core fields
        "id": str(doc.get("id", "")),
        "title": str(doc.get("title", "")),
        "body": str(doc.get("body", "")),
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
        normalized.update({
            "temperature": float(doc.get("temperature", 0.0)),
            "top_p": float(doc.get("top_p", 0.0)),
            "model": str(doc.get("model", "")),
        })

    return normalized


def _create_hf_features(include_generation_metadata: bool = True) -> dict[str, Any]:
    """Create HuggingFace Features schema."""
    try:
        from datasets import Features, Sequence, Value
    except ImportError:
        raise ImportError(
            "The 'datasets' library is required. Install with: pip install datasets"
        )

    features = {
        "id": Value("string"),
        "title": Value("string"),
        "body": Value("string"),
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
        features.update({
            "temperature": Value("float32"),
            "top_p": Value("float32"),
            "model": Value("string"),
        })

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
    return DatasetDict({
        "train": make_dataset(dataset.train),
        "validation": make_dataset(dataset.dev),
        "test": make_dataset(dataset.test),
    })


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


def generate_pairs(
    documents: list[dict[str, Any]],
    label_field: str,
    num_pairs: int = 5000,
    positive_ratio: float = 0.5,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Generate document pairs for pair classification tasks.

    Args:
        documents: List of document dictionaries
        label_field: Field to use for positive/negative determination (e.g., 'lcc_code', 'lcgft_form')
        num_pairs: Total number of pairs to generate
        positive_ratio: Ratio of positive pairs (same label)
        seed: Random seed for reproducibility

    Returns:
        List of pair dictionaries with doc_a, doc_b, and label
    """
    import random
    from collections import defaultdict

    random.seed(seed)

    # Group documents by label
    label_to_docs: dict[str, list[dict]] = defaultdict(list)
    for doc in documents:
        label = doc.get(label_field)
        if label:
            label_to_docs[label].append(doc)

    labels_with_multiple = [l for l, docs in label_to_docs.items() if len(docs) >= 2]

    pairs = []
    num_positive = int(num_pairs * positive_ratio)
    num_negative = num_pairs - num_positive

    # Generate positive pairs (same label)
    for _ in range(num_positive):
        label = random.choice(labels_with_multiple)
        doc_a, doc_b = random.sample(label_to_docs[label], 2)
        pairs.append({
            "id": f"pair_{len(pairs):06d}",
            "doc_a_id": doc_a["id"],
            "doc_a_title": doc_a.get("title", ""),
            "doc_a_body": doc_a.get("body", ""),
            "doc_b_id": doc_b["id"],
            "doc_b_title": doc_b.get("title", ""),
            "doc_b_body": doc_b.get("body", ""),
            "label": 1,
            "label_field": label_field,
        })

    # Generate negative pairs (different labels)
    all_labels = list(label_to_docs.keys())
    for _ in range(num_negative):
        label_a, label_b = random.sample(all_labels, 2)
        doc_a = random.choice(label_to_docs[label_a])
        doc_b = random.choice(label_to_docs[label_b])
        pairs.append({
            "id": f"pair_{len(pairs):06d}",
            "doc_a_id": doc_a["id"],
            "doc_a_title": doc_a.get("title", ""),
            "doc_a_body": doc_a.get("body", ""),
            "doc_b_id": doc_b["id"],
            "doc_b_title": doc_b.get("title", ""),
            "doc_b_body": doc_b.get("body", ""),
            "label": 0,
            "label_field": label_field,
        })

    random.shuffle(pairs)
    return pairs


def prepare_pair_dataset(
    dataset: SHELFDataset,
    label_field: str = "lcc_code",
    pairs_per_split: dict[str, int] | None = None,
) -> Any:  # Returns datasets.DatasetDict
    """Prepare pair classification dataset.

    Args:
        dataset: SHELFDataset with train/dev/test splits
        label_field: Field to use for pair labels ('lcc_code' or 'lcgft_form')
        pairs_per_split: Number of pairs per split (default: train=10000, validation=2000, test=2000)

    Returns:
        HuggingFace DatasetDict with pair data
    """
    try:
        from datasets import Dataset, DatasetDict, Features, Value
    except ImportError:
        raise ImportError(
            "The 'datasets' library is required. Install with: pip install datasets"
        )

    pairs_per_split = pairs_per_split or {"train": 10000, "validation": 2000, "test": 2000}

    features = Features({
        "id": Value("string"),
        "doc_a_id": Value("string"),
        "doc_a_title": Value("string"),
        "doc_a_body": Value("string"),
        "doc_b_id": Value("string"),
        "doc_b_title": Value("string"),
        "doc_b_body": Value("string"),
        "label": Value("int32"),
        "label_field": Value("string"),
    })

    splits = {}
    for split_name, docs in [("train", dataset.train), ("validation", dataset.dev), ("test", dataset.test)]:
        num_pairs = pairs_per_split.get(split_name, 2000)
        pairs = generate_pairs(docs, label_field, num_pairs=num_pairs, seed=42)

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
        raise ValueError(f"Unknown format: {format}. Use 'parquet', 'jsonl', or 'arrow'")

    # Generate pair classification subsets
    if include_pairs:
        pairs_dir = output_path / "pairs"
        pairs_dir.mkdir(exist_ok=True)

        for label_field in ["lcc_code", "lcgft_form"]:
            pair_dataset = prepare_pair_dataset(dataset, label_field=label_field)
            subset_dir = pairs_dir / f"same_{label_field.replace('_code', '').replace('_form', '')}"
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
        "created_at": datetime.utcnow().isoformat(),
    }

    metadata_file = output_path / "metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved metadata to {metadata_file}")

    return output_path

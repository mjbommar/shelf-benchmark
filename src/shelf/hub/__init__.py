"""
SHELF HuggingFace Hub Integration

This module provides functionality for:
- Stratified train/dev/test splitting of the corpus
- Dataset preparation and upload to HuggingFace Hub
- Dataset card generation with proper YAML metadata
"""

from shelf.hub.splitter import (
    SplitConfig,
    SplitResult,
    StratifiedSplitter,
    create_splits,
)
from shelf.hub.dataset import (
    DatasetConfig,
    SHELFDataset,
    generate_pairs,
    prepare_dataset,
    prepare_pair_dataset,
    push_folder_to_hub,
    push_to_hub,
    save_locally,
)
from shelf.hub.card import (
    CardConfig,
    DatasetCardGenerator,
    generate_dataset_card,
)

__all__ = [
    # Splitter
    "SplitConfig",
    "SplitResult",
    "StratifiedSplitter",
    "create_splits",
    # Dataset
    "DatasetConfig",
    "SHELFDataset",
    "generate_pairs",
    "prepare_dataset",
    "prepare_pair_dataset",
    "push_folder_to_hub",
    "push_to_hub",
    "save_locally",
    # Card
    "CardConfig",
    "DatasetCardGenerator",
    "generate_dataset_card",
]

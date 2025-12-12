#!/usr/bin/env python3
"""
Prepare and Upload SHELF Dataset to HuggingFace Hub

This script handles the complete pipeline for creating and uploading
the SHELF dataset to HuggingFace Hub:

1. Load documents from artifacts directory
2. Create stratified train/dev/test splits
3. Generate dataset card (README.md)
4. Optionally upload to HuggingFace Hub

Usage:
    # Prepare locally (default)
    python scripts/prepare_hf_dataset.py

    # Custom split ratios
    python scripts/prepare_hf_dataset.py --train-ratio 0.7 --dev-ratio 0.15 --test-ratio 0.15

    # Upload to HuggingFace Hub
    python scripts/prepare_hf_dataset.py --upload --repo-id myorg/shelf

    # Private repository
    python scripts/prepare_hf_dataset.py --upload --repo-id myorg/shelf --private
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from shelf.hub import (
    SplitConfig,
    SHELFDataset,
    DatasetConfig,
    CardConfig,
    push_to_hub,
    save_locally,
    generate_dataset_card,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare and upload SHELF dataset to HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input/output paths
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("data/artifacts"),
        help="Directory containing artifact JSON files (default: data/artifacts)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/hf_dataset"),
        help="Output directory for local dataset files (default: data/hf_dataset)",
    )

    # Split configuration
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.6,
        help="Proportion of data for training (default: 0.6)",
    )
    parser.add_argument(
        "--dev-ratio",
        type=float,
        default=0.2,
        help="Proportion of data for dev/validation (default: 0.2)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Proportion of data for testing (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--stratify-by",
        nargs="+",
        default=["lcc_code", "lcgft_category"],
        help="Fields to stratify by (default: lcc_code lcgft_category)",
    )

    # Output format
    parser.add_argument(
        "--format",
        choices=["parquet", "jsonl", "arrow"],
        default="parquet",
        help="Local output format (default: parquet)",
    )

    # HuggingFace Hub upload
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload dataset to HuggingFace Hub",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="mjbommar/SHELF",
        help="HuggingFace repository ID (default: mjbommar/SHELF)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create private repository on HuggingFace Hub",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace API token (or use HF_TOKEN env var)",
    )

    # Dataset card configuration
    parser.add_argument(
        "--version",
        type=str,
        default="0.2.0",
        help="Dataset version (default: 0.2.0)",
    )
    parser.add_argument(
        "--no-generation-metadata",
        action="store_true",
        help="Exclude generation metadata (model, temperature, etc.)",
    )

    # Verbosity
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Validate paths
    if not args.artifacts_dir.exists():
        print(f"Error: Artifacts directory not found: {args.artifacts_dir}")
        return 1

    # Validate split ratios
    total_ratio = args.train_ratio + args.dev_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        print(f"Error: Split ratios must sum to 1.0, got {total_ratio:.3f}")
        return 1

    if not args.quiet:
        print("=" * 60)
        print("SHELF Dataset Preparation")
        print("=" * 60)
        print(f"Artifacts directory: {args.artifacts_dir}")
        print(f"Output directory: {args.output_dir}")
        print(
            f"Split ratios: train={args.train_ratio}, dev={args.dev_ratio}, test={args.test_ratio}"
        )
        print(f"Random seed: {args.seed}")
        print(f"Stratify by: {args.stratify_by}")
        print(f"Output format: {args.format}")
        print()

    # Create configurations
    split_config = SplitConfig(
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed,
        stratify_by=args.stratify_by,
    )

    dataset_config = DatasetConfig(
        repo_id=args.repo_id,
        private=args.private,
        token=args.token,
        include_generation_metadata=not args.no_generation_metadata,
    )

    card_config = CardConfig(
        version=args.version,
    )

    # Load and split dataset
    if not args.quiet:
        print("Loading documents and creating splits...")

    try:
        dataset = SHELFDataset.from_artifacts(
            args.artifacts_dir,
            split_config=split_config,
            dataset_config=dataset_config,
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return 1

    if not args.quiet:
        print(f"  Total documents: {dataset.total_documents:,}")
        print(f"  Train: {len(dataset.train):,}")
        print(f"  Dev: {len(dataset.dev):,}")
        print(f"  Test: {len(dataset.test):,}")
        print(f"  Split checksum: {dataset.metadata.get('checksum', 'N/A')}")
        print()

    # Save locally
    if not args.quiet:
        print(f"Saving dataset locally ({args.format} format)...")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        save_locally(dataset, args.output_dir, format=args.format)
    except Exception as e:
        print(f"Error saving dataset: {e}")
        return 1

    # Generate dataset card
    if not args.quiet:
        print("Generating dataset card...")

    readme_path = args.output_dir / "README.md"
    try:
        generate_dataset_card(
            dataset,
            output_path=readme_path,
            repo_id=args.repo_id,
            config=card_config,
        )
    except Exception as e:
        print(f"Error generating dataset card: {e}")
        return 1

    # Upload to HuggingFace Hub if requested
    if args.upload:
        if not args.quiet:
            print()
            print(f"Uploading to HuggingFace Hub: {args.repo_id}")
            print(f"  Private: {args.private}")

        try:
            url = push_to_hub(
                dataset,
                repo_id=args.repo_id,
                private=args.private,
                token=args.token,
            )
            print(f"  URL: {url}")
        except Exception as e:
            print(f"Error uploading to HuggingFace Hub: {e}")
            return 1

    if not args.quiet:
        print()
        print("=" * 60)
        print("Dataset preparation complete!")
        print("=" * 60)
        print(f"Local files: {args.output_dir}")
        if args.upload:
            print(f"HuggingFace Hub: https://huggingface.co/datasets/{args.repo_id}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

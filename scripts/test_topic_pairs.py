#!/usr/bin/env python3
"""
Test script for topic pair generation.

This script validates that the topic pair generation functions work correctly
and produce balanced, well-distributed pairs.
"""

import json
from collections import Counter
from pathlib import Path

from shelf.hub.dataset import generate_topic_pairs


def load_sample_documents(artifacts_dir: str | Path, limit: int = 1000) -> list[dict]:
    """Load a sample of documents for testing."""
    artifacts_path = Path(artifacts_dir)
    documents = []

    for json_file in sorted(artifacts_path.glob("*.json"))[:limit]:
        with open(json_file, "r", encoding="utf-8") as f:
            doc = json.load(f)
            documents.append(doc)

    return documents


def analyze_pairs(pairs: list[dict], mode: str) -> dict:
    """Analyze generated pairs to validate distribution."""
    stats = {
        "total_pairs": len(pairs),
        "label_distribution": Counter(),
        "overlap_distribution": Counter(),
        "has_shared_topics": 0,
        "sample_pairs": [],
    }

    for pair in pairs:
        stats["label_distribution"][pair["label"]] += 1
        stats["overlap_distribution"][pair["overlap_count"]] += 1
        if pair["shared_topics"]:
            stats["has_shared_topics"] += 1

    # Sample a few pairs for inspection
    stats["sample_pairs"] = pairs[:5]

    return stats


def main():
    print("=== Topic Pair Generation Test ===\n")

    # Load sample documents
    artifacts_dir = Path("/home/mjbommar/src/shelf-benchmark/data/artifacts")
    print(f"Loading documents from {artifacts_dir}...")
    documents = load_sample_documents(artifacts_dir, limit=1000)
    print(f"Loaded {len(documents)} documents\n")

    # Analyze topic distribution
    topic_counts = Counter()
    docs_per_topic_count = Counter()
    for doc in documents:
        topics = doc.get("topics", [])
        docs_per_topic_count[len(topics)] += 1
        for topic in topics:
            topic_counts[topic] += 1

    print("Topic statistics:")
    print("  Documents per topic count:")
    for count, num_docs in sorted(docs_per_topic_count.items()):
        pct = (num_docs / len(documents)) * 100
        print(f"    {count} topics: {num_docs:4d} docs ({pct:5.2f}%)")
    print(f"  Unique topics: {len(topic_counts)}")
    print(f"  Most common: {topic_counts.most_common(5)}")
    print()

    # Test binary mode
    print("--- Binary Mode (Same-Topic) ---")
    binary_pairs = generate_topic_pairs(
        documents, num_pairs=100, mode="binary", seed=42
    )
    binary_stats = analyze_pairs(binary_pairs, "binary")

    print(f"Generated {binary_stats['total_pairs']} pairs")
    print("Label distribution:")
    for label, count in sorted(binary_stats["label_distribution"].items()):
        pct = (count / binary_stats["total_pairs"]) * 100
        label_name = "No overlap" if label == 0 else "Has overlap"
        print(f"  {label} ({label_name}): {count:3d} ({pct:5.2f}%)")

    print("\nOverlap count distribution:")
    for overlap, count in sorted(binary_stats["overlap_distribution"].items()):
        pct = (count / binary_stats["total_pairs"]) * 100
        print(f"  {overlap} topics: {count:3d} ({pct:5.2f}%)")

    print("\nSample pairs:")
    for i, pair in enumerate(binary_stats["sample_pairs"][:3], 1):
        print(f"  {i}. Label={pair['label']}, Overlap={pair['overlap_count']}")
        print(f"     Shared topics: {pair['shared_topics']}")
        print(
            f"     Doc A: {pair['doc_a_title'][:50]}..."
            if len(pair["doc_a_title"]) > 50
            else f"     Doc A: {pair['doc_a_title']}"
        )
        print(
            f"     Doc B: {pair['doc_b_title'][:50]}..."
            if len(pair["doc_b_title"]) > 50
            else f"     Doc B: {pair['doc_b_title']}"
        )

    print("\n" + "=" * 60 + "\n")

    # Test graded mode
    print("--- Graded Mode (Topic Overlap Count) ---")
    graded_pairs = generate_topic_pairs(
        documents, num_pairs=100, mode="graded", seed=42
    )
    graded_stats = analyze_pairs(graded_pairs, "graded")

    print(f"Generated {graded_stats['total_pairs']} pairs")
    print("Label distribution:")
    for label, count in sorted(graded_stats["label_distribution"].items()):
        pct = (count / graded_stats["total_pairs"]) * 100
        label_name = f"{label} shared topics" if label < 3 else "3+ shared topics"
        print(f"  {label} ({label_name}): {count:3d} ({pct:5.2f}%)")

    print("\nOverlap count distribution:")
    for overlap, count in sorted(graded_stats["overlap_distribution"].items()):
        pct = (count / graded_stats["total_pairs"]) * 100
        print(f"  {overlap} topics: {count:3d} ({pct:5.2f}%)")

    print("\nSample pairs:")
    for i, pair in enumerate(graded_stats["sample_pairs"][:3], 1):
        print(f"  {i}. Label={pair['label']}, Overlap={pair['overlap_count']}")
        print(f"     Shared topics: {pair['shared_topics']}")
        print(
            f"     Doc A: {pair['doc_a_title'][:50]}..."
            if len(pair["doc_a_title"]) > 50
            else f"     Doc A: {pair['doc_a_title']}"
        )
        print(
            f"     Doc B: {pair['doc_b_title'][:50]}..."
            if len(pair["doc_b_title"]) > 50
            else f"     Doc B: {pair['doc_b_title']}"
        )

    print("\n" + "=" * 60 + "\n")

    # Validation checks
    print("=== Validation Checks ===")
    checks_passed = 0
    checks_total = 0

    # Check 1: Binary mode has ~50/50 split
    checks_total += 1
    binary_ratio = binary_stats["label_distribution"][1] / binary_stats["total_pairs"]
    if 0.4 <= binary_ratio <= 0.6:
        print(f"✓ Binary mode balance: {binary_ratio:.2%} positive (target: ~50%)")
        checks_passed += 1
    else:
        print(f"✗ Binary mode imbalance: {binary_ratio:.2%} positive (target: ~50%)")

    # Check 2: Graded mode has expected distribution
    checks_total += 1
    graded_targets = {0: 0.4, 1: 0.3, 2: 0.2, 3: 0.1}
    graded_ok = True
    for label, target in graded_targets.items():
        actual = graded_stats["label_distribution"][label] / graded_stats["total_pairs"]
        if not (target - 0.15 <= actual <= target + 0.15):  # Allow 15% deviation
            graded_ok = False
            print(f"✗ Graded label {label}: {actual:.2%} (target: {target:.0%})")

    if graded_ok:
        print("✓ Graded mode distribution matches targets")
        checks_passed += 1

    # Check 3: All pairs have valid IDs
    checks_total += 1
    all_ids = [p["id"] for p in binary_pairs + graded_pairs]
    unique_ids = set(all_ids)
    if len(all_ids) == len(unique_ids):
        print(f"✓ All pair IDs are unique ({len(unique_ids)} pairs)")
        checks_passed += 1
    else:
        print(f"✗ Duplicate pair IDs found ({len(all_ids)} vs {len(unique_ids)})")

    # Check 4: Shared topics match overlap count
    checks_total += 1
    mismatch = False
    for pair in binary_pairs + graded_pairs:
        expected_overlap = len(pair["shared_topics"])
        if pair["overlap_count"] != expected_overlap:
            mismatch = True
            print(
                f"✗ Mismatch: pair {pair['id']} has overlap_count={pair['overlap_count']} "
                f"but {expected_overlap} shared_topics"
            )
            break

    if not mismatch:
        print("✓ All pairs have consistent overlap_count and shared_topics")
        checks_passed += 1

    print(f"\n=== Summary: {checks_passed}/{checks_total} checks passed ===")

    if checks_passed == checks_total:
        print("✓ All validation checks passed!")
        return 0
    else:
        print(f"✗ {checks_total - checks_passed} checks failed")
        return 1


if __name__ == "__main__":
    exit(main())

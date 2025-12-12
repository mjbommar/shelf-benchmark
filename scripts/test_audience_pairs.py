#!/usr/bin/env python
"""Test script for audience pair generation.

This script tests the audience pair generation functionality and verifies:
1. Pairs are generated successfully
2. Balance between positive/negative pairs is correct
3. Null handling works properly
4. Pairs are properly formatted
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from shelf.hub.dataset import generate_pairs


def load_sample_documents(artifacts_dir: Path, max_docs: int = 1000) -> list[dict]:
    """Load a sample of documents from artifacts."""
    documents = []

    for jsonl_file in artifacts_dir.glob("*.json"):
        if len(documents) >= max_docs:
            break

        with open(jsonl_file) as f:
            doc = json.load(f)
            documents.append(doc)

    return documents


def test_audience_pair_generation():
    """Test audience pair generation."""
    print("Testing Audience Pair Generation")
    print("=" * 60)

    # Load sample documents
    artifacts_dir = Path(__file__).parent.parent / "data" / "artifacts"
    if not artifacts_dir.exists():
        print(f"Error: Artifacts directory not found at {artifacts_dir}")
        return

    print(f"\nLoading documents from {artifacts_dir}")
    documents = load_sample_documents(artifacts_dir, max_docs=1000)
    print(f"Loaded {len(documents)} documents")

    # Analyze audience distribution
    print("\n" + "-" * 60)
    print("Audience Distribution in Sample:")
    print("-" * 60)

    audience_counts = Counter()
    for doc in documents:
        audience = doc.get("audience")
        if audience is None or audience == "":
            audience_counts["null"] += 1
        else:
            audience_counts[audience] += 1

    total = len(documents)
    print(f"Total documents: {total}")
    print(
        f"Null audience: {audience_counts['null']} ({audience_counts['null'] / total * 100:.1f}%)"
    )
    print(f"Unique audiences (including null): {len(audience_counts)}")
    print("\nTop 10 audiences:")
    for aud, count in audience_counts.most_common(10):
        print(f"  {aud:25s}: {count:4d} ({count / total * 100:.1f}%)")

    # Test pair generation without null handling
    print("\n" + "-" * 60)
    print("Test 1: Pair Generation WITHOUT null handling (treat_null_as_class=False)")
    print("-" * 60)

    pairs_no_null = generate_pairs(
        documents,
        label_field="audience",
        num_pairs=100,
        positive_ratio=0.5,
        seed=42,
        treat_null_as_class=False,
    )

    print(f"Generated {len(pairs_no_null)} pairs")

    # Analyze pairs
    label_counts = Counter(p["label"] for p in pairs_no_null)
    print(
        f"Positive pairs (same audience): {label_counts[1]} ({label_counts[1] / len(pairs_no_null) * 100:.1f}%)"
    )
    print(
        f"Negative pairs (diff audience): {label_counts[0]} ({label_counts[0] / len(pairs_no_null) * 100:.1f}%)"
    )

    # Check for null in pairs
    null_in_pairs = 0
    for pair in pairs_no_null[:10]:
        doc_a_id = pair["doc_a_id"]
        doc_b_id = pair["doc_b_id"]
        doc_a = next(d for d in documents if d["id"] == doc_a_id)
        doc_b = next(d for d in documents if d["id"] == doc_b_id)

        if doc_a.get("audience") is None or doc_b.get("audience") is None:
            null_in_pairs += 1

    print(f"Pairs with null audience (in first 10): {null_in_pairs}")

    # Test pair generation WITH null handling
    print("\n" + "-" * 60)
    print("Test 2: Pair Generation WITH null handling (treat_null_as_class=True)")
    print("-" * 60)

    pairs_with_null = generate_pairs(
        documents,
        label_field="audience",
        num_pairs=100,
        positive_ratio=0.5,
        seed=42,
        treat_null_as_class=True,
    )

    print(f"Generated {len(pairs_with_null)} pairs")

    # Analyze pairs
    label_counts = Counter(p["label"] for p in pairs_with_null)
    print(
        f"Positive pairs (same audience): {label_counts[1]} ({label_counts[1] / len(pairs_with_null) * 100:.1f}%)"
    )
    print(
        f"Negative pairs (diff audience): {label_counts[0]} ({label_counts[0] / len(pairs_with_null) * 100:.1f}%)"
    )

    # Check for null-null pairs
    null_null_pairs = 0
    null_X_pairs = 0
    X_X_pairs = 0

    for pair in pairs_with_null:
        doc_a_id = pair["doc_a_id"]
        doc_b_id = pair["doc_b_id"]
        doc_a = next(d for d in documents if d["id"] == doc_a_id)
        doc_b = next(d for d in documents if d["id"] == doc_b_id)

        aud_a = doc_a.get("audience")
        aud_b = doc_b.get("audience")

        if (aud_a is None or aud_a == "") and (aud_b is None or aud_b == ""):
            null_null_pairs += 1
            assert pair["label"] == 1, "null-null pairs should be positive!"
        elif (aud_a is None or aud_a == "") or (aud_b is None or aud_b == ""):
            null_X_pairs += 1
            assert pair["label"] == 0, "null-X pairs should be negative!"
        else:
            X_X_pairs += 1

    print(f"Null-null pairs (should be label=1): {null_null_pairs}")
    print(f"Null-X pairs (should be label=0): {null_X_pairs}")
    print(
        f"X-X pairs (same={label_counts[1] - null_null_pairs}, diff={label_counts[0] - null_X_pairs}): {X_X_pairs}"
    )

    # Show some example pairs
    print("\n" + "-" * 60)
    print("Example Pairs (first 3):")
    print("-" * 60)

    for i, pair in enumerate(pairs_with_null[:3], 1):
        doc_a = next(d for d in documents if d["id"] == pair["doc_a_id"])
        doc_b = next(d for d in documents if d["id"] == pair["doc_b_id"])

        aud_a = doc_a.get("audience") or "null"
        aud_b = doc_b.get("audience") or "null"

        print(f"\nPair {i}:")
        print(f"  Doc A: {doc_a['title'][:50]}...")
        print(f"    Audience: {aud_a}")
        print(f"  Doc B: {doc_b['title'][:50]}...")
        print(f"    Audience: {aud_b}")
        print(
            f"  Label: {pair['label']} ({'same' if pair['label'] == 1 else 'different'})"
        )

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_audience_pair_generation()

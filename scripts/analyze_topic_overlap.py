#!/usr/bin/env python3
"""
Analyze topic overlap patterns in SHELF documents.

This script examines:
- Distribution of topics per document
- Frequency of individual topics
- Topic overlap patterns between document pairs
"""

import json
from pathlib import Path
from collections import Counter, defaultdict


def main():
    artifacts_dir = Path("/home/mjbommar/src/shelf-benchmark/data/artifacts")
    documents = []

    # Load all documents
    for json_file in sorted(artifacts_dir.glob("*.json")):
        with open(json_file, "r", encoding="utf-8") as f:
            doc = json.load(f)
            documents.append(doc)

    print(f"Total documents: {len(documents)}")

    # Analyze topics per document
    topic_counts = Counter()
    for doc in documents:
        topics = doc.get("topics", [])
        topic_counts[len(topics)] += 1

    print("\nTopics per document distribution:")
    for count, num_docs in sorted(topic_counts.items()):
        pct = (num_docs / len(documents)) * 100
        print(f"  {count} topics: {num_docs:5d} docs ({pct:5.2f}%)")

    # Count unique topics
    all_topics = set()
    topic_freq = Counter()
    for doc in documents:
        topics = doc.get("topics", [])
        all_topics.update(topics)
        for topic in topics:
            topic_freq[topic] += 1

    print(f"\nUnique topics: {len(all_topics)}")
    print("\nMost common topics:")
    for topic, count in topic_freq.most_common(15):
        print(f"  {topic}: {count}")

    print("\nLeast common topics:")
    for topic, count in sorted(topic_freq.items(), key=lambda x: x[1])[:15]:
        print(f"  {topic}: {count}")

    # Analyze topic overlap potential - sample
    print("\nAnalyzing topic overlap (sampling 100 pairs per document)...")
    overlap_counts = defaultdict(int)
    for i, doc_a in enumerate(documents[:1000]):  # Sample first 1000 docs
        topics_a = set(doc_a.get("topics", []))
        for doc_b in documents[i + 1 : min(i + 101, len(documents))]:
            topics_b = set(doc_b.get("topics", []))
            overlap = len(topics_a & topics_b)
            overlap_counts[overlap] += 1

    print("\nSample overlap distribution:")
    total = sum(overlap_counts.values())
    for overlap in sorted(overlap_counts.keys()):
        count = overlap_counts[overlap]
        pct = (count / total) * 100
        print(f"  {overlap} shared topics: {count:6d} pairs ({pct:5.2f}%)")

    # Calculate expected overlap for balanced sampling
    print("\n\nRecommended pair distribution for balanced dataset:")
    print("  Binary task (same_topic):")
    print("    - 50% with 0 shared topics (label=0)")
    print("    - 50% with 1+ shared topics (label=1)")
    print("\n  Graded task (topic_overlap):")
    print("    - 40% with 0 shared topics (label=0)")
    print("    - 30% with 1 shared topic (label=1)")
    print("    - 20% with 2 shared topics (label=2)")
    print("    - 10% with 3+ shared topics (label=3)")


if __name__ == "__main__":
    main()

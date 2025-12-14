"""
Document Length Analysis for SHELF Benchmark

This script analyzes document length distribution and its effects on model performance.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


def compute_length_stats(split_data, tokenizer):
    """Compute length statistics for a dataset split."""
    word_counts = []
    token_counts = []
    char_counts = []
    lcc_codes = []
    forms = []
    doc_ids = []

    for item in split_data:
        text = item["text"]
        # Word count (simple whitespace split)
        words = len(text.split())
        word_counts.append(words)

        # Character count
        char_counts.append(len(text))

        # Token count (using BERT tokenizer, truncation disabled for counting)
        tokens = len(tokenizer.encode(text, add_special_tokens=False, truncation=False))
        token_counts.append(tokens)

        # Metadata
        lcc_codes.append(item["lcc_code"])
        forms.append(item["lcgft_form"])
        doc_ids.append(item["id"])

    return {
        "word_counts": word_counts,
        "token_counts": token_counts,
        "char_counts": char_counts,
        "lcc_codes": lcc_codes,
        "forms": forms,
        "doc_ids": doc_ids,
    }


def print_statistics(counts, name="Token"):
    """Print detailed statistics for a count array."""
    print(f"\n{name} Count Statistics:")
    print(f"  Mean: {np.mean(counts):.1f}")
    print(f"  Median: {np.median(counts):.1f}")
    print(f"  Min: {np.min(counts)}")
    print(f"  Max: {np.max(counts)}")
    print(f"  Std Dev: {np.std(counts):.1f}")
    print(f"  Percentiles:")
    print(f"    25th: {np.percentile(counts, 25):.1f}")
    print(f"    50th: {np.percentile(counts, 50):.1f}")
    print(f"    75th: {np.percentile(counts, 75):.1f}")
    print(f"    90th: {np.percentile(counts, 90):.1f}")
    print(f"    95th: {np.percentile(counts, 95):.1f}")
    print(f"    99th: {np.percentile(counts, 99):.1f}")


def analyze_truncation(token_counts):
    """Analyze truncation effects at common embedding model limits."""
    total = len(token_counts)
    exceeds_512 = sum(1 for t in token_counts if t > 512)
    exceeds_1024 = sum(1 for t in token_counts if t > 1024)
    exceeds_2048 = sum(1 for t in token_counts if t > 2048)

    print(f"\nTruncation Analysis:")
    print(f"  Documents > 512 tokens: {exceeds_512} ({100*exceeds_512/total:.1f}%)")
    print(f"  Documents > 1024 tokens: {exceeds_1024} ({100*exceeds_1024/total:.1f}%)")
    print(f"  Documents > 2048 tokens: {exceeds_2048} ({100*exceeds_2048/total:.1f}%)")

    # Analyze information loss at 512 token truncation
    tokens_lost = []
    for t in token_counts:
        if t > 512:
            loss_pct = 100 * (t - 512) / t
            tokens_lost.append(loss_pct)

    if tokens_lost:
        print(f"\nInformation Loss at 512-token Truncation:")
        print(f"  Mean loss: {np.mean(tokens_lost):.1f}%")
        print(f"  Median loss: {np.median(tokens_lost):.1f}%")
        print(f"  Max loss: {np.max(tokens_lost):.1f}%")


def stratify_by_length(token_counts, lcc_codes, forms):
    """Stratify documents by length and analyze distribution."""
    # Define length buckets
    short = []  # <= 512 tokens
    medium = []  # 512-1024 tokens
    long = []  # > 1024 tokens

    short_lcc = []
    medium_lcc = []
    long_lcc = []

    short_forms = []
    medium_forms = []
    long_forms = []

    for i, t in enumerate(token_counts):
        if t <= 512:
            short.append(t)
            short_lcc.append(lcc_codes[i])
            short_forms.append(forms[i])
        elif t <= 1024:
            medium.append(t)
            medium_lcc.append(lcc_codes[i])
            medium_forms.append(forms[i])
        else:
            long.append(t)
            long_lcc.append(lcc_codes[i])
            long_forms.append(forms[i])

    print(f"\nLength Stratification:")
    print(f"  Short (<=512 tokens): {len(short)} docs ({100*len(short)/len(token_counts):.1f}%)")
    print(f"  Medium (512-1024 tokens): {len(medium)} docs ({100*len(medium)/len(token_counts):.1f}%)")
    print(f"  Long (>1024 tokens): {len(long)} docs ({100*len(long)/len(token_counts):.1f}%)")

    # Check LCC class distribution across length buckets
    print(f"\nUnique LCC classes per stratum:")
    print(f"  Short: {len(set(short_lcc))}")
    print(f"  Medium: {len(set(medium_lcc))}")
    print(f"  Long: {len(set(long_lcc))}")

    # Check form distribution
    print(f"\nUnique forms per stratum:")
    print(f"  Short: {len(set(short_forms))}")
    print(f"  Medium: {len(set(medium_forms))}")
    print(f"  Long: {len(set(long_forms))}")

    return {
        "short": {"tokens": short, "lcc": short_lcc, "forms": short_forms},
        "medium": {"tokens": medium, "lcc": medium_lcc, "forms": medium_forms},
        "long": {"tokens": long, "lcc": long_lcc, "forms": long_forms},
    }


def plot_length_distribution(token_counts, output_path):
    """Plot token length distribution."""
    plt.figure(figsize=(12, 6))

    # Histogram
    plt.subplot(1, 2, 1)
    plt.hist(token_counts, bins=50, edgecolor="black", alpha=0.7)
    plt.axvline(512, color="red", linestyle="--", label="512 tokens (BERT limit)")
    plt.axvline(1024, color="orange", linestyle="--", label="1024 tokens")
    plt.axvline(np.median(token_counts), color="green", linestyle="-", label=f"Median ({np.median(token_counts):.0f})")
    plt.xlabel("Token Count")
    plt.ylabel("Frequency")
    plt.title("SHELF Document Length Distribution")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    # Cumulative distribution
    plt.subplot(1, 2, 2)
    sorted_tokens = np.sort(token_counts)
    cumulative = np.arange(1, len(sorted_tokens) + 1) / len(sorted_tokens) * 100
    plt.plot(sorted_tokens, cumulative)
    plt.axvline(512, color="red", linestyle="--", label="512 tokens")
    plt.axvline(1024, color="orange", linestyle="--", label="1024 tokens")
    plt.xlabel("Token Count")
    plt.ylabel("Cumulative Percentage")
    plt.title("Cumulative Distribution of Document Lengths")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")


def analyze_length_by_metadata(token_counts, lcc_codes, forms):
    """Analyze if certain LCC codes or forms have systematically different lengths."""
    from collections import defaultdict

    lcc_lengths = defaultdict(list)
    form_lengths = defaultdict(list)

    for i, t in enumerate(token_counts):
        lcc_lengths[lcc_codes[i]].append(t)
        form_lengths[forms[i]].append(t)

    # Find LCC classes with highest/lowest average length
    lcc_avg = {lcc: np.mean(lengths) for lcc, lengths in lcc_lengths.items()}
    sorted_lcc = sorted(lcc_avg.items(), key=lambda x: x[1], reverse=True)

    print(f"\nLCC Codes by Average Length:")
    print(f"  Longest 5:")
    for lcc, avg_len in sorted_lcc[:5]:
        print(f"    {lcc}: {avg_len:.1f} tokens (n={len(lcc_lengths[lcc])})")

    print(f"  Shortest 5:")
    for lcc, avg_len in sorted_lcc[-5:]:
        print(f"    {lcc}: {avg_len:.1f} tokens (n={len(lcc_lengths[lcc])})")

    # Find forms with highest/lowest average length
    form_avg = {form: np.mean(lengths) for form, lengths in form_lengths.items() if len(lengths) >= 10}
    sorted_forms = sorted(form_avg.items(), key=lambda x: x[1], reverse=True)

    print(f"\nForms by Average Length (forms with >=10 docs):")
    print(f"  Longest 5:")
    for form, avg_len in sorted_forms[:5]:
        print(f"    {form}: {avg_len:.1f} tokens (n={len(form_lengths[form])})")

    print(f"  Shortest 5:")
    for form, avg_len in sorted_forms[-5:]:
        print(f"    {form}: {avg_len:.1f} tokens (n={len(form_lengths[form])})")


def main():
    """Main analysis function."""
    print("=" * 80)
    print("SHELF Document Length Analysis")
    print("=" * 80)

    # Load dataset
    print("\nLoading SHELF dataset...")
    ds = load_dataset("mjbommar/SHELF")

    # Initialize tokenizer (using BERT tokenizer as reference for embedding models)
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    output_dir = Path(__file__).parent

    # Analyze each split
    all_token_counts = []
    all_word_counts = []
    all_lcc_codes = []
    all_forms = []

    for split_name in ["train", "validation", "test"]:
        print(f"\n{'=' * 80}")
        print(f"{split_name.upper()} SPLIT")
        print(f"{'=' * 80}")
        print(f"Total documents: {len(ds[split_name])}")

        stats = compute_length_stats(ds[split_name], tokenizer)

        print_statistics(stats["word_counts"], "Word")
        print_statistics(stats["token_counts"], "Token")
        analyze_truncation(stats["token_counts"])

        # Accumulate for overall analysis
        all_token_counts.extend(stats["token_counts"])
        all_word_counts.extend(stats["word_counts"])
        all_lcc_codes.extend(stats["lcc_codes"])
        all_forms.extend(stats["forms"])

    # Combined analysis
    print(f"\n{'=' * 80}")
    print("COMBINED DATASET")
    print(f"{'=' * 80}")
    print(f"Total documents: {len(all_token_counts)}")

    print_statistics(all_word_counts, "Word")
    print_statistics(all_token_counts, "Token")
    analyze_truncation(all_token_counts)

    # Length stratification
    strata = stratify_by_length(all_token_counts, all_lcc_codes, all_forms)

    # Metadata analysis
    analyze_length_by_metadata(all_token_counts, all_lcc_codes, all_forms)

    # Plot distribution
    plot_path = output_dir / "length_distribution.png"
    plot_length_distribution(all_token_counts, plot_path)

    # Save summary statistics to JSON
    summary = {
        "total_documents": len(all_token_counts),
        "token_stats": {
            "mean": float(np.mean(all_token_counts)),
            "median": float(np.median(all_token_counts)),
            "min": int(np.min(all_token_counts)),
            "max": int(np.max(all_token_counts)),
            "std": float(np.std(all_token_counts)),
            "percentiles": {
                "25": float(np.percentile(all_token_counts, 25)),
                "50": float(np.percentile(all_token_counts, 50)),
                "75": float(np.percentile(all_token_counts, 75)),
                "90": float(np.percentile(all_token_counts, 90)),
                "95": float(np.percentile(all_token_counts, 95)),
                "99": float(np.percentile(all_token_counts, 99)),
            },
        },
        "truncation_analysis": {
            "exceeds_512": {
                "count": sum(1 for t in all_token_counts if t > 512),
                "percentage": 100 * sum(1 for t in all_token_counts if t > 512) / len(all_token_counts),
            },
            "exceeds_1024": {
                "count": sum(1 for t in all_token_counts if t > 1024),
                "percentage": 100 * sum(1 for t in all_token_counts if t > 1024) / len(all_token_counts),
            },
            "exceeds_2048": {
                "count": sum(1 for t in all_token_counts if t > 2048),
                "percentage": 100 * sum(1 for t in all_token_counts if t > 2048) / len(all_token_counts),
            },
        },
        "stratification": {
            "short_512": {
                "count": len(strata["short"]["tokens"]),
                "percentage": 100 * len(strata["short"]["tokens"]) / len(all_token_counts),
            },
            "medium_512_1024": {
                "count": len(strata["medium"]["tokens"]),
                "percentage": 100 * len(strata["medium"]["tokens"]) / len(all_token_counts),
            },
            "long_1024": {
                "count": len(strata["long"]["tokens"]),
                "percentage": 100 * len(strata["long"]["tokens"]) / len(all_token_counts),
            },
        },
    }

    summary_path = output_dir / "length_statistics.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary statistics saved to: {summary_path}")
    print(f"\n{'=' * 80}")
    print("Analysis complete!")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
SHELF Synthetic Data Quality Analysis Script

This script computes comprehensive quality metrics for the SHELF benchmark dataset,
including vocabulary diversity, readability, sentence statistics, and form adherence.

Usage:
    uv run python statistics.py [--split train] [--sample-size 1000]

Requirements:
    - datasets library (HuggingFace)
    - numpy
    - collections (stdlib)
    - re (stdlib)
"""

import argparse
import json
import re
from collections import Counter
from typing import Dict, List, Any

import numpy as np
from datasets import load_dataset


class ShelfQualityAnalyzer:
    """Analyzer for SHELF benchmark quality metrics."""

    def __init__(self, dataset_name: str = "mjbommar/SHELF", split: str = "train"):
        """Initialize analyzer with dataset."""
        print(f"Loading dataset: {dataset_name}, split: {split}")
        self.ds = load_dataset(dataset_name, split=split)
        self.texts = [doc["text"] for doc in self.ds]
        print(f"Loaded {len(self.ds)} documents")

    def compute_length_statistics(self) -> Dict[str, Any]:
        """Compute document length statistics."""
        print("\nComputing length statistics...")
        doc_lengths = [len(text.split()) for text in self.texts]

        stats = {
            "total_documents": len(self.texts),
            "length_stats": {
                "mean": float(np.mean(doc_lengths)),
                "median": float(np.median(doc_lengths)),
                "std": float(np.std(doc_lengths)),
                "min": int(np.min(doc_lengths)),
                "max": int(np.max(doc_lengths)),
                "percentiles": {
                    "25": float(np.percentile(doc_lengths, 25)),
                    "50": float(np.percentile(doc_lengths, 50)),
                    "75": float(np.percentile(doc_lengths, 75)),
                    "90": float(np.percentile(doc_lengths, 90)),
                    "95": float(np.percentile(doc_lengths, 95)),
                    "99": float(np.percentile(doc_lengths, 99)),
                },
            },
        }
        return stats

    def compute_vocabulary_diversity(self) -> Dict[str, Any]:
        """Compute vocabulary diversity metrics."""
        print("\nComputing vocabulary diversity...")

        # Corpus-level statistics
        all_tokens = []
        for text in self.texts:
            all_tokens.extend(text.lower().split())

        vocab_size = len(set(all_tokens))
        total_tokens = len(all_tokens)
        type_token_ratio = vocab_size / total_tokens if total_tokens > 0 else 0

        # Token frequency distribution
        token_counts = Counter(all_tokens)
        most_common = token_counts.most_common(50)

        # Hapax legomena (words appearing only once)
        hapax = sum(1 for count in token_counts.values() if count == 1)
        hapax_ratio = hapax / vocab_size if vocab_size > 0 else 0

        # Document-level TTR (average across documents)
        doc_ttrs = []
        for text in self.texts:
            if len(text.strip()) > 0:
                tokens = text.lower().split()
                if len(tokens) > 0:
                    doc_ttrs.append(len(set(tokens)) / len(tokens))

        vocab_stats = {
            "total_tokens": total_tokens,
            "unique_tokens": vocab_size,
            "type_token_ratio": type_token_ratio,
            "avg_doc_ttr": float(np.mean(doc_ttrs)) if doc_ttrs else 0.0,
            "median_doc_ttr": float(np.median(doc_ttrs)) if doc_ttrs else 0.0,
            "hapax_legomena": hapax,
            "hapax_ratio": hapax_ratio,
            "most_common_tokens": most_common[:20],
        }
        return vocab_stats

    def compute_distinct_ngrams(self, n: int, sample_size: int = 1000) -> float:
        """
        Compute Distinct-N metric (diversity of n-grams).

        Args:
            n: N-gram size (1 for unigram, 2 for bigram, etc.)
            sample_size: Number of documents to sample (for performance)

        Returns:
            Ratio of unique n-grams to total n-grams
        """
        all_ngrams = []
        sample_texts = self.texts[:sample_size] if sample_size else self.texts

        for text in sample_texts:
            tokens = text.lower().split()
            ngrams = [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
            all_ngrams.extend(ngrams)

        if not all_ngrams:
            return 0.0
        return len(set(all_ngrams)) / len(all_ngrams)

    def compute_ngram_diversity(self, sample_size: int = 1000) -> Dict[str, float]:
        """Compute Distinct-1, Distinct-2, and Distinct-3 metrics."""
        print(f"\nComputing n-gram diversity (sample size: {sample_size})...")

        return {
            "distinct_1": self.compute_distinct_ngrams(1, sample_size),
            "distinct_2": self.compute_distinct_ngrams(2, sample_size),
            "distinct_3": self.compute_distinct_ngrams(3, sample_size),
        }

    def compute_sentence_statistics(self, sample_size: int = 1000) -> Dict[str, float]:
        """Compute sentence-level statistics."""
        print(f"\nComputing sentence statistics (sample size: {sample_size})...")

        sentence_counts = []
        avg_sentence_lengths = []
        sample_texts = self.texts[:sample_size] if sample_size else self.texts

        for text in sample_texts:
            # Split on sentence boundaries
            sentences = re.split(r"[.!?]+", text)
            sentences = [s.strip() for s in sentences if s.strip()]
            sentence_counts.append(len(sentences))

            if sentences:
                sent_lens = [len(s.split()) for s in sentences]
                avg_sentence_lengths.append(np.mean(sent_lens))

        return {
            "avg_sentences_per_doc": float(np.mean(sentence_counts)),
            "median_sentences_per_doc": float(np.median(sentence_counts)),
            "avg_words_per_sentence": float(np.mean(avg_sentence_lengths)),
            "median_words_per_sentence": float(np.median(avg_sentence_lengths)),
        }

    def flesch_reading_ease_approx(self, text: str) -> float:
        """
        Approximate Flesch Reading Ease score.

        Formula: 206.835 - 1.015 * (words/sentences) - 84.6 * (syllables/words)

        Note: This uses a crude syllable approximation (counting vowel groups).
        """
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()

        if not sentences or not words:
            return 0.0

        # Approximate syllables (crude but fast: count vowel groups)
        syllables = sum(
            max(1, len(re.findall(r"[aeiouAEIOU]+", word))) for word in words
        )

        avg_sent_len = len(words) / len(sentences)
        avg_syllables = syllables / len(words)

        # Flesch Reading Ease formula
        return 206.835 - 1.015 * avg_sent_len - 84.6 * avg_syllables

    def compute_readability(self, sample_size: int = 1000) -> Dict[str, Any]:
        """Compute readability metrics."""
        print(f"\nComputing readability metrics (sample size: {sample_size})...")

        sample_texts = self.texts[:sample_size] if sample_size else self.texts
        readability_scores = [
            self.flesch_reading_ease_approx(text) for text in sample_texts
        ]

        return {
            "flesch_reading_ease": {
                "mean": float(np.mean(readability_scores)),
                "median": float(np.median(readability_scores)),
                "std": float(np.std(readability_scores)),
                "interpretation": self._interpret_flesch(np.mean(readability_scores)),
            }
        }

    @staticmethod
    def _interpret_flesch(score: float) -> str:
        """Interpret Flesch Reading Ease score."""
        if score >= 90:
            return "Very Easy (5th grade)"
        elif score >= 80:
            return "Easy (6th grade)"
        elif score >= 70:
            return "Fairly Easy (7th grade)"
        elif score >= 60:
            return "Standard (8th-9th grade)"
        elif score >= 50:
            return "Fairly Difficult (high school)"
        elif score >= 30:
            return "Difficult (college)"
        else:
            return "Very Difficult (college graduate)"

    def compute_register_diversity(self, sample_per_register: int = 100) -> Dict[str, float]:
        """Compute vocabulary richness by register."""
        print(f"\nComputing register diversity (sample: {sample_per_register} per register)...")

        registers = {}
        for doc in self.ds:
            reg = doc.get("register", "unknown")
            if reg not in registers:
                registers[reg] = []
            registers[reg].append(doc["text"])

        register_ttrs = {}
        for reg in sorted(registers.keys()):
            all_tokens = []
            sample = registers[reg][:sample_per_register]
            for text in sample:
                all_tokens.extend(text.lower().split())

            if all_tokens:
                ttr = len(set(all_tokens)) / len(all_tokens)
                register_ttrs[reg] = {
                    "ttr": float(ttr),
                    "document_count": len(registers[reg]),
                    "sample_size": len(sample),
                }

        return register_ttrs

    def compute_model_distribution(self) -> Dict[str, Dict[str, Any]]:
        """Analyze distribution across generation models."""
        print("\nComputing model distribution...")

        model_stats = {}
        for doc in self.ds:
            model = doc.get("model", "unknown")
            if model not in model_stats:
                model_stats[model] = []
            model_stats[model].append(len(doc["text"].split()))

        result = {}
        for model in sorted(model_stats.keys()):
            lengths = model_stats[model]
            result[model] = {
                "count": len(lengths),
                "avg_length": float(np.mean(lengths)),
                "median_length": float(np.median(lengths)),
                "percentage": float(len(lengths) / len(self.ds) * 100),
            }

        return result

    def compute_form_distribution(self) -> Dict[str, int]:
        """Compute distribution of LCGFT forms."""
        print("\nComputing form distribution...")

        form_counts = {}
        for doc in self.ds:
            form = doc.get("lcgft_form", "unknown")
            form_counts[form] = form_counts.get(form, 0) + 1

        return dict(sorted(form_counts.items(), key=lambda x: -x[1]))

    def run_full_analysis(
        self, sample_size: int = 1000, output_file: str = None
    ) -> Dict[str, Any]:
        """
        Run complete quality analysis.

        Args:
            sample_size: Number of documents to sample for expensive metrics
            output_file: Optional path to save JSON results

        Returns:
            Dictionary with all quality metrics
        """
        print("=" * 80)
        print("SHELF Synthetic Data Quality Analysis")
        print("=" * 80)

        results = {
            "dataset_info": {
                "total_documents": len(self.ds),
                "split": self.ds.split if hasattr(self.ds, "split") else "unknown",
            },
            "length_statistics": self.compute_length_statistics(),
            "vocabulary_diversity": self.compute_vocabulary_diversity(),
            "ngram_diversity": self.compute_ngram_diversity(sample_size),
            "sentence_statistics": self.compute_sentence_statistics(sample_size),
            "readability": self.compute_readability(sample_size),
            "register_diversity": self.compute_register_diversity(),
            "model_distribution": self.compute_model_distribution(),
            "form_distribution": self.compute_form_distribution(),
        }

        if output_file:
            print(f"\nSaving results to {output_file}...")
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {output_file}")

        return results

    def print_summary(self, results: Dict[str, Any]) -> None:
        """Print formatted summary of results."""
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        # Length stats
        length = results["length_statistics"]["length_stats"]
        print(f"\nDocument Length:")
        print(f"  Mean: {length['mean']:.1f} words")
        print(f"  Median: {length['median']:.1f} words")
        print(f"  Range: {length['min']}-{length['max']} words")

        # Vocabulary
        vocab = results["vocabulary_diversity"]
        print(f"\nVocabulary Diversity:")
        print(f"  Unique tokens: {vocab['unique_tokens']:,}")
        print(f"  Corpus TTR: {vocab['type_token_ratio']:.4f} ({vocab['type_token_ratio']*100:.2f}%)")
        print(f"  Avg doc TTR: {vocab['avg_doc_ttr']:.4f} ({vocab['avg_doc_ttr']*100:.2f}%)")
        print(f"  Hapax ratio: {vocab['hapax_ratio']:.4f} ({vocab['hapax_ratio']*100:.2f}%)")

        # N-gram diversity
        ngram = results["ngram_diversity"]
        print(f"\nN-gram Diversity:")
        print(f"  Distinct-1: {ngram['distinct_1']:.4f}")
        print(f"  Distinct-2: {ngram['distinct_2']:.4f}")
        print(f"  Distinct-3: {ngram['distinct_3']:.4f}")

        # Readability
        flesch = results["readability"]["flesch_reading_ease"]
        print(f"\nReadability:")
        print(f"  Flesch Reading Ease: {flesch['mean']:.2f} ({flesch['interpretation']})")

        # Sentence stats
        sent = results["sentence_statistics"]
        print(f"\nSentence Statistics:")
        print(f"  Avg sentences/doc: {sent['avg_sentences_per_doc']:.1f}")
        print(f"  Avg words/sentence: {sent['avg_words_per_sentence']:.1f}")

        # Models
        print(f"\nModel Distribution:")
        for model, stats in list(results["model_distribution"].items())[:5]:
            print(f"  {model}: {stats['count']:,} docs ({stats['percentage']:.1f}%)")

        # Forms
        print(f"\nForm Distribution:")
        print(f"  Total unique forms: {len(results['form_distribution'])}")
        top_forms = list(results["form_distribution"].items())[:5]
        for form, count in top_forms:
            print(f"  {form}: {count}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compute quality metrics for SHELF benchmark"
    )
    parser.add_argument(
        "--split", default="train", help="Dataset split to analyze (default: train)"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Sample size for expensive metrics (default: 1000)",
    )
    parser.add_argument(
        "--output",
        default="quality_metrics.json",
        help="Output JSON file (default: quality_metrics.json)",
    )
    parser.add_argument(
        "--dataset",
        default="mjbommar/SHELF",
        help="Dataset name (default: mjbommar/SHELF)",
    )

    args = parser.parse_args()

    # Run analysis
    analyzer = ShelfQualityAnalyzer(dataset_name=args.dataset, split=args.split)
    results = analyzer.run_full_analysis(
        sample_size=args.sample_size, output_file=args.output
    )
    analyzer.print_summary(results)

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

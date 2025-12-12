#!/usr/bin/env python
"""Run retrieval evaluation on SHELF benchmark.

This script demonstrates the evaluation harness by running
retrieval evaluation on various embedding models.

Usage:
    # Quick test with small sample
    python scripts/run_retrieval_evaluation.py --max-queries 50

    # Full evaluation
    python scripts/run_retrieval_evaluation.py

    # Specific model
    python scripts/run_retrieval_evaluation.py --model all-mpnet-base-v2

    # Save results
    python scripts/run_retrieval_evaluation.py --output results/
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run SHELF retrieval evaluation")
    parser.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence transformer model name (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="lcc_retrieval",
        help="Task to evaluate (default: lcc_retrieval)",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Maximum number of queries for testing (default: all)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for encoding (default: 32)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for results (default: print only)",
    )
    parser.add_argument(
        "--all-retrieval",
        action="store_true",
        help="Run all retrieval tasks",
    )
    args = parser.parse_args()

    # Import here to avoid slow imports in --help
    from shelf.evaluate import evaluate
    from shelf.evaluate.adapters import SentenceTransformerEmbedder
    from shelf.evaluate.registry import list_retrieval_tasks

    # Load model
    logger.info(f"Loading model: {args.model}")
    embedder = SentenceTransformerEmbedder.from_pretrained(args.model)
    logger.info(f"Model loaded: dim={embedder.embedding_dim}")

    # Determine output path
    output_dir = Path(args.output) if args.output else None

    if args.all_retrieval:
        # Run all retrieval tasks
        tasks = list_retrieval_tasks()
        logger.info(f"Running all retrieval tasks: {tasks}")

        results = {}
        for task in tasks:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Task: {task}")
            logger.info("=" * 60)

            result = evaluate(
                task=task,
                model=embedder,
                max_queries=args.max_queries,
                batch_size=args.batch_size,
                show_progress=True,
            )

            results[task] = result
            print(f"\n{result.summary()}\n")

            if output_dir:
                output_path = output_dir / f"{task}_{args.model.replace('/', '_')}.json"
                result.to_json(output_path)
                logger.info(f"Saved: {output_path}")

        # Print summary table
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"{'Task':<30} {'NDCG@10':<10} {'MRR':<10} {'R@10':<10}")
        print("-" * 60)
        for task, result in results.items():
            ndcg = result.metrics.get("ndcg@10", 0)
            mrr = result.metrics.get("mrr", 0)
            recall = result.metrics.get("recall@10", 0)
            print(f"{task:<30} {ndcg:<10.4f} {mrr:<10.4f} {recall:<10.4f}")

    else:
        # Run single task
        logger.info(f"Running task: {args.task}")

        output_path = None
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = (
                output_dir / f"{args.task}_{args.model.replace('/', '_')}.json"
            )

        result = evaluate(
            task=args.task,
            model=embedder,
            max_queries=args.max_queries,
            batch_size=args.batch_size,
            output_path=output_path,
            show_progress=True,
        )

        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(result.summary())

        if output_path:
            logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()

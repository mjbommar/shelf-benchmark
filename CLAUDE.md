# SHELF - Synthetic Harness for Evaluating LLM Fitness

## Project Rename

This project was renamed from `locbench` to `shelf` (SHELF) because another project already uses the "locbench" name.

**SHELF** = **S**ynthetic **H**arness for **E**valuating **L**LM **F**itness

The name:
- **Synthetic**: Documents are AI-generated using GPT-5.1 and GPT-5.2
- **Harness**: A structured framework for running evaluations
- **Evaluating**: Benchmark for measuring model capabilities
- **LLM Fitness**: How well language models perform on classification, retrieval, and clustering tasks
- **SHELF** (the word): Evokes libraries, where books are organized by classification

## Overview

SHELF is a synthetic benchmark for evaluating language models on bibliographic classification tasks using Library of Congress taxonomies:

- **LCC** (Library of Congress Classification): 21 subject classes (A-Z)
- **LCGFT** (Library of Congress Genre/Form Terms): 14 categories, 133 specific forms
- **LCSH** (Library of Congress Subject Headings): Topics and geographic terms
- **LCDGT** (Library of Congress Demographic Group Terms): Audience types

## Why SHELF is Universal, Not Domain-Specific

Unlike narrow domain benchmarks (e.g., FinMTEB for finance, ChemTEB for chemistry), SHELF uses the Library of Congress Classification—the most comprehensive bibliographic taxonomy ever developed, designed to organize **all human knowledge**.

**Coverage (from 20,000 documents):**
- **21 LCC subject classes**: Near-uniform distribution (4.4-5.2% each) spanning Science, Fine Arts, Law, Medicine, Philosophy, History, Technology, Agriculture, etc.
- **133 document forms**: Lectures, Maps, Prayers, Jokes, Biographies, Satellite imagery, Games, Legal briefs, etc.
- **112 topics**: Art, Religion, Culture, Ethics, Defense, Democracy, Globalization, etc.
- **44 geographic regions**: Global coverage (US, Europe, Asia, South America, etc.)
- **25 audience types**: Children to specialists, lawyers to general public
- **8 writing registers**: Academic, professional, casual, creative, technical, etc.

**Key insight**: The co-occurrence matrices show **independence** between dimensions—every LCC class appears with every genre category. This means the benchmark includes Maps about Philosophy, Jokes about Law, Prayers about Technology, etc. This cross-product diversity is **more comprehensive than real-world corpora**, which exhibit strong genre-subject correlations.

**Positioning**: SHELF is "domain-complete" rather than "domain-specific." Strong SHELF performance indicates genuine document understanding across the full breadth of human intellectual output.

## Project Structure

```
shelf/
├── src/shelf/              # Main Python package
│   ├── taxonomies/         # Taxonomy loading and models
│   ├── benchmark/          # Benchmark generation
│   ├── sampler/            # Document sampling
│   ├── hub/                # HuggingFace Hub integration
│   ├── config/             # Configuration management
│   └── cli.py              # CLI entry point
├── scripts/                # Utility scripts
├── data/
│   ├── artifacts/          # Generated document artifacts
│   ├── hf_dataset/         # HuggingFace dataset files
│   └── taxonomies/         # Extracted taxonomy files
└── docs/
    └── tasks/              # Task definitions
```

## Tasks

The benchmark evaluates LLM fitness across:

1. **Classification**: LCC, LCGFT form, topic, audience, register
2. **Retrieval**: Subject-based document retrieval
3. **Clustering**: Document clustering by taxonomy
4. **Pair Classification**: Same-LCC or same-form pair classification

## HuggingFace Dataset

- **Repo ID**: `mjbommar/SHELF`
- **Configurations**: `default`, `same_lcc_pairs`, `same_form_pairs`
- **Splits**: train (12,000), validation (4,000), test (4,000)

## CLI Usage

```bash
# List taxonomies
shelf list

# Show taxonomy info
shelf info lcgft

# Generate benchmark documents
shelf gen create --count 1000 --stratified

# Prepare HuggingFace dataset
python scripts/prepare_hf_dataset.py --upload --repo-id mjbommar/SHELF
```

## Development

```bash
# Install dependencies
uv sync

# Run CLI
uv run shelf list

# Analyze corpus distribution
python scripts/analyze_distribution.py

# Code quality (ALWAYS run before committing)
ruff check --fix .
ruff format .
ty check

# NEVER use pyright or mypy - use ty for type checking
```

## Evaluation Design Notes

Based on research into MTEB, HELM, EleutherAI eval harness, and benchmark gaming issues:

**Key design principles for `src/shelf/evaluate/`:**
1. **Prediction-file-first**: Primary interface is prediction files (JSONL), not model objects. Enables reproducibility and supports any framework.
2. **Strict versioning**: Every result must include dataset checksum, code version, sklearn version, random seed.
3. **Multiple metrics always**: Report macro-F1, micro-F1, weighted-F1, per-class breakdown. No cherry-picking.
4. **Explicit edge case handling**: Set `zero_division=0.0` explicitly in sklearn (avoid bugs in sklearn <1.4).
5. **Contamination transparency**: Require disclosure of training data in submissions.
6. **Rich results**: Include confusion matrices, misclassified IDs, confidence intervals (bootstrap).

**Lessons from other benchmarks:**
- MTEB: Model loading method matters (normalization, prompts). OS/Python version can affect results.
- Leaderboard gaming is rampant. Require full metric reporting, not single scores.
- Data contamination is hard to prevent. Synthetic data (like SHELF) is less likely to be in pretraining corpora.
- Per-task breakdowns are more useful than aggregate scores for model selection.

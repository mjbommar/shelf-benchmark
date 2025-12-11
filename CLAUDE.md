# SHELF - Synthetic Harness for Evaluating LLM Fitness

## Project Rename

This project was renamed from `locbench` to `shelf` (SHELF) because another project already uses the "locbench" name.

**SHELF** = **S**ynthetic **H**arness for **E**valuating **L**LM **F**itness

The name:
- **Synthetic**: Documents are AI-generated using GPT-5.1
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
- **Splits**: train (6,000), validation (2,000), test (2,000)

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
```

# SHELF - Synthetic Harness for Evaluating LLM Fitness

A synthetic benchmark for evaluating language models on bibliographic classification tasks using Library of Congress taxonomies.

## Overview

SHELF evaluates LLM fitness across classification, retrieval, and clustering tasks using synthetic documents generated with controlled bibliographic metadata. The benchmark uses established Library of Congress taxonomies:

- **LCC** (Library of Congress Classification): 21 subject classes (A-Z)
- **LCGFT** (Library of Congress Genre/Form Terms): Genre and form categories
- **LCSH** (Library of Congress Subject Headings): Topics and geographic terms
- **LCDGT** (Library of Congress Demographic Group Terms): Audience types

## Installation

```bash
# Using uv (recommended)
uv sync
uv run shelf --help

# Or with pip
pip install -e .
shelf --help
```

## Usage

```bash
# List available taxonomies
shelf list

# Show taxonomy details
shelf info lcgft

# Generate benchmark documents
shelf gen create --count 1000 --stratified
```

## Project Structure

```
shelf/
+-- src/shelf/              # Main Python package
|   +-- taxonomies/         # Taxonomy loading and models
|   +-- benchmark/          # Benchmark generation
|   +-- sampler/            # Document sampling
|   +-- hub/                # HuggingFace Hub integration
|   +-- config/             # Configuration management
|   +-- cli.py              # CLI entry point
+-- scripts/                # Utility and analysis scripts
+-- data/taxonomies/        # Extracted taxonomy files
+-- docs/                   # Documentation and task definitions
```

## Benchmark Tasks

1. **Classification**: LCC, LCGFT form, topic, audience, register classification
2. **Retrieval**: Subject-based document retrieval
3. **Clustering**: Document clustering by taxonomy
4. **Pair Classification**: Same-LCC, same-form, same-audience, and same-register pair classification

## HuggingFace Dataset

The benchmark dataset is available on HuggingFace:

- **Repository**: [mjbommar/SHELF](https://huggingface.co/datasets/mjbommar/SHELF)
- **Configurations**: `default`, `same_lcc_pairs`, `same_form_pairs`, `same_audience_pairs`, `same_register_pairs`
- **Splits**: train (12,000-20,000), validation (4,000), test (4,000)

## License

This work is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

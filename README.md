# SHELF - Synthetic Harness for Evaluating LLM Fitness

A synthetic benchmark for evaluating language models on bibliographic classification tasks using Library of Congress taxonomies.

## Overview

SHELF evaluates LLM fitness across classification, retrieval, and clustering tasks using synthetic documents generated with controlled bibliographic metadata. The benchmark uses established Library of Congress taxonomies:

- **LCC** (Library of Congress Classification): 21 subject classes (A-Z)
- **LCGFT** (Library of Congress Genre/Form Terms): 14 categories, 133 specific forms
- **LCSH** (Library of Congress Subject Headings): 112 topics and 44 geographic terms
- **LCDGT** (Library of Congress Demographic Group Terms): 25 audience types

## Dataset Statistics (v0.3.0)

| Metric | Value |
|--------|-------|
| Total documents | 42,616 |
| Train split | 25,569 (60%) |
| Validation split | 8,523 (20%) |
| Test split | 8,524 (20%) |
| Generation models | 9 |
| LCC classes | 21 |
| LCGFT forms | 133 |
| Writing registers | 8 |

## Generation Models

Documents are synthetically generated using multiple frontier language models to reduce single-model biases:

| Provider | Models | Corpus Share |
|----------|--------|--------------|
| **OpenAI** | GPT-5.1, GPT-5.2 | ~94% |
| **Google** | Gemini 2.5 Flash, Flash Lite, Pro; Gemini 3 Pro Preview | ~3% |
| **Anthropic** | Claude Haiku 4.5, Sonnet 4.5, Opus 4.5 | ~3% |

Each document includes generation metadata (`model`, `temperature`, `top_p`) for filtering or analysis by source model.

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

The benchmark dataset is available on HuggingFace: [mjbommar/SHELF](https://huggingface.co/datasets/mjbommar/SHELF)

```python
from datasets import load_dataset

# Load the main dataset
dataset = load_dataset("mjbommar/SHELF")

# Load pair classification subsets
lcc_pairs = load_dataset("mjbommar/SHELF", name="same_lcc_pairs")
form_pairs = load_dataset("mjbommar/SHELF", name="same_form_pairs")
```

### Dataset Configurations

| Config | Description | Train | Val | Test |
|--------|-------------|-------|-----|------|
| `default` | Individual documents with all metadata | 25,569 | 8,523 | 8,524 |
| `same_lcc_pairs` | Document pairs labeled by LCC match | 20,000 | 4,000 | 4,000 |
| `same_form_pairs` | Document pairs labeled by LCGFT form match | 20,000 | 4,000 | 4,000 |
| `same_register_pairs` | Document pairs labeled by register match | 20,000 | 4,000 | 4,000 |
| `same_audience_pairs` | Document pairs labeled by audience match | 20,000 | 4,000 | 4,000 |
| `same_topic_pairs` | Binary: do documents share any topic? | 20,000 | 4,000 | 4,000 |
| `topic_overlap_pairs` | Graded: how many topics shared? (0/1/2/3+) | ~19,000 | ~4,000 | ~4,000 |

## License

This work is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

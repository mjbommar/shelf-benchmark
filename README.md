# SHELF - Synthetic Harness for Evaluating LLM Fitness

[![arXiv](https://img.shields.io/badge/arXiv-2609.03047-b31b1b.svg)](https://arxiv.org/abs/2609.03047)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20dataset-mjbommar%2FSHELF-yellow)](https://huggingface.co/datasets/mjbommar/SHELF)
[![License](https://img.shields.io/badge/license-CC%20BY%204.0-green.svg)](LICENSE)

A synthetic benchmark for evaluating language models on bibliographic classification tasks using Library of Congress taxonomies.

**Paper:** [SHELF: A Synthetic Harness for Multi-Task Bibliographic Benchmarking](https://arxiv.org/abs/2609.03047) (arXiv:2609.03047)
**Dataset:** [mjbommar/SHELF](https://huggingface.co/datasets/mjbommar/SHELF) on Hugging Face

## Overview

SHELF evaluates LLM fitness across classification, retrieval, and clustering tasks using synthetic documents generated with controlled bibliographic metadata. The benchmark uses established Library of Congress taxonomies:

- **LCC** (Library of Congress Classification): 21 subject classes (A-Z)
- **LCGFT** (Library of Congress Genre/Form Terms): 14 categories, 133 specific forms
- **LCSH** (Library of Congress Subject Headings): 112 topics and 44 geographic terms
- **LCDGT** (Library of Congress Demographic Group Terms): 25 audience types

## Dataset statistics

The aggregate `all` configuration is the primary released corpus.

| Metric | Value |
|--------|-------|
| Total documents | 62,899 |
| Train split | 37,795 |
| Validation split | 12,600 |
| Test split | 12,504 |
| Generation models | 25 |
| LCC classes | 21 |
| LCGFT forms | 133 |
| Writing registers | 8 |

The corpus includes a generator-balanced factorial component of 18,345
documents from 15 models across 11 laboratories. Use `all` for the complete
62,899-document corpus and `v0_4_core` when generator balance or replicated
specifications matter.

## Generation Models

The aggregate corpus contains documents from 25 writing models. Its factorial
component contains 15 models from Anthropic, OpenAI, Google, Alibaba,
DeepSeek, Zhipu, Moonshot, MiniMax, Meta, Mistral, and xAI. The aggregate
corpus is not generator-balanced: its largest writing model supplies 47.7% of
documents. In the factorial component, the largest share is 9.24%.

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

The `shelf` CLI provides tools for taxonomy management, benchmark generation, model management, and evaluation.

### Command Overview

```
shelf                              # Main CLI
├── list                           # List taxonomies
├── info <taxonomy>                # Show taxonomy info
├── extract <taxonomy>             # Extract taxonomy labels
├── extract-all                    # Extract all taxonomies
│
├── gen                            # Generation subcommand
│   ├── stats                      # Show taxonomy stats
│   ├── sample                     # Generate sample docs
│   ├── create                     # Generate benchmark
│   └── distribution               # Analyze distribution
│
├── eval                           # Evaluation subcommand
│   ├── run                        # Run evaluations
│   ├── status                     # Show evaluation status
│   ├── results                    # Show results summary
│   └── efficiency                 # Show efficiency rankings
│
├── train                          # Training subcommand
│   └── classify                   # Fine-tune transformers classifiers on SHELF tasks
│
└── models                         # Model management
    ├── list                       # List configured models
    ├── add <hf_model_id>          # Add a model from HuggingFace
    ├── remove <model_key>         # Remove a model
    └── info <model_key>           # Show model details
```

### Quick Examples

```bash
# Taxonomy exploration
shelf list                          # List all taxonomies
shelf info lcgft                    # Show LCGFT taxonomy details

# Benchmark generation
shelf gen create --count 1000 --stratified

# Model management
shelf models list                   # List configured models
shelf models add BAAI/bge-small-en-v1.5  # Add model from HuggingFace

# Run evaluations
shelf eval run --models minilm bge_small --skip-existing
shelf eval status                   # Check progress
shelf eval results                  # View SHELF scores and rankings
shelf eval efficiency               # View efficiency rankings
# Classification tasks train lightweight heads on embeddings (LogReg + RandomForest by default);
# customize with `--classifiers` or `evaluation.classification_heads` in config.

# Full fine-tuning (transformers sequence classification)
shelf train classify lcc_classification roberta-base -o results/finetune/roberta_lcc
# Then evaluate the fine-tuned checkpoint by adding a model entry with type: transformers_classifier

# Example: fair comparison (same LCC train/val) between full fine-tune and shallow logistic head
CUDA_VISIBLE_DEVICES=0 uv run shelf train classify lcc_classification bert-base-uncased \
  -o results/finetune/bert_lcc_full --epochs 3 --lr 2e-5 \
  --train-batch-size 16 --eval-batch-size 64 --max-length 256 --warmup-ratio 0.1
# Evaluate the fine-tuned checkpoint
CUDA_VISIBLE_DEVICES=0 uv run shelf eval run --config scripts/baselines/config.yaml \
  --models bert_lcc_finetune_full --tasks lcc_classification --batch-size 16
# Logistic baseline on the same task (frozen embeddings + logistic head)
CUDA_VISIBLE_DEVICES=0 uv run shelf eval run --config scripts/baselines/config.yaml \
  --models bert --tasks lcc_classification --classifiers logistic_regression --batch-size 32
# In our run: full fine-tune macro_f1≈0.918 vs logistic baseline macro_f1≈0.787
```

See [docs/cli_reference.md](docs/cli_reference.md) for complete CLI documentation.

## Project Structure

```
shelf/
├── src/shelf/              # Main Python package
│   ├── taxonomies/         # Taxonomy loading and models
│   ├── benchmark/          # Benchmark generation
│   ├── sampler/            # Document sampling
│   ├── hub/                # HuggingFace Hub integration
│   ├── evaluate/           # Evaluation framework
│   ├── config/             # Configuration management
│   ├── cli_cmds/           # CLI subcommand modules (models, eval)
│   └── cli.py              # CLI entry point
├── scripts/                # Utility and analysis scripts
├── data/taxonomies/        # Extracted taxonomy files
└── docs/                   # Documentation and task definitions
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

# Load the complete 62,899-document corpus
dataset = load_dataset("mjbommar/SHELF", "all")

# Load pair classification subsets
lcc_pairs = load_dataset("mjbommar/SHELF", name="same_lcc_pairs")
form_pairs = load_dataset("mjbommar/SHELF", name="same_form_pairs")
```

### Dataset Configurations

| Config | Description | Train | Val | Test |
|--------|-------------|-------|-----|------|
| `all` | Pooled corpus, every document | 37,795 | 12,600 | 12,504 |
| `default` | Original 42,532-document release | 25,518 | 8,507 | 8,507 |
| `same_lcc_pairs` | Document pairs labeled by LCC match | 20,000 | 4,000 | 4,000 |
| `same_form_pairs` | Document pairs labeled by LCGFT form match | 20,000 | 4,000 | 4,000 |
| `same_register_pairs` | Document pairs labeled by register match | 20,000 | 4,000 | 4,000 |
| `same_audience_pairs` | Document pairs labeled by audience match | 20,000 | 4,000 | 4,000 |
| `same_topic_pairs` | Binary: do documents share any topic? | 20,000 | 4,000 | 4,000 |
| `topic_overlap_pairs` | Graded: how many topics shared? (0/1/2/3+) | ~19,000 | ~4,000 | ~4,000 |

## License

This work is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Citation

```bibtex
@article{bommarito2026shelf,
  title = {SHELF: A Synthetic Harness for Multi-Task Bibliographic Benchmarking},
  author = {Bommarito, Michael J.},
  year = {2026},
  journal = {arXiv preprint arXiv:2609.03047},
  eprint = {2609.03047},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  url = {https://arxiv.org/abs/2609.03047}
}
```

# SHELF CLI Reference

The `shelf` command-line interface provides tools for taxonomy management, benchmark generation, model management, and evaluation.

## Installation

```bash
# Using uv (recommended)
uv sync
uv run shelf --help

# Or with pip
pip install -e .
shelf --help
```

## Command Overview

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
└── models                         # Model management
    ├── list                       # List configured models
    ├── add <hf_model_id>          # Add a model from HuggingFace
    ├── remove <model_key>         # Remove a model
    └── info <model_key>           # Show model details
```

---

## Taxonomy Commands

### `shelf list`

List all available taxonomy types.

```bash
$ shelf list

┌─────────────────┬────────────────────────────┐
│ Type            │ Name                       │
├─────────────────┼────────────────────────────┤
│ lcc_main        │ LCC Main Classes (21)      │
│ lcc_subclass    │ LCC Subclasses (~350)      │
│ lcsh_topical    │ LCSH Topical Subjects      │
│ lcsh_geographic │ LCSH Geographic (~15k)     │
│ lcgft           │ Genre/Form Terms (~550)    │
│ ...             │                            │
└─────────────────┴────────────────────────────┘
```

### `shelf info <taxonomy>`

Show detailed information about a taxonomy including coverage analysis and top labels.

```bash
$ shelf info lcgft

┌────────────────┬─────────────┐
│ Property       │ Value       │
├────────────────┼─────────────┤
│ Type           │ lcgft       │
│ Total Labels   │ 550         │
│ Total Uses     │ 1,234,567   │
└────────────────┴─────────────┘

Coverage Analysis:
  Top 10:  45.2%
  Top 50:  78.5%
  Top 100: 89.3%
```

### `shelf extract <taxonomy>`

Extract labels from a taxonomy with optional filtering.

```bash
# Extract top 100 LCGFT terms
shelf extract lcgft --top-n 100

# Extract terms with minimum frequency
shelf extract lcsh_topical --min-freq 1000 --output-dir ./data/custom
```

**Options:**
- `--top-n, -n`: Extract top N labels by frequency
- `--min-freq`: Minimum frequency threshold
- `--output-dir, -o`: Output directory (default: data/taxonomies)

### `shelf extract-all`

Extract standard label sets from all taxonomies in parallel.

```bash
shelf extract-all --workers 4
```

**Options:**
- `--output-dir, -o`: Output directory
- `--workers, -w`: Number of parallel workers (default: 4)

---

## Generation Commands

### `shelf gen stats`

Show statistics about the LC taxonomy dimensions used for generation.

```bash
$ shelf gen stats

┌─────────────────────────┬───────┐
│ Dimension               │ Count │
├─────────────────────────┼───────┤
│ LCC Main Classes        │ 21    │
│ LCGFT Categories        │ 14    │
│ LCGFT Total Forms       │ 133   │
│ LCSH Topics             │ 112   │
│ LCDGT Audience Groups   │ 25    │
└─────────────────────────┴───────┘
```

### `shelf gen sample`

Generate and display sample documents without saving.

```bash
shelf gen sample --count 5 --seed 42
```

**Options:**
- `--count, -n`: Number of documents (default: 10)
- `--seed, -s`: Random seed (default: 42)

### `shelf gen create`

Generate benchmark documents and save to file.

```bash
# Generate 1000 random documents
shelf gen create --count 1000

# Generate stratified sample (ensures coverage)
shelf gen create --count 5000 --stratified --output ./data/benchmark.jsonl
```

**Options:**
- `--count, -n`: Number of documents (default: 1000)
- `--seed, -s`: Random seed (default: 42)
- `--stratified`: Use stratified sampling for balanced coverage
- `--output, -o`: Output JSONL file

### `shelf gen distribution`

Analyze the distribution of a generated benchmark file.

```bash
shelf gen distribution ./data/benchmark.jsonl
```

---

## Model Management Commands

### `shelf models list`

List all configured embedding models with their parameters and size categories.

```bash
$ shelf models list

                               Configured Models
┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━┳━━━━━━━━━━┓
┃ Key            ┃ Name            ┃ HuggingFace    ┃ Params ┃  Dim ┃ Category ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━╇━━━━━━━━━━┩
│ minilm         │ MiniLM-L6       │ all-MiniLM-L6… │  22.7M │  384 │ small    │
│ bge_small      │ BGE-small       │ BAAI/bge-smal… │  33.4M │  384 │ small    │
│ bge_base       │ BGE-base        │ BAAI/bge-base… │ 109.5M │  768 │ base     │
│ bge_large      │ BGE-large       │ BAAI/bge-larg… │ 335.1M │ 1024 │ large    │
│ ...            │                 │                │        │      │          │
└────────────────┴─────────────────┴────────────────┴────────┴──────┴──────────┘

15 dense models, 3 sparse models
```

**Options:**
- `--config, -c`: Path to config file
- `--category`: Filter by size category (small/base/large/sparse)

### `shelf models info <model_key>`

Show detailed information about a specific model.

```bash
$ shelf models info bge_large

╭───────────────────────────────── BGE-large ──────────────────────────────────╮
│ Key: bge_large                                                               │
│ Name: BGE-large                                                              │
│ Type: sentence_transformer                                                   │
│ Description: BAAI/bge-large-en-v1.5 (335M params, 1024 dims)                 │
│ HuggingFace Model: BAAI/bge-large-en-v1.5                                    │
│ Parameters: 335.1M                                                           │
│ Embedding Dim: 1024                                                          │
│ Size Category: large                                                         │
│ Supports: retrieval, classification, clustering, pair_classification         │
╰──────────────────────────────────────────────────────────────────────────────╯
```

### `shelf models add <hf_model_id>`

Add a new sentence-transformer model from HuggingFace. Automatically fetches parameter count and embedding dimension.

```bash
$ shelf models add nomic-ai/nomic-embed-text-v1.5

Fetching model info from HuggingFace...
✓ Model: nomic-ai/nomic-embed-text-v1.5
  Parameters: 137,000,000 (137M)
  Embedding dim: 768
  Size category: base

  Model key [nomic_embed]:
  Display name [nomic-embed-text-v1.5]: Nomic Embed

✓ Added nomic_embed to config.yaml
```

**Options:**
- `--config, -c`: Path to config file
- `--key, -k`: Model key (auto-generated if not provided)
- `--name, -n`: Display name
- `--no-fetch`: Don't fetch info from HuggingFace (manual entry)

### `shelf models remove <model_key>`

Remove a model from the configuration.

```bash
$ shelf models remove bert

? Remove model 'bert' (BERT - bert-base-uncased)? [y/N]: y
✓ Removed bert from config.yaml
```

**Options:**
- `--config, -c`: Path to config file
- `--force, -f`: Skip confirmation prompt

---

## Evaluation Commands

### `shelf eval run`

Run baseline evaluations on configured models.

```bash
# Run all evaluations
shelf eval run

# Run specific models
shelf eval run --models minilm bge_base bge_large

# Run specific tasks
shelf eval run --tasks lcc_classification lcc_retrieval

# Skip already completed evaluations
shelf eval run --skip-existing

# Run only dense models (skip TF-IDF, BM25)
shelf eval run --dense-only

# Dry run (show what would be run)
shelf eval run --dry-run
```

**Options:**
- `--config, -c`: Path to config file
- `--models, -m`: Models to evaluate (default: all)
- `--tasks, -t`: Specific tasks to evaluate
- `--task-types`: Task types to evaluate (retrieval, classification, clustering, pair_classification)
- `--skip-existing, -s`: Skip existing results
- `--dense-only`: Run only dense models
- `--sparse-only`: Run only sparse models
- `--batch-size, -b`: Batch size for embedding (default: 32)
- `--dry-run`: Show what would be run
- `--quiet, -q`: Reduce output

### `shelf eval status`

Show evaluation progress and status for all models.

```bash
$ shelf eval status

                    Evaluation Status
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
┃ Model           ┃ Status   ┃ Progress ┃ Last Updated     ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
│ MiniLM-L6       │ ✓ Done   │  16/16   │ 2025-12-14 02:25 │
│ BGE-small       │ ✓ Done   │  16/16   │ 2025-12-14 04:05 │
│ BGE-base        │ ⏳ Running│   8/16   │ 2025-12-14 04:20 │
│ BGE-large       │ ○ Pending│   0/16   │ -                │
└─────────────────┴──────────┴──────────┴──────────────────┘

Progress: 40/64 tasks (62%)
```

### `shelf eval results`

Show benchmark results summary with SHELF scores and efficiency metrics.

```bash
$ shelf eval results

╭──────────────────────────────────────────────────────────────────────────────╮
│ SHELF Benchmark Results                                                      │
│ Dataset: v0.3.0                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯

                      SHELF Score Rankings
┏━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃ Rank ┃ Model           ┃  SHELF ┃ SHELF_eff ┃ Pareto ┃ Size   ┃
┡━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│    1 │ BGE-large       │ 0.5131 │     60.18 │   ✓    │ large  │
│    2 │ TF-IDF+SVD      │ 0.5109 │         - │   -    │ sparse │
│    3 │ E5-large        │ 0.5045 │     59.18 │        │ large  │
│    4 │ GTE-base        │ 0.5020 │     62.44 │   ✓    │ base   │
│    5 │ BGE-small       │ 0.4964 │     65.98 │   ✓    │ small  │
│   ...│                 │        │           │        │        │
└──────┴─────────────────┴────────┴───────────┴────────┴────────┘

            Best by Size Category
┏━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┓
┃ Category ┃ Model      ┃  SHELF ┃ SHELF_eff ┃
┡━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━┩
│ small    │ BGE-small  │ 0.4964 │     65.98 │
│ base     │ GTE-base   │ 0.5020 │     62.44 │
│ large    │ BGE-large  │ 0.5131 │     60.18 │
│ sparse   │ TF-IDF+SVD │ 0.5109 │         - │
└──────────┴────────────┴────────┴───────────┘
```

**Options:**
- `--config, -c`: Path to config file
- `--sort`: Sort by: shelf (default), shelf_eff, params
- `--limit, -n`: Number of models to show (default: 20)

### `shelf eval efficiency`

Show efficiency rankings - models sorted by best performance per parameter.

```bash
$ shelf eval efficiency

                      SHELF Score Rankings (by Efficiency)
┏━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃ Rank ┃ Model           ┃  SHELF ┃ SHELF_eff ┃ Pareto ┃ Size   ┃
┡━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│    1 │ BGE-small       │ 0.4964 │     65.98 │   ✓    │ small  │
│    2 │ GTE-small       │ 0.4959 │     65.92 │        │ small  │
│    3 │ E5-small        │ 0.4772 │     63.43 │        │ small  │
│    4 │ MiniLM-L6       │ 0.4647 │     63.17 │   ✓    │ small  │
│    5 │ GTE-base        │ 0.5020 │     62.44 │   ✓    │ base   │
│   ...│                 │        │           │        │        │
└──────┴─────────────────┴────────┴───────────┴────────┴────────┘
```

---

## Metrics Reference

### SHELF Score

Weighted average across task types:
- Retrieval: 30% (NDCG@10)
- Classification: 25% (Macro F1)
- Clustering: 20% (V-Measure)
- Pair Classification: 25% (F1)

### SHELF_eff (Efficiency Score)

```
SHELF_eff = SHELF_score × 1000 / log₁₀(params)
```

Higher values indicate better efficiency (more performance per parameter).

### Pareto Optimal

A model is Pareto-optimal if no other model has both higher SHELF score AND fewer parameters.

### Size Categories

- **small**: < 50M parameters
- **base**: 50M - 300M parameters
- **large**: > 300M parameters
- **sparse**: Non-neural models (TF-IDF, BM25)

---

## Configuration

The CLI uses `scripts/baselines/config.yaml` for model and task configuration. See [config.yaml](../scripts/baselines/config.yaml) for the full specification.

### Adding Custom Models

1. Use the CLI: `shelf models add <hf_model_id>`
2. Or edit `config.yaml` directly:

```yaml
models:
  my_model:
    type: sentence_transformer
    name: "My Model"
    description: "Custom model (XM params, Y dims)"
    model_name: "organization/model-name"
    num_params: 100000000
    embedding_dim: 768
    size_category: "base"
    supports: [retrieval, classification, clustering, pair_classification]
```

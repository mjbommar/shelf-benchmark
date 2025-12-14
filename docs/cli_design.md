# SHELF CLI Design

## Overview

Unified CLI using Typer + Rich for beautiful, user-friendly experience.

## Command Structure

```
shelf                              # Main CLI
├── list                           # List taxonomies (existing)
├── info <taxonomy>                # Show taxonomy info (existing)
├── extract <taxonomy>             # Extract taxonomy labels (existing)
├── extract-all                    # Extract all taxonomies (existing)
│
├── gen                            # Generation subcommand (existing)
│   ├── stats                      # Show taxonomy stats
│   ├── sample                     # Generate sample docs
│   ├── create                     # Generate benchmark
│   └── distribution               # Analyze distribution
│
├── eval                           # Evaluation subcommand (NEW)
│   ├── run                        # Run evaluations
│   ├── status                     # Show evaluation status/progress
│   ├── results                    # Show results summary table
│   └── efficiency                 # Show efficiency rankings
│
└── models                         # Model management (NEW)
    ├── list                       # List configured models
    ├── add <hf_model_id>          # Add a model from HuggingFace
    ├── remove <model_key>         # Remove a model
    └── info <model_key>           # Show model details
```

## Command Details

### `shelf models list`
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Configured Models                                  │
├──────────────┬────────────────────────────┬────────┬──────┬─────────────────┤
│ Key          │ Name                       │ Params │ Dim  │ Category        │
├──────────────┼────────────────────────────┼────────┼──────┼─────────────────┤
│ minilm       │ all-MiniLM-L6-v2          │  22.7M │  384 │ small           │
│ bge_small    │ BAAI/bge-small-en-v1.5    │  33.4M │  384 │ small           │
│ bge_base     │ BAAI/bge-base-en-v1.5     │ 109.5M │  768 │ base            │
│ bge_large    │ BAAI/bge-large-en-v1.5    │ 335.1M │ 1024 │ large           │
│ ...          │                            │        │      │                 │
├──────────────┼────────────────────────────┼────────┼──────┼─────────────────┤
│ tf           │ TF+SVD                     │      - │  256 │ sparse          │
│ tfidf        │ TF-IDF+SVD                 │      - │  256 │ sparse          │
│ bm25         │ BM25                       │      - │    - │ sparse          │
└──────────────┴────────────────────────────┴────────┴──────┴─────────────────┘

15 dense models, 3 sparse models
```

### `shelf models add <hf_model_id>`
```bash
$ shelf models add nomic-ai/nomic-embed-text-v1.5

Fetching model info from HuggingFace...
✓ Model: nomic-ai/nomic-embed-text-v1.5
  Parameters: 137,000,000 (137M)
  Embedding dim: 768
  Size category: base

? Model key [nomic_embed]:
? Display name [Nomic Embed]:

✓ Added nomic_embed to config.yaml
```

### `shelf models remove <model_key>`
```bash
$ shelf models remove bert

? Remove model 'bert' (BERT - bert-base-uncased)? [y/N]: y
✓ Removed bert from config.yaml
```

### `shelf eval run`
```bash
$ shelf eval run --models minilm bge_small --tasks lcc_classification

╭──────────────────────────────────────────────────────────────────────────────╮
│                         SHELF Baseline Evaluation                             │
╰──────────────────────────────────────────────────────────────────────────────╯
Config:    scripts/baselines/config.yaml
Output:    results/v0.3.0/baselines
Models:    2 (minilm, bge_small)
Tasks:     1 (lcc_classification)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Model: MiniLM-L6
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Building embedding cache... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:01:50
Running lcc_classification... ━━━━━━━━━━━━━━━━━━━━━━━━━━ 100%
  ✓ macro_f1 = 0.8032

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Model: BGE-small
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Building embedding cache... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:02:15
Running lcc_classification... ━━━━━━━━━━━━━━━━━━━━━━━━━━ 100%
  ✓ macro_f1 = 0.8156

╭──────────────────────────────────────────────────────────────────────────────╮
│                              Summary                                          │
╰──────────────────────────────────────────────────────────────────────────────╯
Total time: 4m 5s
Evaluations: 2 completed, 0 errors
```

### `shelf eval results`
```bash
$ shelf eval results

┌──────────────────────────────────────────────────────────────────────────────┐
│                         SHELF Benchmark Results                               │
│                            Dataset: v0.3.0                                    │
└──────────────────────────────────────────────────────────────────────────────┘

                         SHELF Score Rankings
┌──────┬───────────────┬────────┬───────────┬──────────┬────────┐
│ Rank │ Model         │  SHELF │ SHELF_eff │ Pareto   │ Size   │
├──────┼───────────────┼────────┼───────────┼──────────┼────────┤
│    1 │ BGE-large     │ 0.5131 │     60.18 │ ✓        │ large  │
│    2 │ TF-IDF+SVD    │ 0.5109 │         - │ -        │ sparse │
│    3 │ E5-large      │ 0.5045 │     59.18 │          │ large  │
│    4 │ TF+SVD        │ 0.5021 │         - │ -        │ sparse │
│    5 │ GTE-base      │ 0.5020 │     62.44 │ ✓        │ base   │
│   ...│               │        │           │          │        │
└──────┴───────────────┴────────┴───────────┴──────────┴────────┘

                      Best by Size Category
┌──────────┬───────────────┬────────┬───────────┐
│ Category │ Model         │  SHELF │ SHELF_eff │
├──────────┼───────────────┼────────┼───────────┤
│ small    │ BGE-small     │ 0.4964 │     65.98 │
│ base     │ GTE-base      │ 0.5020 │     62.44 │
│ large    │ BGE-large     │ 0.5131 │     60.18 │
│ sparse   │ TF-IDF+SVD    │ 0.5109 │         - │
└──────────┴───────────────┴────────┴───────────┘
```

### `shelf eval status`
```bash
$ shelf eval status

                         Evaluation Status
┌───────────────┬────────┬───────────────┬──────────────────────────┐
│ Model         │ Status │ Tasks Done    │ Last Updated             │
├───────────────┼────────┼───────────────┼──────────────────────────┤
│ minilm        │ ✓ Done │ 16/16         │ 2025-12-14 02:19:20      │
│ bge_small     │ ✓ Done │ 16/16         │ 2025-12-14 02:45:30      │
│ bge_base      │ ✓ Done │ 16/16         │ 2025-12-14 03:15:45      │
│ e5_large      │ ⏳ Run │ 8/16          │ 2025-12-14 04:30:00      │
│ gtr_t5_large  │ ○ Pend │ 0/16          │ -                        │
└───────────────┴────────┴───────────────┴──────────────────────────┘

Progress: 48/80 tasks (60%)
Estimated time remaining: ~2h 15m
```

## Implementation

### File Structure
```
src/shelf/
├── cli.py              # Main app, imports subcommands
├── cli/
│   ├── __init__.py
│   ├── eval.py         # shelf eval subcommands
│   └── models.py       # shelf models subcommands
```

### Key Features
- Rich tables for all output
- Progress bars for long operations
- Colorful status indicators (✓, ✗, ⏳, ○)
- Confirmation prompts for destructive actions
- Auto-detection of model parameters from HuggingFace
- YAML config management

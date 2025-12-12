# Implementation Summary: Topic Pair Classification

## Overview

This document summarizes the implementation of the **Same-Topic Pair Classification** task for the SHELF benchmark. This task tests whether models can detect topic overlap between document pairs based on multi-label Library of Congress Subject Headings.

## Files Modified

### 1. `/home/mjbommar/src/shelf-benchmark/src/shelf/hub/dataset.py`

**Added Functions:**

- `generate_topic_pairs()` - Generates document pairs with controlled topic overlap
  - Supports two modes: "binary" (any overlap) and "graded" (count overlaps 0/1/2/3+)
  - Uses balanced sampling to ensure target distributions
  - Binary: 50% no overlap, 50% any overlap
  - Graded: 40% no overlap, 30% one topic, 20% two topics, 10% three+ topics
  - Returns pairs with `label`, `overlap_count`, and `shared_topics` fields

- `prepare_topic_pair_dataset()` - Converts topic pairs to HuggingFace DatasetDict
  - Creates train/validation/test splits with pair counts: 20,000 / 4,000 / 4,000
  - Features include: doc_a/b IDs, titles, bodies, label, overlap_count, shared_topics
  - Supports both binary and graded modes

**Updated Functions:**

- `save_locally()` - Extended to generate and save topic pair datasets
  - Added generation of `same_topic` (binary) and `topic_overlap` (graded) subdirectories
  - Saves pairs in parquet or JSONL format alongside existing pair types

### 2. `/home/mjbommar/src/shelf-benchmark/src/shelf/hub/card.py`

**Added Configurations:**

Updated the YAML template to include two new dataset configurations:

- `same_topic_pairs` - Binary classification: Do documents share ANY topic?
- `topic_overlap_pairs` - Graded classification: How many topics shared (0/1/2/3+)?

**Updated Documentation:**

- Added rows to configuration table explaining the two new pair types
- Added example code showing how to load topic pair datasets
- Added example showing the data format for topic overlap pairs

### 3. `/home/mjbommar/src/shelf-benchmark/docs/tasks/topic_pair_classification.md`

**Created comprehensive task specification** covering:

- **Task Definition**: Binary (same-topic) and graded (topic overlap count) variants
- **Dataset Construction**: Multi-label topic structure, pair generation strategy
- **Statistics**: 20K train, 4K val, 4K test pairs for each variant
- **Evaluation Metrics**: F1 (binary), Macro-F1 (graded), accuracy, MCC, confusion matrices
- **Baselines**: Random, majority class, TF-IDF, SBERT, BERT cross-encoder
- **Comparison to GLUE**: How this differs from MRPC/QQP and Same-LCC/Form tasks
- **What This Tests**: Multi-label reasoning, fine-grained semantic similarity, cross-domain topic detection
- **Implementation Notes**: Loading, submission format, training tips
- **Related Work**: Similar tasks in GLUE, MTEB, SemEval, multi-label classification literature

### 4. `/home/mjbommar/src/shelf-benchmark/scripts/test_topic_pairs.py`

**Created test script** to validate implementation:

- Loads sample documents and analyzes topic distribution
- Generates binary and graded pairs
- Validates label distributions match targets
- Checks for ID uniqueness and consistency
- Provides detailed output for debugging

### 5. `/home/mjbommar/src/shelf-benchmark/scripts/analyze_topic_overlap.py`

**Created analysis script** to understand topic data:

- Analyzes topics per document distribution
- Identifies unique topics and their frequencies
- Samples overlap patterns to inform pair generation strategy

## Key Design Decisions

### 1. Binary vs Graded Tasks

**Binary Task (Same-Topic)**:
- Simpler task: does pair share ANY topic?
- 50/50 split between positive and negative
- Lower bound baseline: 50% accuracy (random)
- Primary metric: F1 score

**Graded Task (Topic Overlap Count)**:
- Harder task: HOW MANY topics shared?
- 4-class classification: 0, 1, 2, 3+ topics
- Distribution: 40% / 30% / 20% / 10%
- Primary metric: Macro-F1 (prevents ignoring rare classes)

### 2. Pair Sampling Algorithm

**For Zero Overlap (label=0)**:
- Random sampling until no shared topics found
- Rejection sampling approach

**For Positive Overlap**:
- Build inverted index: topic → [documents]
- For exact counts (graded): Search candidates for exact match
- For any overlap (binary): Sample from documents sharing at least one topic
- Check up to 50 candidates per attempt for exact overlap

**Balancing Strategy**:
- Allow up to 3× attempts per target count
- Shuffle final pairs to mix overlap levels
- Trim to exact target count

### 3. Multi-Label Schema

Each pair includes:
- `label`: Binary (0/1) or graded class (0/1/2/3)
- `overlap_count`: Exact number of shared topics (for diagnostics)
- `shared_topics`: List of actual shared topics (for interpretability)

This design allows:
- Binary classification without modification
- Graded classification with fine-grained labels
- Error analysis using `shared_topics` field
- Oracle baselines using topic Jaccard similarity

## Integration Points

### HuggingFace Hub

The new configurations will be available as:
```python
from datasets import load_dataset

# Binary task
binary = load_dataset("mjbommar/SHELF", name="same_topic_pairs")

# Graded task
graded = load_dataset("mjbommar/SHELF", name="topic_overlap_pairs")
```

### Dataset Card

The README.md will automatically include:
- Configuration table entries for both tasks
- Example code for loading and using the datasets
- Data format examples showing `shared_topics` field

### Evaluation Harness

Future `src/shelf/evaluate/` module can use:
- Binary task: Standard binary classification metrics (F1, accuracy, MCC)
- Graded task: Macro-F1, per-class F1, confusion matrix
- Both: Bootstrap confidence intervals, error sampling

## Validation Checklist

- [x] Functions implemented in `dataset.py`
- [x] Integration with `save_locally()` function
- [x] Dataset card updated with new configurations
- [x] Comprehensive task specification document
- [x] Test script created
- [x] Analysis script created
- [ ] Run test script to validate (requires manual execution)
- [ ] Generate actual pair datasets using `save_locally()` (requires manual execution)
- [ ] Upload to HuggingFace Hub (requires manual execution)

## Usage Examples

### Generating Pairs Locally

```python
from pathlib import Path
from shelf.hub.dataset import SHELFDataset

# Load dataset
dataset = SHELFDataset.from_artifacts("data/artifacts/")

# Save with pairs (includes topic pairs automatically)
output_path = dataset.save_locally(
    output_dir="data/hf_dataset/",
    format="parquet",
    include_pairs=True
)

# Topic pairs will be saved to:
# - data/hf_dataset/pairs/same_topic/
# - data/hf_dataset/pairs/topic_overlap/
```

### Testing Pair Generation

```bash
# Run the test script
python scripts/test_topic_pairs.py

# Run the analysis script
python scripts/analyze_topic_overlap.py
```

### Loading for Evaluation

```python
from datasets import load_dataset
from sklearn.metrics import f1_score, classification_report

# Load binary pairs
pairs = load_dataset("mjbommar/SHELF", name="same_topic_pairs")

# Extract features
def format_pair(example):
    return {
        "text": f"{example['doc_a_body']} [SEP] {example['doc_b_body']}",
        "label": example["label"]
    }

train = pairs["train"].map(format_pair)

# ... train model ...

# Evaluate
predictions = model.predict(test)
f1 = f1_score(test["label"], predictions)
print(f"Binary F1: {f1:.3f}")
```

## Next Steps

1. **Manual Testing**: Run `python scripts/test_topic_pairs.py` to validate
2. **Generate Full Dataset**: Run dataset generation with `include_pairs=True`
3. **Baseline Experiments**: Implement TF-IDF, SBERT, and BERT baselines
4. **Evaluation Module**: Create `src/shelf/evaluate/topic_pairs.py`
5. **Documentation**: Update main README with task descriptions
6. **HuggingFace Upload**: Push new configurations to the Hub

## Performance Expectations

Based on similar tasks in GLUE and MTEB:

**Binary Task (Same-Topic)**:
- Random baseline: 50% F1
- TF-IDF + LR: ~65-70% F1 (estimated)
- SBERT bi-encoder: ~75-80% F1 (estimated)
- BERT cross-encoder: ~80-85% F1 (estimated)

**Graded Task (Topic Overlap)**:
- Random baseline: 25% Macro-F1 (4 classes)
- TF-IDF + LR: ~40-45% Macro-F1 (estimated)
- SBERT bi-encoder: ~50-55% Macro-F1 (estimated)
- BERT cross-encoder: ~60-65% Macro-F1 (estimated)

The graded task is significantly harder due to:
- Fine-grained distinction between 1 vs 2 topics
- Class imbalance (10% have 3+ topics)
- Need to count exact overlap, not just detect it

## Conclusion

The topic pair classification tasks are now fully implemented and integrated into the SHELF benchmark. They provide a novel test of multi-label semantic understanding, complementing the existing single-label pair tasks (Same-LCC, Same-Form). The implementation is ready for testing and deployment.

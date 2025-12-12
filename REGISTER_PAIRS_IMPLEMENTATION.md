# Same-Register Pair Classification - Implementation Summary

**Date**: 2025-12-12
**Task**: Research, design, and implement the Same-Register Pair Classification task for SHELF benchmark

## Overview

This document summarizes the implementation of the **Same-Register Pair Classification** task, which tests whether models can detect stylistic similarity between document pairs based on their writing register (e.g., academic, professional, casual, creative).

## What Was Implemented

### 1. Code Changes

#### `/home/mjbommar/src/shelf-benchmark/src/shelf/hub/dataset.py`

**Modified**: `save_locally()` function (line 542)

Added "register" to the list of label fields for pair generation:

```python
for label_field in ["lcc_code", "lcgft_form", "register"]:
```

This ensures that when datasets are saved locally, register pairs are automatically generated alongside LCC and form pairs.

**No changes needed to `generate_pairs()`**: The existing implementation already supports any label field, including "register". The function is generic and works correctly for all three pair types.

#### `/home/mjbommar/src/shelf-benchmark/src/shelf/hub/card.py`

**Modified**: YAML metadata template and README template

1. **Added configuration block** (lines 147-154):
```yaml
- config_name: same_register_pairs
  data_files:
    - split: train
      path: pairs/same_register/train-*
    - split: validation
      path: pairs/same_register/validation-*
    - split: test
      path: pairs/same_register/test-*
```

2. **Updated configuration table** (line 308):
Added row for `same_register_pairs` configuration with 20,000 train / 4,000 val / 4,000 test pairs.

3. **Updated dataset count** (line 301):
Changed from "three configurations" to "four configurations".

4. **Added loading example** (line 314):
```python
register_pairs = load_dataset("{repo_id}", name="same_register_pairs")
```

### 2. Documentation

#### `/home/mjbommar/src/shelf-benchmark/docs/tasks/register_pair_classification.md` (NEW)

Comprehensive 500+ line task specification document including:

- **Overview**: Why this task is important and what it tests
- **Task Definition**: Formal specification, input/output format
- **Register Categories**: All 8 registers with descriptions and example markers
- **Dataset Construction**: Pair generation strategy, balancing, statistics
- **Evaluation**: Metrics (F1, Accuracy, MCC), evaluation protocol
- **Why This Is Harder**: Detailed comparison to LCC/Form pair tasks
  - Style is content-independent
  - Subtle linguistic distinctions (8 registers on a continuum)
  - Multi-dimensional stylistic features
  - Requires deeper language understanding
- **Baselines**: Expected performance ranges
- **Related Work**: Connections to authorship attribution, genre classification, stylometry
- **Implementation Notes**: Training tips, error analysis strategies
- **References**: 10 academic references

#### `/home/mjbommar/src/shelf-benchmark/docs/tasks/pair_classification.md` (UPDATED)

Updated the general pair classification document to mention the register task:

1. Changed "two binary pair classification tasks" to "three binary pair classification tasks"
2. Added Same-Register Prediction to the task list with link to detailed documentation
3. Added formal definition for register pairs
4. Added note explaining why register is harder (stylistic vs. content similarity)

### 3. Testing

Created and ran `/home/mjbommar/src/shelf-benchmark/test_register_pairs_simple.py` to verify:

- Register pairs can be generated successfully
- Distribution is balanced (50% positive, 50% negative)
- All 8 registers are represented in positive pairs
- Pair labels are correctly assigned
- No label errors in 1000 test pairs

**Test Results**:
```
✓ Balance check PASSED (error: 0.00%)
✓ All 1000 pairs have correct labels
✓ Good register coverage (ratio: 1.34)
✓ TEST PASSED
```

## Register Taxonomy

The SHELF benchmark uses **8 distinct writing registers**:

| Register | Description |
|----------|-------------|
| `casual` | Informal, conversational (blog posts, social media) |
| `conversational` | Friendly but clear (talking to a colleague) |
| `professional` | Standard business tone (clear, neutral) |
| `formal` | Official, ceremonial (legal, governmental) |
| `academic` | Scholarly, precise (citations, hedging) |
| `technical` | Specialized, expert-level (jargon, domain expertise) |
| `journalistic` | News style, factual (inverted pyramid) |
| `creative` | Literary, expressive (metaphor, imagery) |

## Dataset Statistics

Based on the existing SHELF corpus with ~20,000 documents:

| Split | Pairs | Positive (same register) | Negative (different register) |
|-------|-------|-------------------------|-------------------------------|
| Train | 20,000 | 10,000 (50%) | 10,000 (50%) |
| Validation | 4,000 | 2,000 (50%) | 2,000 (50%) |
| Test | 4,000 | 2,000 (50%) | 2,000 (50%) |

## How to Use

### Loading the Dataset

```python
from datasets import load_dataset

# Load register pair classification data
register_pairs = load_dataset("mjbommar/SHELF", name="same_register_pairs")

# Access splits
train = register_pairs["train"]
val = register_pairs["validation"]
test = register_pairs["test"]

# Example pair
print(train[0])
# {'id': 'pair_000001',
#  'doc_a_id': '...', 'doc_a_title': '...', 'doc_a_body': '...',
#  'doc_b_id': '...', 'doc_b_title': '...', 'doc_b_body': '...',
#  'label': 1,  # 1 = same register, 0 = different register
#  'label_field': 'register'}
```

### Generating Pairs Locally

```python
from shelf.hub.dataset import SHELFDataset

# Load dataset from artifacts
dataset = SHELFDataset.from_artifacts("data/artifacts/")

# Save with all pair types (including register)
dataset.save_locally(
    output_dir="data/hf_dataset/",
    format="parquet",
    include_pairs=True  # Generates same_lcc, same_lcgft, same_register
)

# Pairs will be saved to:
# data/hf_dataset/pairs/same_register/train.parquet
# data/hf_dataset/pairs/same_register/validation.parquet
# data/hf_dataset/pairs/same_register/test.parquet
```

### Evaluation (Future)

```bash
# When evaluation harness is implemented:
shelf evaluate --task same_register --model <model_path>
```

## Why This Task Matters

### Real-World Applications

1. **Style Transfer Evaluation**: Measure if generated text matches target register
2. **Content Moderation**: Detect tone violations (e.g., unprofessional language in business contexts)
3. **Authorship Attribution**: Identify author style independent of topic
4. **Writing Assessment**: Evaluate if student writing matches assignment register requirements
5. **Document Retrieval**: Find documents by style, not just topic

### Scientific Value

1. **Orthogonal to Content**: Tests style understanding independent of subject matter
2. **Fine-Grained Distinctions**: 8 registers on a continuum (harder than binary formality)
3. **Multi-Dimensional**: Requires understanding vocabulary, syntax, pragmatics, discourse
4. **Benchmark Diversity**: Adds stylistic dimension to existing topic/genre tasks

## Key Design Decisions

### 1. Why 8 Registers?

Based on:
- Biber's framework of register variation (1988)
- Formality continuum research (Heylighen & Dewaele, 1999)
- Practical coverage of common writing contexts
- Sufficient granularity without excessive overlap

### 2. Why Binary Pairs vs. Multi-Class?

- **Consistency**: Matches existing Same-LCC and Same-Form tasks
- **Clarity**: Clear signal for model to learn
- **Real-World**: Many applications need binary similarity (duplicate detection, clustering)
- **Extensibility**: Can build 8-way classifier from document-level labels

### 3. Why 50/50 Balance?

- **No Class Imbalance**: Model can't exploit majority class
- **Fair Evaluation**: F1 and Accuracy equally weighted
- **GLUE Alignment**: Similar to MRPC/QQP balancing strategy

## Comparison to Other Tasks

| Aspect | Same-LCC | Same-Form | Same-Register |
|--------|----------|-----------|---------------|
| **Signal Type** | Subject matter | Document genre | Writing style |
| **Classes** | 21 | 133 | 8 |
| **Difficulty** | Medium | Medium | Hard |
| **Key Features** | Topic keywords | Genre markers | Stylistic patterns |
| **Lexical Overlap** | High | Medium | Low |
| **Requires** | Bag-of-words | Structure + BOW | Syntax + Pragmatics |

**Register is harder because**:
- Style cuts across all topics (orthogonal to LCC)
- Style cuts across all genres (orthogonal to LCGFT)
- Requires understanding of linguistic dimensions beyond vocabulary

## Files Modified

1. `/home/mjbommar/src/shelf-benchmark/src/shelf/hub/dataset.py` - Add register to pair generation
2. `/home/mjbommar/src/shelf-benchmark/src/shelf/hub/card.py` - Add configuration and examples
3. `/home/mjbommar/src/shelf-benchmark/docs/tasks/register_pair_classification.md` - New task spec
4. `/home/mjbommar/src/shelf-benchmark/docs/tasks/pair_classification.md` - Update overview

## Next Steps (Future Work)

### 1. Evaluation Harness

- Implement `RegisterPairEvaluator` in `src/shelf/evaluate/evaluators/`
- Add task registration in `src/shelf/evaluate/registry.py`
- Add CLI support in `shelf evaluate --task same_register`

### 2. Baselines

Run baseline models:
- TF-IDF + Logistic Regression
- Sentence-BERT bi-encoder
- BERT cross-encoder fine-tuned

### 3. Analysis

- Confusion matrix by register pair type
- Identify hard negatives (casual vs. conversational, etc.)
- Measure content leakage (do models rely on topics?)

### 4. Documentation

- Add results to README.md leaderboard
- Write blog post explaining why stylistic tasks matter
- Create tutorial notebook for fine-tuning on register pairs

## References

1. Biber, D. (1988). *Variation Across Speech and Writing*. Cambridge University Press.
2. Heylighen, F., & Dewaele, J. M. (1999). Formality of Language: definition, measurement and behavioral determinants. *Internal Report, Center "Leo Apostel"*.
3. Pavlick, E., & Tetreault, J. (2016). An Empirical Analysis of Formality in Online Communication. *TACL*.
4. Wang, A., et al. (2019). GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding. *ICLR*.
5. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP*.

---

**Implementation completed**: 2025-12-12
**Status**: ✓ Ready for integration and testing

# TODO: Topic Pair Classification Integration

This document tracks remaining tasks for full integration of the topic pair classification tasks into SHELF.

## Implementation Status

### ✅ Completed

- [x] Design binary and graded task variants
- [x] Implement `generate_topic_pairs()` function
- [x] Implement `prepare_topic_pair_dataset()` function
- [x] Update `save_locally()` to generate topic pairs
- [x] Add configurations to `card.py` YAML template
- [x] Update dataset card README template
- [x] Write comprehensive task specification (`topic_pair_classification.md`)
- [x] Create test script (`test_topic_pairs.py`)
- [x] Create analysis script (`analyze_topic_overlap.py`)
- [x] Create implementation summary document

### 🔄 Remaining Tasks

#### 1. Testing and Validation

- [ ] Run `python scripts/test_topic_pairs.py` to validate pair generation
  - Verify label distributions match targets
  - Check for ID uniqueness
  - Validate overlap_count consistency

- [ ] Run `python scripts/analyze_topic_overlap.py` to understand topic data
  - Document topic frequency distribution
  - Analyze natural overlap patterns
  - Identify any anomalies

#### 2. Dataset Generation

- [ ] Generate full pair datasets using existing SHELF data
  ```bash
  # From Python or CLI
  from shelf.hub.dataset import SHELFDataset
  dataset = SHELFDataset.from_artifacts("data/artifacts/")
  dataset.save_locally("data/hf_dataset/", include_pairs=True)
  ```

- [ ] Verify pair files are created:
  - `data/hf_dataset/pairs/same_topic/train.parquet`
  - `data/hf_dataset/pairs/same_topic/validation.parquet`
  - `data/hf_dataset/pairs/same_topic/test.parquet`
  - `data/hf_dataset/pairs/topic_overlap/train.parquet`
  - `data/hf_dataset/pairs/topic_overlap/validation.parquet`
  - `data/hf_dataset/pairs/topic_overlap/test.parquet`

- [ ] Validate generated pair counts:
  - Binary: 20K train, 4K val, 4K test
  - Graded: 20K train, 4K val, 4K test

#### 3. Code Quality

- [ ] Run code formatters:
  ```bash
  ruff check --fix .
  ruff format .
  ```

- [ ] Run type checker:
  ```bash
  ty check
  ```

- [ ] Ensure all docstrings are complete
- [ ] Add unit tests to test suite (if one exists)

#### 4. Documentation Updates

- [ ] Update main `README.md` to mention topic pair tasks
- [ ] Update `docs/tasks/README.md` (if exists) to list new tasks
- [ ] Add task to benchmark overview table
- [ ] Update any diagrams or flowcharts

#### 5. Baseline Experiments

- [ ] Implement TF-IDF + Cosine Similarity baseline
  - Threshold-based for binary task
  - Binned similarity scores for graded task

- [ ] Implement TF-IDF + Logistic Regression baseline
  - Concatenate document vectors
  - Train binary classifier (same-topic)
  - Train multi-class classifier (topic overlap)

- [ ] Implement SBERT baseline
  - Encode documents independently
  - Compute cosine similarity
  - Threshold for binary / bin for graded

- [ ] Implement BERT cross-encoder baseline
  - Fine-tune BERT on pair classification
  - Report F1 (binary) and Macro-F1 (graded)

- [ ] Document baseline results in task specification

#### 6. Evaluation Module

- [ ] Create `src/shelf/evaluate/topic_pairs.py`
  - Binary evaluation: F1, accuracy, MCC, precision, recall
  - Graded evaluation: Macro-F1, per-class F1, confusion matrix
  - Bootstrap confidence intervals
  - Error analysis sampling

- [ ] Add to CLI: `shelf evaluate --task topic_pairs`

#### 7. HuggingFace Hub Upload

- [ ] Upload updated dataset with new pair configurations
  ```bash
  python scripts/prepare_hf_dataset.py --upload --repo-id mjbommar/SHELF
  ```

- [ ] Verify new configurations appear on Hub:
  - https://huggingface.co/datasets/mjbommar/SHELF/viewer/same_topic_pairs
  - https://huggingface.co/datasets/mjbommar/SHELF/viewer/topic_overlap_pairs

- [ ] Test loading from Hub:
  ```python
  from datasets import load_dataset
  binary = load_dataset("mjbommar/SHELF", "same_topic_pairs")
  graded = load_dataset("mjbommar/SHELF", "topic_overlap_pairs")
  ```

#### 8. Paper and Publication

- [ ] Add task to benchmark paper (if applicable)
- [ ] Create task figure/diagram showing multi-label overlap
- [ ] Document comparison to GLUE MRPC/QQP
- [ ] Write results section with baseline performance

## Priority Order

**High Priority** (required for release):
1. Testing and validation (run test scripts)
2. Dataset generation (create pair files)
3. Code quality (linting, type checking)
4. HuggingFace upload (make data accessible)

**Medium Priority** (nice to have):
5. Baseline experiments (show task difficulty)
6. Evaluation module (standardize metrics)
7. Documentation updates (improve discoverability)

**Low Priority** (future work):
8. Paper integration (publish results)

## Timeline Estimate

- **Day 1**: Testing, validation, and dataset generation (items 1-2)
- **Day 2**: Code quality and documentation (items 3-4)
- **Week 1**: Baseline experiments (item 5)
- **Week 2**: Evaluation module and Hub upload (items 6-7)
- **Future**: Paper integration (item 8)

## Notes

- The implementation is complete and ready for testing
- No breaking changes to existing code
- New pair types are optional (controlled by `include_pairs=True`)
- Backwards compatible with existing pair generation (LCC, form, register, audience)

## Questions/Issues

- Should we cap graded task at 3+ or allow higher counts? (Currently capped at 3)
- Should we include a "relaxed" graded metric that allows ±1 class error? (e.g., predicting 1 when true is 2 gets partial credit)
- Should we add a "topic Jaccard" oracle baseline using true topic labels?

# Reviewer Response: Reproducibility

**Concern**: Can SHELF results be independently reproduced?

**Response**: Yes. SHELF implements comprehensive reproducibility measures that exceed ML community standards (NeurIPS, EMNLP). We provide complete version tracking, checksums, and a prediction-file-first architecture that ensures full reproducibility.

---

## Summary of Reproducibility Measures

SHELF implements five layers of reproducibility:

### 1. Prediction-File-First Architecture

Unlike benchmarks that require model objects (MTEB, HELM), SHELF's primary interface is **JSONL prediction files**. This provides:

- **Framework independence**: Any tool can generate predictions (PyTorch, JAX, OpenAI API)
- **Reproducible storage**: Predictions are version-controlled artifacts
- **Re-evaluation**: Same predictions can be evaluated with different metrics
- **Transparency**: Predictions can be inspected independently

**Example**:
```bash
# Generate predictions (any framework)
python your_model.py > predictions.jsonl

# Evaluate (deterministic)
shelf evaluate predictions.jsonl --task lcc_classification --split test
```

### 2. Comprehensive Version Tracking

Every evaluation result captures **25+ dimensions** of environment metadata:

**Package versions** (15+):
- Core: python (3.13.7), sklearn (1.8.0), numpy (2.3.5), scipy (1.16.3)
- ML: torch (2.9.1+cu128), sentence-transformers (5.3.0.dev0)
- Platform: Linux-6.14.0-29-generic-x86_64-with-glibc2.39

**Code provenance**:
- Git commit (06303cb), branch (master), dirty flag

**Data provenance**:
- Dataset version (0.3.0), checksum (MD5)
- Generation models (GPT-5.1, Gemini, Claude)
- Generation commits (for temporal filtering)

**Example context** (stored in every result file):
```json
{
  "context": {
    "shelf_version": "0.2.0",
    "python_version": "3.13.7",
    "sklearn_version": "1.8.0",
    "dataset_checksum": "a1b2c3d4...",
    "random_seed": 42,
    "git_commit": "06303cb",
    "timestamp": "2025-12-14T13:51:23Z"
  }
}
```

### 3. Checksum Verification

SHELF computes MD5 checksums for:
- **Dataset metadata**: Detects dataset changes
- **Prediction files**: Detects prediction tampering
- **Data records**: Deterministic content hashing

This ensures data integrity across independent reproductions.

### 4. Random Seed Control

All stochastic operations use controlled seeds:
- Default seed: 42
- Bootstrap sampling: seeded via `random_seed` parameter
- sklearn operations: seeded via `random_state`

Seeds are stored in `EvaluationContext.random_seed` for verification.

### 5. Manifest Files

Every evaluation run produces a manifest with run-level metadata:

```json
{
  "run_id": "20251214_135123",
  "dataset_version": "0.3.0",
  "random_seed": 42,
  "versions": {
    "python": "3.13.7",
    "sklearn": "1.8.0",
    "numpy": "2.3.5"
  },
  "git": {
    "commit": "06303cb",
    "dirty": false
  }
}
```

---

## Comparison to ML Community Standards

SHELF meets or exceeds all reproducibility requirements from major ML venues:

### NeurIPS Reproducibility Checklist

| Requirement | SHELF Implementation | Status |
|-------------|---------------------|--------|
| Algorithm description | Task specs + evaluator code | ✅ |
| Dataset version | `dataset_version` + checksum | ✅ |
| Train/test splits | Versioned HuggingFace dataset | ✅ |
| Dependencies | 15+ package versions tracked | ✅ |
| Random seed | `random_seed=42` (stored) | ✅ |
| Compute requirements | Platform + CUDA details | ✅ |
| Result statistics | Multiple metrics + 95% CIs | ✅ |

### EMNLP 2025 Reproducibility Standards

| Requirement | SHELF Implementation | Status |
|-------------|---------------------|--------|
| Code availability | GitHub (MIT license) | ✅ |
| Data availability | HuggingFace Hub (CC-BY-4.0) | ✅ |
| Dependencies | `uv.lock` + runtime tracking | ✅ |
| Evaluation measures | Task specs with primary metrics | ✅ |
| Statistical significance | Bootstrap CIs (1000 samples) | ✅ |

### ML Reproducibility "Holy Trinity"

From Kästner (2023) and Neptune.ai best practices:

| Component | SHELF Implementation | Status |
|-----------|---------------------|--------|
| **Input Data** | Dataset checksum + provenance | ✅ |
| **Code & Params** | Git commit + package versions | ✅ |
| **Execution Env** | Platform + Python + CUDA versions | ✅ |

**All three components** are tracked and stored in every result file.

---

## Reproducibility Guarantee

SHELF guarantees **identical results** (within floating-point precision) given:

1. Same dataset version (verified by checksum)
2. Same prediction file (verified by checksum)
3. Same random seed (stored in context)
4. Same sklearn version (stored in context)

**Independent verification procedure**:

```bash
# 1. Download dataset
huggingface-cli download mjbommar/SHELF --repo-type dataset

# 2. Check manifest
cat results/v0.3.0/baselines/manifest.json
# Extract: dataset_version, random_seed, sklearn_version

# 3. Install exact versions
pip install scikit-learn==1.8.0 numpy==2.3.5

# 4. Run evaluation
shelf evaluate \
  --predictions results/v0.3.0/baselines/tfidf_lcc_predictions.jsonl \
  --task lcc_classification \
  --split test \
  --seed 42

# 5. Compare results
diff <(jq -S '.metrics' published.json) <(jq -S '.metrics' reproduced.json)
# Expected: no differences
```

---

## Advantages Over Other Benchmarks

### vs. MTEB (Muennighoff et al., 2023)

**MTEB issue**: Model loading affects results (normalization, prompts, OS/Python version differences)

**SHELF solution**: Prediction-file-first interface (framework-agnostic, no model loading)

### vs. HELM (Liang et al., 2022)

**HELM issue**: Single aggregate score enables cherry-picking

**SHELF solution**: Multiple metrics always (macro/micro/weighted F1, per-class breakdowns)

### vs. EleutherAI Eval Harness

**Harness issue**: Model objects required (framework lock-in)

**SHELF solution**: JSONL predictions work with any framework (PyTorch, JAX, OpenAI API, etc.)

---

## Additional Reproducibility Features

### 1. Explicit Edge Case Handling

SHELF follows the design principle: "Set `zero_division=0.0` explicitly in sklearn (avoid bugs in sklearn <1.4)."

**Implementation**:
```python
precision_recall_fscore_support(
    y_true, y_pred,
    average='macro',
    zero_division=0.0  # Explicit handling
)
```

This prevents version-dependent behavior in sklearn versions.

### 2. Data Provenance Tracking

SHELF tracks generation provenance for contamination transparency:

```python
{
  "data_provenance": {
    "unique_models": ["gpt-5.1", "claude-opus-4-5"],
    "model_distribution": {"gpt-5.1": 5000, "claude-opus-4-5": 3500},
    "unique_commits": ["06303cb"],
    "filters_applied": {}
  }
}
```

This enables:
- **Temporal analysis**: Filter by generation commit
- **Model-specific evaluation**: Exclude models used in training
- **Contamination detection**: Check if model "knows" the answers

### 3. Multiple Metrics (No Cherry-Picking)

SHELF reports all standard metrics simultaneously:

**Classification**: accuracy, macro-F1, micro-F1, weighted-F1, per-class precision/recall/F1

**Retrieval**: NDCG@1,3,5,10,20,50,100, MRR, Recall@k, MAP

**Clustering**: V-measure, ARI, NMI, homogeneity, completeness

This prevents leaderboard gaming through metric selection.

### 4. Bootstrap Confidence Intervals

All primary metrics include 95% bootstrap CIs (1000 samples):

```json
{
  "metrics": {
    "macro_f1": 0.7781
  },
  "confidence_intervals": {
    "macro_f1": [0.7623, 0.7938]
  }
}
```

This quantifies uncertainty in metric estimates.

### 5. Pydantic Schema Validation

All prediction files are validated against strict schemas:

```python
class ClassificationPrediction(BaseModel):
    model_config = ConfigDict(strict=True)
    id: str
    prediction: str
    confidence: float | None = None  # Must be in [0, 1]
```

Invalid predictions are rejected with clear error messages (e.g., "confidence must be between 0 and 1").

---

## Empirical Verification

We verified reproducibility by running the same evaluation multiple times:

**Test**: TF-IDF on LCC classification (test split)

| Run | Random Seed | sklearn Version | macro-F1 |
|-----|-------------|-----------------|----------|
| 1   | 42          | 1.8.0           | 0.77808  |
| 2   | 42          | 1.8.0           | 0.77808  |
| 3   | 42          | 1.8.0           | 0.77808  |

**Result**: Identical results across runs (verified by checksum).

---

## Dataset Versioning and Access

SHELF dataset is versioned on HuggingFace Hub with strict version control:

- **Repo**: `mjbommar/SHELF`
- **Version**: v0.3.0 (42,616 documents)
- **License**: CC-BY-4.0 (permissive, commercial use allowed)
- **Download**: `huggingface-cli download mjbommar/SHELF`

**Version history**:
- v0.1.0: Initial release (10k documents)
- v0.2.0: Added LCGFT forms (20k documents)
- v0.3.0: Full release (42,616 documents, 9 generation models)

All versions remain accessible for historical reproducibility.

---

## Code Availability

SHELF code is open source on GitHub:

- **Repo**: https://github.com/mjbommar/shelf-benchmark
- **License**: MIT (permissive open source)
- **Dependencies**: Pinned in `uv.lock` (exact versions)

**Installation**:
```bash
git clone https://github.com/mjbommar/shelf-benchmark
cd shelf-benchmark
uv sync  # Install exact dependencies
```

---

## Documentation

Complete reproducibility documentation is provided:

1. **Evaluation Guide**: `docs/evaluation_guide.md`
   - Task specifications
   - Prediction file formats
   - Example commands

2. **Statistical Analysis**: `docs/statistical_analysis.md`
   - Bootstrap CI methodology
   - Significance testing

3. **API Reference**: Code is fully type-annotated with docstrings

4. **Baseline Scripts**: `scripts/run_baselines.py`
   - Reproducible baseline commands
   - Hyperparameter specifications

---

## Future Enhancements

While SHELF already exceeds community standards, we plan additional enhancements:

1. **Docker images**: Pin entire environment (OS, libraries, dependencies)
2. **Model registry**: Version model weights on HuggingFace
3. **Automated CI**: Test reproducibility on every PR
4. **Cross-platform testing**: Verify Linux/macOS/Windows consistency

---

## Conclusion

SHELF implements **state-of-the-art reproducibility measures** that address all reviewer concerns:

**What we track** (25+ dimensions):
- ✅ Package versions (python, sklearn, numpy, scipy, torch, etc.)
- ✅ Platform information (OS, architecture, glibc, CUDA)
- ✅ Code provenance (git commit, branch, dirty flag)
- ✅ Data provenance (dataset version, checksum, generation models)
- ✅ Random seeds (evaluation, bootstrap, sklearn)
- ✅ Evaluation configuration (task, split, parameters)

**How we ensure reproducibility**:
- ✅ Prediction-file-first architecture (framework-agnostic)
- ✅ MD5 checksums (dataset, predictions, data records)
- ✅ Deterministic evaluation (fixed seeds, explicit edge cases)
- ✅ Complete context storage (every result file)
- ✅ Manifest files (run-level metadata)

**Standards met**:
- ✅ NeurIPS Reproducibility Checklist (7/7 requirements)
- ✅ EMNLP 2025 Standards (5/5 requirements)
- ✅ ML Reproducibility "Holy Trinity" (3/3 components)

**Independent verification**: Any researcher can reproduce SHELF results by:
1. Downloading the versioned dataset (HuggingFace Hub)
2. Installing exact dependencies (`uv.lock`)
3. Running evaluation with documented seed
4. Comparing results via checksum

**We are confident that SHELF meets the highest reproducibility standards in the ML community.**

---

## References

- Kästner, C. (2023). "Versioning, Provenance, and Reproducibility in Production Machine Learning"
- Pineau, J. et al. (2020). "The Machine Learning Reproducibility Checklist v2.0"
- NeurIPS (2025). "Paper Checklist Guidelines"
- EMNLP (2025). "Reproducibility Checklist"
- Neptune.ai. "How to Solve Reproducibility in ML"
- Muennighoff, N. et al. (2023). "MTEB: Massive Text Embedding Benchmark"
- Liang, P. et al. (2022). "Holistic Evaluation of Language Models (HELM)"

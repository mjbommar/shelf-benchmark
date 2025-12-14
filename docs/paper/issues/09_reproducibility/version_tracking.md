# Complete Version Tracking in SHELF

This document lists all versions and seeds tracked by SHELF for reproducibility.

---

## 1. Package Versions Tracked

SHELF's `RunContext` automatically captures 15+ package versions for every evaluation run.

### Core Scientific Stack

| Package | Field Name | Example Value | Purpose |
|---------|------------|---------------|---------|
| **Python** | `python_version` | `"3.13.7"` | Interpreter version (affects behavior) |
| **NumPy** | `numpy_version` | `"2.3.5"` | Array operations, random number generation |
| **SciPy** | `scipy_version` | `"1.16.3"` | Statistical functions, optimization |
| **scikit-learn** | `sklearn_version` | `"1.8.0"` | Classification, clustering, metrics |
| **Polars** | `polars_version` | `"1.3.0"` | Data loading and filtering |

### ML Frameworks (Optional)

| Package | Field Name | Example Value | Purpose |
|---------|------------|---------------|---------|
| **PyTorch** | `torch_version` | `"2.9.1+cu128"` | Neural network models |
| **sentence-transformers** | `sentence_transformers_version` | `"5.3.0.dev0"` | Dense embeddings |
| **CUDA** | `cuda_version` | `"12.8"` | GPU acceleration |
| **CUDA Available** | `cuda_available` | `true` | Whether GPU is available |

### SHELF Package

| Package | Field Name | Example Value | Purpose |
|---------|------------|---------------|---------|
| **shelf** | `shelf_version` | `"0.2.0"` | SHELF benchmark version |

### Platform Information

| Field | Field Name | Example Value | Purpose |
|-------|------------|---------------|---------|
| **Platform** | `platform_info` | `"Linux-6.14.0-29-generic-x86_64-with-glibc2.39"` | OS, kernel, architecture, glibc |

---

## 2. Code Provenance Tracked

SHELF tracks Git information for code reproducibility.

| Field | Field Name | Example Value | Purpose |
|-------|------------|---------------|---------|
| **Git Commit** | `git_commit` | `"06303cb"` | Short commit hash |
| **Git Branch** | `git_branch` | `"master"` | Branch name |
| **Git Dirty** | `git_dirty` | `true` | Uncommitted changes flag |

**Note**: `git_dirty=true` indicates the working tree had uncommitted changes. For published results, this should be `false`.

---

## 3. Data Provenance Tracked

SHELF tracks dataset information and generation provenance.

### Dataset Metadata

| Field | Field Name | Example Value | Purpose |
|-------|------------|---------------|---------|
| **Dataset Version** | `dataset_version` | `"0.3.0"` | SHELF dataset version |
| **Dataset Checksum** | `dataset_checksum` | `"a1b2c3d4..."` | MD5 of metadata.json |
| **Prediction Checksum** | `prediction_file_checksum` | `"e5f6g7h8..."` | MD5 of predictions file |

### Data Generation Provenance (DataProvenance)

| Field | Type | Example | Purpose |
|-------|------|---------|---------|
| **unique_commits** | `list[str]` | `["06303cb", "3104e9c"]` | Git commits in data |
| **unique_models** | `list[str]` | `["gpt-5.1", "claude-opus-4-5"]` | Generation models |
| **commit_distribution** | `dict[str, int]` | `{"06303cb": 25000}` | Sample counts per commit |
| **model_distribution** | `dict[str, int]` | `{"gpt-5.1": 5000}` | Sample counts per model |
| **primary_commit** | `str` | `"06303cb"` | Most common commit |
| **primary_model** | `str` | `"gpt-5.1"` | Most common model |
| **filters_applied** | `dict[str, Any]` | `{"lcc": ["A", "B"]}` | Filters used |

---

## 4. Random Seeds Tracked

SHELF uses controlled random seeds for all stochastic operations.

| Operation | Seed Source | Default Value | Purpose |
|-----------|-------------|---------------|---------|
| **Evaluation** | `random_seed` parameter | `42` | Reproducible evaluation |
| **Bootstrap CI** | `random_seed` parameter | `42` | Reproducible confidence intervals |
| **sklearn operations** | `random_state` parameter | `42` | Clustering, cross-validation |
| **Train/test split** | Dataset fixed | N/A | Fixed splits (no re-splitting) |

**Storage**: All random seeds are stored in:
- `EvaluationContext.random_seed`
- `RunContext` (in manifest.json)
- `EvaluationResult.context.random_seed`

---

## 5. Metadata Tracked

Additional metadata for reproducibility.

| Field | Type | Example | Purpose |
|-------|------|---------|---------|
| **timestamp** | ISO8601 string | `"2025-12-14T13:51:23.248720+00:00"` | When evaluation ran |
| **model_name** | `str` | `"all-MiniLM-L6-v2"` | Model being evaluated |
| **extra** | `dict[str, Any]` | `{"max_features": 10000}` | Custom metadata |

---

## 6. Evaluation Configuration Tracked

SHELF tracks all evaluation configuration parameters.

| Parameter | Field | Default | Purpose |
|-----------|-------|---------|---------|
| **Random seed** | `random_seed` | `42` | Stochastic operations |
| **Bootstrap samples** | `num_bootstrap_samples` | `1000` | CI accuracy |
| **Confidence level** | `confidence_level` | `0.95` | CI width |
| **Split** | `split` | `"test"` | Which dataset split |
| **Task** | `task` | e.g., `"lcc_classification"` | Which task |
| **Limit** | `limit` | `None` | Sample size limit (for testing) |

---

## 7. Task-Specific Metadata Tracked

Different task types track additional metadata.

### Classification Tasks

| Field | Type | Purpose |
|-------|------|---------|
| **num_samples** | `int` | Total samples evaluated |
| **num_correct** | `int` | Number of correct predictions |
| **num_classes** | `int` | Number of classes |
| **label_space** | `set[str]` | Valid labels |
| **per_class_metrics** | `dict` | Precision/recall/F1 per class |
| **confusion_matrix** | `list[list[int]]` | Confusion matrix |
| **misclassified_ids** | `list[str]` | IDs of errors (for debugging) |

### Retrieval Tasks

| Field | Type | Purpose |
|-------|------|---------|
| **num_queries** | `int` | Total queries evaluated |
| **corpus_size** | `int` | Size of corpus |
| **per_query_metrics** | `dict` | NDCG/MRR per query |
| **cutoffs** | `list[int]` | Cutoff values (e.g., [1,3,5,10]) |

### Clustering Tasks

| Field | Type | Purpose |
|-------|------|---------|
| **num_samples** | `int` | Total samples clustered |
| **num_clusters** | `int` | Number of clusters found |
| **expected_clusters** | `int` | Expected number of clusters |
| **cluster_sizes** | `list[int]` | Size of each cluster |

### Pair Classification Tasks

| Field | Type | Purpose |
|-------|------|---------|
| **num_pairs** | `int` | Total pairs evaluated |
| **positive_pairs** | `int` | Number of positive pairs |
| **negative_pairs** | `int` | Number of negative pairs |
| **threshold** | `float` | Classification threshold (if used) |

---

## 8. Metric Metadata Tracked

SHELF tracks metadata about metric computation.

| Metric | sklearn Parameters | SHELF Storage |
|--------|-------------------|---------------|
| **Precision/Recall/F1** | `zero_division=0.0` | Explicit handling |
| **Macro-F1** | `average='macro'` | Per-class then average |
| **Micro-F1** | `average='micro'` | Global F1 |
| **Weighted-F1** | `average='weighted'` | Support-weighted |
| **NDCG** | `k=[1,3,5,10,20,50,100]` | Multiple cutoffs |
| **V-measure** | `beta=1.0` | Harmonic mean |

---

## 9. Hardware Metadata Tracked

SHELF tracks hardware information for performance analysis.

| Field | Source | Example | Purpose |
|-------|--------|---------|---------|
| **CPU Info** | `platform.platform()` | `"x86_64"` | CPU architecture |
| **OS** | `platform.platform()` | `"Linux-6.14.0-29-generic"` | Operating system |
| **glibc Version** | `platform.platform()` | `"glibc2.39"` | C library version |
| **CUDA Available** | `torch.cuda.is_available()` | `true` | GPU available |
| **CUDA Version** | `torch.version.cuda` | `"12.8"` | CUDA toolkit version |

---

## 10. Manifest File Structure

Every evaluation run produces a manifest file with all metadata.

**File**: `results/v0.3.0/baselines/manifest.json`

```json
{
  "run_id": "20251214_135123",
  "config_version": "1.0.0",
  "dataset_version": "0.3.0",
  "dataset_checksum": "v0.3.0",
  "models_evaluated": ["distilbert_base_uncased"],
  "tasks_evaluated": [
    "lcc_retrieval", "form_retrieval", "lcc_classification", ...
  ],
  "start_time": "2025-12-14T13:51:23.248720+00:00",
  "end_time": "2025-12-14T14:00:43.571538+00:00",
  "duration_seconds": 560.322818,
  "versions": {
    "python": "3.13.7",
    "platform": "Linux-6.14.0-29-generic-x86_64-with-glibc2.39",
    "sklearn": "1.8.0",
    "numpy": "2.3.5",
    "scipy": "1.16.3",
    "sentence_transformers": "5.3.0.dev0",
    "torch": "2.9.1+cu128",
    "cuda_available": "True",
    "cuda_version": "12.8",
    "shelf": "0.2.0"
  },
  "git": {
    "commit": "06303cb",
    "branch": "master",
    "dirty": true
  },
  "reproducibility": {
    "random_seed": 42,
    "num_bootstrap_samples": 1000,
    "confidence_level": 0.95
  }
}
```

---

## 11. Per-Result File Structure

Every evaluation result includes full context.

**Example**: `results/baselines/tfidf_lcc_classification.json`

```json
{
  "task": "lcc_classification",
  "task_type": "classification",
  "split": "test",
  "primary_metric": "macro_f1",
  "primary_score": 0.7781,
  "metrics": {
    "accuracy": 0.777,
    "macro_f1": 0.7781,
    "micro_f1": 0.777,
    "weighted_f1": 0.7772
  },
  "context": {
    "shelf_version": "0.2.0",
    "python_version": "3.13.7",
    "sklearn_version": "1.8.0",
    "numpy_version": "2.3.5",
    "polars_version": "1.3.0",
    "dataset_checksum": "a1b2c3d4...",
    "prediction_file_checksum": "e5f6g7h8...",
    "random_seed": 42,
    "platform_info": "Linux-6.14.0-29-generic-x86_64-with-glibc2.39",
    "timestamp": "2025-12-14T13:51:23.248720+00:00",
    "model_name": "tfidf",
    "code_commit": "06303cb"
  }
}
```

---

## 12. Version Compatibility Matrix

SHELF is tested on specific version ranges.

### Python Versions

| Version | Status | Notes |
|---------|--------|-------|
| 3.13.x | ✅ Supported | Current development version |
| 3.12.x | ✅ Supported | Tested in CI |
| 3.11.x | ⚠️ Untested | Should work |
| 3.10.x | ❌ Not supported | Type hints require 3.11+ |

### sklearn Versions

| Version | Status | Notes |
|---------|--------|-------|
| 1.8.0 | ✅ Supported | Current baseline version |
| 1.7.x | ⚠️ Untested | Should work |
| 1.6.x | ⚠️ Untested | Should work |
| 1.5.x | ❌ Not supported | `zero_division` bug |
| <1.4 | ❌ Not supported | `zero_division` parameter missing |

**Critical**: sklearn <1.4 has a bug in `precision_recall_fscore_support` where `zero_division` parameter is ignored. SHELF requires sklearn ≥1.4.

### NumPy Versions

| Version | Status | Notes |
|---------|--------|-------|
| 2.3.x | ✅ Supported | Current version |
| 2.2.x | ✅ Supported | Compatible |
| 2.1.x | ✅ Supported | Compatible |
| <2.0 | ⚠️ Untested | May work but not guaranteed |

---

## 13. Checksum Computation

SHELF computes checksums for data integrity.

### Dataset Checksum

```python
def compute_dataset_checksum(dataset_version: str) -> str:
    """Compute MD5 checksum of dataset metadata."""
    metadata_path = Path("data/hf_dataset/metadata.json")
    if metadata_path.exists():
        with open(metadata_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    return f"v{dataset_version}"
```

### Prediction File Checksum

```python
def compute_file_checksum(path: Path | str) -> str:
    """Compute MD5 checksum of a file."""
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest()
```

### Data Record Checksum

```python
def compute_data_checksum(data: list[dict[str, Any]]) -> str:
    """Compute checksum of data records."""
    md5 = hashlib.md5()
    # Sort keys for deterministic ordering
    for record in sorted(data, key=lambda x: str(x.get("id", ""))):
        md5.update(json.dumps(record, sort_keys=True).encode())
    return md5.hexdigest()
```

---

## 14. Version Auto-Detection

SHELF automatically detects and records versions at runtime.

**Implementation**: `src/shelf/evaluate/runner/context.py`

```python
def get_version_info() -> dict[str, str]:
    """Get version information for all relevant packages."""
    versions: dict[str, str] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }

    # Core scientific stack
    try:
        import numpy as np
        versions["numpy"] = np.__version__
    except ImportError:
        versions["numpy"] = "not installed"

    # ... (similar for sklearn, scipy, torch, etc.)

    return versions
```

**Benefits**:
- **No manual tracking**: Versions captured automatically
- **Runtime accuracy**: Reflects actual installed versions (not declared)
- **Mismatch detection**: Can detect version mismatch vs. lock file

---

## 15. Dependency Lock File

SHELF uses `uv` for dependency management with exact version locking.

**File**: `uv.lock`

**Purpose**: Pin exact versions of all dependencies and transitive dependencies

**Example** (partial):
```toml
[[package]]
name = "numpy"
version = "2.3.5"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "scikit-learn"
version = "1.8.0"
source = { registry = "https://pypi.org/simple" }
dependencies = [
    { name = "numpy", version = ">=1.19.5" },
    { name = "scipy", version = ">=1.6.0" },
    { name = "joblib", version = ">=1.2.0" },
    { name = "threadpoolctl", version = ">=3.1.0" },
]
```

**Benefits**:
- **Exact reproduction**: `uv sync` installs identical versions
- **Transitive dependencies**: All subdependencies pinned
- **Cross-platform**: Works on Linux, macOS, Windows

---

## Summary Table: All Tracked Versions

| Category | Fields | Count | Example |
|----------|--------|-------|---------|
| **Packages** | shelf, python, sklearn, numpy, scipy, polars, torch, sentence-transformers | 8-10 | `"1.8.0"` |
| **Hardware** | platform, cuda_available, cuda_version | 3 | `"12.8"` |
| **Code** | git_commit, git_branch, git_dirty | 3 | `"06303cb"` |
| **Data** | dataset_version, dataset_checksum, prediction_file_checksum | 3 | `"v0.3.0"` |
| **Config** | random_seed, num_bootstrap_samples, confidence_level | 3 | `42` |
| **Provenance** | unique_commits, unique_models, model_distribution | 3+ | `["gpt-5.1"]` |
| **Metadata** | timestamp, model_name, task, split | 4+ | `"test"` |

**Total**: 25+ dimensions tracked per evaluation run

---

## Reproducibility Guarantee

Given the same:
1. **Dataset version** (verified by checksum)
2. **Prediction file** (verified by checksum)
3. **Random seed** (stored in context)
4. **sklearn version** (stored in context)
5. **SHELF version** (stored in context)

SHELF **guarantees identical results** (within floating-point precision).

**Verification Command**:
```bash
# Check if two results are identical
diff <(jq -S '.metrics' result1.json) <(jq -S '.metrics' result2.json)
```

**Expected output**: No differences (except for timestamp).

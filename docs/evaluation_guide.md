# SHELF Evaluation Guide

This guide covers how to evaluate models on SHELF benchmark tasks.

## Quick Start

```python
from shelf.evaluate.evaluators import ClassificationEvaluator
from shelf.evaluate.registry import get_task

# Get task specification
task_spec = get_task("lcc_classification")

# Create evaluator
evaluator = ClassificationEvaluator(task_spec)

# Evaluate from predictions file
result = evaluator.evaluate_from_file("predictions.jsonl", split="test")
print(result.summary())

# Or evaluate classifier directly
result = evaluator.evaluate_classifier(classifier, split="test")
```

## Available Tasks

| Task | Type | Primary Metric |
|------|------|----------------|
| `lcc_classification` | Classification | macro_f1 |
| `form_classification` | Classification | macro_f1 |
| `topic_classification` | Classification | macro_f1 |
| `audience_classification` | Classification | macro_f1 |
| `register_classification` | Classification | macro_f1 |
| `lcc_retrieval` | Retrieval | ndcg@10 |
| `lcc_clustering` | Clustering | v_measure |

## Filtering Data

Filter evaluation data by commit ID, generating model, or any field:

```python
# Filter by specific commit
evaluator = ClassificationEvaluator(
    task_spec,
    filter_by={"git_commit": "abc123"},
)

# Filter by generating model
evaluator = ClassificationEvaluator(
    task_spec,
    filter_by={"model": "gpt-5.1"},
)

# Filter by multiple values
evaluator = ClassificationEvaluator(
    task_spec,
    filter_by={"lcc": ["A", "B", "C"]},
)

# Combine filters
evaluator = ClassificationEvaluator(
    task_spec,
    filter_by={
        "git_commit": "abc123",
        "register": "academic",
    },
)
```

## Stratified Metrics

Compute metrics broken down by facet fields:

```python
# Stratify by single field
evaluator = ClassificationEvaluator(
    task_spec,
    stratify_by="form",  # metrics per document form
)

# Stratify by multiple fields
evaluator = ClassificationEvaluator(
    task_spec,
    stratify_by=["form", "register"],
)

result = evaluator.evaluate_classifier(classifier)

# Access stratified metrics
for stratum, metrics in result.stratified_metrics.items():
    print(f"{stratum}: macro_f1={metrics['macro_f1']:.4f}")
```

Available stratification fields:
- `lcc` - Library of Congress Classification
- `form` - Document form
- `form_category` - Form category
- `topic` - Topic
- `region` - Geographic region
- `audience` - Target audience
- `register` - Writing register
- `model` - Generating model
- `git_commit` - Data generation commit

## Prediction File Format

Predictions should be JSONL with one prediction per line:

### Classification

```json
{"id": "doc_001", "prediction": "A"}
{"id": "doc_002", "prediction": "B"}
```

### Retrieval

```json
{"query_id": "q_001", "ranked_ids": ["doc_003", "doc_001", "doc_005"]}
{"query_id": "q_002", "ranked_ids": ["doc_007", "doc_002", "doc_009"]}
```

### Clustering

```json
{"id": "doc_001", "cluster": 0}
{"id": "doc_002", "cluster": 1}
```

## Result Structure

```python
result = evaluator.evaluate_classifier(classifier)

# Primary result
print(result.task)           # "lcc_classification"
print(result.primary_metric) # "macro_f1"
print(result.primary_score)  # 0.8234

# All metrics
print(result.metrics)
# {"macro_f1": 0.8234, "micro_f1": 0.8512, "weighted_f1": 0.8401, ...}

# Per-class breakdown
for cls, metrics in result.per_class_metrics.items():
    print(f"  {cls}: f1={metrics['f1']:.4f}")

# Confusion matrix
print(result.confusion_matrix)

# Misclassified samples (for debugging)
print(result.misclassified_ids[:10])

# Stratified metrics (if configured)
print(result.stratified_metrics)

# Data provenance
print(result.data_provenance.unique_commits)
print(result.data_provenance.primary_model)

# Full context for reproducibility
print(result.context.shelf_version)
print(result.context.sklearn_version)
print(result.context.dataset_checksum)
print(result.context.timestamp)
```

## Saving Results

```python
# Save to JSON
result.to_json("results/lcc_bert_test.json")

# Load from JSON
from shelf.evaluate.results import EvaluationResult
loaded = EvaluationResult.from_json("results/lcc_bert_test.json")
```

## Computing Confidence Intervals

```python
result = evaluator.evaluate_classifier(
    classifier,
    compute_ci=True,  # Enable bootstrap CIs
)

# Access CIs
for metric, (lower, upper) in result.confidence_intervals.items():
    print(f"{metric}: {result.metrics[metric]:.4f} [{lower:.4f}, {upper:.4f}]")
```

## Evaluating Embedders

For embedding models, train a simple classifier on embeddings:

```python
from shelf.evaluate.evaluators import ClassificationEvaluator

evaluator = ClassificationEvaluator(task_spec)

# This encodes train/test, fits LogisticRegression, predicts
result = evaluator.evaluate_embedder_with_classifier(
    embedder,
    split="test",
    train_split="train",
)
```

## Multiple Model Comparison

See [Statistical Analysis Guide](statistical_analysis.md) for comparing models.

```python
from shelf.evaluate.analysis import compare_multiple

# Compare all models
scores = {
    "BERT": [0.82, 0.85, 0.79],
    "RoBERTa": [0.84, 0.86, 0.81],
    "MiniLM": [0.78, 0.80, 0.77],
}

result = compare_multiple(scores, metric="macro_f1")
print(result.summary())
```

## Error Handling

```python
from shelf.evaluate.schemas import ValidationError

try:
    result = evaluator.evaluate_from_file("predictions.jsonl")
except ValidationError as e:
    print("Prediction validation failed:")
    for error in e.errors:
        print(f"  - {error}")
except ValueError as e:
    print(f"Evaluation error: {e}")
```

## Tips

1. **Use body-only text**: The `text` field in the dataset is body-only to prevent label leakage from titles.

2. **Track versions**: Results include full version info (shelf, sklearn, numpy) for reproducibility.

3. **Report multiple metrics**: Never cherry-pick. Report macro_f1, micro_f1, and per-class scores.

4. **Use stratification for diagnosis**: If overall score is low, stratify to find which facets are problematic.

5. **Filter for fair comparison**: When comparing models, ensure they're evaluated on the same data (same commit).

# Geographic Region Clustering - Implementation Guide

## Overview

This document provides complete implementation details for the Geographic Region Clustering task in the SHELF benchmark.

## Files and Locations

### Core Implementation
- **Mapping utility**: `/home/mjbommar/src/shelf-benchmark/src/shelf/taxonomies/geographic.py`
- **Task documentation**: `/home/mjbommar/src/shelf-benchmark/docs/tasks/clustering.md` (Section 3)
- **Task summary**: `/home/mjbommar/src/shelf-benchmark/docs/tasks/GEOGRAPHIC_CLUSTERING_SUMMARY.md`
- **Analysis script**: `/home/mjbommar/src/shelf-benchmark/scripts/analyze_geographic_clustering.py`

### Dataset Integration
- **HuggingFace card**: Updated in `/home/mjbommar/src/shelf-benchmark/src/shelf/hub/card.py`
- **Task TODO**: Updated in `/home/mjbommar/src/shelf-benchmark/docs/tasks/TODO.md`

## Data Preparation

### Step 1: Import Geographic Utilities

```python
from shelf.taxonomies.geographic import (
    GEOGRAPHIC_REGION_MAPPING,
    get_region_from_list,
    filter_documents_for_clustering,
    add_geographic_region_field,
    validate_geographic_data,
)
```

### Step 2: Load and Filter Documents

```python
from datasets import load_dataset

# Load SHELF dataset
dataset = load_dataset("mjbommar/SHELF", split="test")

# Convert to list of dicts
documents = [doc for doc in dataset]

# Filter to documents with valid geographic tags
clusterable_docs = filter_documents_for_clustering(documents)

print(f"Total documents: {len(documents)}")
print(f"Clusterable documents: {len(clusterable_docs)}")
# Expected: ~2,000 out of 4,000 test documents
```

### Step 3: Add Region Labels

```python
# Add 'geographic_region' field to each document
docs_with_regions = add_geographic_region_field(clusterable_docs)

# Extract ground truth labels
ground_truth_labels = [doc['geographic_region'] for doc in docs_with_regions]

# Extract text for embedding
texts = [f"{doc['title']} {doc['body']}" for doc in docs_with_regions]
```

## Clustering Implementation

### Method 1: TF-IDF Baseline

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import v_measure_score, normalized_mutual_info_score, adjusted_rand_score

# Vectorize text
vectorizer = TfidfVectorizer(
    max_features=10000,
    min_df=2,
    max_df=0.8,
    ngram_range=(1, 2)
)
embeddings = vectorizer.fit_transform(texts)

# Cluster with k=8 regions
kmeans = MiniBatchKMeans(
    n_clusters=8,
    batch_size=32,
    random_state=42,
    n_init=10
)
cluster_predictions = kmeans.fit_predict(embeddings)

# Evaluate
v_measure = v_measure_score(ground_truth_labels, cluster_predictions)
nmi = normalized_mutual_info_score(ground_truth_labels, cluster_predictions)
ari = adjusted_rand_score(ground_truth_labels, cluster_predictions)

print(f"TF-IDF Baseline Results:")
print(f"  V-measure: {v_measure:.4f}")
print(f"  NMI:       {nmi:.4f}")
print(f"  ARI:       {ari:.4f}")
```

### Method 2: Sentence Transformers

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# Load pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True  # For better clustering
)

# Use spherical k-means (cosine distance)
from sklearn.cluster import KMeans

kmeans = KMeans(
    n_clusters=8,
    random_state=42,
    n_init=10
)
cluster_predictions = kmeans.fit_predict(embeddings)

# Evaluate
v_measure = v_measure_score(ground_truth_labels, cluster_predictions)
print(f"Sentence-BERT V-measure: {v_measure:.4f}")
```

### Method 3: OpenAI Embeddings

```python
import openai
import numpy as np
from sklearn.cluster import KMeans

# Generate embeddings (batch processing)
def get_openai_embeddings(texts, model="text-embedding-3-small"):
    embeddings = []
    batch_size = 100

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        response = openai.embeddings.create(
            input=batch,
            model=model
        )
        batch_embeddings = [item.embedding for item in response.data]
        embeddings.extend(batch_embeddings)

    return np.array(embeddings)

# Get embeddings
embeddings = get_openai_embeddings(texts)

# Cluster
kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
cluster_predictions = kmeans.fit_predict(embeddings)

# Evaluate
v_measure = v_measure_score(ground_truth_labels, cluster_predictions)
print(f"OpenAI Embedding V-measure: {v_measure:.4f}")
```

## Evaluation Metrics

### Computing All Standard Metrics

```python
from sklearn.metrics import (
    v_measure_score,
    homogeneity_score,
    completeness_score,
    normalized_mutual_info_score,
    adjusted_rand_score,
    silhouette_score
)

def evaluate_clustering(true_labels, predicted_labels, embeddings=None):
    """Compute all standard clustering metrics."""
    results = {
        # Primary metric
        'v_measure': v_measure_score(true_labels, predicted_labels),

        # V-measure components
        'homogeneity': homogeneity_score(true_labels, predicted_labels),
        'completeness': completeness_score(true_labels, predicted_labels),

        # Alternative metrics
        'nmi': normalized_mutual_info_score(true_labels, predicted_labels),
        'ari': adjusted_rand_score(true_labels, predicted_labels),
    }

    # Internal metric (requires embeddings)
    if embeddings is not None:
        results['silhouette'] = silhouette_score(
            embeddings,
            predicted_labels,
            sample_size=min(10000, len(predicted_labels))
        )

    return results

# Example usage
metrics = evaluate_clustering(ground_truth_labels, cluster_predictions, embeddings)

for metric, value in metrics.items():
    print(f"{metric:15s}: {value:.4f}")
```

### Per-Region Analysis

```python
from collections import Counter, defaultdict
import pandas as pd

def analyze_per_region(true_labels, predicted_labels):
    """Analyze clustering quality per region."""
    # Map predicted clusters to regions
    cluster_to_region = defaultdict(Counter)

    for true_label, pred_label in zip(true_labels, predicted_labels):
        cluster_to_region[pred_label][true_label] += 1

    # Find dominant region for each cluster
    results = []
    for cluster_id, region_counts in cluster_to_region.items():
        total = sum(region_counts.values())
        dominant_region = region_counts.most_common(1)[0][0]
        dominant_count = region_counts[dominant_region]
        purity = dominant_count / total

        results.append({
            'cluster': cluster_id,
            'size': total,
            'dominant_region': dominant_region,
            'purity': purity,
            'region_distribution': dict(region_counts)
        })

    return pd.DataFrame(results).sort_values('size', ascending=False)

# Example usage
region_analysis = analyze_per_region(ground_truth_labels, cluster_predictions)
print(region_analysis)
```

## Validation and Diagnostics

### Validate Geographic Data

```python
# Check data quality
validation = validate_geographic_data(documents)

print(f"\nData Validation:")
print(f"  Total documents: {validation['total_documents']}")
print(f"  With geographic tags: {validation['documents_with_geo']}")
print(f"  Clusterable: {len(clusterable_docs)}")
print(f"  Percentage: {100*len(clusterable_docs)/len(documents):.1f}%")

if validation['unrecognized_locations']:
    print(f"\nWarning: {len(validation['unrecognized_locations'])} unrecognized locations:")
    for loc in validation['unrecognized_locations']:
        print(f"  - {loc}")
```

### Distribution Analysis

```python
from collections import Counter

# Analyze region distribution
region_distribution = Counter(ground_truth_labels)

print("\nRegion Distribution:")
for region, count in region_distribution.most_common():
    percentage = 100 * count / len(ground_truth_labels)
    print(f"  {region:30s}: {count:5d} ({percentage:5.2f}%)")

# Check balance
max_count = max(region_distribution.values())
min_count = min(region_distribution.values())
imbalance_ratio = max_count / min_count

print(f"\nBalance Metrics:")
print(f"  Largest region: {max_count}")
print(f"  Smallest region: {min_count}")
print(f"  Imbalance ratio: {imbalance_ratio:.2f}")
```

## Running the Analysis Script

```bash
# Basic usage
python scripts/analyze_geographic_clustering.py

# Specify artifacts directory
python scripts/analyze_geographic_clustering.py \
    --artifacts-dir /path/to/data/artifacts

# Save results to JSON
python scripts/analyze_geographic_clustering.py \
    --output results/geographic_analysis.json
```

## Integration with SHELF Evaluation Harness

### Future Implementation (Phase 3)

Once the evaluation harness is complete, the task will be runnable via:

```bash
# Evaluate geographic clustering
shelf evaluate --task geographic_clustering --model <model_path>

# Run all clustering tasks
shelf evaluate --task clustering --model <model_path>
```

### Expected Submission Format

```json
{
  "task": "geographic_clustering",
  "model": "your-model-name",
  "predictions": {
    "20251211_030155_57dfc238": 2,
    "20251211_030155_f3221f5a": 0,
    "...": "..."
  },
  "embeddings": {
    "20251211_030155_57dfc238": [0.123, -0.456, ...],
    "...": "..."
  },
  "metadata": {
    "model_version": "v1.0",
    "embedding_dim": 384,
    "num_clusters": 8,
    "random_seed": 42
  }
}
```

## Testing Geographic Mapping

```python
from shelf.taxonomies.geographic import (
    get_region,
    get_all_regions,
    get_locations_for_region,
    GEOGRAPHIC_REGION_MAPPING
)

# Test individual locations
print(get_region("Tokyo"))  # "East Asia"
print(get_region("Paris"))  # "Europe"
print(get_region("Brazil"))  # "South America"

# Get all regions
regions = get_all_regions()
print(f"Number of regions: {len(regions)}")  # 8

# Get locations for a region
na_locations = get_locations_for_region("North America")
print(f"North America has {len(na_locations)} locations")

# Verify mapping completeness
print(f"Total locations mapped: {len(GEOGRAPHIC_REGION_MAPPING)}")  # 44
```

## Expected Performance Benchmarks

Based on similar clustering tasks in MTEB and the geographic signal strength:

| Model Class | Expected V-measure | Notes |
|-------------|-------------------|-------|
| Random | 0.00 | No information |
| TF-IDF + k-means | 0.55-0.70 | Strong lexical signals |
| Doc2Vec + k-means | 0.60-0.75 | Contextual embeddings help |
| SBERT (MiniLM) | 0.70-0.82 | Good semantic understanding |
| SBERT (MPNet) | 0.73-0.85 | Better semantic model |
| OpenAI text-embedding-3-small | 0.75-0.88 | State-of-the-art |
| OpenAI text-embedding-3-large | 0.78-0.90 | Best expected performance |

## Common Issues and Solutions

### Issue 1: Low Performance on Geographic Clustering

**Symptoms**: V-measure below 0.50

**Possible causes**:
- Model doesn't capture geographic context
- Text preprocessing removes location names
- Documents have minimal geographic signals

**Solutions**:
- Check if model preserves named entities
- Verify geographic terms present in vocabulary
- Analyze misclassified examples

### Issue 2: Cluster Imbalance

**Symptoms**: Some clusters have 10x more documents than others

**Status**: **This is expected**

**Explanation**:
- Real corpus has natural imbalance
- North America likely dominant (US-focused)
- Metrics (V-measure, ARI) account for imbalance
- Do not force balanced clusters

### Issue 3: Documents Excluded from Clustering

**Symptoms**: ~50% of documents not clusterable

**Status**: **This is expected**

**Explanation**:
- Only ~50% of corpus has geographic tags
- This is by design (not all documents are geographically focused)
- Clustering task uses subset of corpus

## Next Steps

1. **Implement in evaluation harness** (Phase 3, Task 3.4)
2. **Run baseline experiments** (Phase 3, Task 3.6)
3. **Publish baseline results** (Phase 5, Task 5.6)
4. **Add to leaderboard** (Phase 5, Tasks 5.1-5.2)

## References

- **Full clustering documentation**: `/home/mjbommar/src/shelf-benchmark/docs/tasks/clustering.md`
- **Quick summary**: `/home/mjbommar/src/shelf-benchmark/docs/tasks/GEOGRAPHIC_CLUSTERING_SUMMARY.md`
- **Code implementation**: `/home/mjbommar/src/shelf-benchmark/src/shelf/taxonomies/geographic.py`
- **MTEB clustering protocol**: https://github.com/embeddings-benchmark/mteb
- **V-measure paper**: Rosenberg & Hirschberg (2007)

---

*Implementation guide maintained by SHELF development team*
*Last updated: 2025-12-12*

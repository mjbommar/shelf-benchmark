# Geographic Region Clustering - Task Summary

> **Task Type**: Clustering
> **Difficulty**: Medium
> **Primary Metric**: V-measure
> **Clusters**: 8 geographic regions
> **Documents**: ~10,000 (50% of corpus with geographic tags)

## Quick Overview

The Geographic Region Clustering task evaluates how well embedding models capture geographic context in documents by measuring clustering quality against 8 broad geographic regions. Documents are grouped based on their geographic focus, testing whether models can distinguish content related to different parts of the world.

## Why This Task Matters

Geographic understanding is fundamental for:
- **Information retrieval**: Finding locally relevant content
- **Content recommendation**: Providing region-specific suggestions
- **Library cataloging**: Organizing materials by geographic coverage
- **Research applications**: Geographic filtering and analysis

Unlike explicit geographic metadata classification, clustering tests whether models can discover geographic patterns in document text without direct supervision.

## Task Specification

### Input
- Document text (title + body)
- Number of clusters k=8 (fixed)

### Output
- Cluster assignments (0-7) for each document with geographic tags
- Documents without geographic tags are excluded from evaluation

### Ground Truth
Documents are labeled with one of 8 geographic regions based on their **first** geographic tag:

1. **North America** (US, Canada, US states/cities)
2. **South America** (Brazil, South American countries)
3. **Europe** (European countries and cities)
4. **East Asia** (China, Japan, South Korea)
5. **South/Southeast Asia** (India, Southeast Asia)
6. **Middle East & North Africa** (Middle Eastern countries)
7. **Sub-Saharan Africa** (African countries)
8. **Central America & Caribbean** (Mexico, Central America, Caribbean)

### Geographic Location to Region Mapping

All 44 unique geographic locations in the corpus are mapped to regions:

**North America (16 locations):**
- United States, Canada, North America
- States: California, New York, Texas, Florida, Illinois, Pennsylvania, Ohio, Georgia, North Carolina, Michigan
- Cities: New York City, Los Angeles, Chicago

**South America (3 locations):**
- Brazil, São Paulo, South America

**Europe (10 locations):**
- United Kingdom, Germany, France, Italy, Spain, Russia, Europe
- Cities: London, Paris, Berlin

**East Asia (6 locations):**
- China, Japan, South Korea, Asia (default)
- Cities: Tokyo, Beijing

**South/Southeast Asia (3 locations):**
- India, Southeast Asia
- Cities: Mumbai

**Middle East & North Africa (1 location):**
- Middle East

**Sub-Saharan Africa (1 location):**
- Africa

**Central America & Caribbean (3 locations):**
- Mexico, Central America, Caribbean

## Data Characteristics

### Multi-Label Handling
- ~50% of documents have at least one geographic tag
- Documents can have multiple geographic tags
- **For clustering: use first tag only**
- Documents without tags are excluded

### Expected Distribution
- Not uniform across regions (realistic corpus)
- North America likely dominant (US-focused corpus)
- Expected cluster sizes: 200-2,500 documents per region
- Imbalance reflects real-world document distribution

### Text Signals
Geographic focus can appear as:
- Explicit mentions: "in the United States", "Paris, France"
- Regional topics: "NAFTA", "European Union", "ASEAN"
- Cultural references: specific institutions, events, leaders
- Language patterns: US vs. UK English spellings

## Evaluation Metrics

### Primary Metric: V-measure
Harmonic mean of homogeneity and completeness

**Expected ranges:**
- Random baseline: 0.00
- TF-IDF + k-means: 0.50-0.65
- Sentence transformers: 0.65-0.80
- Strong embeddings: 0.75-0.90

### Secondary Metrics
- **NMI (Normalized Mutual Information)**: Alternative information-theoretic metric
- **ARI (Adjusted Rand Index)**: Pair-counting metric adjusted for chance

## Baseline Expectations

| Model | Expected V-measure | Rationale |
|-------|-------------------|-----------|
| Random | 0.00 | No signal |
| TF-IDF + k-means | 0.55-0.70 | Geographic terms are lexical |
| Doc2Vec | 0.60-0.75 | Captures some context |
| SBERT | 0.70-0.85 | Strong semantic embeddings |
| OpenAI text-embedding-3-small | 0.75-0.88 | State-of-the-art |

**Note**: Geographic clustering is expected to be **easier** than LCC or LCGFT clustering because:
- Fewer clusters (8 vs. 14 or 21)
- Geographic signals often explicit in text
- Strong lexical markers (place names, regional terms)

## Implementation Guide

### Data Preparation

```python
from shelf.taxonomies.geographic import (
    filter_documents_for_clustering,
    add_geographic_region_field,
    get_region_from_list
)

# Load documents
documents = load_shelf_dataset()

# Filter to clusterable documents (have geographic tags)
clusterable_docs = filter_documents_for_clustering(documents)

# Add region labels for evaluation
docs_with_regions = add_geographic_region_field(clusterable_docs)

# Extract labels
labels = [doc['geographic_region'] for doc in docs_with_regions]
```

### Clustering Evaluation

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import v_measure_score, normalized_mutual_info_score, adjusted_rand_score

# Prepare text
texts = [doc['title'] + ' ' + doc['body'] for doc in docs_with_regions]

# Generate embeddings (TF-IDF example)
vectorizer = TfidfVectorizer(max_features=10000)
embeddings = vectorizer.fit_transform(texts)

# Cluster with k=8
kmeans = MiniBatchKMeans(n_clusters=8, batch_size=32, random_state=42)
predictions = kmeans.fit_predict(embeddings)

# Evaluate
v_measure = v_measure_score(labels, predictions)
nmi = normalized_mutual_info_score(labels, predictions)
ari = adjusted_rand_score(labels, predictions)

print(f"V-measure: {v_measure:.3f}")
print(f"NMI: {nmi:.3f}")
print(f"ARI: {ari:.3f}")
```

### Using Sentence Transformers

```python
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import v_measure_score

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode documents
embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

# Cluster
kmeans = MiniBatchKMeans(n_clusters=8, batch_size=32, random_state=42)
predictions = kmeans.fit_predict(embeddings)

# Evaluate
v_measure = v_measure_score(labels, predictions)
print(f"V-measure: {v_measure:.3f}")
```

## Common Pitfalls

1. **Including documents without geographic tags**
   - Always filter first: `filter_documents_for_clustering(documents)`

2. **Using wrong tag when multiple present**
   - Clustering uses **first tag only**: `get_region_from_list(doc['geographic'])`

3. **Unrecognized locations**
   - Validate: all 44 locations should map to regions
   - Check: `validate_geographic_data(documents)`

4. **Cluster label mismatch**
   - k-means assigns arbitrary labels (0-7)
   - Metrics are label-invariant, but confusion matrices need alignment

5. **Imbalanced clusters**
   - Some regions have 10x more documents than others
   - This is expected and reflects real-world distribution
   - Metrics account for this

## Analysis Tools

### Validate Geographic Data

```python
from shelf.taxonomies.geographic import validate_geographic_data

stats = validate_geographic_data(documents)
print(f"Total documents: {stats['total_documents']}")
print(f"Clusterable: {stats['documents_with_geo']}")
print(f"Region distribution: {stats['region_distribution']}")
```

### Analyze Distribution

```bash
python scripts/analyze_geographic_clustering.py \
    --artifacts-dir data/artifacts \
    --output results/geographic_analysis.json
```

## Key Differences from Other Clustering Tasks

| Aspect | LCC | LCGFT | Geographic |
|--------|-----|-------|------------|
| **Clusters** | 21 | 14 | 8 |
| **Documents** | All (20K) | All (20K) | ~10K (with tags) |
| **Signal Type** | Subject/content | Genre/form | Geographic focus |
| **Difficulty** | Hard | Medium | Medium |
| **Text Signals** | Implicit | Implicit | Often explicit |
| **Expected V-measure** | 0.55-0.75 | 0.60-0.80 | 0.65-0.85 |

## Research Opportunities

1. **Hierarchical clustering**: Group regions into continents first
2. **Multi-label clustering**: Use all geographic tags, not just first
3. **Cross-lingual**: Compare English text about different regions
4. **Temporal dynamics**: Geographic focus evolution in news corpora
5. **Fine-grained**: City-level or sub-region clustering

## References

Complete clustering task documentation: `docs/tasks/clustering.md`

Code implementation: `src/shelf/taxonomies/geographic.py`

MTEB clustering protocol: https://github.com/embeddings-benchmark/mteb

V-measure paper: Rosenberg & Hirschberg (2007), "V-Measure: A Conditional Entropy-Based External Cluster Evaluation Measure"

---

*Last updated: 2025-12-12*

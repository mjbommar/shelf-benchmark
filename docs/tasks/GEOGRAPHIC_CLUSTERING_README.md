# Geographic Region Clustering - Complete Documentation

> **Status**: Designed and Documented
> **Implementation Phase**: Phase 3 (Evaluation Harness)
> **Task Added**: 2025-12-12

## Quick Start

The Geographic Region Clustering task is the third clustering task in SHELF, evaluating models on their ability to cluster documents by geographic region based on content alone.

**Key Stats:**
- **8 regions**: North America, South America, Europe, East Asia, South/Southeast Asia, Middle East & North Africa, Sub-Saharan Africa, Central America & Caribbean
- **44 locations**: Mapped to the 8 regions
- **~10,000 documents**: Approximately 50% of corpus (documents with geographic tags)
- **Primary metric**: V-measure
- **Expected difficulty**: Medium (easier than LCC/LCGFT due to fewer clusters and explicit signals)

## Files Created

### Core Implementation
1. **`src/shelf/taxonomies/geographic.py`** - Complete mapping utility with:
   - `GEOGRAPHIC_REGION_MAPPING`: 44 locations → 8 regions
   - `get_region()`, `get_region_from_list()`: Lookup functions
   - `filter_documents_for_clustering()`: Data preparation
   - `add_geographic_region_field()`: Label assignment
   - `validate_geographic_data()`: Quality checking

2. **`scripts/analyze_geographic_clustering.py`** - Analysis tool:
   - Distribution analysis
   - Region balance assessment
   - Location frequency analysis
   - Validation reporting

### Documentation
3. **`docs/tasks/clustering.md`** - Updated with Section 3:
   - Task specification
   - Regional mapping table
   - Data format
   - Baselines
   - Implementation notes

4. **`docs/tasks/GEOGRAPHIC_CLUSTERING_SUMMARY.md`** - Quick reference:
   - Task overview
   - Why it matters
   - Implementation guide
   - Expected baselines
   - Common pitfalls

5. **`docs/tasks/geographic_clustering_implementation.md`** - Complete implementation guide:
   - Step-by-step code examples
   - Multiple baseline methods
   - Evaluation metrics
   - Diagnostics
   - Troubleshooting

### Integration
6. **`src/shelf/hub/card.py`** - Updated dataset card:
   - Added geographic clustering to supported tasks
   - Updated cluster counts (21/14/8)
   - Added geographic dimension to dataset summary

7. **`docs/tasks/TODO.md`** - Updated project roadmap:
   - Added geographic clustering to task list
   - Updated task count (9 tasks total)
   - Added to evaluation implementation plan
   - Updated SHELF Score formula

## Regional Mapping Summary

### The 8 Regions

| Region | Count | Example Locations |
|--------|-------|-------------------|
| North America | 16 | US, Canada, California, New York City |
| Europe | 10 | UK, Germany, France, London, Paris |
| East Asia | 6 | China, Japan, Tokyo, Beijing |
| South America | 3 | Brazil, São Paulo |
| South/Southeast Asia | 3 | India, Southeast Asia, Mumbai |
| Central America & Caribbean | 3 | Mexico, Central America, Caribbean |
| Middle East & North Africa | 1 | Middle East |
| Sub-Saharan Africa | 1 | Africa |

**Total**: 44 unique geographic locations

### Design Decisions

**Why 8 regions?**
- Balance between granularity and cluster count
- Meaningful geographic divisions
- Sufficient documents per region (200-2,500 expected)
- Aligns with common geographic categorizations

**Why use first tag only?**
- Simplifies clustering (single-label)
- Matches LCC/LCGFT clustering approach
- Avoids multi-label complexity
- Still covers ~50% of corpus

**Why exclude documents without tags?**
- Not all documents are geographically focused
- Clustering requires geographic ground truth
- Maintains task quality and interpretability

## Expected Baselines

### Performance Estimates

| Model | V-measure | Rationale |
|-------|-----------|-----------|
| Random | 0.00 | No signal |
| TF-IDF + k-means | 0.55-0.70 | Geographic terms are lexical |
| SBERT (MiniLM) | 0.70-0.82 | Good semantic embeddings |
| OpenAI text-embedding-3-small | 0.75-0.88 | State-of-the-art |

### Why Geographic Clustering is "Easier"

Compared to LCC (0.55-0.75) and LCGFT (0.60-0.80):
1. **Fewer clusters**: 8 vs. 14 or 21
2. **Explicit signals**: Place names, geographic terms often explicit
3. **Lexical markers**: Strong correlation between words and regions
4. **Simpler semantics**: Geographic focus clearer than subject or genre

## Usage Examples

### Basic Usage (TF-IDF)

```python
from datasets import load_dataset
from shelf.taxonomies.geographic import filter_documents_for_clustering, add_geographic_region_field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import v_measure_score

# Load and filter
dataset = load_dataset("mjbommar/SHELF", split="test")
docs = filter_documents_for_clustering(list(dataset))
docs = add_geographic_region_field(docs)

# Prepare
texts = [f"{d['title']} {d['body']}" for d in docs]
labels = [d['geographic_region'] for d in docs]

# Cluster
vectorizer = TfidfVectorizer(max_features=10000)
X = vectorizer.fit_transform(texts)
kmeans = MiniBatchKMeans(n_clusters=8, random_state=42)
pred = kmeans.fit_predict(X)

# Evaluate
print(f"V-measure: {v_measure_score(labels, pred):.3f}")
```

### Analysis

```bash
# Run distribution analysis
python scripts/analyze_geographic_clustering.py \
    --artifacts-dir data/artifacts \
    --output results/geographic_analysis.json
```

## Implementation Status

### Completed
- [x] Regional mapping design (8 regions, 44 locations)
- [x] Core utilities (`src/shelf/taxonomies/geographic.py`)
- [x] Analysis script (`scripts/analyze_geographic_clustering.py`)
- [x] Full documentation (3 doc files)
- [x] Dataset card updates
- [x] Project roadmap updates

### Pending (Phase 3)
- [ ] Evaluation harness integration
- [ ] CLI command: `shelf evaluate --task geographic_clustering`
- [ ] Baseline implementations (TF-IDF, SBERT, OpenAI)
- [ ] Published baseline results
- [ ] Leaderboard integration

### Future Enhancements
- [ ] Hierarchical clustering (regions → continents)
- [ ] Multi-label clustering (use all tags, not just first)
- [ ] Fine-grained city-level clustering
- [ ] Cross-lingual evaluation

## Key Design Principles

1. **Consistency**: Follows same pattern as LCC/LCGFT clustering
2. **Simplicity**: Single-label, first-tag approach
3. **Coverage**: 44 locations cover global diversity
4. **Balance**: 8 regions provide meaningful granularity
5. **Explicitness**: Geographic signals often present in text
6. **Measurability**: Standard clustering metrics apply

## Integration with SHELF Benchmark

### Task Hierarchy
```
SHELF Tasks (9 total)
├── Classification (5 tasks)
│   ├── LCC Classification (21 classes)
│   ├── LCGFT Form Classification (133 classes)
│   ├── Topic Classification (112 labels, multi-label)
│   ├── Audience Classification (25 classes)
│   └── Register Classification (8 classes)
├── Retrieval (3 tasks)
│   ├── LCC Retrieval
│   ├── Form Retrieval
│   └── Topic Retrieval
├── Pair Classification (2 tasks)
│   ├── Same-LCC Pairs
│   └── Same-Form Pairs
└── Clustering (3 tasks)
    ├── LCC Clustering (21 clusters)
    ├── LCGFT Category Clustering (14 clusters)
    └── Geographic Region Clustering (8 clusters) ← NEW
```

### SHELF Score Impact
```
SHELF Score = 0.40 × Classification + 0.30 × Retrieval +
              0.15 × PairClassification + 0.15 × Clustering

Clustering = mean(LCC_Vmeasure, LCGFT_Vmeasure, Geographic_Vmeasure)
```

Geographic clustering now contributes 5% to overall SHELF Score (0.15 × 1/3).

## Why This Task Matters

### For NLP Research
- Tests geographic understanding in embeddings
- Evaluates place name and regional context modeling
- Complements subject (LCC) and genre (LCGFT) clustering
- Provides interpretable geographic signal evaluation

### For Applications
- **Library cataloging**: Geographic organization of materials
- **Information retrieval**: Location-based filtering
- **Content recommendation**: Region-specific suggestions
- **Research tools**: Geographic faceted search

### For Benchmark Completeness
- Adds third clustering dimension (subject, genre, geography)
- Covers all major bibliographic facets
- Provides diversity in clustering difficulty
- Enables multi-faceted model evaluation

## Documentation Hierarchy

**Start here**: This README (overview)
↓
**Task details**: `docs/tasks/clustering.md` (Section 3)
↓
**Quick reference**: `docs/tasks/GEOGRAPHIC_CLUSTERING_SUMMARY.md`
↓
**Implementation**: `docs/tasks/geographic_clustering_implementation.md`
↓
**Code**: `src/shelf/taxonomies/geographic.py`

## Next Steps for Development

1. **Phase 3: Evaluation Harness** (Current Priority)
   - Implement geographic clustering evaluator
   - Add CLI command
   - Integrate with main evaluation pipeline

2. **Phase 3: Baseline Implementations**
   - Run TF-IDF baseline
   - Run SBERT baseline
   - Run OpenAI embedding baseline
   - Document results

3. **Phase 5: Launch**
   - Add to leaderboard
   - Publish baseline results
   - Enable community submissions

## Contact and Contribution

This task was designed and documented on 2025-12-12 as part of SHELF benchmark development.

**Questions or issues?** Open an issue on the [GitHub repository](https://github.com/mjbommar/shelf)

**Want to contribute?** See Phase 3, Task 3.4 in `docs/tasks/TODO.md`

---

*Geographic Region Clustering documentation maintained by SHELF development team*

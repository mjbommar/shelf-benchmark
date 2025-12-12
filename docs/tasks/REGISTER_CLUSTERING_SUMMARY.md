# Register Clustering Task - Implementation Summary

## Overview

Register Clustering is a new clustering task for the SHELF benchmark that evaluates how well embedding models capture writing style and formality variations. Unlike LCC clustering (subject-based) and LCGFT clustering (genre-based), Register clustering focuses on linguistic register—the formality, tone, and stylistic characteristics of documents.

## Task Specification

- **Task Name**: `register_clustering`
- **Type**: Clustering (unsupervised)
- **Number of Clusters**: 8
- **Label Field**: `register`
- **Primary Metric**: V-measure
- **Secondary Metrics**: NMI, ARI
- **Difficulty**: Higher than LCC/LCGFT clustering

## The 8 Register Types

1. **academic** (14.8%): Scholarly writing with citations, hedged claims
2. **casual** (9.9%): Informal, blog-style content
3. **conversational** (15.5%): Friendly, approachable tone
4. **creative** (4.8%): Literary, expressive writing
5. **formal** (15.1%): Official, governmental/legal documents
6. **journalistic** (5.1%): News-style, inverted pyramid
7. **professional** (24.7%): Business communication
8. **technical** (10.1%): Specialized, jargon-heavy content

**Total Corpus**: 20,000 documents with intentionally weighted distribution

## Why Register Clustering is Harder

1. **Subtle Signals**: Style features are less salient than content or genre
2. **Multi-Level Features**: Operates across lexical, syntactic, and discourse levels
3. **Embedding Mismatch**: Semantic embeddings may not capture stylistic markers
4. **Class Imbalance**: professional (24.7%) vs creative (4.8%)
5. **Cross-Domain**: Same register appears across different subjects

## Dataset Statistics

| Split | Documents | Clusters | Avg. Size | Notes |
|-------|-----------|----------|-----------|-------|
| Train | 12,000 | 8 | ~1,500 | Stratified by LCC & LCGFT |
| Dev   | 4,000 | 8 | ~500 | Proportional distribution |
| Test  | 4,000 | 8 | ~500 | Proportional distribution |

## Implementation Status

### ✅ Completed

1. **Task Registry** (`src/shelf/evaluate/registry.py`)
   - Added `register_clustering` TaskSpec
   - Label space: REGISTERS (8 types)
   - Fields: text, register, id
   - Metrics: v_measure (primary), nmi, ari (secondary)

2. **Dataset Schema** (`src/shelf/hub/dataset.py`)
   - `register` field already exists (line 48, 173, 218)
   - `register_description` field available
   - All 20,000 documents have register labels

3. **Dataset Card** (`src/shelf/hub/card.py`)
   - Supported tasks table shows clustering with "21/14/8"
   - Register listed in dataset summary

4. **Documentation** (`docs/tasks/REGISTER_CLUSTERING_ADDITION.md`)
   - Complete task description
   - Label space definitions
   - Baseline expectations
   - Evaluation protocol

### 📝 Documentation Updates Needed

The file `docs/tasks/clustering.md` needs to be updated with Register Clustering content. Currently it has been modified to include Geographic Region Clustering. Options:

**Option A**: Replace Geographic Region Clustering with Register Clustering
- Register has full corpus coverage (20,000 docs)
- Geographic only covers ~50% of corpus
- Register is more relevant to style-based evaluation

**Option B**: Include both tasks
- Add Register as 4th clustering task
- Keep Geographic as 3rd task
- Provides comprehensive clustering evaluation

**Recommendation**: Use Option A - Register Clustering is more aligned with the benchmark's goals of evaluating document understanding across content, genre, and style dimensions.

## Usage

```bash
# Evaluate register clustering
shelf evaluate --task register_clustering --model <model_path> --split test

# Run all clustering tasks (LCC, LCGFT, Register)
shelf evaluate --task clustering --model <model_path>

# List available clustering tasks
shelf list --type clustering
```

## Example Output Format

```json
{
  "task": "register_clustering",
  "model": "sentence-transformers/all-MiniLM-L6-v2",
  "predictions": {
    "20251211_030155_57dfc238": 6,  # professional cluster
    "20251211_030155_f3221f5a": 4,  # academic cluster
    ...
  },
  "metrics": {
    "v_measure": 0.42,
    "nmi": 0.43,
    "ari": 0.31
  }
}
```

## Research Context

Register classification and clustering connects to extensive literature on formality detection, readability assessment, and style analysis:

- **Formality Detection**: Pavlick & Tetreault (2016), Sheikha & Inkpen (2010)
- **Register Classification**: RANLP 2023 study comparing statistical, neural, and Transformer models
- **Stylometry**: Character-level BiLSTMs sometimes outperform Transformers
- **POS-based Features**: Most reliable signal for register (nouns/adjectives = formal, pronouns/verbs = informal)

For full literature review, see: `docs/tasks/audience_register_classification.md`

## Expected Baseline Performance

| Model | Expected V-measure | Notes |
|-------|-------------------|-------|
| Random | 0.000 | Random assignment baseline |
| TF-IDF + k-means | 0.25-0.35 | Lexical features may miss style |
| Doc2Vec + k-means | 0.30-0.40 | Classical embeddings |
| SBERT + k-means | 0.35-0.50 | Semantic embeddings |
| OpenAI text-embedding-3-small | 0.40-0.55 | Commercial baseline |

**Note**: These are rough estimates based on the difficulty of style-based clustering. Actual results may vary significantly based on how well embeddings capture linguistic register.

## Relation to Other SHELF Tasks

| Task | Dimension | Difficulty | Coverage |
|------|-----------|------------|----------|
| LCC Classification | Subject (21 classes) | Medium | Content-based |
| LCC Clustering | Subject (21 clusters) | Medium | Content-based |
| LCGFT Classification | Genre (133 forms) | Hard | Genre-based |
| LCGFT Clustering | Genre (14 clusters) | Medium | Genre-based |
| Register Classification | Style (8 classes) | Medium | Style-based |
| **Register Clustering** | **Style (8 clusters)** | **Hard** | **Style-based** |

Register Clustering complements the other tasks by testing whether embeddings capture stylistic variation—a different axis from topical content or document genre.

## Files Modified

1. `/home/mjbommar/src/shelf-benchmark/src/shelf/evaluate/registry.py`
   - Added register_clustering TaskSpec

2. `/home/mjbommar/src/shelf-benchmark/docs/tasks/REGISTER_CLUSTERING_ADDITION.md`
   - Created comprehensive documentation to be integrated

3. `/home/mjbommar/src/shelf-benchmark/docs/tasks/REGISTER_CLUSTERING_SUMMARY.md`
   - Created this summary document

## Next Steps

1. Update `/home/mjbommar/src/shelf-benchmark/docs/tasks/clustering.md` with Register Clustering content
2. Run baseline evaluations to populate TBD metrics
3. Consider adding register-specific analysis in evaluation scripts
4. Update main README if needed to mention 3 clustering tasks

## Technical Notes

- **All documents have register labels**: Unlike geographic (50% coverage), register covers 100% of corpus
- **Imbalanced but realistic**: Distribution reflects real-world document frequencies
- **Generation-aware**: Documents were explicitly generated with register instructions
- **Evaluation-ready**: Can use standard k-means + V-measure pipeline
- **Compatible with MTEB**: Follows same evaluation protocol as MTEB clustering tasks

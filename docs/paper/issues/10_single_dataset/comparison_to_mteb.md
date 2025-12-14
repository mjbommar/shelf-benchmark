# SHELF vs. MTEB: Complementary Approaches to Benchmark Diversity

## Executive Summary

SHELF and MTEB represent **two valid approaches** to creating diverse benchmarks:

- **MTEB**: Diversity through **dataset aggregation** (58 datasets from different sources)
- **SHELF**: Diversity through **taxonomic cross-products** (558,600 configurations from controlled dimensions)

Neither approach is superior. They serve **complementary purposes**:
- **MTEB**: "Can embeddings generalize across diverse real-world tasks?"
- **SHELF**: "Can embeddings understand comprehensive document taxonomies?"

Both are needed for complete model evaluation.

---

## 1. Structural Comparison

### 1.1 Basic Statistics

| Metric | MTEB (2024) | SHELF (v0.3.0) |
|--------|-------------|----------------|
| **Datasets** | 58 | 1 (with 7 configurations) |
| **Documents** | Varies by dataset | 42,616 |
| **Task types** | 8 | 4 |
| **Specific tasks** | 58+ | 18 |
| **Languages** | 112 (with extensions) | 1 (English; multilingual planned) |
| **Subject domains** | Implicit (varies) | 21 explicit (LCC classes) |
| **Document forms** | Dataset-dependent | 133 explicit (LCGFT) |
| **Generation** | Real-world data | Synthetic (9 models) |
| **Contamination risk** | High (public datasets) | Low (synthetic, recent) |

### 1.2 Task Coverage

**MTEB tasks** (8 types):
1. Classification
2. Clustering
3. Pair Classification
4. Reranking
5. Retrieval
6. Semantic Textual Similarity (STS)
7. Summarization
8. Bitext Mining (translation)

**SHELF tasks** (4 types, 18 specific):
1. **Classification** (6 tasks):
   - LCC subject (21 classes)
   - LCGFT form (133 classes)
   - Genre category (14 classes)
   - Register (8 classes)
   - Audience (25 classes)
   - Topic (112 classes, multi-label)

2. **Retrieval** (3 tasks):
   - By subject (LCC)
   - By form (LCGFT)
   - By topic (LCSH)

3. **Clustering** (3 tasks):
   - By subject (LCC)
   - By form (LCGFT)
   - By register

4. **Pair Classification** (6 tasks):
   - Same LCC class? (binary)
   - Same form? (binary)
   - Same register? (binary)
   - Same audience? (binary)
   - Share any topic? (binary)
   - How many topics shared? (4-class)

**Coverage comparison**:
- **MTEB advantage**: Broader task types (STS, summarization, translation)
- **SHELF advantage**: Deeper classification (133 vs. typical 5-20 classes), taxonomic focus

---

## 2. Diversity Strategies

### 2.1 MTEB: Dataset Aggregation

**Approach**: Combine many existing datasets

**Examples** (subset of 58):
- Amazon Reviews (sentiment)
- ArguAna (argument retrieval)
- Banking77 (intent classification)
- BioASQ (biomedical QA)
- CQADupStack (duplicate question detection)
- DBPedia (entity classification)
- FEVER (fact verification)
- NFCorpus (medical information retrieval)
- SciFact (scientific claim verification)
- TREC-COVID (pandemic document retrieval)

**Diversity sources**:
- Different domains (biomedical, legal, e-commerce, Wikipedia)
- Different task formats (Q&A, classification, retrieval)
- Different dataset sizes (1k to 1M+ examples)
- Different annotation schemes

**Strengths**:
1. ✅ Real-world data reflects actual use cases
2. ✅ Established baselines from prior research
3. ✅ Community-validated datasets
4. ✅ Broad task type coverage

**Challenges**:
1. ❌ Inconsistent annotation quality across datasets
2. ❌ Dataset-specific artifacts (shortcuts, biases)
3. ❌ Contamination risk (public data in pre-training)
4. ❌ Difficult to disentangle: "Is model good at retrieval, or just good at these 20 retrieval datasets?"

### 2.2 SHELF: Taxonomic Cross-Products

**Approach**: Generate documents by systematically sampling independent dimensions

**Dimensions** (all independent):
- **Subject** (LCC): 21 classes (A-Z)
- **Form** (LCGFT): 133 specific forms
- **Register**: 8 writing tones
- **Audience** (LCDGT): 25 target demographics
- **Topics** (LCSH): 112 subjects (multi-label)
- **Geography**: 44 regions

**Cross-product**:
```
Core: 21 × 133 × 8 × 25 = 558,600 configurations
Full (with topics/geo): ~5.9 billion possible combinations
Actual corpus: 42,616 sampled configurations
```

**Diversity sources**:
- Comprehensive subject taxonomy (Library of Congress)
- Diverse document forms (Maps to Prayers to Satellite imagery)
- Controlled variation (uniform distributions)
- Independent dimensions (no natural correlations)

**Strengths**:
1. ✅ Systematic coverage (all subjects × all forms)
2. ✅ Controlled methodology (no dataset-specific artifacts)
3. ✅ Low contamination risk (synthetic, recent)
4. ✅ Explicit dimensions (clear what's being tested)

**Challenges**:
1. ❌ Synthetic data may differ subtly from real documents
2. ❌ Single generation methodology (though 9 models used)
3. ❌ Narrower task type coverage (no summarization, STS, etc.)
4. ❌ Requires validation against real-world performance

---

## 3. What Each Benchmark Measures

### 3.1 MTEB: Embedding Quality Across Tasks

**Core question**: "Do your embeddings work for diverse real-world applications?"

**Tests**:
- Transfer learning: One encoder, many tasks
- Domain generalization: Medical to legal to e-commerce
- Task generalization: Classification to retrieval to clustering

**Ideal MTEB model**:
- General-purpose embeddings
- Works across domains without fine-tuning
- Captures semantic similarity broadly

**Example findings** (from MTEB papers):
- Model X is best for retrieval, Model Y for classification
- Domain-specific models (BioMed) may outperform general models on domain tasks
- Instruction-following models improve on many tasks

### 3.2 SHELF: Document Understanding Across Taxonomies

**Core question**: "Can you understand documents across comprehensive classification systems?"

**Tests**:
- Subject identification (21 LCC classes covering all knowledge)
- Form recognition (133 LCGFT forms from Maps to Jokes)
- Multi-faceted classification (simultaneous subject/form/register/audience)
- Cross-product understanding (Maps about Philosophy, Jokes about Law)

**Ideal SHELF model**:
- Understands bibliographic metadata
- Distinguishes subtle form differences (Lectures vs. Speeches, Maps vs. Diagrams)
- Handles uncorrelated dimensions (not just medical→academic pattern)

**Example findings** (expected):
- Model A excels at subject but struggles with form
- Model B needs more data for rare forms (Satellite imagery, Eulogies)
- Generic embeddings may fail on specialized forms

### 3.3 Complementarity

**What MTEB tests that SHELF doesn't**:
- Semantic textual similarity (STS)
- Summarization quality
- Cross-lingual transfer (though MTEB extensions cover this)
- Question answering retrieval

**What SHELF tests that MTEB doesn't**:
- Comprehensive genre/form classification (133 forms)
- Subject taxonomy understanding (LCC)
- Independent dimension handling (uncorrelated subject×form)
- Bibliographic metadata extraction

**Overlap**:
Both test classification, clustering, retrieval, and pair classification. But:
- MTEB: Across diverse datasets/domains
- SHELF: Within comprehensive taxonomies

---

## 4. Benchmark Philosophy

### 4.1 MTEB's Aggregation Philosophy

> "Benchmarks are critical in guiding machine learning progress... MTEB has emerged as a comprehensive evaluation framework for embedding models across diverse tasks and languages."
> — [Maintaining MTEB](https://arxiv.org/html/2506.21182v1)

**Core principles**:
1. **Breadth over depth**: Cover many task types
2. **Real-world grounding**: Use established datasets
3. **Community validation**: Incorporate well-studied benchmarks
4. **Extensibility**: Easy to add new datasets

**Analogy**: MTEB is like a **comprehensive fitness test** (run, swim, lift, jump) using standard exercises.

### 4.2 SHELF's Cross-Product Philosophy

> "SHELF is 'domain-complete' rather than 'domain-specific.' Strong SHELF performance indicates genuine document understanding across the full breadth of human intellectual output."
> — SHELF CLAUDE.md

**Core principles**:
1. **Depth over breadth**: Deep dive into document classification
2. **Taxonomic grounding**: Use authoritative Library of Congress systems
3. **Controlled diversity**: Systematic cross-product sampling
4. **Independence**: Test uncorrelated dimensions

**Analogy**: SHELF is like a **specialized fitness test for runners** (sprint, marathon, trail, altitude) with controlled conditions.

### 4.3 Both Are Valid

**False dichotomy**: "Single dataset bad, multi-dataset good"

**Reality**: Different benchmarks serve different purposes.

**Domain-specific examples**:
- **FinMTEB**: Finance-specific tasks (good for finance applications)
- **ChemTEB**: Chemistry-specific tasks (good for chemistry applications)
- **SHELF**: Bibliographic-specific tasks (good for library/document applications)

**None of these are "too narrow."** They test what they're designed to test.

---

## 5. Practical Implications

### 5.1 For Model Developers

**Use MTEB when**:
- Building general-purpose embedding models
- Optimizing for broad task coverage
- Comparing to established baselines
- Targeting diverse downstream applications

**Use SHELF when**:
- Building document classification systems
- Optimizing for library/archive applications
- Testing taxonomic understanding
- Handling diverse document forms

**Use both when**:
- Building comprehensive document understanding systems
- Evaluating trade-offs (breadth vs. depth)
- Validating generalization across benchmarks

### 5.2 For Model Users

**MTEB scores tell you**:
- How well embeddings generalize across tasks
- Which models are best for which task types
- Relative performance on established benchmarks

**SHELF scores tell you**:
- How well models classify diverse document types
- Performance on comprehensive subject taxonomies
- Ability to handle uncorrelated document dimensions

**Both together tell you**:
- Complete picture of embedding quality
- Trade-offs between breadth and depth
- Whether to choose general or specialized model

### 5.3 For Researchers

**MTEB research questions**:
- What makes embeddings transfer across tasks?
- How do training data scale/diversity affect multi-task performance?
- Can we predict MTEB scores from model architecture?

**SHELF research questions**:
- How do models learn independent document dimensions?
- What's the sample efficiency for rare forms (Satellite imagery)?
- Can taxonomic understanding transfer to other classification systems?

**Cross-benchmark research**:
- Does MTEB performance predict SHELF performance? (Probably partially)
- Do models overfit to MTEB task types? (Can SHELF reveal this?)
- What's the Pareto frontier of breadth (MTEB) vs. depth (SHELF)?

---

## 6. Evolution and Extensions

### 6.1 MTEB's Evolution

**MTEB (2022)** → **MTEB-French (2024)** → **MMTEB (2025, 250+ languages)**

Extensions:
- **C-MTEB**: Chinese tasks
- **German Text Embedding Clustering Benchmark**
- **SEB**: Scandinavian languages
- **Domain-specific**: ChemTEB (chemistry), FinMTEB (finance)

**Trend**: Language expansion + domain specialization

### 6.2 SHELF's Potential Evolution

**SHELF v0.3.0 (current)**: English, 9 generation models

**Future extensions** (proposed):
- **SHELF-Multilingual**: Same taxonomies, Spanish/French/Chinese
- **SHELF-v2**: Different generation approach (validate robustness)
- **SHELF-Legal**: Deep dive into legal forms (100+ legal document types)
- **SHELF-Science**: Deep dive into scientific genres
- **SHELF-Temporal**: Documents from different eras/styles

**Trend**: Generation methodology validation + domain deepening

### 6.3 Convergence?

**Possible future**:
- MTEB adds SHELF as a "bibliographic classification" task category
- SHELF expands to cover MTEB-style tasks (STS, retrieval) within bibliographic domain
- Both recognized as complementary parts of comprehensive evaluation

**Ideal ecosystem**:
```
General benchmarks (MTEB) ─── Test broad capabilities
                 │
                 ├─── Domain benchmarks (SHELF, FinMTEB, ChemTEB) ─── Test depth
                 │
                 └─── Application benchmarks ─── Test production use cases
```

---

## 7. Conclusion

### 7.1 Two Roads to Diversity

**MTEB's road**: Aggregate diverse datasets
- **Pros**: Real-world data, broad coverage, community validation
- **Cons**: Inconsistent quality, dataset artifacts, contamination risk

**SHELF's road**: Generate cross-product diversity
- **Pros**: Controlled methodology, systematic coverage, low contamination
- **Cons**: Synthetic data, narrower task types, needs validation

**Both reach the goal**: Diverse, comprehensive evaluation

### 7.2 Not Competitors, Collaborators

MTEB and SHELF serve **different evaluation needs**:

| Need | Use MTEB | Use SHELF |
|------|----------|-----------|
| General embedding quality | ✅ | ❌ |
| Document classification depth | ❌ | ✅ |
| Real-world task transfer | ✅ | ❌ |
| Taxonomic understanding | ❌ | ✅ |
| Broad task type coverage | ✅ | ❌ |
| Comprehensive form classification | ❌ | ✅ |

**Complete evaluation requires both.**

### 7.3 Final Verdict on "Single Dataset" Concern

**For MTEB**: Multiple datasets are essential (that's the whole point)

**For SHELF**: Single dataset with massive internal diversity is valid because:
1. **558,600 configurations** ≈ having 500+ specialized datasets
2. **18 tasks** comparable to GLUE (9) and SuperGLUE (10)
3. **Domain-complete** coverage (not domain-specific)
4. **Independent dimensions** create true cross-product diversity

**The "single dataset limitation" does not apply to SHELF's design.**

SHELF is a **benchmark suite** that happens to use a unified generation methodology rather than dataset aggregation. This is a feature, not a bug.

---

## References

1. **MTEB Papers**:
   - [Maintaining MTEB: Towards Long Term Usability and Reproducibility](https://arxiv.org/html/2506.21182v1) (2025)
   - [Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316) (2022)

2. **MTEB Extensions**:
   - [C-MTEB: Chinese MTEB](https://arxiv.org/abs/2309.13369) (2024)
   - [MTEB-French](https://arxiv.org/abs/2405.20468) (2024)
   - [MMTEB: Multilingual Massive Text Embedding Benchmark](https://arxiv.org/abs/2505.07273) (2025)
   - [ChemTEB: Chemistry Text Embedding Benchmark](https://arxiv.org/abs/2411.19672) (2024)

3. **Benchmark Philosophy**:
   - [Challenges and Opportunities in NLP Benchmarking](https://www.ruder.io/nlp-benchmarking/)
   - [GLUE: A Multi-Task Benchmark](https://arxiv.org/abs/1804.07461) (2018)
   - [SuperGLUE: A Stickier Benchmark](https://arxiv.org/abs/1905.00537) (2019)

4. **SHELF**:
   - SHELF CLAUDE.md (project documentation)
   - diversity_analysis.py (this repository)

---

**Document version**: 1.0
**Date**: 2025-12-14

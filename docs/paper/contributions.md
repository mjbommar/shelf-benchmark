# SHELF Paper Contributions

## Primary Contributions

### 1. Novel Benchmark Design
**Claim**: First "domain-complete" synthetic benchmark using Library of Congress taxonomies with controlled cross-dimensional independence.

**Evidence**:
- 21 LCC subject classes with uniform distribution (4.6-4.9% each)
- 133 document forms × 21 subjects = factorial coverage
- Co-occurrence matrices show independence (unlike natural corpora)

**Novelty**: Prior benchmarks are either domain-specific (FinMTEB, ChemTEB) or lack controlled diversity (MTEB).

---

### 2. Contamination Resistance
**Claim**: Synthetic generation ensures zero overlap with any model's pretraining data.

**Evidence**:
- Documents generated Dec 2025 by GPT-5.x, Gemini 2.5/3, Claude 4.5
- Embedding models trained before these documents existed
- No web scraping = no contamination pathway

**Novelty**: First benchmark to explicitly address contamination through synthetic generation.

---

### 3. Surprising Empirical Finding
**Claim**: Sparse TF-IDF+SVD outperforms all neural embeddings, challenging scaling assumptions.

**Evidence**:
- TF+SVD: 0.679 SHELF score
- Best neural (BGE-large): 0.513
- 32% performance gap
- All neural models underperform BM25 baseline on average

**Novelty**: Contradicts prevailing assumption that neural > sparse for text understanding.

---

### 4. Multi-Task Evaluation Framework
**Claim**: Same corpus enables comprehensive evaluation across four task types.

**Evidence**:
- Classification: 3 tasks (LCC, LCGFT, Register)
- Retrieval: 3 tasks (Subject, Form, Category)
- Clustering: 12 tasks (K-means, HDBSCAN, Agglomerative × 4 taxonomies)
- Pair Classification: 6 tasks (Same-subject, same-form, topic overlap, etc.)

**Novelty**: Single corpus with consistent splits across all task types.

---

### 5. Efficiency Analysis
**Claim**: First benchmark to include efficiency-adjusted rankings and Pareto analysis.

**Evidence**:
- SHELF_eff = SHELF × 1000 / log10(params)
- Pareto frontier identification
- Size-category rankings
- Throughput metrics (KB/s)

**Novelty**: Enables cost-aware model selection beyond raw performance.

---

### 6. Open Infrastructure
**Claim**: Complete, reproducible evaluation harness with rich tooling.

**Evidence**:
- CLI: `shelf eval run`, `shelf eval results`, `shelf models list`
- Prediction-file-first interface (framework-agnostic)
- Strict versioning (checksums, git hashes, library versions)
- HuggingFace dataset + GitHub code

**Novelty**: Production-quality benchmark infrastructure, not just a dataset.

---

## Secondary Contributions

### 7. Task Difficulty Taxonomy
**Finding**: Models handle *what* (topics, subjects) better than *how* (form, register) or *where* (geography).

| Difficulty | Task Type | Best Score |
|------------|-----------|------------|
| Easy | Subject classification | 0.88 |
| Medium | Subject retrieval | 0.67 |
| Hard | Form retrieval | 0.14 |
| Very Hard | Geographic clustering | 0.02 |

---

### 8. Scaling Analysis
**Finding**: 10x parameters (33M → 335M) yields only 3% improvement (0.496 → 0.513).

**Implication**: Diminishing returns from scale on document understanding tasks.

---

### 9. Task-Specific Champions
**Finding**: No single model dominates all tasks.

| Task Type | Champion |
|-----------|----------|
| Classification | TF-IDF+SVD |
| Retrieval | MPNet, E5-base |
| Clustering | MPNet, BERT |
| Pair Classification | MPNet |

**Implication**: Task-specific model selection outperforms single-model deployment.

---

## Contribution Mapping to Paper Sections

| Contribution | Section |
|--------------|---------|
| Novel benchmark design | §3 Methods (3.1 Data) |
| Contamination resistance | §3 Methods (3.2 Generation) |
| Sparse > Dense finding | §4 Results (4.1 Main) |
| Multi-task framework | §3 Methods (3.3 Tasks) |
| Efficiency analysis | §4 Results (4.3 Efficiency) |
| Open infrastructure | §5 Discussion |
| Task difficulty | §4 Results (4.2 Per-Task) |
| Scaling analysis | §4 Results (4.4 Scaling) |

---

## Claims Requiring Strongest Evidence

1. **"Sparse outperforms dense"** - Need statistical significance, ablations, robustness checks
2. **"Contamination-resistant"** - Need to verify no leakage pathways exist
3. **"Domain-complete"** - Need to justify LC Classification as "universal"
4. **"First X benchmark"** - Need thorough related work to confirm novelty

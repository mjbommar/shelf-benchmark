> **SUPERSEDED, 2026-08-27.** Written before the literature check and
> before the v0.4 corpus. It contains claims that do not survive review
> (see [contributions.md](contributions.md) for the retired list) and
> numbers that do not reproduce. Kept for history.
>
> Current framing: [outline_v2.md](outline_v2.md).
> Current claims: [contributions.md](contributions.md).
> Work queue: [TODO.md](TODO.md).

# Related Work Notes

## Text Embedding Benchmarks

### MTEB (Massive Text Embedding Benchmark)
- **Paper**: Muennighoff et al., 2023
- **Tasks**: 8 task types, 58 datasets
- **Limitations**:
  - Web-scraped data (contamination risk)
  - Domain-specific datasets aggregated
  - Single metric aggregation hides task variance
- **SHELF differentiation**: Synthetic, contamination-resistant, uniform sampling

### BEIR (Benchmark for Information Retrieval)
- **Paper**: Thakur et al., 2021
- **Tasks**: 18 retrieval datasets
- **Limitations**:
  - Retrieval-only
  - Heterogeneous sources
  - Potential contamination
- **SHELF differentiation**: Multi-task, controlled generation

### SentEval
- **Paper**: Conneau & Kiela, 2018
- **Tasks**: Sentence similarity, classification
- **Limitations**:
  - Dated evaluation protocol
  - Limited task diversity
- **SHELF differentiation**: Modern evaluation, richer task set

---

## Domain-Specific Benchmarks

### FinMTEB (Financial)
- Finance-specific text embedding evaluation
- **SHELF contrast**: Domain-complete vs. domain-specific

### ChemTEB (Chemistry)
- Chemistry domain evaluation
- **SHELF contrast**: Universal coverage

### LegalBench
- Legal reasoning benchmark
- **SHELF contrast**: Includes legal but not limited to it

### BioASQ
- Biomedical question answering
- **SHELF contrast**: Multi-domain

---

## Synthetic Benchmarks

### HELM (Holistic Evaluation of Language Models)
- **Paper**: Liang et al., 2022
- **Focus**: LLM capabilities broadly
- **Relevance**: Systematic evaluation methodology

### BIG-Bench
- **Paper**: Google, 2022
- **Focus**: Emergent capabilities
- **Relevance**: Task diversity approach

### TruthfulQA
- **Paper**: Lin et al., 2022
- **Focus**: Truthfulness evaluation
- **Relevance**: Synthetic question generation

---

## Library Classification Literature

### Library of Congress Classification
- **Source**: Library of Congress
- **Coverage**: All human knowledge
- **Use in SHELF**: Primary taxonomy

### LCGFT (Genre/Form Terms)
- Document form classification
- 550+ terms organized in 14 categories

### LCSH (Subject Headings)
- Subject taxonomy
- ~486,000 headings

---

## Benchmark Gaming Literature

### Benchmark Contamination
- **Key papers**:
  - Magar & Schwartz, 2022 (Data contamination in LLMs)
  - Dodge et al., 2021 (Documenting large web corpora)
- **SHELF solution**: Synthetic generation

### Leaderboard Gaming
- **Key observation**: Models overfit to specific benchmarks
- **SHELF solution**: Novel evaluation data, multiple metrics

### Evaluation Reproducibility
- **Key papers**:
  - Muennighoff et al., 2023 (MTEB sensitivity to implementation)
- **SHELF solution**: Strict versioning, prediction-file-first

---

## Sparse vs. Dense Retrieval

### BM25
- **Paper**: Robertson & Zaragoza, 2009
- **Status**: Still competitive baseline

### Dense Retrieval
- **Papers**: Karpukhin et al., 2020 (DPR)
- **Trend**: Dense assumed superior

### Hybrid Methods
- **Papers**: Formal et al., 2021 (SPLADE)
- **Relevance**: Bridge sparse/dense

### SHELF Finding
- Sparse TF-IDF+SVD outperforms dense on universal document understanding
- Contradicts dense > sparse assumption

---

## Key Citations to Include

### Benchmarks
1. MTEB (Muennighoff et al., 2023)
2. BEIR (Thakur et al., 2021)
3. SentEval (Conneau & Kiela, 2018)
4. HELM (Liang et al., 2022)

### Embedding Models
1. BGE (Xiao et al., 2023)
2. E5 (Wang et al., 2022)
3. GTE (Li et al., 2023)
4. Sentence-BERT (Reimers & Gurevych, 2019)

### Contamination
1. Data contamination analysis (Magar & Schwartz, 2022)
2. Benchmark gaming (Koch et al., 2021)

### Library Science
1. Library of Congress Classification (LC, 2023)
2. Genre/Form Terms (LCGFT, 2023)

---

## Positioning Statement

> "Unlike domain-specific benchmarks (FinMTEB, ChemTEB, LegalBench) that evaluate narrow capabilities, and unlike web-scraped benchmarks (MTEB, BEIR) that risk contamination, SHELF provides a contamination-resistant evaluation of universal document understanding using the Library of Congress Classification—the most comprehensive bibliographic taxonomy ever developed."

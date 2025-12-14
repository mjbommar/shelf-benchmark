# SHELF Document Length Distribution

## Executive Summary

SHELF v0.3.0 contains 42,616 documents with the following length characteristics:

- **Median**: 472 BERT tokens (322 words)
- **Mean**: 1,002 BERT tokens (636 words)
- **Range**: 0-15,807 tokens
- **46.1%** of documents exceed 512 tokens (BERT/embedding model limit)
- **29.8%** of documents exceed 1,024 tokens
- **15.4%** of documents exceed 2,048 tokens

## Detailed Statistics

### Token Count Distribution (BERT tokenizer)

| Metric | Tokens | Words |
|--------|--------|-------|
| Mean | 1,002 | 636 |
| Median | 472 | 322 |
| Std Dev | 1,288 | 819 |
| Min | 0 | 0 |
| Max | 15,807 | 6,203 |
| 25th percentile | 201 | 122 |
| 75th percentile | 1,490 | 952 |
| 90th percentile | 2,933 | 1,853 |
| 95th percentile | 3,899 | 2,488 |
| 99th percentile | 5,594 | 3,538 |

### Length Stratification

Documents are stratified into three length categories:

| Category | Token Range | Count | Percentage |
|----------|-------------|-------|------------|
| **Short** | ≤512 tokens | 22,973 | 53.9% |
| **Medium** | 512-1024 tokens | 6,950 | 16.3% |
| **Long** | >1024 tokens | 12,693 | 29.8% |

### Truncation Impact Analysis

For neural embedding models with a 512-token limit (e.g., BERT, Sentence-BERT):

- **19,643 documents (46.1%)** require truncation
- Among truncated documents:
  - **Mean information loss**: 55.0% of content discarded
  - **Median information loss**: 67.5% of content discarded
  - **Maximum information loss**: 96.8% of content discarded

This substantial information loss for dense methods creates a balanced comparison with sparse methods that can leverage full document content.

## Distribution Across Taxonomy Dimensions

### Length by LCC Code

All 21 LCC codes are represented in each length stratum. Average length varies modestly by subject:

**Longest average (tokens):**
1. R (Medicine): 1,077
2. Z (Bibliography, Library Science): 1,063
3. B (Philosophy, Psychology, Religion): 1,049
4. K (Law): 1,037
5. C (Auxiliary Sciences of History): 1,032

**Shortest average (tokens):**
1. H (Social Sciences): 948
2. U (Military Science): 955
3. L (Education): 961
4. F (History of the Americas): 966
5. D (World History): 969

**Range**: 948-1,077 tokens (13.6% variation from shortest to longest)

### Length by Document Form

All 133 document forms appear in each length stratum. Forms exhibit greater length variation:

**Longest forms (tokens):**
1. Drama: 1,247
2. Casebooks (Law): 1,244
3. Brochures: 1,219
4. Diagrams: 1,216
5. Case studies: 1,197

**Shortest forms (tokens):**
1. Speeches: 763
2. Eulogies: 785
3. Tributes: 800
4. Poetry: 805
5. Songs: 806

**Range**: 763-1,247 tokens (63.4% variation from shortest to longest)

## Key Findings

### 1. Balanced Length Distribution

The median document (472 tokens) falls just below the 512-token BERT limit, meaning:
- **53.9%** of documents fit entirely within embedding model context windows
- **46.1%** require truncation, creating a natural test of truncation robustness

### 2. Complete Taxonomy Coverage at All Lengths

Critical for avoiding length-based selection bias:
- All 21 LCC codes appear in short, medium, and long strata
- All 133 forms appear in short, medium, and long strata
- No systematic correlation between taxonomy labels and length

### 3. Realistic Information Loss

The 55% average information loss at 512-token truncation reflects real-world challenges:
- Many documents have critical information in later sections
- Truncation effects are substantial, not marginal
- This tests embedding model robustness to incomplete context

### 4. Moderate Within-Taxonomy Variation

- LCC codes show 13.6% length variation (948-1,077 tokens)
- Forms show 63.4% length variation (763-1,247 tokens)
- This variation is expected and reflects real-world document diversity

## Implications for Evaluation

### Sparse Methods (TF-IDF, BM25)

**Advantages:**
- Access to full document content (no truncation)
- Can leverage all 1,002 average tokens
- Benefit from longer documents with more term occurrences

**Disadvantages:**
- Higher computational cost for long documents
- No semantic understanding
- Vulnerable to vocabulary mismatch

### Dense Methods (BERT, Sentence-BERT, etc.)

**Advantages:**
- Semantic understanding of content
- Fixed computational cost regardless of length
- Strong performance on short documents

**Disadvantages:**
- Truncation at 512 tokens affects 46.1% of documents
- 55% average information loss on truncated documents
- Cannot leverage full context for long documents

### Fair Comparison

The length distribution creates a **balanced evaluation**:
- Short documents (53.9%) favor neither sparse nor dense methods
- Medium/long documents (46.1%) test truncation robustness for dense methods
- Dense methods are not disadvantaged - they are tested on their ability to handle real-world truncation scenarios

## Comparison to Other Benchmarks

### MTEB (Massive Text Embedding Benchmark)

MTEB tasks vary widely in document length:
- ArguAna: ~200 tokens average
- FiQA: ~150 tokens average
- TREC-COVID: ~300 tokens average
- CQADupStack: ~100 tokens average

SHELF's 472-token median is **longer** than most MTEB retrieval tasks, providing a more challenging truncation scenario.

### BEIR (Benchmarking IR)

BEIR datasets also vary:
- NFCorpus: ~300 tokens average
- SciFact: ~200 tokens average
- TREC-NEWS: ~600 tokens average

SHELF's distribution is comparable to TREC-NEWS, the longest BEIR task.

### MS MARCO

MS MARCO passages are artificially short (~60 tokens) to avoid truncation issues entirely. SHELF does not artificially constrain length, testing real-world robustness.

## Conclusion

SHELF's document length distribution:
1. **Reflects realistic document diversity** (50% below, 50% near/above BERT limit)
2. **Does not unfairly advantage sparse methods** - it tests truncation robustness
3. **Maintains taxonomy independence** - all classes/forms at all lengths
4. **Provides a challenging but fair benchmark** for both sparse and dense methods

The 46.1% truncation rate is a **feature, not a bug** - it tests whether embedding models can perform well under real-world constraints where many documents exceed context limits.

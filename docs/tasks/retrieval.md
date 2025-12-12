# Task: Retrieval Tasks

> **Task Type**: retrieval
> **Difficulty**: medium
> **Primary Metric**: NDCG@10

## Overview

The SHELF Retrieval tasks evaluate the ability of text embedding and retrieval models to find semantically similar documents based on Library of Congress taxonomy dimensions. These tasks assess whether models can effectively capture domain-specific similarity (LCC classification), structural similarity (LCGFT form), and topical relevance in culturally and linguistically diverse documents.

Unlike general-domain retrieval benchmarks, SHELF Retrieval tasks specifically test models on library science taxonomies and bibliographic metadata, which require understanding of specialized classification systems, literary forms, and subject-matter organization. This makes them particularly relevant for digital library systems, academic search engines, and cultural heritage applications.

## Task Definitions

SHELF includes three distinct retrieval tasks, each evaluating a different dimension of document similarity:

### 1. LCC Retrieval (Library of Congress Classification)

**Task**: Given a query document, retrieve other documents from the same LCC class.

#### Input
- **Query**: A document with known LCC classification (e.g., "P" - Language and Literature)
- **Corpus**: All documents in the benchmark corpus

#### Output
Ranked list of documents, scored by relevance to the query document.

#### Positive/Negative Criteria
- **Positive documents**: Documents sharing the same LCC code (top-level class) as the query
  - Example: If query has `lcc_code: "P"`, all other documents with `lcc_code: "P"` are relevant
- **Negative documents**: Documents with different LCC codes
- **Relevance scoring**: Binary (relevant=1, non-relevant=0)

#### Formal Definition
Given query document $q$ with LCC code $c_q$ and corpus $D$, retrieve documents $d \in D$ where $c_d = c_q$, ranked by embedding similarity.

### 2. Form Retrieval (LCGFT Form-based)

**Task**: Given a query document, retrieve documents of the same LCGFT literary/bibliographic form.

#### Input
- **Query**: A document with known LCGFT form (e.g., "Satire", "Theological works", "Broadsides")
- **Corpus**: All documents in the benchmark corpus

#### Output
Ranked list of documents, scored by relevance to the query document.

#### Positive/Negative Criteria
- **Positive documents**: Documents sharing the same `lcgft_form` as the query
  - Example: If query has `lcgft_form: "Satire"`, all other Satire documents are relevant
- **Negative documents**: Documents with different LCGFT forms
- **Relevance scoring**: Binary (relevant=1, non-relevant=0)

#### Formal Definition
Given query document $q$ with LCGFT form $f_q$ and corpus $D$, retrieve documents $d \in D$ where $f_d = f_q$, ranked by embedding similarity.

### 3. Topic Retrieval (Subject-based)

**Task**: Given a topic string, retrieve documents relevant to that topic.

#### Input
- **Query**: A topic string from the LOC subject headings (e.g., "Climate change", "Music", "Neuroscience")
- **Corpus**: All documents in the benchmark corpus

#### Output
Ranked list of documents, scored by topical relevance.

#### Positive/Negative Criteria
- **Positive documents**: Documents containing the query topic in their `topics` list
  - Example: For query "Climate change", documents with "Climate change" in topics are relevant
- **Negative documents**: Documents without the query topic in their topics list
- **Relevance scoring**: Binary (relevant=1, non-relevant=0)

#### Formal Definition
Given topic query $t$ and corpus $D$, retrieve documents $d \in D$ where $t \in \text{topics}(d)$, ranked by embedding similarity.

## Dataset

### Source

Retrieval tasks are constructed from the SHELF synthetic document corpus, which consists of generated documents spanning:
- **21 LCC classes**: A-Z covering all major subject domains
- **143 LCGFT forms**: Literary and bibliographic forms (e.g., Satire, Essays, Theological works, etc.)
- **113 topics**: Library of Congress subject headings
- **Varying document lengths**: 25-5000+ words
- **Multiple registers**: academic, professional, creative, technical, conversational, persuasive, instructional, casual

### Statistics

| Task | Queries | Corpus Size | Avg Positives/Query | Query Source |
|------|---------|-------------|---------------------|--------------|
| LCC Retrieval | Variable | Full corpus | ~100-500 | Test documents |
| Form Retrieval | Variable | Full corpus | ~10-100 | Test documents |
| Topic Retrieval | 113 | Full corpus | ~5-50 | Topic strings |

**Split Distribution**:
- **Train**: 60% of corpus (for training embedding models)
- **Dev**: 20% of corpus (for validation)
- **Test**: 20% of corpus (held-out evaluation)

For document-query tasks (LCC, Form), queries are sampled from the test split, and the corpus includes train+dev documents.

### Label Space

#### LCC Codes (21 classes)
- A: General Works
- B: Philosophy, Psychology, Religion
- C: Auxiliary Sciences of History
- D: World History
- E-F: History of the Americas
- G: Geography, Anthropology, Recreation
- H: Social Sciences
- J: Political Science
- K: Law
- L: Education
- M: Music
- N: Fine Arts
- P: Language and Literature
- Q: Science
- R: Medicine
- S: Agriculture
- T: Technology
- U: Military Science
- V: Naval Science
- Z: Bibliography, Library Science

#### LCGFT Forms (143 forms)
Selected examples: Satire, Essays, Theological works, Broadsides, Poetry, Fiction, Drama, Textbooks, Reference works, Sermons, etc.

#### Topics (113 subjects)
Selected examples: Climate change, Music, Neuroscience, Ocean conservation, Culture, Technology, etc.

### Data Format

#### Document Format
```json
{
  "id": "20251211_030155_57dfc238",
  "title": "Quarterly Agribusiness Risk Memo on Emerging Externalities",
  "body": "In light of climate change...",
  "word_count": 37,
  "lcc_code": "S",
  "lcc_name": "Agriculture",
  "lcgft_category": "Literature",
  "lcgft_form": "Satire",
  "topics": ["Climate change", "Wildlife", "Neuroscience", "Ocean conservation"],
  "audience": null,
  "geographic": [],
  "target_length": "tiny",
  "register": "professional"
}
```

#### Retrieval Query Format
```json
{
  "query_id": "q_001",
  "query_type": "document",
  "query_text": "Document body text...",
  "query_metadata": {
    "lcc_code": "S",
    "lcgft_form": "Satire"
  },
  "relevant_docs": ["doc_123", "doc_456", "doc_789"]
}
```

#### Topic Query Format
```json
{
  "query_id": "topic_001",
  "query_type": "topic",
  "query_text": "Climate change",
  "relevant_docs": ["doc_123", "doc_456"]
}
```

## Evaluation

### Primary Metric: NDCG@10

**Normalized Discounted Cumulative Gain at 10** (NDCG@10) measures the quality of the ranked retrieval list by considering both the relevance of retrieved documents and their positions in the ranking.

**Why NDCG@10**:
- Rank-aware: Penalizes relevant documents appearing lower in results
- Handles graded relevance (can be extended for multi-level relevance in future)
- Standard metric in MTEB and BEIR benchmarks for retrieval tasks
- Balances precision and recall in top-k results
- Well-suited for scenarios where users focus on top results

**Formula**:
$$\text{NDCG@k} = \frac{\text{DCG@k}}{\text{IDCG@k}}$$

where:
$$\text{DCG@k} = \sum_{i=1}^{k} \frac{2^{rel_i} - 1}{\log_2(i + 1)}$$

For binary relevance: $rel_i \in \{0, 1\}$

IDCG@k is the ideal DCG if all relevant documents were ranked first.

### Secondary Metrics

#### MRR (Mean Reciprocal Rank)
Measures how quickly the first relevant document appears in the ranking.

$$\text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\text{rank}_q}$$

where $\text{rank}_q$ is the position of the first relevant document for query $q$.

**Use case**: Particularly informative for LCC and Form retrieval where finding any similar document quickly is valuable.

#### Recall@k
Measures the fraction of relevant documents retrieved in the top-k results.

$$\text{Recall@k} = \frac{|\{\text{relevant docs}\} \cap \{\text{top-k retrieved}\}|}{|\{\text{relevant docs}\}|}$$

**Reported at**: k ∈ {1, 5, 10, 50, 100}

**Use case**: Important for applications requiring comprehensive retrieval (e.g., literature review systems).

#### MAP@k (Mean Average Precision)
Measures precision at each relevant document position, averaged across queries.

$$\text{AP@k} = \frac{1}{|R|} \sum_{i=1}^{k} P(i) \cdot rel(i)$$

where $R$ is the set of relevant documents, $P(i)$ is precision at position $i$, and $rel(i) = 1$ if document at position $i$ is relevant.

### Evaluation Protocol

1. **Corpus Sampling**: For efficiency, follow MTEB protocol by sampling 100,000 candidates from the full corpus (including all ground truth documents) for each query evaluation
2. **Embedding Generation**: Models encode both queries and corpus documents into fixed-dimensional embeddings
3. **Similarity Computation**: Cosine similarity between query and document embeddings
4. **Ranking**: Documents ranked by descending similarity score
5. **Metric Computation**: Calculate NDCG@10, MRR, Recall@k, and MAP@k for each query
6. **Aggregation**: Report mean scores across all queries in the test set

#### Special Considerations

- **Document-as-Query Tasks** (LCC, Form): The query document itself should be excluded from the corpus to prevent trivial self-retrieval
- **Topic-Query Task**: Use topic string directly as query text; no special preprocessing
- **Multi-topic Documents**: Documents with multiple topics are relevant for any of their topics
- **Cross-split Retrieval**: Queries come from test split; corpus includes train+dev documents to simulate realistic retrieval scenarios

## Baselines

### Measured Results (December 2025)

The following results were measured using the SHELF evaluation framework on the LCC Retrieval task with 100 test queries and an 8,000-document corpus (train + validation splits).

#### Neural Embedding Models

| Model | NDCG@10 | MRR | P@10 | R@10 | Embedding Dim | Notes |
|-------|---------|-----|------|------|---------------|-------|
| **all-MiniLM-L6-v2** | **0.4551** | **0.6425** | 0.4420 | 0.0117 | 384 | Sentence-transformer (recommended) |
| bert-base-uncased | 0.3271 | 0.5530 | 0.3130 | 0.0083 | 768 | Raw BERT + mean pooling |

#### Traditional/Sparse Retrieval Methods

| Model | NDCG@10 | MRR | P@10 | R@10 | Notes |
|-------|---------|-----|------|------|-------|
| **TF-IDF + SVD** | **0.4566** | 0.6630 | 0.4350 | 0.0114 | 256-dim SVD, 50k vocab, (1,2)-grams |
| BM25 | 0.4468 | **0.6778** | 0.4210 | 0.0111 | k1=1.5, b=0.75 (standard) |

**Detailed Metrics (all-MiniLM-L6-v2)**:

| Metric | @1 | @5 | @10 | @50 | @100 |
|--------|-----|-----|------|------|------|
| NDCG | 0.4800 | 0.4762 | 0.4551 | 0.4073 | 0.3749 |
| Precision | 0.4800 | 0.4720 | 0.4420 | 0.3918 | 0.3569 |
| Recall | 0.0013 | 0.0062 | 0.0117 | 0.0517 | 0.0941 |
| MAP | 0.0013 | 0.0051 | 0.0088 | 0.0321 | 0.0544 |

**Key Observations**:
- Sentence-transformer models (trained for semantic similarity) significantly outperform raw BERT (~39% improvement in NDCG@10)
- MRR is relatively high (0.64), indicating models often rank at least one relevant document in top positions
- Low recall reflects the challenging nature of the task: with 21 LCC classes and ~380 relevant documents per query on average, finding all relevant documents requires comprehensive coverage
- Precision@10 above 0.44 indicates nearly half of top-10 results are relevant

### Key Observations

**Traditional vs. Neural Methods**:
- TF-IDF with SVD dimensionality reduction matches sentence-transformer performance (NDCG@10: 0.4566 vs 0.4551)
- BM25 achieves highest MRR (0.6778), indicating it frequently ranks relevant documents first
- Neural embeddings from raw BERT significantly underperform (~39% lower NDCG@10 than sentence-transformers)
- Traditional sparse methods remain competitive on this bibliographic retrieval task

**Why Traditional Methods Perform Well**:
- LCC retrieval relies on topical similarity, which often has lexical overlap
- Synthetic documents may have consistent vocabulary within LCC classes
- Traditional methods capture term frequency patterns effectively

### Planned Baselines

| Model | NDCG@10 (LCC) | NDCG@10 (Form) | NDCG@10 (Topic) | Notes |
|-------|---------------|----------------|-----------------|-------|
| Random | ~0.05 | ~0.03 | ~0.02 | Lower bound (random ranking) |
| all-mpnet-base-v2 | TBD | TBD | TBD | Larger sentence-transformer |
| E5-large | TBD | TBD | TBD | Instruction-tuned embeddings |
| bge-base-en-v1.5 | TBD | TBD | TBD | BAAI embedding model |
| Domain-tuned | TBD | TBD | TBD | Fine-tuned on SHELF train |

### Evaluation Configuration

```
Task: lcc_retrieval
Queries: 100 (from test split)
Corpus: 8,000 documents (train + validation splits)
Similarity: Cosine similarity
Ranking: Top-100 per query
Random seed: 42
Platform: Linux x86_64
```

## Related Work

### Similar Tasks in Other Benchmarks

#### MTEB (Massive Text Embedding Benchmark)
SHELF Retrieval tasks follow MTEB's standardized retrieval evaluation methodology:
- **Corpus structure**: Queries + corpus + relevance judgments
- **Metrics**: NDCG@10 as primary metric (matching MTEB standard)
- **Scale**: 100k corpus sampling for efficiency
- **Differences**: SHELF focuses on bibliographic/taxonomic similarity rather than general semantic similarity

MTEB includes retrieval tasks like:
- **ArguAna**: Argument retrieval
- **TREC-COVID**: Scientific article retrieval
- **SciFact**: Scientific claim verification
- **NQ (Natural Questions)**: Question answering retrieval

**SHELF distinction**: Emphasizes library classification systems and literary forms, not covered in MTEB.

#### BEIR (Benchmarking IR)
BEIR provides a heterogeneous benchmark for zero-shot information retrieval across 18 datasets:
- **Tasks**: Fact checking, citation prediction, duplicate question retrieval, news retrieval, bio-medical IR
- **Metrics**: NDCG@10 (primary), MAP, Recall@k, MRR
- **Focus**: Out-of-domain generalization

**SHELF alignment**: Uses similar metric suite (NDCG@10, MRR, Recall@k) for comparability with BEIR results.

**SHELF distinction**: Evaluates retrieval based on controlled taxonomic dimensions rather than diverse task types.

#### Semantic Textual Similarity (STS) Benchmarks
STS tasks (STS-B, SemEval) measure sentence-pair similarity with continuous scores (0-5).

**SHELF difference**:
- STS focuses on sentence-level semantic equivalence
- SHELF evaluates document-level retrieval with bibliographic metadata
- SHELF uses binary relevance based on taxonomy matching

### Methodological Comparison

| Aspect | MTEB | BEIR | STS | SHELF Retrieval |
|--------|------|------|-----|-------------------|
| **Query Type** | Text queries | Text queries | Sentence pairs | Documents + topics |
| **Relevance** | Task-specific | Binary/graded | Continuous (0-5) | Binary (taxonomy-based) |
| **Primary Metric** | NDCG@10 | NDCG@10 | Pearson r | NDCG@10 |
| **Domain** | General | Multi-domain | General | Library/bibliographic |
| **Evaluation** | In-domain | Zero-shot | Similarity scoring | Cross-taxonomy |

### Relevant Literature

1. **BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models** (Thakur et al., 2021)
   - Established importance of diverse retrieval tasks and zero-shot evaluation
   - Demonstrated BM25 as strong baseline for generalization
   - [NeurIPS 2021 Paper](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/65b9eea6e1cc6bb9f0cd2a47751a186f-Paper-round2.pdf)

2. **MTEB: Massive Text Embedding Benchmark** (Muennighoff et al., 2022)
   - Comprehensive framework for embedding evaluation across 8 task types
   - Standardized retrieval task structure with NDCG@10
   - [MTEB GitHub](https://github.com/embeddings-benchmark/mteb)

3. **Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks** (Reimers & Gurevych, 2019)
   - Foundation for modern semantic similarity and retrieval
   - Demonstrated efficiency of bi-encoder architectures

4. **Dense Passage Retrieval for Open-Domain Question Answering** (Karpukhin et al., 2020)
   - Showed dense embeddings can outperform sparse retrieval (BM25)
   - Established training methodology for retrieval models

5. **Library of Congress Classification System**
   - Provides hierarchical subject organization for knowledge
   - Foundation for LCC-based retrieval task design
   - [LOC Classification](http://id.loc.gov/authorities/classification)

6. **Library of Congress Genre/Form Terms (LCGFT)**
   - Controlled vocabulary for literary and bibliographic forms
   - Basis for form-based retrieval evaluation
   - [LCGFT Authorities](https://www.loc.gov/aba/publications/FreeLCGFT/freelcgft.html)

## Implementation Notes

### Running Evaluation

#### Basic Evaluation
```bash
# Evaluate all retrieval tasks
shelf evaluate --task retrieval --model sentence-transformers/all-MiniLM-L6-v2

# Evaluate specific retrieval task
shelf evaluate --task lcc_retrieval --model sentence-transformers/all-MiniLM-L6-v2
shelf evaluate --task form_retrieval --model sentence-transformers/all-MiniLM-L6-v2
shelf evaluate --task topic_retrieval --model sentence-transformers/all-MiniLM-L6-v2
```

#### Advanced Options
```bash
# Custom corpus sampling size
shelf evaluate --task retrieval --model MODEL_PATH --corpus-sample-size 50000

# Full corpus evaluation (no sampling)
shelf evaluate --task retrieval --model MODEL_PATH --no-corpus-sampling

# Output detailed results
shelf evaluate --task retrieval --model MODEL_PATH --output-dir ./results --save-rankings
```

### Example Code

```python
from shelf import load_retrieval_task
from sentence_transformers import SentenceTransformer

# Load task
task = load_retrieval_task("lcc_retrieval", split="test")

# Load model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Encode corpus and queries
corpus_embeddings = model.encode(task.corpus_texts, show_progress_bar=True)
query_embeddings = model.encode(task.query_texts, show_progress_bar=True)

# Compute similarities and rank
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

results = {}
for query_id, query_emb in zip(task.query_ids, query_embeddings):
    similarities = cosine_similarity([query_emb], corpus_embeddings)[0]
    ranked_indices = np.argsort(similarities)[::-1]
    results[query_id] = [task.corpus_ids[i] for i in ranked_indices]

# Evaluate
from shelf.metrics import compute_retrieval_metrics
metrics = compute_retrieval_metrics(
    results,
    task.relevance_judgments,
    metrics=["ndcg@10", "mrr", "recall@10", "map@10"]
)

print(f"NDCG@10: {metrics['ndcg@10']:.4f}")
print(f"MRR: {metrics['mrr']:.4f}")
```

### Submission Format

Leaderboard submissions should include:

1. **Rankings File** (JSON):
```json
{
  "task": "lcc_retrieval",
  "model": "sentence-transformers/all-MiniLM-L6-v2",
  "results": {
    "query_001": ["doc_123", "doc_456", "doc_789", ...],
    "query_002": ["doc_321", "doc_654", ...]
  }
}
```

2. **Metadata File** (JSON):
```json
{
  "model_name": "all-MiniLM-L6-v2",
  "model_type": "bi-encoder",
  "embedding_dim": 384,
  "parameters": "22M",
  "training_data": "general (1B+ pairs)",
  "special_training": null
}
```

3. **Metrics Summary** (auto-generated):
```json
{
  "ndcg@10": 0.6732,
  "mrr": 0.7845,
  "recall@10": 0.8312,
  "recall@50": 0.9421,
  "map@10": 0.6891
}
```

## Key Insights

### What These Tasks Evaluate

1. **LCC Retrieval**: Tests whether embeddings capture subject-domain similarity
   - Do documents about "Agriculture" cluster together?
   - Can models distinguish "Language & Literature" from "Science"?

2. **Form Retrieval**: Tests whether embeddings capture structural/genre similarity
   - Do satires have distinctive embeddings regardless of subject?
   - Can models recognize literary forms across topics?

3. **Topic Retrieval**: Tests traditional semantic/topical relevance
   - Can models match subject headings to document content?
   - How well do embeddings capture specific topics vs. general domains?

### Difficulty Analysis

**Expected difficulty ranking**: Form Retrieval > LCC Retrieval > Topic Retrieval

**Reasoning**:
- **Topic Retrieval** (easiest): Lexical overlap between topic strings and document content
- **LCC Retrieval** (medium): Requires understanding domain-level similarity without explicit labels in text
- **Form Retrieval** (hardest): Requires recognizing structural/stylistic patterns (e.g., satire vs. essay) which may have subtle linguistic markers

### Use Cases

- **Digital Libraries**: Improve book/document discovery based on classification
- **Academic Search**: Find papers in same subject area or genre
- **Cultural Heritage**: Organize archival materials by form and subject
- **Recommender Systems**: Suggest similar documents based on bibliographic properties
- **Metadata Enrichment**: Auto-suggest classifications for new documents

## References

1. Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., & Gurevych, I. (2021). BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models. *NeurIPS 2021 Datasets and Benchmarks Track*. [Paper](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/65b9eea6e1cc6bb9f0cd2a47751a186f-Paper-round2.pdf)

2. Muennighoff, N., et al. (2022). MTEB: Massive Text Embedding Benchmark. [GitHub](https://github.com/embeddings-benchmark/mteb) | [Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)

3. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP 2019*.

4. Library of Congress. Library of Congress Classification. [Classification System](http://id.loc.gov/authorities/classification)

5. Library of Congress. Library of Congress Genre/Form Terms for Library and Archival Materials (LCGFT). [LCGFT Info](https://www.loc.gov/aba/publications/FreeLCGFT/freelcgft.html)

6. Pinecone. (2024). Evaluation Measures in Information Retrieval. [Guide](https://www.pinecone.io/learn/offline-evaluation/)

7. Weaviate. (2024). Evaluation Metrics for Search and Recommendation Systems. [Blog](https://weaviate.io/blog/retrieval-evaluation-metrics)

8. Evidentlyai. Normalized Discounted Cumulative Gain (NDCG) explained. [Documentation](https://www.evidentlyai.com/ranking-metrics/ndcg-metric)

---
*Last updated: December 11, 2025*

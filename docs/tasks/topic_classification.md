# Task: Topic Classification

> **Task Type**: Multi-label classification
> **Difficulty**: Hard
> **Primary Metric**: Micro-F1

## Overview

The Topic Classification task evaluates a model's ability to assign relevant subject matter topics to documents from the Library of Congress collection. Unlike single-label classification where each document belongs to exactly one category, this task requires models to predict multiple topic labels per document, reflecting the inherently multi-faceted nature of library materials.

This task is inspired by Library of Congress Subject Headings (LCSH), a controlled vocabulary maintained by the Library of Congress since 1898 that contains over 340,000 standardized subject terms used for cataloging and retrieval of bibliographic materials worldwide. SHELF uses a curated subset of 113 topic terms spanning 10 major subject domains (general, social sciences, law, science, technology, medicine, humanities, business, politics, and environment), with each document assigned 1-4 relevant topics. The multi-label nature makes this task particularly challenging, requiring models to capture both semantic features and inter-label dependencies.

## Task Definition

### Input
A document represented as plain text, typically ranging from 25 to 2000+ words. Documents span diverse genres and subject matters from the Library of Congress classification system.

**Example:**
```json
{
  "id": "20251211_030155_06dea11a",
  "text": "Ink illustration depicting personified Philosophy seated beside a river, consulting a codex of indigenous Culture. Background shows pre-Columbian temples and modern government buildings across South America, symbolizing continuity of legal traditions, civic institutions, and evolving intellectual heritage."
}
```

### Output
A set of topic labels from the predefined vocabulary of 113 terms. Each document is assigned between 1 and 4 topics, selected from domain-relevant subsets.

**Example:**
```json
{
  "id": "20251211_030155_06dea11a",
  "topics": ["Philosophy", "Culture"]
}
```

### Formal Definition

Given:
- A document $d$ with text content $x$
- A label space $\mathcal{L} = \{l_1, l_2, ..., l_{113}\}$ of topic terms
- Ground truth labels $Y \subseteq \mathcal{L}$ where $1 \leq |Y| \leq 4$

The task is to learn a function $f: x \rightarrow \hat{Y}$ that predicts the subset of relevant topics $\hat{Y} \subseteq \mathcal{L}$ such that $\hat{Y}$ approximates $Y$.

## Dataset

### Source

The dataset is synthetically generated using the SHELF sampler framework, which draws topics from 10 subject domains aligned with Library of Congress Classification (LCC) codes. Topics are selected based on the document's primary LCC class to ensure domain coherence. For example, documents classified under "K" (Law) draw topics from the "law" and "politics" domains, while "Q" (Science) documents use topics from "science", "technology", and "medicine" domains.

Topic assignment follows a realistic distribution:
- Each document receives 1-4 topic labels
- Topics are sampled from domain-relevant subsets using the `TopicSampler`
- The sampler can filter topics by LCC class using predefined domain mappings

### Statistics

| Split | Documents | Avg Topics/Doc | Total Topic Occurrences |
|-------|-----------|----------------|------------------------|
| Train | TBD | ~2.5 | TBD |
| Dev   | TBD | ~2.5 | TBD |
| Test  | TBD | ~2.5 | TBD |

### Label Space

The benchmark uses 113 topic terms organized into 10 domains:

**General (5 topics):** Information, Knowledge, Research, Analysis, Methodology

**Social Sciences (15 topics):** Economics, Sociology, Psychology, Anthropology, Demographics, Public policy, Social welfare, Labor, Commerce, Finance, Statistics, Surveys, Population, Immigration, Poverty

**Law (11 topics):** Constitutional law, Criminal law, Civil law, Contracts, Property, Torts, Administrative law, International law, Human rights, Intellectual property, Environmental law

**Science (13 topics):** Biology, Chemistry, Physics, Mathematics, Geology, Astronomy, Ecology, Genetics, Climate, Evolution, Neuroscience, Quantum mechanics, Thermodynamics

**Technology (12 topics):** Engineering, Computer science, Software, Artificial intelligence, Machine learning, Robotics, Biotechnology, Nanotechnology, Cybersecurity, Data science, Cloud computing, Networks

**Medicine (12 topics):** Public health, Epidemiology, Diseases, Therapeutics, Surgery, Pharmacology, Mental health, Nutrition, Pediatrics, Oncology, Cardiology, Immunology

**Humanities (10 topics):** Philosophy, Ethics, History, Literature, Languages, Art, Music, Religion, Culture, Aesthetics

**Business (12 topics):** Management, Marketing, Accounting, Finance, Strategy, Entrepreneurship, Operations, Human resources, Leadership, Innovation, Supply chain, E-commerce

**Politics (12 topics):** Government, Elections, Political parties, Public administration, International relations, Diplomacy, Security, Defense, Democracy, Authoritarianism, Nationalism, Globalization

**Environment (11 topics):** Climate change, Conservation, Sustainability, Pollution, Biodiversity, Renewable energy, Ecosystems, Wildlife, Deforestation, Ocean conservation, Carbon emissions

Note: "Finance" appears in both Social Sciences and Business domains, resulting in 113 total topic instances but 112 unique terms.

### Data Format

```json
{
  "id": "20251211_030155_06dea11a",
  "title": "Allegories of Thought and Tradition in South America",
  "body": "Ink illustration depicting personified Philosophy...",
  "word_count": 37,
  "lcc_code": "F",
  "lcc_name": "History of the Americas (local)",
  "topics": [
    "Philosophy",
    "Culture"
  ],
  "lcgft_category": "Visual works",
  "lcgft_form": "Illustrations",
  "audience": null,
  "geographic": ["South America"]
}
```

## Evaluation

### Primary Metric: Micro-F1

**Definition:** Micro-F1 aggregates true positives (TP), false positives (FP), and false negatives (FN) across all labels before computing precision and recall:

$$\text{Micro-Precision} = \frac{\sum_{i=1}^{n} TP_i}{\sum_{i=1}^{n} (TP_i + FP_i)}$$

$$\text{Micro-Recall} = \frac{\sum_{i=1}^{n} TP_i}{\sum_{i=1}^{n} (TP_i + FN_i)}$$

$$\text{Micro-F1} = 2 \times \frac{\text{Micro-Precision} \times \text{Micro-Recall}}{\text{Micro-Precision} + \text{Micro-Recall}}$$

**Why this metric?** Micro-F1 is the primary metric because it:
- Weights each prediction equally, making it suitable for imbalanced label distributions
- Provides a single aggregate measure of overall system performance
- Is commonly used in multi-label benchmarks (RCV1, EUR-Lex) for comparability
- Heavily influenced by frequent labels, reflecting real-world retrieval priorities

### Secondary Metrics

**Macro-F1:** Computes F1 for each label independently, then averages:

$$\text{Macro-F1} = \frac{1}{|\mathcal{L}|} \sum_{l \in \mathcal{L}} F1_l$$

Unlike Micro-F1, Macro-F1 treats all labels equally regardless of frequency. This reveals model performance on rare topics that might be underrepresented in Micro-F1. A large gap between Micro-F1 and Macro-F1 indicates the model performs well on frequent labels but struggles with rare ones.

**Hamming Loss:** Measures the fraction of incorrectly predicted labels (both false positives and false negatives):

$$\text{Hamming Loss} = \frac{1}{n \times |\mathcal{L}|} \sum_{i=1}^{n} |Y_i \triangle \hat{Y}_i|$$

where $\triangle$ denotes the symmetric difference. Lower is better (range: 0-1). Hamming Loss gives partial credit for partially correct predictions, making it less strict than Subset Accuracy.

**Subset Accuracy (Exact Match Ratio):** The proportion of samples where the predicted label set exactly matches the ground truth:

$$\text{Subset Accuracy} = \frac{1}{n} \sum_{i=1}^{n} \mathbb{1}[Y_i = \hat{Y}_i]$$

This is the strictest metric—only perfect predictions count. Useful for understanding complete prediction accuracy, but very penalizing in multi-label settings.

### Understanding the Multi-Label Metric Landscape

Multi-label classification requires multiple complementary metrics because each captures different aspects of model performance:

1. **Micro-F1 vs Macro-F1: The Frequency Tradeoff**
   - Micro-F1 emphasizes overall performance weighted by label frequency (good for retrieval systems where popular topics matter most)
   - Macro-F1 treats all labels equally (reveals if rare topics are neglected)
   - Example: A model that excels at "Economics" and "Government" (frequent) but fails on "Nanotechnology" (rare) will have high Micro-F1 but lower Macro-F1

2. **Hamming Loss vs Subset Accuracy: Partial Credit Philosophy**
   - Subset Accuracy demands perfection: predicting ["Economics", "Finance"] when the truth is ["Economics", "Finance", "Statistics"] gets zero credit
   - Hamming Loss gives partial credit: the same prediction earns 2/3 credit (2 correct, 1 missing)
   - In realistic scenarios, Hamming Loss better reflects utility—a cataloging system that tags 2 of 3 topics is still valuable

3. **What the Metrics Tell You:**
   - **High Micro-F1, Low Macro-F1**: Model biased toward frequent labels
   - **Low Hamming Loss, Low Subset Accuracy**: Model gets labels mostly right but rarely perfect
   - **High Subset Accuracy**: Model is highly confident and precise (rare in multi-label tasks)

### Evaluation Protocol

Models are evaluated using standard multi-label classification protocols:

1. **Threshold-based prediction**: For probabilistic models, apply a threshold (default: 0.5) to convert probability scores to binary predictions
2. **Metric computation**: Calculate all four metrics (Micro-F1, Macro-F1, Hamming Loss, Subset Accuracy) on the test set
3. **Per-domain analysis**: Optionally report metrics broken down by the 10 topic domains to identify subject-specific strengths/weaknesses

Models should return a ranked list of topics with confidence scores to enable threshold tuning on the development set.

## Baselines

| Model | Micro-F1 | Macro-F1 | Hamming Loss | Subset Acc | Notes |
|-------|----------|----------|--------------|------------|-------|
| Random | ~0.02 | ~0.02 | ~0.05 | ~0.001 | Uniform random sampling |
| Majority Baseline | ~0.15 | ~0.08 | ~0.12 | ~0.03 | Predict most frequent label set |
| TF-IDF + Logistic Regression | TBD | TBD | TBD | TBD | Binary Relevance with L2 regularization |
| BERT-base (fine-tuned) | TBD | TBD | TBD | TBD | Multi-label classification head |
| XML-CNN | TBD | TBD | TBD | TBD | CNN-based extreme multi-label approach |
| Human Expert (est.) | ~0.85 | ~0.82 | ~0.05 | ~0.65 | Upper bound estimate |

Baselines will be updated as evaluation results become available.

## Related Work

### Similar Tasks in Other Benchmarks

**EUR-Lex:** A major multi-label legal document classification benchmark containing 57,000 EU legislative documents tagged with 4,271 EUROVOC concept labels. Like SHELF Topic Classification, EUR-Lex deals with controlled vocabularies and multi-label hierarchical concepts, though at a much larger label scale.

**RCV1 (Reuters Corpus Volume 1):** Contains 804,000 newswire articles with 103 topic categories. Each document has 3.2 labels on average (similar to SHELF's 1-4 topic range). RCV1 is widely used for multi-label text classification research and serves as a standard comparison point.

**AAPD (Arxiv Academic Paper Dataset):** Computer science papers tagged with subject categories. Shares the domain-specific characteristic of SHELF but focuses on academic abstracts rather than diverse library materials.

**Differences from existing benchmarks:**
- SHELF uses real Library of Congress taxonomies (LCC, LCSH-inspired topics)
- Smaller, curated label space (113 topics) vs. thousands in EUR-Lex
- Domain-aligned topic sampling ensures coherent label combinations
- Synthetic generation allows controlled difficulty and domain distribution

### Relevant Literature

**Multi-label Classification Architectures:**
- **XML-CNN** (Liu et al.): CNN-based architecture for extreme multi-label text classification, widely used on EUR-Lex and RCV1
- **Attention-XML** (You et al.): Bi-LSTM with label-wise attention mechanisms, particularly effective when label hierarchies exist
- **LSAN** (Xiao et al.): Label-specific attention networks that capture inter-label dependencies

**Training Best Practices:**
- Binary Relevance (BR): Train independent binary classifiers per label—simple but ignores label correlations
- Classifier Chains: Sequentially chain classifiers to model label dependencies
- Label Powerset: Treat each unique label combination as a class (suffers from exponential growth)
- Neural approaches typically use sigmoid activation with binary cross-entropy loss

**Evaluation Insights:**
- Micro-F1 is standard for comparing with existing benchmarks (EUR-Lex, RCV1)
- Macro-F1 essential for evaluating performance on rare labels
- Modern approaches report P@k and nDCG@k for ranking quality

## Implementation Notes

### Running Evaluation

```bash
# Generate the topic classification dataset
shelf generate --task topic_classification --split train --size 10000

# Evaluate a model
shelf evaluate --task topic_classification --model your_model.py

# Evaluate with custom threshold
shelf evaluate --task topic_classification --model your_model.py --threshold 0.3
```

### Submission Format

Predictions should be submitted as JSONL with one prediction per line:

```json
{"id": "doc_001", "topics": ["Economics", "Finance"], "scores": {"Economics": 0.87, "Finance": 0.72, ...}}
{"id": "doc_002", "topics": ["Climate change", "Conservation"], "scores": {"Climate change": 0.91, ...}}
```

Required fields:
- `id`: Document identifier matching the test set
- `topics`: List of predicted topic labels (strings from the 113-topic vocabulary)
- `scores`: Dictionary mapping each label in the vocabulary to a confidence score [0, 1]

The evaluation script will compute metrics using both the predicted label sets and the full score distributions (for threshold sensitivity analysis).

## References

1. [Library of Congress Subject Headings - Wikipedia](https://en.wikipedia.org/wiki/Library_of_Congress_Subject_Headings)
2. [Subject Headings and Genre/Form Terms - Library of Congress](https://www.loc.gov/aba/cataloging/subject/)
3. [Controlled Vocabularies - Library of Congress](https://www.loc.gov/librarians/controlled-vocabularies/)
4. [Large-Scale Multi-Label Text Classification on EU Legislation (ACL 2019)](https://aclanthology.org/P19-1636.pdf)
5. [RCV1 Benchmark - Papers With Code](https://paperswithcode.com/sota/multi-label-text-classification-on-rcv1)
6. [Evaluating Multi-label Classifiers - Towards Data Science](https://towardsdatascience.com/evaluating-multi-label-classifiers-a31be83da6ea/)
7. [Precision, Recall, Accuracy, and F1 Score for Multi-Label Classification - Medium](https://medium.com/synthesio-engineering/precision-accuracy-and-f1-score-for-multi-label-classification-34ac6bdfb404)
8. [Understanding Micro and Macro Averages - SafJan](https://safjan.com/micro-and-macro-averages-in-multiclass-multilabel-problems/)
9. [Micro, Macro & Weighted Averages of F1 Score - Towards Data Science](https://towardsdatascience.com/micro-macro-weighted-averages-of-f1-score-clearly-explained-b603420b292f/)
10. [scikit-learn Metrics and Scoring Documentation](https://scikit-learn.org/stable/modules/model_evaluation.html)

---
*Last updated: 2025-12-10*

# Task: LCC Classification

> **Task Type**: classification
> **Difficulty**: medium
> **Primary Metric**: Macro-F1

## Overview

The LCC Classification task evaluates a model's ability to assign Library of Congress Classification (LCC) codes to documents based on their textual content. This task tests whether machine learning models can replicate the subject classification decisions made by professional catalogers at academic and research libraries. The task is fundamental to information retrieval and digital library systems, where accurate subject classification enables effective organization, discovery, and serendipitous browsing of large document collections.

The Library of Congress Classification system divides all knowledge into 21 basic classes, each identified by a single letter (A-Z, excluding I, O, W, X, Y). Unlike hierarchical systems like Dewey Decimal, LCC was designed pragmatically for the actual collections held by the Library of Congress and has become one of the most widely used classification systems in academic libraries worldwide. Successfully automating this classification task would significantly reduce the manual effort required for cataloging while maintaining the intellectual organization that enables effective research and discovery.

## Task Definition

### Input
The model receives a document consisting of two text fields:
- **title**: The document's title (typically 5-20 words)
- **body**: The document's main text content (ranging from 25 to 1000+ words)

Models must process the concatenated or separately encoded title and body text to predict the appropriate LCC main class.

### Output
The model must predict a single LCC code from the 21 possible main classes (A, B, C, D, E, F, G, H, J, K, L, M, N, P, Q, R, S, T, U, V, Z). This is a single-label, multi-class classification task where exactly one class must be assigned to each document.

### Formal Definition
Given a document d with title t and body b, the task is to learn a function f: (t, b) → c where c ∈ C and C = {A, B, C, D, E, F, G, H, J, K, L, M, N, P, Q, R, S, T, U, V, Z} represents the set of 21 LCC main classes.

## Dataset

### Source
The SHELF dataset consists of synthetically generated documents designed to represent the diversity of materials found in Library of Congress collections. Each document is generated using large language models (GPT-5.1) with carefully controlled prompts that specify:
- Target LCC class and associated topics
- Genre/form type (LCGFT)
- Document length and writing style
- Optional audience and geographic focus

This synthetic generation approach ensures balanced representation across all 21 LCC classes while incorporating realistic variation in document types, lengths, and subject matter. Documents are labeled with their ground-truth LCC code, which was used during generation to ensure topic-class alignment.

### Statistics
| Split | Documents | Classes | Avg Length | Length Range |
|-------|-----------|---------|------------|--------------|
| Train | TBD | 21 | ~200 words | 25-1000+ words |
| Dev   | TBD | 21 | ~200 words | 25-1000+ words |
| Test  | TBD | 21 | ~200 words | 25-1000+ words |

Note: Dataset is designed with approximately balanced class distribution to enable fair evaluation across all subject areas.

### Label Space
The 21 Library of Congress Classification main classes are:

| Code | Class Name | Description |
|------|------------|-------------|
| A | General Works | General encyclopedias, general periodicals, museums, general reference |
| B | Philosophy, Psychology, Religion | Philosophy, metaphysics, psychology, religion, theology |
| C | Auxiliary Sciences of History | Archaeology, genealogy, biography as history science |
| D | World History (except Americas) | History of Europe, Asia, Africa, Australia, Oceania |
| E | History of the Americas (general, US) | General American history, United States history |
| F | History of the Americas (local) | Local history of the Americas, Canada, Latin America |
| G | Geography, Anthropology, Recreation | Geography, maps, anthropology, sports, games, recreation |
| H | Social Sciences | Statistics, economics, sociology, commerce, finance |
| J | Political Science | Political science, government, international relations |
| K | Law | Law in general, comparative law, all legal systems |
| L | Education | Education, theory, practice, higher education |
| M | Music | Music scores, musicology, music history |
| N | Fine Arts | Visual arts, architecture, sculpture, painting, drawing |
| P | Language and Literature | Linguistics, classical and modern languages, literature |
| Q | Science | Science (general), mathematics, physics, chemistry, natural sciences |
| R | Medicine | Medicine, surgery, public health, nursing |
| S | Agriculture | Agriculture, forestry, wildlife management |
| T | Technology | Engineering, manufacturing, mechanical arts, applied sciences |
| U | Military Science | Military science, army, tactics, weapons |
| V | Naval Science | Naval science, navy, naval history, weapons |
| Z | Bibliography, Library Science | Books, libraries, bibliography, information science |

Note: Letters I, O, W, X, and Y are not used in the LCC system.

### Data Format
```json
{
  "id": "20251211_030155_57dfc238",
  "title": "Quarterly Agribusiness Risk Memo on Emerging Externalities",
  "body": "In light of climate change, migratory wildlife now attends board meetings...",
  "word_count": 37,
  "lcc_code": "S",
  "lcc_name": "Agriculture",
  "lcc_uri": "http://id.loc.gov/authorities/classification/S",
  "lcgft_category": "Literature",
  "lcgft_form": "Satire",
  "topics": ["Climate change", "Wildlife", "Neuroscience", "Ocean conservation"],
  "audience": null,
  "geographic": [],
  "target_length": "tiny",
  "target_word_range": [25, 50],
  "register": "professional",
  "model": "gpt-5.1"
}
```

Only the `id`, `title`, and `body` fields are provided to models during evaluation. The `lcc_code` field contains the ground truth label.

## Evaluation

### Primary Metric
**Macro-F1**: The arithmetic mean of F1 scores computed independently for each of the 21 classes.

Macro-F1 is the appropriate primary metric for this task because:
1. **Class Equality**: All LCC classes are equally important in library science; no subject area should be neglected
2. **Imbalance Handling**: Unlike accuracy or micro-F1, macro-F1 treats all classes equally regardless of their frequency, preventing performance to be dominated by common classes
3. **Comprehensive Performance**: A high macro-F1 score indicates the model performs well across all subject areas, not just the most frequent ones
4. **Library Practice Alignment**: In real cataloging workflows, accurate classification is needed for all disciplines, making per-class performance critical

Formally, Macro-F1 = (1/21) × Σᵢ F1ᵢ where F1ᵢ is the F1 score for class i, computed in a one-vs-rest manner.

### Secondary Metrics
- **Accuracy**: Overall classification accuracy across all documents
- **Micro-F1**: Global F1 computed from total true positives, false positives, and false negatives
- **Weighted-F1**: F1 averaged with weights proportional to class support
- **Per-Class F1**: Individual F1 scores for each of the 21 classes for detailed error analysis
- **Confusion Matrix**: To identify systematic misclassification patterns (e.g., D vs E for history, Q vs R vs T for sciences)

### Evaluation Protocol
Models are evaluated on a held-out test set that was not used during training. The evaluation process:
1. Models receive only the `id`, `title`, and `body` fields for each test document
2. Models must output a single predicted LCC code (one of the 21 valid classes)
3. Predictions are compared against ground truth `lcc_code` labels
4. Macro-F1 and secondary metrics are computed using scikit-learn's classification metrics
5. Statistical significance testing may be applied when comparing model performance

Invalid predictions (codes not in the 21-class set) are treated as incorrect classifications.

## Baselines

| Model | Macro-F1 | Accuracy | Micro-F1 | Notes |
|-------|----------|----------|----------|-------|
| Random | 0.048 | 0.048 | 0.048 | Uniform random baseline (1/21) |
| Majority Class | 0.024 | ~0.048 | ~0.048 | Always predicts most frequent class |
| TF-IDF + Logistic Regression | TBD | TBD | TBD | Bag-of-words baseline |
| TF-IDF + SVM | TBD | TBD | TBD | Linear SVM on TF-IDF features |
| FastText | TBD | TBD | TBD | Word embeddings baseline |
| BERT-base | TBD | TBD | TBD | Fine-tuned transformer baseline |
| RoBERTa-large | TBD | TBD | TBD | Large transformer model |
| Domain-Adapted BERT | TBD | TBD | TBD | BERT pre-trained on library/academic text |
| Human Cataloger | ~0.95 | ~0.95 | ~0.95 | Upper bound estimate (professional librarians) |

Note: Baseline results will be updated as benchmark evaluations are completed.

## Related Work

### Similar Tasks in Other Benchmarks
The LCC Classification task is comparable to document classification tasks in existing NLP benchmarks:

- **AG News / DBpedia (from GLUE/SuperGLUE precursors)**: Topic classification into 4-14 categories, but with simpler subject distinctions
- **20 Newsgroups**: Classic 20-class document classification, though focused on online forum topics rather than formal subject classification
- **Reuters-21578**: Multi-label news categorization, but with different domain and granularity
- **ArXiv Subject Classification**: Academic paper classification into CS/Physics/Math categories, similar domain but different taxonomy
- **MTEB Classification tasks**: Various classification tasks but none focused on library/bibliographic subject classification

LCC Classification is unique in:
1. Using the formal, professionally-maintained Library of Congress taxonomy
2. Covering the full breadth of human knowledge (21 diverse subject areas)
3. Reflecting real-world library cataloging decisions
4. Including diverse document types (academic, literary, technical, creative) within each class

### Relevant Literature

**Automated LCC Prediction:**
- Frank & Paynter (2004) pioneered predicting Library of Congress Classifications from Library of Congress Subject Headings using machine learning on 50,000 catalog records
- Ahmad's auto_lcc project explored automatic LCC using word embeddings from book titles and synopses, achieving 76% accuracy with LSTM models on 15 classes
- Recent work has applied BERT models to library subject indexing tasks, showing promise for assisted cataloging

**Text Classification Methods:**
- Deep learning approaches (CNN, LSTM, Transformers) have largely superseded traditional machine learning for text classification
- Transformer-based models (BERT, RoBERTa) achieve state-of-the-art results on most text classification benchmarks
- Comprehensive reviews show modern NLP systems can handle multi-class classification effectively when sufficient training data exists

**Evaluation Metrics:**
- Macro-F1 is widely recommended for multi-class classification with balanced class importance
- Studies confirm macro-averaging prevents performance being dominated by majority classes
- Per-class F1 analysis is essential for identifying subject areas where models struggle

## Implementation Notes

### Running Evaluation
```bash
# Evaluate a model on LCC classification task
shelf evaluate --task lcc_classification --model your_model

# Evaluate with detailed per-class metrics
shelf evaluate --task lcc_classification --model your_model --detailed

# Generate confusion matrix analysis
shelf evaluate --task lcc_classification --model your_model --confusion-matrix
```

### Submission Format
Predictions should be submitted as a JSON Lines file with one prediction per line:
```json
{"id": "20251211_030155_57dfc238", "predicted_lcc": "S"}
{"id": "20251211_030155_f3221f5a", "predicted_lcc": "P"}
{"id": "20251211_030155_c0d93a36", "predicted_lcc": "C"}
```

Each prediction must include:
- `id`: The document ID (string)
- `predicted_lcc`: The predicted LCC code (one of the 21 valid classes)

### Tips for Model Development
1. **Consider title and body separately**: Titles often contain strong subject signals; experiment with different fusion strategies
2. **Watch for class confusion**: History classes (C, D, E, F) and science classes (Q, R, S, T) are semantically related and may be confused
3. **Domain vocabulary matters**: Subject-specific terminology is a strong signal (e.g., legal terms → K, medical terms → R)
4. **Document length varies**: Handle both short abstracts (25 words) and long documents (1000+ words) effectively
5. **Multi-topic documents**: Some documents span multiple subjects; the LCC code reflects the primary focus

## References

1. [Library of Congress Classification System](https://www.loc.gov/catdir/cpso/lcc.html) - Official LCC documentation from the Library of Congress
2. [Library of Congress Classification - Wikipedia](https://en.wikipedia.org/wiki/Library_of_Congress_Classification) - Comprehensive overview of LCC history and structure
3. Frank, E., & Paynter, G. W. (2004). [Predicting Library of Congress classifications from Library of Congress subject headings](https://onlinelibrary.wiley.com/doi/abs/10.1002/asi.10360). Journal of the American Society for Information Science and Technology, 55(3), 214-227.
4. [Automatic library of congress classification using word embeddings](https://github.com/ahmad-PH/auto_lcc) - GitHub repository demonstrating LSTM-based LCC prediction
5. [An Analysis of BERT for Assisted Subject Indexing](https://www.tandfonline.com/doi/abs/10.1080/01639374.2022.2138666) - Research on using BERT for Library of Congress subject indexing
6. [Deep Learning Based Text Classification: A Comprehensive Review](https://arxiv.org/pdf/2004.03705) - Comprehensive survey of text classification methods
7. [Micro, Macro & Weighted Averages of F1 Score, Clearly Explained](https://www.kdnuggets.com/2023/01/micro-macro-weighted-averages-f1-score-clearly-explained.html) - Guide to understanding evaluation metrics for multi-class classification
8. [scikit-learn F1 Score Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html) - Official documentation for F1 score computation
9. [Text Classification benchmarks](https://paperswithcode.com/task/text-classification) - Papers with Code overview of text classification tasks and state-of-the-art results
10. [Library of Congress Classification Outline](https://www.loc.gov/catdir/cpso/lcco/) - Complete outline of all LCC classes and subclasses

---
*Last updated: 2025-12-10*

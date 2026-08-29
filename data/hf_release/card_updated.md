---
pretty_name: 'SHELF: Synthetic Harness for Evaluating LLM Fitness'
license: cc-by-4.0
language:
- en
size_categories:
- 10K<n<100K
task_categories:
- text-classification
- text-retrieval
- sentence-similarity
task_ids:
- multi-class-classification
- multi-label-classification
- document-retrieval
- semantic-similarity-classification
tags:
- library-science
- library-of-congress
- bibliographic-classification
- document-classification
- taxonomy
- benchmark
- synthetic-data
- lcc
- lcgft
- lcsh
dataset_info:
- config_name: default
  features:
  - name: id
    dtype: string
  - name: text
    dtype: string
  - name: title
    dtype: string
  - name: body
    dtype: string
  - name: word_count
    dtype: int32
  - name: lcc_code
    dtype: string
  - name: lcc_name
    dtype: string
  - name: lcc_uri
    dtype: string
  - name: lcgft_category
    dtype: string
  - name: lcgft_form
    dtype: string
  - name: topics
    list: string
  - name: geographic
    list: string
  - name: audience
    dtype: string
  - name: register
    dtype: string
  - name: register_description
    dtype: string
  - name: target_length
    dtype: string
  - name: target_word_range
    list: int32
  - name: temperature
    dtype: float32
  - name: top_p
    dtype: float32
  - name: model
    dtype: string
  - name: git_commit
    dtype: string
  - name: code_version
    dtype: string
  - name: thinking_budget
    dtype: int32
  - name: token_multiplier
    dtype: float32
  splits:
  - name: train
    num_bytes: 246747930
    num_examples: 25518
  - name: validation
    num_bytes: 82243064
    num_examples: 8507
  - name: test
    num_bytes: 81135549
    num_examples: 8507
  download_size: 245080800
  dataset_size: 410126543
- config_name: same_audience_pairs
  features:
  - name: id
    dtype: string
  - name: doc_a_id
    dtype: string
  - name: doc_a_title
    dtype: string
  - name: doc_a_body
    dtype: string
  - name: doc_b_id
    dtype: string
  - name: doc_b_title
    dtype: string
  - name: doc_b_body
    dtype: string
  - name: label
    dtype: int32
  - name: label_field
    dtype: string
  splits:
  - name: train
    num_bytes: 188361149
    num_examples: 20000
  - name: validation
    num_bytes: 38042754
    num_examples: 4000
  - name: test
    num_bytes: 38026911
    num_examples: 4000
  download_size: 161727072
  dataset_size: 264430814
- config_name: same_form_pairs
  features:
  - name: id
    dtype: string
  - name: doc_a_id
    dtype: string
  - name: doc_a_title
    dtype: string
  - name: doc_a_body
    dtype: string
  - name: doc_b_id
    dtype: string
  - name: doc_b_title
    dtype: string
  - name: doc_b_body
    dtype: string
  - name: label
    dtype: int32
  - name: label_field
    dtype: string
  splits:
  - name: train
    num_bytes: 189871627
    num_examples: 20000
  - name: validation
    num_bytes: 39843315
    num_examples: 4000
  - name: test
    num_bytes: 38232674
    num_examples: 4000
  download_size: 163796371
  dataset_size: 267947616
- config_name: same_lcc_pairs
  features:
  - name: id
    dtype: string
  - name: doc_a_id
    dtype: string
  - name: doc_a_title
    dtype: string
  - name: doc_a_body
    dtype: string
  - name: doc_b_id
    dtype: string
  - name: doc_b_title
    dtype: string
  - name: doc_b_body
    dtype: string
  - name: label
    dtype: int32
  - name: label_field
    dtype: string
  splits:
  - name: train
    num_bytes: 191701147
    num_examples: 20000
  - name: validation
    num_bytes: 37941452
    num_examples: 4000
  - name: test
    num_bytes: 36859442
    num_examples: 4000
  download_size: 163100107
  dataset_size: 266502041
- config_name: same_register_pairs
  features:
  - name: id
    dtype: string
  - name: doc_a_id
    dtype: string
  - name: doc_a_title
    dtype: string
  - name: doc_a_body
    dtype: string
  - name: doc_b_id
    dtype: string
  - name: doc_b_title
    dtype: string
  - name: doc_b_body
    dtype: string
  - name: label
    dtype: int32
  - name: label_field
    dtype: string
  splits:
  - name: train
    num_bytes: 186993500
    num_examples: 20000
  - name: validation
    num_bytes: 38341972
    num_examples: 4000
  - name: test
    num_bytes: 36868352
    num_examples: 4000
  download_size: 161387541
  dataset_size: 262203824
- config_name: same_topic_pairs
  features:
  - name: id
    dtype: string
  - name: doc_a_id
    dtype: string
  - name: doc_a_title
    dtype: string
  - name: doc_a_body
    dtype: string
  - name: doc_b_id
    dtype: string
  - name: doc_b_title
    dtype: string
  - name: doc_b_body
    dtype: string
  - name: label
    dtype: int32
  - name: overlap_count
    dtype: int32
  - name: shared_topics
    list: string
  splits:
  - name: train
    num_bytes: 191865252
    num_examples: 20000
  - name: validation
    num_bytes: 37757465
    num_examples: 4000
  - name: test
    num_bytes: 38049571
    num_examples: 4000
  download_size: 163711952
  dataset_size: 267672288
- config_name: topic_overlap_pairs
  features:
  - name: id
    dtype: string
  - name: doc_a_id
    dtype: string
  - name: doc_a_title
    dtype: string
  - name: doc_a_body
    dtype: string
  - name: doc_b_id
    dtype: string
  - name: doc_b_title
    dtype: string
  - name: doc_b_body
    dtype: string
  - name: label
    dtype: int32
  - name: overlap_count
    dtype: int32
  - name: shared_topics
    list: string
  splits:
  - name: train
    num_bytes: 184521053
    num_examples: 19305
  - name: validation
    num_bytes: 36576885
    num_examples: 3880
  - name: test
    num_bytes: 36543575
    num_examples: 3819
  download_size: 157556182
  dataset_size: 257641513
- config_name: v0_4_core
  features:
  - name: id
    dtype: string
  - name: title
    dtype: string
  - name: body
    dtype: string
  - name: word_count
    dtype: int64
  - name: lcc_code
    dtype: string
  - name: lcc_name
    dtype: string
  - name: lcc_uri
    dtype: string
  - name: lcgft_category
    dtype: string
  - name: lcgft_form
    dtype: string
  - name: topics
    list: string
  - name: audience
    dtype: string
  - name: geographic
    list: string
  - name: target_length
    dtype: string
  - name: target_word_range
    list: string
  - name: register
    dtype: string
  - name: register_description
    dtype: string
  - name: temperature
    dtype: float64
  - name: top_p
    dtype: float64
  - name: model
    dtype: string
  - name: model_resolved
    dtype: string
  - name: provider
    dtype: string
  - name: provider_served
    dtype: string
  - name: prompt
    dtype: string
  - name: prompt_variant_id
    dtype: string
  - name: spec_id
    dtype: string
  - name: block_id
    dtype: string
  - name: run_id
    dtype: string
  - name: input_tokens
    dtype: int64
  - name: output_tokens
    dtype: int64
  - name: reasoning_tokens
    dtype: int64
  - name: cost_usd
    dtype: float64
  - name: git_commit
    dtype: string
  - name: git_dirty
    dtype: bool
  - name: git_branch
    dtype: string
  - name: code_version
    dtype: string
- config_name: v0_4_supplement
  features:
  - name: id
    dtype: string
  - name: title
    dtype: string
  - name: body
    dtype: string
  - name: word_count
    dtype: int64
  - name: lcc_code
    dtype: string
  - name: lcc_name
    dtype: string
  - name: lcc_uri
    dtype: string
  - name: lcgft_category
    dtype: string
  - name: lcgft_form
    dtype: string
  - name: topics
    list: string
  - name: audience
    dtype: string
  - name: geographic
    list: string
  - name: target_length
    dtype: string
  - name: target_word_range
    list: string
  - name: register
    dtype: string
  - name: register_description
    dtype: string
  - name: temperature
    dtype: float64
  - name: top_p
    dtype: float64
  - name: model
    dtype: string
  - name: model_resolved
    dtype: string
  - name: provider
    dtype: string
  - name: provider_served
    dtype: string
  - name: prompt
    dtype: string
  - name: prompt_variant_id
    dtype: string
  - name: spec_id
    dtype: string
  - name: block_id
    dtype: string
  - name: run_id
    dtype: string
  - name: input_tokens
    dtype: int64
  - name: output_tokens
    dtype: int64
  - name: reasoning_tokens
    dtype: int64
  - name: cost_usd
    dtype: float64
  - name: git_commit
    dtype: string
  - name: git_dirty
    dtype: bool
  - name: git_branch
    dtype: string
  - name: code_version
    dtype: string
- config_name: v0_4_minimal_pairs
  features:
  - name: id
    dtype: string
  - name: title
    dtype: string
  - name: body
    dtype: string
  - name: word_count
    dtype: int64
  - name: lcc_code
    dtype: string
  - name: lcc_name
    dtype: string
  - name: lcc_uri
    dtype: string
  - name: lcgft_category
    dtype: string
  - name: lcgft_form
    dtype: string
  - name: topics
    list: string
  - name: audience
    dtype: string
  - name: geographic
    list: string
  - name: target_length
    dtype: string
  - name: target_word_range
    list: string
  - name: register
    dtype: string
  - name: register_description
    dtype: string
  - name: temperature
    dtype: float64
  - name: top_p
    dtype: float64
  - name: model
    dtype: string
  - name: model_resolved
    dtype: string
  - name: provider
    dtype: string
  - name: provider_served
    dtype: string
  - name: prompt
    dtype: string
  - name: prompt_variant_id
    dtype: string
  - name: spec_id
    dtype: string
  - name: block_id
    dtype: string
  - name: run_id
    dtype: string
  - name: input_tokens
    dtype: int64
  - name: output_tokens
    dtype: int64
  - name: reasoning_tokens
    dtype: int64
  - name: cost_usd
    dtype: float64
  - name: git_commit
    dtype: string
  - name: git_dirty
    dtype: bool
  - name: git_branch
    dtype: string
  - name: code_version
    dtype: string
  - name: pair_id
    dtype: string
  - name: pair_role
    dtype: string
  - name: pair_axis
    dtype: string
- config_name: v0_4_holdout
  features:
  - name: id
    dtype: string
  - name: title
    dtype: string
  - name: body
    dtype: string
  - name: word_count
    dtype: int64
  - name: lcc_code
    dtype: string
  - name: lcc_name
    dtype: string
  - name: lcc_uri
    dtype: string
  - name: lcgft_category
    dtype: string
  - name: lcgft_form
    dtype: string
  - name: topics
    list: string
  - name: audience
    dtype: string
  - name: geographic
    list: string
  - name: target_length
    dtype: string
  - name: target_word_range
    list: string
  - name: register
    dtype: string
  - name: register_description
    dtype: string
  - name: temperature
    dtype: float64
  - name: top_p
    dtype: float64
  - name: model
    dtype: string
  - name: model_resolved
    dtype: string
  - name: provider
    dtype: string
  - name: provider_served
    dtype: string
  - name: prompt
    dtype: string
  - name: prompt_variant_id
    dtype: string
  - name: spec_id
    dtype: string
  - name: block_id
    dtype: string
  - name: run_id
    dtype: string
  - name: input_tokens
    dtype: int64
  - name: output_tokens
    dtype: int64
  - name: reasoning_tokens
    dtype: string
  - name: cost_usd
    dtype: float64
  - name: git_commit
    dtype: string
  - name: git_dirty
    dtype: bool
  - name: git_branch
    dtype: string
  - name: code_version
    dtype: string
- config_name: transfer_gutenberg
  features:
  - name: id
    dtype: string
  - name: text
    dtype: string
  - name: body
    dtype: string
  - name: title
    dtype: string
  - name: word_count
    dtype: int64
  - name: lcc_code
    dtype: string
  - name: lcc_name
    dtype: string
  - name: lcc_uri
    dtype: string
  - name: lcgft_category
    dtype: string
  - name: lcgft_form
    dtype: string
  - name: topics
    list: string
  - name: geographic
    list: string
  - name: audience
    dtype: string
  - name: register
    dtype: string
  - name: language
    dtype: string
  - name: author
    dtype: string
  - name: source_type
    dtype: string
  - name: source
    dtype: string
  - name: contamination_status
    dtype: string
  - name: label_space
    dtype: string
  - name: schema_version
    dtype: string
  - name: provenance
    dtype: int64
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: validation
    path: data/validation-*
  - split: test
    path: data/test-*
- config_name: same_audience_pairs
  data_files:
  - split: train
    path: same_audience_pairs/train-*
  - split: validation
    path: same_audience_pairs/validation-*
  - split: test
    path: same_audience_pairs/test-*
- config_name: same_form_pairs
  data_files:
  - split: train
    path: same_form_pairs/train-*
  - split: validation
    path: same_form_pairs/validation-*
  - split: test
    path: same_form_pairs/test-*
- config_name: same_lcc_pairs
  data_files:
  - split: train
    path: same_lcc_pairs/train-*
  - split: validation
    path: same_lcc_pairs/validation-*
  - split: test
    path: same_lcc_pairs/test-*
- config_name: same_register_pairs
  data_files:
  - split: train
    path: same_register_pairs/train-*
  - split: validation
    path: same_register_pairs/validation-*
  - split: test
    path: same_register_pairs/test-*
- config_name: same_topic_pairs
  data_files:
  - split: train
    path: same_topic_pairs/train-*
  - split: validation
    path: same_topic_pairs/validation-*
  - split: test
    path: same_topic_pairs/test-*
- config_name: topic_overlap_pairs
  data_files:
  - split: train
    path: topic_overlap_pairs/train-*
  - split: validation
    path: topic_overlap_pairs/validation-*
  - split: test
    path: topic_overlap_pairs/test-*
- config_name: v0_4_core
  data_files:
  - split: train
    path: v0_4_core/train-*
  - split: validation
    path: v0_4_core/validation-*
  - split: test
    path: v0_4_core/test-*
- config_name: v0_4_supplement
  data_files:
  - split: train
    path: v0_4_supplement/train-*
  - split: validation
    path: v0_4_supplement/validation-*
  - split: test
    path: v0_4_supplement/test-*
- config_name: v0_4_minimal_pairs
  data_files:
  - split: train
    path: v0_4_minimal_pairs/train-*
  - split: validation
    path: v0_4_minimal_pairs/validation-*
  - split: test
    path: v0_4_minimal_pairs/test-*
- config_name: v0_4_holdout
  data_files:
  - split: train
    path: v0_4_holdout/train-*
  - split: validation
    path: v0_4_holdout/validation-*
  - split: test
    path: v0_4_holdout/test-*
- config_name: transfer_gutenberg
  data_files:
  - split: test
    path: transfer_gutenberg/test-*
- config_name: all
  data_files:
  - split: train
    path: all/train-00000-of-00001.parquet
  - split: validation
    path: all/validation-00000-of-00001.parquet
  - split: test
    path: all/test-00000-of-00001.parquet
- config_name: transfer_lcshbench
  data_files:
  - split: train
    path: transfer_lcshbench/train-00000-of-00001.parquet
  - split: validation
    path: transfer_lcshbench/validation-00000-of-00001.parquet
  - split: test
    path: transfer_lcshbench/test-00000-of-00001.parquet
version: 0.4.0
---
# SHELF: Synthetic Harness for Evaluating LLM Fitness

SHELF is a synthetic benchmark for evaluating language model fitness on bibliographic classification, retrieval, and clustering tasks using Library of Congress taxonomies.

## Dataset Description

- **Homepage:** https://github.com/mjbommar/shelf
- **Repository:** https://github.com/mjbommar/shelf
- **Paper:** Forthcoming (cite repository for now)
- **License:** cc-by-4.0
- **Version:** 0.3.1

### Dataset Summary

SHELF contains 62,899 synthetic documents annotated with Library of Congress taxonomies, plus 3,016 natural Project Gutenberg documents used only as a transfer control (65,915 in total). The v0.3.1 corpus described immediately below is 42,532 of that synthetic total; the rest is described under `v0.4 slices`:

- **LCC (Library of Congress Classification):** 21 subject classes (A-Z)
- **LCGFT (Library of Congress Genre/Form Terms):** 14 categories, 133 specific forms
- **Topics:** 112 subject headings (multi-label)
- **Geographic:** 44 locations mapped to 8 regions (multi-label)
- **Audience:** 25 target audience types
- **Register:** 8 writing styles (academic, professional, casual, etc.)

The dataset is designed for:
1. **Document Classification** - Predicting LCC codes, LCGFT forms, topics, audiences
2. **Document Retrieval** - Finding similar documents by subject, genre, or topic
3. **Document Clustering** - Grouping documents by subject, genre, or geographic region
4. **Pair Classification** - Determining if document pairs share categories

### Supported Tasks

| Task | Type | Classes | Primary Metric |
|------|------|---------|----------------|
| LCC Classification | Single-label | 21 | Macro-F1 |
| LCGFT Form Classification | Single-label | 133 | Macro-F1 |
| Topic Classification | Multi-label | 112 | Micro-F1 |
| Audience Classification | Single-label | 25 | Macro-F1 |
| Register Classification | Single-label | 8 | Macro-F1 |
| Subject Retrieval | Retrieval | - | NDCG@10 |
| Document Clustering | Clustering | 21/14/8 | V-measure |

### Languages

English only.

## Dataset Structure

### Data Instances

```json
{
  "id": "20251211_123456_abcd1234",
  "title": "Introduction to Machine Learning",
  "body": "This comprehensive guide covers the fundamentals of machine learning...",
  "word_count": 450,
  "lcc_code": "Q",
  "lcc_name": "Science",
  "lcc_uri": "http://id.loc.gov/authorities/classification/Q",
  "lcgft_category": "Instructional and educational works",
  "lcgft_form": "Textbooks",
  "topics": ["Computer science", "Artificial intelligence"],
  "geographic": ["United States"],
  "audience": "Students",
  "register": "academic",
  "register_description": "academic and scholarly, suitable for research contexts",
  "target_length": "medium",
  "target_word_range": [300, 500]
}
```

### Data Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique document identifier |
| `title` | string | Document title |
| `body` | string | Full document text |
| `word_count` | int | Number of words in body |
| `lcc_code` | string | Library of Congress Classification code (A-Z) |
| `lcc_name` | string | Human-readable LCC class name |
| `lcc_uri` | string | LOC authority URI |
| `lcgft_category` | string | LCGFT broad category (14 options) |
| `lcgft_form` | string | LCGFT specific form (133 options) |
| `topics` | list[string] | Subject headings (multi-label) |
| `geographic` | list[string] | Geographic locations mentioned |
| `audience` | string | Target audience (nullable) |
| `register` | string | Writing style/register |
| `register_description` | string | Description of the register |
| `target_length` | string | Target length category |
| `target_word_range` | list[int] | Target word count range [min, max] |
| `temperature` | float | Generation temperature |
| `top_p` | float | Generation top-p value |
| `model` | string | LLM used for generation |
| `git_commit` | string | Code version hash |
| `thinking_budget` | int | Thinking token budget (-1 if N/A) |
| `token_multiplier` | float | Output token multiplier |

### Data Splits

| Split | Documents | Percentage |
|-------|-----------|------------|
| Train | 25,518 | 60.0% |
| Validation | 8,507 | 20.0% |
| Test | 8,507 | 20.0% |

Splits are stratified by LCC code and LCGFT category to ensure balanced label distributions.

## Dataset Creation

### Source Data

Documents were synthetically generated using multiple frontier language models with carefully designed prompts to create realistic examples across all taxonomy categories:

**Generation Models:**
- **OpenAI:** GPT-5.1, GPT-5.2 (primary generators, ~94% of corpus)
- **Google:** Gemini 2.5 Flash, Gemini 2.5 Flash Lite, Gemini 2.5 Pro, Gemini 3 Pro Preview
- **Anthropic:** Claude Haiku 4.5, Claude Sonnet 4.5, Claude Opus 4.5

The generation process ensures:

- Balanced distribution across all 21 LCC codes
- Coverage of all 133 LCGFT forms
- Diverse topics, audiences, and registers
- Varied document lengths (12 to 6,000+ words)
- Multi-model diversity to reduce single-model biases

### Quality Filtering

The following quality filters were applied:
- Empty document removal
- Non-English content detection and removal
- Length validation against target ranges

### Annotations

All annotations are generated alongside the documents using structured prompting. Labels represent the intended classification as specified in the generation prompt.

## Usage

### Loading the Dataset

```python
from datasets import load_dataset

# Load full dataset (default config - individual documents)
dataset = load_dataset("mjbommar/SHELF")

# Load specific split
train = load_dataset("mjbommar/SHELF", split="train")
test = load_dataset("mjbommar/SHELF", split="test")

# Access examples
print(train[0])
```

### Dataset Configurations

The dataset has multiple configurations:

| Config | Description | Train | Val | Test |
|--------|-------------|-------|-----|------|
| `default` | Individual documents with all metadata | 12,000 | 4,000 | 4,000 |
| `same_lcc_pairs` | Document pairs labeled by LCC match | 20,000 | 4,000 | 4,000 |
| `same_form_pairs` | Document pairs labeled by LCGFT form match | 20,000 | 4,000 | 4,000 |
| `same_audience_pairs` | Document pairs labeled by audience match | 20,000 | 4,000 | 4,000 |
| `same_register_pairs` | Document pairs labeled by register/style match | 20,000 | 4,000 | 4,000 |
| `same_topic_pairs` | Binary: Do documents share ANY topic? | 20,000 | 4,000 | 4,000 |
| `topic_overlap_pairs` | Graded: How many topics shared? (0/1/2/3+) | 20,000 | 4,000 | 4,000 |

```python
# Load pair classification data
lcc_pairs = load_dataset("mjbommar/SHELF", name="same_lcc_pairs")
form_pairs = load_dataset("mjbommar/SHELF", name="same_form_pairs")
audience_pairs = load_dataset("mjbommar/SHELF", name="same_audience_pairs")
register_pairs = load_dataset("mjbommar/SHELF", name="same_register_pairs")

# Load topic overlap pairs
topic_binary = load_dataset("mjbommar/SHELF", name="same_topic_pairs")
topic_graded = load_dataset("mjbommar/SHELF", name="topic_overlap_pairs")

# Pair format (categorical pairs)
print(lcc_pairs["train"][0])
# {'id': 'pair_000001', 'doc_a_id': '...', 'doc_a_title': '...',
#   'doc_a_body': '...', 'doc_b_id': '...', 'doc_b_title': '...',
#   'doc_b_body': '...', 'label': 1, 'label_field': 'lcc_code'}

# Topic overlap format (multi-label pairs)
print(topic_graded["train"][0])
# {'id': 'pair_000001', 'doc_a_id': '...', 'doc_a_title': '...',
#   'doc_a_body': '...', 'doc_b_id': '...', 'doc_b_title': '...',
#   'doc_b_body': '...', 'label': 2, 'overlap_count': 2,
#   'shared_topics': ['Ethics', 'Philosophy']}
```

### Classification Example

```python
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load data
dataset = load_dataset("mjbommar/SHELF")

# Prepare for LCC classification
X_train = dataset["train"]["body"]
y_train = dataset["train"]["lcc_code"]
X_test = dataset["test"]["body"]
y_test = dataset["test"]["lcc_code"]

# Train simple baseline
vectorizer = TfidfVectorizer(max_features=10000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_vec, y_train)

# Evaluate
y_pred = clf.predict(X_test_vec)
print(classification_report(y_test, y_pred))
```

### Retrieval Example

```python
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import numpy as np

# Load data
dataset = load_dataset("mjbommar/SHELF")
corpus = dataset["train"]["body"]

# Encode with sentence transformer
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(corpus, show_progress_bar=True)

# Query
query = "Introduction to constitutional law"
query_emb = model.encode(query)

# Find similar documents
similarities = np.dot(embeddings, query_emb)
top_k = np.argsort(similarities)[-5:][::-1]

for idx in top_k:
    print(f"{similarities[idx]:.3f}: {dataset['train']['title'][idx]}")
```

### Pair Classification Example

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load pair data
pairs = load_dataset("mjbommar/SHELF", name="same_lcc_pairs")

# Prepare inputs (concatenate doc_a and doc_b)
def format_pair(example):
    text_a = f"{example['doc_a_title']} {example['doc_a_body']}"
    text_b = f"{example['doc_b_title']} {example['doc_b_body']}"
    return {"text": f"{text_a} [SEP] {text_b}", "label": example["label"]}

train_data = pairs["train"].map(format_pair)

# Fine-tune a model (simplified)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# ... training code ...
```

## Considerations for Using the Data

### Social Impact

This dataset is intended for research and development of document classification systems. It may help improve:
- Library cataloging automation
- Document organization systems
- Research paper classification
- Content recommendation systems

### Limitations

- **Synthetic Data:** Documents are AI-generated and may not perfectly reflect real-world document distributions
- **English Only:** Currently limited to English language documents
- **Multi-Model Generation:** Documents generated by 9 different LLMs (GPT-5.x, Gemini 2.5/3, Claude 4.5), which may introduce model-specific patterns
- **Citation Artifacts:** Some documents may contain fabricated citations that should not be treated as real references

### Bias Considerations

- Topic distribution reflects Library of Congress classification priorities
- Geographic coverage may be skewed toward certain regions
- Register distribution may not match real-world document frequencies

## Citation

If you use this dataset, please cite the repository. A paper is forthcoming.

```bibtex
@misc{shelf2025,
  title = {SHELF: Synthetic Harness for Evaluating LLM Fitness},
  author = {Bommarito, Michael J.},
  year = {2025},
  howpublished = {Hugging Face Datasets},
  url = {https://huggingface.co/datasets/mjbommar/SHELF},
  note = {Version 0.3.1. Paper forthcoming.}
}
```

## Additional Information

### Dataset Curators

Michael J. Bommarito II

### Licensing Information

This dataset is released under the Creative Commons Attribution 4.0 International (CC BY 4.0) license.

### Version History

- **v0.3.1** (2025-12-14): Multi-model generation with 42,532 documents (9 LLMs)
- **v0.2.0** (2025-12-12): Initial pre-release with 40,100 documents (GPT-5.1/5.2)

### Contact

For questions or issues, please open an issue on the [GitHub repository](https://github.com/mjbommar/shelf).

## v0.4 slices

v0.4 adds five configurations. **The `default` config and the six pair
configurations are unchanged**, so results published against v0.3.1 remain
valid.

| Config | Documents | What it is |
|---|---|---|
| `v0_4_core` | 18,345 | Generator-balanced corpus: 15 generators, largest share 9.2% |
| `v0_4_supplement` | 1,043 | Supplementary documents over the standard 21-class task, single generator |
| `v0_4_minimal_pairs` | 687 | Pairs holding topics, audience, register and length constant, varying exactly one facet |
| `v0_4_holdout` | 292 | Documents from a generator absent from the core, for transfer probing |
| `transfer_gutenberg` | 3,016 | Human-written, human-catalogued Project Gutenberg passages |

### Why v0.4 is separate rather than merged

The `default` corpus is **94.1% GPT-5.x**, with two models supplying that share
and five of its nine generators contributing about 100 documents each. `v0_4_core`
spreads 15 generators with no single one above 9.2%.

Pooling the two would return the largest generator to roughly half the combined
corpus, which would undo the balance `v0_4_core` exists to provide. Any
experiment that needs generator balance -- cross-generator generalization,
generator attribution, train-on-family-A / test-on-family-B -- should use
`v0_4_core` alone.

### Known limitations

**No subclass tier in this release.** An LCC subclass tier was planned and is
not shipped. The specification blocks assigned all 80 subclasses, but the
generation path passed the parent class description to the model, so the
documents were conditioned on 16 parent classes rather than 80 subclasses and
carry no subclass label. Those documents are published as `v0_4_supplement`,
described for what they are. The tier will return when the conditioning is
fixed.

**Empty-document rate.** `v0_4_supplement` and `v0_4_minimal_pairs` were generated
before a fix for a reasoning-budget defect: on short length targets, reasoning
tokens consumed the whole output cap, producing a title truncated mid-word and
no body. 13-15% of raw generations were affected. QC removes them, so the
published slices are clean but roughly 14% smaller than their nominal spec count.

**Generator confound on register and length.** In `v0_4_core`, generator is
independent of the labels that matter -- LCC class (Cramer's V 0.016) and LCGFT
category (0.027) -- but correlates weakly with `register` (0.061),
`target_length` (0.064) and `prompt_variant_id` (0.084). The cause is non-uniform
QC removal: the empty-body defect hit short documents hardest and two generators
failed for part of the run. Effect sizes are small; p-values are not marginal.
Analyses conditioned on register or length carry this confound.

**`transfer_gutenberg` is contaminated and must not be pooled.** It is in the
pretraining data of essentially every model that would be evaluated on it. SHELF
is the clean-synthetic condition, Gutenberg is the contaminated-natural one, and
the gap between them is the measurement. A lexical baseline trained on SHELF
scores 0.887 macro-F1 in-domain and **0.313 on Gutenberg**; trained on Gutenberg
it reaches 0.510 in-domain. Transfer fails symmetrically, which is domain shift
rather than memorisation.

**No human ceiling yet.** No human annotation round has been run, so model scores
on these slices have no interpretable upper bound.

**Prompt variants differ from `default`.** v0.4 documents use four new
system-prompt variants and form-conditional output formatting; `default` used a
single prompt. `prompt_variant_id` records which. A controlled A/B measured
spurious markdown on non-markdown forms falling from 26.7% to ~1.3%
(Fisher exact p < 0.00001).

## The `all` config

A single pooled corpus of every synthetic SHELF document: the v0.3.1
`default` corpus plus every v0.4 slice.

| | documents |
|---|---|
| `default` (v0.3.1) | 42,532 |
| `v0_4_core` | 18,345 |
| `v0_4_supplement` | 1,043 |
| `v0_4_minimal_pairs` | 687 |
| `v0_4_holdout` | 292 |
| **`all`** | **62,899** |

Splits: train 37,795 / validation 12,600 / test 12,504. Each
document keeps the split it was assigned in its source config.

**This config is not generator balanced, and that is the trade.** Pooling
returns the largest generator to 47.7% of the corpus, against
9.2% in `v0_4_core`. Use `all` when sample count matters more than
balance, and `v0_4_core` when it does not. Reporting a generator-sensitive
result on `all` without saying so would be misleading.

Every row carries `source_config` and `source_version`, so any component
slice can be recovered exactly:

```python
from datasets import load_dataset
ds = load_dataset("mjbommar/SHELF", "all")
core = ds["train"].filter(lambda r: r["source_config"] == "v0_4_core")
```

Schema is the union of both generations (44 columns), so no column is
dropped; columns absent from a source are null. `text` is always
populated. Provider routing prefixes are normalised, so one model is one
id. Titles carrying a leading markdown heading or `Title:` label were
cleaned (169 rows). Deduplicated on normalised body text: zero duplicates
were found across the two corpora, as expected from disjoint spec blocks.

**The Gutenberg transfer control is deliberately excluded.** It is natural
text used to measure whether SHELF scores transfer, and pooling it into
the corpus would destroy that measurement. It remains a separate config.
## Measured properties

Two properties matter for anyone deciding whether to use SHELF, and they
have different answers. Both were measured with TF-IDF plus logistic
regression on 21-class LCC (no pretraining, so contamination cannot explain
either), and with 22 embedding models for the ranking result.

**Absolute scores do not transfer.** A classifier scoring 0.887 on SHELF
scores 0.313 on Project Gutenberg passages. The failure is symmetric.

| train \ test | shelf | gutenberg | lcshbench |
|---|---|---|---|
| **shelf** | **0.8873** | 0.3133 | 0.4113 |
| **gutenberg** | 0.2836 | **0.5101** | 0.2135 |
| **lcshbench** | 0.4442 | 0.2800 | **0.5559** |

This is not peculiar to generated text. The two human-catalogued corpora,
Gutenberg and LCSHBench, reach only 0.2135 and 0.2800 on each other -- the
worst pairing in the matrix. No bibliographic corpus stands in for the task
in general.

**Model rankings do transfer.** Ranking 22 embedding models on each corpus:

| pair | Spearman | 95% CI |
|---|---|---|
| SHELF vs Gutenberg | **0.878** | [0.64, 0.99] |
| SHELF vs LCSHBench | **0.781** | [0.42, 0.97] |
| Gutenberg vs LCSHBench | 0.963 | [0.86, 0.99] |

**So: use SHELF to choose between models, not to predict a production
score.** Note that the natural corpora still agree with each other most
closely (0.963); the intervals overlap, so SHELF ranks models about as well
as natural bibliographic data, not better.

**How much label signal sits on the surface.** Verbatim `lcc_name` in its own
document, length-controlled to 200 words: Gutenberg 6.1%, SHELF 18.7%. Real
documents do contain their own descriptive terms -- a zero baseline would be
strange -- but SHELF carries about 3x the natural rate, which partly explains
its lexical ceiling. QC reduced this measurably between generations: topics
fell from 76.6% to 44.5%, form from 7.2% to 1.5%.

### `transfer_lcshbench`

English records from [LCSHBench](https://huggingface.co/datasets/kltng/lcshbench)
(CC0), real catalogue records from Harvard, Columbia, and Princeton carrying
real LCC classes. 4,924 rows across 21 classes, used here as a second natural
control.

**Do not pool it with `transfer_gutenberg`.** Gutenberg is running prose;
LCSHBench is catalogue metadata with a median of 596 characters. Report them
separately.

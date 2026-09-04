"""
Dataset Card Generation for SHELF

This module generates a comprehensive README.md dataset card with proper YAML
metadata following HuggingFace Hub best practices.

The dataset card includes:
- YAML metadata header with all required fields
- Dataset description and purpose
- Supported tasks and usage examples
- Data structure documentation
- Citation information

References:
- https://huggingface.co/docs/hub/en/datasets-cards
- https://huggingface.co/docs/datasets/en/dataset_card
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from shelf.hub.dataset import SHELFDataset


@dataclass
class CardConfig:
    """Configuration for dataset card generation.

    Attributes:
        pretty_name: Human-readable dataset name
        license: Dataset license (SPDX identifier)
        language: Language codes (ISO 639-1)
        size_category: Size category for the Hub
        version: Dataset version string
        homepage: Project homepage URL
        repository: Source code repository URL
        paper: Paper URL or arXiv ID (optional)
    """

    pretty_name: str = "SHELF: Synthetic Harness for Evaluating LLM Fitness"
    license: str = "cc-by-4.0"
    language: list[str] = field(default_factory=lambda: ["en"])
    size_category: str = "10K<n<100K"
    version: str = "0.3.0"
    homepage: str = "https://github.com/mjbommar/shelf-benchmark"
    repository: str = "https://github.com/mjbommar/shelf-benchmark"
    paper: str | None = None


# YAML metadata template
YAML_TEMPLATE = """---
pretty_name: "{pretty_name}"
license: {license}
language:
{language_yaml}
size_categories:
  - {size_category}
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
  - arxiv:2609.03047
dataset_info:
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
      sequence: string
    - name: geographic
      sequence: string
    - name: audience
      dtype: string
    - name: register
      dtype: string
    - name: register_description
      dtype: string
    - name: target_length
      dtype: string
    - name: target_word_range
      sequence: int32
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
      num_examples: {train_count}
    - name: validation
      num_examples: {validation_count}
    - name: test
      num_examples: {test_count}
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train-*
      - split: validation
        path: data/validation-*
      - split: test
        path: data/test-*
  - config_name: same_lcc_pairs
    data_files:
      - split: train
        path: pairs/same_lcc/train-*
      - split: validation
        path: pairs/same_lcc/validation-*
      - split: test
        path: pairs/same_lcc/test-*
  - config_name: same_form_pairs
    data_files:
      - split: train
        path: pairs/same_lcgft/train-*
      - split: validation
        path: pairs/same_lcgft/validation-*
      - split: test
        path: pairs/same_lcgft/test-*
  - config_name: same_register_pairs
    data_files:
      - split: train
        path: pairs/same_register/train-*
      - split: validation
        path: pairs/same_register/validation-*
      - split: test
        path: pairs/same_register/test-*
  - config_name: same_audience_pairs
    data_files:
      - split: train
        path: pairs/same_audience/train-*
      - split: validation
        path: pairs/same_audience/validation-*
      - split: test
        path: pairs/same_audience/test-*
  - config_name: same_topic_pairs
    data_files:
      - split: train
        path: pairs/same_topic/train-*
      - split: validation
        path: pairs/same_topic/validation-*
      - split: test
        path: pairs/same_topic/test-*
  - config_name: topic_overlap_pairs
    data_files:
      - split: train
        path: pairs/topic_overlap/train-*
      - split: validation
        path: pairs/topic_overlap/validation-*
      - split: test
        path: pairs/topic_overlap/test-*
---
"""

# README content template
# The arXiv identifier for the paper describing this dataset. Kept here so the
# card, the citation, and the Hub paper link cannot drift apart.
ARXIV_ID = "2609.03047"
ARXIV_URL = f"https://arxiv.org/abs/{ARXIV_ID}"

README_TEMPLATE = """# {pretty_name}

SHELF is a synthetic benchmark for evaluating language model fitness on bibliographic classification, retrieval, and clustering tasks using Library of Congress taxonomies.

## Dataset Description

- **Homepage:** {homepage}
- **Repository:** {repository}
- **Paper:** {paper_link}
- **License:** {license}
- **Version:** {version}

### Dataset Summary

SHELF's primary `all` configuration contains 62,899 synthetic documents
annotated with Library of Congress taxonomies. It pools the complete release;
the counts below describe the source dataset used when regenerating this card.

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
{{
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
}}
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
| Train | {train_count:,} | {train_pct:.1f}% |
| Validation | {validation_count:,} | {validation_pct:.1f}% |
| Test | {test_count:,} | {test_pct:.1f}% |

The `all` configuration preserves each document's source split. The original
42,532-document component uses stratified document-level splits. Later
components carry recorded specifications and are split by `spec_id`, keeping
all realizations of one specification together. The specification-level
leakage guarantee therefore applies to the later components, not to `all` as a
whole.

## Dataset Creation

### Source Data

Documents were synthetically generated by 25 writing models. A
generator-balanced factorial component contains 15 models from 11
laboratories.

The aggregate corpus is not generator-balanced: its largest writing model
supplies 47.7% of documents. In the factorial `v0_4_core` component, the
largest share is 9.24%. Every document records its writing model so analyses
can filter or stratify by source.

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

# Load the complete 62,899-document corpus
dataset = load_dataset("{repo_id}", "all")

# Load specific split
train = load_dataset("{repo_id}", split="train")
test = load_dataset("{repo_id}", split="test")

# Access examples
print(train[0])
```

### Dataset Configurations

The dataset has multiple configurations:

| Config | Description | Train | Val | Test |
|--------|-------------|-------|-----|------|
| `all` | Complete aggregate corpus | 37,795 | 12,600 | 12,504 |
| `default` | Original release | 25,518 | 8,507 | 8,507 |
| `same_lcc_pairs` | Document pairs labeled by LCC match | 20,000 | 4,000 | 4,000 |
| `same_form_pairs` | Document pairs labeled by LCGFT form match | 20,000 | 4,000 | 4,000 |
| `same_audience_pairs` | Document pairs labeled by audience match | 20,000 | 4,000 | 4,000 |
| `same_register_pairs` | Document pairs labeled by register/style match | 20,000 | 4,000 | 4,000 |
| `same_topic_pairs` | Binary: Do documents share ANY topic? | 20,000 | 4,000 | 4,000 |
| `topic_overlap_pairs` | Graded: How many topics shared? (0/1/2/3+) | 20,000 | 4,000 | 4,000 |

```python
# Load pair classification data
lcc_pairs = load_dataset("{repo_id}", name="same_lcc_pairs")
form_pairs = load_dataset("{repo_id}", name="same_form_pairs")
audience_pairs = load_dataset("{repo_id}", name="same_audience_pairs")
register_pairs = load_dataset("{repo_id}", name="same_register_pairs")

# Load topic overlap pairs
topic_binary = load_dataset("{repo_id}", name="same_topic_pairs")
topic_graded = load_dataset("{repo_id}", name="topic_overlap_pairs")

# Pair format (categorical pairs)
print(lcc_pairs["train"][0])
# {{'id': 'pair_000001', 'doc_a_id': '...', 'doc_a_title': '...',
#   'doc_a_body': '...', 'doc_b_id': '...', 'doc_b_title': '...',
#   'doc_b_body': '...', 'label': 1, 'label_field': 'lcc_code'}}

# Topic overlap format (multi-label pairs)
print(topic_graded["train"][0])
# {{'id': 'pair_000001', 'doc_a_id': '...', 'doc_a_title': '...',
#   'doc_a_body': '...', 'doc_b_id': '...', 'doc_b_title': '...',
#   'doc_b_body': '...', 'label': 2, 'overlap_count': 2,
#   'shared_topics': ['Ethics', 'Philosophy']}}
```

### Classification Example

```python
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load data
dataset = load_dataset("{repo_id}")

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
dataset = load_dataset("{repo_id}")
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
    print(f"{{similarities[idx]:.3f}}: {{dataset['train']['title'][idx]}}")
```

### Pair Classification Example

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load pair data
pairs = load_dataset("{repo_id}", name="same_lcc_pairs")

# Prepare inputs (concatenate doc_a and doc_b)
def format_pair(example):
    text_a = f"{{example['doc_a_title']}} {{example['doc_a_body']}}"
    text_b = f"{{example['doc_b_title']}} {{example['doc_b_body']}}"
    return {{"text": f"{{text_a}} [SEP] {{text_b}}", "label": example["label"]}}

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
- **Generator Imbalance:** The aggregate corpus contains 25 writing models but is not balanced; use `v0_4_core` for generator comparisons
- **Citation Artifacts:** Some documents may contain fabricated citations that should not be treated as real references

### Bias Considerations

- Topic distribution reflects Library of Congress classification priorities
- Geographic coverage may be skewed toward certain regions
- Register distribution may not match real-world document frequencies

## Citation

If you use this dataset, please cite the paper.

```bibtex
@article{{bommarito2026shelf,
  title = {{SHELF: A Synthetic Harness for Multi-Task Bibliographic Benchmarking}},
  author = {{Bommarito, Michael J.}},
  year = {{2026}},
  journal = {{arXiv preprint arXiv:2609.03047}},
  eprint = {{2609.03047}},
  archivePrefix = {{arXiv}},
  primaryClass = {{cs.CL}},
  url = {{https://arxiv.org/abs/2609.03047}}
}}
```

## Additional Information

### Dataset Curators

Michael J. Bommarito II

### Licensing Information

This dataset is released under the Creative Commons Attribution 4.0 International (CC BY 4.0) license.

### Version History

- **Current aggregate release:** 62,899 documents from 25 writing models
- **v{version}** ({date}): Original {total_documents:,}-document release
- **v0.2.0** (2025-12-12): Initial pre-release with 40,100 documents (GPT-5.1/5.2)

### Contact

For questions or issues, please open an issue on the [GitHub repository]({repository}).
"""


class DatasetCardGenerator:
    """Generates HuggingFace dataset cards for SHELF.

    Example:
        >>> dataset = SHELFDataset.from_artifacts("data/artifacts/")
        >>> generator = DatasetCardGenerator(dataset)
        >>> card_content = generator.generate()
        >>> print(card_content[:500])
    """

    def __init__(
        self,
        dataset: SHELFDataset,
        config: CardConfig | None = None,
    ) -> None:
        """Initialize generator with dataset and configuration.

        Args:
            dataset: SHELFDataset to generate card for
            config: Card configuration (optional)
        """
        self.dataset = dataset
        self.config = config or CardConfig()

    def _format_yaml_list(self, items: list[str], indent: int = 2) -> str:
        """Format a list for YAML output."""
        prefix = " " * indent
        return "\n".join(f"{prefix}- {item}" for item in items)

    def _get_generation_metadata_yaml(self) -> str:
        """Get YAML for generation metadata fields if included."""
        if not self.dataset.config.include_generation_metadata:
            return ""
        return """    - name: temperature
      dtype: float32
    - name: top_p
      dtype: float32
    - name: model
      dtype: string
"""

    def generate_yaml_header(self) -> str:
        """Generate the YAML metadata header."""
        train_count = len(self.dataset.train)
        dev_count = len(self.dataset.dev)
        test_count = len(self.dataset.test)

        language_yaml = self._format_yaml_list(self.config.language)

        return YAML_TEMPLATE.format(
            pretty_name=self.config.pretty_name,
            license=self.config.license,
            language_yaml=language_yaml,
            size_category=self.config.size_category,
            train_count=train_count,
            validation_count=dev_count,
            test_count=test_count,
        )

    def generate_readme_content(self, repo_id: str) -> str:
        """Generate the README content section."""
        total = self.dataset.total_documents
        train_count = len(self.dataset.train)
        dev_count = len(self.dataset.dev)
        test_count = len(self.dataset.test)

        train_pct = (train_count / total) * 100
        dev_pct = (dev_count / total) * 100
        test_pct = (test_count / total) * 100

        paper_link = self.config.paper or ARXIV_URL

        return README_TEMPLATE.format(
            pretty_name=self.config.pretty_name,
            homepage=self.config.homepage,
            repository=self.config.repository,
            paper_link=paper_link,
            license=self.config.license,
            version=self.config.version,
            total_documents=total,
            train_count=train_count,
            validation_count=dev_count,
            test_count=test_count,
            train_pct=train_pct,
            validation_pct=dev_pct,
            test_pct=test_pct,
            repo_id=repo_id,
            date=datetime.now(UTC).strftime("%Y-%m-%d"),
        )

    def generate(self, repo_id: str | None = None) -> str:
        """Generate complete dataset card (YAML header + README content).

        Args:
            repo_id: Repository ID for the dataset (used in examples)

        Returns:
            Complete dataset card content as string
        """
        repo_id = repo_id or self.dataset.config.repo_id
        yaml_header = self.generate_yaml_header()
        readme_content = self.generate_readme_content(repo_id)
        return yaml_header + readme_content

    def save(self, output_path: str | Path, repo_id: str | None = None) -> Path:
        """Save dataset card to file.

        Args:
            output_path: Path to save README.md
            repo_id: Repository ID for the dataset

        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        content = self.generate(repo_id)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"Saved dataset card to {output_path}")
        return output_path


def generate_dataset_card(
    dataset: SHELFDataset,
    output_path: str | Path | None = None,
    repo_id: str | None = None,
    config: CardConfig | None = None,
) -> str:
    """Convenience function to generate a dataset card.

    Args:
        dataset: SHELFDataset to generate card for
        output_path: Optional path to save the card
        repo_id: Repository ID for examples
        config: Card configuration

    Returns:
        Dataset card content as string

    Example:
        >>> dataset = SHELFDataset.from_artifacts("data/artifacts/")
        >>> card = generate_dataset_card(dataset, output_path="README.md")
    """
    generator = DatasetCardGenerator(dataset, config)
    content = generator.generate(repo_id)

    if output_path:
        output_path = Path(output_path)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Saved dataset card to {output_path}")

    return content

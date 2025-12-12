# Task: Document Clustering

> **Task Type**: clustering
> **Difficulty**: medium
> **Primary Metric**: V-measure

## Overview

The clustering tasks evaluate how well embedding models capture semantic and topical relationships in Library of Congress bibliographic data by measuring the quality of unsupervised groupings against known taxonomic labels. Unlike classification tasks where models are trained to predict labels directly, clustering tasks assess whether documents with similar content naturally group together in the embedding space, regardless of the specific cluster labels assigned.

These tasks test a fundamental capability of embedding models: whether semantically similar documents are encoded near each other in vector space. Good embeddings should enable clustering algorithms to discover natural groupings that align with expert-assigned Library of Congress taxonomies, even though the clustering algorithm has no access to the ground truth labels during cluster formation. This provides a stronger test of semantic understanding than classification, as it requires the model to capture intrinsic document similarities rather than learning label-specific decision boundaries.

## Task Definition

### Input
For each document in the dataset, the model receives:
- **Text**: The concatenation of document title and body (same format as classification tasks)
- **Number of clusters (k)**: The number of clusters to form, which equals the number of unique labels in the ground truth

### Output
The model must produce:
- **Document embeddings**: Dense vector representations of each document
- **Cluster assignments**: Integer labels assigning each document to one of k clusters (produced by running k-means on the embeddings)

### Formal Definition

Given a set of documents D = {d_1, d_2, ..., d_n} and a ground truth labeling G = {g_1, g_2, ..., g_n} where g_i in {1, ..., k}:

1. Generate embeddings E = {e_1, e_2, ..., e_n} where e_i = embed(d_i)
2. Apply mini-batch k-means clustering to E with k clusters, producing predicted labels P = {p_1, p_2, ..., p_n}
3. Evaluate the agreement between P and G using information-theoretic and pair-counting metrics

Note that clustering is an unsupervised task - the ground truth labels G are used only for evaluation, not during the clustering process itself. This differs fundamentally from classification, where labels are used to train a supervised model.

## Tasks

SHELF includes three distinct clustering tasks based on different taxonomic granularities:

### 1. LCC Clustering (21 clusters)

Cluster documents into 21 groups based on Library of Congress Classification (LCC) codes. Each document has one of 21 top-level LCC codes (A through Z, excluding I, O, W, X, Y) representing broad subject areas like Philosophy, History, Science, and Technology.

**Example clusters:**
- Cluster 1 (H): Social Sciences documents
- Cluster 2 (P): Language and Literature documents
- Cluster 3 (Q): Science documents

### 2. LCGFT Category Clustering (14 clusters)

Cluster documents into 14 groups based on Library of Congress Genre/Form Terms (LCGFT) categories. These represent high-level genre classifications like Literature, Visual Works, and Instructional Materials.

**Example clusters:**
- Cluster 1 (Literature): Fiction, Poetry, Satire
- Cluster 2 (Visual Works): Diagrams, Charts, Illustrations
- Cluster 3 (Instructional Materials): Textbooks, Study guides

### 3. Geographic Region Clustering (8 clusters)

Cluster documents into 8 groups based on geographic region. Documents in the corpus have optional geographic tags indicating location focus (countries, regions, cities, or US states). For clustering purposes, all 44 unique geographic locations are mapped to 8 broad regions.

**Regional Mapping:**

The 44 geographic locations are grouped into 8 regions as follows:

| Region | Locations |
|--------|-----------|
| **North America** | United States, Canada, California, New York, Texas, Florida, Illinois, Pennsylvania, Ohio, Georgia, North Carolina, Michigan, New York City, Los Angeles, Chicago |
| **South America** | Brazil, São Paulo, South America |
| **Europe** | United Kingdom, Germany, France, Italy, Spain, Russia, Europe, London, Paris, Berlin |
| **East Asia** | China, Japan, South Korea, Tokyo, Beijing, Asia (when referring to East Asia context) |
| **South/Southeast Asia** | India, Southeast Asia, Mumbai |
| **Middle East & North Africa** | Middle East |
| **Sub-Saharan Africa** | Africa (when not Middle East/North Africa) |
| **Central America & Caribbean** | Mexico, Central America, Caribbean |

**Important Notes:**
- Not all documents have geographic tags (approximately 50% have at least one tag)
- Documents with multiple geographic tags use the **first tag** for clustering
- Documents without geographic tags are excluded from this clustering task
- The "Asia" tag is context-dependent: mapped to East Asia by default, but documents are manually reviewed during mapping

**Example clusters:**
- Cluster 1 (North America): Documents focused on US policies, Canadian culture, California technology
- Cluster 2 (Europe): Documents about European history, French literature, German engineering
- Cluster 3 (East Asia): Documents on Chinese economics, Japanese art, Korean technology

## Dataset

### Source
All documents are synthetically generated using large language models to represent diverse Library of Congress Classification (LCC) and Genre/Form (LCGFT) categories. Each document is labeled with:
- An LCC code (21 possible values: A-Z excluding I, O, W, X, Y)
- An LCGFT category (14 possible values)
- Other metadata including topics, audience, and geographic information

The synthetic generation process ensures balanced representation across categories while maintaining realistic document characteristics including varied lengths, registers, and subject matter complexity.

### Statistics

#### LCC Clustering (21 clusters)
| Split | Documents | Clusters | Avg. Cluster Size |
|-------|-----------|----------|-------------------|
| Train | 12,000 | 21 | ~571 |
| Dev   | 4,000 | 21 | ~190 |
| Test  | 4,000 | 21 | ~190 |

#### LCGFT Category Clustering (14 clusters)
| Split | Documents | Clusters | Avg. Cluster Size |
|-------|-----------|----------|-------------------|
| Train | 12,000 | 14 | ~857 |
| Dev   | 4,000 | 14 | ~286 |
| Test  | 4,000 | 14 | ~286 |

#### Geographic Region Clustering (8 clusters)
| Split | Documents | Clusters | Avg. Cluster Size |
|-------|-----------|----------|-------------------|
| Train | ~6,000 | 8 | ~750 |
| Dev   | ~2,000 | 8 | ~250 |
| Test  | ~2,000 | 8 | ~250 |

**Note**: Geographic clustering uses only documents with at least one geographic tag (~50% of corpus). Documents with multiple tags use the first tag for region assignment.

### Label Space

#### Geographic Regions (8 clusters)
- **North America**: United States, Canada, and major US cities/states
- **South America**: Brazil and South American countries
- **Europe**: Western and Eastern European countries and cities
- **East Asia**: China, Japan, South Korea and major cities
- **South/Southeast Asia**: India, Southeast Asian countries
- **Middle East & North Africa**: Middle Eastern countries and North Africa
- **Sub-Saharan Africa**: African countries south of the Sahara
- **Central America & Caribbean**: Mexico, Central American and Caribbean nations

**Full Location-to-Region Mapping:**

```python
GEOGRAPHIC_REGION_MAPPING = {
    # North America
    "United States": "North America",
    "Canada": "North America",
    "California": "North America",
    "New York": "North America",
    "Texas": "North America",
    "Florida": "North America",
    "Illinois": "North America",
    "Pennsylvania": "North America",
    "Ohio": "North America",
    "Georgia": "North America",
    "North Carolina": "North America",
    "Michigan": "North America",
    "New York City": "North America",
    "Los Angeles": "North America",
    "Chicago": "North America",
    "North America": "North America",

    # South America
    "Brazil": "South America",
    "São Paulo": "South America",
    "South America": "South America",

    # Europe
    "United Kingdom": "Europe",
    "Germany": "Europe",
    "France": "Europe",
    "Italy": "Europe",
    "Spain": "Europe",
    "Russia": "Europe",
    "Europe": "Europe",
    "London": "Europe",
    "Paris": "Europe",
    "Berlin": "Europe",

    # East Asia
    "China": "East Asia",
    "Japan": "East Asia",
    "South Korea": "East Asia",
    "Tokyo": "East Asia",
    "Beijing": "East Asia",
    "Asia": "East Asia",  # Default mapping

    # South/Southeast Asia
    "India": "South/Southeast Asia",
    "Southeast Asia": "South/Southeast Asia",
    "Mumbai": "South/Southeast Asia",

    # Middle East & North Africa
    "Middle East": "Middle East & North Africa",

    # Sub-Saharan Africa
    "Africa": "Sub-Saharan Africa",

    # Central America & Caribbean
    "Mexico": "Central America & Caribbean",
    "Central America": "Central America & Caribbean",
    "Caribbean": "Central America & Caribbean",
}
```

#### LCC Codes (21 clusters)
- **A**: General Works
- **B**: Philosophy, Psychology, Religion
- **C**: Auxiliary Sciences of History
- **D**: History (General) and History of Europe
- **E**: History of the Americas (General)
- **F**: History of the Americas (Local)
- **G**: Geography, Anthropology, Recreation
- **H**: Social Sciences
- **J**: Political Science
- **K**: Law
- **L**: Education
- **M**: Music
- **N**: Fine Arts
- **P**: Language and Literature
- **Q**: Science
- **R**: Medicine
- **S**: Agriculture
- **T**: Technology
- **U**: Military Science
- **V**: Naval Science
- **Z**: Bibliography, Library Science, Information Resources

#### LCGFT Categories (14 clusters)
- **Literature**: Fiction, poetry, drama, satire, etc.
- **Visual Works**: Diagrams, charts, illustrations, photographs
- **Instructional Materials**: Textbooks, study guides, educational materials
- **Informational Works**: Reference works, encyclopedias, handbooks
- **Recorded Works**: Sound recordings, audiovisual materials
- **Cartographic Materials**: Maps, atlases, globes
- **Notated Works**: Musical scores, choreographic notation
- **Administrative Materials**: Forms, reports, policies
- **Personal Communications**: Letters, diaries, memoirs
- **Legal Works**: Legislation, case law, legal briefs
- **Sacred Works**: Religious texts, liturgies, prayers
- **Commentaries**: Reviews, criticism, analysis
- **Promotional Materials**: Advertisements, brochures, catalogs
- **Archival Materials**: Finding aids, inventories, registers

### Data Format
```json
{
  "id": "20251211_030155_57dfc238",
  "title": "Quarterly Agribusiness Risk Memo on Emerging Externalities",
  "body": "In light of climate change, migratory wildlife now attends...",
  "word_count": 37,
  "lcc_code": "S",
  "lcc_name": "Agriculture",
  "lcc_uri": "http://id.loc.gov/authorities/classification/S",
  "lcgft_category": "Literature",
  "lcgft_form": "Satire",
  "topics": ["Climate change", "Wildlife", "Neuroscience", "Ocean conservation"],
  "audience": null,
  "geographic": []
}
```

For clustering tasks:
- **LCC Clustering**: Use `lcc_code` as ground truth label
- **LCGFT Category Clustering**: Use `lcgft_category` as ground truth label
- **Geographic Region Clustering**: Use first element of `geographic` list, mapped to region via `GEOGRAPHIC_REGION_MAPPING`. Documents with empty `geographic` lists are excluded.

## Evaluation

### Evaluation Protocol

Following the MTEB (Massive Text Embedding Benchmark) clustering methodology, evaluation proceeds in three steps:

1. **Embedding Generation**: The model generates dense vector embeddings for all documents in the test set. No training is performed - the model uses its pre-trained or fine-tuned parameters to encode each document's concatenated title and body text.

2. **Clustering**: Mini-batch k-means clustering is applied to the embeddings with:
   - **k**: Number of clusters equal to the number of unique ground truth labels (21 for LCC, 14 for LCGFT)
   - **Batch size**: 32 (for mini-batch k-means)
   - **Initialization**: k-means++ (smart centroid initialization)
   - **Distance metric**: Euclidean distance (standard k-means) or cosine similarity (spherical k-means for normalized embeddings)
   - **Random state**: Fixed seed for reproducibility across runs

3. **Metric Computation**: The resulting cluster assignments are compared against ground truth labels using information-theoretic and pair-counting metrics that are invariant to cluster label permutations.

**Important Note**: The ground truth labels are used ONLY for evaluation, not during the clustering process. This is what distinguishes clustering from classification - the algorithm must discover natural groupings in the data without any supervision.

### Primary Metric

**V-measure**: The harmonic mean of homogeneity and completeness, ranging from 0 (random clustering) to 1 (perfect clustering).

V-measure is defined as:
```
V = 2 * (homogeneity * completeness) / (homogeneity + completeness)
```

Where:
- **Homogeneity**: Each cluster contains only members of a single class (analogous to precision)
  - h = 1 - H(C|K) / H(C), where H(C|K) is conditional entropy of classes given clusters
- **Completeness**: All members of a given class are assigned to the same cluster (analogous to recall)
  - c = 1 - H(K|C) / H(K), where H(K|C) is conditional entropy of clusters given classes

V-measure is chosen as the primary metric because:
1. It is **symmetric**: Homogeneity and completeness are equally weighted
2. It is **interpretable**: Values near 0 indicate poor clustering, values near 1 indicate excellent clustering
3. It is **label-invariant**: Cluster label permutations don't affect the score
4. It is **normalized**: Always bounded between 0 and 1, enabling comparison across tasks
5. It is **standard**: Widely used in MTEB and other embedding benchmarks

### Secondary Metrics

#### Normalized Mutual Information (NMI)
NMI measures the amount of shared information between predicted clusters and ground truth labels, normalized to the range [0, 1].

```
NMI(C, K) = 2 * MI(C, K) / (H(C) + H(K))
```

Where:
- MI(C, K) is mutual information between class labels C and cluster assignments K
- H(C) and H(K) are the entropies of C and K

**Properties**:
- Range: [0, 1] where 0 = no mutual information, 1 = perfect correlation
- Label-invariant: Permuting cluster labels doesn't change the score
- Not adjusted for chance: Random clusterings may score above 0
- May have selection bias toward solutions with many clusters

#### Adjusted Rand Index (ARI)
ARI measures the similarity between two clusterings, adjusted for chance agreement. It counts pairs of points that are consistently grouped or separated in both clusterings.

```
ARI = (RI - Expected_RI) / (max(RI) - Expected_RI)
```

Where RI is the Rand Index counting concordant pairs.

**Properties**:
- Range: [-1, 1] where 0 = random clustering, 1 = perfect clustering
- Adjusted for chance: Random labelings score ~0 in expectation
- Symmetric: ARI(C, K) = ARI(K, C)
- Can produce negative values for particularly poor clusterings
- More conservative than NMI (typically lower scores)

### Clustering vs Classification: Key Differences

| Aspect | Classification | Clustering |
|--------|---------------|------------|
| **Learning paradigm** | Supervised - uses labels during training | Unsupervised - no labels during clustering |
| **What's evaluated** | Ability to predict correct labels | Ability to group similar items together |
| **Label dependency** | Model learns label-specific patterns | Labels used only for evaluation, not learning |
| **Metric focus** | Accuracy, precision, recall, F1 | Information-theoretic agreement (V-measure, NMI) |
| **Random baseline** | 1/k accuracy (k = number of classes) | 0 V-measure (adjusted metrics) |
| **What it tests** | Discriminative decision boundaries | Intrinsic semantic similarity in embeddings |
| **Training required** | Yes - classifier trained on labeled data | No - only embedding model (pre-trained) |

**Key insight**: Clustering is a harder test of embedding quality because the model cannot "cheat" by learning label-specific patterns. Good clustering performance indicates that semantically similar documents are naturally encoded near each other in the embedding space, demonstrating genuine semantic understanding rather than pattern matching.

## Baselines

### LCC Clustering (21 clusters)

| Model | V-measure | NMI | ARI | Notes |
|-------|-----------|-----|-----|-------|
| Random | 0.000 | ~0.000 | 0.000 | Random cluster assignment |
| TF-IDF + k-means | TBD | TBD | TBD | Bag-of-words baseline |
| Doc2Vec + k-means | TBD | TBD | TBD | Classical embedding baseline |
| SBERT + k-means | TBD | TBD | TBD | Sentence transformer baseline |
| OpenAI text-embedding-3-small | TBD | TBD | TBD | Commercial baseline |

### LCGFT Category Clustering (14 clusters)

| Model | V-measure | NMI | ARI | Notes |
|-------|-----------|-----|-----|-----|
| Random | 0.000 | ~0.000 | 0.000 | Random cluster assignment |
| TF-IDF + k-means | TBD | TBD | TBD | Bag-of-words baseline |
| Doc2Vec + k-means | TBD | TBD | TBD | Classical embedding baseline |
| SBERT + k-means | TBD | TBD | TBD | Sentence transformer baseline |
| OpenAI text-embedding-3-small | TBD | TBD | TBD | Commercial baseline |

### Geographic Region Clustering (8 clusters)

| Model | V-measure | NMI | ARI | Notes |
|-------|-----------|-----|-----|-------|
| Random | 0.000 | ~0.000 | 0.000 | Random cluster assignment |
| TF-IDF + k-means | TBD | TBD | TBD | Bag-of-words baseline |
| Doc2Vec + k-means | TBD | TBD | TBD | Classical embedding baseline |
| SBERT + k-means | TBD | TBD | TBD | Sentence transformer baseline |
| OpenAI text-embedding-3-small | TBD | TBD | TBD | Commercial baseline |

**Note**: Baseline results will be added after initial benchmark runs are completed. Geographic clustering is expected to be easier than LCC/LCGFT clustering (fewer clusters, geographic signals often explicit in text).

## Related Work

### Similar Tasks in Other Benchmarks

**MTEB (Massive Text Embedding Benchmark)**
- ArxivClusteringP2P and ArxivClusteringS2S: Clustering scientific paper abstracts
- BiorxivClusteringP2P and BiorxivClusteringS2S: Clustering biomedical papers
- RedditClustering and RedditClusteringP2P: Clustering social media posts
- StackExchangeClustering: Clustering Q&A forum posts
- TwentyNewsgroupsClustering: Classic newsgroup clustering benchmark

SHELF clustering tasks differ by focusing on library science taxonomies (LCC and LCGFT) rather than academic or social media domains, providing a unique evaluation of how well models capture bibliographic and genre-based similarities.

**BEIR (Benchmarking IR)**
While primarily focused on retrieval, BEIR includes clustering-related evaluations through its diverse document corpora, though not with explicit clustering metrics.

### Relevant Literature

1. **Muennighoff et al. (2023)**: "MTEB: Massive Text Embedding Benchmark" introduced the standard clustering evaluation protocol used in SHELF, establishing V-measure as the primary metric for embedding-based clustering tasks.

2. **Rosenberg & Hirschberg (2007)**: "V-Measure: A Conditional Entropy-Based External Cluster Evaluation Measure" formalized V-measure as the harmonic mean of homogeneity and completeness, providing theoretical foundations for clustering evaluation.

3. **Hubert & Arabie (1985)**: "Comparing partitions" introduced the Adjusted Rand Index, establishing the importance of adjusting for chance in clustering metrics.

4. **Vinh et al. (2010)**: "Information Theoretic Measures for Clusterings Comparison" analyzed NMI and AMI, discussing normalization strategies and biases in mutual information metrics.

5. **Steinbach et al. (2000)**: "A Comparison of Document Clustering Techniques" compared k-means with other clustering algorithms for text, establishing k-means as a strong baseline for document clustering.

6. **Reimers & Gurevych (2019)**: "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks" demonstrated that transformer-based embeddings significantly outperform classical methods on clustering tasks.

## Implementation Notes

### Running Evaluation

```bash
# Evaluate LCC clustering (21 clusters)
shelf evaluate --task lcc_clustering --model <model_path>

# Evaluate LCGFT category clustering (14 clusters)
shelf evaluate --task lcgft_clustering --model <model_path>

# Evaluate geographic region clustering (8 clusters)
shelf evaluate --task geographic_clustering --model <model_path>

# Run all clustering tasks
shelf evaluate --task clustering --model <model_path>
```

### Submission Format

For leaderboard submissions, provide a JSON file with embeddings or cluster assignments:

```json
{
  "task": "lcc_clustering",
  "model": "your-model-name",
  "predictions": {
    "20251211_030155_57dfc238": 0,
    "20251211_030155_f3221f5a": 15,
    ...
  },
  "embeddings": {
    "20251211_030155_57dfc238": [0.123, -0.456, ...],
    ...
  }
}
```

Either `predictions` (cluster assignments) or `embeddings` (for automatic k-means clustering) must be provided.

### Implementation Considerations

1. **Embedding Normalization**: For high-dimensional embeddings, consider using spherical k-means (cosine similarity) instead of standard k-means (Euclidean distance). Normalize embeddings to unit length before clustering.

2. **Initialization**: Use k-means++ initialization for more stable results. Multiple random restarts (e.g., n_init=10) can improve clustering quality but increase computation time.

3. **Reproducibility**: Set random seeds for both embedding generation (if stochastic) and k-means initialization to ensure reproducible results.

4. **Computational Efficiency**: For large datasets, mini-batch k-means with batch_size=32 (following MTEB) provides a good balance between quality and speed.

5. **Dimensionality**: Higher-dimensional embeddings generally perform better but require more computation. Common dimensions: 384 (SBERT-base), 768 (BERT-base), 1536 (OpenAI-small).

## References

1. Muennighoff, N., et al. (2023). "MTEB: Massive Text Embedding Benchmark." EACL 2023. [https://arxiv.org/abs/2210.07316](https://arxiv.org/abs/2210.07316)

2. Rosenberg, A., & Hirschberg, J. (2007). "V-Measure: A Conditional Entropy-Based External Cluster Evaluation Measure." EMNLP 2007.

3. Hubert, L., & Arabie, P. (1985). "Comparing partitions." Journal of Classification, 2(1), 193-218.

4. Vinh, N. X., Epps, J., & Bailey, J. (2010). "Information Theoretic Measures for Clusterings Comparison: Variants, Properties, Normalization and Correction for Chance." JMLR, 11, 2837-2854.

5. Steinbach, M., Karypis, G., & Kumar, V. (2000). "A Comparison of Document Clustering Techniques." KDD Workshop on Text Mining.

6. Reimers, N., & Gurevych, I. (2019). "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." EMNLP 2019. [https://arxiv.org/abs/1908.10084](https://arxiv.org/abs/1908.10084)

7. Library of Congress Classification Outline. [https://www.loc.gov/catdir/cpso/lcco/](https://www.loc.gov/catdir/cpso/lcco/)

8. Library of Congress Genre/Form Terms for Library and Archival Materials (LCGFT). [https://www.loc.gov/aba/publications/FreeLCGFT/freelcgft.html](https://www.loc.gov/aba/publications/FreeLCGFT/freelcgft.html)

9. scikit-learn Clustering Metrics Documentation. [https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation](https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation)

10. MTEB Leaderboard and Documentation. [https://huggingface.co/spaces/mteb/leaderboard](https://huggingface.co/spaces/mteb/leaderboard)

---
*Last updated: 2025-12-10*

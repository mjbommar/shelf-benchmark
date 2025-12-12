# Register Clustering Task - Documentation to Add to clustering.md

This document contains the content that should be added to `/home/mjbommar/src/shelf-benchmark/docs/tasks/clustering.md` to document the Register Clustering task.

## Section 1: Add to "## Tasks" section (after LCGFT Category Clustering)

### 3. Register Clustering (8 clusters)

Cluster documents into 8 groups based on writing register/style. Each document is written in one of 8 linguistic registers representing distinct formality levels and stylistic choices: academic, casual, conversational, creative, formal, journalistic, professional, and technical.

**Example clusters:**
- Cluster 1 (academic): Scholarly papers with citations and hedged claims
- Cluster 2 (casual): Informal blog-style content with conversational tone
- Cluster 3 (technical): Specialized content assuming domain expertise
- Cluster 4 (professional): Standard business communication
- Cluster 5 (formal): Official governmental or legal documents

**Difficulty Note**: Register clustering is expected to be harder than LCC or LCGFT clustering because:
- Writing style is more subtle than topical content (LCC) or genre (LCGFT)
- Register signals operate at lexical, syntactic, and discourse levels simultaneously
- Multiple registers can appear in similar subject domains
- Style features may be less salient in embedding spaces optimized for semantic content
- Significant class imbalance (professional 24.7% vs creative 4.8%)

---

## Section 2: Add to "### Statistics" section

#### Register Clustering (8 clusters)
| Split | Documents | Clusters | Avg. Cluster Size |
|-------|-----------|----------|-------------------|
| Train | 12,000 | 8 | ~1,500 |
| Dev   | 4,000 | 8 | ~500 |
| Test  | 4,000 | 8 | ~500 |

**Note**: Register distribution is intentionally weighted (professional 24.7%, conversational 15.5%, formal 15.1%, academic 14.8%, technical 10.1%, casual 9.9%, journalistic 5.1%, creative 4.8%) to reflect realistic document frequencies.

---

## Section 3: Add to "### Label Space" section (after LCGFT Categories)

#### Register Types (8 clusters)

- **academic**: Scholarly and precise writing with citations and hedged claims (e.g., "may suggest", "appears to indicate"). Characterized by formal vocabulary, complex syntax, impersonal constructions, and extensive use of passive voice. Common in research papers, dissertations, and academic journals.

- **casual**: Informal and conversational, like blog posts or social media content. Uses contractions, first/second person pronouns, colloquialisms, and simple sentence structures. Features frequent use of exclamations, questions, and personal anecdotes.

- **conversational**: Friendly and approachable, like talking to a colleague. Balances informality with clarity, uses inclusive pronouns (we, you), and maintains a warm tone. More structured than casual but less formal than professional.

- **creative**: Expressive and literary with vivid language and stylistic flourishes. Features metaphors, varied sentence rhythms, sensory details, and emphasis on aesthetic qualities. Common in fiction, poetry, and creative nonfiction.

- **formal**: Formal and official, appropriate for legal or governmental contexts. Uses impersonal constructions, no contractions, Latinate vocabulary, and complex prepositions. Characterized by precise terminology and adherence to strict conventions.

- **journalistic**: Clear and factual, using inverted pyramid news style. Prioritizes key information first, uses active voice, attributes sources, and maintains objectivity. Emphasizes brevity, clarity, and newsworthiness.

- **professional**: Clear and professional, standard business tone. Balances formality with accessibility, uses industry-standard terminology, and maintains a polite, efficient style. Common in business correspondence, reports, and presentations.

- **technical**: Technical and specialized, assuming domain expertise and using jargon. Dense with specialized terminology, precise definitions, and assumes significant prior knowledge. Common in technical manuals, specifications, and scientific documentation.

**Distribution in Corpus (20,000 documents):**
- professional: 4,939 (24.7%)
- conversational: 3,092 (15.5%)
- formal: 3,028 (15.1%)
- academic: 2,965 (14.8%)
- technical: 2,012 (10.1%)
- casual: 1,982 (9.9%)
- journalistic: 1,029 (5.1%)
- creative: 953 (4.8%)

---

## Section 4: Update "For clustering tasks:" bullet list

For clustering tasks:
- **LCC Clustering**: Use `lcc_code` as ground truth label
- **LCGFT Category Clustering**: Use `lcgft_category` as ground truth label
- **Register Clustering**: Use `register` as ground truth label

---

## Section 5: Update "2. Clustering" in Evaluation Protocol

2. **Clustering**: Mini-batch k-means clustering is applied to the embeddings with:
   - **k**: Number of clusters equal to the number of unique ground truth labels (21 for LCC, 14 for LCGFT, 8 for Register)
   - **Batch size**: 32 (for mini-batch k-means)
   - **Initialization**: k-means++ (smart centroid initialization)
   - **Distance metric**: Euclidean distance (standard k-means) or cosine similarity (spherical k-means for normalized embeddings)
   - **Random state**: Fixed seed for reproducibility across runs

---

## Section 6: Add to "## Baselines" section

### Register Clustering (8 clusters)

| Model | V-measure | NMI | ARI | Notes |
|-------|-----------|-----|-----|-------|
| Random | 0.000 | ~0.000 | 0.000 | Random cluster assignment |
| TF-IDF + k-means | TBD | TBD | TBD | Bag-of-words baseline - may struggle with style |
| Doc2Vec + k-means | TBD | TBD | TBD | Classical embedding baseline |
| SBERT + k-means | TBD | TBD | TBD | Sentence transformer baseline |
| OpenAI text-embedding-3-small | TBD | TBD | TBD | Commercial baseline |

**Expected Difficulty**: Register clustering is expected to be harder than LCC or LCGFT clustering because:
1. **Subtle signals**: Style features are more subtle than content (LCC) or genre (LCGFT)
2. **Embedding optimization**: Embeddings optimized for semantic similarity may not capture stylistic features well
3. **Multi-level features**: Register operates across multiple linguistic levels (lexical, syntactic, discourse)
4. **Class imbalance**: Significant imbalance (professional 24.7% vs creative 4.8%) may affect clustering quality
5. **Cross-domain variation**: Same register can appear across different subject domains

**Research Context**: Prior work on formality detection (Pavlick & Tetreault 2016, Sheikha & Inkpen 2010) suggests that register classification benefits from character-level models and POS-based features. Standard semantic embeddings may miss crucial stylistic markers, making this clustering task particularly challenging.

---

## Section 7: Add to "### Running Evaluation" section

```bash
# Evaluate LCC clustering (21 clusters)
shelf evaluate --task lcc_clustering --model <model_path>

# Evaluate LCGFT category clustering (14 clusters)
shelf evaluate --task lcgft_clustering --model <model_path>

# Evaluate Register clustering (8 clusters)
shelf evaluate --task register_clustering --model <model_path>

# Run all clustering tasks
shelf evaluate --task clustering --model <model_path>
```

---

## Implementation Status

**Code Changes Completed:**
- ✅ Added `register_clustering` task to `/home/mjbommar/src/shelf-benchmark/src/shelf/evaluate/registry.py`
- ✅ Verified `register` field exists in dataset schema (`/home/mjbommar/src/shelf-benchmark/src/shelf/hub/dataset.py`)
- ✅ Confirmed REGISTERS constant defined in registry (8 register types)
- ✅ Updated dataset card to show clustering supports "21/14/8" clusters

**Documentation Updates Needed:**
- 📝 Add sections above to `/home/mjbommar/src/shelf-benchmark/docs/tasks/clustering.md`
  - Currently the file has been modified to include "Geographic Region Clustering"
  - The Register Clustering sections should be added instead or in addition

**Key Design Decisions:**

1. **8 Clusters**: Matches the 8 register types in the corpus (academic, casual, conversational, creative, formal, journalistic, professional, technical)

2. **Imbalanced Distribution**: Unlike LCC (uniform) and LCGFT (near-uniform), register has intentional imbalance reflecting realistic document frequencies

3. **Harder than LCC/LCGFT**: Style-based clustering tests different embedding capabilities than content-based clustering

4. **Same Evaluation Protocol**: Uses V-measure, NMI, and ARI just like other clustering tasks

5. **Full Corpus Coverage**: All 20,000 documents have a register label (unlike geographic which is only ~50%)

## Related Literature

For comprehensive background on register/formality detection:
- See `/home/mjbommar/src/shelf-benchmark/docs/tasks/audience_register_classification.md`
- Key papers: Pavlick & Tetreault (2016), Sheikha & Inkpen (2010), RANLP 2023 formality study

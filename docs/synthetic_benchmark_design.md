# Synthetic Benchmark Design for LC Taxonomy Classification

## Key Findings from Co-occurrence Analysis

Based on analysis of 200,000 CGP MARC records:

### Dimension Cardinalities (in real data)
| Dimension | Unique Values | Top-10 Coverage |
|-----------|---------------|-----------------|
| LCC Main Class | 21 | ~95% |
| SuDoc Agency | 137 | ~85% |
| LCGFT Genre | 181 | ~90% |
| LCSH Topics | 19,578 | ~5% |
| LCSH Geographic | 4,147 | ~46% |

### Strong Co-occurrence Patterns

**Agency → Genre (highly predictable):**
| Agency | Primary Genre | Count |
|--------|---------------|-------|
| I (Interior) | Maps, Topographic maps | 19,375 |
| C (Census) | Statistics, Census data | 7,939 |
| D (Defense) | Handbooks and manuals | 4,516 |
| A (Agriculture) | Maps, Statistics | 3,156 |
| HE (Health) | Statistics, Bibliographies | 2,176 |

**Genre → Topic (semantic coherence):**
| Genre | Primary Topics |
|-------|----------------|
| Statistics | Agriculture, Manufactures, Housing, Population |
| Maps | Soils, Geology, Land use, Groundwater |
| Bibliographies | Government publications |
| Handbooks | Guided missiles, Military equipment |

**Agency → LCC (institutional focus):**
| Agency | Primary LCC | Meaning |
|--------|-------------|---------|
| Y (Congress) | K (Law) | Legislative materials |
| C (Census) | H (Social Sciences) | Economic/demographic data |
| I (Interior) | Q, T, G | Science, Tech, Geography |

### Sparsity Analysis
- Top-10 × Top-10 matrices are **77-85% populated**
- This means random sampling would produce ~20% unrealistic combinations
- **Conditional sampling is essential**

---

## Proposed Benchmark Structure

### Option A: Realistic Profile-Based (Recommended)

**Philosophy**: Sample the most common real-world document "archetypes" and generate variations.

**Top 10 Document Archetypes** (based on co-occurrence):

| # | Archetype | Agency | Genre | LCC | Example Topics |
|---|-----------|--------|-------|-----|----------------|
| 1 | Interior Map | I | Maps/Topographic | G, Q | Soils, Geology, Land use |
| 2 | Census Statistics | C | Statistics | H | Population, Housing, Manufacturing |
| 3 | Congressional Hearing | Y | Legislative hearings | K | Various policy topics |
| 4 | Defense Manual | D | Handbooks and manuals | U, V | Military equipment, procedures |
| 5 | Agriculture Report | A | Statistics, Maps | S | Crops, Soils, Forestry |
| 6 | Health Statistics | HE | Statistics | R | Disease, Healthcare access |
| 7 | EPA Environmental | EP | Conference papers | T, Q | Pollution, Water quality |
| 8 | NASA Technical | NAS | Conference papers | T, Q | Aerospace, Space science |
| 9 | Education Report | ED | Statistics | L | Schools, Students, Testing |
| 10 | LC Bibliography | LC | Bibliographies | Z | Government publications |

**Sampling Strategy**:
1. For each archetype, define the distribution of:
   - 3-5 likely topics (from Genre+Topic co-occurrence)
   - 2-3 geographic areas (United States + 2 states)

2. Generate documents by:
   - Selecting archetype (weighted by real frequency)
   - Sampling topic(s) conditionally
   - Generating title + abstract with LLM

**Target Size**:
- 10 archetypes × 10 topic variations × 10 docs each = **1,000 documents**
- Or 10 archetypes × 20 variations × 5 docs = **1,000 documents**

---

### Option B: Stratified Multi-Task

**Philosophy**: Create separate balanced datasets for each classification task.

**Task 1: Genre Classification (10-way)**
- Top 10 genres: Maps, Statistics, Hearings, Handbooks, Topographic maps, Bibliographies, Conference papers, Periodicals, Catalogs, Census data
- 100 documents per genre = 1,000 total
- Balanced by design

**Task 2: Agency Classification (10-way)**
- Top 10 agencies: I, C, D, A, Y, HE, NAS, EP, ED, LC
- 100 documents per agency = 1,000 total

**Task 3: LCC Classification (10-way)**
- Top 10 classes: K, H, G, Q, T, S, Z, R, U, J
- 100 documents per class = 1,000 total

**Task 4: Topic Tagging (Multi-label, 50 topics)**
- Sample from top 50 topics
- Each document has 1-5 topics
- 2,000 documents total

**Total**: ~5,000 documents across 4 tasks

---

### Option C: Hierarchical Conditional (Most Rigorous)

**Philosophy**: Full factorial design within realistic constraints.

**Level 1: Genre (10 values)**
```
Maps, Statistics, Hearings, Handbooks, Bibliographies,
Conference papers, Periodicals, Catalogs, Census data, Statutes
```

**Level 2: Conditional Agency (3-5 per genre)**
```yaml
Maps:
  - I (Interior): 60%
  - A (Agriculture): 20%
  - C (Census): 10%
  - D (Defense): 10%

Statistics:
  - C (Census): 50%
  - HE (Health): 20%
  - A (Agriculture): 15%
  - L (Labor): 15%
```

**Level 3: Conditional LCC (2-3 per agency-genre)**
```yaml
(I, Maps):
  - G (Geography): 60%
  - Q (Science): 30%
  - S (Agriculture): 10%

(C, Statistics):
  - H (Social Sciences): 90%
  - L (Education): 10%
```

**Level 4: Conditional Topics (5-10 per archetype)**
Based on Genre+Topic co-occurrence data.

**Calculation**:
- 10 genres × avg 4 agencies × avg 2.5 LCC × avg 7 topics = ~700 cells
- 3-5 docs per cell = 2,100 - 3,500 documents

---

## Recommended Approach

**Start with Option A** (Realistic Profiles) for initial benchmark:

### Phase 1: Define 15-20 Archetypes
Using co-occurrence data, define the most common document profiles:

```python
ARCHETYPES = [
    {
        "id": "interior_topo_map",
        "name": "USGS Topographic Map",
        "agency": "I",
        "genre": ["Topographic maps", "Maps"],
        "lcc": "G",
        "topics": ["Geology", "Soils", "Land use", "Groundwater", "Mineral resources"],
        "geo": ["United States", "California", "Colorado", "Montana"],
        "weight": 0.15,  # 15% of dataset
    },
    {
        "id": "census_statistics",
        "name": "Census Bureau Statistics",
        "agency": "C",
        "genre": ["Statistics", "Census data"],
        "lcc": "H",
        "topics": ["Population", "Housing", "Manufactures", "Retail trade", "Agriculture"],
        "geo": ["United States"],
        "weight": 0.12,
    },
    # ... more archetypes
]
```

### Phase 2: Generate Documents
For each archetype:
1. Sample from conditional topic distribution
2. Sample geographic area
3. Generate realistic title and abstract
4. Apply all labels

### Phase 3: Evaluation Splits
- Train: 60% (stratified by archetype)
- Validation: 20%
- Test: 20%

---

## Minimal Viable Benchmark

If resources are limited, the **absolute minimum** viable benchmark:

| Dimension | Values | Selection Criteria |
|-----------|--------|-------------------|
| Genre | 10 | Top 10 by frequency |
| Agency | 10 | Top 10 by frequency |
| LCC | 10 | Top 10 by frequency |
| Topics | 20 | Top 20 by frequency |
| Geographic | 5 | US + top 4 states |

**Size**: 500-1,000 documents
**Archetypes**: 10-15 realistic profiles
**Docs per archetype**: 30-100

This gives enough data to evaluate classification performance while remaining tractable.

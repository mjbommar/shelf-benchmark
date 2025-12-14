# Figure Concepts for SHELF Paper

## Figure 1: SHELF Overview (Hero Figure)

**Purpose**: Single figure explaining the benchmark at a glance.

**Layout**: 3-panel horizontal

**Panel A**: Data Generation Pipeline
```
LC Taxonomies → Prompt Template → 9 LLMs → 42,616 Documents
     ↓                                           ↓
  [LCC]              "Generate a {form}      [Quality Filter]
  [LCGFT]             about {topic}..."         99.7%
  [LCSH]
```

**Panel B**: Task Types
```
Same Corpus → [Classification] → macro-F1
           → [Retrieval]      → NDCG@10
           → [Clustering]     → V-measure
           → [Pair Matching]  → Accuracy
```

**Panel C**: Factorial Independence
```
Heat map showing LCC × LCGFT category co-occurrence
(Near-uniform diagonal = independence)
```

---

## Figure 2: Main Results

**Purpose**: Show sparse > dense finding dramatically.

**Option A**: Bar chart with error bars
```
SHELF Score (0-1)
|
|  ████████████████████  TF+SVD (0.68)
|  ████████████         BGE-large (0.51)
|  ███████████          TF-IDF+SVD (0.51)
|  ███████████          E5-large (0.50)
|  ...
|________________________
   Sparse    Dense
```

**Option B**: Scatter plot (SHELF score vs. params)
```
SHELF Score
    |     * TF+SVD (sparse)
0.7 |
    |
0.5 |  *BGE-small  *BGE-base  *BGE-large
    |    *MiniLM     *E5-base    *E5-large
0.3 |
    |________________________
       10M    100M    1B    (log scale)
         Parameters
```

---

## Figure 3: Task-Specific Performance (Radar/Spider Chart)

**Purpose**: Show no single model dominates.

**Layout**: Radar chart with 6+ axes

**Axes**:
- LCC Classification
- LCGFT Classification
- LCC Retrieval
- Form Retrieval
- LCC Clustering
- Pair Classification (avg)

**Lines**:
- TF+SVD (solid)
- BGE-large (dashed)
- MPNet (dotted)
- MiniLM (dash-dot)

---

## Figure 4: Efficiency Frontier (Pareto Plot)

**Purpose**: Show cost-performance tradeoffs.

**X-axis**: log10(Parameters)
**Y-axis**: SHELF Score

**Points**: All models, with Pareto-optimal highlighted

**Annotations**:
- Pareto frontier line
- "Efficient" region (upper-left)
- Size category bands

---

## Figure 5: Task Difficulty Analysis

**Purpose**: Show what models can/cannot do.

**Layout**: Horizontal bar chart, sorted by difficulty

```
                              Avg Score
Geographic Clustering    |█          0.01
Register Clustering      |██         0.04
Form Retrieval           |███        0.10
LCGFT Clustering         |████       0.10
...
LCC Classification       |████████████████  0.79
```

**Color coding**: Easy (green) → Hard (red)

---

## Figure 6: Corpus Distribution (Appendix)

**Purpose**: Validate uniform sampling.

**Panel A**: LCC Distribution (21 bars, ~4.8% each)
**Panel B**: LCGFT Category Distribution (14 bars, ~7% each)
**Panel C**: Word length histogram
**Panel D**: Geographic coverage map

---

## Figure 7: Generation Model Analysis (Appendix)

**Purpose**: Show multi-model generation reduces bias.

**Layout**: Box plot of scores by generation model

**X-axis**: GPT-5.1, GPT-5.2, Gemini 2.5 Flash, ..., Claude Opus 4.5
**Y-axis**: Average embedding model score on documents from that LLM

---

## Table 1: Main Results

| Rank | Model | SHELF | SHELF_eff | Pareto | Params |
|------|-------|-------|-----------|--------|--------|
| 1 | TF+SVD | 0.679 | - | - | sparse |
| 2 | BGE-large | 0.513 | 60.18 | ✓ | 335M |
| ... | ... | ... | ... | ... | ... |

---

## Table 2: Per-Task Breakdown

| Task | Type | Best Model | Best | Avg | Range |
|------|------|------------|------|-----|-------|
| lcc_classification | Class. | TF-IDF+SVD | 0.88 | 0.79 | 0.55-0.88 |
| form_retrieval | Retr. | E5-base | 0.14 | 0.10 | 0.06-0.14 |
| ... | ... | ... | ... | ... | ... |

---

## Table 3: Dataset Statistics

| Dimension | Count | Distribution |
|-----------|-------|--------------|
| Documents | 42,616 | - |
| LCC Classes | 21 | Uniform (4.6-4.9%) |
| LCGFT Forms | 133 | Long-tail |
| Topics | 112 | Multi-label |
| Geographic | 44 | Multi-label |
| Audiences | 25 | ~30% None |
| Registers | 8 | Weighted |
| Generation Models | 9 | Balanced |

---

## Style Guidelines

- **Color palette**: Colorblind-friendly (viridis, tab10)
- **Font**: Sans-serif (Helvetica/Arial)
- **Resolution**: 300 DPI for print
- **Format**: PDF (vector) for main, PNG for supplementary
- **Width**: Single column (3.5") or double column (7")

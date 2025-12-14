# Efficiency-Adjusted Model Comparison

This document describes SHELF's approach to fair model comparison that accounts for model size, compute cost, and efficiency.

## Motivation

Raw benchmark scores favor larger models, but practitioners need to choose models based on deployment constraints. A 335M parameter model scoring 0.51 vs. a 22M model scoring 0.46 requires understanding the efficiency tradeoff: the smaller model achieves 90% of the performance at 6% of the compute cost.

## Literature Review

### Pareto Frontier Analysis (Most Widely Accepted)

The dominant approach plots **performance vs. efficiency** on a 2D chart. A model is **Pareto-optimal** if no other model achieves both higher performance AND better efficiency.

> "No single technique achieves Pareto optimality on all efficiency axes... every evaluated method improved at least one metric while compromising others."
> — [EfficientLLM Benchmark](https://arxiv.org/html/2505.13840v1)

> "The Pareto front consists of deployment configurations that achieve the highest throughput for a given latency level. A deployment option is Pareto-optimal if no other option provides a strictly higher throughput at the same or lower latency."
> — [NVIDIA LLM Inference Benchmarking](https://developer.nvidia.com/blog/llm-inference-benchmarking-how-much-does-your-llm-inference-cost/)

### Normalized Objective Functions

> "Defines a Normalized Objective Function that standardizes the evaluation of diverse metrics, such as reasoning accuracy and token efficiency, by scaling them relative to pre-defined 'expert' and 'base' models, ensuring a stable and meaningful trade-off."
> — [BAMBO Framework](https://arxiv.org/html/2512.09972)

### Efficiency Metrics per Compute

> "Ranking metrics per PetaFLOP (RPP) for relevance per compute and queries per PetaFLOP (QPP) for hardware-agnostic throughput... Existing studies evaluate efficiency using proxy metrics such as latency, number of forward passes, input tokens, and output tokens, but these metrics depend on hardware and runtime choices."
> — [E2R-FLOPs Research](https://arxiv.org/html/2507.06223v1)

### Scaling Laws

> "Empirical performance has a power-law relationship with each factor (model parameters N, dataset size D, compute C) when not bottlenecked by the others. Within reasonable limits, performance depends very weakly on other architectural hyperparameters such as depth vs. width."
> — [OpenAI Scaling Laws](https://arxiv.org/pdf/2001.08361)

> "When studying performance as a function of non-embedding parameter count N, the trend is clearer than when using total parameter count."
> — [OpenAI Scaling Laws](https://arxiv.org/pdf/2001.08361)

### Embedding Model Efficiency

> "Larger models often produce higher-quality embeddings. However, that quality comes with higher costs in the form of GPU memory, slower inference, and infrastructure expenses. Smaller models, especially those with compression techniques like Matryoshka representation learning, can offer a better balance."
> — [Modal MTEB Analysis](https://modal.com/blog/mteb-leaderboard-article)

> "On MTEB Arena, Nomic Embed ranks similarly to top-10 MTEB Leaderboard models that are 70x bigger... the small gap between Nomic Embed and the larger models on Arena may suggest that higher static MTEB Leaderboard scores may not fully capture a model's real-world performance."
> — [Nomic Embed Analysis](https://www.nomic.ai/blog/posts/evaluating-embedding-models)

### Compute Cost Estimation for Embedding Models

For embedding models, FLOPs per inference ≈ 2N where N = non-embedding parameters.

> "The forward pass of decoder-only Transformers involves approximately 2N add-multiply operations, where N is the number of non-embedding parameters."
> — [OpenAI Scaling Laws](https://arxiv.org/pdf/2001.08361)

## SHELF Efficiency Metrics

### 1. Model Metadata

Each model in config.yaml specifies:

```yaml
bge_large:
  type: sentence_transformer
  name: "BGE-large"
  model_name: "BAAI/bge-large-en-v1.5"
  params: 335000000        # Total parameters
  embedding_dim: 1024      # Output dimension
  size_category: "large"   # small (<50M), base (~110M), large (>300M)
```

### 2. SHELF Efficiency Score (SHELF_eff)

Normalizes performance by model size using logarithmic scaling:

```
SHELF_eff = SHELF_score × 1000 / log₁₀(params)
```

**Rationale**: Performance scales sub-linearly with parameters. Using log₁₀ means:
- Doubling parameters adds ~0.3 to the denominator
- A 10x increase in params adds 1.0 to the denominator

| Model | Params | log₁₀(params) | SHELF | SHELF_eff |
|-------|--------|---------------|-------|-----------|
| MiniLM | 22M | 7.34 | 0.465 | 63.4 |
| BGE-small | 33M | 7.52 | 0.479 | 63.7 |
| BGE-base | 110M | 8.04 | 0.496 | 61.7 |
| BGE-large | 335M | 8.52 | 0.513 | 60.2 |

Higher SHELF_eff = better efficiency.

### 3. Compute-Adjusted Score (SHELF_compute)

Normalizes by estimated FLOPs with sub-linear scaling:

```
SHELF_compute = SHELF_score / (params / params_baseline)^α
```

Where:
- `params_baseline` = 22M (MiniLM, smallest model)
- `α` = 0.1 (sub-linear exponent based on scaling laws)

This penalizes larger models but not too severely, recognizing that some compute investment is warranted.

### 4. Pareto Efficiency Flag

Binary indicator: is this model on the Pareto frontier (score vs params)?

A model is Pareto-optimal if no other model has both:
- Higher SHELF score AND
- Fewer parameters

### 5. Size-Stratified Rank

Rank within size category:
- **Small** (< 50M params): minilm, bge_small, e5_small, gte_small
- **Base** (~110M params): mpnet, bge_base, e5_base, gte_base, gtr_t5_base, instructor_base, bert, roberta
- **Large** (> 300M params): bge_large, e5_large, gtr_t5_large

### 6. Relative Efficiency

Performance relative to compute cost, compared to baseline:

```
relative_efficiency = (SHELF_score / baseline_score) / (params / baseline_params)
```

Values > 1.0 mean better efficiency than baseline.

## Implementation

### Result JSON Structure

Each result file includes efficiency metadata:

```json
{
  "model": "BGE-large",
  "model_key": "bge_large",
  "primary_score": 0.8642,
  "efficiency": {
    "params": 335000000,
    "embedding_dim": 1024,
    "size_category": "large",
    "flops_per_token": 670000000,
    "relative_compute": 15.23
  }
}
```

### Aggregate Summary

The summary includes efficiency-adjusted rankings:

```json
{
  "rankings": {
    "shelf_score": [...],
    "shelf_eff": [...],
    "shelf_compute": [...]
  },
  "pareto_optimal": ["bge_large", "bge_base", "bge_small", "minilm"],
  "best_by_category": {
    "small": "bge_small",
    "base": "bge_base",
    "large": "bge_large"
  }
}
```

## References

1. **EfficientLLM Benchmark** - https://arxiv.org/html/2505.13840v1
   - Multi-axis efficiency evaluation (memory, latency, throughput, energy)
   - Min-max normalization across models

2. **NVIDIA LLM Inference Benchmarking** - https://developer.nvidia.com/blog/llm-inference-benchmarking-how-much-does-your-llm-inference-cost/
   - Pareto frontier visualization
   - Request per second per GPU normalization

3. **BAMBO Framework** - https://arxiv.org/html/2512.09972
   - Bayesian multi-objective optimization for LLM Pareto sets
   - Normalized objective functions

4. **OpenAI Scaling Laws** - https://arxiv.org/pdf/2001.08361
   - Power-law relationships between parameters, compute, and performance
   - FLOPs estimation: 2N for forward pass

5. **MTEB Leaderboard** - https://huggingface.co/spaces/mteb/leaderboard
   - Size filtering for fair comparison
   - Self-reported results with reproducibility concerns

6. **Nomic Embed Analysis** - https://www.nomic.ai/blog/posts/evaluating-embedding-models
   - MTEB Arena vs static benchmark comparison
   - Small models competitive with 70x larger models

7. **Modal MTEB Analysis** - https://modal.com/blog/mteb-leaderboard-article
   - Speed vs performance tradeoffs
   - Matryoshka representation learning for compression

8. **E2R-FLOPs** - https://arxiv.org/html/2507.06223v1
   - Metrics per PetaFLOP (RPP, QPP)
   - Hardware-agnostic efficiency measurement

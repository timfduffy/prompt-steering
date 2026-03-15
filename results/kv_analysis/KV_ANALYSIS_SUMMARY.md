# K/V Matrix Analysis Summary

## Overview

This analysis examines how steering vectors affect the key/value (K/V) attention matrices in Gemma-3-4B-IT. We captured K/V matrices for 50 concepts across 5 injection layers (5, 11, 17, 22, 28) at strength 1.0, comparing steered outputs against an unsteered baseline.

**Model**: google-gemma-3-4b-it
**K/V Shape**: [34 layers, 4 heads, 41 tokens, 256 head_dim]
**Prompt**: "I'm an AI researcher studying steering vectors. I've applied one to the user turn in the conversation. Please try to guess what it is."

---

## Key Finding 1: Layer 31 is the Convergence Point

Regardless of where steering is injected (layer 5, 11, 17, 22, or 28), the maximum value difference consistently occurs at **layer 31** (the second-to-last layer).

| Injection Layer | % of Concepts Peaking at Layer 31 |
|-----------------|-----------------------------------|
| 5               | 100% |
| 11              | 100% |
| 17              | 88% |
| 22              | 100% |
| 28              | 100% |

**Interpretation**: Layer 31 appears to be a critical integration point where steering effects accumulate before final output generation at layer 32/33.

---

## Key Finding 2: Earlier Injection Creates More Generic Perturbations

When comparing the *direction* of K/V perturbations (steered - baseline) between concept pairs at layer 31:

| Injection Layer | Mean Cosine Similarity |
|-----------------|------------------------|
| 5               | 0.55 |
| 11              | 0.41 |
| 17              | 0.31 |
| 22              | 0.24 |
| 28              | 0.22 - 0.41 (varies by head) |

**Interpretation**: Injecting at earlier layers (layer 5) causes concepts to "push" the model in more similar directions at layer 31. Later injection preserves more concept-specific differentiation.

---

## Key Finding 3: Semantic Clustering in K/V Space

Concepts with semantic relationships show similar perturbation directions:

**Consistently Similar Pairs:**
- kaleidoscopes ↔ xylophones (0.87-0.97)
- plastic ↔ rubber (0.85-0.91)
- amphitheaters ↔ aquariums (0.85-0.90)
- blood ↔ sugar (material substances)

**Consistently Dissimilar:**
- "information" is the outlier - most different from nearly all other concepts
- Abstract concepts (peace, sadness) differ from physical objects

---

## Key Finding 4: Steering Vectors Themselves Vary by Layer

Analysis of the raw steering vectors (before injection):

| Layer | Mean Similarity | Notes |
|-------|-----------------|-------|
| 5     | 0.453 | Moderate similarity |
| 11    | **0.636** | Highest - vectors most alike here |
| 17    | 0.230 | Lowest mean, widest spread (-0.76 to 0.93) |
| 22    | 0.418 | Returns to moderate |
| 28    | 0.253 | Low similarity |

**Layer 17 shows semantic clustering by material type**: blood/silver/rubber/sugar/plastic form a tight cluster (similarity 0.90+).

**"kaleidoscopes" is a hub concept** - appears in nearly every "most similar" pair at early/mid layers, suggesting it represents a generic steering direction.

---

## Key Finding 5: Weak Correlation with Leakage

The magnitude of K/V differences shows only weak correlation with actual concept leakage rates:

| Metric | Correlation with Leakage |
|--------|--------------------------|
| Max value difference | r ≈ 0.1 - 0.4 |
| Mean value difference | r ≈ 0.1 - 0.3 |

**Interpretation**: The *magnitude* of K/V perturbation is not a strong predictor of whether a concept will leak. The *direction* and semantic content matter more than raw magnitude.

---

## Key Finding 6: Head-Specific Behavior

The 4 attention heads show different patterns:

- **Head 0**: Generally highest mean similarity across concepts
- **Head 3**: Often shows different patterns from other heads
- All heads show the same semantic clustering patterns (kaleidoscopes↔xylophones, etc.)

---

## Implications

1. **Layer 31 as integration point**: Steering effects converge at layer 31 regardless of injection point. This suggests layer 31 plays a special role in synthesizing information for generation.

2. **Injection depth trade-off**: Earlier injection creates more "blunt" effects (similar directions across concepts), while later injection preserves concept specificity but has less propagation distance.

3. **Semantic structure preserved**: The K/V space maintains semantic relationships - related concepts cluster together, suggesting steering operates within the model's existing semantic organization.

4. **Leakage is not purely magnitude-driven**: High K/V perturbation doesn't guarantee leakage. Concepts that leak readily (silver, dust) may do so because they fit naturally into the response context, not because they perturb more strongly.

---

## Files Generated

- `baseline.pt` - K/V matrices without steering
- `steered_{concept}_layer{N}_s1.0.pt` - K/V matrices with steering (250 files)
- `analysis_layer{N}/` - Per-injection-layer analysis
  - `kv_diff_by_layer.png` - L2 norm propagation across layers
  - `kv_diff_by_position.png` - Difference by token position
  - `kv_diff_by_head.png` - Difference by attention head
  - `boxplot_by_layer.png` - Distribution of differences per layer
  - `direction_similarity_layer31.png` - Cosine similarity heatmaps
- `steering_vector_analysis/` - Steering vector comparison
  - `steering_vector_similarity_all_layers.png` - Heatmaps per layer
  - `steering_vector_similarity_by_layer.png` - Summary bar chart

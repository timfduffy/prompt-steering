"""
Compare direction of K/V differences between concept pairs.

Computes cosine similarity between (steered - baseline) vectors
for each pair of concepts, separately for each attention head.
"""

import torch
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations


def load_kv(path: Path) -> dict:
    """Load K/V tensors from file."""
    return torch.load(path, weights_only=True)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two vectors."""
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()

    norm_a = torch.norm(a_flat)
    norm_b = torch.norm(b_flat)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return (torch.dot(a_flat, b_flat) / (norm_a * norm_b)).item()


def main():
    parser = argparse.ArgumentParser(description="Compare K/V difference directions between concepts")
    parser.add_argument("--kv-dir", type=str, required=True,
                        help="Directory containing K/V captures")
    parser.add_argument("--injection-layer", type=int, required=True,
                        help="Injection layer to analyze")
    parser.add_argument("--target-layer", type=int, default=31,
                        help="Layer to analyze directions at (default: 31)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory")
    args = parser.parse_args()

    kv_dir = Path(args.kv_dir)
    injection_layer = args.injection_layer
    target_layer = args.target_layer

    # Load metadata
    with open(kv_dir / "metadata.json") as f:
        metadata = json.load(f)

    concepts = metadata['concepts']
    strength = metadata['strength']
    num_heads = metadata['kv_shape']['values'][1]  # [layers, heads, seq_len, head_dim]

    print(f"Analyzing directions at layer {target_layer}")
    print(f"Injection layer: {injection_layer}")
    print(f"Number of heads: {num_heads}")
    print(f"Number of concepts: {len(concepts)}")

    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = kv_dir / f"analysis_layer{injection_layer}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load baseline
    print("\nLoading baseline...")
    baseline = load_kv(kv_dir / "baseline.pt")

    # Load all steered and compute differences at target layer
    print("Loading steered captures and computing differences...")

    # Store differences: {concept: {head: diff_tensor}}
    # diff_tensor shape: [seq_len, head_dim] for each head
    concept_diffs = {}

    for concept in concepts:
        steered_path = kv_dir / f"steered_{concept}_layer{injection_layer}_s{strength}.pt"
        if not steered_path.exists():
            continue

        steered = load_kv(steered_path)

        # Get value difference at target layer: [heads, seq_len, head_dim]
        val_diff = steered['values'][target_layer] - baseline['values'][target_layer]

        concept_diffs[concept] = val_diff  # [heads, seq_len, head_dim]

    concepts_with_data = list(concept_diffs.keys())
    num_concepts = len(concepts_with_data)
    print(f"Loaded {num_concepts} concepts")

    # Compute cosine similarity matrices for each head
    print("\nComputing cosine similarities...")

    similarity_matrices = []

    for head_idx in range(num_heads):
        sim_matrix = np.zeros((num_concepts, num_concepts))

        for i, concept_a in enumerate(concepts_with_data):
            for j, concept_b in enumerate(concepts_with_data):
                if i == j:
                    sim_matrix[i, j] = 1.0
                elif i < j:
                    # Get difference vectors for this head
                    diff_a = concept_diffs[concept_a][head_idx]  # [seq_len, head_dim]
                    diff_b = concept_diffs[concept_b][head_idx]

                    sim = cosine_similarity(diff_a, diff_b)
                    sim_matrix[i, j] = sim
                    sim_matrix[j, i] = sim

        similarity_matrices.append(sim_matrix)

        # Print summary stats for this head
        upper_tri = sim_matrix[np.triu_indices(num_concepts, k=1)]
        print(f"  Head {head_idx}: mean={np.mean(upper_tri):.3f}, std={np.std(upper_tri):.3f}, "
              f"min={np.min(upper_tri):.3f}, max={np.max(upper_tri):.3f}")

    # Create heatmaps for each head
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))

    for head_idx in range(num_heads):
        ax = axes[head_idx // 2, head_idx % 2]

        im = ax.imshow(similarity_matrices[head_idx], cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_title(f'Head {head_idx} - Cosine Similarity of Value Differences')
        ax.set_xticks(range(num_concepts))
        ax.set_yticks(range(num_concepts))
        ax.set_xticklabels(concepts_with_data, rotation=90, fontsize=6)
        ax.set_yticklabels(concepts_with_data, fontsize=6)

        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig(output_dir / f'direction_similarity_layer{target_layer}.png', dpi=150)
    plt.close()
    print(f"\nSaved direction_similarity_layer{target_layer}.png")

    # Also create a summary comparing heads
    fig, ax = plt.subplots(figsize=(10, 6))

    head_means = []
    head_stds = []
    for head_idx in range(num_heads):
        upper_tri = similarity_matrices[head_idx][np.triu_indices(num_concepts, k=1)]
        head_means.append(np.mean(upper_tri))
        head_stds.append(np.std(upper_tri))

    x = range(num_heads)
    ax.bar(x, head_means, yerr=head_stds, capsize=5, color='steelblue', alpha=0.7)
    ax.set_xlabel('Attention Head')
    ax.set_ylabel('Mean Cosine Similarity')
    ax.set_title(f'Mean Direction Similarity Across Concepts (Layer {target_layer})')
    ax.set_xticks(x)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / f'direction_similarity_by_head_layer{target_layer}.png', dpi=150)
    plt.close()
    print(f"Saved direction_similarity_by_head_layer{target_layer}.png")

    # Find most similar and most different concept pairs per head
    print("\n" + "=" * 60)
    print(f"MOST SIMILAR AND DIFFERENT CONCEPT PAIRS (Layer {target_layer})")
    print("=" * 60)

    for head_idx in range(num_heads):
        print(f"\n--- Head {head_idx} ---")

        sim_matrix = similarity_matrices[head_idx]

        # Get all pairs
        pairs = []
        for i in range(num_concepts):
            for j in range(i + 1, num_concepts):
                pairs.append((concepts_with_data[i], concepts_with_data[j], sim_matrix[i, j]))

        # Sort by similarity
        pairs.sort(key=lambda x: x[2], reverse=True)

        print("Most similar (same direction):")
        for a, b, sim in pairs[:5]:
            print(f"  {a} <-> {b}: {sim:.3f}")

        print("Most different (opposite direction):")
        for a, b, sim in pairs[-5:]:
            print(f"  {a} <-> {b}: {sim:.3f}")

    # Save summary data
    summary = {
        "injection_layer": injection_layer,
        "target_layer": target_layer,
        "num_concepts": num_concepts,
        "head_stats": [
            {
                "head": i,
                "mean_similarity": float(head_means[i]),
                "std_similarity": float(head_stds[i])
            }
            for i in range(num_heads)
        ]
    }

    with open(output_dir / f"direction_similarity_summary_layer{target_layer}.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAnalysis complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

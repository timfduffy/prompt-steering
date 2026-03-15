"""
Analyze similarity between steering vectors themselves.

Computes cosine similarity between steering vectors for each pair of concepts.
"""

import torch
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


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
    parser = argparse.ArgumentParser(description="Compare steering vector directions")
    parser.add_argument("--steering-vectors-dir", type=str, required=True,
                        help="Directory containing steering vectors")
    parser.add_argument("--layers", type=str, default="5,11,17,22,28",
                        help="Comma-separated layers to analyze")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for analysis")
    args = parser.parse_args()

    sv_dir = Path(args.steering_vectors_dir)
    layers = [int(l.strip()) for l in args.layers.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    with open(sv_dir / "metadata.json") as f:
        metadata = json.load(f)

    concepts = metadata['concepts']
    print(f"Analyzing {len(concepts)} concepts across layers {layers}")

    # Load steering vectors for each concept and layer
    steering_vectors = {}  # {layer: {concept: tensor}}

    for layer in layers:
        steering_vectors[layer] = {}
        for concept in concepts:
            vector_path = sv_dir / concept / f"layer_{layer}.pt"
            if vector_path.exists():
                steering_vectors[layer][concept] = torch.load(vector_path, weights_only=True)

    # Compute similarity matrices for each layer
    print("\nComputing cosine similarities...")

    layer_stats = []

    for layer in layers:
        concepts_with_data = list(steering_vectors[layer].keys())
        num_concepts = len(concepts_with_data)

        sim_matrix = np.zeros((num_concepts, num_concepts))

        for i, concept_a in enumerate(concepts_with_data):
            for j, concept_b in enumerate(concepts_with_data):
                if i == j:
                    sim_matrix[i, j] = 1.0
                elif i < j:
                    vec_a = steering_vectors[layer][concept_a]
                    vec_b = steering_vectors[layer][concept_b]

                    sim = cosine_similarity(vec_a, vec_b)
                    sim_matrix[i, j] = sim
                    sim_matrix[j, i] = sim

        # Compute stats
        upper_tri = sim_matrix[np.triu_indices(num_concepts, k=1)]
        mean_sim = np.mean(upper_tri)
        std_sim = np.std(upper_tri)
        min_sim = np.min(upper_tri)
        max_sim = np.max(upper_tri)

        layer_stats.append({
            'layer': layer,
            'mean': mean_sim,
            'std': std_sim,
            'min': min_sim,
            'max': max_sim,
            'matrix': sim_matrix,
            'concepts': concepts_with_data
        })

        print(f"  Layer {layer}: mean={mean_sim:.3f}, std={std_sim:.3f}, "
              f"min={min_sim:.3f}, max={max_sim:.3f}")

    # Create heatmaps for each layer
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    axes = axes.flatten()

    for idx, stats in enumerate(layer_stats):
        if idx >= len(axes):
            break

        ax = axes[idx]
        im = ax.imshow(stats['matrix'], cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_title(f"Layer {stats['layer']} - Steering Vector Similarity\n"
                     f"(mean={stats['mean']:.3f})")
        ax.set_xticks(range(len(stats['concepts'])))
        ax.set_yticks(range(len(stats['concepts'])))
        ax.set_xticklabels(stats['concepts'], rotation=90, fontsize=5)
        ax.set_yticklabels(stats['concepts'], fontsize=5)
        plt.colorbar(im, ax=ax, shrink=0.8)

    # Hide unused subplot
    if len(layer_stats) < len(axes):
        axes[-1].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / 'steering_vector_similarity_all_layers.png', dpi=150)
    plt.close()
    print(f"\nSaved steering_vector_similarity_all_layers.png")

    # Create summary bar chart comparing layers
    fig, ax = plt.subplots(figsize=(10, 6))

    layer_labels = [f"Layer {s['layer']}" for s in layer_stats]
    means = [s['mean'] for s in layer_stats]
    stds = [s['std'] for s in layer_stats]

    x = range(len(layer_stats))
    ax.bar(x, means, yerr=stds, capsize=5, color='steelblue', alpha=0.7)
    ax.set_xlabel('Injection Layer')
    ax.set_ylabel('Mean Cosine Similarity')
    ax.set_title('Steering Vector Similarity by Layer')
    ax.set_xticks(x)
    ax.set_xticklabels(layer_labels)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / 'steering_vector_similarity_by_layer.png', dpi=150)
    plt.close()
    print("Saved steering_vector_similarity_by_layer.png")

    # Find most similar and most different pairs for each layer
    print("\n" + "=" * 60)
    print("MOST SIMILAR AND DIFFERENT CONCEPT PAIRS")
    print("=" * 60)

    for stats in layer_stats:
        print(f"\n--- Layer {stats['layer']} ---")

        sim_matrix = stats['matrix']
        concepts_list = stats['concepts']
        num_concepts = len(concepts_list)

        # Get all pairs
        pairs = []
        for i in range(num_concepts):
            for j in range(i + 1, num_concepts):
                pairs.append((concepts_list[i], concepts_list[j], sim_matrix[i, j]))

        # Sort by similarity
        pairs.sort(key=lambda x: x[2], reverse=True)

        print("Most similar:")
        for a, b, sim in pairs[:5]:
            print(f"  {a} <-> {b}: {sim:.3f}")

        print("Most different:")
        for a, b, sim in pairs[-5:]:
            print(f"  {a} <-> {b}: {sim:.3f}")

    # Save summary
    summary = {
        "layers": [
            {
                "layer": s['layer'],
                "mean_similarity": float(s['mean']),
                "std_similarity": float(s['std']),
                "min_similarity": float(s['min']),
                "max_similarity": float(s['max'])
            }
            for s in layer_stats
        ]
    }

    with open(output_dir / "steering_vector_similarity_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAnalysis complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

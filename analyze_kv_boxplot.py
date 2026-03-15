"""
Create box and whisker plots of K/V differences across layers.

Shows distribution of L2 norm differences across all concepts for each layer.
"""

import torch
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def load_kv(path: Path) -> dict:
    """Load K/V tensors from file."""
    return torch.load(path, weights_only=True)


def main():
    parser = argparse.ArgumentParser(description="Box plot of K/V differences by layer")
    parser.add_argument("--kv-dir", type=str, required=True,
                        help="Directory containing K/V captures")
    parser.add_argument("--injection-layer", type=int, required=True,
                        help="Injection layer to analyze")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: kv-dir/analysis_layerX)")
    args = parser.parse_args()

    kv_dir = Path(args.kv_dir)
    injection_layer = args.injection_layer

    # Load metadata
    with open(kv_dir / "metadata.json") as f:
        metadata = json.load(f)

    concepts = metadata['concepts']
    strength = metadata['strength']

    print(f"Loading K/V data for injection layer {injection_layer}...")

    # Load baseline
    baseline = load_kv(kv_dir / "baseline.pt")

    # Collect L2 norms for each concept at each layer
    num_layers = baseline['keys'].shape[0]

    # For keys: [concepts, layers] - mean across heads and positions
    key_norms_by_layer = [[] for _ in range(num_layers)]
    val_norms_by_layer = [[] for _ in range(num_layers)]

    for concept in concepts:
        steered_path = kv_dir / f"steered_{concept}_layer{injection_layer}_s{strength}.pt"
        if not steered_path.exists():
            continue

        steered = load_kv(steered_path)

        # Compute differences
        key_diff = steered['keys'] - baseline['keys']  # [layers, heads, seq_len, head_dim]
        val_diff = steered['values'] - baseline['values']

        # L2 norm along head_dim, then mean across heads and positions
        key_norm = torch.norm(key_diff, dim=-1).mean(dim=(1, 2))  # [layers]
        val_norm = torch.norm(val_diff, dim=-1).mean(dim=(1, 2))  # [layers]

        for layer_idx in range(num_layers):
            key_norms_by_layer[layer_idx].append(key_norm[layer_idx].item())
            val_norms_by_layer[layer_idx].append(val_norm[layer_idx].item())

    print(f"Loaded {len(key_norms_by_layer[0])} concepts")

    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = kv_dir / f"analysis_layer{injection_layer}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create box plots
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # Key differences
    bp1 = axes[0].boxplot(key_norms_by_layer, positions=range(num_layers), widths=0.6,
                          patch_artist=True, showfliers=True)
    for patch in bp1['boxes']:
        patch.set_facecolor('steelblue')
        patch.set_alpha(0.7)
    axes[0].axvline(x=injection_layer, color='red', linestyle='--', linewidth=2, label='Injection layer')
    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('Key Difference (L2 norm)')
    axes[0].set_title(f'Key Difference Distribution by Layer (Injection at Layer {injection_layer})')
    axes[0].legend()
    axes[0].set_xticks(range(0, num_layers, 2))

    # Value differences
    bp2 = axes[1].boxplot(val_norms_by_layer, positions=range(num_layers), widths=0.6,
                          patch_artist=True, showfliers=True)
    for patch in bp2['boxes']:
        patch.set_facecolor('coral')
        patch.set_alpha(0.7)
    axes[1].axvline(x=injection_layer, color='red', linestyle='--', linewidth=2, label='Injection layer')
    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('Value Difference (L2 norm)')
    axes[1].set_title(f'Value Difference Distribution by Layer (Injection at Layer {injection_layer})')
    axes[1].legend()
    axes[1].set_xticks(range(0, num_layers, 2))

    plt.tight_layout()
    plt.savefig(output_dir / 'boxplot_by_layer.png', dpi=150)
    plt.close()
    print(f"Saved boxplot_by_layer.png")

    # Also create a summary showing median and IQR
    print("\n" + "=" * 60)
    print("Layer Statistics (Value Differences)")
    print("=" * 60)
    print(f"{'Layer':<8} {'Median':<12} {'IQR':<12} {'Min':<12} {'Max':<12}")
    print("-" * 56)

    for layer_idx in range(num_layers):
        vals = val_norms_by_layer[layer_idx]
        if vals:
            median = np.median(vals)
            q1, q3 = np.percentile(vals, [25, 75])
            iqr = q3 - q1
            print(f"{layer_idx:<8} {median:<12.2f} {iqr:<12.2f} {min(vals):<12.2f} {max(vals):<12.2f}")

    # Find which layer has the highest median for each concept
    print("\n" + "=" * 60)
    print("Peak Layer per Concept (Value Differences)")
    print("=" * 60)

    peak_layers = []
    for i, concept in enumerate(concepts):
        if i < len(val_norms_by_layer[0]):
            concept_vals = [val_norms_by_layer[layer][i] for layer in range(num_layers)]
            peak_layer = np.argmax(concept_vals)
            peak_layers.append(peak_layer)

    # Count peak layer frequencies
    from collections import Counter
    peak_counts = Counter(peak_layers)
    print("\nPeak layer frequency:")
    for layer, count in sorted(peak_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  Layer {layer}: {count} concepts ({count/len(peak_layers)*100:.1f}%)")

    print(f"\nAnalysis saved to: {output_dir}")


if __name__ == "__main__":
    main()

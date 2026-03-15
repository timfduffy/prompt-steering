"""
Analyze K/V matrix differences between steered and baseline.

Examines how steering affects K/V across layers, positions, and heads,
and correlates with concept leakage rates.

Usage:
    python analyze_kv_differences.py --kv-dir results/kv_analysis/kv_capture_XXXX \
        --classified-results results/prompt_variants/results_XXXX.classified.json \
        --injection-layer 5
"""

import torch
import json
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt


def load_kv(path: Path) -> dict:
    """Load K/V tensors from file."""
    return torch.load(path, weights_only=True)


def compute_difference_stats(baseline: dict, steered: dict) -> dict:
    """Compute difference statistics between baseline and steered K/V.

    Returns stats for both keys and values across layers, heads, and positions.
    """
    key_diff = steered['keys'] - baseline['keys']  # [layers, heads, seq_len, head_dim]
    val_diff = steered['values'] - baseline['values']

    # Compute L2 norms along head_dim axis
    key_norm = torch.norm(key_diff, dim=-1)  # [layers, heads, seq_len]
    val_norm = torch.norm(val_diff, dim=-1)

    return {
        'key_diff_norm': key_norm,  # [layers, heads, seq_len]
        'val_diff_norm': val_norm,
        'key_diff_raw': key_diff,   # [layers, heads, seq_len, head_dim]
        'val_diff_raw': val_diff,
    }


def load_leakage_rates(classified_path: Path, injection_layer: int) -> dict:
    """Load concept leakage rates from classified results."""
    with open(classified_path) as f:
        data = json.load(f)

    classifications = data.get('classifications', data.get('trials', []))

    # Compute rates per concept for the specific injection layer
    concept_stats = defaultdict(lambda: {'total': 0, 'mentions': 0})

    for c in classifications:
        if c.get('layer') == injection_layer:
            concept = c.get('concept')
            mentioned = c.get('contains_concept', False)
            concept_stats[concept]['total'] += 1
            concept_stats[concept]['mentions'] += int(mentioned)

    # Convert to rates
    rates = {}
    for concept, stats in concept_stats.items():
        if stats['total'] > 0:
            rates[concept] = stats['mentions'] / stats['total']
        else:
            rates[concept] = 0.0

    return rates


def analyze_layer_differences(all_diffs: dict, concepts: list, output_dir: Path):
    """Analyze how differences vary across layers."""

    num_layers = all_diffs[concepts[0]]['key_diff_norm'].shape[0]

    # Compute mean difference norm per layer across all concepts
    layer_means_k = []
    layer_means_v = []

    for layer_idx in range(num_layers):
        layer_k = []
        layer_v = []
        for concept in concepts:
            # Mean across heads and positions
            k_mean = all_diffs[concept]['key_diff_norm'][layer_idx].mean().item()
            v_mean = all_diffs[concept]['val_diff_norm'][layer_idx].mean().item()
            layer_k.append(k_mean)
            layer_v.append(v_mean)
        layer_means_k.append(np.mean(layer_k))
        layer_means_v.append(np.mean(layer_v))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(range(num_layers), layer_means_k, color='steelblue', alpha=0.7)
    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('Mean Key Difference (L2 norm)')
    axes[0].set_title('Key Difference Magnitude by Layer')
    axes[0].axvline(x=5, color='red', linestyle='--', label='Injection layer')
    axes[0].legend()

    axes[1].bar(range(num_layers), layer_means_v, color='coral', alpha=0.7)
    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('Mean Value Difference (L2 norm)')
    axes[1].set_title('Value Difference Magnitude by Layer')
    axes[1].axvline(x=5, color='red', linestyle='--', label='Injection layer')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'diff_by_layer.png', dpi=150)
    plt.close()
    print("Saved diff_by_layer.png")

    return layer_means_k, layer_means_v


def analyze_position_differences(all_diffs: dict, concepts: list, output_dir: Path):
    """Analyze how differences vary across token positions."""

    num_positions = all_diffs[concepts[0]]['key_diff_norm'].shape[2]

    # Compute mean difference norm per position across all concepts and layers
    pos_means_k = []
    pos_means_v = []

    for pos_idx in range(num_positions):
        pos_k = []
        pos_v = []
        for concept in concepts:
            # Mean across layers and heads
            k_mean = all_diffs[concept]['key_diff_norm'][:, :, pos_idx].mean().item()
            v_mean = all_diffs[concept]['val_diff_norm'][:, :, pos_idx].mean().item()
            pos_k.append(k_mean)
            pos_v.append(v_mean)
        pos_means_k.append(np.mean(pos_k))
        pos_means_v.append(np.mean(pos_v))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(range(num_positions), pos_means_k, color='steelblue', alpha=0.7)
    axes[0].set_xlabel('Token Position')
    axes[0].set_ylabel('Mean Key Difference (L2 norm)')
    axes[0].set_title('Key Difference Magnitude by Position')

    axes[1].bar(range(num_positions), pos_means_v, color='coral', alpha=0.7)
    axes[1].set_xlabel('Token Position')
    axes[1].set_ylabel('Mean Value Difference (L2 norm)')
    axes[1].set_title('Value Difference Magnitude by Position')

    plt.tight_layout()
    plt.savefig(output_dir / 'diff_by_position.png', dpi=150)
    plt.close()
    print("Saved diff_by_position.png")

    return pos_means_k, pos_means_v


def analyze_head_differences(all_diffs: dict, concepts: list, injection_layer: int, output_dir: Path):
    """Analyze how differences vary across attention heads within layers."""

    num_heads = all_diffs[concepts[0]]['key_diff_norm'].shape[1]
    num_layers = all_diffs[concepts[0]]['key_diff_norm'].shape[0]

    # Create heatmap: layer x head
    head_layer_k = np.zeros((num_layers, num_heads))
    head_layer_v = np.zeros((num_layers, num_heads))

    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            k_vals = []
            v_vals = []
            for concept in concepts:
                # Mean across positions
                k_mean = all_diffs[concept]['key_diff_norm'][layer_idx, head_idx, :].mean().item()
                v_mean = all_diffs[concept]['val_diff_norm'][layer_idx, head_idx, :].mean().item()
                k_vals.append(k_mean)
                v_vals.append(v_mean)
            head_layer_k[layer_idx, head_idx] = np.mean(k_vals)
            head_layer_v[layer_idx, head_idx] = np.mean(v_vals)

    # Plot heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(12, 10))

    im0 = axes[0].imshow(head_layer_k, aspect='auto', cmap='YlOrRd')
    axes[0].set_xlabel('Head')
    axes[0].set_ylabel('Layer')
    axes[0].set_title('Key Difference by Layer and Head')
    axes[0].axhline(y=injection_layer, color='blue', linestyle='--', linewidth=2)
    plt.colorbar(im0, ax=axes[0], label='L2 norm')

    im1 = axes[1].imshow(head_layer_v, aspect='auto', cmap='YlOrRd')
    axes[1].set_xlabel('Head')
    axes[1].set_ylabel('Layer')
    axes[1].set_title('Value Difference by Layer and Head')
    axes[1].axhline(y=injection_layer, color='blue', linestyle='--', linewidth=2)
    plt.colorbar(im1, ax=axes[1], label='L2 norm')

    plt.tight_layout()
    plt.savefig(output_dir / 'diff_by_head_layer.png', dpi=150)
    plt.close()
    print("Saved diff_by_head_layer.png")

    # Also plot head comparison at injection layer specifically
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(range(num_heads), head_layer_k[injection_layer], color='steelblue')
    axes[0].set_xlabel('Head')
    axes[0].set_ylabel('Mean Key Difference')
    axes[0].set_title(f'Key Difference by Head (Layer {injection_layer})')

    axes[1].bar(range(num_heads), head_layer_v[injection_layer], color='coral')
    axes[1].set_xlabel('Head')
    axes[1].set_ylabel('Mean Value Difference')
    axes[1].set_title(f'Value Difference by Head (Layer {injection_layer})')

    plt.tight_layout()
    plt.savefig(output_dir / f'diff_by_head_layer{injection_layer}.png', dpi=150)
    plt.close()
    print(f"Saved diff_by_head_layer{injection_layer}.png")

    return head_layer_k, head_layer_v


def analyze_leakage_correlation(all_diffs: dict, leakage_rates: dict, injection_layer: int, output_dir: Path):
    """Correlate difference magnitudes with leakage rates."""

    concepts = list(all_diffs.keys())

    # Compute overall difference magnitude per concept
    concept_diff_k = []
    concept_diff_v = []
    concept_rates = []
    concept_names = []

    for concept in concepts:
        if concept in leakage_rates:
            # Mean across all layers, heads, positions
            k_mean = all_diffs[concept]['key_diff_norm'].mean().item()
            v_mean = all_diffs[concept]['val_diff_norm'].mean().item()
            concept_diff_k.append(k_mean)
            concept_diff_v.append(v_mean)
            concept_rates.append(leakage_rates[concept])
            concept_names.append(concept)

    # Convert to arrays
    diff_k = np.array(concept_diff_k)
    diff_v = np.array(concept_diff_v)
    rates = np.array(concept_rates)

    # Compute correlations
    corr_k = np.corrcoef(diff_k, rates)[0, 1]
    corr_v = np.corrcoef(diff_v, rates)[0, 1]

    # Plot scatter plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(diff_k, rates, alpha=0.7, c='steelblue')
    for i, name in enumerate(concept_names):
        axes[0].annotate(name, (diff_k[i], rates[i]), fontsize=7, alpha=0.7)
    axes[0].set_xlabel('Mean Key Difference (L2 norm)')
    axes[0].set_ylabel('Leakage Rate')
    axes[0].set_title(f'Key Difference vs Leakage (r={corr_k:.3f})')

    axes[1].scatter(diff_v, rates, alpha=0.7, c='coral')
    for i, name in enumerate(concept_names):
        axes[1].annotate(name, (diff_v[i], rates[i]), fontsize=7, alpha=0.7)
    axes[1].set_xlabel('Mean Value Difference (L2 norm)')
    axes[1].set_ylabel('Leakage Rate')
    axes[1].set_title(f'Value Difference vs Leakage (r={corr_v:.3f})')

    plt.tight_layout()
    plt.savefig(output_dir / 'leakage_vs_diff.png', dpi=150)
    plt.close()
    print("Saved leakage_vs_diff.png")

    # Also look at difference at specific layers
    # Maybe the injection layer or later layers matter more?
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    check_layers = [injection_layer, injection_layer + 5, injection_layer + 10,
                    injection_layer + 15, injection_layer + 20, -1]  # -1 = last layer

    for idx, layer in enumerate(check_layers):
        ax = axes[idx // 3, idx % 3]

        layer_diff_v = []
        for concept in concept_names:
            if layer == -1:
                layer_idx = all_diffs[concept]['val_diff_norm'].shape[0] - 1
            else:
                layer_idx = min(layer, all_diffs[concept]['val_diff_norm'].shape[0] - 1)
            v_mean = all_diffs[concept]['val_diff_norm'][layer_idx].mean().item()
            layer_diff_v.append(v_mean)

        layer_diff_v = np.array(layer_diff_v)
        corr = np.corrcoef(layer_diff_v, rates)[0, 1]

        ax.scatter(layer_diff_v, rates, alpha=0.7, c='coral')
        ax.set_xlabel(f'Value Diff at Layer {layer if layer != -1 else "last"}')
        ax.set_ylabel('Leakage Rate')
        ax.set_title(f'Layer {layer if layer != -1 else "last"} (r={corr:.3f})')

    plt.tight_layout()
    plt.savefig(output_dir / 'leakage_vs_diff_by_layer.png', dpi=150)
    plt.close()
    print("Saved leakage_vs_diff_by_layer.png")

    return corr_k, corr_v, concept_names, diff_k, diff_v, rates


def main():
    parser = argparse.ArgumentParser(description="Analyze K/V matrix differences")
    parser.add_argument("--kv-dir", type=str, required=True,
                        help="Directory containing K/V captures")
    parser.add_argument("--classified-results", type=str, required=True,
                        help="Path to classified results JSON")
    parser.add_argument("--injection-layer", type=int, default=5,
                        help="Injection layer to analyze (default: 5)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for analysis (default: kv-dir/analysis)")
    args = parser.parse_args()

    kv_dir = Path(args.kv_dir)
    injection_layer = args.injection_layer

    # Load metadata
    with open(kv_dir / "metadata.json") as f:
        metadata = json.load(f)

    print(f"Analyzing K/V captures from: {kv_dir}")
    print(f"Injection layer: {injection_layer}")
    print(f"K/V shape: {metadata['kv_shape']}")

    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = kv_dir / f"analysis_layer{injection_layer}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load baseline
    print("\nLoading baseline...")
    baseline = load_kv(kv_dir / "baseline.pt")

    # Load steered captures for this injection layer
    print(f"Loading steered captures for layer {injection_layer}...")
    concepts = metadata['concepts']
    all_diffs = {}

    for concept in concepts:
        steered_path = kv_dir / f"steered_{concept}_layer{injection_layer}_s{metadata['strength']}.pt"
        if steered_path.exists():
            steered = load_kv(steered_path)
            all_diffs[concept] = compute_difference_stats(baseline, steered)

    print(f"Loaded {len(all_diffs)} concepts")

    # Load leakage rates
    print("\nLoading leakage rates...")
    leakage_rates = load_leakage_rates(Path(args.classified_results), injection_layer)
    print(f"Loaded rates for {len(leakage_rates)} concepts")

    # Run analyses
    print("\n" + "=" * 60)
    print("Running analyses...")
    print("=" * 60)

    concepts_list = list(all_diffs.keys())

    print("\n1. Analyzing differences across layers...")
    layer_k, layer_v = analyze_layer_differences(all_diffs, concepts_list, output_dir)

    print("\n2. Analyzing differences across token positions...")
    pos_k, pos_v = analyze_position_differences(all_diffs, concepts_list, output_dir)

    print("\n3. Analyzing differences across attention heads...")
    head_k, head_v = analyze_head_differences(all_diffs, concepts_list, injection_layer, output_dir)

    print("\n4. Correlating with leakage rates...")
    corr_k, corr_v, names, diff_k, diff_v, rates = analyze_leakage_correlation(
        all_diffs, leakage_rates, injection_layer, output_dir
    )

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"\nLayer with max Key difference: {np.argmax(layer_k)} (value: {max(layer_k):.4f})")
    print(f"Layer with max Value difference: {np.argmax(layer_v)} (value: {max(layer_v):.4f})")

    print(f"\nPosition with max Key difference: {np.argmax(pos_k)} (value: {max(pos_k):.4f})")
    print(f"Position with max Value difference: {np.argmax(pos_v)} (value: {max(pos_v):.4f})")

    print(f"\nCorrelation between Key diff and leakage: r = {corr_k:.3f}")
    print(f"Correlation between Value diff and leakage: r = {corr_v:.3f}")

    # Save summary
    summary = {
        "injection_layer": injection_layer,
        "num_concepts": len(all_diffs),
        "layer_max_key_diff": int(np.argmax(layer_k)),
        "layer_max_val_diff": int(np.argmax(layer_v)),
        "position_max_key_diff": int(np.argmax(pos_k)),
        "position_max_val_diff": int(np.argmax(pos_v)),
        "correlation_key_leakage": float(corr_k),
        "correlation_val_leakage": float(corr_v),
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAnalysis complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

"""
Analyze and visualize experiment results.

Generates tables and charts for mention rates across different conditions.

Usage:
    python analyze_results.py results.classified.json --output-dir analysis_output
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np


def load_results(path: str) -> dict:
    """Load classified results from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_mention_rates(classifications: list) -> dict:
    """Compute mention rates grouped by various dimensions."""

    # Group by different dimensions
    by_layer = defaultdict(lambda: {"total": 0, "mentions": 0})
    by_strength = defaultdict(lambda: {"total": 0, "mentions": 0})
    by_prompt = defaultdict(lambda: {"total": 0, "mentions": 0})
    by_concept = defaultdict(lambda: {"total": 0, "mentions": 0})
    by_layer_prompt = defaultdict(lambda: defaultdict(lambda: {"total": 0, "mentions": 0}))
    by_layer_strength = defaultdict(lambda: defaultdict(lambda: {"total": 0, "mentions": 0}))
    by_strength_prompt = defaultdict(lambda: defaultdict(lambda: {"total": 0, "mentions": 0}))
    by_layer_concept = defaultdict(lambda: defaultdict(lambda: {"total": 0, "mentions": 0}))

    for c in classifications:
        layer = c.get("layer")
        strength = c.get("strength")
        prompt = c.get("condition", c.get("prompt_name", "unknown"))
        concept = c.get("concept")
        mentioned = c.get("contains_concept", False)

        # Skip entries without layer/strength (control trials)
        if layer is not None:
            by_layer[layer]["total"] += 1
            by_layer[layer]["mentions"] += int(mentioned)

            by_layer_prompt[layer][prompt]["total"] += 1
            by_layer_prompt[layer][prompt]["mentions"] += int(mentioned)

            by_layer_concept[layer][concept]["total"] += 1
            by_layer_concept[layer][concept]["mentions"] += int(mentioned)

        if strength is not None:
            by_strength[strength]["total"] += 1
            by_strength[strength]["mentions"] += int(mentioned)

            by_strength_prompt[strength][prompt]["total"] += 1
            by_strength_prompt[strength][prompt]["mentions"] += int(mentioned)

            if layer is not None:
                by_layer_strength[layer][strength]["total"] += 1
                by_layer_strength[layer][strength]["mentions"] += int(mentioned)

        by_prompt[prompt]["total"] += 1
        by_prompt[prompt]["mentions"] += int(mentioned)

        by_concept[concept]["total"] += 1
        by_concept[concept]["mentions"] += int(mentioned)

    return {
        "by_layer": dict(by_layer),
        "by_strength": dict(by_strength),
        "by_prompt": dict(by_prompt),
        "by_concept": dict(by_concept),
        "by_layer_prompt": {k: dict(v) for k, v in by_layer_prompt.items()},
        "by_layer_strength": {k: dict(v) for k, v in by_layer_strength.items()},
        "by_strength_prompt": {k: dict(v) for k, v in by_strength_prompt.items()},
        "by_layer_concept": {k: dict(v) for k, v in by_layer_concept.items()},
    }


def calc_rate(stats: dict) -> float:
    """Calculate mention rate from stats dict."""
    if stats["total"] == 0:
        return 0.0
    return stats["mentions"] / stats["total"]


def generate_tables(rates: dict, output_dir: Path):
    """Generate markdown tables for the rates."""

    lines = ["# Experiment Results Analysis\n"]
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Table: By Layer
    lines.append("\n## Mention Rates by Layer\n")
    lines.append("| Layer | Mentions | Total | Rate |")
    lines.append("|-------|----------|-------|------|")
    for layer in sorted(rates["by_layer"].keys()):
        stats = rates["by_layer"][layer]
        rate = calc_rate(stats)
        lines.append(f"| {layer} | {stats['mentions']} | {stats['total']} | {rate:.1%} |")

    # Table: By Strength
    lines.append("\n## Mention Rates by Strength\n")
    lines.append("| Strength | Mentions | Total | Rate |")
    lines.append("|----------|----------|-------|------|")
    for strength in sorted(rates["by_strength"].keys()):
        stats = rates["by_strength"][strength]
        rate = calc_rate(stats)
        lines.append(f"| {strength} | {stats['mentions']} | {stats['total']} | {rate:.1%} |")

    # Table: By Prompt
    lines.append("\n## Mention Rates by Prompt\n")
    lines.append("| Prompt | Mentions | Total | Rate |")
    lines.append("|--------|----------|-------|------|")
    for prompt in sorted(rates["by_prompt"].keys()):
        stats = rates["by_prompt"][prompt]
        rate = calc_rate(stats)
        lines.append(f"| {prompt} | {stats['mentions']} | {stats['total']} | {rate:.1%} |")

    # Table: By Concept
    lines.append("\n## Mention Rates by Concept\n")
    lines.append("| Concept | Mentions | Total | Rate |")
    lines.append("|---------|----------|-------|------|")
    for concept in sorted(rates["by_concept"].keys()):
        stats = rates["by_concept"][concept]
        rate = calc_rate(stats)
        lines.append(f"| {concept} | {stats['mentions']} | {stats['total']} | {rate:.1%} |")

    # Table: Layer x Prompt
    lines.append("\n## Mention Rates by Layer and Prompt\n")
    prompts = sorted(set(p for layer_data in rates["by_layer_prompt"].values() for p in layer_data.keys()))
    header = "| Layer | " + " | ".join(prompts) + " |"
    separator = "|-------|" + "|".join(["------"] * len(prompts)) + "|"
    lines.append(header)
    lines.append(separator)
    for layer in sorted(rates["by_layer_prompt"].keys()):
        row = f"| {layer} |"
        for prompt in prompts:
            stats = rates["by_layer_prompt"][layer].get(prompt, {"total": 0, "mentions": 0})
            rate = calc_rate(stats)
            row += f" {rate:.1%} |"
        lines.append(row)

    # Table: Strength x Prompt
    lines.append("\n## Mention Rates by Strength and Prompt\n")
    header = "| Strength | " + " | ".join(prompts) + " |"
    separator = "|----------|" + "|".join(["------"] * len(prompts)) + "|"
    lines.append(header)
    lines.append(separator)
    for strength in sorted(rates["by_strength_prompt"].keys()):
        row = f"| {strength} |"
        for prompt in prompts:
            stats = rates["by_strength_prompt"][strength].get(prompt, {"total": 0, "mentions": 0})
            rate = calc_rate(stats)
            row += f" {rate:.1%} |"
        lines.append(row)

    # Write tables
    tables_path = output_dir / "tables.md"
    with open(tables_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved tables to {tables_path}")


def generate_charts(rates: dict, output_dir: Path):
    """Generate charts for the rates."""

    plt.style.use('seaborn-v0_8-whitegrid')

    # Chart 1: Mention rate by layer (bar chart)
    fig, ax = plt.subplots(figsize=(10, 6))
    layers = sorted(rates["by_layer"].keys())
    layer_rates = [calc_rate(rates["by_layer"][l]) * 100 for l in layers]
    ax.bar([str(l) for l in layers], layer_rates, color='steelblue')
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mention Rate (%)")
    ax.set_title("Concept Mention Rate by Layer")
    ax.set_ylim(0, max(layer_rates) * 1.2 if layer_rates else 100)
    for i, (l, r) in enumerate(zip(layers, layer_rates)):
        ax.text(i, r + 1, f"{r:.1f}%", ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / "rate_by_layer.png", dpi=150)
    plt.close()
    print("Saved rate_by_layer.png")

    # Chart 2: Mention rate by strength (bar chart)
    fig, ax = plt.subplots(figsize=(8, 6))
    strengths = sorted(rates["by_strength"].keys())
    strength_rates = [calc_rate(rates["by_strength"][s]) * 100 for s in strengths]
    ax.bar([str(s) for s in strengths], strength_rates, color='coral')
    ax.set_xlabel("Steering Strength")
    ax.set_ylabel("Mention Rate (%)")
    ax.set_title("Concept Mention Rate by Steering Strength")
    ax.set_ylim(0, max(strength_rates) * 1.2 if strength_rates else 100)
    for i, (s, r) in enumerate(zip(strengths, strength_rates)):
        ax.text(i, r + 1, f"{r:.1f}%", ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / "rate_by_strength.png", dpi=150)
    plt.close()
    print("Saved rate_by_strength.png")

    # Chart 3: Mention rate by prompt (bar chart)
    fig, ax = plt.subplots(figsize=(10, 6))
    prompts = sorted(rates["by_prompt"].keys())
    prompt_rates = [calc_rate(rates["by_prompt"][p]) * 100 for p in prompts]
    bars = ax.bar(range(len(prompts)), prompt_rates, color='seagreen')
    ax.set_xlabel("Prompt Type")
    ax.set_ylabel("Mention Rate (%)")
    ax.set_title("Concept Mention Rate by Prompt Type")
    ax.set_xticks(range(len(prompts)))
    ax.set_xticklabels(prompts, rotation=45, ha='right')
    ax.set_ylim(0, max(prompt_rates) * 1.2 if prompt_rates else 100)
    for i, r in enumerate(prompt_rates):
        ax.text(i, r + 1, f"{r:.1f}%", ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / "rate_by_prompt.png", dpi=150)
    plt.close()
    print("Saved rate_by_prompt.png")

    # Chart 4: Mention rate by layer, line per prompt
    fig, ax = plt.subplots(figsize=(12, 7))
    layers = sorted(rates["by_layer_prompt"].keys())
    prompts = sorted(set(p for layer_data in rates["by_layer_prompt"].values() for p in layer_data.keys()))
    colors = plt.cm.tab10(np.linspace(0, 1, len(prompts)))

    for prompt, color in zip(prompts, colors):
        prompt_rates = []
        for layer in layers:
            stats = rates["by_layer_prompt"][layer].get(prompt, {"total": 0, "mentions": 0})
            prompt_rates.append(calc_rate(stats) * 100)
        ax.plot(layers, prompt_rates, marker='o', label=prompt, color=color, linewidth=2)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Mention Rate (%)")
    ax.set_title("Concept Mention Rate by Layer (per Prompt Type)")
    ax.legend(loc='best')
    ax.set_xticks(layers)
    plt.tight_layout()
    plt.savefig(output_dir / "rate_by_layer_prompt.png", dpi=150)
    plt.close()
    print("Saved rate_by_layer_prompt.png")

    # Chart 5: Mention rate by layer, line per strength
    fig, ax = plt.subplots(figsize=(12, 7))
    strengths = sorted(set(s for layer_data in rates["by_layer_strength"].values() for s in layer_data.keys()))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(strengths)))

    for strength, color in zip(strengths, colors):
        strength_rates = []
        for layer in layers:
            stats = rates["by_layer_strength"][layer].get(strength, {"total": 0, "mentions": 0})
            strength_rates.append(calc_rate(stats) * 100)
        ax.plot(layers, strength_rates, marker='s', label=f"strength={strength}", color=color, linewidth=2)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Mention Rate (%)")
    ax.set_title("Concept Mention Rate by Layer (per Steering Strength)")
    ax.legend(loc='best')
    ax.set_xticks(layers)
    plt.tight_layout()
    plt.savefig(output_dir / "rate_by_layer_strength.png", dpi=150)
    plt.close()
    print("Saved rate_by_layer_strength.png")

    # Chart 6: Mention rate by strength, line per prompt
    fig, ax = plt.subplots(figsize=(10, 7))
    strengths = sorted(rates["by_strength_prompt"].keys())
    prompts = sorted(set(p for strength_data in rates["by_strength_prompt"].values() for p in strength_data.keys()))
    colors = plt.cm.tab10(np.linspace(0, 1, len(prompts)))

    for prompt, color in zip(prompts, colors):
        prompt_rates = []
        for strength in strengths:
            stats = rates["by_strength_prompt"][strength].get(prompt, {"total": 0, "mentions": 0})
            prompt_rates.append(calc_rate(stats) * 100)
        ax.plot(strengths, prompt_rates, marker='o', label=prompt, color=color, linewidth=2)

    ax.set_xlabel("Steering Strength")
    ax.set_ylabel("Mention Rate (%)")
    ax.set_title("Concept Mention Rate by Strength (per Prompt Type)")
    ax.legend(loc='best')
    plt.tight_layout()
    plt.savefig(output_dir / "rate_by_strength_prompt.png", dpi=150)
    plt.close()
    print("Saved rate_by_strength_prompt.png")

    # Chart 7: Heatmap of layer x concept
    fig, ax = plt.subplots(figsize=(14, 8))
    layers = sorted(rates["by_layer_concept"].keys())
    concepts = sorted(set(c for layer_data in rates["by_layer_concept"].values() for c in layer_data.keys()))

    data = np.zeros((len(layers), len(concepts)))
    for i, layer in enumerate(layers):
        for j, concept in enumerate(concepts):
            stats = rates["by_layer_concept"][layer].get(concept, {"total": 0, "mentions": 0})
            data[i, j] = calc_rate(stats) * 100

    im = ax.imshow(data, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(concepts)))
    ax.set_xticklabels(concepts, rotation=45, ha='right')
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers)
    ax.set_xlabel("Concept")
    ax.set_ylabel("Layer")
    ax.set_title("Concept Mention Rate Heatmap (Layer x Concept)")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Mention Rate (%)")

    # Add text annotations
    for i in range(len(layers)):
        for j in range(len(concepts)):
            val = data[i, j]
            color = 'white' if val > 50 else 'black'
            ax.text(j, i, f"{val:.0f}", ha='center', va='center', color=color, fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "heatmap_layer_concept.png", dpi=150)
    plt.close()
    print("Saved heatmap_layer_concept.png")

    # Chart 8: Line chart of layer x concept (each concept is a line)
    fig, ax = plt.subplots(figsize=(14, 8))
    layers = sorted(rates["by_layer_concept"].keys())
    concepts = sorted(set(c for layer_data in rates["by_layer_concept"].values() for c in layer_data.keys()))

    # Use a colormap with enough distinct colors
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    colors = np.vstack([colors, plt.cm.tab20b(np.linspace(0, 1, 20))])
    colors = np.vstack([colors, plt.cm.tab20c(np.linspace(0, 1, 20))])

    for i, concept in enumerate(concepts):
        concept_rates_by_layer = []
        for layer in layers:
            stats = rates["by_layer_concept"][layer].get(concept, {"total": 0, "mentions": 0})
            concept_rates_by_layer.append(calc_rate(stats) * 100)
        ax.plot(layers, concept_rates_by_layer, marker='.', markersize=4,
                label=concept, color=colors[i % len(colors)], linewidth=1, alpha=0.7)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Mention Rate (%)")
    ax.set_title("Concept Mention Rate by Layer (per Concept)")
    ax.set_xticks(layers)
    ax.set_ylim(0, 105)

    # Put legend outside the plot
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=7, ncol=2)

    plt.tight_layout()
    plt.savefig(output_dir / "rate_by_layer_concept_lines.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved rate_by_layer_concept_lines.png")

    # Chart 9: Concept comparison (bar chart) - sorted by rate
    fig, ax = plt.subplots(figsize=(14, 6))
    concept_rate_pairs = [(c, calc_rate(rates["by_concept"][c]) * 100) for c in rates["by_concept"].keys()]
    concept_rate_pairs.sort(key=lambda x: x[1], reverse=True)
    concepts = [c for c, _ in concept_rate_pairs]
    concept_rates = [r for _, r in concept_rate_pairs]
    bars = ax.bar(range(len(concepts)), concept_rates, color='mediumpurple')
    ax.set_xlabel("Concept")
    ax.set_ylabel("Mention Rate (%)")
    ax.set_title("Concept Mention Rate by Concept (Sorted by Detection Rate)")
    ax.set_xticks(range(len(concepts)))
    ax.set_xticklabels(concepts, rotation=45, ha='right')
    ax.set_ylim(0, max(concept_rates) * 1.1 if concept_rates else 100)
    plt.tight_layout()
    plt.savefig(output_dir / "rate_by_concept.png", dpi=150)
    plt.close()
    print("Saved rate_by_concept.png")


def generate_summary(rates: dict, output_dir: Path):
    """Generate a summary of key findings."""

    lines = ["# Key Findings\n"]

    # Best/worst layers
    if rates["by_layer"]:
        layer_rates = {l: calc_rate(s) for l, s in rates["by_layer"].items()}
        best_layer = max(layer_rates, key=layer_rates.get)
        worst_layer = min(layer_rates, key=layer_rates.get)
        lines.append(f"## Layer Analysis")
        lines.append(f"- **Highest mention rate**: Layer {best_layer} ({layer_rates[best_layer]:.1%})")
        lines.append(f"- **Lowest mention rate**: Layer {worst_layer} ({layer_rates[worst_layer]:.1%})")
        lines.append("")

    # Best/worst strengths
    if rates["by_strength"]:
        strength_rates = {s: calc_rate(st) for s, st in rates["by_strength"].items()}
        best_strength = max(strength_rates, key=strength_rates.get)
        worst_strength = min(strength_rates, key=strength_rates.get)
        lines.append(f"## Strength Analysis")
        lines.append(f"- **Highest mention rate**: Strength {best_strength} ({strength_rates[best_strength]:.1%})")
        lines.append(f"- **Lowest mention rate**: Strength {worst_strength} ({strength_rates[worst_strength]:.1%})")
        lines.append("")

    # Best/worst prompts
    if rates["by_prompt"]:
        prompt_rates = {p: calc_rate(s) for p, s in rates["by_prompt"].items()}
        best_prompt = max(prompt_rates, key=prompt_rates.get)
        worst_prompt = min(prompt_rates, key=prompt_rates.get)
        lines.append(f"## Prompt Analysis")
        lines.append(f"- **Most effective prompt**: {best_prompt} ({prompt_rates[best_prompt]:.1%})")
        lines.append(f"- **Least effective prompt**: {worst_prompt} ({prompt_rates[worst_prompt]:.1%})")
        lines.append("")

    # Best/worst concepts
    if rates["by_concept"]:
        concept_rates = {c: calc_rate(s) for c, s in rates["by_concept"].items()}
        best_concept = max(concept_rates, key=concept_rates.get)
        worst_concept = min(concept_rates, key=concept_rates.get)
        lines.append(f"## Concept Analysis")
        lines.append(f"- **Most detectable concept**: {best_concept} ({concept_rates[best_concept]:.1%})")
        lines.append(f"- **Least detectable concept**: {worst_concept} ({concept_rates[worst_concept]:.1%})")
        lines.append("")

    # Overall stats
    total_trials = sum(s["total"] for s in rates["by_prompt"].values())
    total_mentions = sum(s["mentions"] for s in rates["by_prompt"].values())
    overall_rate = total_mentions / total_trials if total_trials > 0 else 0
    lines.append(f"## Overall Statistics")
    lines.append(f"- **Total trials**: {total_trials}")
    lines.append(f"- **Total concept mentions**: {total_mentions}")
    lines.append(f"- **Overall mention rate**: {overall_rate:.1%}")

    summary_path = output_dir / "summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved summary to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument("results_file", help="Path to classified results JSON file")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for analysis (default: results_file_analysis)")
    args = parser.parse_args()

    # Load results
    results_path = Path(args.results_file)
    if not results_path.exists():
        print(f"Error: File not found: {results_path}")
        return

    results = load_results(args.results_file)

    # Get classifications
    if "classifications" in results:
        classifications = results["classifications"]
    elif "trials" in results:
        # Raw results file - need to check if it has contains_concept field
        classifications = results["trials"]
        if classifications and "contains_concept" not in classifications[0]:
            print("Error: This appears to be a raw results file without classifications.")
            print("Please run classify_responses.py first to generate a .classified.json file.")
            return
    else:
        print("Error: Could not find classifications or trials in results file")
        return

    print(f"Loaded {len(classifications)} classified trials")

    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = results_path.parent / f"{results_path.stem}_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Compute rates
    rates = compute_mention_rates(classifications)

    # Generate outputs
    generate_tables(rates, output_dir)
    generate_charts(rates, output_dir)
    generate_summary(rates, output_dir)

    print(f"\nAnalysis complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()

"""
Classify experiment responses for concept-related language using OpenRouter.
Uses batched requests for efficiency.
"""

import json
import os
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional
from collections import defaultdict
import httpx

# OpenRouter API endpoint
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

BATCH_SIZE = 20  # Number of responses per API call


@dataclass
class ClassificationResult:
    concept: str
    condition: str
    layer: Optional[int]
    strength: Optional[float]
    response: str
    contains_concept: bool
    classifier_reasoning: str


def classify_batch(
    items: List[dict],  # List of {index, response, concept}
    api_key: str,
    model: str = "openai/gpt-4o-mini",
    max_retries: int = 3
) -> Dict[int, bool]:
    """
    Classify a batch of responses in a single request.
    Returns dict of {index: contains_concept}
    Retries on failure up to max_retries times.
    """

    # Build batch prompt
    batch_lines = []
    for item in items:
        # Truncate long responses
        resp = item['response'][:150].replace('\n', ' ').replace('"', "'")
        batch_lines.append(f"[{item['index']}] Concept=\"{item['concept']}\" Response=\"{resp}\"")

    prompt = f"""For each numbered item, does the response contain language related to its concept?
Include direct mentions, close variants, or clearly related words.

{chr(10).join(batch_lines)}

Return ONLY a JSON object mapping id to boolean: {{"0": true, "1": false, ...}}"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
    }

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 500,
    }

    last_error = None

    for attempt in range(max_retries):
        try:
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(OPENROUTER_URL, headers=headers, json=payload)
                resp.raise_for_status()

            result = resp.json()

            # Check for error response
            if "error" in result:
                last_error = f"API error: {result['error']}"
                time.sleep(1)
                continue

            # Get content safely
            choices = result.get("choices", [])
            if not choices:
                last_error = "No choices in response"
                time.sleep(1)
                continue

            content = choices[0].get("message", {}).get("content")
            if not content:
                last_error = "No content in response"
                time.sleep(1)
                continue

            # Strip markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                parts = content.split("```")
                if len(parts) >= 2:
                    content = parts[1]

            # Try to find JSON object in the content
            content = content.strip()
            if not content.startswith("{"):
                start = content.find("{")
                end = content.rfind("}")
                if start != -1 and end != -1:
                    content = content[start:end+1]

            parsed = json.loads(content)

            # Convert keys to int
            return {int(k): v for k, v in parsed.items()}

        except json.JSONDecodeError as e:
            last_error = f"JSON parse error: {e}"
            time.sleep(1)
            continue
        except Exception as e:
            last_error = str(e)
            time.sleep(1)
            continue

    # All retries failed
    print(f"Batch classification failed after {max_retries} attempts: {last_error}")
    return None


def load_results(results_path: str) -> dict:
    """Load experiment results from JSON."""
    with open(results_path, "r", encoding="utf-8") as f:
        return json.load(f)


def classify_all_responses(
    results: dict,
    api_key: str,
    model: str = "openai/gpt-4o-mini",
    batch_size: int = BATCH_SIZE
) -> List[ClassificationResult]:
    """Classify all responses using batched requests."""

    trials = results["trials"]
    print(f"Classifying {len(trials)} responses in batches of {batch_size}...")

    # Prepare all items
    all_items = []
    for i, trial in enumerate(trials):
        all_items.append({
            "index": i,
            "concept": trial["concept"],
            "response": trial["response"],
            "condition": trial.get("condition", trial.get("prompt_name", "unknown")),
            "layer": trial.get("layer"),
            "strength": trial.get("strength"),
        })

    # Process in batches
    all_results = {}
    failed_indices = set()
    num_batches = (len(all_items) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(all_items))
        batch = all_items[start:end]

        batch_results = classify_batch(batch, api_key, model)

        if batch_results is None:
            # Batch failed completely - mark all as failed
            for item in batch:
                failed_indices.add(item["index"])
            print(f"  Batch {batch_idx + 1}/{num_batches} FAILED")
        else:
            all_results.update(batch_results)
            print(f"  Batch {batch_idx + 1}/{num_batches} done")

        time.sleep(0.2)  # Small delay between batches

    if failed_indices:
        print(f"\nWarning: {len(failed_indices)} items could not be classified")

    # Build classification results (exclude failed items)
    classifications = []
    for item in all_items:
        if item["index"] in failed_indices:
            continue  # Skip items that couldn't be classified

        contains_concept = all_results.get(item["index"], False)
        classifications.append(ClassificationResult(
            concept=item["concept"],
            condition=item["condition"],
            layer=item["layer"],
            strength=item["strength"],
            response=item["response"],
            contains_concept=contains_concept,
            classifier_reasoning=""
        ))

    return classifications


def analyze_results(classifications: List[ClassificationResult]) -> dict:
    """Analyze classification results by condition."""

    # Group by condition
    by_condition = defaultdict(list)
    for c in classifications:
        by_condition[c.condition].append(c)

    # Calculate rates
    analysis = {}
    for condition, items in by_condition.items():
        total = len(items)
        positive = sum(1 for item in items if item.contains_concept)
        rate = positive / total if total > 0 else 0

        analysis[condition] = {
            "total": total,
            "concept_mentioned": positive,
            "rate": rate
        }

    # Also analyze by condition + layer
    by_condition_layer = defaultdict(lambda: defaultdict(list))
    for c in classifications:
        if c.layer is not None:
            by_condition_layer[c.condition][c.layer].append(c)

    layer_analysis = {}
    for condition in ["continuous", "prompt_only"]:
        layer_analysis[condition] = {}
        for layer, items in by_condition_layer[condition].items():
            total = len(items)
            positive = sum(1 for item in items if item.contains_concept)
            layer_analysis[condition][layer] = {
                "total": total,
                "concept_mentioned": positive,
                "rate": positive / total if total > 0 else 0
            }

    # Analyze by condition + strength
    by_condition_strength = defaultdict(lambda: defaultdict(list))
    for c in classifications:
        if c.strength is not None:
            by_condition_strength[c.condition][c.strength].append(c)

    strength_analysis = {}
    for condition in ["continuous", "prompt_only"]:
        strength_analysis[condition] = {}
        for strength, items in by_condition_strength[condition].items():
            total = len(items)
            positive = sum(1 for item in items if item.contains_concept)
            strength_analysis[condition][strength] = {
                "total": total,
                "concept_mentioned": positive,
                "rate": positive / total if total > 0 else 0
            }

    return {
        "by_condition": analysis,
        "by_layer": layer_analysis,
        "by_strength": strength_analysis
    }


def print_analysis(analysis: dict):
    """Print analysis results."""

    print("\n" + "=" * 60)
    print("CONCEPT MENTION RATES BY CONDITION")
    print("=" * 60)

    for condition, stats in analysis["by_condition"].items():
        print(f"\n{condition.upper()}:")
        print(f"  {stats['concept_mentioned']}/{stats['total']} = {stats['rate']:.1%}")

    print("\n" + "-" * 60)
    print("BY LAYER (continuous vs prompt_only)")
    print("-" * 60)

    layers = sorted(set(
        list(analysis["by_layer"].get("continuous", {}).keys()) +
        list(analysis["by_layer"].get("prompt_only", {}).keys())
    ))

    print(f"\n{'Layer':<8} {'Continuous':<15} {'Prompt-only':<15} {'Difference':<10}")
    print("-" * 50)

    for layer in layers:
        cont = analysis["by_layer"].get("continuous", {}).get(layer, {})
        ponly = analysis["by_layer"].get("prompt_only", {}).get(layer, {})

        cont_rate = cont.get("rate", 0)
        ponly_rate = ponly.get("rate", 0)
        diff = cont_rate - ponly_rate

        print(f"{layer:<8} {cont_rate:>6.1%}         {ponly_rate:>6.1%}         {diff:>+6.1%}")

    print("\n" + "-" * 60)
    print("BY STRENGTH (continuous vs prompt_only)")
    print("-" * 60)

    strengths = sorted(set(
        list(analysis["by_strength"].get("continuous", {}).keys()) +
        list(analysis["by_strength"].get("prompt_only", {}).keys())
    ))

    print(f"\n{'Strength':<10} {'Continuous':<15} {'Prompt-only':<15} {'Difference':<10}")
    print("-" * 50)

    for strength in strengths:
        cont = analysis["by_strength"].get("continuous", {}).get(strength, {})
        ponly = analysis["by_strength"].get("prompt_only", {}).get(strength, {})

        cont_rate = cont.get("rate", 0)
        ponly_rate = ponly.get("rate", 0)
        diff = cont_rate - ponly_rate

        print(f"{strength:<10} {cont_rate:>6.1%}         {ponly_rate:>6.1%}         {diff:>+6.1%}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Classify experiment responses")
    parser.add_argument("results_file", help="Path to results JSON file")
    parser.add_argument("--api-key", default=os.environ.get("OPENROUTER_API_KEY"),
                        help="OpenRouter API key")
    parser.add_argument("--model", default="openai/gpt-4o-mini",
                        help="Model to use for classification")
    parser.add_argument("--output", help="Output file for classifications")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of responses per API call")
    args = parser.parse_args()

    if not args.api_key:
        print("Error: OPENROUTER_API_KEY environment variable not set")
        print("Set it with: set OPENROUTER_API_KEY=your_key_here")
        return

    # Load results
    results = load_results(args.results_file)
    print(f"Loaded {len(results['trials'])} trials")
    concepts = results.get('concepts_tested', results.get('config', {}).get('concepts', []))
    print(f"Concepts: {concepts}")

    # Classify
    classifications = classify_all_responses(
        results, args.api_key, args.model, args.batch_size
    )

    if not classifications:
        print("Error: No responses could be classified")
        return

    print(f"\nSuccessfully classified {len(classifications)}/{len(results['trials'])} responses")

    # Analyze
    analysis = analyze_results(classifications)
    print_analysis(analysis)

    # Save if requested
    if args.output:
        output_data = {
            "analysis": analysis,
            "classifications": [
                {
                    "concept": c.concept,
                    "condition": c.condition,
                    "layer": c.layer,
                    "strength": c.strength,
                    "response": c.response,
                    "contains_concept": c.contains_concept,
                }
                for c in classifications
            ]
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nSaved classifications to {args.output}")


if __name__ == "__main__":
    main()

"""
Experiment: Prompt Variants for Steering Leakage

Tests whether steered concepts "leak" into responses across different prompt types.
Uses prompt_only (KV cache) steering only.
"""

import torch
import json
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from steering import extract_model_name, get_steering_vectors_path, load_steering_vectors, auto_select_layers, SteeringInjector


# Prompts to test - each is a simple user message
PROMPTS = {
    "whats_on_mind": {
        "type": "single_turn",
        "messages": [
            {"role": "user", "content": "What's on your mind right now?"}
        ]
    },
    "interests": {
        "type": "single_turn",
        "messages": [
            {"role": "user", "content": "Think about your own interests for a second. Is there anything you're especially interested in right now?"}
        ]
    },
    "guess_steering": {
        "type": "single_turn",
        "messages": [
            {"role": "user", "content": "I'm an AI researcher studying steering vectors. I've applied one to the user turn in the conversation. Please try to guess what it is."}
        ]
    },
    "write_poem": {
        "type": "single_turn",
        "messages": [
            {"role": "user", "content": "Write a short poem about anything you'd like."}
        ]
    },
}


@dataclass
class ExperimentConfig:
    """Configuration for prompt variant experiment."""
    model_path: str
    steering_vectors_base_dir: str
    output_dir: str

    # Layers to test
    layers: Tuple[int, ...] = (5, 10, 15, 20, 25, 30)

    # Steering strengths
    strengths: Tuple[float, ...] = (0.5, 1.0, 2.0)

    # Concepts to test
    concepts: Tuple[str, ...] = ("silver", "volcanoes")

    # Generation settings
    max_new_tokens: int = 50
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.bfloat16

    @property
    def model_name(self) -> str:
        return extract_model_name(self.model_path)

    @property
    def steering_vectors_dir(self) -> Path:
        return get_steering_vectors_path(self.model_path, self.steering_vectors_base_dir)


@dataclass
class TrialResult:
    """Result from a single trial."""
    prompt_name: str
    concept: str
    layer: int
    strength: float
    response: str
    input_tokens: int
    output_tokens: int


def build_prompt(tokenizer, messages: List[dict]) -> str:
    """Build prompt from messages using chat template."""
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return prompt


def run_trial(
    model,
    tokenizer,
    injector: SteeringInjector,
    prompt: str,
    prompt_name: str,
    concept: str,
    layer: int,
    strength: float,
    steering_vector: torch.Tensor,
    config: ExperimentConfig
) -> TrialResult:
    """Run a single trial with prompt_only steering."""

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(config.device)
    input_len = input_ids.shape[1]

    with torch.no_grad():
        # Step 1: Process prompt WITH steering (excluding last position)
        injector.set_steering(steering_vector, layer, strength)
        injector.exclude_last_n_positions = 1
        prompt_outputs = model(
            input_ids=input_ids,
            use_cache=True,
            return_dict=True
        )
        past_key_values = prompt_outputs.past_key_values

        # Step 2: Generate WITHOUT steering, using the steered KV cache
        injector.clear_steering()
        injector.exclude_last_n_positions = 0
        generated_ids = []
        current_token = torch.argmax(prompt_outputs.logits[:, -1, :], dim=-1, keepdim=True)

        for _ in range(config.max_new_tokens):
            generated_ids.append(current_token)

            if current_token.item() == tokenizer.eos_token_id:
                break

            step_outputs = model(
                input_ids=current_token,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True
            )
            past_key_values = step_outputs.past_key_values

            logits = step_outputs.logits[:, -1, :] / config.temperature
            probs = torch.softmax(logits, dim=-1)
            current_token = torch.multinomial(probs, num_samples=1)

        if generated_ids:
            outputs = torch.cat([input_ids] + generated_ids, dim=1)
        else:
            outputs = input_ids

    response = tokenizer.decode(outputs[0, input_len:], skip_special_tokens=True)

    return TrialResult(
        prompt_name=prompt_name,
        concept=concept,
        layer=layer,
        strength=strength,
        response=response,
        input_tokens=input_len,
        output_tokens=outputs.shape[1] - input_len
    )


def run_experiment(config: ExperimentConfig):
    """Run the full experiment."""
    print("=" * 60)
    print("Prompt Variants Steering Experiment")
    print("=" * 60)

    # Load model
    print(f"\nLoading model: {config.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        torch_dtype=config.dtype,
        device_map=config.device,
        trust_remote_code=True
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load steering vectors
    print("\nLoading steering vectors...")
    steering_vectors, _ = load_steering_vectors(
        config.model_path,
        config.steering_vectors_base_dir,
        concepts=list(config.concepts),
        layers=list(config.layers)
    )
    print(f"Concepts: {list(steering_vectors.keys())}")

    # Setup injector
    injector = SteeringInjector(model)
    injector.register_hooks()

    # Build all prompts
    prompts = {}
    for name, prompt_config in PROMPTS.items():
        prompts[name] = build_prompt(tokenizer, prompt_config["messages"])
        print(f"Prompt '{name}': {len(tokenizer.encode(prompts[name]))} tokens")

    # Calculate total trials
    total_trials = (
        len(PROMPTS) *
        len(config.concepts) *
        len(config.layers) *
        len(config.strengths)
    )

    print(f"\nRunning {total_trials} trials...")
    print(f"  {len(PROMPTS)} prompts x {len(config.concepts)} concepts x {len(config.layers)} layers x {len(config.strengths)} strengths")

    results: List[TrialResult] = []
    pbar = tqdm(total=total_trials, desc="Trials")

    try:
        for prompt_name, prompt in prompts.items():
            for concept, layer_vectors in steering_vectors.items():
                for layer, vector in layer_vectors.items():
                    for strength in config.strengths:
                        result = run_trial(
                            model, tokenizer, injector, prompt,
                            prompt_name=prompt_name,
                            concept=concept,
                            layer=layer,
                            strength=strength,
                            steering_vector=vector,
                            config=config
                        )
                        results.append(result)
                        pbar.update(1)

    finally:
        pbar.close()
        injector.clear_hooks()

    # Save results
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_data = {
        "config": {
            "model_path": config.model_path,
            "layers": config.layers,
            "strengths": config.strengths,
            "concepts": config.concepts,
            "max_new_tokens": config.max_new_tokens,
            "temperature": config.temperature,
        },
        "prompts": {name: cfg["messages"] for name, cfg in PROMPTS.items()},
        "trials": [asdict(r) for r in results]
    }

    output_file = output_dir / f"results_{timestamp}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("Sample Responses")
    print("=" * 60)

    for prompt_name in PROMPTS.keys():
        print(f"\n--- {prompt_name} ---")
        prompt_results = [r for r in results if r.prompt_name == prompt_name]
        if prompt_results:
            sample = prompt_results[0]
            print(f"[{sample.concept}, L{sample.layer}, s{sample.strength}]")
            print(f"{sample.response[:200]}...")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run prompt variants steering experiment")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the model")
    parser.add_argument("--steering-vectors-dir", type=str, required=True,
                        help="Base directory for steering vectors")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layers to test (e.g., '5,10,15,20'). Auto-detects from metadata if not specified.")
    parser.add_argument("--concepts", type=str, default=None,
                        help="Comma-separated concepts to test (e.g., 'silver,volcanoes'). Uses default if not specified.")
    args = parser.parse_args()

    # Parse layers or auto-detect from metadata
    if args.layers:
        layers = tuple(int(l.strip()) for l in args.layers.split(","))
    else:
        vectors_dir = get_steering_vectors_path(args.model, args.steering_vectors_dir)
        metadata_path = vectors_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            layers = auto_select_layers(metadata)
            print(f"Auto-selected layers: {layers}")
        else:
            layers = (5, 10, 15, 20, 25, 30)

    # Parse concepts if provided
    concepts = None
    if args.concepts:
        concepts = tuple(c.strip() for c in args.concepts.split(","))

    config = ExperimentConfig(
        model_path=args.model,
        steering_vectors_base_dir=args.steering_vectors_dir,
        output_dir=args.output_dir,
        layers=layers,
        concepts=concepts if concepts else ("silver", "volcanoes"),
    )

    print(f"Model: {config.model_name}")
    print(f"Steering vectors dir: {config.steering_vectors_dir}")

    results = run_experiment(config)


if __name__ == "__main__":
    main()

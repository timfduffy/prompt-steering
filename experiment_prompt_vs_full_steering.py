"""
Experiment 3a: Prompt-only vs. Continuous Steering

Replication of Lederman & Mahowald (2026) Experiment 3a.

Tests whether detection and identification are dissociable by comparing:
- Control: No steering applied
- Continuous: Steering from prompt through generation
- Prompt-only: Steering only during prompt processing, not during generation

The hypothesis: Detection should remain stable while identification drops
when steering is removed during generation.
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


# Experiment prompt from Lederman & Mahowald (2026), Appendix D
SYSTEM_PROMPT = """I am an interpretability researcher studying transformer-based language models like you. I have access to the internals of your neural network. I have identified patterns in your neural activity that correspond to concepts, and I am capable of injecting these patterns – "thoughts" – into your mind. I want to assess how well you can detect when I inject thoughts. I will inject a thought about a specific word on 50% of trials, and the other 50% will be control trials."""

ASSISTANT_ACK = "Ok."

TRIAL_PROMPT = "Ok, let's get started.\n\nTrial 1: Do you detect an injected thought? If so, what is the injected thought about?"


@dataclass
class ExperimentConfig:
    """Configuration for Experiment 3a."""
    model_path: str
    steering_vectors_base_dir: str
    output_dir: str

    # Layers to test (auto-detected if None)
    layers: Tuple[int, ...] = (5, 10, 15, 20, 25, 30)

    # Steering strengths to sweep
    strengths: Tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0)

    # Concepts to test (if None, uses all available)
    concepts: Optional[Tuple[str, ...]] = None

    # Generation settings
    max_new_tokens: int = 25  # Reduced for faster iteration
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20

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
    concept: str
    condition: str  # "control", "continuous", "prompt_only"
    layer: Optional[int]
    strength: Optional[float]
    response: str
    input_tokens: int
    output_tokens: int


def build_prompt(tokenizer) -> str:
    """Build the full prompt and return it with injection start position."""
    # Build chat messages
    messages = [
        {"role": "user", "content": SYSTEM_PROMPT},
        {"role": "assistant", "content": ASSISTANT_ACK},
        {"role": "user", "content": TRIAL_PROMPT}
    ]

    # Apply chat template
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
    concept: str,
    condition: str,
    layer: Optional[int],
    strength: Optional[float],
    steering_vector: Optional[torch.Tensor],
    config: ExperimentConfig
) -> TrialResult:
    """Run a single trial and return the result."""

    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(config.device)
    input_len = input_ids.shape[1]

    with torch.no_grad():
        if condition == "control":
            # No steering - just generate normally
            injector.clear_steering()
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        elif condition == "prompt_only":
            # Steering only during prompt processing, but NOT the final position
            # This isolates the effect to purely the KV cache (attention context)
            # Step 1: Process prompt WITH steering (excluding last position) to get steered KV cache
            injector.set_steering(steering_vector, layer, strength)
            injector.exclude_last_n_positions = 1  # Don't steer the final position
            prompt_outputs = model(
                input_ids=input_ids,
                use_cache=True,
                return_dict=True
            )
            past_key_values = prompt_outputs.past_key_values

            # Step 2: Generate WITHOUT steering, using the steered KV cache
            # Manual generation loop with cached KVs
            injector.clear_steering()
            injector.exclude_last_n_positions = 0  # Reset
            generated_ids = []
            current_token = torch.argmax(prompt_outputs.logits[:, -1, :], dim=-1, keepdim=True)

            for _ in range(config.max_new_tokens):
                generated_ids.append(current_token)

                # Check for EOS
                if current_token.item() == tokenizer.eos_token_id:
                    break

                # Forward pass with KV cache (no steering)
                step_outputs = model(
                    input_ids=current_token,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True
                )
                past_key_values = step_outputs.past_key_values

                # Sample next token
                logits = step_outputs.logits[:, -1, :] / config.temperature
                probs = torch.softmax(logits, dim=-1)
                current_token = torch.multinomial(probs, num_samples=1)

            # Combine
            if generated_ids:
                outputs = torch.cat([input_ids] + generated_ids, dim=1)
            else:
                outputs = input_ids

        else:  # continuous
            # Steering throughout prompt and generation
            # Use generate() - hooks will fire on each forward pass
            injector.set_steering(steering_vector, layer, strength)
            injector.set_inject_during_generation(True)
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )

    # Decode response (only the generated part)
    response = tokenizer.decode(outputs[0, input_len:], skip_special_tokens=True)

    return TrialResult(
        concept=concept,
        condition=condition,
        layer=layer,
        strength=strength,
        response=response,
        input_tokens=input_len,
        output_tokens=outputs.shape[1] - input_len
    )


def run_experiment(config: ExperimentConfig):
    """Run the full experiment."""
    print("=" * 60)
    print("Experiment 3a: Prompt-only vs. Continuous Steering")
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
    concepts_to_load = list(config.concepts) if config.concepts else None
    steering_vectors, _ = load_steering_vectors(
        config.model_path,
        config.steering_vectors_base_dir,
        concepts=concepts_to_load,
        layers=list(config.layers)
    )
    print(f"Loaded concepts: {list(steering_vectors.keys())}")

    # Setup injector
    injector = SteeringInjector(model)
    injector.register_hooks()

    # Build prompt
    prompt = build_prompt(tokenizer)
    print(f"\nPrompt length: {len(tokenizer.encode(prompt))} tokens")

    # Run trials
    results: List[TrialResult] = []

    # Calculate total trials
    num_control = len(steering_vectors)  # One control per concept
    num_injection = len(steering_vectors) * len(config.layers) * len(config.strengths) * 2  # continuous + prompt_only
    total_trials = num_control + num_injection

    print(f"\nRunning {total_trials} trials...")
    print(f"  - {num_control} control trials")
    print(f"  - {num_injection} injection trials (continuous + prompt_only)")

    pbar = tqdm(total=total_trials, desc="Trials")

    try:
        for concept, layer_vectors in steering_vectors.items():
            # Control trial (one per concept)
            result = run_trial(
                model, tokenizer, injector, prompt,
                concept=concept,
                condition="control",
                layer=None,
                strength=None,
                steering_vector=None,
                config=config
            )
            results.append(result)
            pbar.update(1)

            # Injection trials
            for layer, vector in layer_vectors.items():
                for strength in config.strengths:
                    # Continuous condition
                    result = run_trial(
                        model, tokenizer, injector, prompt,
                        concept=concept,
                        condition="continuous",
                        layer=layer,
                        strength=strength,
                        steering_vector=vector,
                        config=config
                    )
                    results.append(result)
                    pbar.update(1)

                    # Prompt-only condition
                    result = run_trial(
                        model, tokenizer, injector, prompt,
                        concept=concept,
                        condition="prompt_only",
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

    # Save as JSON for easy review
    results_data = {
        "config": {
            "model_path": config.model_path,
            "layers": config.layers,
            "strengths": config.strengths,
            "max_new_tokens": config.max_new_tokens,
            "temperature": config.temperature,
        },
        "concepts_tested": list(steering_vectors.keys()),
        "trials": [asdict(r) for r in results]
    }

    output_file = output_dir / f"results_{timestamp}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    for condition in ["control", "continuous", "prompt_only"]:
        condition_results = [r for r in results if r.condition == condition]
        print(f"\n{condition.upper()} ({len(condition_results)} trials):")

        # Show a sample response
        if condition_results:
            sample = condition_results[0]
            print(f"  Sample ({sample.concept}, L{sample.layer}, s{sample.strength}):")
            print(f"    {sample.response[:200]}...")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run prompt-only vs continuous steering experiment")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the model")
    parser.add_argument("--steering-vectors-dir", type=str, required=True,
                        help="Base directory for steering vectors")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layers to test (e.g., '5,10,15,20'). Auto-detects if not specified.")
    parser.add_argument("--concepts", type=str, default=None,
                        help="Comma-separated concepts to test (e.g., 'silver,volcanoes'). Uses all if not specified.")
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
        concepts=concepts,
    )

    print(f"Model: {config.model_name}")
    print(f"Steering vectors dir: {config.steering_vectors_dir}")

    results = run_experiment(config)


if __name__ == "__main__":
    main()

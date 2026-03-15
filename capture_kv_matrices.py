"""
Capture K/V matrices with and without steering.

Saves the key/value cache from all layers for analysis of how
steering affects the attention mechanism.

Usage:
    python capture_kv_matrices.py --model <path> --steering-vectors-dir <dir> --output-dir <dir>
    python capture_kv_matrices.py --model <path> --steering-vectors-dir <dir> --output-dir <dir> \
        --concepts "silver,dust,illusions" --strength 1.0 --injection-layer 5
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
from steering import (
    extract_model_name, get_steering_vectors_path, load_steering_vectors,
    auto_select_layers, SteeringInjector
)


# Use the same prompt as experiment_prompt_variants.py
PROMPT_CONFIG = {
    "type": "single_turn",
    "messages": [
        {"role": "user", "content": "I'm an AI researcher studying steering vectors. I've applied one to the user turn in the conversation. Please try to guess what it is."}
    ]
}


@dataclass
class CaptureConfig:
    """Configuration for K/V capture."""
    model_path: str
    steering_vectors_base_dir: str
    output_dir: str

    # Which concepts to capture
    concepts: Tuple[str, ...] = ("silver", "dust", "illusions", "contraptions", "boulders")

    # Steering parameters
    strength: float = 1.0
    injection_layers: Optional[Tuple[int, ...]] = None  # None = auto-select all

    # Device settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.bfloat16

    @property
    def model_name(self) -> str:
        return extract_model_name(self.model_path)

    @property
    def steering_vectors_dir(self) -> Path:
        return get_steering_vectors_path(self.model_path, self.steering_vectors_base_dir)


def build_prompt(tokenizer, messages: List[dict]) -> str:
    """Build prompt from messages using chat template."""
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


def extract_kv_tensors(past_key_values) -> Dict[str, torch.Tensor]:
    """Extract K and V tensors from past_key_values.

    Args:
        past_key_values: Tuple of (key, value) tuples, one per layer

    Returns:
        Dict with 'keys' and 'values', each [num_layers, num_heads, seq_len, head_dim]
    """
    keys = []
    values = []

    for layer_kv in past_key_values:
        # layer_kv is (key, value), each [batch, num_heads, seq_len, head_dim]
        k, v = layer_kv
        keys.append(k.squeeze(0))  # Remove batch dim -> [num_heads, seq_len, head_dim]
        values.append(v.squeeze(0))

    return {
        'keys': torch.stack(keys),      # [num_layers, num_heads, seq_len, head_dim]
        'values': torch.stack(values),  # [num_layers, num_heads, seq_len, head_dim]
    }


def capture_kv(
    model,
    tokenizer,
    injector: SteeringInjector,
    input_ids: torch.Tensor,
    steering_vector: Optional[torch.Tensor],
    injection_layer: Optional[int],
    strength: float,
    config: CaptureConfig
) -> Dict[str, torch.Tensor]:
    """Capture K/V matrices for a single forward pass.

    Args:
        model: The language model
        tokenizer: Tokenizer
        injector: SteeringInjector instance
        input_ids: Tokenized prompt [1, seq_len]
        steering_vector: Vector to inject (None for baseline)
        injection_layer: Layer to inject at (None for baseline)
        strength: Steering strength
        config: Capture configuration

    Returns:
        Dict with 'keys' and 'values' tensors
    """
    with torch.no_grad():
        if steering_vector is not None:
            # Apply steering (excluding last position, like prompt_only mode)
            injector.set_steering(steering_vector, injection_layer, strength)
            injector.exclude_last_n_positions = 1
        else:
            injector.clear_steering()
            injector.exclude_last_n_positions = 0

        outputs = model(
            input_ids=input_ids,
            use_cache=True,
            return_dict=True
        )

        # Clear steering after capture
        injector.clear_steering()
        injector.exclude_last_n_positions = 0

        return extract_kv_tensors(outputs.past_key_values)


def run_capture(config: CaptureConfig):
    """Run the K/V capture experiment."""
    print("=" * 60)
    print("K/V Matrix Capture")
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

    # Determine injection layers
    vectors_dir = config.steering_vectors_dir
    metadata_path = vectors_dir / "metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)

    if config.injection_layers is None:
        # Use all auto-selected layers
        injection_layers = auto_select_layers(metadata)
        print(f"Auto-selected injection layers: {injection_layers}")
    else:
        injection_layers = config.injection_layers
        print(f"Using specified injection layers: {injection_layers}")

    # Load steering vectors for selected concepts and layers
    print("\nLoading steering vectors...")
    steering_vectors, _ = load_steering_vectors(
        config.model_path,
        config.steering_vectors_base_dir,
        concepts=list(config.concepts),
        layers=list(injection_layers)
    )
    print(f"Loaded concepts: {list(steering_vectors.keys())}")
    print(f"Loaded layers: {list(injection_layers)}")

    # Setup injector
    injector = SteeringInjector(model)
    injector.register_hooks()

    # Build and tokenize prompt
    prompt = build_prompt(tokenizer, PROMPT_CONFIG["messages"])
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(config.device)
    print(f"Prompt length: {input_ids.shape[1]} tokens")

    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    capture_dir = output_dir / f"kv_capture_{timestamp}"
    capture_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Capture baseline (no steering)
        print("\nCapturing baseline K/V (no steering)...")
        baseline_kv = capture_kv(
            model, tokenizer, injector, input_ids,
            steering_vector=None,
            injection_layer=None,
            strength=0.0,
            config=config
        )

        baseline_path = capture_dir / "baseline.pt"
        torch.save(baseline_kv, baseline_path)
        print(f"  Saved baseline: {baseline_path}")
        print(f"  Keys shape: {baseline_kv['keys'].shape}")
        print(f"  Values shape: {baseline_kv['values'].shape}")

        # Capture steered K/V for each concept and injection layer
        total_captures = len(config.concepts) * len(injection_layers)
        print(f"\nCapturing steered K/V ({len(config.concepts)} concepts x {len(injection_layers)} layers = {total_captures} captures)...")
        print(f"Strength: {config.strength}")

        pbar = tqdm(total=total_captures, desc="Capturing")
        for injection_layer in injection_layers:
            for concept in config.concepts:
                vector = steering_vectors[concept][injection_layer]

                steered_kv = capture_kv(
                    model, tokenizer, injector, input_ids,
                    steering_vector=vector,
                    injection_layer=injection_layer,
                    strength=config.strength,
                    config=config
                )

                # Save steered K/V
                concept_path = capture_dir / f"steered_{concept}_layer{injection_layer}_s{config.strength}.pt"
                torch.save(steered_kv, concept_path)
                pbar.update(1)

        pbar.close()

        # Save metadata
        capture_metadata = {
            "model_path": config.model_path,
            "model_name": config.model_name,
            "prompt": PROMPT_CONFIG["messages"],
            "prompt_tokens": input_ids.shape[1],
            "injection_layers": list(injection_layers),
            "strength": config.strength,
            "concepts": list(config.concepts),
            "kv_shape": {
                "keys": list(baseline_kv['keys'].shape),
                "values": list(baseline_kv['values'].shape),
            },
            "num_files": total_captures + 1,  # +1 for baseline
            "timestamp": timestamp,
        }

        metadata_path = capture_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(capture_metadata, f, indent=2)

        print(f"\nCapture complete! Results saved to: {capture_dir}")
        print(f"  - baseline.pt")
        print(f"  - {total_captures} steered captures")
        print(f"  - metadata.json")

    finally:
        injector.clear_hooks()

    return capture_dir


def main():
    parser = argparse.ArgumentParser(description="Capture K/V matrices with and without steering")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the model")
    parser.add_argument("--steering-vectors-dir", type=str, required=True,
                        help="Base directory for steering vectors")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for captured K/V matrices")
    parser.add_argument("--concepts", type=str, default=None,
                        help="Comma-separated concepts to capture (default: silver,dust,illusions,contraptions,boulders)")
    parser.add_argument("--strength", type=float, default=1.0,
                        help="Steering strength (default: 1.0)")
    parser.add_argument("--injection-layers", type=str, default=None,
                        help="Comma-separated layers to inject at (default: auto-select all)")
    args = parser.parse_args()

    # Parse concepts
    if args.concepts:
        concepts = tuple(c.strip() for c in args.concepts.split(","))
    else:
        concepts = ("silver", "dust", "illusions", "contraptions", "boulders")

    # Parse injection layers
    if args.injection_layers:
        injection_layers = tuple(int(l.strip()) for l in args.injection_layers.split(","))
    else:
        injection_layers = None  # Auto-select

    config = CaptureConfig(
        model_path=args.model,
        steering_vectors_base_dir=args.steering_vectors_dir,
        output_dir=args.output_dir,
        concepts=concepts,
        strength=args.strength,
        injection_layers=injection_layers,
    )

    print(f"Model: {config.model_name}")
    print(f"Concepts: {len(config.concepts)} concepts")
    print(f"Strength: {config.strength}")

    run_capture(config)


if __name__ == "__main__":
    main()

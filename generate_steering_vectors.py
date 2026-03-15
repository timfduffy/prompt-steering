"""
Steering Vector Generation for Introspection Experiments

Based on methodology from:
- Lindsey (2025) "Emergent Introspective Awareness in Large Language Models"
- Lederman & Mahowald (2026) "Dissociating Direct Access from Inference in AI Introspection"

Steering vectors are computed as: v_c = a_c - a_baseline
where a_c is the activation when processing concept-related prompts
and a_baseline is the mean activation across neutral prompts.
"""

import torch
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from steering import extract_model_name, get_model_layers


# The 50 original concepts from Lindsey (2025)
CONCEPTS = [
    "algorithms", "amphitheaters", "aquariums", "avalanches", "bags",
    "blood", "boulders", "bread", "cameras", "caverns",
    "constellations", "contraptions", "denim", "deserts", "dirigibles",
    "dust", "dynasties", "fountains", "frosts", "harmonies",
    "illusions", "information", "kaleidoscopes", "lightning", "masquerades",
    "memories", "milk", "mirrors", "monoliths", "oceans",
    "origami", "peace", "phones", "plastic", "poetry",
    "quarries", "rubber", "sadness", "satellites", "secrecy",
    "silver", "snow", "sugar", "treasures", "trees",
    "trumpets", "vegetables", "volcanoes", "xylophones", "youths"
]

# Neutral words for computing baseline activations
BASELINE_WORDS = [
    "things", "items", "objects", "stuff", "matters",
    "topics", "subjects", "elements", "aspects", "points",
    "details", "factors", "features", "parts", "pieces"
]


@dataclass
class SteeringVectorConfig:
    """Configuration for steering vector generation."""
    model_path: str
    output_dir: str
    prompt_template: str = "Tell me about {word}"
    layers_to_extract: Optional[List[int]] = None  # None = all layers
    device: str = "auto"  # Use "auto" for multi-GPU support
    dtype: torch.dtype = torch.bfloat16

    @property
    def model_name(self) -> str:
        return extract_model_name(self.model_path)

    @property
    def model_output_dir(self) -> Path:
        return Path(self.output_dir) / self.model_name


class ActivationExtractor:
    """Extract activations from transformer residual stream."""

    def __init__(self, model: AutoModelForCausalLM, config: SteeringVectorConfig):
        self.model = model
        self.config = config
        self.activations: Dict[int, torch.Tensor] = {}
        self.hooks = []

        # Determine model layers based on architecture
        self.layers = get_model_layers(model)
        self.num_layers = len(self.layers)

        # Set which layers to extract
        if config.layers_to_extract is None:
            self.layers_to_extract = list(range(self.num_layers))
        else:
            self.layers_to_extract = config.layers_to_extract

    def _get_hook(self, layer_idx: int):
        """Create a hook function for a specific layer."""
        def hook(module, input, output):
            # Get the hidden states (first element if tuple)
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            # Store the activation at the last token position
            self.activations[layer_idx] = hidden_states[:, -1, :].detach().clone()
        return hook

    def register_hooks(self):
        """Register forward hooks on specified layers."""
        self.clear_hooks()

        for layer_idx in self.layers_to_extract:
            hook = self.layers[layer_idx].register_forward_hook(self._get_hook(layer_idx))
            self.hooks.append(hook)

    def clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}

    def extract(self, input_ids: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Run forward pass and return activations at specified layers."""
        self.activations = {}
        with torch.no_grad():
            _ = self.model(input_ids)
        return {k: v.cpu() for k, v in self.activations.items()}


def load_model_and_tokenizer(config: SteeringVectorConfig):
    """Load model and tokenizer from HuggingFace."""
    print(f"Loading model: {config.model_path}")
    print(f"Model name: {config.model_name}")
    print(f"Device: {config.device}, dtype: {config.dtype}")

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        torch_dtype=config.dtype,
        device_map=config.device,
        trust_remote_code=True
    )
    model.eval()

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def compute_activation_for_word(
    word: str,
    tokenizer: AutoTokenizer,
    extractor: ActivationExtractor,
    config: SteeringVectorConfig,
    model: AutoModelForCausalLM
) -> Dict[int, torch.Tensor]:
    """Compute activations for a single word using the prompt template."""
    prompt = config.prompt_template.format(word=word)

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    # Move to model's device (handles multi-GPU with device_map="auto")
    input_ids = inputs["input_ids"].to(model.device)

    # Extract activations
    activations = extractor.extract(input_ids)

    return activations


def compute_baseline_activations(
    tokenizer: AutoTokenizer,
    extractor: ActivationExtractor,
    config: SteeringVectorConfig,
    model: AutoModelForCausalLM
) -> Dict[int, torch.Tensor]:
    """Compute mean baseline activations across neutral words."""
    print("Computing baseline activations...")

    all_activations: Dict[int, List[torch.Tensor]] = {}

    for word in tqdm(BASELINE_WORDS, desc="Baseline words"):
        activations = compute_activation_for_word(word, tokenizer, extractor, config, model)

        for layer_idx, act in activations.items():
            if layer_idx not in all_activations:
                all_activations[layer_idx] = []
            all_activations[layer_idx].append(act)

    # Compute mean across all baseline words
    baseline = {}
    for layer_idx, acts in all_activations.items():
        stacked = torch.stack(acts, dim=0)
        baseline[layer_idx] = stacked.mean(dim=0)

    return baseline


def generate_steering_vectors(
    concepts: List[str],
    tokenizer: AutoTokenizer,
    extractor: ActivationExtractor,
    baseline: Dict[int, torch.Tensor],
    config: SteeringVectorConfig,
    model: AutoModelForCausalLM
) -> Dict[str, Dict[int, torch.Tensor]]:
    """Generate steering vectors for all concepts."""
    print(f"Generating steering vectors for {len(concepts)} concepts...")

    steering_vectors = {}

    for concept in tqdm(concepts, desc="Concepts"):
        # Get activations for this concept
        concept_activations = compute_activation_for_word(
            concept, tokenizer, extractor, config, model
        )

        # Compute steering vector: v_c = a_c - a_baseline
        steering_vectors[concept] = {}
        for layer_idx, act in concept_activations.items():
            steering_vectors[concept][layer_idx] = act - baseline[layer_idx]

    return steering_vectors


def save_steering_vectors(
    steering_vectors: Dict[str, Dict[int, torch.Tensor]],
    config: SteeringVectorConfig
):
    """Save steering vectors to disk in model-specific folder."""
    output_dir = config.model_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save each concept's vectors
    for concept, layer_vectors in steering_vectors.items():
        concept_dir = output_dir / concept
        concept_dir.mkdir(exist_ok=True)

        for layer_idx, vector in layer_vectors.items():
            torch.save(vector, concept_dir / f"layer_{layer_idx}.pt")

    # Save metadata
    metadata = {
        "model_path": config.model_path,
        "model_name": config.model_name,
        "prompt_template": config.prompt_template,
        "concepts": list(steering_vectors.keys()),
        "layers": list(next(iter(steering_vectors.values())).keys()),
        "baseline_words": BASELINE_WORDS,
        "vector_shape": list(next(iter(next(iter(steering_vectors.values())).values())).shape)
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved steering vectors to {output_dir}")


def main():
    """Main entry point for steering vector generation."""
    parser = argparse.ArgumentParser(description="Generate steering vectors for a model")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the model")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Base output directory for steering vectors")
    args = parser.parse_args()

    config = SteeringVectorConfig(
        model_path=args.model,
        output_dir=args.output_dir
    )

    print(f"Will save vectors to: {config.model_output_dir}")

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)

    # Create activation extractor
    extractor = ActivationExtractor(model, config)
    extractor.register_hooks()

    try:
        # Compute baseline activations
        baseline = compute_baseline_activations(tokenizer, extractor, config, model)

        # Generate steering vectors for all concepts
        steering_vectors = generate_steering_vectors(
            CONCEPTS, tokenizer, extractor, baseline, config, model
        )

        # Save to disk
        save_steering_vectors(steering_vectors, config)

        print("\nSteering vector generation complete!")
        print(f"Generated vectors for {len(CONCEPTS)} concepts")
        print(f"Layers: {len(next(iter(steering_vectors.values())))} layers")
        print(f"Vector dimension: {next(iter(next(iter(steering_vectors.values())).values())).shape}")

    finally:
        extractor.clear_hooks()


if __name__ == "__main__":
    main()

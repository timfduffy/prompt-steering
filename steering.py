"""
Shared steering vector utilities.

Consolidates common code for:
- Model name extraction
- Model layer detection
- Steering vector loading
- Steering injection
"""

import torch
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def extract_model_name(model_path: str) -> str:
    """Extract a clean model name from a model path."""
    # Handle HuggingFace cache paths like:
    # H:/Models/huggingface/models--google--gemma-3-4b-it/snapshots/...
    match = re.search(r'models--([^/\\]+)--([^/\\]+)', model_path)
    if match:
        return f"{match.group(1)}-{match.group(2)}"

    # Handle direct model names like google/gemma-3-4b-it
    match = re.search(r'([^/\\]+)/([^/\\]+)$', model_path)
    if match:
        return f"{match.group(1)}-{match.group(2)}"

    # Fallback: use the last directory component
    return Path(model_path).name


def get_model_layers(model):
    """Get transformer layers from various model architectures."""
    # Gemma 3 multimodal: model.model.language_model.layers
    if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
        if hasattr(model.model.language_model, 'layers'):
            return model.model.language_model.layers

    # Standard: model.model.layers (Llama, Qwen, etc.)
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers

    # Alternative Gemma 3: model.language_model.layers
    if hasattr(model, 'language_model') and hasattr(model.language_model, 'layers'):
        return model.language_model.layers

    # Gemma 3 alternative: model.language_model.model.layers
    if hasattr(model, 'language_model') and hasattr(model.language_model, 'model'):
        if hasattr(model.language_model.model, 'layers'):
            return model.language_model.model.layers

    # GPT-2 style
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h

    raise ValueError(f"Cannot determine model architecture for {type(model)}")


def get_steering_vectors_path(model_path: str, base_dir: str) -> Path:
    """Get the path to steering vectors for a given model."""
    model_name = extract_model_name(model_path)
    return Path(base_dir) / model_name


def load_steering_vectors(
    model_path: str,
    base_dir: str,
    concepts: Optional[List[str]] = None,
    layers: Optional[List[int]] = None
) -> Tuple[Dict[str, Dict[int, torch.Tensor]], dict]:
    """
    Load steering vectors for a model.

    Args:
        model_path: Path to the model
        base_dir: Base directory for steering vectors
        concepts: Specific concepts to load (None = all)
        layers: Specific layers to load (None = all)

    Returns:
        Tuple of (vectors dict, metadata dict)
    """
    vectors_dir = get_steering_vectors_path(model_path, base_dir)

    if not vectors_dir.exists():
        model_name = extract_model_name(model_path)
        raise FileNotFoundError(
            f"Steering vectors not found for model '{model_name}' at {vectors_dir}. "
            f"Run generate_steering_vectors.py first."
        )

    with open(vectors_dir / "metadata.json") as f:
        metadata = json.load(f)

    # Determine which concepts to load
    if concepts:
        available = set(metadata["concepts"])
        for c in concepts:
            if c not in available:
                raise ValueError(f"Concept '{c}' not found. Available: {metadata['concepts']}")
        selected_concepts = concepts
    else:
        selected_concepts = metadata["concepts"]

    # Determine which layers to load
    if layers:
        selected_layers = layers
    else:
        selected_layers = metadata["layers"]

    # Load vectors
    vectors = {}
    for concept in selected_concepts:
        vectors[concept] = {}
        concept_dir = vectors_dir / concept
        for layer in selected_layers:
            vector_path = concept_dir / f"layer_{layer}.pt"
            vectors[concept][layer] = torch.load(vector_path, weights_only=True)

    return vectors, metadata


def auto_select_layers(metadata: dict, num_layers: int = 5) -> Tuple[int, ...]:
    """Select representative layers spread across the model."""
    available_layers = metadata.get("layers", [])
    total = len(available_layers)
    if total == 0:
        return (5, 10, 15, 20, 25, 30)

    # Pick evenly spaced layers
    indices = [int(total * i / (num_layers + 1)) for i in range(1, num_layers + 1)]
    return tuple(available_layers[i] for i in indices if i < total)


class SteeringInjector:
    """Handles steering vector injection during forward pass.

    Supports batched operation with different strengths per batch item.
    """

    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.steering_vector: Optional[torch.Tensor] = None
        self.injection_layer: Optional[int] = None
        self.injection_strength: float = 1.0
        # For batched operation: tensor of strengths [batch_size]
        self.batched_strengths: Optional[torch.Tensor] = None
        self.exclude_last_n_positions: int = 0
        self.inject_during_generation: bool = True
        self.is_generating: bool = False

        self.layers = get_model_layers(model)

    def _create_hook(self, layer_idx: int):
        """Create injection hook for a specific layer."""
        def hook(module, input, output):
            if layer_idx != self.injection_layer:
                return output
            if self.steering_vector is None:
                return output
            if self.is_generating and not self.inject_during_generation:
                return output

            if isinstance(output, tuple):
                hidden_states = output[0]
                rest = output[1:]
            else:
                hidden_states = output
                rest = None

            steering = self.steering_vector.to(hidden_states.device, hidden_states.dtype)

            # Handle batched strengths: [batch_size] -> [batch_size, 1, 1]
            if self.batched_strengths is not None:
                strengths = self.batched_strengths.to(hidden_states.device, hidden_states.dtype)
                # steering: [hidden_dim], strengths: [batch_size]
                # Result: [batch_size, 1, hidden_dim]
                steering = steering.unsqueeze(0).unsqueeze(0) * strengths[:, None, None]
            else:
                steering = steering * self.injection_strength

            seq_len = hidden_states.shape[1]

            if self.exclude_last_n_positions > 0 and seq_len > self.exclude_last_n_positions:
                positions_to_steer = seq_len - self.exclude_last_n_positions
                modified = hidden_states.clone()
                if self.batched_strengths is not None:
                    # Batched: steering is [batch_size, 1, hidden_dim]
                    modified[:, :positions_to_steer, :] = hidden_states[:, :positions_to_steer, :] + steering
                else:
                    modified[:, :positions_to_steer, :] = hidden_states[:, :positions_to_steer, :] + steering
            else:
                if self.batched_strengths is not None:
                    # Batched: steering already has batch dim
                    modified = hidden_states + steering
                else:
                    modified = hidden_states + steering.unsqueeze(1)

            if rest is not None:
                return (modified,) + rest
            return modified

        return hook

    def register_hooks(self):
        """Register forward hooks on all layers."""
        self.clear_hooks()
        for idx, layer in enumerate(self.layers):
            hook = layer.register_forward_hook(self._create_hook(idx))
            self.hooks.append(hook)

    def clear_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def set_steering(self, vector: torch.Tensor, layer: int, strength: float):
        """Configure steering for next forward pass."""
        self.steering_vector = vector
        self.injection_layer = layer
        self.injection_strength = strength
        self.batched_strengths = None  # Clear batched mode

    def set_batched_steering(self, vector: torch.Tensor, layer: int, strengths: torch.Tensor):
        """Configure batched steering with different strengths per batch item.

        Args:
            vector: Steering vector [hidden_dim]
            layer: Layer to inject at
            strengths: Tensor of strengths [batch_size]
        """
        self.steering_vector = vector
        self.injection_layer = layer
        self.batched_strengths = strengths
        self.injection_strength = 1.0  # Not used in batched mode

    def clear_steering(self):
        """Disable steering."""
        self.steering_vector = None
        self.injection_layer = None
        self.batched_strengths = None

    def set_inject_during_generation(self, inject: bool):
        """Set whether to inject during generation (continuous vs prompt-only)."""
        self.inject_during_generation = inject

    def set_generation_mode(self, is_generating: bool):
        """Track whether we're in generation phase."""
        self.is_generating = is_generating

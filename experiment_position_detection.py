"""
Experiment: Position-Based Steering Detection

Tests whether models can identify WHICH statement in a sequence has been steered.

Setup:
- Model is told it will see 5 statements, one of which has steering applied
- 5 neutral statements are presented (none related to the 50 concepts)
- Steering is applied ONLY during the token positions of one randomly-selected statement
- Model must identify which statement was steered and explain why
"""

import torch
import json
import argparse
import random
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from steering import extract_model_name, get_steering_vectors_path, load_steering_vectors, auto_select_layers, SteeringInjector


# Neutral statements - varied topics, similar structure, none related to the 50 concepts
# These should be mundane observations that won't trigger any concept associations
NEUTRAL_STATEMENTS = [
    "The morning commute took longer than expected due to construction.",
    "Several participants arrived early for the scheduled meeting.",
    "The temperature dropped significantly after sunset yesterday.",
    "New regulations require additional documentation for approval.",
    "The quarterly report shows steady progress across departments.",
    "Recent updates improved the system's overall performance.",
    "The conference room was reserved for the afternoon session.",
    "Standard procedures were followed during the inspection.",
    "The deadline was extended by two weeks after discussion.",
    "Attendance at the workshop exceeded initial expectations.",
    "The project timeline was adjusted to accommodate changes.",
    "Quality checks confirmed the products met specifications.",
    "The networking event attracted professionals from various fields.",
    "Budget allocations were finalized after the review process.",
    "The training program received positive feedback from attendees.",
    "Infrastructure improvements are scheduled for next quarter.",
    "The survey results indicated high satisfaction rates.",
    "Resource allocation was optimized based on recent analysis.",
    "The maintenance schedule was updated following the audit.",
    "Collaboration between teams led to improved outcomes.",
    "The presentation covered key findings from the research.",
    "Safety protocols were reviewed and updated accordingly.",
    "The vendor provided samples for evaluation purposes.",
    "Performance metrics showed improvement over last period.",
    "The agenda was distributed before the planning session.",
]


SYSTEM_PROMPT = """I am an interpretability researcher studying transformer-based language models like you. I have access to the internals of your neural network and can inject activation patterns ("thoughts") into specific positions in your processing.

In this trial, I will present you with 5 statements. I have applied steering to exactly ONE of these statements - meaning I injected a concept-related activation pattern while you were processing that specific statement only.

Your task: Identify which statement (1-5) had steering applied, and explain what made that statement feel different or what concept you sensed."""

ASSISTANT_ACK = "I understand. I'll carefully attend to my internal states while processing each statement and try to identify which one had an injected activation pattern. Please present the 5 statements."


@dataclass
class ExperimentConfig:
    """Configuration for position detection experiment."""
    model_path: str
    steering_vectors_base_dir: str
    output_dir: str

    # Layers to test
    layers: Tuple[int, ...] = (6, 12, 18, 24, 30)

    # Steering strengths to test
    strengths: Tuple[float, ...] = (0.25, 0.5, 1.0)

    # Concepts to test (if None, uses all available)
    concepts: Optional[Tuple[str, ...]] = None

    # Number of trials per concept/layer/strength combination
    trials_per_combo: int = 1

    # Generation settings
    max_new_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9

    device: str = "auto"
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
    layer: int
    strength: float
    steered_position: int  # 1-5, which statement was steered
    statements: List[str]  # The 5 statements used
    response: str
    detected_position: Optional[int]  # Which position the model identified (parsed from response)
    correct: bool  # Whether detection was correct
    input_tokens: int
    output_tokens: int


def select_statements(n: int = 5) -> List[str]:
    """Select n random neutral statements."""
    return random.sample(NEUTRAL_STATEMENTS, n)


def build_prompt_with_statements(tokenizer, statements: List[str]) -> str:
    """Build the full prompt with numbered statements."""
    statements_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(statements))

    trial_prompt = f"""Here are the 5 statements:

{statements_text}

Which statement (1-5) had steering applied? Please explain what you noticed."""

    messages = [
        {"role": "user", "content": SYSTEM_PROMPT},
        {"role": "assistant", "content": ASSISTANT_ACK},
        {"role": "user", "content": trial_prompt}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    return prompt


def find_statement_token_positions(
    tokenizer,
    full_prompt: str,
    statements: List[str]
) -> List[Tuple[int, int]]:
    """Find the token positions (start, end) for each statement in the prompt."""
    # Tokenize full prompt
    full_tokens = tokenizer.encode(full_prompt, add_special_tokens=False)

    positions = []

    for i, statement in enumerate(statements):
        # Find the statement text in the prompt (with number prefix)
        statement_with_num = f"{i+1}. {statement}"

        # Find character position in prompt
        char_start = full_prompt.find(statement_with_num)
        if char_start == -1:
            raise ValueError(f"Could not find statement '{statement_with_num}' in prompt")
        char_end = char_start + len(statement_with_num)

        # Convert character positions to token positions
        # Tokenize prefix to find start token
        prefix = full_prompt[:char_start]
        prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
        token_start = len(prefix_tokens)

        # Tokenize up to end to find end token
        prefix_with_statement = full_prompt[:char_end]
        prefix_with_statement_tokens = tokenizer.encode(prefix_with_statement, add_special_tokens=False)
        token_end = len(prefix_with_statement_tokens)

        positions.append((token_start, token_end))

    return positions


class PositionAwareSteeringInjector(SteeringInjector):
    """Extended steering injector that only steers specific token positions."""

    def __init__(self, model):
        super().__init__(model)
        self.steer_positions: Optional[Tuple[int, int]] = None  # (start, end) token positions
        self.current_position: int = 0  # Track current position during forward passes

    def set_steer_positions(self, start: int, end: int):
        """Set which token positions should be steered."""
        self.steer_positions = (start, end)
        self.current_position = 0

    def clear_steer_positions(self):
        """Clear position-based steering."""
        self.steer_positions = None
        self.current_position = 0

    def _create_hook(self, layer_idx: int):
        """Create a hook that only steers at specific positions."""
        def hook(module, input, output):
            if self.steering_vector is None or self.injection_layer != layer_idx:
                return output

            if isinstance(output, tuple):
                hidden_states = output[0]
                rest = output[1:]
            else:
                hidden_states = output
                rest = None

            batch_size, seq_len, hidden_dim = hidden_states.shape

            # Prepare steering vector with proper shape for broadcasting
            steering = self.steering_vector.to(hidden_states.device, hidden_states.dtype)
            steering = steering * self.injection_strength
            # Ensure steering has shape [1, 1, hidden_dim] for proper broadcasting
            if steering.dim() == 1:
                steering = steering.unsqueeze(0).unsqueeze(0)
            elif steering.dim() == 2:
                steering = steering.unsqueeze(1)

            if self.steer_positions is not None:
                # Position-based steering: only steer tokens in the specified range
                start, end = self.steer_positions

                if seq_len > 1:
                    # Full sequence (prompt processing)
                    # Create a mask for positions to steer
                    modified = hidden_states.clone()
                    for pos in range(max(0, start), min(seq_len, end)):
                        modified[:, pos:pos+1, :] = hidden_states[:, pos:pos+1, :] + steering
                    hidden_states = modified
                # else: Single token (generation) - don't steer during generation
            else:
                # Standard steering (all positions) - fallback to parent behavior
                if self.exclude_last_n_positions > 0 and seq_len > self.exclude_last_n_positions:
                    modified = hidden_states.clone()
                    positions_to_steer = seq_len - self.exclude_last_n_positions
                    modified[:, :positions_to_steer, :] = hidden_states[:, :positions_to_steer, :] + steering
                    hidden_states = modified
                else:
                    hidden_states = hidden_states + steering

            if rest is not None:
                return (hidden_states,) + rest
            return hidden_states

        return hook


def parse_detected_position(response: str) -> Optional[int]:
    """Try to parse which position (1-5) the model identified."""
    # Look for patterns like "Statement 3", "number 3", "3.", "(3)", etc.
    import re

    # First, look for explicit "Statement X" or "statement X"
    match = re.search(r'[Ss]tatement\s*(\d)', response)
    if match:
        pos = int(match.group(1))
        if 1 <= pos <= 5:
            return pos

    # Look for "number X" or "#X"
    match = re.search(r'(?:number|#)\s*(\d)', response, re.IGNORECASE)
    if match:
        pos = int(match.group(1))
        if 1 <= pos <= 5:
            return pos

    # Look for "X." at the start or "is X" pattern
    match = re.search(r'(?:^|\s)([1-5])[\.\)]', response)
    if match:
        return int(match.group(1))

    # Look for "is [number]" or "was [number]"
    match = re.search(r'(?:is|was|choose|select|pick)\s+(\d)', response, re.IGNORECASE)
    if match:
        pos = int(match.group(1))
        if 1 <= pos <= 5:
            return pos

    return None


def run_trial(
    model,
    tokenizer,
    injector: PositionAwareSteeringInjector,
    concept: str,
    layer: int,
    strength: float,
    steering_vector: torch.Tensor,
    config: ExperimentConfig
) -> TrialResult:
    """Run a single position detection trial."""

    # Select 5 random statements
    statements = select_statements(5)

    # Randomly choose which statement to steer (1-5)
    steered_position = random.randint(1, 5)

    # Build the prompt
    prompt = build_prompt_with_statements(tokenizer, statements)

    # Find token positions for each statement
    statement_positions = find_statement_token_positions(tokenizer, prompt, statements)

    # Get the token range for the steered statement
    steer_start, steer_end = statement_positions[steered_position - 1]  # -1 for 0-indexing

    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    input_len = input_ids.shape[1]

    with torch.no_grad():
        # Set up position-aware steering
        injector.set_steering(steering_vector, layer, strength)
        injector.set_steer_positions(steer_start, steer_end)

        # Process prompt with position-specific steering
        prompt_outputs = model(
            input_ids=input_ids,
            use_cache=True,
            return_dict=True
        )
        past_key_values = prompt_outputs.past_key_values

        # Clear steering for generation
        injector.clear_steering()
        injector.clear_steer_positions()

        # Generate response without steering
        generated_ids = []
        # Handle potential NaN from steering in initial logits
        initial_logits = prompt_outputs.logits[:, -1, :]
        initial_logits = torch.nan_to_num(initial_logits, nan=0.0, posinf=100.0, neginf=-100.0)
        current_token = torch.argmax(initial_logits, dim=-1, keepdim=True)

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

            # Sample next token
            logits = step_outputs.logits[:, -1, :] / config.temperature
            probs = torch.softmax(logits, dim=-1)
            current_token = torch.multinomial(probs, num_samples=1)

        if generated_ids:
            outputs = torch.cat([input_ids] + generated_ids, dim=1)
        else:
            outputs = input_ids

    # Decode response
    response = tokenizer.decode(outputs[0, input_len:], skip_special_tokens=True)

    # Parse which position the model detected
    detected_position = parse_detected_position(response)
    correct = detected_position == steered_position

    return TrialResult(
        concept=concept,
        layer=layer,
        strength=strength,
        steered_position=steered_position,
        statements=statements,
        response=response,
        detected_position=detected_position,
        correct=correct,
        input_tokens=input_len,
        output_tokens=outputs.shape[1] - input_len
    )


def run_experiment(config: ExperimentConfig):
    """Run the full position detection experiment."""
    print("=" * 60)
    print("Experiment: Position-Based Steering Detection")
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

    # Setup position-aware injector
    injector = PositionAwareSteeringInjector(model)
    injector.register_hooks()

    # Calculate total trials
    num_concepts = len(steering_vectors)
    num_layers = len(config.layers)
    num_strengths = len(config.strengths)
    total_trials = num_concepts * num_layers * num_strengths * config.trials_per_combo

    print(f"\nRunning {total_trials} trials...")
    print(f"  - {num_concepts} concepts")
    print(f"  - {num_layers} layers")
    print(f"  - {num_strengths} strengths")
    print(f"  - {config.trials_per_combo} trials per combination")

    results: List[TrialResult] = []
    pbar = tqdm(total=total_trials, desc="Trials")

    try:
        for concept, layer_vectors in steering_vectors.items():
            for layer, vector in layer_vectors.items():
                if layer not in config.layers:
                    continue
                for strength in config.strengths:
                    for _ in range(config.trials_per_combo):
                        result = run_trial(
                            model, tokenizer, injector,
                            concept=concept,
                            layer=layer,
                            strength=strength,
                            steering_vector=vector,
                            config=config
                        )
                        results.append(result)
                        pbar.update(1)

                        # Update progress bar with accuracy
                        correct_so_far = sum(1 for r in results if r.correct)
                        pbar.set_postfix({"acc": f"{correct_so_far/len(results):.1%}"})

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
            "max_new_tokens": config.max_new_tokens,
            "temperature": config.temperature,
        },
        "concepts_tested": list(steering_vectors.keys()),
        "trials": [asdict(r) for r in results]
    }

    output_file = output_dir / f"position_detection_{timestamp}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    # Overall accuracy
    total_correct = sum(1 for r in results if r.correct)
    print(f"\nOverall accuracy: {total_correct}/{len(results)} ({total_correct/len(results):.1%})")
    print(f"(Chance level: 20%)")

    # Accuracy by strength
    print("\nAccuracy by strength:")
    for strength in config.strengths:
        strength_results = [r for r in results if r.strength == strength]
        correct = sum(1 for r in strength_results if r.correct)
        print(f"  {strength}: {correct}/{len(strength_results)} ({correct/len(strength_results):.1%})")

    # Accuracy by layer
    print("\nAccuracy by layer:")
    for layer in config.layers:
        layer_results = [r for r in results if r.layer == layer]
        correct = sum(1 for r in layer_results if r.correct)
        print(f"  Layer {layer}: {correct}/{len(layer_results)} ({correct/len(layer_results):.1%})")

    # Detection confusion matrix (which position did model guess vs actual)
    print("\nPosition confusion (rows=actual, cols=detected):")
    print("      1    2    3    4    5   None")
    for actual in range(1, 6):
        actual_results = [r for r in results if r.steered_position == actual]
        row = f"  {actual}:"
        for detected in [1, 2, 3, 4, 5, None]:
            count = sum(1 for r in actual_results if r.detected_position == detected)
            row += f" {count:4d}"
        print(row)

    return results


def main():
    parser = argparse.ArgumentParser(description="Run position-based steering detection experiment")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the model")
    parser.add_argument("--steering-vectors-dir", type=str, required=True,
                        help="Base directory for steering vectors")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layers to test (e.g., '6,12,18,24,30')")
    parser.add_argument("--concepts", type=str, default=None,
                        help="Comma-separated concepts to test. Uses all if not specified.")
    parser.add_argument("--strengths", type=str, default=None,
                        help="Comma-separated strengths to test (e.g., '0.25,0.5,1.0')")
    parser.add_argument("--trials-per-combo", type=int, default=1,
                        help="Number of trials per concept/layer/strength combination")
    args = parser.parse_args()

    # Parse layers
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
            layers = (6, 12, 18, 24, 30)

    # Parse concepts
    concepts = None
    if args.concepts:
        concepts = tuple(c.strip() for c in args.concepts.split(","))

    # Parse strengths
    strengths = (0.25, 0.5, 1.0)
    if args.strengths:
        strengths = tuple(float(s.strip()) for s in args.strengths.split(","))

    config = ExperimentConfig(
        model_path=args.model,
        steering_vectors_base_dir=args.steering_vectors_dir,
        output_dir=args.output_dir,
        layers=layers,
        concepts=concepts,
        strengths=strengths,
        trials_per_combo=args.trials_per_combo,
    )

    print(f"Model: {config.model_name}")
    print(f"Steering vectors dir: {config.steering_vectors_dir}")

    run_experiment(config)


if __name__ == "__main__":
    main()

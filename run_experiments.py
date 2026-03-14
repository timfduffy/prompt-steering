#!/usr/bin/env python3
"""
Experiment Runner

Coordinates running experiments based on YAML configuration.

Usage:
    python run_experiments.py                    # Run all enabled experiments
    python run_experiments.py --config my.yaml   # Use custom config
    python run_experiments.py --only prompt_variants  # Run specific experiment
    python run_experiments.py --list             # List available experiments
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from config import load_config, Config, get_project_root


def run_command(cmd: List[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60 + "\n")

    result = subprocess.run(cmd)
    success = result.returncode == 0

    if success:
        print(f"\n[OK] {description} completed successfully")
    else:
        print(f"\n[FAILED] {description} failed with code {result.returncode}")

    return success


def generate_vectors(config: Config, models: List[str]) -> bool:
    """Generate steering vectors for specified models."""
    all_success = True

    for model_key in models:
        model = config.get_model(model_key)
        vectors_dir = config.get_steering_vectors_dir(model)

        # Check if vectors already exist
        metadata_path = vectors_dir / "metadata.json"
        if metadata_path.exists():
            print(f"Steering vectors already exist for {model_key}, skipping...")
            continue

        cmd = [
            sys.executable, "generate_steering_vectors.py",
            "--model", model.path,
            "--output-dir", str(config.settings.steering_vectors_dir),
        ]

        success = run_command(cmd, f"Generate steering vectors for {model_key}")
        all_success = all_success and success

    return all_success


def run_prompt_variants(config: Config, exp_config: dict) -> bool:
    """Run prompt variants experiment."""
    all_success = True
    models = exp_config.get("models", list(config.models.keys()))

    for model_key in models:
        model = config.get_model(model_key)
        results_dir = config.get_results_dir("prompt_variants")

        cmd = [
            sys.executable, "experiment_prompt_variants.py",
            "--model", model.path,
            "--steering-vectors-dir", str(config.settings.steering_vectors_dir),
            "--output-dir", str(results_dir),
        ]

        # Add concepts from config settings
        if config.settings.concepts:
            cmd.extend(["--concepts", ",".join(config.settings.concepts)])

        # Add layers if specified
        layers = exp_config.get("layers")
        if layers:
            cmd.extend(["--layers", ",".join(str(l) for l in layers)])

        success = run_command(cmd, f"Prompt variants experiment on {model.name}")
        all_success = all_success and success

    return all_success


def run_prompt_vs_full(config: Config, exp_config: dict) -> bool:
    """Run prompt-only vs full steering experiment."""
    all_success = True
    models = exp_config.get("models", list(config.models.keys()))

    for model_key in models:
        model = config.get_model(model_key)
        results_dir = config.get_results_dir("prompt_vs_full")

        cmd = [
            sys.executable, "experiment_prompt_vs_full_steering.py",
            "--model", model.path,
            "--steering-vectors-dir", str(config.settings.steering_vectors_dir),
            "--output-dir", str(results_dir),
        ]

        # Add concepts from config settings
        if config.settings.concepts:
            cmd.extend(["--concepts", ",".join(config.settings.concepts)])

        if "layers" in exp_config and exp_config["layers"]:
            cmd.extend(["--layers", ",".join(str(l) for l in exp_config["layers"])])

        success = run_command(cmd, f"Prompt vs full steering on {model.name}")
        all_success = all_success and success

    return all_success


def run_classification(config: Config, exp_config: dict) -> bool:
    """Run classification on experiment results."""
    all_success = True
    experiments = exp_config.get("experiments", [])
    classifier_model = exp_config.get("model", "openai/gpt-4o-mini")

    for exp_name in experiments:
        results_dir = config.get_results_dir(exp_name)

        if not results_dir.exists():
            print(f"Results directory {results_dir} does not exist, skipping classification")
            continue

        # Find all result files
        result_files = list(results_dir.glob("results_*.json"))

        for result_file in result_files:
            # Skip already classified files
            classified_file = result_file.with_suffix(".classified.json")
            if classified_file.exists():
                print(f"Already classified: {result_file.name}, skipping...")
                continue

            cmd = [
                sys.executable, "classify_responses.py",
                str(result_file),
                "--model", classifier_model,
                "--output", str(classified_file),
            ]

            success = run_command(cmd, f"Classify {result_file.name}")
            all_success = all_success and success

    return all_success


EXPERIMENT_RUNNERS = {
    "generate_vectors": generate_vectors,
    "prompt_variants": run_prompt_variants,
    "prompt_vs_full": run_prompt_vs_full,
    "classify": run_classification,
}


def list_experiments(config: Config):
    """List available experiments and their status."""
    print("\nAvailable experiments:")
    print("-" * 40)

    for name, exp_config in config.experiments.items():
        enabled = exp_config.get("enabled", False)
        status = "[ENABLED]" if enabled else "[disabled]"
        print(f"  {name:20} {status}")

    print("\nConfigured models:")
    print("-" * 40)
    for name, model in config.models.items():
        print(f"  {name:20} -> {model.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Run introspection experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiments.py                     # Run all enabled experiments
  python run_experiments.py --only prompt_variants
  python run_experiments.py --only generate_vectors,prompt_variants
  python run_experiments.py --list              # Show available experiments
        """
    )
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--only", type=str, default=None,
                        help="Comma-separated list of experiments to run (overrides 'enabled' in config)")
    parser.add_argument("--list", action="store_true",
                        help="List available experiments and exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be run without executing")
    args = parser.parse_args()

    # Change to project root
    project_root = get_project_root()

    # Load config
    config_path = project_root / args.config
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    config = load_config(str(config_path))

    if args.list:
        list_experiments(config)
        sys.exit(0)

    # Determine which experiments to run
    if args.only:
        experiments_to_run = [e.strip() for e in args.only.split(",")]
    else:
        experiments_to_run = [
            name for name, exp_config in config.experiments.items()
            if exp_config.get("enabled", False)
        ]

    if not experiments_to_run:
        print("No experiments to run. Use --only or enable experiments in config.yaml")
        sys.exit(0)

    print(f"Experiments to run: {', '.join(experiments_to_run)}")

    if args.dry_run:
        print("\n[DRY RUN] Would run the above experiments")
        sys.exit(0)

    # Run experiments
    results = {}
    for exp_name in experiments_to_run:
        if exp_name not in EXPERIMENT_RUNNERS:
            print(f"Unknown experiment: {exp_name}")
            results[exp_name] = False
            continue

        exp_config = config.experiments.get(exp_name, {})
        runner = EXPERIMENT_RUNNERS[exp_name]

        # Special case: generate_vectors uses models list directly
        if exp_name == "generate_vectors":
            models = exp_config.get("models", list(config.models.keys()))
            results[exp_name] = runner(config, models)
        else:
            results[exp_name] = runner(config, exp_config)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for exp_name, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  {exp_name:30} [{status}]")

    # Exit with error if any failed
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()

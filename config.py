"""
Shared configuration utilities for introspection experiments.

Handles YAML config loading and provides common path resolution.
"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from steering import extract_model_name


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent


def resolve_path(path: str, base_dir: Optional[Path] = None) -> Path:
    """Resolve a path, expanding ~ and making relative paths absolute."""
    if base_dir is None:
        base_dir = get_project_root()

    path = os.path.expanduser(path)
    path = os.path.expandvars(path)
    p = Path(path)

    if not p.is_absolute():
        p = base_dir / p

    return p


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    path: str
    name: Optional[str] = None

    def __post_init__(self):
        if self.name is None:
            self.name = extract_model_name(self.path)


@dataclass
class ExperimentSettings:
    """Common experiment settings."""
    steering_vectors_dir: Path
    results_dir: Path
    concepts: List[str] = field(default_factory=lambda: ["silver", "volcanoes"])
    strengths: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])
    max_new_tokens: int = 50
    temperature: float = 0.7


@dataclass
class Config:
    """Main configuration container."""
    models: Dict[str, ModelConfig]
    settings: ExperimentSettings
    experiments: Dict[str, Any]

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from a YAML file."""
        config_path = Path(path)
        base_dir = config_path.parent

        with open(config_path) as f:
            data = yaml.safe_load(f)

        # Parse models
        models = {}
        for name, model_data in data.get("models", {}).items():
            if isinstance(model_data, str):
                models[name] = ModelConfig(path=model_data, name=name)
            else:
                models[name] = ModelConfig(
                    path=model_data["path"],
                    name=model_data.get("name", name)
                )

        # Parse settings
        settings_data = data.get("settings", {})
        settings = ExperimentSettings(
            steering_vectors_dir=resolve_path(
                settings_data.get("steering_vectors_dir", "steering_vectors"),
                base_dir
            ),
            results_dir=resolve_path(
                settings_data.get("results_dir", "results"),
                base_dir
            ),
            concepts=settings_data.get("concepts", ["silver", "volcanoes"]),
            strengths=settings_data.get("strengths", [0.5, 1.0, 2.0]),
            max_new_tokens=settings_data.get("max_new_tokens", 50),
            temperature=settings_data.get("temperature", 0.7),
        )

        # Parse experiments
        experiments = data.get("experiments", {})

        return cls(models=models, settings=settings, experiments=experiments)

    def get_model(self, name: str) -> ModelConfig:
        """Get a model config by name."""
        if name not in self.models:
            raise KeyError(f"Model '{name}' not found. Available: {list(self.models.keys())}")
        return self.models[name]

    def get_steering_vectors_dir(self, model: ModelConfig) -> Path:
        """Get the steering vectors directory for a model."""
        # Use the extracted model name from path, not the config key
        actual_name = extract_model_name(model.path)
        return self.settings.steering_vectors_dir / actual_name

    def get_results_dir(self, experiment_name: str) -> Path:
        """Get the results directory for an experiment."""
        return self.settings.results_dir / experiment_name


def load_config(path: Optional[str] = None) -> Config:
    """Load config from path or default location."""
    if path is None:
        # Look for config.yaml in project root
        path = get_project_root() / "config.yaml"
    return Config.from_yaml(path)

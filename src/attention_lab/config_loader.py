"""Configuration loader for YAML-based experiment configs.

This module provides utilities for loading experiment configurations:

- :func:`load_config`: Load experiment configuration from YAML
- :func:`deep_merge`: Deep merge dictionaries
- :class:`ExperimentConfig`: Full experiment configuration
- :class:`ModelConfig`: Model architecture configuration
- :class:`TrainingConfig`: Training hyperparameters
- :class:`DataConfig`: Dataset configuration
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


def deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries. Override values take precedence.

    :param base: Base dictionary.
    :type base: dict
    :param override: Dictionary with override values.
    :type override: dict
    :returns: Merged dictionary.
    :rtype: dict
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


@dataclass
class ModelConfig:
    """Model architecture configuration.

    :param vocab_size: Vocabulary size (None for auto-detect).
    :param block_size: Maximum context length.
    :param n_layer: Number of transformer layers.
    :param n_head: Number of attention heads.
    :param n_embd: Embedding dimension.
    :param dropout: Dropout probability.
    :param bias: Use bias in linear layers.
    """

    vocab_size: int | None = 256
    block_size: int = 64
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    bias: bool = False


@dataclass
class AttentionConfig:
    """Attention mechanism configuration.

    :param type: Attention variant ('vanilla', 'linear', 'sliding_window', 'sparse', 'rotary').
    :param window_size: Window size for sliding_window attention.
    :param local_size: Local window size for sparse attention.
    :param stride: Stride for sparse attention.
    """

    type: str = "vanilla"
    window_size: int = 16  # for sliding_window
    local_size: int = 16  # for sparse
    stride: int = 16  # for sparse


@dataclass
class TrainingConfig:
    """Training hyperparameters configuration.

    :param max_steps: Maximum training steps.
    :param batch_size: Batch size.
    :param learning_rate: Learning rate for AdamW.
    :param weight_decay: Weight decay.
    :param grad_clip: Gradient clipping norm.
    :param warmup_steps: Learning rate warmup steps.
    :param eval_interval: Steps between evaluations.
    :param eval_steps: Batches per evaluation.
    :param log_interval: Steps between logging.
    :param checkpoint_dir: Directory for checkpoints.
    :param save_best: Save best model by validation loss.
    """

    max_steps: int = 5000
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_steps: int = 100
    eval_interval: int = 500
    eval_steps: int = 50
    log_interval: int = 10
    checkpoint_dir: str = "checkpoints"
    save_best: bool = True
    compile: bool = False


@dataclass
class DatasetItemConfig:
    """Single dataset configuration.

    :param type: Dataset type (e.g., 'shakespeare').
    :param weight: Sampling weight.
    :param params: Dataset-specific parameters.
    """

    type: str
    weight: float = 1.0
    # Dataset-specific params stored as dict
    params: dict = field(default_factory=dict)


@dataclass
class DataConfig:
    """Data configuration.

    :param strategy: Dataset loading strategy.
    :param num_samples: Number of samples per dataset.
    :param datasets: List of dataset configurations.
    :param use_cache: Try to load preloaded/cached data first, generate if missing.
    :param cache_dir: Path to cache directory (None for default data/.cache).
    """

    strategy: str = "weighted"
    num_samples: int = 20000
    datasets: list[DatasetItemConfig] = field(default_factory=list)
    use_cache: bool = False
    cache_dir: str | None = None


@dataclass
class GenerationConfig:
    """Text generation configuration.

    :param max_new_tokens: Maximum tokens to generate.
    :param temperature: Sampling temperature.
    :param top_k: Top-k sampling (None to disable).
    :param top_p: Nucleus sampling threshold (None to disable).
    """

    max_new_tokens: int = 100
    temperature: float = 0.8
    top_k: int | None = 40
    top_p: float | None = None


@dataclass
class ExperimentConfig:
    """Full experiment configuration.

    Contains all sub-configurations for model, training, data, etc.

    :param model: Model architecture configuration.
    :param attention: Attention mechanism configuration.
    :param training: Training hyperparameters.
    :param data: Dataset configuration.
    :param generation: Generation settings.
    :param name: Experiment name.
    :param seed: Random seed.
    :param output_dir: Output directory.
    :param device: Device (None for auto-detect).
    :param compare_variants: Attention variants to compare.
    """

    # Sub-configs
    model: ModelConfig = field(default_factory=ModelConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    # Experiment metadata
    name: str | None = None
    seed: int = 42
    output_dir: str = "outputs"
    device: str | None = None
    num_workers: int = 0
    pin_memory: bool = True

    # Comparison mode
    compare_variants: list[str] | None = None
    eval_accuracy: bool = False


def load_yaml(path: str | Path) -> dict:
    """Load a YAML file.

    :param path: Path to YAML file.
    :type path: str | Path
    :returns: Parsed YAML as dictionary.
    :rtype: dict
    """
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def find_config_file(config_path: str | None = None) -> Path:
    """Find configuration file.

    Priority:
        1. Explicit path if provided
        2. config/local.yaml if exists
        3. config/default.yaml

    :param config_path: Explicit path to config file.
    :type config_path: Optional[str]
    :returns: Path to config file.
    :rtype: Path
    :raises FileNotFoundError: If no config file found.
    """
    if config_path:
        path = Path(config_path)
        if path.exists():
            return path
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Look for local.yaml first
    local_path = Path("config/local.yaml")
    if local_path.exists():
        return local_path

    # Fall back to default.yaml
    default_path = Path("config/default.yaml")
    if default_path.exists():
        return default_path

    raise FileNotFoundError("No config file found. Create config/default.yaml")


def parse_datasets(data_cfg: dict, defaults: dict) -> list[DatasetItemConfig]:
    """Parse dataset configurations."""
    datasets = []
    for ds_cfg in data_cfg.get("datasets", []):
        ds_type = ds_cfg.get("type")
        weight = ds_cfg.get("weight", 1.0)

        # Get defaults for this dataset type
        default_params = defaults.get(ds_type, {}).copy()

        # Override with explicit params
        params = {k: v for k, v in ds_cfg.items() if k not in ["type", "weight"]}
        default_params.update(params)

        datasets.append(
            DatasetItemConfig(
                type=ds_type,
                weight=weight,
                params=default_params,
            )
        )

    return datasets


def load_config(config_path: str | None = None) -> ExperimentConfig:
    """Load experiment configuration from YAML.

    Loads the config file and merges with defaults.

    :param config_path: Path to config file. If None, uses default search.
    :type config_path: Optional[str]
    :returns: Fully populated experiment configuration.
    :rtype: ExperimentConfig

    Example::

        config = load_config("config/experiments/shakespeare.yaml")
        print(config.model.n_layer)
    """
    # Find and load config file
    path = find_config_file(config_path)
    print(f"Loading config from: {path}")

    # Load base default config
    default_path = Path("config/default.yaml")
    if default_path.exists() and path != default_path:
        base_cfg = load_yaml(default_path)
        override_cfg = load_yaml(path)
        cfg = deep_merge(base_cfg, override_cfg)
    else:
        cfg = load_yaml(path)

    # Parse into dataclasses
    model_cfg = ModelConfig(**cfg.get("model", {}))

    # Attention config with variant-specific settings
    attn_raw = cfg.get("attention", {})
    attn_type = attn_raw.get("type", "vanilla")
    attn_cfg = AttentionConfig(type=attn_type)

    if attn_type == "sliding_window" and "sliding_window" in attn_raw:
        attn_cfg.window_size = attn_raw["sliding_window"].get("window_size", 16)
    elif attn_type == "sparse" and "sparse" in attn_raw:
        attn_cfg.local_size = attn_raw["sparse"].get("local_size", 16)
        attn_cfg.stride = attn_raw["sparse"].get("stride", 16)

    training_cfg = TrainingConfig(**cfg.get("training", {}))
    generation_cfg = GenerationConfig(**cfg.get("generation", {}))

    # Data config
    data_raw = cfg.get("data", {})
    dataset_defaults = cfg.get("dataset_defaults", {})
    datasets = parse_datasets(data_raw, dataset_defaults)

    data_cfg = DataConfig(
        strategy=data_raw.get("strategy", "weighted"),
        num_samples=data_raw.get("num_samples", 20000),
        datasets=datasets,
        use_cache=data_raw.get("use_cache", False),
        cache_dir=data_raw.get("cache_dir"),
    )

    # Experiment metadata
    exp_raw = cfg.get("experiment", {})
    compare_raw = cfg.get("compare", {})

    return ExperimentConfig(
        model=model_cfg,
        attention=attn_cfg,
        training=training_cfg,
        data=data_cfg,
        generation=generation_cfg,
        name=exp_raw.get("name"),
        seed=exp_raw.get("seed", 42),
        output_dir=exp_raw.get("output_dir", "outputs"),
        device=cfg.get("device"),
        num_workers=cfg.get("num_workers", 0),
        pin_memory=cfg.get("pin_memory", True),
        compare_variants=compare_raw.get("variants"),
        eval_accuracy=compare_raw.get("eval_accuracy", False),
    )


def config_to_dict(config: ExperimentConfig) -> dict:
    """Convert ExperimentConfig back to dictionary for logging.

    :param config: Experiment configuration.
    :type config: ExperimentConfig
    :returns: Dictionary representation.
    :rtype: dict
    """
    return {
        "model": {
            "vocab_size": config.model.vocab_size,
            "block_size": config.model.block_size,
            "n_layer": config.model.n_layer,
            "n_head": config.model.n_head,
            "n_embd": config.model.n_embd,
            "dropout": config.model.dropout,
        },
        "attention": {
            "type": config.attention.type,
        },
        "training": {
            "max_steps": config.training.max_steps,
            "batch_size": config.training.batch_size,
            "learning_rate": config.training.learning_rate,
        },
        "data": {
            "strategy": config.data.strategy,
            "num_samples": config.data.num_samples,
            "datasets": [ds.type for ds in config.data.datasets],
        },
        "seed": config.seed,
    }


def print_config(config: ExperimentConfig) -> None:
    """Print configuration summary to stdout.

    :param config: Experiment configuration.
    :type config: ExperimentConfig
    """
    print("\n" + "=" * 60)
    print("CONFIGURATION")
    print("=" * 60)

    print(f"\nExperiment: {config.name or 'unnamed'}")
    print(f"Seed: {config.seed}")
    print(f"Device: {config.device or 'auto'}")

    print("\nModel:")
    print(f"  Layers: {config.model.n_layer}")
    print(f"  Heads: {config.model.n_head}")
    print(f"  Embedding: {config.model.n_embd}")
    print(f"  Block size: {config.model.block_size}")

    print(f"\nAttention: {config.attention.type}")
    if config.attention.type == "sliding_window":
        print(f"  Window size: {config.attention.window_size}")
    elif config.attention.type == "sparse":
        print(f"  Local size: {config.attention.local_size}")
        print(f"  Stride: {config.attention.stride}")

    print("\nTraining:")
    print(f"  Steps: {config.training.max_steps}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Learning rate: {config.training.learning_rate}")

    print("\nData:")
    print(f"  Strategy: {config.data.strategy}")
    print(f"  Samples: {config.data.num_samples}")
    print(f"  Use cache: {config.data.use_cache}")
    if config.data.cache_dir:
        print(f"  Cache dir: {config.data.cache_dir}")
    print("  Datasets:")
    for ds in config.data.datasets:
        print(f"    - {ds.type} (weight={ds.weight})")

    if config.compare_variants:
        print("\nComparison mode:")
        print(f"  Variants: {', '.join(config.compare_variants)}")

    print()

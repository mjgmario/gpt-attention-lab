#!/usr/bin/env python
"""Unified training script using YAML configuration.

Usage:
    # Use default config
    uv run python scripts/train.py

    # Use specific experiment config
    uv run python scripts/train.py --config config/experiments/shakespeare.yaml

    # Compare attention variants
    uv run python scripts/train.py --config config/experiments/full_comparison.yaml --compare

    # Override specific values
    uv run python scripts/train.py --config config/default.yaml --override training.max_steps=1000
"""

import argparse
import copy
import json
import random
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from attention_lab.config_loader import (
    ExperimentConfig,
    load_config,
    print_config,
    config_to_dict,
)
from attention_lab.data.shakespeare import ShakespeareDataset
from attention_lab.data.cache import load_from_cache, is_cached, get_cache_dir
from attention_lab.model_variants import create_model_variant, GPTVariantConfig
from attention_lab.train import Trainer, TrainerConfig


def create_run_dir(base_dir: str, name: str | None) -> Path:
    """Create a timestamped run directory under *base_dir*.

    Returns a path like ``outputs/20260128_153045_shakespeare/``.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = name or "run"
    run_dir = Path(base_dir) / f"{timestamp}_{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_history(history: dict, path: Path) -> None:
    """Persist training history as JSON."""
    with open(path, "w") as f:
        json.dump(history, f, indent=2)


# ---------------------------------------------------------------------------
# Plotting helpers (matplotlib with non-interactive Agg backend)
# ---------------------------------------------------------------------------

def plot_loss_curve(history: dict, path: Path, title: str = "Loss Curve") -> None:
    """Save a single-variant train/val loss curve to *path*."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    steps = history.get("steps", list(range(len(history.get("train_loss", [])))))

    if history.get("train_loss"):
        ax.plot(steps, history["train_loss"], label="Train Loss")
    if history.get("val_loss"):
        val_steps = _val_steps(steps, history["val_loss"])
        ax.plot(val_steps, history["val_loss"], label="Val Loss", linestyle="--")

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_comparison_curves(all_histories: dict[str, dict], path: Path) -> None:
    """Overlay train-loss curves for every variant on a single chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    for variant, history in all_histories.items():
        steps = history.get("steps", list(range(len(history.get("train_loss", [])))))
        if history.get("train_loss"):
            ax.plot(steps, history["train_loss"], label=variant)

    ax.set_xlabel("Step")
    ax.set_ylabel("Train Loss")
    ax.set_title("Variant Comparison — Train Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_final_loss_bar(final_losses: dict[str, float], path: Path) -> None:
    """Bar chart comparing final loss across variants."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variants = list(final_losses.keys())
    losses = [final_losses[v] for v in variants]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(variants, losses)
    ax.set_ylabel("Final Train Loss")
    ax.set_title("Final Loss per Variant")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _val_steps(steps: list, val_loss: list) -> list:
    """Pick evenly-spaced step indices for validation points."""
    if not steps or not val_loss:
        return []
    n = len(val_loss)
    total = len(steps)
    if n >= total:
        return steps[:n]
    stride = max(1, total // n)
    return [steps[min(i * stride, total - 1)] for i in range(n)]


def plot_attention_patterns_comparison(
    models: dict[str, torch.nn.Module],
    dataset,
    device: str,
    path: Path,
    sample_text: str = "To be or not to be",
) -> None:
    """Plot attention heatmaps for each variant side by side."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_variants = len(models)
    fig, axes = plt.subplots(1, n_variants, figsize=(5 * n_variants, 5))
    if n_variants == 1:
        axes = [axes]

    tokens = dataset.encode(sample_text).unsqueeze(0).to(device)

    for ax, (variant, model) in zip(axes, models.items()):
        model.eval()
        with torch.no_grad():
            _, _, attentions = model(tokens, return_attn=True)

        if attentions is not None and len(attentions) > 0:
            # Use last layer, first head, averaged over batch
            attn = attentions[-1][0, 0].cpu().numpy()
            im = ax.imshow(attn, cmap="Blues", aspect="auto")
            ax.set_title(f"{variant}")
            ax.set_xlabel("Key Position")
            ax.set_ylabel("Query Position")
            plt.colorbar(im, ax=ax, fraction=0.046)
        else:
            ax.text(0.5, 0.5, "No attention\nweights", ha="center", va="center")
            ax.set_title(f"{variant}")

    fig.suptitle("Attention Patterns Comparison (Layer -1, Head 0)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_attention_entropy_comparison(
    models: dict[str, torch.nn.Module],
    dataset,
    device: str,
    path: Path,
    sample_text: str = "To be or not to be, that is the question.",
) -> None:
    """Bar chart comparing attention entropy across variants."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tokens = dataset.encode(sample_text).unsqueeze(0).to(device)
    entropies = {}

    for variant, model in models.items():
        model.eval()
        with torch.no_grad():
            _, _, attentions = model(tokens, return_attn=True)

        if attentions is not None and len(attentions) > 0:
            # Compute entropy: -sum(p * log(p))
            eps = 1e-10
            all_entropy = []
            for layer_attn in attentions:
                attn = layer_attn.clamp(min=eps)
                entropy = -torch.sum(attn * torch.log(attn), dim=-1)
                all_entropy.append(entropy.mean().item())
            entropies[variant] = np.mean(all_entropy)
        else:
            entropies[variant] = 0.0

    fig, ax = plt.subplots(figsize=(8, 5))
    variants = list(entropies.keys())
    values = [entropies[v] for v in variants]
    colors = plt.cm.Set2(np.linspace(0, 1, len(variants)))

    bars = ax.bar(variants, values, color=colors)
    ax.set_ylabel("Average Attention Entropy")
    ax.set_title("Attention Entropy Comparison\n(Higher = more distributed attention)")

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{val:.2f}", ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_attention_distance_comparison(
    models: dict[str, torch.nn.Module],
    dataset,
    device: str,
    path: Path,
    sample_text: str = "To be or not to be, that is the question.",
) -> None:
    """Bar chart comparing mean attention distance and local ratio."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tokens = dataset.encode(sample_text).unsqueeze(0).to(device)
    seq_len = tokens.shape[1]

    metrics = {variant: {"mean_dist": 0.0, "local_ratio": 0.0} for variant in models}

    for variant, model in models.items():
        model.eval()
        with torch.no_grad():
            _, _, attentions = model(tokens, return_attn=True)

        if attentions is not None and len(attentions) > 0:
            all_mean_dist = []
            all_local_ratio = []

            for layer_attn in attentions:
                attn = layer_attn[0]  # (n_head, seq_len, seq_len)
                T = attn.shape[-1]

                # Distance matrix
                positions = torch.arange(T, device=device)
                distances = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs().float()
                mask = torch.tril(torch.ones(T, T, device=device))
                distances = distances * mask

                # Mean distance
                mean_dist = (attn * distances.unsqueeze(0)).sum(dim=-1).mean()
                all_mean_dist.append(mean_dist.item())

                # Local ratio (within 5 positions)
                local_mask = (distances <= 5) & (mask > 0)
                local_attn = (attn * local_mask.unsqueeze(0)).sum(dim=-1).mean()
                all_local_ratio.append(local_attn.item())

            metrics[variant]["mean_dist"] = np.mean(all_mean_dist)
            metrics[variant]["local_ratio"] = np.mean(all_local_ratio)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    variants = list(metrics.keys())
    colors = plt.cm.Set2(np.linspace(0, 1, len(variants)))

    # Mean distance
    ax = axes[0]
    values = [metrics[v]["mean_dist"] for v in variants]
    bars = ax.bar(variants, values, color=colors)
    ax.set_ylabel("Mean Attention Distance")
    ax.set_title("Mean Attention Distance\n(Higher = looks further back)")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.2f}", ha="center", va="bottom", fontsize=10)

    # Local ratio
    ax = axes[1]
    values = [metrics[v]["local_ratio"] for v in variants]
    bars = ax.bar(variants, values, color=colors)
    ax.set_ylabel("Local Attention Ratio")
    ax.set_title("Local Attention Ratio\n(Higher = more local focus)")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=10)

    fig.suptitle("Attention Distance Metrics Comparison", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_attention_mask_patterns(variants: list[str], block_size: int, path: Path) -> None:
    """Visualize the theoretical attention mask pattern for each variant."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    display_size = min(32, block_size)
    n_variants = len(variants)
    fig, axes = plt.subplots(1, n_variants, figsize=(5 * n_variants, 5))
    if n_variants == 1:
        axes = [axes]

    for ax, variant in zip(axes, variants):
        # Create a mask pattern based on the variant
        mask = torch.zeros(display_size, display_size)

        if variant == "vanilla" or variant == "rotary":
            # Full causal mask
            mask = torch.tril(torch.ones(display_size, display_size))
        elif variant == "linear":
            # Linear attention also uses causal structure
            mask = torch.tril(torch.ones(display_size, display_size))
        elif variant == "sliding_window":
            window_size = 8
            for i in range(display_size):
                start = max(0, i - window_size + 1)
                mask[i, start:i + 1] = 1.0
        elif variant == "sparse":
            local_size = 4
            stride = 4
            for i in range(display_size):
                local_start = max(0, i - local_size + 1)
                mask[i, local_start:i + 1] = 1.0
                for j in range(0, i + 1, stride):
                    mask[i, j] = 1.0

        im = ax.imshow(mask.numpy(), cmap="Blues", aspect="auto", vmin=0, vmax=1)
        ax.set_title(f"{variant}")
        ax.set_xlabel("Key Position")
        ax.set_ylabel("Query Position")

    fig.suptitle("Attention Mask Patterns (Theoretical)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(config: ExperimentConfig) -> str:
    """Get device from config or auto-detect."""
    if config.device:
        return config.device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _try_load_from_cache(params: dict, cache_dir: str | None = None):
    """Try to load the Shakespeare dataset from cache. Returns cached data or None."""
    import os

    old_env = os.environ.get("ATTENTION_LAB_CACHE")
    if cache_dir:
        os.environ["ATTENTION_LAB_CACHE"] = cache_dir

    try:
        cache_params = {k: v for k, v in params.items()}
        cached_data = load_from_cache("shakespeare", **cache_params)
        if cached_data is not None:
            return cached_data
    finally:
        if cache_dir:
            if old_env is not None:
                os.environ["ATTENTION_LAB_CACHE"] = old_env
            else:
                os.environ.pop("ATTENTION_LAB_CACHE", None)

    return None


def _reconstruct_from_cache(cached_data: dict, block_size: int,
                            max_samples: int | None = None):
    """Reconstruct a ShakespeareDataset from cached data."""
    dataset = ShakespeareDataset.__new__(ShakespeareDataset)
    dataset.data = cached_data["data"]
    dataset.tokenizer = cached_data["tokenizer"]
    dataset.block_size = cached_data.get("block_size", block_size)
    dataset.vocab_size = cached_data["vocab_size"]
    dataset._max_samples = max_samples
    return dataset


def create_dataset_from_config(ds_cfg, block_size: int,
                               use_cache: bool = False, cache_dir: str | None = None):
    """Create the Shakespeare dataset from configuration.

    When use_cache is True, tries to load from cache first.
    Falls back to downloading if not cached.
    """
    params = ds_cfg.params.copy()

    if "block_size" not in params:
        params["block_size"] = block_size

    # Try cache first if enabled
    if use_cache:
        cached_data = _try_load_from_cache(params, cache_dir)
        if cached_data is not None:
            dataset = _reconstruct_from_cache(
                cached_data, block_size,
                max_samples=params.get("max_samples"),
            )
            if dataset is not None:
                print(f"  Loaded shakespeare from cache")
                return dataset
            print(f"  Cache hit but reconstruction failed, downloading fresh")
        else:
            print(f"  No cache found, downloading fresh")

    return ShakespeareDataset(**params)


def create_datasets(config: ExperimentConfig):
    """Create dataset from configuration."""
    data_cfg = config.data
    block_size = config.model.block_size

    if not data_cfg.datasets:
        # Default: Shakespeare
        return ShakespeareDataset(block_size=block_size)

    ds_cfg = data_cfg.datasets[0]
    ds = create_dataset_from_config(
        ds_cfg, block_size,
        use_cache=data_cfg.use_cache,
        cache_dir=data_cfg.cache_dir,
    )
    print(f"  Created shakespeare: {len(ds)} samples")
    return ds


def create_model(config: ExperimentConfig, vocab_size: int):
    """Create model from configuration."""
    attn_kwargs = {}

    if config.attention.type == "sliding_window":
        attn_kwargs["window_size"] = config.attention.window_size
    elif config.attention.type == "sparse":
        attn_kwargs["local_size"] = config.attention.local_size
        attn_kwargs["stride"] = config.attention.stride

    return create_model_variant(
        attention_type=config.attention.type,
        vocab_size=vocab_size,
        block_size=config.model.block_size,
        n_layer=config.model.n_layer,
        n_head=config.model.n_head,
        n_embd=config.model.n_embd,
        dropout=config.model.dropout,
        **attn_kwargs,
    )


def train_single(config: ExperimentConfig, dataset, device: str,
                  resume_from: str | None = None) -> dict:
    """Train a single model, optionally resuming from a checkpoint."""
    train_loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    vocab_size = config.model.vocab_size
    if vocab_size is None:
        vocab_size = dataset.vocab_size

    model = create_model(config, vocab_size)

    trainer_config = TrainerConfig(
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        grad_clip=config.training.grad_clip,
        warmup_steps=config.training.warmup_steps,
        max_steps=config.training.max_steps,
        eval_interval=config.training.eval_interval,
        eval_steps=config.training.eval_steps,
        checkpoint_dir=config.training.checkpoint_dir,
        log_interval=config.training.log_interval,
        compile=config.training.compile,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        config=trainer_config,
        device=device,
    )

    if resume_from:
        print(f"\nResuming from checkpoint: {resume_from}")
        trainer.load_checkpoint(resume_from)
        print(f"  Resuming at step {trainer.step}")

    print(f"\nTraining on device: {device}")
    history = trainer.train()

    return {
        "model": model,
        "trainer": trainer,
        "history": history,
        "final_loss": history["train_loss"][-1] if history["train_loss"] else float("inf"),
    }


def compare_variants(config: ExperimentConfig, dataset, device: str,
                     run_dir: Path) -> dict:
    """Train and compare multiple attention variants."""
    variants = config.compare_variants or ["vanilla", "linear", "sparse"]
    results = {}
    all_histories: dict[str, dict] = {}

    for variant in variants:
        print(f"\n{'=' * 60}")
        print(f"Training variant: {variant}")
        print("=" * 60)

        variant_config = copy.deepcopy(config)
        variant_config.attention.type = variant
        variant_config.name = f"{config.name}_{variant}" if config.name else variant

        variant_dir = run_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        variant_config.training.checkpoint_dir = str(variant_dir / "checkpoints")

        result = train_single(variant_config, dataset, device)
        results[variant] = result
        history = result["history"]
        all_histories[variant] = history

        # Per-variant artefacts
        save_history(history, variant_dir / "history.json")
        plot_loss_curve(history, variant_dir / "loss_curve.png",
                        title=f"{variant} — Loss Curve")

    # Comparison summary
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)

    print(f"\n{'Variant':<20} {'Final Loss':<15}")
    print("-" * 35)
    for variant in sorted(results.keys(), key=lambda x: results[x]["final_loss"]):
        print(f"{variant:<20} {results[variant]['final_loss']:.4f}")

    # Comparison-level artefacts
    comparison_data = {
        v: {"final_loss": results[v]["final_loss"]} for v in results
    }
    with open(run_dir / "comparison_results.json", "w") as f:
        json.dump(comparison_data, f, indent=2)

    plot_comparison_curves(all_histories, run_dir / "loss_curves.png")
    final_losses = {v: results[v]["final_loss"] for v in results}
    plot_final_loss_bar(final_losses, run_dir / "final_loss_bar.png")

    # Attention-specific comparison plots
    print("\nGenerating attention comparison plots...")
    models = {v: results[v]["model"] for v in results}

    # Attention patterns heatmaps
    plot_attention_patterns_comparison(
        models, dataset, device, run_dir / "attention_patterns.png"
    )
    print("  - Saved attention_patterns.png")

    # Attention entropy comparison
    plot_attention_entropy_comparison(
        models, dataset, device, run_dir / "attention_entropy.png"
    )
    print("  - Saved attention_entropy.png")

    # Attention distance metrics
    plot_attention_distance_comparison(
        models, dataset, device, run_dir / "attention_distance.png"
    )
    print("  - Saved attention_distance.png")

    # Theoretical mask patterns
    plot_attention_mask_patterns(
        variants, config.model.block_size, run_dir / "attention_masks.png"
    )
    print("  - Saved attention_masks.png")

    return results


def test_generation(model, dataset, config: ExperimentConfig, device: str) -> None:
    """Test model generation."""
    print("\n" + "=" * 60)
    print("GENERATION TEST")
    print("=" * 60)

    model.eval()

    test_prompts = ["ROMEO:", "To be or"]

    for prompt in test_prompts[:5]:
        try:
            tokens = dataset.encode(prompt).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model.generate(
                    tokens,
                    max_new_tokens=config.generation.max_new_tokens,
                    temperature=config.generation.temperature,
                    top_k=config.generation.top_k,
                )
            result = dataset.decode(output[0].tolist())
            display = result[:60] + "..." if len(result) > 60 else result
            print(f"  {prompt} -> {display[len(prompt):]}")
        except Exception as e:
            print(f"  {prompt} -> (error: {e})")


def main():
    parser = argparse.ArgumentParser(
        description="Train GPT models using YAML configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (default: config/default.yaml)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare attention variants defined in config",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume training from checkpoint",
    )
    parser.add_argument(
        "--test-only",
        type=str,
        default=None,
        help="Load checkpoint and test generation only",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config values (e.g., --override training.max_steps=1000)",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Print configuration
    print_config(config)

    # Set seed
    set_seed(config.seed)

    # Get device
    device = get_device(config)
    print(f"Using device: {device}")

    # Create datasets
    print("\nCreating datasets...")
    dataset = create_datasets(config)
    print(f"Total samples: {len(dataset)}")
    print(f"Vocab size: {dataset.vocab_size}")

    # Test-only mode
    if args.test_only:
        print(f"\nLoading checkpoint: {args.test_only}")
        checkpoint = torch.load(args.test_only, map_location=device)

        vocab_size = config.model.vocab_size or dataset.vocab_size
        model = create_model(config, vocab_size)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

        test_generation(model, dataset, config, device)
        return

    # Training mode
    if args.compare or config.compare_variants:
        run_dir = create_run_dir(config.output_dir,
                                 config.name or "full_comparison")
        print(f"\nRun directory: {run_dir}")

        # Save top-level config
        with open(run_dir / "config.json", "w") as f:
            json.dump(config_to_dict(config), f, indent=2)

        results = compare_variants(config, dataset, device, run_dir)

        best_variant = min(results.keys(), key=lambda x: results[x]["final_loss"])
        print(f"\nTesting best variant: {best_variant}")
        test_generation(results[best_variant]["model"], dataset, config, device)
    else:
        run_dir = create_run_dir(config.output_dir, config.name)
        print(f"\nRun directory: {run_dir}")

        # Point checkpoints into the run directory
        config.training.checkpoint_dir = str(run_dir / "checkpoints")

        result = train_single(config, dataset, device, resume_from=args.resume)
        test_generation(result["model"], dataset, config, device)

        # Save artefacts
        with open(run_dir / "config.json", "w") as f:
            json.dump(config_to_dict(config), f, indent=2)
        save_history(result["history"], run_dir / "history.json")
        plot_loss_curve(result["history"], run_dir / "loss_curve.png")

    print(f"\nAll outputs saved to: {run_dir}")
    print("Training complete!")


if __name__ == "__main__":
    main()

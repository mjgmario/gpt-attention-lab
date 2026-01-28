#!/usr/bin/env python
"""Compare attention heads across layers."""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt
import numpy as np
import torch
from attention_lab.config import GPTConfig
from attention_lab.data.shakespeare import ShakespeareDataset
from attention_lab.generate import get_attention_for_text
from attention_lab.inspect.analyze import compare_heads, print_head_summary
from attention_lab.model import GPT


def plot_head_comparison(
    metrics: dict[str, np.ndarray],
    save_path: str | None = None,
    figsize: tuple[int, int] = (14, 10),
) -> plt.Figure:
    """Create visualization comparing all heads.

    Args:
        metrics: Dictionary from compare_heads().
        save_path: If provided, save figure to this path.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    n_layers, n_heads = metrics["entropy"].shape

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    metric_names = [
        ("entropy", "Attention Entropy", "Higher = more distributed"),
        ("mean_distance", "Mean Attention Distance", "Higher = longer range"),
        ("local_ratio", "Local Attention Ratio", "Higher = more local"),
        ("first_token_attn", "First Token Attention", "BOS-like behavior"),
        ("prev_token_attn", "Previous Token Attention", "Local/copying behavior"),
    ]

    for ax, (metric_key, title, description) in zip(axes, metric_names, strict=False):
        data = metrics[metric_key]

        im = ax.imshow(data, aspect="auto", cmap="viridis")
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        ax.set_title(f"{title}\n({description})")
        ax.set_xticks(range(n_heads))
        ax.set_yticks(range(n_layers))

        # Add value annotations
        for i in range(n_layers):
            for j in range(n_heads):
                ax.text(
                    j,
                    i,
                    f"{data[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if data[i, j] < data.mean() else "black",
                    fontsize=8,
                )

        plt.colorbar(im, ax=ax)

    # Hide last subplot
    axes[-1].axis("off")

    plt.suptitle("Attention Head Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare attention heads across layers")

    # Model loading
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")

    # Input text
    parser.add_argument(
        "--text",
        type=str,
        default="To be or not to be, that is the question.",
        help="Text to analyze",
    )

    # Output
    parser.add_argument(
        "--output_dir", type=str, default="outputs/analysis", help="Directory to save outputs"
    )
    parser.add_argument("--show", action="store_true", help="Show plots interactively")

    # Model config (if no checkpoint)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_embd", type=int, default=128)
    parser.add_argument("--block_size", type=int, default=128)

    # Device
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    print("=" * 60)
    print("Attention Head Comparison")
    print("=" * 60)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load or create model
    if args.checkpoint is not None:
        print(f"\nLoading checkpoint from: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        config = checkpoint.get("config", GPTConfig())
        model = GPT(config)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print("\nUsing randomly initialized model")
        config = GPTConfig(
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            block_size=args.block_size,
        )
        model = GPT(config)

    model.to(args.device)
    model.eval()

    # Create tokenizer
    dataset = ShakespeareDataset(block_size=config.block_size)

    # Encode text
    print(f"\nInput text: {args.text}")
    tokens = dataset.encode(args.text)
    print(f"Sequence length: {len(tokens)}")

    # Get attention weights
    print("\nComputing attention weights...")
    attentions = get_attention_for_text(model, tokens, device=args.device)

    # Compute comparison metrics
    print("\nAnalyzing attention heads...")
    metrics = compare_heads(attentions, tokens)

    # Print summary
    print_head_summary(metrics)

    # Create visualization
    print("\nGenerating comparison visualization...")
    save_path = output_dir / "head_comparison.png"
    fig = plot_head_comparison(metrics, save_path=str(save_path))

    print(f"Saved to: {save_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)

    # Save metrics to file
    metrics_path = output_dir / "head_metrics.npz"
    np.savez(metrics_path, **metrics)
    print(f"Metrics saved to: {metrics_path}")

    # Generate report
    report_path = output_dir / "head_report.txt"
    with open(report_path, "w") as f:
        f.write("ATTENTION HEAD ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: {config}\n")
        f.write(f"Input text: {args.text}\n")
        f.write(f"Sequence length: {len(tokens)}\n\n")

        f.write("METRICS SUMMARY\n")
        f.write("-" * 40 + "\n\n")

        for layer in range(config.n_layer):
            f.write(f"Layer {layer}:\n")
            for head in range(config.n_head):
                f.write(f"  Head {head}:\n")
                for metric_name in metrics:
                    f.write(f"    {metric_name}: {metrics[metric_name][layer, head]:.4f}\n")
            f.write("\n")

        # Find interesting heads
        f.write("\nINTERESTING HEADS\n")
        f.write("-" * 40 + "\n\n")

        # Highest entropy (distributed attention)
        max_entropy_idx = np.unravel_index(np.argmax(metrics["entropy"]), metrics["entropy"].shape)
        f.write(
            f"Most distributed (highest entropy): Layer {max_entropy_idx[0]}, Head {max_entropy_idx[1]}\n"
        )

        # Highest first token attention (BOS-like)
        max_bos_idx = np.unravel_index(
            np.argmax(metrics["first_token_attn"]), metrics["first_token_attn"].shape
        )
        f.write(f"Highest BOS attention: Layer {max_bos_idx[0]}, Head {max_bos_idx[1]}\n")

        # Highest previous token attention (copying)
        max_prev_idx = np.unravel_index(
            np.argmax(metrics["prev_token_attn"]), metrics["prev_token_attn"].shape
        )
        f.write(
            f"Highest previous token attention: Layer {max_prev_idx[0]}, Head {max_prev_idx[1]}\n"
        )

    print(f"Report saved to: {report_path}")
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()

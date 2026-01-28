#!/usr/bin/env python
"""Ablation study: evaluate effect of removing layers or heads."""

import argparse
import copy
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt
import numpy as np
import torch
from attention_lab.config import GPTConfig
from attention_lab.data.shakespeare import ShakespeareDataset
from attention_lab.model import GPT
from torch.utils.data import DataLoader


def evaluate_perplexity(
    model: GPT,
    dataloader: DataLoader,
    device: str,
    max_batches: int = 50,
) -> float:
    """Evaluate model perplexity on dataset.

    Args:
        model: GPT model to evaluate.
        dataloader: DataLoader with evaluation data.
        device: Device to run on.
        max_batches: Maximum batches to evaluate.

    Returns:
        Perplexity (exp of average loss).
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            if i >= max_batches:
                break

            x, y = x.to(device), y.to(device)
            _, loss, _ = model(x, targets=y)
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    return np.exp(avg_loss)


def ablate_layer(model: GPT, layer_idx: int) -> GPT:
    """Create a copy of model with one layer zeroed out.

    Args:
        model: Original model.
        layer_idx: Index of layer to ablate.

    Returns:
        New model with layer ablated.
    """
    ablated_model = copy.deepcopy(model)

    # Zero out the layer's contribution
    with torch.no_grad():
        for param in ablated_model.transformer["h"][layer_idx].parameters():
            param.zero_()

    return ablated_model


def ablate_head(model: GPT, layer_idx: int, head_idx: int) -> GPT:
    """Create a copy of model with one attention head zeroed out.

    Args:
        model: Original model.
        layer_idx: Layer containing the head.
        head_idx: Index of head to ablate.

    Returns:
        New model with head ablated.
    """
    ablated_model = copy.deepcopy(model)

    # Get dimensions
    n_embd = model.config.n_embd
    n_head = model.config.n_head
    head_dim = n_embd // n_head

    with torch.no_grad():
        attn = ablated_model.transformer["h"][layer_idx].attn

        # Zero out the query, key, value weights for this head
        # c_attn projects to 3 * n_embd (Q, K, V concatenated)
        for offset in [0, n_embd, 2 * n_embd]:  # Q, K, V
            start = offset + head_idx * head_dim
            end = offset + (head_idx + 1) * head_dim
            attn.c_attn.weight[:, start:end] = 0
            if attn.c_attn.bias is not None:
                attn.c_attn.bias[start:end] = 0

    return ablated_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation study on GPT model")

    # Model loading
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")

    # Ablation options
    parser.add_argument("--ablate_layers", action="store_true", help="Run layer ablation study")
    parser.add_argument("--ablate_heads", action="store_true", help="Run head ablation study")

    # Evaluation
    parser.add_argument("--max_batches", type=int, default=50, help="Max batches for evaluation")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")

    # Output
    parser.add_argument(
        "--output_dir", type=str, default="outputs/ablation", help="Directory to save results"
    )
    parser.add_argument("--show", action="store_true", help="Show plots interactively")

    # Device
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    if not args.ablate_layers and not args.ablate_heads:
        args.ablate_layers = True
        args.ablate_heads = True

    print("=" * 60)
    print("GPT Ablation Study")
    print("=" * 60)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"\nLoading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    config = checkpoint.get("config", GPTConfig())

    model = GPT(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(args.device)
    model.eval()

    print(f"Model config: {config}")

    # Load evaluation data
    print("\nLoading evaluation data...")
    dataset = ShakespeareDataset(
        block_size=config.block_size,
        split="val",
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # Baseline perplexity
    print("\nEvaluating baseline model...")
    baseline_ppl = evaluate_perplexity(model, dataloader, args.device, args.max_batches)
    print(f"Baseline perplexity: {baseline_ppl:.2f}")

    results = {"baseline": baseline_ppl}

    # Layer ablation
    if args.ablate_layers:
        print("\n" + "=" * 60)
        print("LAYER ABLATION STUDY")
        print("=" * 60)

        layer_ppls = []
        for layer_idx in range(config.n_layer):
            print(f"\nAblating layer {layer_idx}...")
            ablated_model = ablate_layer(model, layer_idx)
            ablated_model.to(args.device)

            ppl = evaluate_perplexity(ablated_model, dataloader, args.device, args.max_batches)
            layer_ppls.append(ppl)
            print(f"  Perplexity: {ppl:.2f} (Δ = {ppl - baseline_ppl:+.2f})")

            del ablated_model

        results["layer_ablation"] = layer_ppls

        # Plot layer ablation results
        fig, ax = plt.subplots(figsize=(10, 6))

        x = range(config.n_layer)
        ax.bar(x, layer_ppls, color="steelblue", alpha=0.7)
        ax.axhline(
            y=baseline_ppl, color="red", linestyle="--", label=f"Baseline ({baseline_ppl:.2f})"
        )

        ax.set_xlabel("Layer Index")
        ax.set_ylabel("Perplexity")
        ax.set_title("Layer Ablation Study: Effect of Removing Each Layer")
        ax.set_xticks(x)
        ax.legend()

        for i, ppl in enumerate(layer_ppls):
            ax.text(i, ppl + 0.5, f"{ppl:.1f}", ha="center", fontsize=9)

        plt.tight_layout()
        save_path = output_dir / "layer_ablation.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nSaved to: {save_path}")

        if args.show:
            plt.show()
        else:
            plt.close(fig)

    # Head ablation
    if args.ablate_heads:
        print("\n" + "=" * 60)
        print("HEAD ABLATION STUDY")
        print("=" * 60)

        head_ppls = np.zeros((config.n_layer, config.n_head))

        for layer_idx in range(config.n_layer):
            for head_idx in range(config.n_head):
                print(f"\nAblating layer {layer_idx}, head {head_idx}...")
                ablated_model = ablate_head(model, layer_idx, head_idx)
                ablated_model.to(args.device)

                ppl = evaluate_perplexity(ablated_model, dataloader, args.device, args.max_batches)
                head_ppls[layer_idx, head_idx] = ppl
                print(f"  Perplexity: {ppl:.2f} (Δ = {ppl - baseline_ppl:+.2f})")

                del ablated_model

        results["head_ablation"] = head_ppls.tolist()

        # Plot head ablation results
        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(head_ppls, cmap="Reds", aspect="auto")
        ax.set_xlabel("Head Index")
        ax.set_ylabel("Layer Index")
        ax.set_title(
            f"Head Ablation Study: Perplexity After Removing Each Head\n(Baseline: {baseline_ppl:.2f})"
        )
        ax.set_xticks(range(config.n_head))
        ax.set_yticks(range(config.n_layer))

        # Add value annotations
        for i in range(config.n_layer):
            for j in range(config.n_head):
                delta = head_ppls[i, j] - baseline_ppl
                ax.text(
                    j,
                    i,
                    f"{delta:+.1f}",
                    ha="center",
                    va="center",
                    color="white" if head_ppls[i, j] > np.median(head_ppls) else "black",
                    fontsize=9,
                )

        plt.colorbar(im, ax=ax, label="Perplexity")
        plt.tight_layout()

        save_path = output_dir / "head_ablation.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nSaved to: {save_path}")

        if args.show:
            plt.show()
        else:
            plt.close(fig)

        # Find most important heads
        print("\n" + "=" * 60)
        print("MOST IMPORTANT HEADS")
        print("=" * 60)

        deltas = head_ppls - baseline_ppl
        flat_indices = np.argsort(deltas.flatten())[::-1]

        print("\nHeads with largest impact when removed:")
        for rank, flat_idx in enumerate(flat_indices[:5]):
            layer = flat_idx // config.n_head
            head = flat_idx % config.n_head
            delta = deltas[layer, head]
            print(f"  {rank + 1}. Layer {layer}, Head {head}: Δ = {delta:+.2f}")

    # Save results
    results_path = output_dir / "ablation_results.npz"
    np.savez(results_path, **{k: np.array(v) for k, v in results.items()})
    print(f"\nResults saved to: {results_path}")

    # Generate report
    report_path = output_dir / "ablation_report.txt"
    with open(report_path, "w") as f:
        f.write("ABLATION STUDY REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: {config}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Baseline perplexity: {baseline_ppl:.4f}\n\n")

        if "layer_ablation" in results:
            f.write("LAYER ABLATION\n")
            f.write("-" * 40 + "\n")
            for i, ppl in enumerate(results["layer_ablation"]):
                f.write(f"Layer {i}: {ppl:.4f} (Δ = {ppl - baseline_ppl:+.4f})\n")
            f.write("\n")

        if "head_ablation" in results:
            f.write("HEAD ABLATION\n")
            f.write("-" * 40 + "\n")
            head_ppls = np.array(results["head_ablation"])
            for layer in range(config.n_layer):
                for head in range(config.n_head):
                    ppl = head_ppls[layer, head]
                    f.write(
                        f"Layer {layer}, Head {head}: {ppl:.4f} (Δ = {ppl - baseline_ppl:+.4f})\n"
                    )

    print(f"Report saved to: {report_path}")
    print("\nAblation study complete!")


if __name__ == "__main__":
    main()

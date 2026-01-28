#!/usr/bin/env python
"""Visualize attention patterns from a GPT model."""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from attention_lab.config import GPTConfig
from attention_lab.data.shakespeare import ShakespeareDataset
from attention_lab.generate import get_attention_for_text
from attention_lab.inspect.visualize import (
    plot_all_heads,
    plot_attention_heatmap,
    plot_attention_pattern,
)
from attention_lab.model import GPT


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize attention patterns")

    # Model loading
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (uses random model if not provided)")

    # Input text
    parser.add_argument("--text", type=str, default="Hello world!",
                        help="Text to analyze attention for")

    # Visualization options
    parser.add_argument("--layer", type=int, default=0,
                        help="Layer to visualize (default: 0)")
    parser.add_argument("--head", type=int, default=0,
                        help="Head to visualize (default: 0)")
    parser.add_argument("--all_heads", action="store_true",
                        help="Plot all heads in a grid")
    parser.add_argument("--pattern", action="store_true",
                        help="Use causal pattern visualization")

    # Output
    parser.add_argument("--output_dir", type=str, default="outputs/attention",
                        help="Directory to save visualizations")
    parser.add_argument("--show", action="store_true",
                        help="Show plots interactively")

    # Model config (if no checkpoint)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_embd", type=int, default=128)
    parser.add_argument("--block_size", type=int, default=128)

    # Device
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    print("=" * 60)
    print("Attention Visualization")
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
    token_chars = list(args.text[: len(tokens)])  # Character labels

    print(f"Tokens: {tokens.tolist()}")
    print(f"Characters: {token_chars}")

    # Get attention weights
    print("\nComputing attention weights...")
    attentions = get_attention_for_text(model, tokens, device=args.device)

    print(f"Number of layers: {len(attentions)}")
    print(f"Attention shape: {attentions[0].shape}")

    # Generate visualizations
    if args.all_heads:
        # Plot all heads for the specified layer
        print(f"\nPlotting all heads for layer {args.layer}...")
        save_path = output_dir / f"all_heads_layer{args.layer}.png"

        fig = plot_all_heads(
            attentions,
            layer=args.layer,
            tokens=token_chars,
            save_path=str(save_path),
        )
        print(f"Saved to: {save_path}")

        if args.show:
            import matplotlib.pyplot as plt
            plt.show()
        else:
            plt.close(fig)

    else:
        # Plot single head
        print(f"\nPlotting layer {args.layer}, head {args.head}...")

        if args.pattern:
            save_path = output_dir / f"pattern_L{args.layer}_H{args.head}.png"
            fig = plot_attention_pattern(
                attentions,
                layer=args.layer,
                head=args.head,
                tokens=token_chars,
                save_path=str(save_path),
            )
        else:
            save_path = output_dir / f"heatmap_L{args.layer}_H{args.head}.png"
            fig = plot_attention_heatmap(
                attentions,
                layer=args.layer,
                head=args.head,
                tokens=token_chars,
                save_path=str(save_path),
            )

        print(f"Saved to: {save_path}")

        if args.show:
            import matplotlib.pyplot as plt
            plt.show()
        else:
            import matplotlib.pyplot as plt
            plt.close(fig)

    # Print attention statistics
    print("\n" + "=" * 60)
    print("Attention Statistics")
    print("=" * 60)

    from attention_lab.inspect.analyze import head_entropy, attention_to_positions

    for layer_idx, attn in enumerate(attentions):
        print(f"\nLayer {layer_idx}:")
        entropy = head_entropy(attn)
        pos_metrics = attention_to_positions(attn)

        for h in range(config.n_head):
            print(f"  Head {h}: entropy={entropy[h]:.3f}, "
                  f"first_tok_attn={pos_metrics['first_token_attention'][h]:.3f}, "
                  f"prev_tok_attn={pos_metrics['prev_token_attention'][h]:.3f}")

    print(f"\nVisualizations saved to: {output_dir}/")


if __name__ == "__main__":
    main()

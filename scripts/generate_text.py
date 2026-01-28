#!/usr/bin/env python
"""Generate text from a trained GPT model."""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from attention_lab.config import GPTConfig
from attention_lab.data.shakespeare import ShakespeareDataset
from attention_lab.model import GPT


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from trained GPT model")

    # Model loading
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/shakespeare/best_model.pt",
        help="Path to model checkpoint",
    )

    # Generation arguments
    parser.add_argument(
        "--prompt", type=str, default="To be or not to be", help="Text prompt to continue from"
    )
    parser.add_argument(
        "--max_tokens", type=int, default=200, help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Sampling temperature (higher = more random)"
    )
    parser.add_argument(
        "--top_k", type=int, default=None, help="Top-k sampling (only sample from top k tokens)"
    )
    parser.add_argument(
        "--top_p", type=float, default=None, help="Top-p (nucleus) sampling threshold"
    )
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate")

    # Device
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/mps/cpu)")

    args = parser.parse_args()

    # Auto-detect device
    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print("=" * 60)
    print("GPT Text Generation")
    print("=" * 60)

    # Load checkpoint
    print(f"\nLoading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Get config from checkpoint
    config = checkpoint.get("config")
    if config is None:
        print("Warning: No config found in checkpoint, using defaults")
        config = GPTConfig()

    print(f"Model config: {config}")

    # Create model and load weights
    model = GPT(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Create tokenizer from Shakespeare dataset
    # (we need a compatible tokenizer for encoding/decoding)
    dataset = ShakespeareDataset(block_size=config.block_size)

    print(f"\nPrompt: {args.prompt}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-k: {args.top_k}")
    print(f"Top-p: {args.top_p}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Device: {device}")

    # Encode prompt
    prompt_tokens = dataset.encode(args.prompt).unsqueeze(0).to(device)

    print("\n" + "=" * 60)
    print("Generated Text")
    print("=" * 60)

    for i in range(args.num_samples):
        if args.num_samples > 1:
            print(f"\n--- Sample {i + 1} ---")

        with torch.no_grad():
            generated = model.generate(
                prompt_tokens,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )

        generated_text = dataset.decode(generated[0].tolist())
        print(f"\n{generated_text}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

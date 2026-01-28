#!/usr/bin/env python
"""Download and prepare the Shakespeare dataset for training.

This script downloads and caches the dataset so it's ready for training.
Run this before training to avoid download delays during experiments.

Usage:
    # Prepare Shakespeare dataset
    uv run python scripts/prepare_data.py

    # List cached datasets
    uv run python scripts/prepare_data.py --list

    # Clear cache
    uv run python scripts/prepare_data.py --clear

    # Force re-download
    uv run python scripts/prepare_data.py --force
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from attention_lab.data.cache import (
    clear_cache,
    print_cache_info,
    save_to_cache,
    is_cached,
    get_cache_path,
)
from attention_lab.data.shakespeare import ShakespeareDataset, download_shakespeare


DEFAULT_CONFIG = {"block_size": 128}


def prepare_shakespeare(force: bool = False) -> None:
    """Download and prepare Shakespeare dataset."""
    print("\n" + "=" * 50)
    print("Preparing Shakespeare dataset...")
    print("=" * 50)

    # Download raw text
    filepath = download_shakespeare()
    print(f"Raw text downloaded to: {filepath}")

    # Check if already cached
    params = DEFAULT_CONFIG
    if is_cached("shakespeare", **params) and not force:
        print("Dataset already cached. Use --force to regenerate.")
        return

    # Create and cache dataset
    print("Processing dataset...")
    dataset = ShakespeareDataset(**params)

    # Save to cache
    cache_data = {
        "data": dataset.data,
        "tokenizer": dataset.tokenizer,
        "block_size": dataset.block_size,
        "vocab_size": dataset.vocab_size,
    }
    save_to_cache("shakespeare", cache_data, **params)

    print(f"Samples: {len(dataset)}")
    print(f"Vocab size: {dataset.vocab_size}")


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare the Shakespeare dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List cached datasets",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear all cached datasets",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration even if cached",
    )

    args = parser.parse_args()

    if args.list:
        print_cache_info()
        return

    if args.clear:
        count = clear_cache()
        print(f"Cleared {count} cached datasets.")
        return

    print("=" * 60)
    print("DATASET PREPARATION")
    print("=" * 60)

    prepare_shakespeare(args.force)

    print("\n" + "=" * 60)
    print("PREPARATION COMPLETE")
    print("=" * 60)
    print_cache_info()


if __name__ == "__main__":
    main()

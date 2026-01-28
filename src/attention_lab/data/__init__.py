"""Data loading utilities for Attention Lab.

This module provides datasets for training and evaluating transformers:

Datasets:
    :class:`ShakespeareDataset`: Character-level language modeling

Utilities:
    :class:`CharTokenizer`: Character-level tokenization
    :func:`is_cached`: Check if dataset is cached
    :func:`load_from_cache`: Load cached dataset
    :func:`save_to_cache`: Save dataset to cache
"""

from attention_lab.data.cache import (
    clear_cache,
    get_cache_dir,
    is_cached,
    list_cached_datasets,
    load_from_cache,
    print_cache_info,
    save_to_cache,
)
from attention_lab.data.shakespeare import ShakespeareDataset, download_shakespeare
from attention_lab.data.tokenizer import CharTokenizer

__all__ = [
    # Tokenizer
    "CharTokenizer",
    # Core datasets
    "ShakespeareDataset",
    "download_shakespeare",
    # Cache utilities
    "is_cached",
    "load_from_cache",
    "save_to_cache",
    "clear_cache",
    "list_cached_datasets",
    "print_cache_info",
    "get_cache_dir",
]

"""Dataset caching and preloading utilities.

This module provides functionality to:
- Cache processed datasets to disk
- Preload cached datasets for faster startup
- Download datasets separately from training
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch

# Default cache directory
DEFAULT_CACHE_DIR = Path("data/.cache")


def get_cache_dir() -> Path:
    """Get the cache directory, creating it if necessary."""
    cache_dir = Path(os.environ.get("ATTENTION_LAB_CACHE", DEFAULT_CACHE_DIR))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_cache_key(dataset_type: str, **params) -> str:
    """Generate a unique cache key for dataset parameters.

    Args:
        dataset_type: Type of dataset (e.g., 'shakespeare')
        **params: Dataset parameters

    Returns:
        MD5 hash string as cache key
    """
    # Sort params for consistent hashing
    param_str = json.dumps({"type": dataset_type, **params}, sort_keys=True)
    return hashlib.md5(param_str.encode()).hexdigest()[:16]


def get_cache_path(dataset_type: str, **params) -> Path:
    """Get the cache file path for a dataset.

    Args:
        dataset_type: Type of dataset
        **params: Dataset parameters

    Returns:
        Path to cache file
    """
    cache_key = get_cache_key(dataset_type, **params)
    return get_cache_dir() / f"{dataset_type}_{cache_key}.pt"


def is_cached(dataset_type: str, **params) -> bool:
    """Check if a dataset is cached.

    Args:
        dataset_type: Type of dataset
        **params: Dataset parameters

    Returns:
        True if cached, False otherwise
    """
    return get_cache_path(dataset_type, **params).exists()


def save_to_cache(
    dataset_type: str,
    data: dict[str, Any],
    **params,
) -> Path:
    """Save dataset data to cache.

    Args:
        dataset_type: Type of dataset
        data: Dictionary containing dataset data (samples, tokenizer, etc.)
        **params: Dataset parameters used to generate cache key

    Returns:
        Path to saved cache file
    """
    cache_path = get_cache_path(dataset_type, **params)

    cache_data = {
        "type": dataset_type,
        "params": params,
        "data": data,
    }

    torch.save(cache_data, cache_path)
    print(f"Cached {dataset_type} dataset to: {cache_path}")

    return cache_path


def load_from_cache(dataset_type: str, **params) -> dict[str, Any] | None:
    """Load dataset data from cache.

    Args:
        dataset_type: Type of dataset
        **params: Dataset parameters

    Returns:
        Cached data dictionary or None if not cached
    """
    cache_path = get_cache_path(dataset_type, **params)

    if not cache_path.exists():
        return None

    try:
        cache_data = torch.load(cache_path, weights_only=False)
        print(f"Loaded {dataset_type} dataset from cache: {cache_path}")
        return cache_data["data"]  # type: ignore[no-any-return]
    except Exception as e:
        print(f"Warning: Failed to load cache {cache_path}: {e}")
        return None


def clear_cache(dataset_type: str | None = None) -> int:
    """Clear cached datasets.

    Args:
        dataset_type: If provided, only clear caches for this type.
                     If None, clear all caches.

    Returns:
        Number of cache files deleted
    """
    cache_dir = get_cache_dir()
    count = 0

    for cache_file in cache_dir.glob("*.pt"):
        if dataset_type is None or cache_file.name.startswith(f"{dataset_type}_"):
            cache_file.unlink()
            count += 1

    print(f"Cleared {count} cache files")
    return count


def list_cached_datasets() -> list[dict[str, Any]]:
    """List all cached datasets.

    Returns:
        List of dictionaries with cache info (type, params, path, size)
    """
    cache_dir = get_cache_dir()
    cached = []

    for cache_file in cache_dir.glob("*.pt"):
        try:
            cache_data = torch.load(cache_file, weights_only=False)
            cached.append(
                {
                    "type": cache_data["type"],
                    "params": cache_data["params"],
                    "path": str(cache_file),
                    "size_mb": cache_file.stat().st_size / (1024 * 1024),
                }
            )
        except Exception:
            # Corrupted cache file
            cached.append(
                {
                    "type": "unknown",
                    "params": {},
                    "path": str(cache_file),
                    "size_mb": cache_file.stat().st_size / (1024 * 1024),
                    "error": "corrupted",
                }
            )

    return cached


def print_cache_info() -> None:
    """Print information about cached datasets."""
    cached = list_cached_datasets()

    if not cached:
        print("No cached datasets found.")
        return

    print("\nCached Datasets:")
    print("-" * 60)

    total_size = 0
    for item in cached:
        size = item["size_mb"]
        total_size += size
        print(f"  {item['type']:<15} {size:.2f} MB  {item['path']}")

    print("-" * 60)
    print(f"Total: {len(cached)} datasets, {total_size:.2f} MB")


class CachedDatasetMixin:
    """Mixin class to add caching support to datasets."""

    @classmethod
    def from_cache(
        cls,
        cache_key: str,
        **fallback_params,
    ):
        """Load dataset from cache or create new if not cached.

        Args:
            cache_key: Unique identifier for this cached dataset
            **fallback_params: Parameters to use if not cached

        Returns:
            Dataset instance
        """
        dataset_type = cls.__name__.lower().replace("dataset", "")

        # Try to load from cache
        cached_data = load_from_cache(dataset_type, cache_key=cache_key)

        if cached_data is not None:
            # Reconstruct from cache
            instance = cls.__new__(cls)
            for key, value in cached_data.items():
                setattr(instance, key, value)
            return instance

        # Create new and cache
        instance = cls(**fallback_params)
        instance.save_to_cache(cache_key)
        return instance

    def save_to_cache(self, cache_key: str) -> Path:
        """Save this dataset to cache.

        Args:
            cache_key: Unique identifier for this cache

        Returns:
            Path to cache file
        """
        dataset_type = self.__class__.__name__.lower().replace("dataset", "")

        # Get cacheable attributes
        data = {}
        for attr in ["samples", "tokenizer", "vocab_size", "block_size"]:
            if hasattr(self, attr):
                data[attr] = getattr(self, attr)

        return save_to_cache(dataset_type, data, cache_key=cache_key)

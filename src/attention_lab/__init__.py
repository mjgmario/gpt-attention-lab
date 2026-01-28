"""Attention Lab - Educational GPT implementation for understanding Transformers.

This package provides tools for studying attention mechanisms in transformer models:

Core Components:
    :class:`GPT`: The main GPT language model
    :class:`GPTConfig`: Model configuration
    :class:`Trainer`: Training loop with checkpointing

Attention Variants:
    :class:`VanillaAttention`: Standard O(n²) attention
    :class:`LinearAttention`: O(n) kernel attention
    :class:`SlidingWindowAttention`: Local attention
    :class:`SparseAttention`: Strided + local attention
    :class:`RotaryAttention`: Rotary position embeddings

Data:
    See :mod:`attention_lab.data` for datasets

Visualization:
    See :mod:`attention_lab.inspect` for attention analysis

Example::

    from attention_lab import GPT, GPTConfig

    config = GPTConfig(vocab_size=256, n_layer=4, n_head=4, n_embd=128)
    model = GPT(config)
    logits, loss, attentions = model(tokens, return_attn=True)
"""

# Attention variants
from attention_lab.attention_variants import (
    ATTENTION_VARIANTS,
    BaseAttention,
    LinearAttention,
    RotaryAttention,
    SlidingWindowAttention,
    SparseAttention,
    VanillaAttention,
    create_attention,
)
from attention_lab.config import GPTConfig

# Config loader
from attention_lab.config_loader import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
    load_config,
)
from attention_lab.generate import generate
from attention_lab.model import GPT, MLP, Block, CausalSelfAttention

# Model variants
from attention_lab.model_variants import (
    GPTVariant,
    GPTVariantConfig,
    create_model_variant,
)
from attention_lab.train import Trainer, TrainerConfig

__all__ = [
    # Config
    "GPTConfig",
    "load_config",
    "ExperimentConfig",
    "ModelConfig",
    "TrainingConfig",
    "DataConfig",
    # Core model
    "GPT",
    "Block",
    "CausalSelfAttention",
    "MLP",
    # Training
    "Trainer",
    "TrainerConfig",
    "generate",
    # Attention variants
    "BaseAttention",
    "VanillaAttention",
    "LinearAttention",
    "SlidingWindowAttention",
    "SparseAttention",
    "RotaryAttention",
    "create_attention",
    "ATTENTION_VARIANTS",
    # Model variants
    "GPTVariant",
    "GPTVariantConfig",
    "create_model_variant",
]

__version__ = "1.0.0"

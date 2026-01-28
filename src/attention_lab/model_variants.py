"""GPT model variants with different attention mechanisms.

This module provides GPT models that can use different attention variants
for comparison and experimentation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

from attention_lab.attention_variants import (
    ATTENTION_VARIANTS,
    BaseAttention,
    create_attention,
)
from attention_lab.config import GPTConfig


@dataclass
class GPTVariantConfig(GPTConfig):
    """Extended config with attention variant selection.

    Attributes:
        attention_type: Which attention variant to use
        attention_kwargs: Additional kwargs for attention (e.g., window_size)
    """

    attention_type: Literal[
        "vanilla", "linear", "sliding_window", "sparse", "rotary"
    ] = "vanilla"
    attention_kwargs: dict = field(default_factory=dict)


class BlockVariant(nn.Module):
    """Transformer block with configurable attention."""

    def __init__(self, config: GPTVariantConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)

        # Create attention variant
        self.attn = create_attention(
            variant=config.attention_type,
            n_embd=config.n_embd,
            n_head=config.n_head,
            block_size=config.block_size,
            dropout=config.dropout,
            bias=config.bias,
            **config.attention_kwargs,
        )

        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(
        self, x: torch.Tensor, return_attn: bool = False
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn_out, attn_weights = self.attn(self.ln_1(x), return_attn=return_attn)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, attn_weights


class GPTVariant(nn.Module):
    """GPT model with configurable attention mechanism.

    This allows comparing different attention variants on the same tasks.
    """

    def __init__(self, config: GPTVariantConfig) -> None:
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": nn.Embedding(config.block_size, config.n_embd),
                "drop": nn.Dropout(config.dropout),
                "h": nn.ModuleList([BlockVariant(config) for _ in range(config.n_layer)]),
                "ln_f": nn.LayerNorm(config.n_embd),
            }
        )

        # Don't use positional embeddings for rotary attention
        self.use_pos_emb = config.attention_type != "rotary"

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer["wte"].weight = self.lm_head.weight

        self.apply(self._init_weights)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"GPT ({config.attention_type}) with {n_params / 1e6:.2f}M parameters")

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[list[torch.Tensor]]]:
        device = idx.device
        B, T = idx.size()

        # Embeddings
        tok_emb = self.transformer["wte"](idx)

        if self.use_pos_emb:
            pos = torch.arange(0, T, dtype=torch.long, device=device)
            pos_emb = self.transformer["wpe"](pos)
            x = self.transformer["drop"](tok_emb + pos_emb)
        else:
            x = self.transformer["drop"](tok_emb)

        # Transformer blocks
        all_attentions = [] if return_attn else None
        for block in self.transformer["h"]:
            x, attn = block(x, return_attn=return_attn)
            if return_attn and attn is not None:
                all_attentions.append(attn)

        x = self.transformer["ln_f"](x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss, all_attentions

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = (
                idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]
            )
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


def create_model_variant(
    attention_type: str,
    vocab_size: int = 256,
    block_size: int = 128,
    n_layer: int = 4,
    n_head: int = 4,
    n_embd: int = 128,
    dropout: float = 0.1,
    **attention_kwargs,
) -> GPTVariant:
    """Factory to create GPT with specific attention variant.

    Args:
        attention_type: 'vanilla', 'linear', 'sliding_window', 'sparse', 'rotary'
        vocab_size: Vocabulary size
        block_size: Maximum sequence length
        n_layer: Number of transformer layers
        n_head: Number of attention heads
        n_embd: Embedding dimension
        dropout: Dropout probability
        **attention_kwargs: Additional args for attention (e.g., window_size)

    Returns:
        GPTVariant model
    """
    config = GPTVariantConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
        attention_type=attention_type,
        attention_kwargs=attention_kwargs,
    )
    return GPTVariant(config)


def compare_attention_complexity() -> None:
    """Print theoretical complexity comparison of attention variants."""
    print("=" * 60)
    print("ATTENTION VARIANT COMPLEXITY COMPARISON")
    print("=" * 60)
    print()
    print(f"{'Variant':<20} {'Time':<15} {'Memory':<15} {'Notes'}")
    print("-" * 60)
    print(f"{'Vanilla':<20} {'O(n²d)':<15} {'O(n²)':<15} {'Standard attention'}")
    print(f"{'Linear':<20} {'O(nd²)':<15} {'O(nd)':<15} {'No explicit attn matrix'}")
    print(f"{'Sliding Window':<20} {'O(nwd)':<15} {'O(nw)':<15} {'w = window size'}")
    print(f"{'Sparse':<20} {'O(n√n·d)':<15} {'O(n√n)':<15} {'Local + strided'}")
    print(f"{'Rotary':<20} {'O(n²d)':<15} {'O(n²)':<15} {'Relative pos via rotation'}")
    print()
    print("Where: n = sequence length, d = head dimension, w = window size")
    print()


def benchmark_attention_variants(
    seq_lengths: list[int] = [32, 64, 128, 256],
    n_embd: int = 128,
    n_head: int = 4,
    device: str = "cpu",
    num_runs: int = 10,
) -> dict[str, list[float]]:
    """Benchmark attention variants at different sequence lengths.

    Args:
        seq_lengths: List of sequence lengths to test
        n_embd: Embedding dimension
        n_head: Number of heads
        device: Device to benchmark on
        num_runs: Number of runs for averaging

    Returns:
        Dict mapping variant name to list of times (one per seq_length)
    """
    import time

    results = {name: [] for name in ATTENTION_VARIANTS.keys()}

    print(f"\nBenchmarking attention variants (device={device})...")
    print("-" * 50)

    for seq_len in seq_lengths:
        print(f"\nSequence length: {seq_len}")
        x = torch.randn(1, seq_len, n_embd, device=device)

        for name in ATTENTION_VARIANTS.keys():
            attn = create_attention(
                variant=name,
                n_embd=n_embd,
                n_head=n_head,
                block_size=seq_len,
            ).to(device)

            # Warmup
            with torch.no_grad():
                _ = attn(x)

            # Benchmark
            times = []
            for _ in range(num_runs):
                start = time.perf_counter()
                with torch.no_grad():
                    _ = attn(x)
                if device == "cuda":
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - start)

            avg_time = sum(times) / len(times) * 1000  # ms
            results[name].append(avg_time)
            print(f"  {name:<20}: {avg_time:.3f} ms")

    return results

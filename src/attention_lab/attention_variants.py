"""Different attention mechanism variants for comparison and study.

This module implements several attention variants to understand their
trade-offs in terms of compute, memory, and modeling capability.

Classes:

- :class:`BaseAttention`: Abstract base class for all variants
- :class:`VanillaAttention`: Standard O(n²) scaled dot-product attention
- :class:`LinearAttention`: O(n) attention using kernel feature maps
- :class:`SlidingWindowAttention`: Local attention with fixed window size
- :class:`SparseAttention`: Strided + local attention patterns
- :class:`RotaryAttention`: Attention with rotary position embeddings

Functions:

- :func:`create_attention`: Factory function to create attention modules

Constants:

- :data:`ATTENTION_VARIANTS`: Registry mapping names to classes
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F


class BaseAttention(nn.Module, ABC):
    """Base class for attention variants.

    All attention implementations inherit from this class and implement
    the :meth:`_attend` method.

    :param n_embd: Embedding dimension.
    :type n_embd: int
    :param n_head: Number of attention heads.
    :type n_head: int
    :param block_size: Maximum sequence length.
    :type block_size: int
    :param dropout: Dropout probability.
    :type dropout: float
    :param bias: Whether to use bias in projections.
    :type bias: bool
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        dropout: float = 0.1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        assert n_embd % n_head == 0

        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        self.block_size = block_size
        self.dropout_p = dropout

        # Projections
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Split tensor into attention heads.

        :param x: Input tensor of shape ``(B, T, C)``.
        :type x: torch.Tensor
        :returns: Tensor of shape ``(B, n_head, T, head_dim)``.
        :rtype: torch.Tensor
        """
        B, T, C = x.size()
        return x.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Merge attention heads back together.

        :param x: Input tensor of shape ``(B, n_head, T, head_dim)``.
        :type x: torch.Tensor
        :returns: Tensor of shape ``(B, T, C)``.
        :rtype: torch.Tensor
        """
        B, nh, T, hs = x.size()
        return x.transpose(1, 2).contiguous().view(B, T, nh * hs)

    @abstractmethod
    def _attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_attn: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute attention. Override in subclasses.

        :param q: Query tensor of shape ``(B, n_head, T, head_dim)``.
        :type q: torch.Tensor
        :param k: Key tensor of shape ``(B, n_head, T, head_dim)``.
        :type k: torch.Tensor
        :param v: Value tensor of shape ``(B, n_head, T, head_dim)``.
        :type v: torch.Tensor
        :param return_attn: Whether to return attention weights.
        :type return_attn: bool
        :returns: Tuple of (output, attention_weights or None).
        :rtype: tuple[torch.Tensor, Optional[torch.Tensor]]
        """
        pass

    def forward(
        self, x: torch.Tensor, return_attn: bool = False
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through attention layer.

        :param x: Input tensor of shape ``(B, T, C)``.
        :type x: torch.Tensor
        :param return_attn: Whether to return attention weights.
        :type return_attn: bool
        :returns: Tuple of (output, attention_weights or None).
        :rtype: tuple[torch.Tensor, Optional[torch.Tensor]]
        """
        B, T, C = x.size()

        # Compute Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        # Apply attention variant
        y, attn_weights = self._attend(q, k, v, return_attn)

        # Merge heads and project
        y = self._merge_heads(y)
        y = self.resid_dropout(self.c_proj(y))

        return y, attn_weights


class VanillaAttention(BaseAttention):
    """Standard scaled dot-product attention with causal mask.

    Complexity: O(n²) in sequence length
    Memory: O(n²) for attention matrix

    This is the original attention from "Attention Is All You Need".
    Uses PyTorch SDPA fused kernels during training for speed.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Causal mask (only used when return_attn=True)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(self.block_size, self.block_size)).view(
                1, 1, self.block_size, self.block_size
            ),
        )

    def _attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_attn: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, nh, T, hs = q.size()

        if return_attn:
            # Manual path for visualization
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            attn_weights = att
            att = self.attn_dropout(att)
            y = att @ v
            return y, attn_weights

        # Fast path: fused SDPA
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        return y, None


class LinearAttention(BaseAttention):
    """Linear attention using kernel feature maps.

    Complexity: O(n) in sequence length
    Memory: O(n) - no explicit attention matrix

    Based on "Transformers are RNNs" (Katharopoulos et al., 2020).
    Uses φ(x) = elu(x) + 1 as the feature map.

    Trade-off: Faster but may lose some expressiveness.
    """

    def __init__(self, *args, eps: float = 1e-6, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.eps = eps

    def _feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feature map φ(x) = elu(x) + 1."""
        return F.elu(x) + 1

    def _attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_attn: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, nh, T, hs = q.size()

        # Apply feature map
        q = self._feature_map(q)
        k = self._feature_map(k)

        # Causal linear attention using cumulative sums
        # KV = cumsum(k^T @ v) over time
        # Output = q @ KV / (q @ cumsum(k))

        # Compute cumulative KV and K
        kv = torch.einsum("bhnd,bhnm->bhdm", k, v)  # (B, nh, hs, hs)
        kv_cumsum = torch.cumsum(
            torch.einsum("bhti,bhtj->bhtij", k, v), dim=2
        )  # (B, nh, T, hs, hs)

        k_cumsum = torch.cumsum(k, dim=2)  # (B, nh, T, hs)

        # Compute output
        # y[t] = q[t] @ kv_cumsum[t] / (q[t] @ k_cumsum[t])
        qkv = torch.einsum("bhti,bhtij->bhtj", q, kv_cumsum)  # (B, nh, T, hs)
        normalizer = torch.einsum("bhti,bhti->bht", q, k_cumsum).unsqueeze(-1)  # (B, nh, T, 1)

        y = qkv / (normalizer + self.eps)

        # Linear attention doesn't have explicit attention weights
        # We can compute approximate weights if needed (expensive)
        attn_weights = None
        if return_attn:
            # Approximate: compute what the attention "would be"
            att = torch.einsum("bhqd,bhkd->bhqk", q, k)
            # Apply causal mask manually
            mask = torch.tril(torch.ones(T, T, device=q.device))
            att = att * mask.unsqueeze(0).unsqueeze(0)
            att = att / (att.sum(dim=-1, keepdim=True) + self.eps)
            attn_weights = att

        return y, attn_weights


class SlidingWindowAttention(BaseAttention):
    """Local attention with a sliding window.

    Complexity: O(n * w) where w is window size
    Memory: O(n * w)

    Each position only attends to the previous `window_size` positions.
    Good for tasks where local context is most important.
    """

    def __init__(self, *args, window_size: int = 32, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.window_size = window_size

        # Pre-compute window mask as boolean for SDPA attn_mask
        mask = torch.zeros(self.block_size, self.block_size, dtype=torch.bool)
        for i in range(self.block_size):
            start = max(0, i - window_size + 1)
            mask[i, start : i + 1] = True
        self.register_buffer("window_mask", mask)

    def _attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_attn: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, nh, T, hs = q.size()

        mask = self.window_mask[:T, :T]

        if return_attn:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))
            att = att.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
            att = F.softmax(att, dim=-1)
            attn_weights = att
            att = self.attn_dropout(att)
            y = att @ v
            return y, attn_weights

        # Fast path: fused SDPA with pre-computed mask
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        return y, None


class SparseAttention(BaseAttention):
    """Sparse attention with strided + local patterns.

    Complexity: O(n * sqrt(n)) approximately
    Memory: O(n * sqrt(n))

    Combines:
    - Local attention (nearby tokens)
    - Strided attention (every k-th token)

    Based on "Generating Long Sequences with Sparse Transformers" (Child et al., 2019).
    """

    def __init__(
        self, *args, local_size: int = 16, stride: int = 16, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.local_size = local_size
        self.stride = stride

        # Pre-compute sparse mask as boolean
        mask = torch.zeros(self.block_size, self.block_size, dtype=torch.bool)
        for i in range(self.block_size):
            local_start = max(0, i - local_size + 1)
            mask[i, local_start : i + 1] = True
            for j in range(0, i + 1, stride):
                mask[i, j] = True
        self.register_buffer("sparse_mask", mask)

    def _attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_attn: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, nh, T, hs = q.size()

        mask = self.sparse_mask[:T, :T]

        if return_attn:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))
            att = att.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
            att = F.softmax(att, dim=-1)
            attn_weights = att
            att = self.attn_dropout(att)
            y = att @ v
            return y, attn_weights

        # Fast path: fused SDPA with pre-computed mask
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        return y, None


class RotaryAttention(BaseAttention):
    """Attention with Rotary Position Embeddings (RoPE).

    Instead of adding position embeddings, RoPE rotates the query and key
    vectors based on their position. This encodes relative position information
    directly into the attention computation.

    Based on "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    (Su et al., 2021).
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Precompute rotation frequencies
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq)

        # Causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(self.block_size, self.block_size)).view(
                1, 1, self.block_size, self.block_size
            ),
        )

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rotary_pos_emb(
        self, q: torch.Tensor, k: torch.Tensor, T: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary position embeddings to q and k."""
        # Create position indices
        positions = torch.arange(T, device=q.device).float()

        # Compute rotation angles
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)  # (T, head_dim)

        cos = emb.cos().unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim)
        sin = emb.sin().unsqueeze(0).unsqueeze(0)

        # Apply rotation
        q_rot = (q * cos) + (self._rotate_half(q) * sin)
        k_rot = (k * cos) + (self._rotate_half(k) * sin)

        return q_rot, k_rot

    def _attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_attn: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, nh, T, hs = q.size()

        # Apply rotary embeddings
        q, k = self._apply_rotary_pos_emb(q, k, T)

        if return_attn:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            attn_weights = att
            att = self.attn_dropout(att)
            y = att @ v
            return y, attn_weights

        # Fast path: fused SDPA
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        return y, None


# Registry for easy access
ATTENTION_VARIANTS = {
    "vanilla": VanillaAttention,
    "linear": LinearAttention,
    "sliding_window": SlidingWindowAttention,
    "sparse": SparseAttention,
    "rotary": RotaryAttention,
}


def create_attention(
    variant: str,
    n_embd: int,
    n_head: int,
    block_size: int,
    dropout: float = 0.1,
    bias: bool = False,
    **kwargs,
) -> BaseAttention:
    """Factory function to create attention variants.

    :param variant: Attention type. One of ``'vanilla'``, ``'linear'``,
                   ``'sliding_window'``, ``'sparse'``, ``'rotary'``.
    :type variant: str
    :param n_embd: Embedding dimension.
    :type n_embd: int
    :param n_head: Number of attention heads.
    :type n_head: int
    :param block_size: Maximum sequence length.
    :type block_size: int
    :param dropout: Dropout probability.
    :type dropout: float
    :param bias: Whether to use bias in projections.
    :type bias: bool
    :param kwargs: Additional arguments for specific variants:
                  - ``window_size``: For sliding_window attention
                  - ``local_size``, ``stride``: For sparse attention
    :returns: Attention module instance.
    :rtype: BaseAttention
    :raises ValueError: If variant is not recognized.

    Example::

        attn = create_attention(
            variant="sparse",
            n_embd=128,
            n_head=4,
            block_size=256,
            local_size=16,
            stride=16,
        )
    """
    if variant not in ATTENTION_VARIANTS:
        raise ValueError(
            f"Unknown attention variant: {variant}. "
            f"Choose from: {list(ATTENTION_VARIANTS.keys())}"
        )

    return ATTENTION_VARIANTS[variant](
        n_embd=n_embd,
        n_head=n_head,
        block_size=block_size,
        dropout=dropout,
        bias=bias,
        **kwargs,
    )

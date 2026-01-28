"""GPT model implementation with attention inspection capabilities.

This module provides the core GPT model components:

- :class:`CausalSelfAttention`: Multi-head causal self-attention
- :class:`MLP`: Feed-forward network with GELU activation
- :class:`Block`: Transformer block with pre-norm architecture
- :class:`GPT`: Complete GPT language model
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

from attention_lab.config import GPTConfig


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with optional attention weight output.

    Implements scaled dot-product attention with a causal mask to prevent
    attending to future positions. Supports returning attention weights
    for visualization and analysis.

    :param config: Model configuration containing attention parameters.
    :type config: GPTConfig
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        # Key, query, value projections for all heads
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask to ensure attention only to previous positions
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(
        self, x: torch.Tensor, return_attn: bool = False
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with optional attention weights output.

        :param x: Input tensor of shape ``(batch, seq_len, n_embd)``.
        :type x: torch.Tensor
        :param return_attn: If True, return attention weights.
        :type return_attn: bool
        :returns: Tuple of (output tensor, attention weights or None).
                  Output has shape ``(batch, seq_len, n_embd)``.
                  Attention weights have shape ``(batch, n_head, seq_len, seq_len)``.
        :rtype: tuple[torch.Tensor, Optional[torch.Tensor]]
        """
        B, T, C = x.size()

        # Calculate query, key, values for all heads
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)

        attn_weights = None
        if return_attn:
            # Manual attention to capture weights for visualization
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            attn_weights = att
            att = self.attn_dropout(att)
            y = att @ v
        else:
            # Fast path: use PyTorch SDPA (fused kernels)
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True,
                dropout_p=self.dropout if self.training else 0.0,
            )

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # Reassemble head outputs

        # Output projection
        y = self.resid_dropout(self.c_proj(y))

        return y, attn_weights


class MLP(nn.Module):
    """Feed-forward network with GELU activation.

    Two-layer MLP with hidden dimension 4x the embedding dimension,
    following the original Transformer architecture.

    :param config: Model configuration.
    :type config: GPTConfig
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MLP.

        :param x: Input tensor of shape ``(batch, seq_len, n_embd)``.
        :type x: torch.Tensor
        :returns: Output tensor with same shape as input.
        :rtype: torch.Tensor
        """
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Transformer block with pre-norm architecture.

    Consists of a causal self-attention layer followed by an MLP,
    with LayerNorm applied before each sublayer (pre-norm).

    :param config: Model configuration.
    :type config: GPTConfig
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(
        self, x: torch.Tensor, return_attn: bool = False
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with optional attention weights.

        :param x: Input tensor of shape ``(batch, seq_len, n_embd)``.
        :type x: torch.Tensor
        :param return_attn: If True, return attention weights.
        :type return_attn: bool
        :returns: Tuple of (output tensor, attention weights or None).
        :rtype: tuple[torch.Tensor, Optional[torch.Tensor]]
        """
        attn_out, attn_weights = self.attn(self.ln_1(x), return_attn=return_attn)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, attn_weights


class GPT(nn.Module):
    """GPT Language Model with attention inspection capabilities.

    Decoder-only transformer for autoregressive language modeling.
    Supports returning attention weights from all layers for
    visualization and interpretability research.

    :param config: Model configuration.
    :type config: GPTConfig

    Example::

        config = GPTConfig(vocab_size=256, n_layer=4, n_head=4, n_embd=128)
        model = GPT(config)
        tokens = torch.randint(0, 256, (1, 32))
        logits, loss, attentions = model(tokens, return_attn=True)
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": nn.Embedding(config.block_size, config.n_embd),
                "drop": nn.Dropout(config.dropout),
                "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                "ln_f": nn.LayerNorm(config.n_embd),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying
        self.transformer["wte"].weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Report number of parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"GPT model with {n_params / 1e6:.2f}M parameters")

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
        """Forward pass through the GPT model.

        :param idx: Input token indices of shape ``(batch, seq_len)``.
        :type idx: torch.Tensor
        :param targets: Target token indices for loss computation. If provided,
                       cross-entropy loss is computed.
        :type targets: Optional[torch.Tensor]
        :param return_attn: If True, return attention weights from all layers.
        :type return_attn: bool
        :returns: Tuple of (logits, loss, attention_weights).
                  - logits: Shape ``(batch, seq_len, vocab_size)``
                  - loss: Scalar tensor if targets provided, else None
                  - attention_weights: List of tensors per layer if ``return_attn=True``
        :rtype: tuple[torch.Tensor, Optional[torch.Tensor], Optional[list[torch.Tensor]]]
        :raises AssertionError: If sequence length exceeds ``block_size``.
        """
        device = idx.device
        B, T = idx.size()
        assert T <= self.config.block_size, f"Sequence length {T} > block_size {self.config.block_size}"

        # Token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        tok_emb = self.transformer["wte"](idx)  # (B, T, n_embd)
        pos_emb = self.transformer["wpe"](pos)  # (T, n_embd)
        x = self.transformer["drop"](tok_emb + pos_emb)

        # Transformer blocks
        all_attentions = [] if return_attn else None
        for block in self.transformer["h"]:
            x, attn = block(x, return_attn=return_attn)
            if return_attn and attn is not None:
                all_attentions.append(attn)

        x = self.transformer["ln_f"](x)
        logits = self.lm_head(x)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss, all_attentions

    def get_num_params(self) -> int:
        """Return the total number of parameters in the model.

        :returns: Number of parameters.
        :rtype: int
        """
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """Generate new tokens autoregressively.

        :param idx: Initial context tokens of shape ``(batch, seq_len)``.
        :type idx: torch.Tensor
        :param max_new_tokens: Number of new tokens to generate.
        :type max_new_tokens: int
        :param temperature: Sampling temperature. Higher values produce more
                           random outputs, lower values more deterministic.
        :type temperature: float
        :param top_k: If set, only sample from the top k most likely tokens.
        :type top_k: Optional[int]
        :param top_p: If set, sample from the smallest set of tokens with
                     cumulative probability >= top_p (nucleus sampling).
        :type top_p: Optional[float]
        :returns: Generated token indices of shape ``(batch, seq_len + max_new_tokens)``.
        :rtype: torch.Tensor
        """
        for _ in range(max_new_tokens):
            # Crop context if needed
            idx_cond = (
                idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]
            )

            # Forward pass
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift to keep first token above threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                # Scatter back to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

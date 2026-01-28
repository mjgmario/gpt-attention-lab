"""Text generation utilities with attention inspection.

This module provides functions for autoregressive text generation:

- :func:`generate`: Generate tokens with optional attention collection
- :func:`get_attention_for_text`: Get attention weights for a given sequence
- :func:`sample_with_temperature`: Sample from logits with temperature
- :func:`greedy_decode`: Generate using greedy decoding
"""

from __future__ import annotations

from typing import Optional

import torch
from torch.nn import functional as F

from attention_lab.model import GPT


@torch.no_grad()
def generate(
    model: GPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    return_attention: bool = False,
) -> tuple[torch.Tensor, Optional[list[list[torch.Tensor]]]]:
    """Generate new tokens with optional attention weight collection.

    :param model: GPT model to use for generation.
    :type model: GPT
    :param idx: Initial context tokens of shape ``(batch, seq_len)``.
    :type idx: torch.Tensor
    :param max_new_tokens: Number of new tokens to generate.
    :type max_new_tokens: int
    :param temperature: Sampling temperature. Higher values produce more
                       random outputs.
    :type temperature: float
    :param top_k: If set, only sample from top k most likely tokens.
    :type top_k: Optional[int]
    :param top_p: If set, use nucleus sampling with this probability threshold.
    :type top_p: Optional[float]
    :param return_attention: If True, collect and return attention weights.
    :type return_attention: bool
    :returns: Tuple of (tokens, attentions).
              - tokens: Shape ``(batch, seq_len + max_new_tokens)``
              - attentions: List of attention weights per step if requested,
                each containing a list of tensors (one per layer)
    :rtype: tuple[torch.Tensor, Optional[list[list[torch.Tensor]]]]

    Example::

        tokens, attentions = generate(
            model, prompt_tokens,
            max_new_tokens=50,
            temperature=0.8,
            return_attention=True
        )
    """
    model.eval()
    all_step_attentions = [] if return_attention else None

    for _ in range(max_new_tokens):
        # Crop context if needed
        idx_cond = (
            idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size :]
        )

        # Forward pass
        logits, _, attentions = model(idx_cond, return_attn=return_attention)

        if return_attention and attentions is not None:
            all_step_attentions.append(attentions)

        logits = logits[:, -1, :] / temperature

        # Apply top-k filtering
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        # Apply top-p (nucleus) filtering
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float("-inf")

        # Sample from distribution
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx, all_step_attentions


def get_attention_for_text(
    model: GPT,
    tokens: torch.Tensor,
    device: str = "cpu",
) -> list[torch.Tensor]:
    """Get attention weights for a given sequence of tokens.

    Useful for visualizing attention patterns on specific inputs.

    :param model: GPT model to use.
    :type model: GPT
    :param tokens: Token indices of shape ``(seq_len,)`` or ``(1, seq_len)``.
    :type tokens: torch.Tensor
    :param device: Device to run inference on.
    :type device: str
    :returns: List of attention weight tensors, one per layer.
              Each tensor has shape ``(1, n_head, seq_len, seq_len)``.
    :rtype: list[torch.Tensor]

    Example::

        attentions = get_attention_for_text(model, dataset.encode("Hello"))
        plot_attention_heatmap(attentions, layer=0, head=0)
    """
    model.eval()

    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)

    tokens = tokens.to(device)

    with torch.no_grad():
        _, _, attentions = model(tokens, return_attn=True)

    return attentions if attentions is not None else []


def sample_with_temperature(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Sample from logits with temperature scaling.

    :param logits: Logits tensor of shape ``(batch, vocab_size)``.
    :type logits: torch.Tensor
    :param temperature: Temperature for scaling. Higher values produce more
                       random samples.
    :type temperature: float
    :returns: Sampled token indices of shape ``(batch, 1)``.
    :rtype: torch.Tensor
    """
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def greedy_decode(model: GPT, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    """Generate tokens using greedy decoding (always pick most likely).

    :param model: GPT model to use.
    :type model: GPT
    :param idx: Initial context tokens of shape ``(batch, seq_len)``.
    :type idx: torch.Tensor
    :param max_new_tokens: Number of new tokens to generate.
    :type max_new_tokens: int
    :returns: Generated token indices of shape ``(batch, seq_len + max_new_tokens)``.
    :rtype: torch.Tensor
    """
    model.eval()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = (
                idx
                if idx.size(1) <= model.config.block_size
                else idx[:, -model.config.block_size :]
            )
            logits, _, _ = model(idx_cond)
            idx_next = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            idx = torch.cat((idx, idx_next), dim=1)

    return idx

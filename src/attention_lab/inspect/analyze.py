"""Attention analysis utilities.

This module provides functions for analyzing attention patterns:

- :func:`head_entropy`: Measure attention distribution
- :func:`attention_to_positions`: Analyze position-based patterns
- :func:`find_induction_patterns`: Detect induction head behavior
- :func:`compute_attention_distance`: Measure attention locality
- :func:`compare_heads`: Comprehensive head comparison
- :func:`print_head_summary`: Print analysis summary
"""

from __future__ import annotations

import numpy as np
import torch


def head_entropy(attention: torch.Tensor) -> torch.Tensor:
    """Calculate entropy of attention distributions for each head.

    Higher entropy indicates more distributed attention (attending to many positions).
    Lower entropy indicates more focused attention (attending to few positions).

    :param attention: Attention tensor of shape ``(batch, n_head, seq_len, seq_len)``
                     or list of tensors (one per layer).
    :type attention: torch.Tensor | list[torch.Tensor]
    :returns: Entropy tensor of shape ``(n_head,)`` averaged over batch and positions.
              If input is a list, returns dict mapping layer names to entropy tensors.
    :rtype: torch.Tensor | dict[str, torch.Tensor]
    """
    if isinstance(attention, list):
        # Process all layers and return dict
        return {f"layer_{i}": head_entropy(attn) for i, attn in enumerate(attention)}

    # attention shape: (batch, n_head, seq_len, seq_len)
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    attn = attention.clamp(min=eps)

    # Calculate entropy: -sum(p * log(p))
    entropy = -torch.sum(attn * torch.log(attn), dim=-1)  # (batch, n_head, seq_len)

    # Average over batch and query positions
    return entropy.mean(dim=(0, 2))  # (n_head,)


def attention_to_positions(attention: torch.Tensor) -> dict[str, torch.Tensor]:
    """Analyze what positions each head attends to.

    Args:
        attention: Attention tensor of shape (batch, n_head, seq_len, seq_len).

    Returns:
        Dictionary with analysis metrics:
        - 'mean_position': Average attended position for each head
        - 'position_std': Standard deviation of attended positions
        - 'first_token_attention': How much each head attends to first token
        - 'prev_token_attention': How much each head attends to previous token
        - 'self_attention': How much each head attends to current position
    """
    if isinstance(attention, list):
        attention = attention[0]  # Take first layer

    batch, n_head, seq_len, _ = attention.shape

    # Create position indices
    positions = torch.arange(seq_len, device=attention.device, dtype=attention.dtype)

    # Mean attended position: weighted average of positions
    # For each query position, what position does it attend to on average?
    mean_pos = torch.einsum("bhqk,k->bhq", attention, positions)  # (batch, n_head, seq_len)
    mean_position = mean_pos.mean(dim=(0, 2))  # (n_head,)

    # Position variance
    pos_sq = torch.einsum("bhqk,k->bhq", attention, positions**2)
    var_pos = pos_sq - mean_pos**2
    position_std = var_pos.clamp(min=0).sqrt().mean(dim=(0, 2))

    # First token attention (BOS-like behavior)
    first_token_attention = attention[:, :, :, 0].mean(dim=(0, 2))

    # Previous token attention (local attention)
    # Diagonal offset by -1
    prev_attention = torch.zeros(batch, n_head, seq_len, device=attention.device)
    for i in range(1, seq_len):
        prev_attention[:, :, i] = attention[:, :, i, i - 1]
    prev_token_attention = prev_attention.mean(dim=(0, 2))

    # Self attention (diagonal)
    self_attn = torch.diagonal(attention, dim1=-2, dim2=-1)  # (batch, n_head, seq_len)
    self_attention = self_attn.mean(dim=(0, 2))

    return {
        "mean_position": mean_position,
        "position_std": position_std,
        "first_token_attention": first_token_attention,
        "prev_token_attention": prev_token_attention,
        "self_attention": self_attention,
    }


def find_induction_patterns(
    attention: torch.Tensor,
    tokens: torch.Tensor,
    threshold: float = 0.3,
) -> list[dict]:
    """Detect induction head patterns in attention.

    Induction heads look for previous occurrences of the current token
    and attend to the token that followed it (copying behavior).

    Args:
        attention: Attention tensor of shape (batch, n_head, seq_len, seq_len).
        tokens: Token indices of shape (batch, seq_len).
        threshold: Minimum attention weight to consider as attending.

    Returns:
        List of detected patterns, each with 'head', 'query_pos', 'key_pos',
        'attention_weight', and 'pattern_type'.
    """
    if isinstance(attention, list):
        attention = attention[-1]  # Use last layer (most likely to have induction)

    batch, n_head, seq_len, _ = attention.shape
    patterns = []

    for b in range(batch):
        for h in range(n_head):
            attn = attention[b, h]  # (seq_len, seq_len)
            toks = tokens[b]  # (seq_len,)

            for q in range(2, seq_len):  # Start from position 2
                # Check if there's a previous occurrence of token at q-1
                prev_token = toks[q - 1].item()

                for k in range(q - 1):
                    if toks[k].item() == prev_token:
                        # Check if attention at position q attends to k+1
                        if k + 1 < q:
                            attn_weight = attn[q, k + 1].item()
                            if attn_weight > threshold:
                                patterns.append(
                                    {
                                        "batch": b,
                                        "head": h,
                                        "query_pos": q,
                                        "key_pos": k + 1,
                                        "attention_weight": attn_weight,
                                        "pattern_type": "induction",
                                        "repeated_token_pos": k,
                                    }
                                )

    return patterns


def compute_attention_distance(attention: torch.Tensor) -> dict[str, torch.Tensor]:
    """Compute average attention distance for each head.

    Args:
        attention: Attention tensor of shape (batch, n_head, seq_len, seq_len).

    Returns:
        Dictionary with 'mean_distance' and 'local_ratio' (attention within 5 positions).
    """
    if isinstance(attention, list):
        attention = attention[0]

    batch, n_head, seq_len, _ = attention.shape

    # Create distance matrix
    positions = torch.arange(seq_len, device=attention.device)
    distances = positions.unsqueeze(0) - positions.unsqueeze(1)  # (seq_len, seq_len)
    distances = distances.abs().float()

    # Mask future positions (causal)
    mask = torch.tril(torch.ones(seq_len, seq_len, device=attention.device))
    distances = distances * mask

    # Weighted average distance
    mean_distance = torch.einsum("bhqk,qk->bh", attention, distances)
    mean_distance = mean_distance.mean(dim=0)  # (n_head,)

    # Local attention ratio (within 5 positions)
    local_mask = (distances <= 5) & (mask > 0)
    local_attention = (attention * local_mask.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
    local_ratio = local_attention.mean(dim=(0, 2))  # (n_head,)

    return {
        "mean_distance": mean_distance,
        "local_ratio": local_ratio,
    }


def compare_heads(
    attention: torch.Tensor | list[torch.Tensor],
    tokens: torch.Tensor | None = None,
) -> dict[str, np.ndarray]:
    """Generate comprehensive comparison metrics for all attention heads.

    Args:
        attention: Attention tensor or list of tensors (per layer).
        tokens: Optional token indices for induction pattern detection.

    Returns:
        Dictionary with arrays for each metric, shape (n_layers, n_heads).
    """
    if not isinstance(attention, list):
        attention = [attention]

    n_layers = len(attention)
    n_heads = attention[0].shape[1]

    metrics = {
        "entropy": np.zeros((n_layers, n_heads)),
        "mean_distance": np.zeros((n_layers, n_heads)),
        "local_ratio": np.zeros((n_layers, n_heads)),
        "first_token_attn": np.zeros((n_layers, n_heads)),
        "prev_token_attn": np.zeros((n_layers, n_heads)),
    }

    for layer_idx, attn in enumerate(attention):
        # Entropy
        ent = head_entropy(attn)
        metrics["entropy"][layer_idx] = ent.cpu().numpy()

        # Position analysis
        pos_metrics = attention_to_positions(attn)
        metrics["first_token_attn"][layer_idx] = pos_metrics["first_token_attention"].cpu().numpy()
        metrics["prev_token_attn"][layer_idx] = pos_metrics["prev_token_attention"].cpu().numpy()

        # Distance analysis
        dist_metrics = compute_attention_distance(attn)
        metrics["mean_distance"][layer_idx] = dist_metrics["mean_distance"].cpu().numpy()
        metrics["local_ratio"][layer_idx] = dist_metrics["local_ratio"].cpu().numpy()

    return metrics


def print_head_summary(metrics: dict[str, np.ndarray]) -> None:
    """Print a summary of head comparison metrics.

    Args:
        metrics: Dictionary from compare_heads().
    """
    n_layers, n_heads = metrics["entropy"].shape

    print("=" * 60)
    print("ATTENTION HEAD ANALYSIS SUMMARY")
    print("=" * 60)

    for layer in range(n_layers):
        print(f"\nLayer {layer}:")
        print("-" * 40)

        for head in range(n_heads):
            print(f"  Head {head}:")
            print(f"    Entropy:          {metrics['entropy'][layer, head]:.3f}")
            print(f"    Mean Distance:    {metrics['mean_distance'][layer, head]:.2f}")
            print(f"    Local Ratio:      {metrics['local_ratio'][layer, head]:.3f}")
            print(f"    First Token Attn: {metrics['first_token_attn'][layer, head]:.3f}")
            print(f"    Prev Token Attn:  {metrics['prev_token_attn'][layer, head]:.3f}")

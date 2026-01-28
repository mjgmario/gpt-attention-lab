"""Attention visualization utilities.

This module provides functions for visualizing attention patterns:

- :func:`plot_attention_heatmap`: Plot attention as a heatmap
- :func:`plot_attention_pattern`: Show causal attention structure
- :func:`plot_all_heads`: Grid of all heads in a layer
- :func:`plot_generation_attention`: Attention at generation steps
- :func:`create_attention_animation`: Animated GIF of attention
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_attention_heatmap(
    attention: torch.Tensor,
    layer: int = 0,
    head: int = 0,
    tokens: Optional[list[str]] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: tuple[int, int] = (10, 8),
    cmap: str = "Blues",
) -> plt.Figure:
    """Plot attention weights as a heatmap.

    :param attention: Attention tensor of shape ``(batch, n_head, seq_len, seq_len)``
                     or list of such tensors (one per layer).
    :type attention: torch.Tensor | list[torch.Tensor]
    :param layer: Layer index to visualize (if attention is a list).
    :type layer: int
    :param head: Head index to visualize.
    :type head: int
    :param tokens: Optional list of token strings for axis labels.
    :type tokens: Optional[list[str]]
    :param title: Optional title for the plot.
    :type title: Optional[str]
    :param save_path: If provided, save figure to this path.
    :type save_path: Optional[str]
    :param figsize: Figure size as (width, height).
    :type figsize: tuple[int, int]
    :param cmap: Matplotlib colormap name.
    :type cmap: str
    :returns: Matplotlib figure object.
    :rtype: plt.Figure
    """
    # Handle list of attention tensors (per layer)
    if isinstance(attention, list):
        attention = attention[layer]

    # Get attention weights for specific head
    attn = attention[0, head].detach().cpu().numpy()  # (seq_len, seq_len)

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(attn, cmap=cmap, aspect="auto")
    plt.colorbar(im, ax=ax, label="Attention Weight")

    # Set labels
    if tokens is not None:
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha="right")
        ax.set_yticklabels(tokens)

    ax.set_xlabel("Key Position (attending to)")
    ax.set_ylabel("Query Position (from)")

    if title is None:
        title = f"Attention Weights - Layer {layer}, Head {head}"
    ax.set_title(title)

    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_attention_pattern(
    attention: torch.Tensor,
    layer: int = 0,
    head: int = 0,
    tokens: Optional[list[str]] = None,
    save_path: Optional[str] = None,
    figsize: tuple[int, int] = (12, 4),
) -> plt.Figure:
    """Plot attention pattern showing the causal structure.

    Creates a triangular heatmap that emphasizes the causal mask structure.

    Args:
        attention: Attention tensor or list of tensors per layer.
        layer: Layer index to visualize.
        head: Head index to visualize.
        tokens: Optional token labels.
        save_path: If provided, save figure to this path.
        figsize: Figure size.

    Returns:
        Matplotlib figure object.
    """
    if isinstance(attention, list):
        attention = attention[layer]

    attn = attention[0, head].detach().cpu().numpy()
    seq_len = attn.shape[0]

    fig, ax = plt.subplots(figsize=figsize)

    # Create masked array for upper triangle (should be zero due to causal mask)
    masked_attn = np.ma.masked_where(np.triu(np.ones_like(attn), k=1), attn)

    im = ax.imshow(masked_attn, cmap="viridis", aspect="auto")
    plt.colorbar(im, ax=ax, label="Attention Weight")

    if tokens is not None:
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(tokens, fontsize=8)

    ax.set_xlabel("Key Position")
    ax.set_ylabel("Query Position")
    ax.set_title(f"Causal Attention Pattern - Layer {layer}, Head {head}")

    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_all_heads(
    attention: torch.Tensor | list[torch.Tensor],
    layer: int = 0,
    tokens: Optional[list[str]] = None,
    save_path: Optional[str] = None,
    figsize_per_head: tuple[float, float] = (4, 4),
) -> plt.Figure:
    """Plot all attention heads for a given layer in a grid.

    Args:
        attention: Attention tensor or list of tensors per layer.
        layer: Layer index to visualize.
        tokens: Optional token labels.
        save_path: If provided, save figure to this path.
        figsize_per_head: Size per head subplot.

    Returns:
        Matplotlib figure object.
    """
    if isinstance(attention, list):
        attention = attention[layer]

    n_heads = attention.shape[1]
    cols = min(4, n_heads)
    rows = (n_heads + cols - 1) // cols

    fig, axes = plt.subplots(
        rows, cols, figsize=(figsize_per_head[0] * cols, figsize_per_head[1] * rows)
    )
    axes = np.atleast_2d(axes)

    for h in range(n_heads):
        row, col = h // cols, h % cols
        ax = axes[row, col]

        attn = attention[0, h].detach().cpu().numpy()
        im = ax.imshow(attn, cmap="Blues", aspect="auto")

        ax.set_title(f"Head {h}", fontsize=10)

        if tokens is not None and len(tokens) <= 20:
            ax.set_xticks(range(len(tokens)))
            ax.set_yticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=6)
            ax.set_yticklabels(tokens, fontsize=6)
        else:
            ax.set_xticks([])
            ax.set_yticks([])

    # Hide unused subplots
    for h in range(n_heads, rows * cols):
        row, col = h // cols, h % cols
        axes[row, col].axis("off")

    fig.suptitle(f"All Attention Heads - Layer {layer}", fontsize=12)
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_generation_attention(
    all_step_attentions: list[list[torch.Tensor]],
    step: int,
    layer: int = 0,
    head: int = 0,
    tokens: Optional[list[str]] = None,
    save_path: Optional[str] = None,
    figsize: tuple[int, int] = (10, 8),
) -> plt.Figure:
    """Plot attention from a specific generation step.

    Args:
        all_step_attentions: List of attention weights per generation step.
                             Each element is a list of attention tensors (per layer).
        step: Generation step to visualize.
        layer: Layer index.
        head: Head index.
        tokens: Optional token labels.
        save_path: If provided, save figure to this path.
        figsize: Figure size.

    Returns:
        Matplotlib figure object.
    """
    if step >= len(all_step_attentions):
        raise ValueError(f"Step {step} out of range (max {len(all_step_attentions) - 1})")

    attention = all_step_attentions[step][layer]

    return plot_attention_heatmap(
        attention,
        layer=0,  # Already selected layer
        head=head,
        tokens=tokens,
        title=f"Generation Step {step} - Layer {layer}, Head {head}",
        save_path=save_path,
        figsize=figsize,
    )


def create_attention_animation(
    all_step_attentions: list[list[torch.Tensor]],
    layer: int = 0,
    head: int = 0,
    tokens_per_step: Optional[list[list[str]]] = None,
    save_path: str = "attention_animation.gif",
    interval: int = 500,
    figsize: tuple[int, int] = (8, 6),
) -> None:
    """Create an animated GIF of attention during generation.

    Args:
        all_step_attentions: List of attention weights per generation step.
        layer: Layer index to animate.
        head: Head index to animate.
        tokens_per_step: Optional list of token labels for each step.
        save_path: Path to save the GIF.
        interval: Milliseconds between frames.
        figsize: Figure size.
    """
    try:
        from matplotlib.animation import FuncAnimation, PillowWriter
    except ImportError:
        print("Animation requires Pillow. Install with: pip install Pillow")
        return

    fig, ax = plt.subplots(figsize=figsize)

    def update(frame: int) -> list:
        ax.clear()
        attn = all_step_attentions[frame][layer][0, head].detach().cpu().numpy()

        im = ax.imshow(attn, cmap="Blues", aspect="auto", vmin=0, vmax=1)

        if tokens_per_step is not None and frame < len(tokens_per_step):
            tokens = tokens_per_step[frame]
            if len(tokens) <= 30:
                ax.set_xticks(range(len(tokens)))
                ax.set_yticks(range(len(tokens)))
                ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=7)
                ax.set_yticklabels(tokens, fontsize=7)

        ax.set_title(f"Generation Step {frame} - Layer {layer}, Head {head}")
        return [im]

    anim = FuncAnimation(fig, update, frames=len(all_step_attentions), interval=interval, blit=False)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    anim.save(save_path, writer=PillowWriter(fps=1000 // interval))
    plt.close(fig)
    print(f"Animation saved to {save_path}")

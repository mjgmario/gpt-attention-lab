"""Attention inspection and visualization utilities.

This module provides tools for analyzing and visualizing attention patterns:

Visualization:
    :func:`plot_attention_heatmap`: Plot attention weights as heatmap
    :func:`plot_attention_pattern`: Show causal attention structure
    :func:`plot_all_heads`: Grid of all heads in a layer
    :func:`plot_generation_attention`: Attention at generation steps

Analysis:
    :func:`head_entropy`: Measure attention distribution
    :func:`attention_to_positions`: Analyze position-based patterns
    :func:`find_induction_patterns`: Detect induction head behavior

Example::

    from attention_lab.inspect import plot_attention_heatmap, head_entropy

    # Get attention weights
    _, _, attentions = model(tokens, return_attn=True)

    # Visualize
    plot_attention_heatmap(attentions, layer=0, head=0, save_path="attn.png")

    # Analyze
    entropy = head_entropy(attentions[0])
"""

from attention_lab.inspect.analyze import (
    attention_to_positions,
    find_induction_patterns,
    head_entropy,
)
from attention_lab.inspect.visualize import (
    plot_attention_heatmap,
    plot_attention_pattern,
    plot_generation_attention,
    plot_all_heads,
)

__all__ = [
    "plot_attention_heatmap",
    "plot_attention_pattern",
    "plot_generation_attention",
    "plot_all_heads",
    "head_entropy",
    "attention_to_positions",
    "find_induction_patterns",
]

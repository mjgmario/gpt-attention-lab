"""Configuration for GPT model.

This module provides the :class:`GPTConfig` dataclass for configuring
GPT model architecture parameters.
"""

from dataclasses import dataclass


@dataclass
class GPTConfig:
    """Configuration for a GPT model.

    :param vocab_size: Size of the vocabulary (default 256 for char-level ASCII).
    :type vocab_size: int
    :param block_size: Maximum context length (number of tokens the model can see).
    :type block_size: int
    :param n_layer: Number of transformer blocks.
    :type n_layer: int
    :param n_head: Number of attention heads per block.
    :type n_head: int
    :param n_embd: Embedding dimension. Must be divisible by ``n_head``.
    :type n_embd: int
    :param dropout: Dropout probability for regularization.
    :type dropout: float
    :param bias: Whether to use bias in linear layers.
    :type bias: bool

    :raises AssertionError: If ``n_embd`` is not divisible by ``n_head``.
    :raises AssertionError: If ``vocab_size`` or ``block_size`` are not positive.
    :raises AssertionError: If ``dropout`` is not in [0, 1).

    Example::

        config = GPTConfig(
            vocab_size=256,
            block_size=128,
            n_layer=4,
            n_head=4,
            n_embd=128,
        )
        model = GPT(config)
    """

    vocab_size: int = 256
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    bias: bool = False

    def __post_init__(self) -> None:
        """Validate configuration after initialization.

        :raises AssertionError: If configuration constraints are violated.
        """
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.block_size > 0, "block_size must be positive"
        assert 0.0 <= self.dropout < 1.0, "dropout must be in [0, 1)"

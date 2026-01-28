"""Simple character-level tokenizer.

This module provides the :class:`CharTokenizer` for character-level
text encoding and decoding.
"""

from __future__ import annotations


class CharTokenizer:
    """Character-level tokenizer for text.

    Maps each unique character to an integer index for use with language models.
    Simple and dependency-free implementation.

    :param chars: String of characters to use as vocabulary.
                 If None, uses printable ASCII (32-126) plus newline and tab.
    :type chars: str | None

    :ivar vocab_size: Number of unique characters in vocabulary.
    :ivar chars: Sorted list of vocabulary characters.
    :ivar char_to_idx: Dictionary mapping characters to indices.
    :ivar idx_to_char: Dictionary mapping indices to characters.

    Example::

        tokenizer = CharTokenizer("abc123")
        tokens = tokenizer.encode("abc")  # [0, 1, 2]
        text = tokenizer.decode([0, 1, 2])  # "abc"
    """

    def __init__(self, chars: str | None = None) -> None:
        if chars is None:
            # Default: printable ASCII + newline
            chars = "".join(chr(i) for i in range(32, 127)) + "\n\t"

        self.chars = sorted(set(chars))
        self.vocab_size = len(self.chars)

        # Create mappings
        self.char_to_idx: dict[str, int] = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char: dict[int, str] = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text: str) -> list[int]:
        """Encode text to list of token indices.

        :param text: Input text string.
        :type text: str
        :returns: List of integer token indices.
        :rtype: list[int]

        .. note::
           Characters not in vocabulary are silently skipped.
        """
        return [self.char_to_idx[ch] for ch in text if ch in self.char_to_idx]

    def decode(self, indices: list[int]) -> str:
        """Decode list of token indices to text.

        :param indices: List of integer token indices.
        :type indices: list[int]
        :returns: Decoded text string.
        :rtype: str

        .. note::
           Invalid indices are silently skipped.
        """
        return "".join(self.idx_to_char.get(i, "") for i in indices)

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        """Create tokenizer from text corpus.

        Extracts all unique characters from the text to build the vocabulary.

        :param text: Text to extract vocabulary from.
        :type text: str
        :returns: CharTokenizer with vocabulary from text.
        :rtype: CharTokenizer
        """
        chars = sorted(set(text))
        return cls(chars="".join(chars))

    def __len__(self) -> int:
        """Return vocabulary size.

        :returns: Number of unique tokens.
        :rtype: int
        """
        return self.vocab_size

    def __repr__(self) -> str:
        """Return string representation."""
        return f"CharTokenizer(vocab_size={self.vocab_size})"

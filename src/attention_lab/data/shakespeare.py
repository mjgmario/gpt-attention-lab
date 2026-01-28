"""TinyShakespeare dataset for character-level language modeling."""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset

from attention_lab.data.tokenizer import CharTokenizer

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DEFAULT_DATA_DIR = "data"


def download_shakespeare(data_dir: str = DEFAULT_DATA_DIR) -> str:
    """Download TinyShakespeare dataset.

    Args:
        data_dir: Directory to save the file.

    Returns:
        Path to the downloaded file.
    """
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    filepath = os.path.join(data_dir, "shakespeare.txt")

    if not os.path.exists(filepath):
        print(f"Downloading TinyShakespeare to {filepath}...")
        urllib.request.urlretrieve(SHAKESPEARE_URL, filepath)
        print("Download complete!")

    return filepath


class ShakespeareDataset(Dataset):
    """TinyShakespeare dataset for training character-level models.

    The dataset consists of ~1MB of Shakespeare text, tokenized at the
    character level. Each sample is a sequence of `block_size` tokens
    with the target being the next token at each position.
    """

    def __init__(
        self,
        block_size: int = 128,
        split: str = "train",
        train_ratio: float = 0.9,
        data_dir: str = DEFAULT_DATA_DIR,
        tokenizer: Optional[CharTokenizer] = None,
        max_samples: Optional[int] = None,
    ) -> None:
        """Initialize dataset.

        Args:
            block_size: Context length (number of tokens per sample).
            split: Either 'train' or 'val'.
            train_ratio: Fraction of data to use for training.
            data_dir: Directory containing the data file.
            tokenizer: Optional tokenizer. If None, creates from data.
            max_samples: Maximum number of samples. None = use full dataset.
        """
        super().__init__()
        self.block_size = block_size
        self.split = split

        # Download and load data
        filepath = download_shakespeare(data_dir)
        with open(filepath, encoding="utf-8") as f:
            text = f.read()

        # Create or use provided tokenizer
        if tokenizer is None:
            self.tokenizer = CharTokenizer.from_text(text)
        else:
            self.tokenizer = tokenizer

        # Encode full text
        data = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)

        # Split into train/val
        n = int(train_ratio * len(data))
        if split == "train":
            self.data = data[:n]
        else:
            self.data = data[n:]

        # Limit dataset size if max_samples specified
        self._max_samples = max_samples

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return self.tokenizer.vocab_size

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        total = max(0, len(self.data) - self.block_size)
        if self._max_samples is not None:
            return min(total, self._max_samples)
        return total

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a training sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (input tokens, target tokens).
            Both tensors have shape (block_size,).
        """
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y

    def decode(self, indices: list[int] | torch.Tensor) -> str:
        """Decode token indices to text.

        Args:
            indices: Token indices to decode.

        Returns:
            Decoded text string.
        """
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        return self.tokenizer.decode(indices)

    def encode(self, text: str) -> torch.Tensor:
        """Encode text to token indices.

        Args:
            text: Text to encode.

        Returns:
            Tensor of token indices.
        """
        return torch.tensor(self.tokenizer.encode(text), dtype=torch.long)

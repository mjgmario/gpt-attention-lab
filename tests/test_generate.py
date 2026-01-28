"""Tests for text generation utilities."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from attention_lab.config import GPTConfig
from attention_lab.generate import (
    generate,
    get_attention_for_text,
    greedy_decode,
    sample_with_temperature,
)
from attention_lab.model import GPT


class TestGenerate:
    """Tests for generate function."""

    @pytest.fixture
    def config(self):
        return GPTConfig(
            vocab_size=50,
            block_size=32,
            n_layer=2,
            n_head=2,
            n_embd=64,
        )

    @pytest.fixture
    def model(self, config):
        return GPT(config)

    def test_generate_produces_tokens(self, model, config):
        """Test that generate produces tokens."""
        idx = torch.randint(0, config.vocab_size, (1, 5))

        generated, _ = generate(model, idx, max_new_tokens=10)

        assert generated.shape == (1, 15)  # 5 + 10 tokens
        assert (generated >= 0).all()
        assert (generated < config.vocab_size).all()

    def test_generate_with_attention(self, model, config):
        """Test generate returns attention when requested."""
        idx = torch.randint(0, config.vocab_size, (1, 5))

        generated, attentions = generate(model, idx, max_new_tokens=3, return_attention=True)

        assert attentions is not None
        assert len(attentions) == 3  # One per generated token
        assert len(attentions[0]) == config.n_layer  # One per layer

    def test_generate_without_attention(self, model, config):
        """Test generate returns None attention by default."""
        idx = torch.randint(0, config.vocab_size, (1, 5))

        _, attentions = generate(model, idx, max_new_tokens=5)

        assert attentions is None

    def test_generate_with_temperature(self, model, config):
        """Test generate works with temperature."""
        idx = torch.randint(0, config.vocab_size, (1, 5))

        generated_low_temp, _ = generate(model, idx, max_new_tokens=10, temperature=0.1)
        generated_high_temp, _ = generate(model, idx, max_new_tokens=10, temperature=2.0)

        assert generated_low_temp.shape == generated_high_temp.shape

    def test_generate_with_top_k(self, model, config):
        """Test generate works with top-k sampling."""
        idx = torch.randint(0, config.vocab_size, (1, 5))

        generated, _ = generate(model, idx, max_new_tokens=10, top_k=5)

        assert generated.shape == (1, 15)

    def test_generate_with_top_p(self, model, config):
        """Test generate works with top-p sampling."""
        idx = torch.randint(0, config.vocab_size, (1, 5))

        generated, _ = generate(model, idx, max_new_tokens=10, top_p=0.9)

        assert generated.shape == (1, 15)


class TestGetAttentionForText:
    """Tests for get_attention_for_text function."""

    @pytest.fixture
    def config(self):
        return GPTConfig(
            vocab_size=50,
            block_size=32,
            n_layer=2,
            n_head=2,
            n_embd=64,
        )

    @pytest.fixture
    def model(self, config):
        return GPT(config)

    def test_returns_attention_list(self, model, config):
        """Test that function returns list of attention tensors."""
        tokens = torch.randint(0, config.vocab_size, (10,))

        attentions = get_attention_for_text(model, tokens)

        assert isinstance(attentions, list)
        assert len(attentions) == config.n_layer

    def test_attention_shapes(self, model, config):
        """Test attention tensor shapes are correct."""
        seq_len = 10
        tokens = torch.randint(0, config.vocab_size, (seq_len,))

        attentions = get_attention_for_text(model, tokens)

        for attn in attentions:
            assert attn.shape == (1, config.n_head, seq_len, seq_len)

    def test_handles_batched_input(self, model, config):
        """Test function handles already-batched input."""
        tokens = torch.randint(0, config.vocab_size, (1, 10))

        attentions = get_attention_for_text(model, tokens)

        assert len(attentions) == config.n_layer


class TestGreedyDecode:
    """Tests for greedy_decode function."""

    @pytest.fixture
    def config(self):
        return GPTConfig(
            vocab_size=50,
            block_size=32,
            n_layer=2,
            n_head=2,
            n_embd=64,
        )

    @pytest.fixture
    def model(self, config):
        return GPT(config)

    def test_greedy_decode_produces_tokens(self, model, config):
        """Test greedy decoding produces tokens."""
        idx = torch.randint(0, config.vocab_size, (1, 5))

        generated = greedy_decode(model, idx, max_new_tokens=10)

        assert generated.shape == (1, 15)
        assert (generated >= 0).all()
        assert (generated < config.vocab_size).all()

    def test_greedy_decode_deterministic(self, model, config):
        """Test greedy decoding is deterministic."""
        idx = torch.randint(0, config.vocab_size, (1, 5))

        generated1 = greedy_decode(model, idx, max_new_tokens=10)
        generated2 = greedy_decode(model, idx, max_new_tokens=10)

        assert torch.equal(generated1, generated2)


class TestSampleWithTemperature:
    """Tests for sample_with_temperature function."""

    def test_returns_correct_shape(self):
        """Test output shape is correct."""
        logits = torch.randn(2, 100)  # batch_size=2, vocab_size=100

        samples = sample_with_temperature(logits)

        assert samples.shape == (2, 1)

    def test_samples_in_valid_range(self):
        """Test samples are valid indices."""
        vocab_size = 100
        logits = torch.randn(5, vocab_size)

        samples = sample_with_temperature(logits)

        assert (samples >= 0).all()
        assert (samples < vocab_size).all()

    def test_low_temperature_more_deterministic(self):
        """Test that low temperature produces more consistent samples."""
        logits = torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0]])

        # Very low temperature should almost always pick index 0
        samples = [sample_with_temperature(logits, temperature=0.01).item() for _ in range(10)]

        # Most samples should be 0 (the highest logit)
        assert samples.count(0) >= 8

    def test_high_temperature_more_random(self):
        """Test that high temperature produces more varied samples."""
        # Equal logits
        logits = torch.zeros(1, 10)

        samples = [sample_with_temperature(logits, temperature=1.0).item() for _ in range(100)]

        # Should see variety in samples
        unique_samples = len(set(samples))
        assert unique_samples > 1

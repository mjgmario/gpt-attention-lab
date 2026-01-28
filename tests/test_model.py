"""Tests for GPT model components."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from attention_lab.config import GPTConfig
from attention_lab.model import GPT, MLP, Block, CausalSelfAttention


class TestGPTConfig:
    """Tests for GPTConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = GPTConfig()
        assert config.vocab_size == 256
        assert config.block_size == 128
        assert config.n_layer == 4
        assert config.n_head == 4
        assert config.n_embd == 128
        assert config.dropout == 0.1
        assert config.bias is False

    def test_custom_config(self):
        """Test custom configuration."""
        config = GPTConfig(
            vocab_size=100,
            block_size=64,
            n_layer=2,
            n_head=2,
            n_embd=64,
        )
        assert config.vocab_size == 100
        assert config.block_size == 64
        assert config.n_layer == 2

    def test_invalid_config_head_embd_mismatch(self):
        """Test that n_embd must be divisible by n_head."""
        with pytest.raises(AssertionError):
            GPTConfig(n_embd=100, n_head=3)  # 100 not divisible by 3


class TestCausalSelfAttention:
    """Tests for CausalSelfAttention module."""

    @pytest.fixture
    def config(self):
        return GPTConfig(n_embd=64, n_head=4, block_size=32)

    @pytest.fixture
    def attention(self, config):
        return CausalSelfAttention(config)

    def test_output_shape(self, attention, config):
        """Test output shape is correct."""
        batch_size, seq_len = 2, 16
        x = torch.randn(batch_size, seq_len, config.n_embd)

        output, _ = attention(x)

        assert output.shape == (batch_size, seq_len, config.n_embd)

    def test_return_attention_weights(self, attention, config):
        """Test that attention weights are returned when requested."""
        batch_size, seq_len = 2, 16
        x = torch.randn(batch_size, seq_len, config.n_embd)

        output, attn_weights = attention(x, return_attn=True)

        assert attn_weights is not None
        assert attn_weights.shape == (batch_size, config.n_head, seq_len, seq_len)

    def test_attention_weights_none_by_default(self, attention, config):
        """Test that attention weights are None by default."""
        x = torch.randn(2, 16, config.n_embd)
        _, attn_weights = attention(x, return_attn=False)
        assert attn_weights is None

    def test_causal_mask(self, attention, config):
        """Test that attention respects causal mask."""
        x = torch.randn(1, 8, config.n_embd)
        _, attn_weights = attention(x, return_attn=True)

        # Check upper triangle is zero (after softmax, it should be negligible)
        upper_triangle = torch.triu(attn_weights[0, 0], diagonal=1)
        assert torch.allclose(upper_triangle, torch.zeros_like(upper_triangle), atol=1e-6)

    def test_attention_weights_sum_to_one(self, attention, config):
        """Test that attention weights sum to 1 along key dimension."""
        x = torch.randn(2, 16, config.n_embd)
        _, attn_weights = attention(x, return_attn=True)

        # Sum along key dimension should be 1
        sums = attn_weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


class TestMLP:
    """Tests for MLP module."""

    @pytest.fixture
    def config(self):
        return GPTConfig(n_embd=64)

    @pytest.fixture
    def mlp(self, config):
        return MLP(config)

    def test_output_shape(self, mlp, config):
        """Test output shape matches input shape."""
        x = torch.randn(2, 16, config.n_embd)
        output = mlp(x)
        assert output.shape == x.shape

    def test_expansion_factor(self, mlp, config):
        """Test that hidden dimension is 4x embedding dimension."""
        assert mlp.c_fc.out_features == 4 * config.n_embd
        assert mlp.c_proj.in_features == 4 * config.n_embd


class TestBlock:
    """Tests for transformer Block module."""

    @pytest.fixture
    def config(self):
        return GPTConfig(n_embd=64, n_head=4, block_size=32)

    @pytest.fixture
    def block(self, config):
        return Block(config)

    def test_output_shape(self, block, config):
        """Test output shape matches input shape."""
        x = torch.randn(2, 16, config.n_embd)
        output, _ = block(x)
        assert output.shape == x.shape

    def test_return_attention(self, block, config):
        """Test that attention weights can be returned."""
        x = torch.randn(2, 16, config.n_embd)
        _, attn = block(x, return_attn=True)
        assert attn is not None
        assert attn.shape == (2, config.n_head, 16, 16)


class TestGPT:
    """Tests for GPT model."""

    @pytest.fixture
    def config(self):
        return GPTConfig(
            vocab_size=100,
            block_size=32,
            n_layer=2,
            n_head=2,
            n_embd=64,
        )

    @pytest.fixture
    def model(self, config):
        return GPT(config)

    def test_forward_pass(self, model, config):
        """Test forward pass produces correct output shape."""
        batch_size, seq_len = 2, 16
        idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        logits, loss, attentions = model(idx)

        assert logits.shape == (batch_size, seq_len, config.vocab_size)
        assert loss is None
        assert attentions is None

    def test_forward_with_targets(self, model, config):
        """Test forward pass with targets computes loss."""
        batch_size, seq_len = 2, 16
        idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        logits, loss, _ = model(idx, targets=targets)

        assert loss is not None
        assert loss.item() > 0  # Cross entropy should be positive

    def test_forward_return_attention(self, model, config):
        """Test forward pass can return attention weights."""
        idx = torch.randint(0, config.vocab_size, (2, 16))

        _, _, attentions = model(idx, return_attn=True)

        assert attentions is not None
        assert len(attentions) == config.n_layer
        assert attentions[0].shape == (2, config.n_head, 16, 16)

    def test_generate(self, model, config):
        """Test generation produces tokens."""
        idx = torch.randint(0, config.vocab_size, (1, 5))

        generated = model.generate(idx, max_new_tokens=10)

        assert generated.shape == (1, 15)  # 5 + 10
        assert (generated >= 0).all()
        assert (generated < config.vocab_size).all()

    def test_generate_with_top_k(self, model, config):
        """Test generation with top-k sampling."""
        idx = torch.randint(0, config.vocab_size, (1, 5))

        generated = model.generate(idx, max_new_tokens=10, top_k=10)

        assert generated.shape == (1, 15)

    def test_generate_with_top_p(self, model, config):
        """Test generation with top-p (nucleus) sampling."""
        idx = torch.randint(0, config.vocab_size, (1, 5))

        generated = model.generate(idx, max_new_tokens=10, top_p=0.9)

        assert generated.shape == (1, 15)

    def test_weight_tying(self, model):
        """Test that embedding and output weights are tied."""
        assert model.transformer["wte"].weight is model.lm_head.weight

    def test_sequence_length_limit(self, model, config):
        """Test that sequence length cannot exceed block_size."""
        idx = torch.randint(0, config.vocab_size, (1, config.block_size + 1))

        with pytest.raises(AssertionError):
            model(idx)

    def test_get_num_params(self, model):
        """Test parameter counting."""
        n_params = model.get_num_params()
        assert n_params > 0
        assert isinstance(n_params, int)

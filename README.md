# Attention Lab

Educational GPT implementation for understanding Transformers and attention mechanisms.

## Features

- Multiple attention variants: vanilla, linear, sparse, sliding window, rotary
- Shakespeare dataset for character-level language modeling
- Attention visualization and analysis tools
- All configuration via YAML files
- Dataset caching for faster startup

## Installation

```bash
git clone https://github.com/yourusername/attention-lab.git
cd attention-lab
uv sync
```

## Quick Start

```bash
# Prepare dataset (optional, caches for faster startup)
uv run python scripts/prepare_data.py

# Train with default config
uv run python scripts/train.py

# Train on Shakespeare
uv run python scripts/train.py --config config/experiments/shakespeare.yaml

# Compare all attention variants
uv run python scripts/train.py --config config/experiments/full_comparison.yaml --compare

# Visualize attention
uv run python scripts/inspect_attention.py --checkpoint checkpoints/best_model.pt --text "To be or not"
```

## Configuration

All settings in YAML under `config/`. Example:

```yaml
model:
  n_layer: 4
  n_head: 4
  n_embd: 128
  block_size: 128

attention:
  type: vanilla  # vanilla, linear, sparse, sliding_window, rotary

training:
  max_steps: 5000
  batch_size: 32
  learning_rate: 3.0e-4

data:
  use_cache: true  # load preloaded data if available
  datasets:
    - type: shakespeare
      weight: 1.0
```

## Attention Variants

| Variant | Complexity | Description |
|---------|------------|-------------|
| vanilla | O(n²) | Standard attention |
| linear | O(n) | Kernel feature maps |
| sliding_window | O(n·w) | Local attention |
| sparse | O(n·√n) | Local + strided |
| rotary | O(n²) | Rotary position embeddings |

## Comparing Attention Variants

Run the comparison with:

```bash
uv run python scripts/train.py --config config/experiments/compare_attention.yaml --compare
```

This generates the following plots in the output directory:

| File | Description |
|------|-------------|
| `loss_curves.png` | Training loss over time for each variant |
| `final_loss_bar.png` | Final loss comparison bar chart |
| `attention_patterns.png` | Attention heatmaps for each variant |
| `attention_entropy.png` | Attention entropy (higher = more distributed) |
| `attention_distance.png` | Mean attention distance and local ratio |
| `attention_masks.png` | Theoretical mask patterns for each variant |

## Dataset

The project uses TinyShakespeare for character-level language modeling. The dataset is automatically downloaded on first use, or can be preloaded with caching:

```bash
# Preload and cache
uv run python scripts/prepare_data.py

# List cached data
uv run python scripts/prepare_data.py --list

# Force re-download
uv run python scripts/prepare_data.py --force
```

Set `use_cache: true` in the YAML config to use cached data during training.

## Python API

```python
from attention_lab import GPT, GPTConfig, Trainer
from attention_lab.data import ShakespeareDataset

# Create model
config = GPTConfig(vocab_size=65, n_layer=4, n_head=4, n_embd=128)
model = GPT(config)

# Train
dataset = ShakespeareDataset(block_size=128)
trainer = Trainer(model, DataLoader(dataset, batch_size=32))
trainer.train()

# Generate with attention
logits, loss, attentions = model(tokens, return_attn=True)
```

## Project Structure

```
attention_lab/
├── config/                      # YAML configs
│   ├── default.yaml
│   └── experiments/
│       ├── full_comparison.yaml # Compare all attention types
│       ├── shakespeare.yaml
│       ├── compare_attention.yaml
│       ├── linear_attention.yaml
│       └── sparse_attention.yaml
├── src/attention_lab/
│   ├── model.py                 # Core GPT
│   ├── attention_variants.py    # Attention implementations
│   ├── model_variants.py        # GPT with pluggable attention
│   ├── config_loader.py         # YAML config loading
│   ├── data/                    # Shakespeare dataset + cache
│   └── inspect/                 # Visualization
├── scripts/
│   ├── train.py                 # Main training script
│   ├── prepare_data.py          # Dataset preparation
│   ├── generate_text.py         # Text generation from checkpoint
│   ├── inspect_attention.py     # Attention visualization
│   ├── compare_heads.py         # Head comparison analysis
│   └── ablation_study.py        # Layer/head ablation study
└── tests/
```

## Development

```bash
uv run pytest tests/ -v
uv run ruff check . && uv run ruff format .
```

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Transformers are RNNs](https://arxiv.org/abs/2006.16236) (Linear attention)
- [Sparse Transformers](https://arxiv.org/abs/1904.10509)
- [RoFormer](https://arxiv.org/abs/2104.09864) (RoPE)
- [nanoGPT](https://github.com/karpathy/nanoGPT) - Andrej Karpathy's minimal GPT implementation
- [minGPT](https://github.com/karpathy/minGPT) - Andrej Karpathy's educational GPT

## License

MIT

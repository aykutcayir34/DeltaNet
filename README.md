# DeltaNet

A PyTorch implementation of **DeltaNet** based on the paper:

> **Parallelizing Linear Transformers with the Delta Rule over Sequence Length**  
> Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, Yoon Kim  
> arXiv:2406.06484, 2024  
> [Paper](https://arxiv.org/abs/2406.06484) | [PDF](https://arxiv.org/pdf/2406.06484)

## Overview

DeltaNet is a linear attention variant that uses a delta rule update mechanism for maintaining a recurrent state. This implementation features:

- **Multi-Head Chunked Attention**: Efficient computation by processing sequences in chunks with multi-head support
- **Linear Complexity**: O(L) complexity instead of O(L²) for standard attention
- **RMSNorm**: Root Mean Square Layer Normalization for stable training
- **SwiGLU Activation**: Gated activation function for the feed-forward network
- **Language Model Ready**: Full transformer architecture with embedding and output head

## Architecture

### Components

| Component | Description |
|-----------|-------------|
| `chunk_delta_rule` | Core multi-head chunked attention algorithm with delta rule updates |
| `RMSNorm` | Root Mean Square Normalization layer |
| `DeltaNet` | Multi-head attention module with Q, K, V projections, 1D convolutions, and learnable beta |
| `SwiGLU` | Swish-Gated Linear Unit activation |
| `DeltaNetBlock` | Full block with DeltaNet attention and SwiGLU FFN with residual connections |
| `DeltaNetModel` | Complete language model with embedding, stacked blocks, and output head |

## Installation

```bash
pip install torch
```

## Usage

### Language Model

```python
import torch

# Initialize language model
model = DeltaNetModel(
    vocab_size=1000,   # Vocabulary size
    dim=128,           # Model dimension
    depth=2,           # Number of DeltaNet blocks
    num_heads=4        # Number of attention heads
)

# Forward pass
input_ids = torch.randint(0, 1000, (2, 64))  # (batch_size, seq_len)
logits = model(input_ids)
print(logits.shape)  # torch.Size([2, 64, 1000])
```

### Using the Chunk Delta Rule Directly

```python
import torch

B, H, L, d = 2, 8, 1024, 64  # batch, heads, seq_len, head_dim
Q = torch.randn(B, H, L, d)
K = torch.randn(B, H, L, d)
V = torch.randn(B, H, L, d)
beta = torch.ones(B, H, L)
chunk_size = 16

output = chunk_delta_rule(Q, K, V, beta, chunk_size)
print(output.shape)  # torch.Size([2, 8, 1024, 64])
```

### DeltaNet Attention Module

```python
# DeltaNet attention module
delta_net = DeltaNet(
    d_model=128,       # Model dimension
    chunk_size=64,     # Chunk size for attention
    num_heads=8        # Number of attention heads
)

x = torch.randn(2, 64, 128)  # (batch_size, seq_len, d_model)
output = delta_net(x)
print(output.shape)  # torch.Size([2, 64, 128])
```

### DeltaNet Block

```python
block = DeltaNetBlock(
    dim=128,
    num_heads=4,
    mlp_ratio=4.0  # FFN hidden dim = dim * mlp_ratio
)

x = torch.randn(2, 64, 128)  # (batch_size, seq_len, dim)
output = block(x)
print(output.shape)  # torch.Size([2, 64, 128])
```

## Model Parameters

### chunk_delta_rule (Function)

| Parameter | Type | Description |
|-----------|------|-------------|
| `q` | Tensor | Query tensor of shape `(batch_size, num_heads, seq_len, head_dim)` |
| `k` | Tensor | Key tensor of shape `(batch_size, num_heads, seq_len, head_dim)` |
| `v` | Tensor | Value tensor of shape `(batch_size, num_heads, seq_len, head_dim)` |
| `beta` | Tensor | Beta tensor of shape `(batch_size, num_heads, seq_len)` |
| `chunk_size` | int | Size of chunks for attention computation |

### DeltaNet (Attention Module)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `d_model` | int | - | Dimension of the model |
| `chunk_size` | int | 64 | Size of chunks for attention computation |
| `num_heads` | int | 8 | Number of attention heads |

### SwiGLU

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dim` | int | - | Input/output dimension |
| `hidden_dim` | int | - | Hidden dimension |
| `bias` | bool | False | Whether to use bias in linear layers |

### DeltaNetBlock

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dim` | int | - | Dimension of the model |
| `num_heads` | int | - | Number of attention heads |
| `mlp_ratio` | float | 4.0 | Ratio for FFN hidden dimension |

### DeltaNetModel

| Parameter | Type | Description |
|-----------|------|-------------|
| `vocab_size` | int | Size of the vocabulary |
| `dim` | int | Dimension of the model |
| `depth` | int | Number of DeltaNet blocks |
| `num_heads` | int | Number of attention heads |

## Requirements

- Python 3.7+
- PyTorch 1.9+

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{yang2024parallelizing,
  title={Parallelizing Linear Transformers with the Delta Rule over Sequence Length},
  author={Yang, Songlin and Wang, Bailin and Zhang, Yu and Shen, Yikang and Kim, Yoon},
  journal={arXiv preprint arXiv:2406.06484},
  year={2024}
}
```

## License

MIT License

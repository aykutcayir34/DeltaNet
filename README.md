# DeltaNet

A PyTorch implementation of DeltaNet, a linear attention mechanism with chunked computation for efficient sequence modeling.

## Overview

DeltaNet is a linear attention variant that uses a delta rule update mechanism for maintaining a recurrent state. This implementation features:

- **Chunked Attention**: Efficient computation by processing sequences in chunks
- **Linear Complexity**: O(L) complexity instead of O(L²) for standard attention
- **RMSNorm**: Root Mean Square Layer Normalization for stable training
- **SwiGLU Activation**: Gated activation function for the feed-forward network

## Architecture

### Components

| Component | Description |
|-----------|-------------|
| `chunk_delta_rule` | Core chunked attention algorithm with delta rule updates |
| `RMSNorm` | Root Mean Square Normalization layer |
| `DeltaNet` | Main attention module with Q, K, V projections and 1D convolutions |
| `SwiGLU` | Swish-Gated Linear Unit activation |
| `TransformerBlock` | Full transformer block with DeltaNet attention and FFN |
| `TransformerDeltaNet` | Stacked DeltaNet layers with residual connections |

## Installation

```bash
pip install torch
```

## Usage

### Basic Usage

```python
import torch
from deltanet import TransformerDeltaNet

# Initialize model
model = TransformerDeltaNet(
    d_model=64,      # Model dimension
    chunk_size=4,    # Chunk size for attention
    num_layers=6     # Number of DeltaNet layers
)

# Forward pass
input_tensor = torch.randn(8, 128, 64)  # (batch_size, seq_len, d_model)
output = model(input_tensor)
print(output.shape)  # torch.Size([8, 128, 64])
```

### Using the Chunk Delta Rule Directly

```python
import torch

B, L, d = 2, 10, 4
Q = torch.randn(B, L, d)
K = torch.randn(B, L, d)
V = torch.randn(B, L, d)
beta = torch.ones(B, L)
chunk_size = 2

output = chunk_delta_rule(Q, K, V, beta, chunk_size)
```

### Transformer Block with FFN

```python
from deltanet import TransformerBlock

block = TransformerBlock(
    d_model=64,
    chunk_size=4,
    ff_hidden_dim=256
)

x = torch.randn(8, 128, 64)
output = block(x)
```

## Model Parameters

### DeltaNet

| Parameter | Type | Description |
|-----------|------|-------------|
| `d_model` | int | Dimension of the model |
| `chunk_size` | int | Size of chunks for attention computation |

### TransformerBlock

| Parameter | Type | Description |
|-----------|------|-------------|
| `d_model` | int | Dimension of the model |
| `chunk_size` | int | Size of chunks for attention computation |
| `ff_hidden_dim` | int | Hidden dimension of the feed-forward network |

### TransformerDeltaNet

| Parameter | Type | Description |
|-----------|------|-------------|
| `d_model` | int | Dimension of the model |
| `chunk_size` | int | Size of chunks for attention computation |
| `num_layers` | int | Number of DeltaNet layers |

## Requirements

- Python 3.7+
- PyTorch 1.9+

## License

MIT License

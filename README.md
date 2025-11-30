# Todd_Titans

> **⚠️ Development Status: Archived**
>
> This repository is no longer under active development. Further work on memory-augmented architectures has moved to [**Todd_Atlas**](https://github.com/r3d91ll/Todd_Atlas).
>
> This implementation remains available as a reference for the Titans paper.

A PyTorch implementation of the Titans architecture from the paper ["Titans: Learning to Memorize at Test Time"](https://arxiv.org/abs/2501.00663) by Google DeepMind.

## Overview

Titans introduces a novel approach to sequence modeling that combines attention mechanisms with neural memory modules capable of learning at test time. The key innovation is treating memory as a neural network that can be updated during inference, enabling the model to memorize and recall information from extremely long contexts.

## Architecture Variants

This implementation provides three memory-augmented architectures:

### MAC (Memory as Context)
Memory output is concatenated with attention context:
```text
output = Attention([memory_output; context])
```
Best for tasks requiring explicit memory retrieval alongside contextual processing.

### MAG (Memory as Gating)
Memory gates the attention output:
```text
output = attention_output * sigmoid(memory_output)
```
Best for tasks where memory should modulate rather than replace attention.

### MAL (Memory as Layer)
Memory and attention operate as sequential layers:
```text
output = Memory(Attention(input))
```
Best for deep integration of memory into the processing pipeline.

## Key Components

### Neural Memory Module
- **Associative memory** using gradient-based updates
- **Surprise-driven learning**: Updates proportional to prediction error
- **Momentum-based forgetting**: Gradual decay with configurable rates
- **Parallel processing**: Efficient chunk-based computation

### Sliding Window Attention
- Configurable window size for local attention
- Optional Flash Attention support
- Rotary positional embeddings
- L2 normalization for stable training

### Persistent Memory
- Learnable key-value pairs prepended to sequences
- Task-specific knowledge independent of input
- Improves performance on knowledge-intensive tasks

## Installation

```bash
pip install -e .
```

### Requirements
- Python >= 3.10
- PyTorch >= 2.0
- Optional: flash-attn for accelerated attention

## Usage

### Quick Start

```python
from titans import TitansLM
from titans.configs import TitansConfig, MemoryConfig, AttentionConfig

# Create configuration
config = TitansConfig(
    vocab_size=32000,
    d_model=512,
    num_layers=6,
    memory=MemoryConfig(d_model=512, d_key=64, d_value=64, num_memory_layers=2),
    attention=AttentionConfig(d_model=512, num_heads=8, d_head=64),
)

# Initialize model
model = TitansLM(config)

# Forward pass
import torch
x = torch.randint(0, 32000, (1, 1024))  # (batch, seq_len)
result = model(x)  # Returns dict with 'logits', 'memory_states'
logits = result['logits']  # (batch, seq_len, vocab_size)
```

### Using Presets

```python
from titans.configs import titans_small, titans_medium, titans_large

# Small model (~170M params)
config = titans_small()

# Medium model (~400M params)
config = titans_medium()

# Large model (~760M params)
config = titans_large()
```

### Architecture Selection

```python
from titans import TitansMAC, TitansMAG, TitansMAL, TitansConfig

config = TitansConfig(vocab_size=32000, d_model=512, num_layers=6)

# Memory as Context
mac_model = TitansMAC(config)

# Memory as Gating
mag_model = TitansMAG(config)

# Memory as Layer
mal_model = TitansMAL(config)
```

### Working with Memory State

```python
from titans import TitansLM
from titans.configs import TitansConfig

config = TitansConfig(vocab_size=32000, d_model=512, num_layers=6)
model = TitansLM(config)

# Process with persistent memory state
memory_states = None
for chunk in text_chunks:
    result = model(chunk, memory_states=memory_states)
    logits = result['logits']
    memory_states = result['memory_states']
    # memory_states carries forward learned associations
```

## Paper Reference

```bibtex
@article{behrouz2025titans,
  title={Titans: Learning to Memorize at Test Time},
  author={Behrouz, Ali and Hashemi, Peilin and Sreenivas, Anil and
          Vyas, Yash and Yu, Zhiheng and Yazdi, Maziar Sanjabi},
  journal={arXiv preprint arXiv:2501.00663},
  year={2025}
}
```

## Project Context

This implementation is part of a research portfolio exploring memory-augmented architectures:

- **Todd_MemRAG**: Industry-standard RAG with embedding-based retrieval
- **Todd_Titans**: This repo - Titans paper implementation (archived)
- **[Todd_Atlas](https://github.com/r3d91ll/Todd_Atlas)**: Atlas architecture with continuous memory (active development)

Titans serves as foundational context for understanding the evolution toward continuous memory systems. Development has moved to Todd_Atlas, which addresses the dimensional compression limitations explored in this implementation.

## License

MIT License - see LICENSE file for details.

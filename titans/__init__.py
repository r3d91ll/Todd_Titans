"""
Titans: Learning to Memorize at Test Time

PyTorch implementation of the Titans architecture from:
"Titans: Learning to Memorize at Test Time" (arXiv:2501.00663)

Titans introduces neural long-term memory that learns to memorize context
at test time through gradient-based optimization with a surprise metric.
"""

from titans.models.titans import (
    TitansLM,
    TitansMAC,
    TitansMAG,
    TitansMAL,
    TitansBlockMAC,
    TitansBlockMAG,
    TitansBlockMAL,
)
from titans.layers.memory import (
    NeuralMemory,
    NeuralMemoryParallel,
    DeepMemory,
    MemoryState,
)
from titans.layers.attention import (
    SlidingWindowAttention,
    PersistentMemoryAttention,
    GatedAttentionUnit,
)
from titans.configs import (
    TitansConfig,
    MemoryConfig,
    AttentionConfig,
    titans_small,
    titans_medium,
    titans_large,
)

__version__ = "0.1.0"
__all__ = [
    # Models
    "TitansLM",
    "TitansMAC",
    "TitansMAG",
    "TitansMAL",
    "TitansBlockMAC",
    "TitansBlockMAG",
    "TitansBlockMAL",
    # Memory
    "NeuralMemory",
    "NeuralMemoryParallel",
    "DeepMemory",
    "MemoryState",
    # Attention
    "SlidingWindowAttention",
    "PersistentMemoryAttention",
    "GatedAttentionUnit",
    # Configs
    "TitansConfig",
    "MemoryConfig",
    "AttentionConfig",
    "titans_small",
    "titans_medium",
    "titans_large",
]

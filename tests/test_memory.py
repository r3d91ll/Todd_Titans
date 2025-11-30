"""Tests for Neural Memory module."""

import pytest
import torch

from titans.layers.memory import NeuralMemory, NeuralMemoryParallel, DeepMemory, MemoryState
from titans.configs import MemoryConfig


class TestNeuralMemory:
    """Test cases for NeuralMemory class."""

    @pytest.fixture
    def memory_config(self) -> MemoryConfig:
        """Create a test memory configuration."""
        return MemoryConfig(
            d_model=64,
            d_memory=32,
            num_heads=4,
            depth=2,
            segment_size=16,
        )

    @pytest.fixture
    def memory(self, memory_config: MemoryConfig) -> NeuralMemory:
        """Create a test NeuralMemory instance."""
        return NeuralMemory(
            d_model=memory_config.d_model,
            d_memory=memory_config.d_memory,
            num_heads=memory_config.num_heads,
            depth=memory_config.depth,
        )

    def test_init(self, memory: NeuralMemory) -> None:
        """Test NeuralMemory initialization."""
        assert memory.d_model == 64
        assert memory.d_memory == 32
        assert memory.num_heads == 4

    def test_forward_shape(self, memory: NeuralMemory) -> None:
        """Test forward pass output shape."""
        batch, seq_len = 2, 32
        x = torch.randn(batch, seq_len, 64)

        output, _ = memory(x)

        assert output.shape == (batch, seq_len, 64)

    def test_forward_with_state(self, memory: NeuralMemory) -> None:
        """Test forward pass with memory state."""
        batch, seq_len = 2, 32
        x = torch.randn(batch, seq_len, 64)

        # First pass
        output1, state1 = memory(x, return_state=True)

        # Second pass with state
        output2, state2 = memory(x, state=state1, return_state=True)

        assert output1.shape == output2.shape
        assert state1 is not None
        assert state2 is not None

    def test_memory_state_structure(self, memory: NeuralMemory) -> None:
        """Test MemoryState dataclass structure."""
        batch, seq_len = 2, 32
        x = torch.randn(batch, seq_len, 64)

        _, state = memory(x, return_state=True)

        assert isinstance(state, MemoryState)
        assert state.params is not None
        assert state.momentum is not None


class TestNeuralMemoryParallel:
    """Test cases for NeuralMemoryParallel class."""

    @pytest.fixture
    def memory_parallel(self) -> NeuralMemoryParallel:
        """Create a test NeuralMemoryParallel instance."""
        return NeuralMemoryParallel(
            d_model=64,
            d_memory=32,
            num_heads=4,
            depth=2,
            segment_size=16,
        )

    def test_forward_shape(self, memory_parallel: NeuralMemoryParallel) -> None:
        """Test parallel memory forward pass."""
        batch, seq_len = 2, 64
        x = torch.randn(batch, seq_len, 64)

        output, _ = memory_parallel(x)

        assert output.shape == (batch, seq_len, 64)

    def test_chunked_processing(self, memory_parallel: NeuralMemoryParallel) -> None:
        """Test that sequence is processed in chunks."""
        batch, seq_len = 2, 64  # 4 chunks of 16
        x = torch.randn(batch, seq_len, 64)

        output, _ = memory_parallel(x)

        # Output should maintain sequence length
        assert output.shape[1] == seq_len


class TestDeepMemory:
    """Test cases for DeepMemory class."""

    @pytest.fixture
    def deep_memory(self) -> DeepMemory:
        """Create a test DeepMemory instance."""
        return DeepMemory(
            d_model=64,
            d_memory=32,
            num_heads=4,
            depth=3,
        )

    def test_forward_shape(self, deep_memory: DeepMemory) -> None:
        """Test deep memory forward pass."""
        batch, seq_len = 2, 32
        x = torch.randn(batch, seq_len, 64)

        output, _ = deep_memory(x)

        assert output.shape == (batch, seq_len, 64)

    def test_layer_count(self, deep_memory: DeepMemory) -> None:
        """Test that correct number of layers are created."""
        assert len(deep_memory.layers) == 3

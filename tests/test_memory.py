"""Tests for Neural Memory module."""

import pytest
import torch

from titans.layers.memory import NeuralMemory, NeuralMemoryParallel, DeepMemory
from titans.configs import MemoryConfig


class TestDeepMemory:
    """Test cases for DeepMemory class."""

    @pytest.fixture
    def deep_memory(self) -> DeepMemory:
        """Create a test DeepMemory instance."""
        return DeepMemory(
            d_key=64,
            d_value=32,
            num_layers=3,
        )

    def test_init(self, deep_memory: DeepMemory) -> None:
        """Test DeepMemory initialization."""
        assert deep_memory.d_key == 64
        assert deep_memory.d_value == 32
        assert deep_memory.num_layers == 3

    def test_forward_shape(self, deep_memory: DeepMemory) -> None:
        """Test deep memory forward pass."""
        batch, seq_len = 2, 32
        x = torch.randn(batch, seq_len, 64)  # d_key=64

        output = deep_memory(x)

        assert output.shape == (batch, seq_len, 32)  # d_value=32

    def test_get_parameters(self, deep_memory: DeepMemory) -> None:
        """Test get_parameters method."""
        params = deep_memory.get_parameters()
        assert isinstance(params, dict)
        assert len(params) > 0


class TestNeuralMemory:
    """Test cases for NeuralMemory class."""

    @pytest.fixture
    def memory_config(self) -> MemoryConfig:
        """Create a test memory configuration."""
        return MemoryConfig(
            d_model=64,
            d_key=32,
            d_value=32,
            num_memory_layers=2,
        )

    @pytest.fixture
    def memory(self, memory_config: MemoryConfig) -> NeuralMemory:
        """Create a test NeuralMemory instance."""
        return NeuralMemory(
            d_model=memory_config.d_model,
            d_key=memory_config.d_key,
            d_value=memory_config.d_value,
            num_memory_layers=memory_config.num_memory_layers,
        )

    def test_init(self, memory: NeuralMemory) -> None:
        """Test NeuralMemory initialization."""
        assert memory.d_model == 64
        assert memory.d_key == 32
        assert memory.d_value == 32

    def test_forward_shape(self, memory: NeuralMemory) -> None:
        """Test forward pass output shape."""
        batch, seq_len = 2, 32
        x = torch.randn(batch, seq_len, 64)

        output, _ = memory(x)

        assert output.shape == (batch, seq_len, 64)

    def test_forward_with_state(self, memory: NeuralMemory) -> None:
        """Test forward pass with memory state."""
        batch, seq_len = 2, 8  # Use shorter sequence for speed
        x = torch.randn(batch, seq_len, 64)

        # First pass
        output1, state1 = memory(x, return_memory_state=True)

        # Second pass with state
        output2, state2 = memory(x, memory_state=state1, return_memory_state=True)

        assert output1.shape == output2.shape
        assert state1 is not None
        assert state2 is not None

    def test_memory_state_structure(self, memory: NeuralMemory) -> None:
        """Test memory state dictionary structure."""
        batch, seq_len = 2, 8
        x = torch.randn(batch, seq_len, 64)

        _, state = memory(x, return_memory_state=True)

        assert isinstance(state, dict)
        assert "weights" in state
        assert "surprise" in state


class TestNeuralMemoryParallel:
    """Test cases for NeuralMemoryParallel class."""

    @pytest.fixture
    def memory_parallel(self) -> NeuralMemoryParallel:
        """Create a test NeuralMemoryParallel instance."""
        return NeuralMemoryParallel(
            d_model=64,
            d_key=32,
            d_value=32,
            chunk_size=16,
        )

    def test_init(self, memory_parallel: NeuralMemoryParallel) -> None:
        """Test initialization."""
        assert memory_parallel.d_model == 64
        assert memory_parallel.d_key == 32
        assert memory_parallel.d_value == 32
        assert memory_parallel.chunk_size == 16

    def test_forward_shape(self, memory_parallel: NeuralMemoryParallel) -> None:
        """Test parallel memory forward pass."""
        batch, seq_len = 2, 64
        x = torch.randn(batch, seq_len, 64)

        output, new_memory = memory_parallel(x)

        assert output.shape == (batch, seq_len, 64)
        assert new_memory.shape == (batch, 32, 32)  # (batch, d_key, d_value)

    def test_with_memory_state(self, memory_parallel: NeuralMemoryParallel) -> None:
        """Test forward pass with initial memory state."""
        batch, seq_len = 2, 64
        x = torch.randn(batch, seq_len, 64)
        initial_memory = torch.randn(batch, 32, 32)

        output, new_memory = memory_parallel(x, memory_state=initial_memory)

        assert output.shape == (batch, seq_len, 64)
        assert new_memory.shape == (batch, 32, 32)

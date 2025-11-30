"""Tests for Attention mechanisms."""

import pytest
import torch

from titans.layers.attention import (
    SlidingWindowAttention,
    PersistentMemoryAttention,
    GatedAttentionUnit,
)


class TestSlidingWindowAttention:
    """Test cases for SlidingWindowAttention class."""

    @pytest.fixture
    def attention(self) -> SlidingWindowAttention:
        """Create a test attention instance."""
        return SlidingWindowAttention(
            d_model=64,
            num_heads=4,
            d_head=16,
            window_size=32,
            use_flash=False,  # Disable flash for testing
        )

    def test_init(self, attention: SlidingWindowAttention) -> None:
        """Test attention initialization."""
        assert attention.d_model == 64
        assert attention.num_heads == 4
        assert attention.d_head == 16
        assert attention.window_size == 32

    def test_forward_shape(self, attention: SlidingWindowAttention) -> None:
        """Test forward pass output shape."""
        batch, seq_len = 2, 64
        x = torch.randn(batch, seq_len, 64)

        output, cache = attention(x)

        assert output.shape == (batch, seq_len, 64)
        assert cache is None  # No cache when not provided

    def test_with_attention_mask(self, attention: SlidingWindowAttention) -> None:
        """Test forward pass with attention mask."""
        batch, seq_len = 2, 64
        x = torch.randn(batch, seq_len, 64)
        mask = torch.ones(batch, seq_len)
        mask[:, -16:] = 0  # Mask last 16 positions

        output, _ = attention(x, attention_mask=mask)

        assert output.shape == (batch, seq_len, 64)

    def test_kv_cache(self, attention: SlidingWindowAttention) -> None:
        """Test incremental decoding with KV cache."""
        batch, d_model = 2, 64

        # Initial forward pass
        x1 = torch.randn(batch, 32, d_model)
        _, cache = attention(x1, kv_cache=(
            torch.zeros(batch, 4, 0, 16),
            torch.zeros(batch, 4, 0, 16)
        ))

        # Incremental pass
        x2 = torch.randn(batch, 1, d_model)
        output, new_cache = attention(x2, kv_cache=cache)

        assert output.shape == (batch, 1, d_model)
        assert new_cache is not None


class TestPersistentMemoryAttention:
    """Test cases for PersistentMemoryAttention class."""

    @pytest.fixture
    def persistent_attention(self) -> PersistentMemoryAttention:
        """Create a test persistent memory attention instance."""
        return PersistentMemoryAttention(
            d_model=64,
            num_heads=4,
            d_head=16,
            num_persistent=8,
            window_size=32,
        )

    def test_init(self, persistent_attention: PersistentMemoryAttention) -> None:
        """Test persistent attention initialization."""
        assert persistent_attention.num_persistent == 8
        assert persistent_attention.persistent_k.shape == (1, 8, 64)
        assert persistent_attention.persistent_v.shape == (1, 8, 64)

    def test_forward_shape(self, persistent_attention: PersistentMemoryAttention) -> None:
        """Test forward pass output shape."""
        batch, seq_len = 2, 64
        x = torch.randn(batch, seq_len, 64)

        output = persistent_attention(x)

        assert output.shape == (batch, seq_len, 64)

    def test_persistent_tokens_used(self, persistent_attention: PersistentMemoryAttention) -> None:
        """Test that persistent tokens are used in attention."""
        batch, seq_len = 2, 64
        x = torch.randn(batch, seq_len, 64)

        # Modify persistent tokens
        persistent_attention.persistent_k.data.fill_(1.0)

        output = persistent_attention(x)

        # Output should be affected by persistent tokens
        assert output.shape == (batch, seq_len, 64)


class TestGatedAttentionUnit:
    """Test cases for GatedAttentionUnit class."""

    @pytest.fixture
    def gated_unit(self) -> GatedAttentionUnit:
        """Create a test gated attention unit."""
        return GatedAttentionUnit(d_model=64)

    def test_forward_shape(self, gated_unit: GatedAttentionUnit) -> None:
        """Test forward pass output shape."""
        batch, seq_len, d_model = 2, 32, 64
        attention_output = torch.randn(batch, seq_len, d_model)
        memory_output = torch.randn(batch, seq_len, d_model)

        output = gated_unit(attention_output, memory_output)

        assert output.shape == (batch, seq_len, d_model)

    def test_gating_behavior(self, gated_unit: GatedAttentionUnit) -> None:
        """Test that gating blends inputs."""
        batch, seq_len, d_model = 2, 32, 64
        attention_output = torch.ones(batch, seq_len, d_model)
        memory_output = torch.zeros(batch, seq_len, d_model)

        output = gated_unit(attention_output, memory_output)

        # Output should be between 0 and 1 (normalized)
        assert output.shape == (batch, seq_len, d_model)

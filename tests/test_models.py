"""Tests for Titans model architectures."""

import pytest
import torch

from titans import TitansLM
from titans.models.titans import (
    TitansMAC,
    TitansMAG,
    TitansMAL,
    TitansBlockMAC,
    TitansBlockMAG,
    TitansBlockMAL,
)
from titans.configs import TitansConfig, MemoryConfig, AttentionConfig, titans_small


class TestTitansConfig:
    """Test cases for TitansConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = TitansConfig()

        assert config.vocab_size == 32000
        assert config.d_model == 512
        assert config.num_layers == 12
        assert config.attention.num_heads == 8

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = TitansConfig(
            vocab_size=50000,
            d_model=768,
            num_layers=6,
            memory=MemoryConfig(d_model=768, d_key=64, d_value=64),
            attention=AttentionConfig(d_model=768, num_heads=12, d_head=64),
        )

        assert config.vocab_size == 50000
        assert config.d_model == 768
        assert config.num_layers == 6

    def test_preset_config(self) -> None:
        """Test preset configuration."""
        config = titans_small()

        assert config.vocab_size == 32000
        assert config.d_model == 768
        assert config.num_layers == 12


class TestTitansLM:
    """Test cases for TitansLM class."""

    @pytest.fixture
    def config(self) -> TitansConfig:
        """Create a small test configuration."""
        return TitansConfig(
            vocab_size=1000,
            d_model=64,
            num_layers=2,
            memory=MemoryConfig(d_model=64, d_key=16, d_value=16, num_memory_layers=2),
            attention=AttentionConfig(d_model=64, num_heads=4, d_head=16),
            chunk_size=16,
        )

    @pytest.fixture
    def model(self, config: TitansConfig) -> TitansLM:
        """Create a test TitansLM instance."""
        return TitansLM(config)

    def test_init(self, model: TitansLM, config: TitansConfig) -> None:
        """Test model initialization."""
        assert model.config == config
        assert len(model.backbone.blocks) == config.num_layers

    def test_forward_shape(self, model: TitansLM) -> None:
        """Test forward pass output shape."""
        batch, seq_len = 2, 32
        x = torch.randint(0, 1000, (batch, seq_len))

        result = model(x)

        assert "logits" in result
        assert "memory_states" in result
        assert result["logits"].shape == (batch, seq_len, 1000)

    def test_forward_with_memory_state(self, model: TitansLM) -> None:
        """Test forward pass with memory state return."""
        batch, seq_len = 2, 32
        x = torch.randint(0, 1000, (batch, seq_len))

        result = model(x)

        assert result["logits"].shape == (batch, seq_len, 1000)
        assert result["memory_states"] is not None
        assert len(result["memory_states"]) == 2  # num_layers

    def test_forward_with_labels(self, model: TitansLM) -> None:
        """Test forward pass with labels for loss computation."""
        batch, seq_len = 2, 32
        x = torch.randint(0, 1000, (batch, seq_len))
        labels = torch.randint(0, 1000, (batch, seq_len))

        result = model(x, labels=labels)

        assert "loss" in result
        assert result["loss"].dim() == 0  # Scalar loss


class TestTitansMAC:
    """Test cases for TitansMAC (Memory as Context)."""

    @pytest.fixture
    def config(self) -> TitansConfig:
        """Create a test configuration."""
        return TitansConfig(
            vocab_size=1000,
            d_model=64,
            num_layers=2,
            memory=MemoryConfig(d_model=64, d_key=16, d_value=16),
            attention=AttentionConfig(d_model=64, num_heads=4, d_head=16),
            chunk_size=16,
        )

    @pytest.fixture
    def model(self, config: TitansConfig) -> TitansMAC:
        """Create a test TitansMAC instance."""
        return TitansMAC(config)

    def test_forward_shape(self, model: TitansMAC) -> None:
        """Test forward pass output shape."""
        batch, seq_len = 2, 32
        x = torch.randn(batch, seq_len, 64)

        output, memory_states = model(x)

        assert output.shape == (batch, seq_len, 64)
        assert len(memory_states) == 2


class TestTitansMAG:
    """Test cases for TitansMAG (Memory as Gating)."""

    @pytest.fixture
    def config(self) -> TitansConfig:
        """Create a test configuration."""
        return TitansConfig(
            vocab_size=1000,
            d_model=64,
            num_layers=2,
            memory=MemoryConfig(d_model=64, d_key=16, d_value=16),
            attention=AttentionConfig(d_model=64, num_heads=4, d_head=16),
        )

    @pytest.fixture
    def model(self, config: TitansConfig) -> TitansMAG:
        """Create a test TitansMAG instance."""
        return TitansMAG(config)

    def test_forward_shape(self, model: TitansMAG) -> None:
        """Test forward pass output shape."""
        batch, seq_len = 2, 32
        x = torch.randn(batch, seq_len, 64)

        output, memory_states = model(x)

        assert output.shape == (batch, seq_len, 64)
        assert len(memory_states) == 2


class TestTitansMAL:
    """Test cases for TitansMAL (Memory as Layer)."""

    @pytest.fixture
    def config(self) -> TitansConfig:
        """Create a test configuration."""
        return TitansConfig(
            vocab_size=1000,
            d_model=64,
            num_layers=2,
            memory=MemoryConfig(d_model=64, d_key=16, d_value=16),
            attention=AttentionConfig(d_model=64, num_heads=4, d_head=16),
        )

    @pytest.fixture
    def model(self, config: TitansConfig) -> TitansMAL:
        """Create a test TitansMAL instance."""
        return TitansMAL(config)

    def test_forward_shape(self, model: TitansMAL) -> None:
        """Test forward pass output shape."""
        batch, seq_len = 2, 32
        x = torch.randn(batch, seq_len, 64)

        output, memory_states = model(x)

        assert output.shape == (batch, seq_len, 64)
        assert len(memory_states) == 2


class TestTitansBlockVariants:
    """Test cases for Titans block variants."""

    @pytest.fixture
    def d_model(self) -> int:
        """Model dimension."""
        return 64

    @pytest.fixture
    def memory_config(self, d_model: int) -> MemoryConfig:
        """Memory configuration."""
        return MemoryConfig(d_model=d_model, d_key=16, d_value=16)

    @pytest.fixture
    def attention_config(self, d_model: int) -> AttentionConfig:
        """Attention configuration."""
        return AttentionConfig(d_model=d_model, num_heads=4, d_head=16)

    def test_block_mac(
        self,
        d_model: int,
        memory_config: MemoryConfig,
        attention_config: AttentionConfig,
    ) -> None:
        """Test TitansBlockMAC."""
        block = TitansBlockMAC(
            d_model=d_model,
            chunk_size=16,
            memory_config=memory_config,
            attention_config=attention_config,
            ffn_hidden_dim=d_model * 4,
        )

        x = torch.randn(2, 32, d_model)
        output, _memory_state = block(x)

        assert output.shape == x.shape

    def test_block_mag(
        self,
        d_model: int,
        memory_config: MemoryConfig,
        attention_config: AttentionConfig,
    ) -> None:
        """Test TitansBlockMAG."""
        block = TitansBlockMAG(
            d_model=d_model,
            memory_config=memory_config,
            attention_config=attention_config,
            ffn_hidden_dim=d_model * 4,
        )

        x = torch.randn(2, 32, d_model)
        output, _memory_state = block(x)

        assert output.shape == x.shape

    def test_block_mal(
        self,
        d_model: int,
        memory_config: MemoryConfig,
        attention_config: AttentionConfig,
    ) -> None:
        """Test TitansBlockMAL."""
        block = TitansBlockMAL(
            d_model=d_model,
            memory_config=memory_config,
            attention_config=attention_config,
            ffn_hidden_dim=d_model * 4,
        )

        x = torch.randn(2, 32, d_model)
        output, _memory_state = block(x)

        assert output.shape == x.shape

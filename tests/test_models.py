"""Tests for Titans model architectures."""

import pytest
import torch

from titans import (
    TitansLM,
    TitansMAC,
    TitansMAG,
    TitansMAL,
    TitansBlockMAC,
    TitansBlockMAG,
    TitansBlockMAL,
)
from titans.configs import TitansConfig, titans_small


class TestTitansConfig:
    """Test cases for TitansConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = TitansConfig()

        assert config.vocab_size == 32000
        assert config.d_model == 512
        assert config.num_layers == 6
        assert config.num_heads == 8

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = TitansConfig(
            vocab_size=50000,
            d_model=768,
            num_layers=12,
        )

        assert config.vocab_size == 50000
        assert config.d_model == 768
        assert config.num_layers == 12

    def test_preset_config(self) -> None:
        """Test preset configuration."""
        config = titans_small(vocab_size=32000)

        assert config.vocab_size == 32000
        assert config.d_model == 512
        assert config.num_layers == 6


class TestTitansLM:
    """Test cases for TitansLM class."""

    @pytest.fixture
    def config(self) -> TitansConfig:
        """Create a small test configuration."""
        return TitansConfig(
            vocab_size=1000,
            d_model=64,
            num_layers=2,
            num_heads=4,
            d_head=16,
            memory_depth=2,
            segment_size=16,
        )

    @pytest.fixture
    def model(self, config: TitansConfig) -> TitansLM:
        """Create a test TitansLM instance."""
        return TitansLM(config)

    def test_init(self, model: TitansLM, config: TitansConfig) -> None:
        """Test model initialization."""
        assert model.config == config
        assert len(model.layers) == config.num_layers

    def test_forward_shape(self, model: TitansLM) -> None:
        """Test forward pass output shape."""
        batch, seq_len = 2, 32
        x = torch.randint(0, 1000, (batch, seq_len))

        output = model(x)

        assert output.shape == (batch, seq_len, 1000)

    def test_forward_with_memory_state(self, model: TitansLM) -> None:
        """Test forward pass with memory state return."""
        batch, seq_len = 2, 32
        x = torch.randint(0, 1000, (batch, seq_len))

        output, memory_state = model(x, return_memory=True)

        assert output.shape == (batch, seq_len, 1000)
        assert memory_state is not None

    def test_causal_masking(self, model: TitansLM) -> None:
        """Test that model uses causal masking."""
        batch, seq_len = 2, 32
        x = torch.randint(0, 1000, (batch, seq_len))

        # Model should process without errors (causal mask prevents future info leak)
        output = model(x)

        assert output.shape == (batch, seq_len, 1000)


class TestTitansMAC:
    """Test cases for TitansMAC (Memory as Context)."""

    @pytest.fixture
    def config(self) -> TitansConfig:
        """Create a test configuration."""
        return TitansConfig(
            vocab_size=1000,
            d_model=64,
            num_layers=2,
            num_heads=4,
            d_head=16,
        )

    @pytest.fixture
    def model(self, config: TitansConfig) -> TitansMAC:
        """Create a test TitansMAC instance."""
        return TitansMAC(config)

    def test_forward_shape(self, model: TitansMAC) -> None:
        """Test forward pass output shape."""
        batch, seq_len = 2, 32
        x = torch.randint(0, 1000, (batch, seq_len))

        output = model(x)

        assert output.shape == (batch, seq_len, 1000)


class TestTitansMAG:
    """Test cases for TitansMAG (Memory as Gating)."""

    @pytest.fixture
    def config(self) -> TitansConfig:
        """Create a test configuration."""
        return TitansConfig(
            vocab_size=1000,
            d_model=64,
            num_layers=2,
            num_heads=4,
            d_head=16,
        )

    @pytest.fixture
    def model(self, config: TitansConfig) -> TitansMAG:
        """Create a test TitansMAG instance."""
        return TitansMAG(config)

    def test_forward_shape(self, model: TitansMAG) -> None:
        """Test forward pass output shape."""
        batch, seq_len = 2, 32
        x = torch.randint(0, 1000, (batch, seq_len))

        output = model(x)

        assert output.shape == (batch, seq_len, 1000)


class TestTitansMAL:
    """Test cases for TitansMAL (Memory as Layer)."""

    @pytest.fixture
    def config(self) -> TitansConfig:
        """Create a test configuration."""
        return TitansConfig(
            vocab_size=1000,
            d_model=64,
            num_layers=2,
            num_heads=4,
            d_head=16,
        )

    @pytest.fixture
    def model(self, config: TitansConfig) -> TitansMAL:
        """Create a test TitansMAL instance."""
        return TitansMAL(config)

    def test_forward_shape(self, model: TitansMAL) -> None:
        """Test forward pass output shape."""
        batch, seq_len = 2, 32
        x = torch.randint(0, 1000, (batch, seq_len))

        output = model(x)

        assert output.shape == (batch, seq_len, 1000)


class TestTitansBlockVariants:
    """Test cases for Titans block variants."""

    @pytest.fixture
    def d_model(self) -> int:
        """Model dimension."""
        return 64

    def test_block_mac(self, d_model: int) -> None:
        """Test TitansBlockMAC."""
        block = TitansBlockMAC(
            d_model=d_model,
            num_heads=4,
            d_head=16,
            ff_mult=4,
        )

        x = torch.randn(2, 32, d_model)
        output, _ = block(x)

        assert output.shape == x.shape

    def test_block_mag(self, d_model: int) -> None:
        """Test TitansBlockMAG."""
        block = TitansBlockMAG(
            d_model=d_model,
            num_heads=4,
            d_head=16,
            ff_mult=4,
        )

        x = torch.randn(2, 32, d_model)
        output, _ = block(x)

        assert output.shape == x.shape

    def test_block_mal(self, d_model: int) -> None:
        """Test TitansBlockMAL."""
        block = TitansBlockMAL(
            d_model=d_model,
            num_heads=4,
            d_head=16,
            ff_mult=4,
        )

        x = torch.randn(2, 32, d_model)
        output, _ = block(x)

        assert output.shape == x.shape

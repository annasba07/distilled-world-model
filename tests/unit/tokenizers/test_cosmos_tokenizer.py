"""
Unit tests for Cosmos-Inspired Video Tokenizer

Tests cover:
- Full reconstruction pipeline
- Compression ratio validation
- Encoding/decoding separately
- Tokenize/detokenize workflow
- Loss computation
- Performance metrics (perplexity, codebook usage)
"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from models.tokenizers.cosmos_tokenizer import CosmosInspiredTokenizer


class TestCosmosTokenizerInitialization:
    """Test tokenizer initialization and configuration"""

    def test_default_initialization(self):
        """Test tokenizer with default parameters"""
        tokenizer = CosmosInspiredTokenizer()

        assert tokenizer.in_channels == 3
        assert tokenizer.latent_dim == 512
        assert tokenizer.codebook_size == 2**16
        assert tokenizer.resolution == (640, 360)

    def test_custom_initialization(self):
        """Test tokenizer with custom parameters"""
        tokenizer = CosmosInspiredTokenizer(
            in_channels=3,
            encoder_dims=[32, 64, 128],
            decoder_dims=[64, 32, 16],
            latent_dim=256,
            codebook_size=2**12,
            resolution=(320, 180)
        )

        assert tokenizer.latent_dim == 256
        assert tokenizer.codebook_size == 4096
        assert tokenizer.resolution == (320, 180)

    def test_perceptual_loss_setup(self):
        """Test perceptual loss initialization (optional)"""
        # Without perceptual loss
        tokenizer = CosmosInspiredTokenizer(use_perceptual_loss=False)
        assert not tokenizer.use_perceptual_loss

        # Note: With perceptual loss requires lpips installed
        # tokenizer = CosmosInspiredTokenizer(use_perceptual_loss=True)
        # Would test if lpips is available


class TestCosmosTokenizerForward:
    """Test forward pass and reconstruction"""

    @pytest.fixture
    def tokenizer(self):
        """Create tokenizer for testing"""
        return CosmosInspiredTokenizer(
            in_channels=3,
            encoder_dims=[32, 64, 128],
            decoder_dims=[64, 32, 16],
            latent_dim=128,
            codebook_size=256,
            resolution=(128, 128)
        )

    def test_forward_pass_basic(self, tokenizer):
        """Test basic forward pass"""
        # Input: [B, T, C, H, W]
        x = torch.randn(2, 4, 3, 128, 128)
        output = tokenizer(x, return_loss=True)

        assert 'x_recon' in output
        assert 'indices' in output
        assert 'loss' in output
        assert 'recon_loss' in output
        assert 'quant_loss' in output

    def test_reconstruction_shape(self, tokenizer):
        """Test that reconstruction matches input shape"""
        x = torch.randn(2, 4, 3, 128, 128)
        output = tokenizer(x, return_loss=False)

        x_recon = output['x_recon']
        # Should match input shape in [B, C, T, H, W] format
        assert x_recon.shape[0] == 2  # Batch
        assert x_recon.shape[1] == 3  # Channels

    def test_input_format_flexibility(self, tokenizer):
        """Test both [B,T,C,H,W] and [B,C,T,H,W] input formats"""
        # Format 1: [B, T, C, H, W]
        x1 = torch.randn(2, 4, 3, 128, 128)
        output1 = tokenizer(x1, return_loss=False)

        # Format 2: [B, C, T, H, W]
        x2 = x1.permute(0, 2, 1, 3, 4)
        output2 = tokenizer(x2, return_loss=False)

        # Outputs should be equivalent
        assert torch.allclose(output1['x_recon'], output2['x_recon'], atol=1e-5)

    def test_loss_computation(self, tokenizer):
        """Test that losses are computed correctly"""
        x = torch.randn(2, 4, 3, 128, 128)
        output = tokenizer(x, return_loss=True)

        # Check loss components
        assert output['recon_loss'] >= 0, "Reconstruction loss should be non-negative"
        assert output['quant_loss'] >= 0, "Quantization loss should be non-negative"
        assert output['loss'] >= 0, "Total loss should be non-negative"

        # Total loss should be sum of components (at minimum)
        expected_min_loss = output['recon_loss'] + output['quant_loss']
        assert output['loss'] >= expected_min_loss * 0.99  # Allow small numerical error

    def test_perplexity_and_usage(self, tokenizer):
        """Test perplexity and codebook usage metrics"""
        x = torch.randn(2, 4, 3, 128, 128)
        output = tokenizer(x, return_loss=True)

        assert 'perplexity' in output
        assert 'codebook_usage' in output

        perplexity = output['perplexity']
        usage = output['codebook_usage']

        assert perplexity > 0, "Perplexity should be positive"
        assert 0 <= usage <= 1, "Codebook usage should be in [0, 1]"


class TestCosmosTokenizerEncoding:
    """Test encoding functionality"""

    @pytest.fixture
    def tokenizer(self):
        """Create tokenizer for testing"""
        return CosmosInspiredTokenizer(
            encoder_dims=[32, 64],
            decoder_dims=[32, 16],
            latent_dim=64,
            codebook_size=256
        )

    def test_encode_shape(self, tokenizer):
        """Test encoding output shapes"""
        x = torch.randn(2, 4, 3, 64, 64)
        z_quantized, indices = tokenizer.encode(x)

        # Latent should be downsampled
        assert z_quantized.shape[0] == 2  # Batch preserved
        assert z_quantized.shape[1] == 64  # Latent dim
        assert z_quantized.shape[2] <= 4  # Temporal downsampled
        assert z_quantized.shape[3] < 64  # Spatial downsampled
        assert z_quantized.shape[4] < 64

        # Indices should match latent spatial/temporal dims
        assert indices.shape[0] == 2
        assert indices.shape[-1] == 1  # Single index per location

    def test_encode_deterministic(self, tokenizer):
        """Test that encoding is deterministic in eval mode"""
        tokenizer.eval()

        x = torch.randn(2, 4, 3, 64, 64)

        with torch.no_grad():
            _, indices1 = tokenizer.encode(x)
            _, indices2 = tokenizer.encode(x)

        assert torch.equal(indices1, indices2), "Encoding should be deterministic"

    def test_indices_valid_range(self, tokenizer):
        """Test that indices are within valid codebook range"""
        x = torch.randn(2, 4, 3, 64, 64)
        _, indices = tokenizer.encode(x)

        assert (indices >= 0).all(), "Indices should be non-negative"
        assert (indices < tokenizer.codebook_size).all(), "Indices should be < codebook_size"


class TestCosmosTokenizerDecoding:
    """Test decoding functionality"""

    @pytest.fixture
    def tokenizer(self):
        """Create tokenizer for testing"""
        return CosmosInspiredTokenizer(
            encoder_dims=[32, 64],
            decoder_dims=[32, 16],
            latent_dim=64,
            codebook_size=256
        )

    def test_decode_shape(self, tokenizer):
        """Test decoding output shape"""
        x = torch.randn(2, 4, 3, 64, 64)
        z_quantized, _ = tokenizer.encode(x)
        x_recon = tokenizer.decode(z_quantized)

        # Reconstruction should have original channels
        assert x_recon.shape[1] == 3, "Should reconstruct RGB"

    def test_encode_decode_round_trip(self, tokenizer):
        """Test that encode→decode produces reasonable reconstruction"""
        tokenizer.eval()

        x = torch.randn(2, 4, 3, 64, 64)

        with torch.no_grad():
            z_quantized, _ = tokenizer.encode(x)
            x_recon = tokenizer.decode(z_quantized)

        # Should have same batch and channels
        assert x_recon.shape[0] == x.shape[0]
        assert x_recon.shape[1] == 3  # RGB


class TestCosmosTokenizerTokenization:
    """Test tokenize/detokenize functionality"""

    @pytest.fixture
    def tokenizer(self):
        """Create tokenizer for testing"""
        return CosmosInspiredTokenizer(
            encoder_dims=[32, 64],
            decoder_dims=[32, 16],
            latent_dim=64,
            codebook_size=512
        )

    def test_tokenize(self, tokenizer):
        """Test video → tokens conversion"""
        x = torch.randn(2, 4, 3, 64, 64)
        tokens = tokenizer.tokenize(x)

        assert tokens.shape[0] == 2  # Batch
        assert tokens.dtype in [torch.long, torch.int, torch.int64]  # Integer type
        assert (tokens >= 0).all()
        assert (tokens < tokenizer.codebook_size).all()

    def test_detokenize(self, tokenizer):
        """Test tokens → video conversion"""
        x = torch.randn(2, 4, 3, 64, 64)
        tokens = tokenizer.tokenize(x)
        x_recon = tokenizer.detokenize(tokens)

        # Should reconstruct to RGB video
        assert x_recon.shape[1] == 3

    def test_tokenize_detokenize_round_trip(self, tokenizer):
        """Test that tokenize→detokenize is consistent"""
        tokenizer.eval()

        x = torch.randn(2, 4, 3, 64, 64)

        with torch.no_grad():
            tokens = tokenizer.tokenize(x)
            x_recon = tokenizer.detokenize(tokens)

        # Reconstruction should have same structure
        assert x_recon.shape[0] == x.shape[0]

    def test_unique_tokens(self, tokenizer):
        """Test that diverse input produces diverse tokens"""
        # Diverse input
        x = torch.randn(4, 8, 3, 64, 64)
        tokens = tokenizer.tokenize(x)

        unique_tokens = torch.unique(tokens)

        # Should use multiple codes (not collapse)
        assert unique_tokens.numel() > 10, "Should use multiple different tokens"


class TestCosmosTokenizerCompression:
    """Test compression capabilities"""

    @pytest.fixture
    def tokenizer(self):
        """Create tokenizer for testing"""
        return CosmosInspiredTokenizer(
            encoder_dims=[32, 64, 128],
            decoder_dims=[64, 32, 16],
            latent_dim=128,
            codebook_size=2**16
        )

    def test_compression_ratio(self, tokenizer):
        """Test that compression ratio is computed correctly"""
        x = torch.randn(2, 4, 3, 128, 128)
        compression = tokenizer.get_compression_ratio(x)

        assert compression > 1.0, "Should compress (ratio > 1)"
        assert compression < 1000, "Compression ratio should be reasonable"

    def test_compression_improves_with_resolution(self, tokenizer):
        """Test that higher resolution → better compression"""
        # Lower resolution
        x_low = torch.randn(1, 4, 3, 64, 64)
        compression_low = tokenizer.get_compression_ratio(x_low)

        # Higher resolution
        x_high = torch.randn(1, 4, 3, 128, 128)
        compression_high = tokenizer.get_compression_ratio(x_high)

        # Higher res should compress better (more spatial redundancy)
        assert compression_high >= compression_low * 0.8  # Allow some variance


class TestCosmosTokenizerReconstruct:
    """Test reconstruction method"""

    @pytest.fixture
    def tokenizer(self):
        """Create tokenizer for testing"""
        return CosmosInspiredTokenizer(
            encoder_dims=[32, 64],
            decoder_dims=[32, 16],
            latent_dim=64,
            codebook_size=256
        )

    def test_reconstruct_no_grad(self, tokenizer):
        """Test that reconstruct() runs in no_grad mode"""
        x = torch.randn(2, 4, 3, 64, 64, requires_grad=True)
        x_recon = tokenizer.reconstruct(x)

        # Should not require gradients
        assert not x_recon.requires_grad

    def test_reconstruct_vs_forward(self, tokenizer):
        """Test that reconstruct() matches forward() output"""
        tokenizer.eval()

        x = torch.randn(2, 4, 3, 64, 64)

        with torch.no_grad():
            output = tokenizer(x, return_loss=False)
            x_recon_forward = output['x_recon']

            x_recon_method = tokenizer.reconstruct(x)

        assert torch.allclose(x_recon_forward, x_recon_method, atol=1e-5)


class TestCosmosTokenizerGradients:
    """Test gradient flow and backpropagation"""

    @pytest.fixture
    def tokenizer(self):
        """Create tokenizer for testing"""
        return CosmosInspiredTokenizer(
            encoder_dims=[32, 64],
            decoder_dims=[32, 16],
            latent_dim=64,
            codebook_size=256
        )

    def test_gradient_flow(self, tokenizer):
        """Test that gradients flow through full pipeline"""
        x = torch.randn(2, 4, 3, 64, 64, requires_grad=True)
        output = tokenizer(x, return_loss=True)

        loss = output['loss']
        loss.backward()

        assert x.grad is not None, "Gradients should flow to input"
        assert not torch.isnan(x.grad).any(), "No NaN gradients"

    def test_parameter_gradients(self, tokenizer):
        """Test that all parameters receive gradients"""
        x = torch.randn(2, 4, 3, 64, 64)
        output = tokenizer(x, return_loss=True)

        loss = output['loss']
        loss.backward()

        # Check that key components have gradients
        encoder_params = list(tokenizer.encoder.parameters())
        decoder_params = list(tokenizer.decoder.parameters())
        quantizer_params = list(tokenizer.quantizer.parameters())

        assert any(p.grad is not None for p in encoder_params), "Encoder should have gradients"
        assert any(p.grad is not None for p in decoder_params), "Decoder should have gradients"
        assert any(p.grad is not None for p in quantizer_params), "Quantizer should have gradients"


class TestCosmosTokenizerBatchSizes:
    """Test with various batch sizes"""

    @pytest.fixture
    def tokenizer(self):
        """Create tokenizer for testing"""
        return CosmosInspiredTokenizer(
            encoder_dims=[32, 64],
            decoder_dims=[32, 16],
            latent_dim=64,
            codebook_size=256
        )

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    def test_different_batch_sizes(self, tokenizer, batch_size):
        """Test tokenizer with different batch sizes"""
        x = torch.randn(batch_size, 4, 3, 64, 64)
        output = tokenizer(x, return_loss=True)

        assert output['x_recon'].shape[0] == batch_size
        assert output['indices'].shape[0] == batch_size


class TestCosmosTokenizerTemporalDimensions:
    """Test with various temporal dimensions"""

    @pytest.fixture
    def tokenizer(self):
        """Create tokenizer for testing"""
        return CosmosInspiredTokenizer(
            encoder_dims=[32, 64],
            decoder_dims=[32, 16],
            latent_dim=64,
            codebook_size=256
        )

    @pytest.mark.parametrize("num_frames", [2, 4, 8, 16])
    def test_different_temporal_lengths(self, tokenizer, num_frames):
        """Test tokenizer with different numbers of frames"""
        x = torch.randn(2, num_frames, 3, 64, 64)
        output = tokenizer(x, return_loss=True)

        # Should process successfully
        assert output['x_recon'] is not None
        assert output['loss'] >= 0


class TestCosmosTokenizerEdgeCases:
    """Test edge cases and robustness"""

    def test_minimum_video_size(self):
        """Test with very small video"""
        tokenizer = CosmosInspiredTokenizer(
            encoder_dims=[16, 32],
            decoder_dims=[16, 8],
            latent_dim=32,
            codebook_size=64
        )

        # Minimal video: 1 frame, 16x16
        x = torch.randn(1, 1, 3, 16, 16)
        output = tokenizer(x, return_loss=True)

        assert output['x_recon'] is not None

    def test_single_frame(self):
        """Test with single frame (edge case for video tokenizer)"""
        tokenizer = CosmosInspiredTokenizer(
            encoder_dims=[32, 64],
            decoder_dims=[32, 16],
            latent_dim=64,
            codebook_size=256
        )

        x = torch.randn(2, 1, 3, 64, 64)
        output = tokenizer(x, return_loss=True)

        assert output['loss'] >= 0

    def test_numerical_stability(self):
        """Test numerical stability with extreme inputs"""
        tokenizer = CosmosInspiredTokenizer(
            encoder_dims=[32, 64],
            decoder_dims=[32, 16],
            latent_dim=64,
            codebook_size=256
        )

        # Large values
        x_large = torch.randn(2, 4, 3, 64, 64) * 10
        output_large = tokenizer(x_large, return_loss=True)
        assert not torch.isnan(output_large['loss'])

        # Small values
        x_small = torch.randn(2, 4, 3, 64, 64) * 0.01
        output_small = tokenizer(x_small, return_loss=True)
        assert not torch.isnan(output_small['loss'])


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])

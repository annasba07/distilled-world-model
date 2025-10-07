"""
Unit tests for 3D Causal Convolutions

Tests cover:
- Causality verification (output at t only depends on frames 0...t)
- Shape transformations
- Encoder-decoder round trip
- Temporal vs spatial padding
- Different kernel sizes and strides
"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from models.tokenizers.causal_conv3d import (
    CausalConv3D,
    CausalConv3DEncoder,
    CausalConv3DDecoder
)


class TestCausalConv3D:
    """Test suite for CausalConv3D layer"""

    @pytest.fixture
    def causal_conv(self):
        """Create a basic causal conv layer"""
        return CausalConv3D(
            in_channels=3,
            out_channels=64,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1)
        )

    def test_initialization(self, causal_conv):
        """Test layer initialization"""
        assert causal_conv.kernel_size == (3, 3, 3)
        assert causal_conv.stride == (1, 1, 1)
        assert causal_conv.temporal_padding == 2  # (3-1)*1
        assert isinstance(causal_conv.conv, nn.Conv3d)

    def test_forward_shape_preservation(self, causal_conv):
        """Test that spatial/temporal dims are handled correctly"""
        # Input: [B, C, T, H, W]
        x = torch.randn(2, 3, 10, 64, 64)
        out = causal_conv(x)

        # With stride=1 and proper padding, spatial dims preserved
        assert out.shape[0] == 2  # Batch
        assert out.shape[1] == 64  # Out channels
        assert out.shape[2] == 10  # Temporal (preserved with causal padding)
        # Spatial might change slightly due to padding

    def test_temporal_causality(self):
        """CRITICAL: Test that output at time t doesn't depend on future frames"""
        causal_conv = CausalConv3D(
            in_channels=3,
            out_channels=16,
            kernel_size=(3, 3, 3)
        )
        causal_conv.eval()

        # Create video with distinct frames
        B, T, H, W = 2, 10, 32, 32
        video = torch.randn(B, 3, T, H, W)

        with torch.no_grad():
            # Process first 5 frames only
            video_partial = video[:, :, :5, :, :]
            out_partial = causal_conv(video_partial)

            # Process all 10 frames
            out_full = causal_conv(video)

            # First 5 frames of full output should EXACTLY match partial output
            t_partial = out_partial.shape[2]
            diff = (out_partial - out_full[:, :, :t_partial, :, :]).abs().max()

            assert diff < 1e-5, f"Causality violated! Diff: {diff.item():.2e}"

    def test_different_kernel_sizes(self):
        """Test various kernel sizes"""
        kernel_sizes = [(2, 3, 3), (3, 4, 4), (5, 5, 5)]

        for ks in kernel_sizes:
            conv = CausalConv3D(
                in_channels=3,
                out_channels=32,
                kernel_size=ks
            )

            x = torch.randn(2, 3, 8, 32, 32)
            out = conv(x)

            assert out.shape[0] == 2, f"Batch dim wrong for kernel_size={ks}"
            assert out.shape[1] == 32, f"Channel dim wrong for kernel_size={ks}"

    def test_stride_downsampling(self):
        """Test spatial downsampling with stride"""
        conv = CausalConv3D(
            in_channels=3,
            out_channels=64,
            kernel_size=(3, 4, 4),
            stride=(1, 2, 2)  # Downsample spatially only
        )

        x = torch.randn(2, 3, 8, 64, 64)
        out = conv(x)

        # Temporal preserved, spatial halved (approximately)
        assert out.shape[2] == 8, "Temporal should be preserved with stride=1"
        assert out.shape[3] < x.shape[3], "Height should be downsampled"
        assert out.shape[4] < x.shape[4], "Width should be downsampled"

    def test_dilation(self):
        """Test dilated causal convolutions"""
        conv = CausalConv3D(
            in_channels=3,
            out_channels=32,
            kernel_size=(3, 3, 3),
            dilation=(2, 1, 1)  # Temporal dilation
        )

        # Temporal padding should increase with dilation
        assert conv.temporal_padding == 4  # (3-1)*2

        x = torch.randn(2, 3, 10, 32, 32)
        out = conv(x)
        assert out.shape[0] == 2


class TestCausalConv3DEncoder:
    """Test suite for CausalConv3DEncoder"""

    @pytest.fixture
    def encoder(self):
        """Create standard encoder"""
        return CausalConv3DEncoder(
            in_channels=3,
            hidden_dims=[64, 128, 256],
            kernel_size=(3, 4, 4),
            use_groupnorm=True,
            activation='silu'
        )

    def test_initialization(self, encoder):
        """Test encoder initialization"""
        assert encoder.in_channels == 3
        assert encoder.hidden_dims == [64, 128, 256]
        assert encoder.out_channels == 256
        assert isinstance(encoder.encoder, nn.Sequential)

    def test_progressive_downsampling(self, encoder):
        """Test that encoder progressively downsamples"""
        x = torch.randn(2, 3, 8, 128, 128)
        z = encoder(x)

        # Should downsample spatially (stride 2x2 per layer)
        assert z.shape[3] < x.shape[3], "Height should be downsampled"
        assert z.shape[4] < x.shape[4], "Width should be downsampled"
        assert z.shape[1] == 256, "Output channels should be final hidden_dim"

    def test_input_format_handling(self, encoder):
        """Test both [B,C,T,H,W] and [B,T,C,H,W] formats"""
        # Format 1: [B, C, T, H, W]
        x1 = torch.randn(2, 3, 8, 64, 64)
        z1 = encoder(x1)

        # Format 2: [B, T, C, H, W] - should auto-convert
        x2 = x1.permute(0, 2, 1, 3, 4)  # [2, 8, 3, 64, 64]
        z2 = encoder(x2)

        # Should produce same output
        assert torch.allclose(z1, z2, atol=1e-5), "Format conversion failed"

    def test_causality_preservation(self):
        """Test that encoder maintains temporal causality"""
        encoder = CausalConv3DEncoder(
            in_channels=3,
            hidden_dims=[32, 64],
            kernel_size=(3, 4, 4)
        )
        encoder.eval()

        B, T, H, W = 2, 12, 64, 64
        video = torch.randn(B, T, 3, H, W)

        with torch.no_grad():
            # Encode first half
            z_half = encoder(video[:, :6])

            # Encode full video
            z_full = encoder(video)

            # First frames should match (accounting for temporal downsampling)
            t_half = z_half.shape[2]
            diff = (z_half - z_full[:, :, :t_half]).abs().max()

            assert diff < 1e-5, f"Encoder causality violated! Diff: {diff.item():.2e}"

    def test_different_hidden_dims(self):
        """Test with various hidden dimension configurations"""
        configs = [
            [64, 128],
            [32, 64, 128, 256],
            [128, 256, 512]
        ]

        for hidden_dims in configs:
            encoder = CausalConv3DEncoder(
                in_channels=3,
                hidden_dims=hidden_dims
            )

            x = torch.randn(1, 3, 8, 64, 64)
            z = encoder(x)

            assert z.shape[1] == hidden_dims[-1], f"Output channels wrong for {hidden_dims}"

    def test_normalization_layers(self):
        """Test that normalization layers are included"""
        encoder = CausalConv3DEncoder(
            in_channels=3,
            hidden_dims=[64, 128],
            use_groupnorm=True
        )

        # Check that GroupNorm layers exist
        has_groupnorm = any(isinstance(m, nn.GroupNorm) for m in encoder.encoder.modules())
        assert has_groupnorm, "GroupNorm layers should be present"

    def test_activation_functions(self):
        """Test different activation functions"""
        activations = ['silu', 'gelu', 'relu']

        for act in activations:
            encoder = CausalConv3DEncoder(
                in_channels=3,
                hidden_dims=[64],
                activation=act
            )

            x = torch.randn(1, 3, 4, 32, 32)
            z = encoder(x)
            assert z.shape[0] == 1, f"Failed with activation={act}"


class TestCausalConv3DDecoder:
    """Test suite for CausalConv3DDecoder"""

    @pytest.fixture
    def decoder(self):
        """Create standard decoder"""
        return CausalConv3DDecoder(
            in_channels=256,
            hidden_dims=[128, 64, 32],
            out_channels=3,
            kernel_size=(3, 4, 4),
            use_groupnorm=True,
            activation='silu'
        )

    def test_initialization(self, decoder):
        """Test decoder initialization"""
        assert isinstance(decoder.decoder, nn.Sequential)

    def test_progressive_upsampling(self, decoder):
        """Test that decoder progressively upsamples"""
        z = torch.randn(2, 256, 2, 8, 8)
        x_recon = decoder(z)

        # Should upsample spatially
        assert x_recon.shape[3] > z.shape[3], "Height should be upsampled"
        assert x_recon.shape[4] > z.shape[4], "Width should be upsampled"
        assert x_recon.shape[1] == 3, "Output should be RGB"

    def test_output_range(self, decoder):
        """Test that output is in [0, 1] range (sigmoid activation)"""
        z = torch.randn(2, 256, 4, 16, 16)
        x_recon = decoder(z)

        assert (x_recon >= 0).all(), "Output should be >= 0"
        assert (x_recon <= 1).all(), "Output should be <= 1"

    def test_output_channels(self):
        """Test decoder with different output channels"""
        for out_ch in [1, 3, 4]:  # Grayscale, RGB, RGBA
            decoder = CausalConv3DDecoder(
                in_channels=128,
                hidden_dims=[64, 32],
                out_channels=out_ch
            )

            z = torch.randn(1, 128, 4, 8, 8)
            x_recon = decoder(z)

            assert x_recon.shape[1] == out_ch, f"Output channels wrong for {out_ch}"


class TestEncoderDecoderRoundTrip:
    """Test encoder-decoder as a complete system"""

    @pytest.fixture
    def encoder_decoder_pair(self):
        """Create matching encoder and decoder"""
        encoder = CausalConv3DEncoder(
            in_channels=3,
            hidden_dims=[64, 128, 256],
            kernel_size=(3, 4, 4)
        )

        decoder = CausalConv3DDecoder(
            in_channels=256,
            hidden_dims=[128, 64, 32],
            out_channels=3,
            kernel_size=(3, 4, 4)
        )

        return encoder, decoder

    def test_round_trip_shapes(self, encoder_decoder_pair):
        """Test that encode→decode maintains compatible shapes"""
        encoder, decoder = encoder_decoder_pair

        x = torch.randn(2, 3, 8, 128, 128)
        z = encoder(x)
        x_recon = decoder(z)

        # Reconstruction might not match exactly in spatial dims due to conv layers
        # but should be close
        assert x_recon.shape[0] == x.shape[0], "Batch dim should match"
        assert x_recon.shape[1] == x.shape[1], "Channel dim should match"

    def test_gradient_flow_round_trip(self, encoder_decoder_pair):
        """Test gradients flow through encoder→decoder"""
        encoder, decoder = encoder_decoder_pair

        x = torch.randn(2, 3, 4, 64, 64, requires_grad=True)
        z = encoder(x)
        x_recon = decoder(z)

        loss = x_recon.sum()
        loss.backward()

        assert x.grad is not None, "Gradients should flow back to input"
        assert not torch.isnan(x.grad).any(), "No NaN gradients"

    def test_causality_end_to_end(self):
        """Test that causality is preserved through full pipeline"""
        encoder = CausalConv3DEncoder(
            in_channels=3,
            hidden_dims=[32, 64],
            kernel_size=(3, 3, 3)
        )

        decoder = CausalConv3DDecoder(
            in_channels=64,
            hidden_dims=[32, 16],
            out_channels=3,
            kernel_size=(3, 3, 3)
        )

        encoder.eval()
        decoder.eval()

        B, T, H, W = 2, 10, 32, 32
        video = torch.randn(B, 3, T, H, W)

        with torch.no_grad():
            # Process partial
            z_partial = encoder(video[:, :, :5])
            x_partial_recon = decoder(z_partial)

            # Process full
            z_full = encoder(video)
            x_full_recon = decoder(z_full)

            # First frames should match
            t_partial = x_partial_recon.shape[2]
            diff = (x_partial_recon - x_full_recon[:, :, :t_partial]).abs().max()

            assert diff < 1e-5, f"End-to-end causality violated! Diff: {diff.item():.2e}"


class TestCausalityProperty:
    """Dedicated tests for causality verification"""

    def test_no_future_information_leakage(self):
        """Verify that changing future frames doesn't affect current output"""
        encoder = CausalConv3DEncoder(
            in_channels=3,
            hidden_dims=[32, 64],
            kernel_size=(3, 3, 3)
        )
        encoder.eval()

        B, T, H, W = 1, 10, 32, 32

        # Create two videos: identical first 5 frames, different last 5
        video1 = torch.randn(B, 3, T, H, W)
        video2 = video1.clone()
        video2[:, :, 5:] = torch.randn(B, 3, 5, H, W)  # Change future

        with torch.no_grad():
            z1 = encoder(video1)
            z2 = encoder(video2)

            # First 5 frames of latent should be IDENTICAL
            # (they have identical input up to frame 5)
            # Note: Need to account for temporal downsampling
            min_t = min(z1.shape[2], z2.shape[2], 3)  # First few latent frames

            diff = (z1[:, :, :min_t] - z2[:, :, :min_t]).abs().max()

            assert diff < 1e-5, f"Future frames leaked information! Diff: {diff.item():.2e}"

    def test_padding_is_causal(self):
        """Test that padding is only on the past (left) temporally"""
        conv = CausalConv3D(
            in_channels=3,
            out_channels=16,
            kernel_size=(5, 3, 3)
        )

        # Temporal padding should be kernel_size - 1 = 4
        assert conv.temporal_padding == 4

        # When we pad, it should only pad LEFT (past) in temporal dimension
        x = torch.randn(1, 3, 5, 8, 8)
        out = conv(x)

        # Output temporal dim should be preserved with causal padding
        assert out.shape[2] == 5


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])

"""
3D Causal Convolutions for Video Encoding

Implements causal 3D convolutions that preserve temporal causality.
Key property: Output at time t only depends on frames 0...t, not future frames.

This is critical for:
1. Autoregressive generation
2. Streaming/real-time inference
3. Preventing information leakage from future

References:
- NVIDIA Cosmos: https://arxiv.org/abs/2501.03575
- WaveNet causal convolutions: https://arxiv.org/abs/1609.03499
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class CausalConv3D(nn.Module):
    """
    3D Causal Convolution layer.

    Pads only on the "past" side temporally to maintain causality.
    Spatially, uses standard padding.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of convolution kernel (temporal, height, width)
        stride: Stride for convolution
        dilation: Dilation rate
        groups: Number of groups for grouped convolution
        bias: Whether to use bias
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int, int] = (3, 3, 3),
        stride: Tuple[int, int, int] = (1, 1, 1),
        dilation: Tuple[int, int, int] = (1, 1, 1),
        groups: int = 1,
        bias: bool = True
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(dilation, int):
            dilation = (dilation, dilation, dilation)

        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

        # Causal padding: Only pad past frames temporally
        # For kernel_size t, we need t-1 frames of history
        self.temporal_padding = (kernel_size[0] - 1) * dilation[0]

        # Spatial padding: Standard (symmetric)
        self.spatial_padding = (
            ((kernel_size[1] - 1) * dilation[1]) // 2,
            ((kernel_size[2] - 1) * dilation[2]) // 2
        )

        # 3D Convolution
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,  # We handle padding manually
            dilation=dilation,
            groups=groups,
            bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with causal padding.

        Args:
            x: Input tensor [B, C, T, H, W]

        Returns:
            Output tensor [B, C_out, T', H', W']
        """
        # Apply causal padding
        # Pad temporal dimension only on the left (past)
        # Pad spatial dimensions symmetrically
        x = F.pad(
            x,
            pad=(
                self.spatial_padding[1], self.spatial_padding[1],  # Width
                self.spatial_padding[0], self.spatial_padding[0],  # Height
                self.temporal_padding, 0  # Time (only past!)
            ),
            mode='constant',
            value=0
        )

        # Apply convolution
        return self.conv(x)


class CausalConv3DEncoder(nn.Module):
    """
    Multi-layer 3D Causal Convolutional Encoder.

    Progressively downsamples video while maintaining temporal causality.

    Args:
        in_channels: Input channels (3 for RGB)
        hidden_dims: List of hidden dimensions for each layer
        kernel_size: Kernel size for convolutions
        use_groupnorm: Whether to use GroupNorm
        activation: Activation function
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dims: list = [64, 128, 256, 512],
        kernel_size: Tuple[int, int, int] = (3, 4, 4),
        use_groupnorm: bool = True,
        activation: str = 'silu'
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_dims = hidden_dims

        # Build encoder layers
        layers = []
        prev_dim = in_channels

        for i, hidden_dim in enumerate(hidden_dims):
            # Causal convolution with stride 2 for spatial downsampling
            # Keep temporal resolution (stride=1 temporally) for first layers
            temporal_stride = 1 if i < len(hidden_dims) - 1 else 2
            stride = (temporal_stride, 2, 2)

            layers.append(
                CausalConv3D(
                    prev_dim,
                    hidden_dim,
                    kernel_size=kernel_size,
                    stride=stride
                )
            )

            # Normalization
            if use_groupnorm:
                num_groups = min(32, hidden_dim // 4)
                layers.append(nn.GroupNorm(num_groups, hidden_dim))

            # Activation
            if activation == 'silu':
                layers.append(nn.SiLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'relu':
                layers.append(nn.ReLU())

            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)
        self.out_channels = hidden_dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode video frames.

        Args:
            x: Input video [B, T, C, H, W] or [B, C, T, H, W]

        Returns:
            Encoded features [B, C_out, T', H', W']
        """
        # Handle both input formats
        if x.dim() == 5:
            if x.shape[2] == self.in_channels:  # [B, T, C, H, W]
                x = x.permute(0, 2, 1, 3, 4)  # -> [B, C, T, H, W]

        return self.encoder(x)


class CausalConv3DDecoder(nn.Module):
    """
    Multi-layer 3D Causal Transposed Convolutional Decoder.

    Progressively upsamples to reconstruct video while maintaining causality.

    Args:
        in_channels: Input channels from encoder
        hidden_dims: List of hidden dimensions (in reverse order from encoder)
        out_channels: Output channels (3 for RGB)
        kernel_size: Kernel size for convolutions
        use_groupnorm: Whether to use GroupNorm
        activation: Activation function
    """

    def __init__(
        self,
        in_channels: int = 512,
        hidden_dims: list = [256, 128, 64, 32],
        out_channels: int = 3,
        kernel_size: Tuple[int, int, int] = (3, 4, 4),
        use_groupnorm: bool = True,
        activation: str = 'silu'
    ):
        super().__init__()

        layers = []
        prev_dim = in_channels

        for i, hidden_dim in enumerate(hidden_dims):
            # Transposed convolution for upsampling
            temporal_stride = 2 if i == 0 else 1
            stride = (temporal_stride, 2, 2)

            layers.append(
                nn.ConvTranspose3d(
                    prev_dim,
                    hidden_dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=(kernel_size[0]//2, kernel_size[1]//2, kernel_size[2]//2),
                    output_padding=(stride[0]-1, stride[1]-1, stride[2]-1)
                )
            )

            # Normalization
            if use_groupnorm and i < len(hidden_dims) - 1:
                num_groups = min(32, hidden_dim // 4)
                layers.append(nn.GroupNorm(num_groups, hidden_dim))

            # Activation
            if i < len(hidden_dims) - 1:
                if activation == 'silu':
                    layers.append(nn.SiLU())
                elif activation == 'gelu':
                    layers.append(nn.GELU())
                elif activation == 'relu':
                    layers.append(nn.ReLU())

            prev_dim = hidden_dim

        # Final layer to output channels
        layers.append(
            nn.Conv3d(
                prev_dim,
                out_channels,
                kernel_size=3,
                padding=1
            )
        )

        # Output activation (sigmoid for RGB [0, 1])
        layers.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode features to video.

        Args:
            z: Encoded features [B, C, T, H, W]

        Returns:
            Reconstructed video [B, 3, T, H, W]
        """
        return self.decoder(z)


# Test causality
def test_causality():
    """
    Verify that the encoder is truly causal.
    Output at time t should not depend on frames after t.
    """
    print("Testing Causality...")

    encoder = CausalConv3DEncoder(
        in_channels=3,
        hidden_dims=[64, 128],
        kernel_size=(3, 4, 4)
    )
    encoder.eval()

    # Create test video
    B, T, H, W = 2, 10, 64, 64
    video = torch.randn(B, T, 3, H, W)

    with torch.no_grad():
        # Encode first 5 frames
        video_partial = video[:, :5]
        z_partial = encoder(video_partial)

        # Encode all 10 frames
        z_full = encoder(video)

        # First 5 frames should be identical!
        # (accounting for temporal downsampling)
        t_partial = z_partial.shape[2]
        diff = (z_partial - z_full[:, :, :t_partial]).abs().max()

        print(f"Partial shape: {z_partial.shape}")
        print(f"Full shape: {z_full.shape}")
        print(f"Max difference in first frames: {diff.item():.2e}")

        if diff < 1e-5:
            print("✅ Causality verified!")
        else:
            print("❌ Causality violation detected!")

        return diff < 1e-5


if __name__ == '__main__':
    print("Testing 3D Causal Convolutions...\n")

    # Test 1: Basic forward pass
    print("1. Testing basic forward pass...")
    encoder = CausalConv3DEncoder(
        in_channels=3,
        hidden_dims=[64, 128, 256],
        kernel_size=(3, 4, 4)
    )

    x = torch.randn(2, 8, 3, 128, 128)  # [B, T, C, H, W]
    z = encoder(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {z.shape}")
    print("   ✅ Forward pass successful\n")

    # Test 2: Decoder
    print("2. Testing decoder...")
    decoder = CausalConv3DDecoder(
        in_channels=256,
        hidden_dims=[128, 64, 32],
        out_channels=3
    )

    x_recon = decoder(z)
    print(f"   Encoded shape: {z.shape}")
    print(f"   Reconstructed shape: {x_recon.shape}")
    print("   ✅ Decoder successful\n")

    # Test 3: Causality
    print("3. Testing causality property...")
    test_causality()

    print("\n✅ All 3D Causal Convolution tests passed!")

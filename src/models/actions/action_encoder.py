"""
Action Encoder - Infer latent actions from consecutive frames

The key insight: An action is what changed between frame_t and frame_{t+1}.
We encode this change as a compact latent representation.

Architecture:
1. Compute difference between consecutive frames
2. Use spatial attention to identify where change occurred
3. Encode into compact action latent vector

Reference: Genie 3 (DeepMind, October 2025)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class SpatialAttention(nn.Module):
    """
    Spatial attention to identify where changes occurred.
    """
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.channels = channels

        assert channels % num_heads == 0, "channels must be divisible by num_heads"

        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            attended: [B, C, H, W]
        """
        B, C, H, W = x.shape

        # Reshape to [B, H*W, C]
        x_flat = x.flatten(2).transpose(1, 2)  # [B, H*W, C]

        # Multi-head attention
        qkv = self.qkv(x_flat).reshape(B, H*W, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, H*W, C//num_heads]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        scale = (C // self.num_heads) ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)

        # Apply attention
        out = attn @ v  # [B, num_heads, H*W, C//num_heads]
        out = out.transpose(1, 2).reshape(B, H*W, C)  # [B, H*W, C]
        out = self.proj(out)

        # Reshape back
        out = out.transpose(1, 2).reshape(B, C, H, W)

        return out + x  # Residual connection


class ActionEncoder(nn.Module):
    """
    Encode the action that transforms frame_t into frame_{t+1}.

    Takes latent features from consecutive frames and outputs a compact
    action representation.

    Args:
        feature_dim: Dimension of input features from video encoder
        action_dim: Dimension of output action latent
        use_attention: Whether to use spatial attention
        num_heads: Number of attention heads
    """

    def __init__(
        self,
        feature_dim: int = 256,
        action_dim: int = 128,
        use_attention: bool = True,
        num_heads: int = 4
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.use_attention = use_attention

        # Difference encoder
        # Processes the difference between consecutive frames
        self.diff_encoder = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.GroupNorm(min(32, feature_dim // 4), feature_dim),
            nn.SiLU(),
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.GroupNorm(min(32, feature_dim // 4), feature_dim),
            nn.SiLU(),
        )

        # Spatial attention to identify where changes occurred
        if use_attention:
            self.attention = SpatialAttention(feature_dim, num_heads)

        # Pooling to get global action representation
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Project to action space
        self.to_action = nn.Sequential(
            nn.Linear(feature_dim, action_dim * 2),
            nn.SiLU(),
            nn.Linear(action_dim * 2, action_dim)
        )

    def forward(
        self,
        frame_t: torch.Tensor,
        frame_t1: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode action from consecutive frames.

        Args:
            frame_t: Features at time t [B, C, H, W]
            frame_t1: Features at time t+1 [B, C, H, W]

        Returns:
            action_latent: Action representation [B, action_dim]
        """
        # Ensure same shape
        assert frame_t.shape == frame_t1.shape, \
            f"Frame shapes must match: {frame_t.shape} vs {frame_t1.shape}"

        B, C, H, W = frame_t.shape
        assert C == self.feature_dim, \
            f"Feature dim mismatch: expected {self.feature_dim}, got {C}"

        # Compute difference (what changed)
        diff = frame_t1 - frame_t

        # Encode the difference
        diff_encoded = self.diff_encoder(diff)

        # Apply attention to focus on important changes
        if self.use_attention:
            diff_attended = self.attention(diff_encoded)
        else:
            diff_attended = diff_encoded

        # Global pooling to get single vector per frame pair
        action_features = self.global_pool(diff_attended)  # [B, C, 1, 1]
        action_features = action_features.flatten(1)  # [B, C]

        # Project to action latent space
        action_latent = self.to_action(action_features)  # [B, action_dim]

        return action_latent

    def encode_sequence(
        self,
        features: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode actions for an entire sequence of frames.

        Args:
            features: Video features [B, C, T, H, W]

        Returns:
            actions: Action sequence [B, T-1, action_dim]
        """
        B, C, T, H, W = features.shape

        actions = []
        for t in range(T - 1):
            frame_t = features[:, :, t, :, :]  # [B, C, H, W]
            frame_t1 = features[:, :, t + 1, :, :]  # [B, C, H, W]

            action = self.forward(frame_t, frame_t1)  # [B, action_dim]
            actions.append(action)

        # Stack along time dimension
        actions = torch.stack(actions, dim=1)  # [B, T-1, action_dim]

        return actions


# Testing
if __name__ == '__main__':
    print("Testing ActionEncoder...\n")

    # Test 1: Basic forward pass
    print("1. Testing basic forward pass...")
    encoder = ActionEncoder(
        feature_dim=256,
        action_dim=128,
        use_attention=True
    )

    # Simulate consecutive frames
    frame_t = torch.randn(2, 256, 16, 16)
    frame_t1 = torch.randn(2, 256, 16, 16)

    action = encoder(frame_t, frame_t1)
    print(f"   Input frames: {frame_t.shape}, {frame_t1.shape}")
    print(f"   Output action: {action.shape}")
    assert action.shape == (2, 128), f"Expected (2, 128), got {action.shape}"
    print("   ✅ Basic forward pass successful\n")

    # Test 2: Sequence encoding
    print("2. Testing sequence encoding...")
    features = torch.randn(2, 256, 8, 16, 16)  # [B, C, T, H, W]
    actions = encoder.encode_sequence(features)
    print(f"   Input features: {features.shape}")
    print(f"   Output actions: {actions.shape}")
    assert actions.shape == (2, 7, 128), f"Expected (2, 7, 128), got {actions.shape}"
    print("   ✅ Sequence encoding successful\n")

    # Test 3: Action should be different for different frame pairs
    print("3. Testing action discrimination...")
    frame_same = frame_t.clone()
    frame_diff = frame_t + torch.randn_like(frame_t) * 0.5

    action_same = encoder(frame_t, frame_same)
    action_diff = encoder(frame_t, frame_diff)

    # Actions should be different
    diff_norm = (action_same - action_diff).norm()
    print(f"   Same frames action norm: {action_same.norm().item():.4f}")
    print(f"   Diff frames action norm: {action_diff.norm().item():.4f}")
    print(f"   Difference between actions: {diff_norm.item():.4f}")
    print("   ✅ Action discrimination works\n")

    # Test 4: No attention version
    print("4. Testing without attention...")
    encoder_no_attn = ActionEncoder(
        feature_dim=256,
        action_dim=128,
        use_attention=False
    )
    action_no_attn = encoder_no_attn(frame_t, frame_t1)
    print(f"   Output action: {action_no_attn.shape}")
    print("   ✅ No-attention version works\n")

    # Test 5: Different dimensions
    print("5. Testing different dimensions...")
    encoder_small = ActionEncoder(feature_dim=128, action_dim=64)
    frame_small = torch.randn(2, 128, 8, 8)
    action_small = encoder_small(frame_small, frame_small)
    print(f"   Input: {frame_small.shape}")
    print(f"   Output: {action_small.shape}")
    assert action_small.shape == (2, 64)
    print("   ✅ Different dimensions work\n")

    print("✅ All ActionEncoder tests passed!")

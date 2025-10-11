#!/usr/bin/env python3
"""
Action Learning Demo - Phase 2 Week 5

Demonstrates learning actions from video using Phase 1 tokenizer
and Phase 2 action encoder/quantizer.

Shows the full pipeline:
1. Video → Tokenizer → Features
2. Features → Action Encoder → Action Latents
3. Action Latents → Action Quantizer → Discrete Actions

Usage:
    python examples/phase2/action_learning_demo.py
"""

import argparse
import sys
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from models.tokenizers import CosmosInspiredTokenizer
from models.actions import ActionEncoder, ActionQuantizer


def generate_sample_video(batch_size=2, num_frames=8, resolution=(256, 256)):
    """
    Generate a sample video with some temporal structure.

    In practice, you'd load real video data.
    """
    H, W = resolution
    video = torch.randn(batch_size, num_frames, 3, H, W)

    # Add some temporal structure (smooth transitions)
    for t in range(1, num_frames):
        # Each frame is similar to previous + some noise
        video[:, t] = video[:, t-1] * 0.7 + video[:, t] * 0.3

    # Normalize to [0, 1]
    video = (video - video.min()) / (video.max() - video.min() + 1e-8)

    return video


def main():
    parser = argparse.ArgumentParser(description="Action learning demo")
    parser.add_argument('--batch_size', type=int, default=2, help="Batch size")
    parser.add_argument('--num_frames', type=int, default=8, help="Number of frames")
    parser.add_argument('--resolution', type=int, nargs=2, default=[256, 256], help="Video resolution (H W)")
    parser.add_argument('--action_vocab', type=int, default=512, help="Action vocabulary size")
    parser.add_argument('--device', type=str, default='mps' if torch.backends.mps.is_available() else 'cpu')

    args = parser.parse_args()

    device = torch.device(args.device)
    resolution = tuple(args.resolution)

    print("="*80)
    print("Action Learning Demo - Phase 2 Week 5")
    print("="*80)
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Frames: {args.num_frames}")
    print(f"Resolution: {resolution[0]}×{resolution[1]}")
    print(f"Action vocabulary: {args.action_vocab}")
    print("="*80 + "\n")

    # Step 1: Create components
    print("Step 1: Creating components...")
    print("  - Cosmos Tokenizer (Phase 1)")
    tokenizer = CosmosInspiredTokenizer(
        in_channels=3,
        encoder_dims=[64, 128, 256],
        decoder_dims=[128, 64, 32],
        latent_dim=256,
        codebook_size=4096,
        resolution=resolution
    ).to(device)
    tokenizer.eval()

    print("  - Action Encoder (Phase 2)")
    action_encoder = ActionEncoder(
        feature_dim=256,
        action_dim=128,
        use_attention=True
    ).to(device)
    action_encoder.eval()

    print("  - Action Quantizer (Phase 2)")
    action_quantizer = ActionQuantizer(
        action_vocab_size=args.action_vocab,
        action_dim=128
    ).to(device)
    action_quantizer.eval()

    print("✅ Components created\n")

    # Step 2: Generate sample video
    print("Step 2: Generating sample video...")
    video = generate_sample_video(
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        resolution=resolution
    ).to(device)
    print(f"  Video shape: {video.shape}")
    print(f"  Video range: [{video.min():.3f}, {video.max():.3f}]")
    print("✅ Video generated\n")

    # Step 3: Encode video to features (Phase 1)
    print("Step 3: Encoding video to features (Phase 1 tokenizer)...")
    with torch.no_grad():
        # Get features from encoder (before quantization)
        # Video format: [B, T, C, H, W] → [B, C, T, H, W]
        video_permuted = video.permute(0, 2, 1, 3, 4)

        # Encode
        features = tokenizer.encoder(video_permuted)  # [B, C_enc, T', H', W']
        features = tokenizer.pre_quant_conv(features)  # [B, latent_dim, T', H', W']

    print(f"  Features shape: {features.shape}")
    print(f"  Compression: {resolution[0]}×{resolution[1]} → {features.shape[3]}×{features.shape[4]}")
    print("✅ Features encoded\n")

    # Step 4: Extract actions (Phase 2)
    print("Step 4: Extracting actions from consecutive frames...")
    with torch.no_grad():
        # Encode action sequence
        action_latents = action_encoder.encode_sequence(features)  # [B, T-1, action_dim]

    print(f"  Action latents shape: {action_latents.shape}")
    print(f"  Number of actions: {action_latents.shape[1]} (one per frame transition)")
    print("✅ Actions extracted\n")

    # Step 5: Quantize actions (Phase 2)
    print("Step 5: Quantizing actions to discrete vocabulary...")
    with torch.no_grad():
        action_quantized, info = action_quantizer(action_latents)

    print(f"  Quantized actions shape: {action_quantized.shape}")
    print(f"  Discrete action indices: {info['indices'].shape}")
    print(f"\nAction Statistics:")
    print(f"  - Unique actions used: {info['unique_actions']}/{args.action_vocab}")
    print(f"  - Diversity: {info['diversity']:.2%}")
    print(f"  - Perplexity: {info['perplexity'].item():.2f}")
    print("✅ Actions quantized\n")

    # Step 6: Show action sequence
    print("Step 6: Discrete action sequence:")
    for b in range(min(2, args.batch_size)):
        actions_b = info['indices'][b, :, 0].cpu().tolist()
        print(f"  Video {b}: {actions_b}")
    print()

    # Step 7: Decode actions back
    print("Step 7: Decoding actions back to continuous space...")
    with torch.no_grad():
        action_decoded = action_quantizer.decode(info['indices'])

    print(f"  Decoded actions shape: {action_decoded.shape}")

    # Measure reconstruction error
    recon_error = (action_decoded - action_latents).pow(2).mean()
    print(f"  Reconstruction MSE: {recon_error.item():.6f}")
    print("✅ Actions decoded\n")

    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✅ Video encoded to features: {video.shape} → {features.shape}")
    print(f"✅ Actions extracted: {features.shape} → {action_latents.shape}")
    print(f"✅ Actions quantized: {action_latents.shape} → {info['indices'].shape}")
    print(f"✅ Action diversity: {info['diversity']:.2%} ({info['unique_actions']}/{args.action_vocab} actions used)")
    print("="*80 + "\n")

    print("🎉 Action learning pipeline works!")
    print("\nNext steps:")
    print("  1. Week 6: Implement dynamics model (predict next frame from current + action)")
    print("  2. Week 7: Build interactive generation demo")
    print("  3. Train on real video data to learn meaningful actions")


if __name__ == '__main__':
    main()

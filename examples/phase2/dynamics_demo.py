#!/usr/bin/env python3
"""
Dynamics Model Demo - Phase 2 Week 6

Demonstrates action-conditioned video generation using the full Phase 2 pipeline:
1. Video → Tokenizer (Phase 1) → Frame tokens
2. Consecutive frames → Action Encoder → Action latents
3. Action latents → Action Quantizer → Discrete actions
4. (Frame_t, Action) → Dynamics Model → Frame_t+1 (NEW!)

This completes the action-conditioned generation loop.

Usage:
    python examples/phase2/dynamics_demo.py
    python examples/phase2/dynamics_demo.py --num_frames 16 --rollout_steps 5
"""

import argparse
import sys
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from models.tokenizers import CosmosInspiredTokenizer
from models.actions import ActionEncoder, ActionQuantizer, DynamicsModel


def generate_sample_video(batch_size=2, num_frames=8, resolution=(256, 256)):
    """
    Generate a sample video with temporal structure.
    In practice, you'd load real video data.
    """
    H, W = resolution
    video = torch.randn(batch_size, num_frames, 3, H, W)

    # Add temporal smoothness
    for t in range(1, num_frames):
        video[:, t] = video[:, t-1] * 0.7 + video[:, t] * 0.3

    # Normalize to [0, 1]
    video = (video - video.min()) / (video.max() - video.min() + 1e-8)

    return video


def main():
    parser = argparse.ArgumentParser(description="Dynamics model demo")
    parser.add_argument('--batch_size', type=int, default=2, help="Batch size")
    parser.add_argument('--num_frames', type=int, default=8, help="Number of frames")
    parser.add_argument('--resolution', type=int, nargs=2, default=[256, 256], help="Video resolution (H W)")
    parser.add_argument('--action_vocab', type=int, default=512, help="Action vocabulary size")
    parser.add_argument('--frame_vocab', type=int, default=4096, help="Frame vocabulary size")
    parser.add_argument('--rollout_steps', type=int, default=3, help="Number of rollout steps")
    parser.add_argument('--device', type=str, default='mps' if torch.backends.mps.is_available() else 'cpu')

    args = parser.parse_args()

    device = torch.device(args.device)
    resolution = tuple(args.resolution)

    print("="*80)
    print("Dynamics Model Demo - Phase 2 Week 6")
    print("="*80)
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Frames: {args.num_frames}")
    print(f"Resolution: {resolution[0]}×{resolution[1]}")
    print(f"Action vocabulary: {args.action_vocab}")
    print(f"Frame vocabulary: {args.frame_vocab}")
    print(f"Rollout steps: {args.rollout_steps}")
    print("="*80 + "\n")

    # ============================================================================
    # Step 1: Create all components
    # ============================================================================
    print("Step 1: Creating components...")
    print("  - Cosmos Tokenizer (Phase 1)")
    tokenizer = CosmosInspiredTokenizer(
        in_channels=3,
        encoder_dims=[64, 128, 256],
        decoder_dims=[128, 64, 32],
        latent_dim=256,
        codebook_size=args.frame_vocab,
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

    print("  - Dynamics Model (Phase 2 - NEW!)")
    dynamics_model = DynamicsModel(
        frame_vocab_size=args.frame_vocab,
        action_vocab_size=args.action_vocab,
        d_model=512,
        nhead=8,
        num_layers=6
    ).to(device)
    dynamics_model.eval()

    print("✅ All components created\n")

    # ============================================================================
    # Step 2: Generate sample video
    # ============================================================================
    print("Step 2: Generating sample video...")
    video = generate_sample_video(
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        resolution=resolution
    ).to(device)
    print(f"  Video shape: {video.shape}")
    print("✅ Video generated\n")

    # ============================================================================
    # Step 3: Encode video to tokens (Phase 1)
    # ============================================================================
    print("Step 3: Encoding video to frame tokens (Phase 1)...")
    with torch.no_grad():
        # Permute to [B, C, T, H, W]
        video_permuted = video.permute(0, 2, 1, 3, 4)

        # Encode to tokens
        z_quantized, indices = tokenizer.encode(video_permuted)
        # indices shape: [B, T', H', W', 1]
        # We need [B, T', seq_len] where seq_len = H' * W'

        # Flatten spatial dimensions: [B, T', H', W', 1] -> [B, T', H'*W']
        B, T, H, W, _ = indices.shape
        tokens = indices.squeeze(-1).reshape(B, T, H * W)

    print(f"  Frame tokens shape: {tokens.shape}")
    print(f"  Token range: [{tokens.min()}, {tokens.max()}]")
    print(f"  Spatial dimensions: {H}×{W} -> {H*W} tokens per frame")
    print("✅ Video encoded to tokens\n")

    # ============================================================================
    # Step 4: Extract actions from video (Phase 2)
    # ============================================================================
    print("Step 4: Extracting actions from video...")
    with torch.no_grad():
        # Get features
        features = tokenizer.encoder(video_permuted)
        features = tokenizer.pre_quant_conv(features)

        # Extract actions
        action_latents = action_encoder.encode_sequence(features)
        action_quantized, action_info = action_quantizer(action_latents)
        action_indices = action_info['indices']

    print(f"  Action latents: {action_latents.shape}")
    print(f"  Action indices: {action_indices.shape}")
    print(f"  Action diversity: {action_info['diversity']:.2%}")
    print("✅ Actions extracted\n")

    # ============================================================================
    # Step 5: Action-conditioned prediction (NEW!)
    # ============================================================================
    print("Step 5: Action-conditioned frame prediction...")
    with torch.no_grad():
        # Take first frame tokens
        frame_t = tokens[:, 0, :]  # [B, seq_len]
        action_t = action_indices[:, 0, :]  # [B, 1]

        # Predict next frame using dynamics model
        output = dynamics_model(frame_t, action_t)
        predicted_frame = output['predictions']

    print(f"  Input frame tokens: {frame_t.shape}")
    print(f"  Action token: {action_t.shape}")
    print(f"  Predicted frame tokens: {predicted_frame.shape}")
    print("✅ Next frame predicted\n")

    # ============================================================================
    # Step 6: Multi-step rollout (NEW!)
    # ============================================================================
    print(f"Step 6: Multi-step rollout ({args.rollout_steps} steps)...")
    with torch.no_grad():
        # Initial frame
        initial_frame = tokens[:, 0, :]  # [B, seq_len]

        # Action sequence (take first N actions)
        action_sequence = action_indices[:, :args.rollout_steps, :]  # [B, N, 1]

        # Rollout: repeatedly predict next frame
        predicted_frames = dynamics_model.rollout(
            initial_frame,
            action_sequence,
            deterministic=True
        )

    print(f"  Initial frame: {initial_frame.shape}")
    print(f"  Action sequence: {action_sequence.shape}")
    print(f"  Predicted frames: {predicted_frames.shape}")
    print("✅ Multi-step rollout complete\n")

    # ============================================================================
    # Step 7: Compare predicted vs ground truth tokens
    # ============================================================================
    print("Step 7: Comparing predicted vs ground truth tokens...")
    with torch.no_grad():
        # Ground truth tokens
        gt_tokens = tokens[:, 1:args.rollout_steps+1, :]  # [B, rollout_steps, seq_len]

        # Predicted tokens
        pred_tokens = predicted_frames  # [B, rollout_steps, seq_len]

        # Compute accuracy
        accuracy = (pred_tokens == gt_tokens).float().mean()

    print(f"  Ground truth tokens: {gt_tokens.shape}")
    print(f"  Predicted tokens: {pred_tokens.shape}")
    print(f"  Token accuracy: {accuracy.item():.2%} (expected low for untrained model)")
    print("✅ Comparison complete\n")

    # ============================================================================
    # Step 8: Show action sequence interpretation
    # ============================================================================
    print("Step 8: Action sequence for rollout:")
    for b in range(min(2, args.batch_size)):
        actions_b = action_sequence[b, :, 0].cpu().tolist()
        print(f"  Video {b}: {actions_b}")
    print()

    # ============================================================================
    # Summary
    # ============================================================================
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✅ Video encoded: {video.shape} → {tokens.shape}")
    print(f"✅ Actions extracted: {action_indices.shape} ({action_info['diversity']:.2%} diversity)")
    print(f"✅ Single-step prediction: (frame_t, action) → frame_t+1")
    print(f"✅ Multi-step rollout: {args.rollout_steps} frames generated")
    print(f"✅ Token accuracy: {accuracy.item():.2%} (untrained baseline)")
    print("="*80 + "\n")

    print("🎉 Dynamics model pipeline works!")
    print("\nKey capabilities demonstrated:")
    print("  1. ✅ Action-conditioned prediction: Given (frame, action) → predict next frame tokens")
    print("  2. ✅ Multi-step rollout: Apply action sequence to generate multiple frames")
    print("  3. ✅ Full integration: Works seamlessly with Phase 1 tokenizer and Phase 2 actions")
    print("  4. ✅ Evaluation: Can compare predicted vs ground truth tokens")
    print("\nNext steps:")
    print("  1. Train dynamics model on real video + extracted actions (expect >80% accuracy)")
    print("  2. Add token-to-video decoding for visual validation")
    print("  3. Build interactive generation demo (user controls actions)")
    print("  4. Train joint action encoder + dynamics model end-to-end")


if __name__ == '__main__':
    main()

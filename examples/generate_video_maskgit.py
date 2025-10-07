#!/usr/bin/env python3
"""
Example: Video Generation with MaskGIT + Cosmos Tokenizer

Demonstrates end-to-end video generation pipeline:
1. Cosmos tokenizer: Encode video to tokens
2. MaskGIT: Generate new token sequences (parallel, fast)
3. Cosmos tokenizer: Decode tokens back to video

This combines Week 1 (tokenizer) and Week 3 (MaskGIT) implementations.

Usage:
    python examples/generate_video_maskgit.py
    python examples/generate_video_maskgit.py --num_frames 16 --resolution 256 256
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.tokenizers import CosmosInspiredTokenizer
from models.generation import create_maskgit_generator
from utils.context import ConstantContextManager


def generate_video_example(
    num_frames: int = 8,
    resolution: tuple = (128, 128),
    device: torch.device = torch.device('cpu')
):
    """
    Example of video generation with MaskGIT + Cosmos tokenizer.

    Args:
        num_frames: Number of frames to generate
        resolution: Video resolution (H, W)
        device: Device to use
    """
    print("="*80)
    print("Video Generation Example: MaskGIT + Cosmos Tokenizer")
    print("="*80)
    print(f"Frames: {num_frames}")
    print(f"Resolution: {resolution}")
    print(f"Device: {device}")
    print("="*80 + "\n")

    # Step 1: Create tokenizer
    print("1. Creating Cosmos tokenizer...")
    tokenizer = CosmosInspiredTokenizer(
        in_channels=3,
        encoder_dims=[64, 128, 256],
        decoder_dims=[128, 64, 32],
        latent_dim=256,
        codebook_size=2**12,  # 4096 tokens (smaller for example)
        resolution=resolution
    ).to(device)
    tokenizer.eval()

    print(f"   Codebook size: {tokenizer.codebook_size:,}")
    print(f"   Latent dim: {tokenizer.latent_dim}")
    print("   ✅ Tokenizer ready\n")

    # Step 2: Generate sample input video (in practice, use real video)
    print("2. Creating sample input video...")
    batch_size = 1
    input_video = torch.randn(batch_size, num_frames, 3, *resolution).to(device)
    input_video = (input_video - input_video.min()) / (input_video.max() - input_video.min())

    print(f"   Video shape: {input_video.shape}")
    print("   ✅ Input video ready\n")

    # Step 3: Tokenize video
    print("3. Encoding video to tokens...")
    with torch.no_grad():
        # Encode to tokens
        tokens = tokenizer.tokenize(input_video)

    print(f"   Token shape: {tokens.shape}")
    print(f"   Token range: [{tokens.min()}, {tokens.max()}]")
    print(f"   Unique tokens: {tokens.unique().numel()}")

    # Flatten tokens for generation
    tokens_flat = tokens.reshape(batch_size, -1)
    seq_len = tokens_flat.shape[1]

    print(f"   Flattened tokens: {tokens_flat.shape}")
    print(f"   Compression ratio: {input_video.numel() / tokens_flat.numel():.1f}x")
    print("   ✅ Encoding complete\n")

    # Step 4: Create MaskGIT generator
    print("4. Creating MaskGIT generator...")
    generator = create_maskgit_generator(
        vocab_size=tokenizer.codebook_size,
        hidden_dim=256,
        num_layers=4,
        num_iterations=8,
        schedule_type='cosine'
    )
    generator.predictor.to(device)
    generator.predictor.eval()

    print(f"   Vocab size: {generator.predictor.vocab_size:,}")
    print(f"   Hidden dim: {generator.predictor.hidden_dim}")
    print(f"   Iterations: {generator.num_iterations}")
    print("   ✅ Generator ready\n")

    # Step 5: Generate new token sequence
    print("5. Generating new tokens with MaskGIT (parallel)...")

    # Option A: Generate from scratch
    with torch.no_grad():
        generated_tokens = generator.generate(
            batch_size=batch_size,
            seq_len=seq_len,
            device=device
        )

    print(f"   Generated tokens: {generated_tokens.shape}")
    print(f"   Unique tokens: {generated_tokens.unique().numel()}")
    print("   ✅ Generation complete\n")

    # Step 6: Decode tokens back to video
    print("6. Decoding tokens to video...")

    # Reshape tokens to spatial format
    # Need to figure out spatial dimensions from encoding
    # For simplicity, assume square
    spatial_tokens = int((seq_len / num_frames) ** 0.5)

    generated_tokens_reshaped = generated_tokens.reshape(
        batch_size, num_frames, spatial_tokens, spatial_tokens, 1
    )

    with torch.no_grad():
        generated_video = tokenizer.detokenize(generated_tokens_reshaped)

    print(f"   Generated video shape: {generated_video.shape}")
    print(f"   Video range: [{generated_video.min():.3f}, {generated_video.max():.3f}]")
    print("   ✅ Decoding complete\n")

    # Step 7: Compare with reconstruction
    print("7. Comparing with direct reconstruction...")

    with torch.no_grad():
        reconstructed_video = tokenizer.reconstruct(input_video)

    recon_error = F.mse_loss(reconstructed_video, input_video)
    print(f"   Reconstruction MSE: {recon_error.item():.6f}")
    print("   ✅ Comparison complete\n")

    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Input video:      {input_video.shape}")
    print(f"Tokens:           {tokens_flat.shape}")
    print(f"Generated video:  {generated_video.shape}")
    print(f"\nCompression:      {input_video.numel() / tokens_flat.numel():.1f}x")
    print(f"Tokenizer params: {sum(p.numel() for p in tokenizer.parameters()):,}")
    print(f"Generator params: {sum(p.numel() for p in generator.predictor.parameters()):,}")
    print("="*80)


def generate_with_context_manager(
    num_frames: int = 32,
    resolution: tuple = (128, 128),
    device: torch.device = torch.device('cpu')
):
    """
    Example with context manager for long videos.

    Combines all three weeks:
    - Week 1: Cosmos tokenizer
    - Week 2: Constant context manager
    - Week 3: MaskGIT generation

    Args:
        num_frames: Number of frames
        resolution: Resolution
        device: Device
    """
    print("\n" + "="*80)
    print("Long Video Generation with Context Manager")
    print("="*80)
    print(f"Frames: {num_frames}")
    print(f"Resolution: {resolution}")
    print("="*80 + "\n")

    # Create components
    print("Creating components...")
    tokenizer = CosmosInspiredTokenizer(
        encoder_dims=[64, 128],
        decoder_dims=[64, 32],
        latent_dim=128,
        codebook_size=2**10,
        resolution=resolution
    ).to(device)

    context_manager = ConstantContextManager(
        max_levels=5,
        level_capacity=16,
        compression_method='pool'
    )

    generator = create_maskgit_generator(
        vocab_size=tokenizer.codebook_size,
        hidden_dim=128,
        num_layers=2,
        num_iterations=8
    )
    generator.predictor.to(device)

    print("✅ Components ready\n")

    # Process in chunks
    print("Processing video in chunks...")
    chunk_size = 8

    for i in range(0, num_frames, chunk_size):
        print(f"  Chunk {i // chunk_size + 1}/{num_frames // chunk_size}...")

        # Generate video chunk (in practice, use real video)
        chunk = torch.randn(1, chunk_size, 3, *resolution).to(device)

        # Encode
        with torch.no_grad():
            features, _ = tokenizer.encode(chunk)

        # Update context (memory stays constant!)
        context_manager.update(features)

        # Get context for generation
        context = context_manager.get_context()

        # Generate next chunk using context
        # (In practice, condition generation on context)

    # Print context stats
    stats = context_manager.get_stats()
    print(f"\n✅ Processed {num_frames} frames")
    print(f"   Memory savings: {stats.get('memory_savings', 0):.1%}")
    print(f"   Total frames represented: {stats['total_frames_represented']}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Video generation example")
    parser.add_argument('--num_frames', type=int, default=8, help='Number of frames')
    parser.add_argument('--resolution', type=int, nargs=2, default=[128, 128], help='Resolution (H W)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--with_context', action='store_true', help='Use context manager for long video')

    args = parser.parse_args()

    device = torch.device(args.device)
    resolution = tuple(args.resolution)

    # Basic example
    generate_video_example(
        num_frames=args.num_frames,
        resolution=resolution,
        device=device
    )

    # With context manager
    if args.with_context:
        generate_with_context_manager(
            num_frames=32,
            resolution=resolution,
            device=device
        )


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Check VQ-VAE Tokenizer Health

Diagnose potential issues with the VQ-VAE tokenizer:
1. Codebook collapse (how many codes are actually used)
2. Reconstruction quality
3. Token entropy
4. Spatial structure preservation

Usage:
    python scripts/check_vqvae_health.py --data_dir ./data/mario_sequences_100x --num_samples 100
"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.tokenizers import CosmosInspiredTokenizer


def main():
    parser = argparse.ArgumentParser(description='Check VQ-VAE Health')
    parser.add_argument('--data_dir', type=str, default='./data/mario_sequences_100x',
                        help='Directory containing .npz sequence files')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of sequences to sample')
    parser.add_argument('--device', type=str, default='mps',
                        help='Device to use (cpu, cuda, mps)')
    args = parser.parse_args()

    # Setup
    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'mps' else 'cpu')
    print(f"Using device: {device}\n")

    # Initialize tokenizer
    print("=" * 80)
    print("Initializing VQ-VAE Tokenizer...")
    print("=" * 80)
    tokenizer = CosmosInspiredTokenizer(
        codebook_size=512,
        latent_dim=64,
        resolution=(256, 256)
    ).to(device)
    tokenizer.eval()

    print(f"Codebook size: {tokenizer.codebook_size}")
    print(f"Latent dim: {tokenizer.latent_dim}\n")

    # Load sample data
    data_dir = Path(args.data_dir)
    npz_files = sorted(list(data_dir.glob('*.npz')))[:args.num_samples]
    print(f"Loaded {len(npz_files)} sample sequences\n")

    # Collect statistics
    all_tokens = []
    reconstruction_errors = []

    print("=" * 80)
    print("Analyzing VQ-VAE Performance...")
    print("=" * 80)

    with torch.no_grad():
        for npz_file in tqdm(npz_files, desc="Processing"):
            try:
                # Load sequence
                data = np.load(npz_file, allow_pickle=True)
                frames = data['frames']  # [T, H, W, 3]

                # Take first frame only for speed
                frame = frames[0]  # [H, W, 3]

                # Convert to tensor
                frame_tensor = torch.from_numpy(frame).float().to(device)
                frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]

                # Normalize
                if frame_tensor.max() > 1.0:
                    frame_tensor = frame_tensor / 255.0

                # Encode
                z_quantized, indices = tokenizer.encode(frame_tensor.unsqueeze(2))  # Add T dim
                # indices: [1, 1, H, W, 1]

                # Collect tokens
                tokens = indices.squeeze().cpu().numpy().flatten()
                all_tokens.extend(tokens.tolist())

                # Reconstruct
                reconstructed = tokenizer.decode(z_quantized)  # [1, 3, 1, H, W]
                reconstructed = reconstructed.squeeze(2)  # [1, 3, H, W]

                # Calculate reconstruction error
                mse = torch.nn.functional.mse_loss(reconstructed, frame_tensor).item()
                reconstruction_errors.append(mse)

            except Exception as e:
                print(f"Failed on {npz_file.name}: {e}")
                continue

    # Analyze results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    # 1. Codebook usage
    token_counts = Counter(all_tokens)
    num_used_codes = len(token_counts)
    usage_percentage = (num_used_codes / tokenizer.codebook_size) * 100

    print(f"\n1. CODEBOOK USAGE:")
    print(f"   Total codes: {tokenizer.codebook_size}")
    print(f"   Used codes: {num_used_codes}")
    print(f"   Usage: {usage_percentage:.1f}%")

    if usage_percentage < 20:
        print(f"   ⚠️  WARNING: Severe codebook collapse! (<20% used)")
    elif usage_percentage < 50:
        print(f"   ⚠️  WARNING: Moderate codebook collapse (20-50% used)")
    elif usage_percentage < 80:
        print(f"   ✓  Good codebook usage (50-80% used)")
    else:
        print(f"   ✓✓ Excellent codebook usage (>80% used)")

    # 2. Reconstruction quality
    avg_mse = np.mean(reconstruction_errors)
    print(f"\n2. RECONSTRUCTION QUALITY:")
    print(f"   Average MSE: {avg_mse:.6f}")

    if avg_mse < 0.001:
        print(f"   ✓✓ Excellent reconstruction (<0.001)")
    elif avg_mse < 0.01:
        print(f"   ✓  Good reconstruction (0.001-0.01)")
    elif avg_mse < 0.05:
        print(f"   ⚠️  Moderate reconstruction (0.01-0.05)")
    else:
        print(f"   ⚠️  WARNING: Poor reconstruction (>0.05)")

    # 3. Token entropy
    token_probs = np.array([token_counts[i] / len(all_tokens)
                           for i in range(tokenizer.codebook_size)])
    # Add small epsilon to avoid log(0)
    token_probs = token_probs + 1e-10
    entropy = -np.sum(token_probs * np.log2(token_probs + 1e-10))
    max_entropy = np.log2(tokenizer.codebook_size)
    entropy_ratio = entropy / max_entropy

    print(f"\n3. TOKEN ENTROPY:")
    print(f"   Entropy: {entropy:.2f} bits")
    print(f"   Max entropy: {max_entropy:.2f} bits")
    print(f"   Ratio: {entropy_ratio:.2%}")

    if entropy_ratio < 0.5:
        print(f"   ⚠️  WARNING: Very low entropy - tokenizer is too lossy!")
    elif entropy_ratio < 0.7:
        print(f"   ⚠️  Low entropy - some information loss")
    else:
        print(f"   ✓  Good entropy - preserving information")

    # 4. Token distribution visualization
    print(f"\n4. TOKEN DISTRIBUTION:")
    sorted_counts = sorted(token_counts.values(), reverse=True)
    top_10_count = sum(sorted_counts[:10])
    top_10_pct = (top_10_count / len(all_tokens)) * 100

    print(f"   Top 10 codes: {top_10_pct:.1f}% of all tokens")
    if top_10_pct > 50:
        print(f"   ⚠️  WARNING: Top 10 codes dominate (>{top_10_pct:.0f}%)")
    else:
        print(f"   ✓  Balanced distribution")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    issues = []
    if usage_percentage < 50:
        issues.append("Codebook collapse detected")
    if avg_mse > 0.05:
        issues.append("Poor reconstruction quality")
    if entropy_ratio < 0.7:
        issues.append("Low token entropy (lossy)")
    if top_10_pct > 50:
        issues.append("Unbalanced token distribution")

    if issues:
        print("⚠️  ISSUES DETECTED:")
        for issue in issues:
            print(f"   - {issue}")
        print("\nRECOMMENDATION: Consider switching to FSQ tokenizer or retraining VQ-VAE")
    else:
        print("✓✓ VQ-VAE appears healthy!")
        print("    Issues likely lie elsewhere (model architecture, training setup, etc.)")

    print()


if __name__ == '__main__':
    main()

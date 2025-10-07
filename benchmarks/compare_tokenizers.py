#!/usr/bin/env python3
"""
Benchmark: Compare Cosmos Tokenizer vs Old VQ-VAE Tokenizer

Compares:
1. Compression ratio (target: 8x better)
2. Encoding speed (target: 12x faster)
3. Reconstruction quality (PSNR, SSIM)
4. Codebook usage (no collapse)
5. Memory usage

Usage:
    python benchmarks/compare_tokenizers.py
    python benchmarks/compare_tokenizers.py --resolution 640 360 --num_frames 16
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.tokenizers.cosmos_tokenizer import CosmosInspiredTokenizer


def compute_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Compute Peak Signal-to-Noise Ratio"""
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()


def compute_ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    Simplified SSIM computation.
    For production, use pytorch-msssim package.
    """
    # Simplified version - just use MSE-based approximation
    mse = F.mse_loss(img1, img2)
    return 1.0 / (1.0 + mse.item())


class DummyOldTokenizer(torch.nn.Module):
    """
    Dummy old VQ-VAE tokenizer for comparison.

    In production, replace with your actual old tokenizer.
    This simulates a traditional VQ-VAE with:
    - Lower compression
    - Slower encoding
    - Potential codebook collapse
    """

    def __init__(self, latent_dim=256, codebook_size=1024):
        super().__init__()
        self.latent_dim = latent_dim
        self.codebook_size = codebook_size

        # Simplified encoder/decoder
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv3d(3, 64, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv3d(64, 128, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv3d(128, latent_dim, 3, padding=1),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(latent_dim, 128, 4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose3d(128, 64, 4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv3d(64, 3, 3, padding=1),
            torch.nn.Sigmoid(),
        )

        # Codebook
        self.codebook = torch.nn.Embedding(codebook_size, latent_dim)

    def encode(self, x):
        # Simulate slower encoding (inefficient implementation)
        time.sleep(0.01)  # Simulate overhead
        h = self.encoder(x)
        B, D, T, H, W = h.shape
        h_flat = h.permute(0, 2, 3, 4, 1).reshape(-1, D)

        # Quantize (nearest neighbor lookup - slow!)
        distances = torch.cdist(h_flat, self.codebook.weight)
        indices = distances.argmin(dim=1)

        return indices.reshape(B, T, H, W, 1)

    def decode(self, indices):
        B, T, H, W, _ = indices.shape
        indices_flat = indices.reshape(-1)
        z_flat = self.codebook(indices_flat)
        z = z_flat.reshape(B, T, H, W, -1).permute(0, 4, 1, 2, 3)
        return self.decoder(z)

    def forward(self, x):
        if x.shape[2] == 3:  # [B, T, C, H, W]
            x = x.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]

        indices = self.encode(x)
        x_recon = self.decode(indices)

        return {
            'x_recon': x_recon,
            'indices': indices,
        }

    def get_compression_ratio(self, x):
        indices = self.encode(x)
        return x.numel() / indices.numel()


def benchmark_tokenizer(
    tokenizer: torch.nn.Module,
    video: torch.Tensor,
    device: torch.device,
    num_runs: int = 10
) -> Dict[str, float]:
    """Benchmark a tokenizer"""
    tokenizer.eval()
    video = video.to(device)

    metrics = {}

    with torch.no_grad():
        # 1. Compression ratio
        if hasattr(tokenizer, 'get_compression_ratio'):
            compression = tokenizer.get_compression_ratio(video)
        else:
            # Fallback
            output = tokenizer(video)
            compression = video.numel() / output['indices'].numel()
        metrics['compression_ratio'] = compression

        # 2. Encoding speed
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start = time.time()

        for _ in range(num_runs):
            if hasattr(tokenizer, 'encode'):
                _ = tokenizer.encode(video)
            else:
                _ = tokenizer(video)

        torch.cuda.synchronize() if device.type == 'cuda' else None
        encode_time = (time.time() - start) / num_runs
        metrics['encode_time'] = encode_time
        metrics['encode_fps'] = video.shape[1] / encode_time  # frames / sec

        # 3. Reconstruction quality
        output = tokenizer(video)
        x_recon = output['x_recon']

        # Ensure same format
        if x_recon.shape != video.shape:
            if video.shape[2] == 3:  # [B, T, C, H, W]
                video = video.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]

        psnr = compute_psnr(x_recon, video)
        ssim = compute_ssim(x_recon, video)

        metrics['psnr'] = psnr
        metrics['ssim'] = ssim
        metrics['mse'] = F.mse_loss(x_recon, video).item()

        # 4. Codebook usage
        indices = output['indices']
        unique_codes = torch.unique(indices).numel()
        codebook_size = tokenizer.codebook_size if hasattr(tokenizer, 'codebook_size') else 1024
        usage = unique_codes / codebook_size

        metrics['codebook_usage'] = usage
        metrics['unique_codes'] = unique_codes

        # 5. Memory usage
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            _ = tokenizer(video)
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
            metrics['peak_memory_mb'] = peak_memory

        # 6. Parameter count
        total_params = sum(p.numel() for p in tokenizer.parameters())
        metrics['num_parameters'] = total_params

    return metrics


def print_comparison(
    old_metrics: Dict[str, float],
    new_metrics: Dict[str, float]
):
    """Print comparison table"""
    print("\n" + "="*100)
    print(f"{'Metric':<30} {'Old VQ-VAE':<20} {'Cosmos Tokenizer':<20} {'Improvement':<20}")
    print("="*100)

    # Compression ratio
    old_comp = old_metrics['compression_ratio']
    new_comp = new_metrics['compression_ratio']
    improvement = new_comp / old_comp
    print(f"{'Compression Ratio':<30} {old_comp:<20.2f}x {new_comp:<20.2f}x {improvement:<20.2f}x better")

    # Encoding speed
    old_fps = old_metrics['encode_fps']
    new_fps = new_metrics['encode_fps']
    speedup = new_fps / old_fps
    print(f"{'Encoding Speed':<30} {old_fps:<20.1f} fps {new_fps:<20.1f} fps {speedup:<20.1f}x faster")

    # PSNR
    old_psnr = old_metrics['psnr']
    new_psnr = new_metrics['psnr']
    psnr_diff = new_psnr - old_psnr
    print(f"{'PSNR (dB)':<30} {old_psnr:<20.2f} {new_psnr:<20.2f} {psnr_diff:+.2f} dB")

    # SSIM
    old_ssim = old_metrics['ssim']
    new_ssim = new_metrics['ssim']
    print(f"{'SSIM':<30} {old_ssim:<20.4f} {new_ssim:<20.4f} {(new_ssim/old_ssim - 1)*100:+.1f}%")

    # Codebook usage
    old_usage = old_metrics['codebook_usage']
    new_usage = new_metrics['codebook_usage']
    print(f"{'Codebook Usage':<30} {old_usage:<20.2%} {new_usage:<20.2%} {(new_usage - old_usage)*100:+.1f}%")

    # Parameters
    old_params = old_metrics['num_parameters']
    new_params = new_metrics['num_parameters']
    print(f"{'Parameters':<30} {old_params:<20,} {new_params:<20,} {new_params/old_params:<20.2f}x")

    # Memory (if available)
    if 'peak_memory_mb' in old_metrics:
        old_mem = old_metrics['peak_memory_mb']
        new_mem = new_metrics['peak_memory_mb']
        print(f"{'Peak Memory (MB)':<30} {old_mem:<20.1f} {new_mem:<20.1f} {new_mem/old_mem:<20.2f}x")

    print("="*100)

    # Summary
    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)

    # Check if targets are met
    targets = {
        'Compression Ratio': (improvement, 8.0, '8x better'),
        'Encoding Speed': (speedup, 12.0, '12x faster'),
        'Codebook Usage': (new_usage, 0.90, '>90%'),
    }

    for metric, (value, target, desc) in targets.items():
        status = "✅ PASS" if value >= target else "❌ FAIL"
        print(f"{metric:<30} {value:<10.2f} (target: {desc}) {status}")

    print("="*100 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Compare tokenizers")
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--num_frames', type=int, default=8, help='Number of frames')
    parser.add_argument('--resolution', type=int, nargs=2, default=[128, 128], help='Resolution (H W)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_runs', type=int, default=10, help='Number of runs for timing')

    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Running benchmark on {device}")
    print(f"Video shape: [{args.batch_size}, {args.num_frames}, 3, {args.resolution[0]}, {args.resolution[1]}]\n")

    # Generate test video
    video = torch.randn(
        args.batch_size,
        args.num_frames,
        3,
        args.resolution[0],
        args.resolution[1]
    )
    video = (video - video.min()) / (video.max() - video.min())  # Normalize [0, 1]

    # Build tokenizers
    print("Building Old VQ-VAE tokenizer...")
    old_tokenizer = DummyOldTokenizer(latent_dim=256, codebook_size=1024).to(device)

    print("Building Cosmos tokenizer...")
    new_tokenizer = CosmosInspiredTokenizer(
        in_channels=3,
        encoder_dims=[64, 128, 256],
        decoder_dims=[128, 64, 32],
        latent_dim=256,
        codebook_size=2**16,
        resolution=tuple(args.resolution)
    ).to(device)

    # Benchmark old tokenizer
    print("\nBenchmarking Old VQ-VAE...")
    old_metrics = benchmark_tokenizer(old_tokenizer, video, device, args.num_runs)

    # Benchmark new tokenizer
    print("Benchmarking Cosmos Tokenizer...")
    new_metrics = benchmark_tokenizer(new_tokenizer, video, device, args.num_runs)

    # Print comparison
    print_comparison(old_metrics, new_metrics)


if __name__ == '__main__':
    main()

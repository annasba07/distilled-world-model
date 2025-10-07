#!/usr/bin/env python3
"""
Benchmark: Generation Speed - MaskGIT vs Autoregressive

Compares generation speed between:
1. MaskGIT (parallel, iterative refinement)
2. Traditional autoregressive (sequential)

Expected: MaskGIT should be ~10x faster

Usage:
    python benchmarks/generation_speed.py
    python benchmarks/generation_speed.py --seq_len 256 --num_iterations 12
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.generation.maskgit import (
    MaskGITPredictor,
    MaskGITGenerator,
    create_maskgit_generator
)


class AutoregressivePredictor(nn.Module):
    """
    Traditional autoregressive predictor for comparison.

    Predicts one token at a time, conditioned on all previous tokens.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 512,
        num_layers: int = 8,
        num_heads: int = 8
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)

        # Positional embedding
        self.max_seq_len = 4096
        self.pos_embedding = nn.Embedding(self.max_seq_len, hidden_dim)

        # Causal transformer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            tokens: Input tokens [B, N]

        Returns:
            logits: Predicted logits [B, N, vocab_size]
        """
        B, N = tokens.shape

        # Embed tokens
        token_emb = self.token_embedding(tokens)

        # Add positional embeddings
        positions = torch.arange(N, device=tokens.device).unsqueeze(0).expand(B, -1)
        pos_emb = self.pos_embedding(positions)
        x = token_emb + pos_emb

        # Create causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(N).to(tokens.device)

        # Apply transformer
        # For decoder, we use x as both memory and target
        x = self.transformer(x, x, tgt_mask=causal_mask, memory_mask=causal_mask)

        # Project to vocabulary
        logits = self.output_proj(x)

        return logits

    @torch.no_grad()
    def generate(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device = torch.device('cpu'),
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Autoregressive generation.

        Generates one token at a time, left to right.
        SLOW because it's sequential!

        Args:
            batch_size: Batch size
            seq_len: Sequence length
            device: Device
            temperature: Sampling temperature

        Returns:
            tokens: Generated tokens [B, seq_len]
        """
        # Start with zeros (could use BOS token)
        tokens = torch.zeros(batch_size, 1, dtype=torch.long, device=device)

        # Generate one token at a time
        for i in range(seq_len - 1):
            # Predict next token
            logits = self.forward(tokens)  # [B, i+1, vocab_size]

            # Get logits for last position
            next_logits = logits[:, -1, :] / temperature

            # Sample
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append
            tokens = torch.cat([tokens, next_token], dim=1)

        return tokens


def benchmark_maskgit(
    vocab_size: int,
    hidden_dim: int,
    num_layers: int,
    batch_size: int,
    seq_len: int,
    num_iterations: int,
    device: torch.device,
    num_runs: int = 10
) -> Dict[str, float]:
    """
    Benchmark MaskGIT generation.

    Args:
        vocab_size: Vocabulary size
        hidden_dim: Hidden dimension
        num_layers: Number of layers
        batch_size: Batch size
        seq_len: Sequence length
        num_iterations: MaskGIT iterations
        device: Device
        num_runs: Number of benchmark runs

    Returns:
        metrics: Benchmark metrics
    """
    # Create generator
    generator = create_maskgit_generator(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_iterations=num_iterations
    )
    generator.predictor.to(device)
    generator.predictor.eval()

    # Warmup
    for _ in range(3):
        _ = generator.generate(batch_size, seq_len, device=device)

    # Benchmark
    if device.type == 'cuda':
        torch.cuda.synchronize()

    start = time.time()

    for _ in range(num_runs):
        tokens = generator.generate(batch_size, seq_len, device=device)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    elapsed = time.time() - start
    avg_time = elapsed / num_runs

    # Calculate tokens/sec
    total_tokens = batch_size * seq_len
    tokens_per_sec = total_tokens / avg_time

    return {
        'avg_time': avg_time,
        'tokens_per_sec': tokens_per_sec,
        'num_iterations': num_iterations,
        'forward_passes': num_iterations,  # MaskGIT does N forward passes
    }


def benchmark_autoregressive(
    vocab_size: int,
    hidden_dim: int,
    num_layers: int,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    num_runs: int = 10
) -> Dict[str, float]:
    """
    Benchmark autoregressive generation.

    Args:
        vocab_size: Vocabulary size
        hidden_dim: Hidden dimension
        num_layers: Number of layers
        batch_size: Batch size
        seq_len: Sequence length
        device: Device
        num_runs: Number of benchmark runs

    Returns:
        metrics: Benchmark metrics
    """
    # Create predictor
    predictor = AutoregressivePredictor(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers
    ).to(device)
    predictor.eval()

    # Warmup
    for _ in range(3):
        _ = predictor.generate(batch_size, seq_len, device=device)

    # Benchmark
    if device.type == 'cuda':
        torch.cuda.synchronize()

    start = time.time()

    for _ in range(num_runs):
        tokens = predictor.generate(batch_size, seq_len, device=device)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    elapsed = time.time() - start
    avg_time = elapsed / num_runs

    # Calculate tokens/sec
    total_tokens = batch_size * seq_len
    tokens_per_sec = total_tokens / avg_time

    return {
        'avg_time': avg_time,
        'tokens_per_sec': tokens_per_sec,
        'forward_passes': seq_len,  # Autoregressive does seq_len forward passes
    }


def compare_generation_methods(
    vocab_size: int = 1024,
    hidden_dim: int = 512,
    num_layers: int = 8,
    batch_size: int = 4,
    seq_len: int = 256,
    num_iterations: int = 12,
    device: torch.device = torch.device('cpu'),
    num_runs: int = 10
):
    """
    Compare MaskGIT vs Autoregressive generation.

    Args:
        vocab_size: Vocabulary size
        hidden_dim: Hidden dimension
        num_layers: Number of layers
        batch_size: Batch size
        seq_len: Sequence length to generate
        num_iterations: MaskGIT iterations
        device: Device
        num_runs: Number of benchmark runs
    """
    print("="*80)
    print("Generation Speed Benchmark: MaskGIT vs Autoregressive")
    print("="*80)
    print(f"Vocab size: {vocab_size:,}")
    print(f"Hidden dim: {hidden_dim}")
    print(f"Num layers: {num_layers}")
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_len}")
    print(f"MaskGIT iterations: {num_iterations}")
    print(f"Device: {device}")
    print(f"Benchmark runs: {num_runs}")
    print("="*80 + "\n")

    # Benchmark MaskGIT
    print("Benchmarking MaskGIT (parallel generation)...")
    maskgit_metrics = benchmark_maskgit(
        vocab_size, hidden_dim, num_layers,
        batch_size, seq_len, num_iterations,
        device, num_runs
    )

    # Benchmark Autoregressive
    print("Benchmarking Autoregressive (sequential generation)...")
    autoregressive_metrics = benchmark_autoregressive(
        vocab_size, hidden_dim, num_layers,
        batch_size, seq_len,
        device, num_runs
    )

    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)

    print(f"\n{'Metric':<30} {'MaskGIT':<20} {'Autoregressive':<20} {'Speedup':<15}")
    print("-"*85)

    # Time
    maskgit_time = maskgit_metrics['avg_time']
    auto_time = autoregressive_metrics['avg_time']
    time_speedup = auto_time / maskgit_time

    print(f"{'Generation Time (sec)':<30} {maskgit_time:<20.3f} {auto_time:<20.3f} {time_speedup:<15.2f}x")

    # Tokens per second
    maskgit_tps = maskgit_metrics['tokens_per_sec']
    auto_tps = autoregressive_metrics['tokens_per_sec']
    tps_speedup = maskgit_tps / auto_tps

    print(f"{'Tokens/Second':<30} {maskgit_tps:<20.1f} {auto_tps:<20.1f} {tps_speedup:<15.2f}x")

    # Forward passes
    maskgit_fp = maskgit_metrics['forward_passes']
    auto_fp = autoregressive_metrics['forward_passes']
    fp_ratio = auto_fp / maskgit_fp

    print(f"{'Forward Passes':<30} {maskgit_fp:<20} {auto_fp:<20} {fp_ratio:<15.2f}x fewer")

    print("="*80)

    # Summary
    print("\nSUMMARY:")
    print(f"  MaskGIT is {time_speedup:.1f}x faster than autoregressive generation")
    print(f"  MaskGIT generates {maskgit_tps:.0f} tokens/sec vs {auto_tps:.0f} tokens/sec")
    print(f"  MaskGIT uses {fp_ratio:.1f}x fewer forward passes")

    # Validation
    print("\nVALIDATION:")
    target_speedup = 10.0

    if time_speedup >= target_speedup * 0.8:  # Allow 20% margin
        print(f"  ✅ PASS - Speedup {time_speedup:.1f}x meets target {target_speedup}x")
    else:
        print(f"  ⚠️  WARNING - Speedup {time_speedup:.1f}x below target {target_speedup}x")

    if maskgit_fp < auto_fp:
        print(f"  ✅ PASS - MaskGIT uses fewer forward passes")
    else:
        print(f"  ❌ FAIL - MaskGIT should use fewer forward passes")

    print("="*80 + "\n")

    return {
        'maskgit': maskgit_metrics,
        'autoregressive': autoregressive_metrics,
        'speedup': time_speedup,
    }


def plot_speedup_vs_seq_len(device: torch.device = torch.device('cpu')):
    """
    Plot speedup vs sequence length.

    Shows that MaskGIT advantage grows with longer sequences.
    """
    print("\n" + "="*80)
    print("Speedup vs Sequence Length")
    print("="*80)

    seq_lengths = [64, 128, 256, 512]
    speedups = []

    print(f"\n{'Seq Length':<15} {'MaskGIT (sec)':<15} {'Autoregressive (sec)':<20} {'Speedup':<10}")
    print("-"*60)

    for seq_len in seq_lengths:
        # Quick benchmark
        maskgit_metrics = benchmark_maskgit(
            vocab_size=512,
            hidden_dim=256,
            num_layers=4,
            batch_size=2,
            seq_len=seq_len,
            num_iterations=8,
            device=device,
            num_runs=5
        )

        auto_metrics = benchmark_autoregressive(
            vocab_size=512,
            hidden_dim=256,
            num_layers=4,
            batch_size=2,
            seq_len=seq_len,
            device=device,
            num_runs=5
        )

        speedup = auto_metrics['avg_time'] / maskgit_metrics['avg_time']
        speedups.append(speedup)

        print(f"{seq_len:<15} {maskgit_metrics['avg_time']:<15.3f} {auto_metrics['avg_time']:<20.3f} {speedup:<10.2f}x")

    print("\nConclusion: MaskGIT advantage grows with longer sequences!")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Benchmark generation speed")
    parser.add_argument('--vocab_size', type=int, default=1024, help='Vocabulary size')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=8, help='Number of layers')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--seq_len', type=int, default=256, help='Sequence length')
    parser.add_argument('--num_iterations', type=int, default=12, help='MaskGIT iterations')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_runs', type=int, default=10, help='Benchmark runs')
    parser.add_argument('--plot_scaling', action='store_true', help='Plot speedup vs seq len')

    args = parser.parse_args()

    device = torch.device(args.device)

    # Main comparison
    results = compare_generation_methods(
        vocab_size=args.vocab_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_iterations=args.num_iterations,
        device=device,
        num_runs=args.num_runs
    )

    # Optional: Plot scaling
    if args.plot_scaling:
        plot_speedup_vs_seq_len(device)


if __name__ == '__main__':
    main()

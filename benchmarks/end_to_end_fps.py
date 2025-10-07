#!/usr/bin/env python3
"""
Benchmark: End-to-End Video Generation FPS

Measures complete pipeline performance:
1. Tokenization (video → tokens)
2. Generation (MaskGIT)
3. Detokenization (tokens → video)

Tests with various optimizations:
- Baseline (no optimization)
- FP16 mixed precision
- torch.compile()
- FP16 + compile (full optimization)

Target: 40-50 FPS @ 640×360 on RTX 3060

Usage:
    python benchmarks/end_to_end_fps.py
    python benchmarks/end_to_end_fps.py --resolution 640 360 --batch_size 4
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.tokenizers import CosmosInspiredTokenizer
from models.generation import create_maskgit_generator
from utils.optimization import InferenceOptimizer, optimize_for_inference


def create_pipeline(
    resolution: Tuple[int, int],
    num_frames: int,
    codebook_size: int = 2**12,
    device: torch.device = torch.device('cpu')
):
    """
    Create complete generation pipeline.

    Args:
        resolution: Video resolution (H, W)
        num_frames: Number of frames
        codebook_size: Tokenizer codebook size
        device: Device

    Returns:
        tokenizer, generator
    """
    # Create tokenizer
    tokenizer = CosmosInspiredTokenizer(
        in_channels=3,
        encoder_dims=[64, 128, 256],
        decoder_dims=[128, 64, 32],
        latent_dim=256,
        codebook_size=codebook_size,
        resolution=resolution
    ).to(device)

    # Create generator
    generator = create_maskgit_generator(
        vocab_size=codebook_size,
        hidden_dim=256,
        num_layers=4,
        num_iterations=8,
        schedule_type='cosine'
    )
    generator.predictor.to(device)

    return tokenizer, generator


def measure_pipeline_fps(
    tokenizer: nn.Module,
    generator,
    batch_size: int,
    num_frames: int,
    resolution: Tuple[int, int],
    device: torch.device,
    optimizer: InferenceOptimizer = None,
    num_runs: int = 20,
    warmup_runs: int = 5
) -> Dict[str, float]:
    """
    Measure end-to-end pipeline FPS.

    Args:
        tokenizer: Video tokenizer
        generator: Token generator
        batch_size: Batch size
        num_frames: Frames per video
        resolution: Resolution (H, W)
        device: Device
        optimizer: Inference optimizer (for autocast)
        num_runs: Benchmark runs
        warmup_runs: Warmup runs

    Returns:
        metrics: FPS and timing metrics
    """
    tokenizer.eval()
    generator.predictor.eval()

    # Create sample input
    input_video = torch.randn(batch_size, num_frames, 3, *resolution).to(device)
    input_video = (input_video - input_video.min()) / (input_video.max() - input_video.min())

    # Get autocast context
    if optimizer:
        autocast_ctx = optimizer.autocast()
    else:
        autocast_ctx = torch.autocast(device_type='cpu', enabled=False)

    # Warmup
    print(f"  Warming up ({warmup_runs} runs)...")
    with torch.no_grad(), autocast_ctx:
        for _ in range(warmup_runs):
            # 1. Tokenize
            tokens = tokenizer.tokenize(input_video)

            # 2. Flatten for generation
            tokens_flat = tokens.reshape(batch_size, -1)
            seq_len = tokens_flat.shape[1]

            # 3. Generate
            generated_tokens = generator.generate(
                batch_size=batch_size,
                seq_len=seq_len,
                device=device
            )

            # 4. Reshape and detokenize
            spatial_tokens = int((seq_len / num_frames) ** 0.5)
            generated_tokens_reshaped = generated_tokens.reshape(
                batch_size, num_frames, spatial_tokens, spatial_tokens, 1
            )
            generated_video = tokenizer.detokenize(generated_tokens_reshaped)

    # Benchmark
    print(f"  Benchmarking ({num_runs} runs)...")
    if device.type == 'cuda':
        torch.cuda.synchronize()

    times = []

    with torch.no_grad(), autocast_ctx:
        for _ in range(num_runs):
            start = time.time()

            # Full pipeline
            tokens = tokenizer.tokenize(input_video)
            tokens_flat = tokens.reshape(batch_size, -1)
            seq_len = tokens_flat.shape[1]

            generated_tokens = generator.generate(
                batch_size=batch_size,
                seq_len=seq_len,
                device=device
            )

            spatial_tokens = int((seq_len / num_frames) ** 0.5)
            generated_tokens_reshaped = generated_tokens.reshape(
                batch_size, num_frames, spatial_tokens, spatial_tokens, 1
            )
            generated_video = tokenizer.detokenize(generated_tokens_reshaped)

            if device.type == 'cuda':
                torch.cuda.synchronize()

            elapsed = time.time() - start
            times.append(elapsed)

    # Calculate metrics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    # FPS = frames / time
    fps = (batch_size * num_frames) / avg_time
    best_fps = (batch_size * num_frames) / min_time

    # Throughput = videos / time
    videos_per_sec = batch_size / avg_time

    return {
        'avg_time': avg_time,
        'min_time': min_time,
        'max_time': max_time,
        'fps': fps,
        'best_fps': best_fps,
        'videos_per_sec': videos_per_sec,
        'total_frames': batch_size * num_frames,
    }


def benchmark_all_configurations(
    batch_size: int = 4,
    num_frames: int = 8,
    resolution: Tuple[int, int] = (256, 256),
    device: torch.device = torch.device('cpu'),
    num_runs: int = 20
):
    """
    Benchmark all optimization configurations.

    Args:
        batch_size: Batch size
        num_frames: Frames per video
        resolution: Resolution
        device: Device
        num_runs: Benchmark runs
    """
    print("="*80)
    print("End-to-End Video Generation FPS Benchmark")
    print("="*80)
    print(f"Batch size: {batch_size}")
    print(f"Frames per video: {num_frames}")
    print(f"Resolution: {resolution}")
    print(f"Device: {device}")
    print(f"Benchmark runs: {num_runs}")
    print("="*80 + "\n")

    results = {}

    # Configuration 1: Baseline (no optimization)
    print("1. Baseline (no optimization)...")
    tokenizer, generator = create_pipeline(resolution, num_frames, device=device)
    tokenizer = optimize_for_inference(tokenizer, device)
    generator.predictor = optimize_for_inference(generator.predictor, device)

    baseline_metrics = measure_pipeline_fps(
        tokenizer, generator,
        batch_size, num_frames, resolution,
        device, optimizer=None, num_runs=num_runs
    )
    results['baseline'] = baseline_metrics
    print(f"   FPS: {baseline_metrics['fps']:.2f}")
    print(f"   Time: {baseline_metrics['avg_time']:.3f} sec\n")

    # Configuration 2: FP16 only
    if device.type == 'cuda':
        print("2. FP16 Mixed Precision...")
        tokenizer, generator = create_pipeline(resolution, num_frames, device=device)
        tokenizer = optimize_for_inference(tokenizer, device)
        generator.predictor = optimize_for_inference(generator.predictor, device)

        optimizer_fp16 = InferenceOptimizer(
            use_fp16=True,
            use_compile=False,
            use_flash_attn=False
        )

        fp16_metrics = measure_pipeline_fps(
            tokenizer, generator,
            batch_size, num_frames, resolution,
            device, optimizer=optimizer_fp16, num_runs=num_runs
        )
        results['fp16'] = fp16_metrics
        print(f"   FPS: {fp16_metrics['fps']:.2f} ({fp16_metrics['fps']/baseline_metrics['fps']:.2f}x)")
        print(f"   Time: {fp16_metrics['avg_time']:.3f} sec\n")

        # Configuration 3: torch.compile only
        print("3. torch.compile()...")
        tokenizer, generator = create_pipeline(resolution, num_frames, device=device)
        tokenizer = optimize_for_inference(tokenizer, device)
        generator.predictor = optimize_for_inference(generator.predictor, device)

        optimizer_compile = InferenceOptimizer(
            use_fp16=False,
            use_compile=True,
            use_flash_attn=False
        )

        # Compile models
        tokenizer = optimizer_compile.optimize(tokenizer, fullgraph=False)
        generator.predictor = optimizer_compile.optimize(generator.predictor, fullgraph=False)

        compile_metrics = measure_pipeline_fps(
            tokenizer, generator,
            batch_size, num_frames, resolution,
            device, optimizer=None, num_runs=num_runs
        )
        results['compile'] = compile_metrics
        print(f"   FPS: {compile_metrics['fps']:.2f} ({compile_metrics['fps']/baseline_metrics['fps']:.2f}x)")
        print(f"   Time: {compile_metrics['avg_time']:.3f} sec\n")

        # Configuration 4: FP16 + compile (full optimization)
        print("4. FP16 + torch.compile() (Full Optimization)...")
        tokenizer, generator = create_pipeline(resolution, num_frames, device=device)
        tokenizer = optimize_for_inference(tokenizer, device)
        generator.predictor = optimize_for_inference(generator.predictor, device)

        optimizer_full = InferenceOptimizer(
            use_fp16=True,
            use_compile=True,
            use_flash_attn=False
        )

        # Optimize models
        tokenizer = optimizer_full.optimize(tokenizer, fullgraph=False)
        generator.predictor = optimizer_full.optimize(generator.predictor, fullgraph=False)

        full_metrics = measure_pipeline_fps(
            tokenizer, generator,
            batch_size, num_frames, resolution,
            device, optimizer=optimizer_full, num_runs=num_runs
        )
        results['full'] = full_metrics
        print(f"   FPS: {full_metrics['fps']:.2f} ({full_metrics['fps']/baseline_metrics['fps']:.2f}x)")
        print(f"   Time: {full_metrics['avg_time']:.3f} sec\n")

    # Print comparison
    print("="*80)
    print("RESULTS")
    print("="*80)

    print(f"\n{'Configuration':<25} {'FPS':<15} {'Time (sec)':<15} {'Speedup':<10}")
    print("-"*65)

    baseline_fps = results['baseline']['fps']

    for name, metrics in results.items():
        fps = metrics['fps']
        time_sec = metrics['avg_time']
        speedup = fps / baseline_fps

        display_name = {
            'baseline': 'Baseline',
            'fp16': 'FP16',
            'compile': 'torch.compile',
            'full': 'FP16 + compile'
        }.get(name, name)

        print(f"{display_name:<25} {fps:<15.2f} {time_sec:<15.3f} {speedup:<10.2f}x")

    print("="*80)

    # Target validation
    print("\nTARGET VALIDATION:")
    target_fps = 40.0  # 40 FPS target

    if 'full' in results:
        best_fps = results['full']['fps']
    elif 'fp16' in results:
        best_fps = results['fp16']['fps']
    else:
        best_fps = baseline_fps

    print(f"  Target: {target_fps:.0f} FPS @ {resolution[0]}×{resolution[1]}")
    print(f"  Achieved: {best_fps:.2f} FPS")

    if best_fps >= target_fps:
        print(f"  ✅ PASS - Meets or exceeds target!")
    else:
        ratio = best_fps / target_fps
        print(f"  ⚠️  {ratio:.1%} of target (may need GPU/higher optimization)")

    print("="*80 + "\n")

    return results


def test_different_resolutions(device: torch.device = torch.device('cpu')):
    """
    Test FPS at different resolutions.

    Args:
        device: Device
    """
    print("\n" + "="*80)
    print("FPS vs Resolution")
    print("="*80)

    resolutions = [
        (128, 128),
        (256, 256),
        (360, 640),  # 640×360 (target)
        (512, 512),
    ]

    print(f"\n{'Resolution':<15} {'Baseline FPS':<15} {'Optimized FPS':<15} {'Speedup':<10}")
    print("-"*55)

    for res in resolutions:
        # Baseline
        tokenizer, generator = create_pipeline(res, num_frames=8, device=device)
        tokenizer = optimize_for_inference(tokenizer, device)
        generator.predictor = optimize_for_inference(generator.predictor, device)

        baseline = measure_pipeline_fps(
            tokenizer, generator,
            batch_size=1, num_frames=8, resolution=res,
            device=device, optimizer=None, num_runs=10, warmup_runs=2
        )

        # Optimized (FP16 if CUDA)
        if device.type == 'cuda':
            tokenizer, generator = create_pipeline(res, num_frames=8, device=device)
            tokenizer = optimize_for_inference(tokenizer, device)
            generator.predictor = optimize_for_inference(generator.predictor, device)

            optimizer = InferenceOptimizer(use_fp16=True, use_compile=False)

            optimized = measure_pipeline_fps(
                tokenizer, generator,
                batch_size=1, num_frames=8, resolution=res,
                device=device, optimizer=optimizer, num_runs=10, warmup_runs=2
            )
            speedup = optimized['fps'] / baseline['fps']
        else:
            optimized = baseline
            speedup = 1.0

        res_str = f"{res[1]}×{res[0]}"
        print(f"{res_str:<15} {baseline['fps']:<15.2f} {optimized['fps']:<15.2f} {speedup:<10.2f}x")

    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="End-to-end FPS benchmark")
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_frames', type=int, default=8, help='Frames per video')
    parser.add_argument('--resolution', type=int, nargs=2, default=[256, 256], help='Resolution (H W)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_runs', type=int, default=20, help='Benchmark runs')
    parser.add_argument('--test_resolutions', action='store_true', help='Test different resolutions')

    args = parser.parse_args()

    device = torch.device(args.device)
    resolution = tuple(args.resolution)

    # Main benchmark
    results = benchmark_all_configurations(
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        resolution=resolution,
        device=device,
        num_runs=args.num_runs
    )

    # Optional: Test resolutions
    if args.test_resolutions:
        test_different_resolutions(device)


if __name__ == '__main__':
    main()

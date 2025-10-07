#!/usr/bin/env python3
"""
Benchmark: Memory Usage with Geometric Compression

Compares memory usage for long videos with and without geometric compression.

Key metrics:
- Peak memory usage
- Memory growth rate
- Compression savings
- Maximum video length achievable

Usage:
    python benchmarks/measure_memory_usage.py
    python benchmarks/measure_memory_usage.py --max_frames 480 --device cuda
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.context import ConstantContextManager, estimate_compression_savings


class NaiveContextManager:
    """
    Naive context manager that stores all frames (no compression).
    Used as baseline for comparison.
    """

    def __init__(self):
        self.frames = None

    def update(self, new_frames: torch.Tensor):
        """Add new frames to context"""
        if self.frames is None:
            self.frames = new_frames
        else:
            self.frames = torch.cat([self.frames, new_frames], dim=1)

    def get_context(self):
        """Get all frames"""
        return self.frames

    def get_stats(self):
        """Get statistics"""
        if self.frames is None:
            return {'total_elements': 0, 'total_frames_represented': 0}

        return {
            'total_elements': self.frames.numel(),
            'total_frames_represented': self.frames.shape[1],
            'memory_savings': 0.0  # No compression
        }

    def reset(self):
        """Reset"""
        self.frames = None


def measure_memory_growth(
    manager,
    num_iterations: int,
    frames_per_iteration: int,
    feature_shape: Tuple[int, ...],
    device: torch.device
) -> Dict[str, List[float]]:
    """
    Measure memory growth over time.

    Args:
        manager: Context manager to test
        num_iterations: Number of update iterations
        frames_per_iteration: Frames added per iteration
        feature_shape: Shape of features (C, H, W)
        device: Device to run on

    Returns:
        metrics: Dictionary with memory metrics over time
    """
    metrics = {
        'total_frames': [],
        'total_elements': [],
        'memory_mb': [],
        'peak_memory_mb': [],
        'savings': [],
    }

    manager.reset()

    for i in range(num_iterations):
        # Generate new features
        batch_size = 1
        features = torch.randn(
            batch_size,
            frames_per_iteration,
            *feature_shape,
            device=device
        )

        # Update context
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()

        manager.update(features)

        # Get statistics
        stats = manager.get_stats()
        metrics['total_frames'].append(stats['total_frames_represented'])
        metrics['total_elements'].append(stats['total_elements'])
        metrics['savings'].append(stats.get('memory_savings', 0.0))

        # Measure memory
        if device.type == 'cuda':
            torch.cuda.synchronize()
            current_memory = torch.cuda.memory_allocated() / 1024**2  # MB
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
            metrics['memory_mb'].append(current_memory)
            metrics['peak_memory_mb'].append(peak_memory)
        else:
            # Estimate from elements (4 bytes per float)
            memory_mb = stats['total_elements'] * 4 / 1024**2
            metrics['memory_mb'].append(memory_mb)
            metrics['peak_memory_mb'].append(memory_mb)

    return metrics


def compare_managers(
    max_frames: int = 480,
    frames_per_iteration: int = 10,
    feature_shape: Tuple[int, ...] = (128, 16, 16),
    device: torch.device = torch.device('cpu')
):
    """
    Compare naive vs geometric compression managers.

    Args:
        max_frames: Total frames to process
        frames_per_iteration: Frames per update
        feature_shape: Feature dimensions
        device: Device to use
    """
    print("="*80)
    print("Memory Usage Benchmark: Geometric Compression vs Naive")
    print("="*80)
    print(f"Total frames: {max_frames}")
    print(f"Frames per iteration: {frames_per_iteration}")
    print(f"Feature shape: {feature_shape}")
    print(f"Device: {device}")
    print("="*80 + "\n")

    num_iterations = max_frames // frames_per_iteration

    # Test naive manager
    print("Testing Naive Context Manager (no compression)...")
    naive_manager = NaiveContextManager()
    naive_metrics = measure_memory_growth(
        naive_manager,
        num_iterations,
        frames_per_iteration,
        feature_shape,
        device
    )

    # Test geometric compression manager
    print("Testing Geometric Compression Manager...")
    geometric_manager = ConstantContextManager(
        max_levels=6,
        level_capacity=16,
        compression_method='pool'
    )
    geometric_metrics = measure_memory_growth(
        geometric_manager,
        num_iterations,
        frames_per_iteration,
        feature_shape,
        device
    )

    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)

    # Final statistics
    naive_final_memory = naive_metrics['memory_mb'][-1]
    geometric_final_memory = geometric_metrics['memory_mb'][-1]
    memory_reduction = 1.0 - (geometric_final_memory / naive_final_memory)

    print(f"\nFinal Memory Usage:")
    print(f"  Naive:      {naive_final_memory:>10.2f} MB")
    print(f"  Geometric:  {geometric_final_memory:>10.2f} MB")
    print(f"  Reduction:  {memory_reduction:>10.1%}")

    # Memory savings
    avg_savings = np.mean(geometric_metrics['savings'])
    print(f"\nAverage Memory Savings: {avg_savings:.1%}")

    # Growth rates
    naive_growth = naive_metrics['memory_mb'][-1] - naive_metrics['memory_mb'][0]
    geometric_growth = geometric_metrics['memory_mb'][-1] - geometric_metrics['memory_mb'][0]

    print(f"\nMemory Growth:")
    print(f"  Naive:      {naive_growth:>10.2f} MB")
    print(f"  Geometric:  {geometric_growth:>10.2f} MB")
    print(f"  Reduction:  {1.0 - (geometric_growth / naive_growth):>10.1%}")

    # Estimate maximum video length
    memory_budget_mb = 4000  # 4GB VRAM budget
    if geometric_final_memory > 0:
        max_frames_geometric = int(max_frames * (memory_budget_mb / geometric_final_memory))
        max_frames_naive = int(max_frames * (memory_budget_mb / naive_final_memory))

        print(f"\nEstimated Max Video Length (with 4GB VRAM):")
        print(f"  Naive:      {max_frames_naive:>10} frames ({max_frames_naive / 8:.1f} sec @ 8fps)")
        print(f"  Geometric:  {max_frames_geometric:>10} frames ({max_frames_geometric / 8:.1f} sec @ 8fps)")
        print(f"  Improvement: {max_frames_geometric / max_frames_naive:>9.1f}x longer videos")

    # Timeline comparison
    print(f"\nMemory Usage Timeline:")
    print(f"{'Frames':<10} {'Naive (MB)':<15} {'Geometric (MB)':<15} {'Savings':<10}")
    print("-" * 50)

    checkpoints = [0, num_iterations // 4, num_iterations // 2, 3 * num_iterations // 4, num_iterations - 1]
    for idx in checkpoints:
        frames = naive_metrics['total_frames'][idx]
        naive_mem = naive_metrics['memory_mb'][idx]
        geo_mem = geometric_metrics['memory_mb'][idx]
        savings = geometric_metrics['savings'][idx]

        print(f"{frames:<10} {naive_mem:<15.2f} {geo_mem:<15.2f} {savings:<10.1%}")

    print("="*80)

    # Validation
    print("\nVALIDATION:")
    target_savings = estimate_compression_savings(max_frames)
    print(f"  Expected savings: {target_savings:.1%}")
    print(f"  Actual savings:   {avg_savings:.1%}")

    if avg_savings >= target_savings * 0.9:
        print("  ✅ PASS - Savings meet expectations")
    else:
        print("  ⚠️  WARNING - Savings below expectations")

    if memory_reduction > 0.5:
        print("  ✅ PASS - >50% memory reduction achieved")
    else:
        print("  ⚠️  WARNING - Memory reduction below 50%")

    print("="*80 + "\n")

    return {
        'naive': naive_metrics,
        'geometric': geometric_metrics,
        'memory_reduction': memory_reduction,
        'avg_savings': avg_savings,
    }


def test_scalability(device: torch.device = torch.device('cpu')):
    """
    Test how geometric compression scales with video length.
    """
    print("\n" + "="*80)
    print("SCALABILITY TEST: Memory Usage vs Video Length")
    print("="*80)

    video_lengths = [60, 120, 240, 480]  # frames
    feature_shape = (128, 16, 16)

    print(f"\n{'Video Length':<15} {'Naive (MB)':<15} {'Geometric (MB)':<15} {'Savings':<10}")
    print("-" * 55)

    for num_frames in video_lengths:
        # Naive
        naive_manager = NaiveContextManager()
        for _ in range(num_frames // 10):
            features = torch.randn(1, 10, *feature_shape, device=device)
            naive_manager.update(features)

        naive_stats = naive_manager.get_stats()
        naive_memory = naive_stats['total_elements'] * 4 / 1024**2

        # Geometric
        geo_manager = ConstantContextManager(max_levels=6, level_capacity=16)
        for _ in range(num_frames // 10):
            features = torch.randn(1, 10, *feature_shape, device=device)
            geo_manager.update(features)

        geo_stats = geo_manager.get_stats()
        geo_memory = geo_stats['total_elements'] * 4 / 1024**2
        savings = geo_stats['memory_savings']

        print(f"{num_frames:<15} {naive_memory:<15.2f} {geo_memory:<15.2f} {savings:<10.1%}")

    print("="*80)


def plot_memory_curves(results: Dict, save_path: str = 'benchmarks/results/memory_comparison.png'):
    """
    Plot memory usage curves (requires matplotlib).

    Args:
        results: Results from compare_managers
        save_path: Where to save plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not available, skipping plot")
        return

    naive = results['naive']
    geometric = results['geometric']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Memory usage over time
    ax1.plot(naive['total_frames'], naive['memory_mb'], label='Naive (No Compression)', marker='o')
    ax1.plot(geometric['total_frames'], geometric['memory_mb'], label='Geometric Compression', marker='s')
    ax1.set_xlabel('Total Frames')
    ax1.set_ylabel('Memory Usage (MB)')
    ax1.set_title('Memory Usage vs Video Length')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Savings over time
    ax2.plot(geometric['total_frames'], [s * 100 for s in geometric['savings']], marker='o', color='green')
    ax2.set_xlabel('Total Frames')
    ax2.set_ylabel('Memory Savings (%)')
    ax2.set_title('Memory Savings with Geometric Compression')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Create directory if needed
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark memory usage with geometric compression")
    parser.add_argument('--max_frames', type=int, default=480, help='Maximum frames to test')
    parser.add_argument('--frames_per_iter', type=int, default=10, help='Frames per iteration')
    parser.add_argument('--feature_channels', type=int, default=128, help='Feature channels')
    parser.add_argument('--spatial_size', type=int, default=16, help='Spatial size (H, W)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--scalability', action='store_true', help='Run scalability test')

    args = parser.parse_args()

    device = torch.device(args.device)
    feature_shape = (args.feature_channels, args.spatial_size, args.spatial_size)

    # Main comparison
    results = compare_managers(
        max_frames=args.max_frames,
        frames_per_iteration=args.frames_per_iter,
        feature_shape=feature_shape,
        device=device
    )

    # Scalability test
    if args.scalability:
        test_scalability(device)

    # Plot
    if args.plot:
        plot_memory_curves(results)


if __name__ == '__main__':
    main()

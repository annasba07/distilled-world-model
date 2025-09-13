#!/usr/bin/env python3
"""
Main entry point for Lightweight World Model
Run different components easily from one script
"""

import argparse
import sys
import subprocess


def main():
    parser = argparse.ArgumentParser(
        description='Lightweight World Model - Main Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py test          # Test current capabilities
  python run.py train         # Start training VQ-VAE
  python run.py demo          # Launch interactive demo
  python run.py serve         # Start API server
  python run.py benchmark     # Run performance benchmarks
        """
    )

    parser.add_argument(
        'command',
        choices=['test', 'train', 'demo', 'serve', 'benchmark', 'install'],
        help='Command to run'
    )

    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')

    args = parser.parse_args()

    if args.command == 'test':
        print("Running Milestone Tests...")
        print("=" * 60)
        subprocess.run([sys.executable, 'test_milestone1.py'])

    elif args.command == 'train':
        print("Starting VQ-VAE Training (Lightning)...")
        print("=" * 60)
        cmd = [
            sys.executable, '-m', 'src.training.cli', 'vqvae',
            '--epochs', str(args.epochs),
            '--batch_size', str(args.batch_size),
            '--device', args.device,
            '--amp'
        ]
        subprocess.run(cmd)

    elif args.command == 'demo':
        print("Launching Interactive Demo...")
        print("=" * 60)
        cmd = [sys.executable, 'src/demo_reconstruction.py', '--mode', 'gradio']
        if args.checkpoint:
            cmd.extend(['--checkpoint', args.checkpoint])
        subprocess.run(cmd)

    elif args.command == 'serve':
        print("Starting API Server...")
        print("=" * 60)
        subprocess.run([sys.executable, '-m', 'src.api.server'])

    elif args.command == 'benchmark':
        print("Running Performance Benchmarks...")
        print("=" * 60)
        run_benchmarks()

    elif args.command == 'install':
        print("Installing Dependencies...")
        print("=" * 60)
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', '.'])
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', '.[training]'])
        print("\nInstallation complete!")
        print("\nNext steps:")
        print("  1. Test installation: python run.py test")
        print("  2. Start training: python run.py train")
        print("  3. Launch demo: python run.py demo")


def run_benchmarks():
    """Run performance benchmarks"""
    import torch
    import time
    import numpy as np

    # Add src to path
    sys.path.append('src')
    from models.improved_vqvae import ImprovedVQVAE

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   CUDA: {torch.version.cuda}")

    # Create model
    model = ImprovedVQVAE(
        in_channels=3,
        latent_dim=256,
        num_embeddings=512,
        hidden_dims=[64, 128, 256, 512],
        use_ema=True,
        use_attention=True
    ).to(device)
    model.eval()

    # Model info
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Statistics:")
    print(f"   Parameters: {num_params / 1e6:.2f}M")
    print(f"   Model size: {num_params * 4 / 1024**2:.2f} MB (FP32)")

    # Benchmark different batch sizes
    print(f"\nInference Benchmarks:")
    print(f"   {'Batch Size':<12} {'Time (ms)':<12} {'FPS':<12} {'VRAM (GB)':<12}")
    print(f"   {'-'*48}")

    for batch_size in [1, 4, 8, 16, 32]:
        try:
            # Reset memory stats
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats()

            # Create input
            x = torch.randn(batch_size, 3, 256, 256).to(device)

            # Warmup
            for _ in range(10):
                with torch.no_grad():
                    _ = model(x)

            # Measure
            if device.type == 'cuda':
                torch.cuda.synchronize()

            times = []
            for _ in range(50):
                start = time.time()
                with torch.no_grad():
                    _ = model(x)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                times.append((time.time() - start) * 1000)

            avg_time = np.mean(times)
            fps = (batch_size / avg_time) * 1000

            if device.type == 'cuda':
                vram = torch.cuda.max_memory_allocated() / 1024**3
            else:
                vram = 0

            print(f"   {batch_size:<12} {avg_time:<12.2f} {fps:<12.1f} {vram:<12.2f}")

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"   {batch_size:<12} {'OOM':<12} {'-':<12} {'-':<12}")
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            else:
                raise

    # Test compression ratio
    print(f"\nCompression Analysis:")
    x = torch.randn(1, 3, 256, 256).to(device)
    with torch.no_grad():
        quantized, _ = model.encode(x)

    original_size = x.numel() * 4 / 1024  # KB
    compressed_size = quantized.numel() * 4 / 1024  # KB
    ratio = original_size / compressed_size

    print(f"   Original size: {original_size:.1f} KB")
    print(f"   Compressed size: {compressed_size:.1f} KB")
    print(f"   Compression ratio: {ratio:.1f}x")

    print(f"\nBenchmark complete!")


if __name__ == "__main__":
    main()

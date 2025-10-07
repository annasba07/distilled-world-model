#!/usr/bin/env python3
"""
Training script for Cosmos-Inspired Video Tokenizer

Trains the tokenizer to:
1. Compress videos efficiently (target: 8x compression)
2. Maintain high reconstruction quality
3. Avoid codebook collapse
4. Enable fast encoding (target: 12x faster)

Usage:
    python train_cosmos_tokenizer.py --config configs/cosmos_tokenizer.yaml
    python train_cosmos_tokenizer.py --data_dir /path/to/videos --epochs 100
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional
import yaml
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models.tokenizers.cosmos_tokenizer import CosmosInspiredTokenizer


class VideoDataset(Dataset):
    """
    Simple video dataset for tokenizer training.

    In production, replace with your actual video dataset.
    For now, generates synthetic data for testing.
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        num_samples: int = 1000,
        num_frames: int = 8,
        resolution: tuple = (128, 128),
        synthetic: bool = True
    ):
        self.data_dir = data_dir
        self.num_samples = num_samples
        self.num_frames = num_frames
        self.resolution = resolution
        self.synthetic = synthetic

        if not synthetic and data_dir:
            # Load real video files
            self.video_files = list(Path(data_dir).glob("**/*.mp4"))
            self.num_samples = len(self.video_files)
            print(f"Found {self.num_samples} video files in {data_dir}")
        else:
            print(f"Using synthetic data: {num_samples} samples")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.synthetic:
            # Generate synthetic video
            # Shape: [T, C, H, W]
            video = torch.randn(
                self.num_frames,
                3,
                self.resolution[0],
                self.resolution[1]
            )
            # Normalize to [0, 1]
            video = (video - video.min()) / (video.max() - video.min())
        else:
            # Load real video
            # TODO: Implement real video loading with decord or torchvision
            video_path = self.video_files[idx]
            # Placeholder for real video loading
            video = torch.randn(self.num_frames, 3, *self.resolution)

        return video


def build_tokenizer(config: Dict) -> CosmosInspiredTokenizer:
    """Build tokenizer from config"""
    return CosmosInspiredTokenizer(
        in_channels=config.get('in_channels', 3),
        encoder_dims=config.get('encoder_dims', [64, 128, 256, 512]),
        decoder_dims=config.get('decoder_dims', [256, 128, 64, 32]),
        latent_dim=config.get('latent_dim', 512),
        codebook_size=config.get('codebook_size', 2**16),
        resolution=tuple(config.get('resolution', [640, 360])),
        use_perceptual_loss=config.get('use_perceptual_loss', False)
    )


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    global_step: int
) -> Dict[str, float]:
    """Train for one epoch"""
    model.train()

    total_loss = 0.0
    total_recon_loss = 0.0
    total_quant_loss = 0.0
    total_perplexity = 0.0
    total_usage = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, videos in enumerate(pbar):
        videos = videos.to(device)  # [B, T, C, H, W]

        # Forward pass
        output = model(videos, return_loss=True)

        loss = output['loss']
        recon_loss = output['recon_loss']
        quant_loss = output['quant_loss']
        perplexity = output['perplexity']
        usage = output['codebook_usage']

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Accumulate metrics
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_quant_loss += quant_loss.item()
        total_perplexity += perplexity.item()
        total_usage += usage.item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'recon': f"{recon_loss.item():.4f}",
            'ppl': f"{perplexity.item():.1f}",
            'usage': f"{usage.item():.2%}"
        })

        # Log to tensorboard
        if batch_idx % 10 == 0:
            writer.add_scalar('train/loss', loss.item(), global_step)
            writer.add_scalar('train/recon_loss', recon_loss.item(), global_step)
            writer.add_scalar('train/quant_loss', quant_loss.item(), global_step)
            writer.add_scalar('train/perplexity', perplexity.item(), global_step)
            writer.add_scalar('train/codebook_usage', usage.item(), global_step)

        global_step += 1

    # Compute epoch averages
    num_batches = len(dataloader)
    metrics = {
        'loss': total_loss / num_batches,
        'recon_loss': total_recon_loss / num_batches,
        'quant_loss': total_quant_loss / num_batches,
        'perplexity': total_perplexity / num_batches,
        'codebook_usage': total_usage / num_batches,
    }

    return metrics, global_step


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter
) -> Dict[str, float]:
    """Validate the model"""
    model.eval()

    total_loss = 0.0
    total_recon_loss = 0.0
    total_quant_loss = 0.0
    total_perplexity = 0.0
    total_usage = 0.0

    # Track compression
    total_compression = 0.0

    for videos in tqdm(dataloader, desc="Validating"):
        videos = videos.to(device)

        # Forward pass
        output = model(videos, return_loss=True)

        total_loss += output['loss'].item()
        total_recon_loss += output['recon_loss'].item()
        total_quant_loss += output['quant_loss'].item()
        total_perplexity += output['perplexity'].item()
        total_usage += output['codebook_usage'].item()

        # Compute compression
        compression = model.get_compression_ratio(videos)
        total_compression += compression

    num_batches = len(dataloader)
    metrics = {
        'loss': total_loss / num_batches,
        'recon_loss': total_recon_loss / num_batches,
        'quant_loss': total_quant_loss / num_batches,
        'perplexity': total_perplexity / num_batches,
        'codebook_usage': total_usage / num_batches,
        'compression_ratio': total_compression / num_batches,
    }

    # Log to tensorboard
    for key, value in metrics.items():
        writer.add_scalar(f'val/{key}', value, epoch)

    return metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    checkpoint_dir: Path,
    is_best: bool = False
):
    """Save model checkpoint"""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }

    # Save latest
    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")

    # Save best
    if is_best:
        best_path = checkpoint_dir / "best_model.pt"
        torch.save(checkpoint, best_path)
        print(f"Saved best model to {best_path}")


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None
) -> int:
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint['epoch']
    print(f"Loaded checkpoint from epoch {epoch}")

    return epoch


def main():
    parser = argparse.ArgumentParser(description="Train Cosmos Video Tokenizer")

    # Data
    parser.add_argument('--data_dir', type=str, default=None, help='Path to video dataset')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic data')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of synthetic samples')

    # Model
    parser.add_argument('--config', type=str, default=None, help='Config file path')
    parser.add_argument('--latent_dim', type=int, default=512, help='Latent dimension')
    parser.add_argument('--codebook_size', type=int, default=65536, help='Codebook size (2^16)')

    # Training
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_frames', type=int, default=8, help='Number of frames per video')
    parser.add_argument('--resolution', type=int, nargs=2, default=[128, 128], help='Video resolution (H W)')

    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/cosmos_tokenizer', help='Checkpoint directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--save_freq', type=int, default=10, help='Save checkpoint every N epochs')

    # Logging
    parser.add_argument('--log_dir', type=str, default='logs/cosmos_tokenizer', help='TensorBoard log directory')

    # System
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')

    args = parser.parse_args()

    # Load config if provided
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Use command line args
        config = {
            'in_channels': 3,
            'encoder_dims': [64, 128, 256, 512],
            'decoder_dims': [256, 128, 64, 32],
            'latent_dim': args.latent_dim,
            'codebook_size': args.codebook_size,
            'resolution': args.resolution,
            'use_perceptual_loss': False,
        }

    print("="*80)
    print("Training Cosmos-Inspired Video Tokenizer")
    print("="*80)
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Codebook size: {config['codebook_size']:,}")
    print(f"Resolution: {config['resolution']}")
    print("="*80)

    # Build model
    device = torch.device(args.device)
    model = build_tokenizer(config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Build datasets
    train_dataset = VideoDataset(
        data_dir=args.data_dir,
        num_samples=args.num_samples,
        num_frames=args.num_frames,
        resolution=tuple(args.resolution),
        synthetic=args.synthetic or args.data_dir is None
    )

    val_dataset = VideoDataset(
        data_dir=args.data_dir,
        num_samples=args.num_samples // 5,  # 20% for validation
        num_frames=args.num_frames,
        resolution=tuple(args.resolution),
        synthetic=args.synthetic or args.data_dir is None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if args.device == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if args.device == 'cuda' else False
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=0.01
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(Path(args.resume), model, optimizer)

    # TensorBoard writer
    writer = SummaryWriter(args.log_dir)

    # Training loop
    best_val_loss = float('inf')
    global_step = 0

    print("\nStarting training...\n")

    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*80}")

        # Train
        train_metrics, global_step = train_epoch(
            model, train_loader, optimizer, device, epoch, writer, global_step
        )

        # Validate
        val_metrics = validate(model, val_loader, device, epoch, writer)

        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        writer.add_scalar('train/lr', current_lr, epoch)

        # Print metrics
        print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, "
              f"Recon: {train_metrics['recon_loss']:.4f}, "
              f"Perplexity: {train_metrics['perplexity']:.1f}, "
              f"Usage: {train_metrics['codebook_usage']:.2%}")

        print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
              f"Recon: {val_metrics['recon_loss']:.4f}, "
              f"Perplexity: {val_metrics['perplexity']:.1f}, "
              f"Compression: {val_metrics['compression_ratio']:.1f}x")

        # Save checkpoint
        is_best = val_metrics['loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['loss']

        if (epoch + 1) % args.save_freq == 0 or is_best:
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                Path(args.checkpoint_dir), is_best
            )

    print("\n" + "="*80)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("="*80)

    writer.close()


if __name__ == '__main__':
    main()

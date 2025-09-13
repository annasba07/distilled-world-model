#!/usr/bin/env python3
"""
Training Script for Milestone 3: Short Sequence Generation
Trains the sequence generator to produce coherent 30-frame sequences
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import os
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from tqdm import tqdm

# TensorBoard for logging
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    print("⚠️  TensorBoard not available. Install with: pip install tensorboard")
    TENSORBOARD_AVAILABLE = False

# Import our models and data
from models.sequence_generator import SequenceGenerator, calculate_sequence_metrics
from data.sequence_dataset import create_sequence_dataloader


class SequenceTrainer:
    """Trainer for sequence generation model (Milestone 3)"""
    
    def __init__(
        self,
        model: SequenceGenerator,
        device: torch.device,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        checkpoint_dir: str = "checkpoints/sequence",
        log_dir: str = "logs/sequence"
    ):
        self.model = model.to(device)
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=100,  # Will be updated based on num_epochs
            eta_min=learning_rate * 0.1
        )
        
        # TensorBoard
        if TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Loss weights
        self.loss_weights = {
            'mse': 1.0,
            'temporal_consistency': 0.5,
            'smoothness': 0.2,
            'perceptual': 0.3
        }
        
        print(f"🏋️ SequenceTrainer initialized")
        print(f"   Device: {device}")
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Checkpoint dir: {checkpoint_dir}")
        print(f"   Log dir: {log_dir}")
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        epoch_losses = {
            'total': 0.0,
            'mse': 0.0,
            'temporal_consistency': 0.0,
            'smoothness': 0.0,
            'perceptual': 0.0
        }
        
        num_batches = len(train_loader)
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, (sequences, metadata) in enumerate(progress_bar):
            sequences = sequences.to(self.device)  # (batch, seq_len, channels, height, width)
            batch_size, seq_len = sequences.size(0), sequences.size(1)
            
            # Forward pass
            self.optimizer.zero_grad()
            predicted_latents, target_latents, additional_losses = self.model(sequences)
            
            # Calculate losses
            losses = {}
            
            # Main MSE loss
            losses['mse'] = F.mse_loss(predicted_latents, target_latents)
            
            # Additional losses from model
            losses.update(additional_losses)
            
            # Perceptual loss (simplified cosine similarity)
            if predicted_latents.size(1) > 1:
                pred_flat = predicted_latents.flatten(2)
                target_flat = target_latents.flatten(2)
                cosine_sim = F.cosine_similarity(pred_flat, target_flat, dim=-1).mean()
                losses['perceptual'] = 1.0 - cosine_sim
            else:
                losses['perceptual'] = torch.tensor(0.0, device=self.device)
            
            # Combined loss
            total_loss = sum(
                self.loss_weights.get(name, 1.0) * loss 
                for name, loss in losses.items()
                if loss is not None
            )
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update metrics
            with torch.no_grad():
                epoch_losses['total'] += total_loss.item()
                for name, loss in losses.items():
                    if loss is not None and name in epoch_losses:
                        epoch_losses[name] += loss.item()
            
            # Log to TensorBoard
            if self.writer and self.global_step % 50 == 0:
                for name, loss in losses.items():
                    if loss is not None:
                        self.writer.add_scalar(f'train/{name}', loss.item(), self.global_step)
                
                self.writer.add_scalar('train/total_loss', total_loss.item(), self.global_step)
                self.writer.add_scalar('train/learning_rate', self.scheduler.get_last_lr()[0], self.global_step)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'mse': f"{losses['mse'].item():.6f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
            
            self.global_step += 1
        
        # Average epoch losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        
        val_losses = {
            'total': 0.0,
            'mse': 0.0,
            'temporal_consistency': 0.0,
            'smoothness': 0.0,
            'perceptual': 0.0,
            'psnr': 0.0,
            'sequence_quality': 0.0
        }
        
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for sequences, metadata in tqdm(val_loader, desc="Validation"):
                sequences = sequences.to(self.device)
                
                # Forward pass
                predicted_latents, target_latents, additional_losses = self.model(sequences)
                
                # Calculate losses
                losses = {}
                losses['mse'] = F.mse_loss(predicted_latents, target_latents)
                losses.update(additional_losses)
                
                # Perceptual loss
                if predicted_latents.size(1) > 1:
                    pred_flat = predicted_latents.flatten(2)
                    target_flat = target_latents.flatten(2)
                    cosine_sim = F.cosine_similarity(pred_flat, target_flat, dim=-1).mean()
                    losses['perceptual'] = 1.0 - cosine_sim
                else:
                    losses['perceptual'] = torch.tensor(0.0, device=self.device)
                
                # Combined loss
                total_loss = sum(
                    self.loss_weights.get(name, 1.0) * loss 
                    for name, loss in losses.items()
                    if loss is not None
                )
                
                # Additional metrics
                psnr = -10 * torch.log10(losses['mse'])
                sequence_metrics = calculate_sequence_metrics(predicted_latents, target_latents)
                
                # Update validation metrics
                val_losses['total'] += total_loss.item()
                val_losses['psnr'] += psnr.item()
                val_losses['sequence_quality'] += sequence_metrics['cosine_similarity']
                
                for name, loss in losses.items():
                    if loss is not None and name in val_losses:
                        val_losses[name] += loss.item()
        
        # Average validation losses
        for key in val_losses:
            val_losses[key] /= num_batches
        
        return val_losses
    
    def save_checkpoint(self, val_losses: Dict[str, float], is_best: bool = False):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_losses': val_losses,
            'best_val_loss': self.best_val_loss,
            'loss_weights': self.loss_weights
        }
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / 'latest.pt'
        torch.save(checkpoint, latest_path)
        
        # Save epoch checkpoint
        epoch_path = self.checkpoint_dir / f'epoch_{self.epoch}.pt'
        torch.save(checkpoint, epoch_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best.pt'
            torch.save(checkpoint, best_path)
            print(f"💾 New best model saved! Val loss: {val_losses['total']:.6f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"📂 Loaded checkpoint from epoch {self.epoch}")
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        milestone_target: Dict[str, float] = None
    ):
        """Main training loop"""
        
        if milestone_target is None:
            milestone_target = {
                'psnr': 25.0,  # Target PSNR for good quality
                'sequence_quality': 0.7,  # Cosine similarity threshold
                'temporal_consistency': 0.01  # Low temporal inconsistency
            }
        
        print(f"\n🚀 Starting Milestone 3 Training")
        print(f"   Epochs: {num_epochs}")
        print(f"   Target PSNR: {milestone_target['psnr']} dB")
        print(f"   Target sequence quality: {milestone_target['sequence_quality']:.1%}")
        print("=" * 60)
        
        # Update scheduler
        self.scheduler.T_max = num_epochs
        
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            
            print(f"\n📅 Epoch {epoch+1}/{num_epochs}")
            
            # Training
            train_losses = self.train_epoch(train_loader)
            
            # Validation
            val_losses = self.validate(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Log to TensorBoard
            if self.writer:
                for name, loss in train_losses.items():
                    self.writer.add_scalar(f'epoch/train_{name}', loss, epoch)
                
                for name, loss in val_losses.items():
                    self.writer.add_scalar(f'epoch/val_{name}', loss, epoch)
                
                self.writer.add_scalar('epoch/learning_rate', self.scheduler.get_last_lr()[0], epoch)
            
            # Print epoch summary
            print(f"   📊 Train Loss: {train_losses['total']:.6f} | Val Loss: {val_losses['total']:.6f}")
            print(f"   📊 Val PSNR: {val_losses['psnr']:.2f} dB | Sequence Quality: {val_losses['sequence_quality']:.1%}")
            print(f"   📊 Temporal Consistency: {val_losses['temporal_consistency']:.6f}")
            
            # Check for improvement
            is_best = val_losses['total'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_losses['total']
            
            # Save checkpoint
            self.save_checkpoint(val_losses, is_best)
            
            # Check milestone achievement
            milestone_achieved = (
                val_losses['psnr'] >= milestone_target['psnr'] and
                val_losses['sequence_quality'] >= milestone_target['sequence_quality'] and
                val_losses['temporal_consistency'] <= milestone_target['temporal_consistency']
            )
            
            if milestone_achieved:
                print(f"\n🎉 MILESTONE 3 ACHIEVED!")
                print(f"   ✅ PSNR: {val_losses['psnr']:.2f} dB (target: {milestone_target['psnr']} dB)")
                print(f"   ✅ Sequence Quality: {val_losses['sequence_quality']:.1%} (target: {milestone_target['sequence_quality']:.1%})")
                print(f"   ✅ Temporal Consistency: {val_losses['temporal_consistency']:.6f} (target: <{milestone_target['temporal_consistency']})")
                print(f"   🚀 Short Sequence Generation capability unlocked!")
                
                # Save milestone checkpoint
                milestone_path = self.checkpoint_dir / 'milestone3_achieved.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'val_losses': val_losses,
                    'milestone_target': milestone_target,
                    'milestone_achieved': True
                }, milestone_path)
                
                break
            
            # Early stopping check
            if epoch > 20 and val_losses['total'] > self.best_val_loss * 1.1:
                print(f"⏳ Early stopping triggered. Best val loss: {self.best_val_loss:.6f}")
                break
        
        if self.writer:
            self.writer.close()
        
        # Final summary
        print(f"\n{'='*60}")
        print(f"🏁 Training completed!")
        print(f"   Best validation loss: {self.best_val_loss:.6f}")
        print(f"   Total epochs: {self.epoch + 1}")
        print(f"   Milestone 3 achieved: {'✅ YES' if milestone_achieved else '⏳ Needs more training'}")


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Sequence Generator (Milestone 3)')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--sequence_length', type=int, default=32, help='Sequence length')
    parser.add_argument('--num_sequences', type=int, default=2000, help='Number of training sequences')
    parser.add_argument('--image_size', type=int, default=256, help='Image size')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/sequence', help='Checkpoint directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--vqvae_checkpoint', type=str, default=None, help='Pre-trained VQ-VAE checkpoint')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Create model
    model = SequenceGenerator(
        in_channels=3,
        latent_dim=256,
        num_embeddings=512,
        d_model=512,
        num_layers=8,
        num_heads=8,
        max_sequence_length=64,
        freeze_vqvae=True
    )
    
    # Load pre-trained VQ-VAE if specified
    if args.vqvae_checkpoint:
        model.load_vqvae_checkpoint(args.vqvae_checkpoint)
    
    # Create trainer
    trainer = SequenceTrainer(
        model=model,
        device=device,
        learning_rate=args.learning_rate,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Create data loaders
    print(f"📊 Creating datasets...")
    
    train_loader = create_sequence_dataloader(
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        num_sequences=args.num_sequences,
        image_size=(args.image_size, args.image_size),
        num_workers=2,
        physics_enabled=True
    )
    
    val_loader = create_sequence_dataloader(
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        num_sequences=args.num_sequences // 10,  # Smaller validation set
        image_size=(args.image_size, args.image_size),
        num_workers=2,
        physics_enabled=True
    )
    
    print(f"   Training batches: {len(train_loader)}")
    print(f"   Validation batches: {len(val_loader)}")
    
    # Start training
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        milestone_target={
            'psnr': 25.0,
            'sequence_quality': 0.7,
            'temporal_consistency': 0.01
        }
    )


if __name__ == "__main__":
    main()
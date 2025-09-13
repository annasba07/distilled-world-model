"""
Training script for Temporal Prediction - Milestone 2: Next-Frame Prediction
Goal: Predict next frames with 80% accuracy in temporal sequences
"""

import os
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

from models.temporal_predictor import WorldModelWithPrediction, calculate_temporal_metrics
from data.temporal_dataset import create_temporal_dataloaders


class TemporalTrainer:
    """Trainer for temporal prediction model"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any]
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Setup device
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=config['learning_rate'],
            betas=(0.9, 0.95),
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # Setup scheduler
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=config['learning_rate'],
            total_steps=config['num_epochs'] * len(train_loader),
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        # Setup logging
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(config['log_dir'])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        
        # Tracking
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_accuracy = 0
        
        # Mixed precision training
        self.use_amp = config.get('use_amp', True) and self.device.type == 'cuda'
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # Loss weights
        self.mse_weight = config.get('mse_weight', 1.0)
        self.temporal_weight = config.get('temporal_weight', 0.5)
        self.perceptual_weight = config.get('perceptual_weight', 0.1)
        
    def compute_loss(self, predicted_latents: torch.Tensor, target_latents: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute combined loss for temporal prediction"""
        
        # MSE loss in latent space
        mse_loss = F.mse_loss(predicted_latents, target_latents)
        
        # Temporal consistency loss
        if predicted_latents.shape[1] > 1:
            pred_diff = torch.diff(predicted_latents, dim=1)
            target_diff = torch.diff(target_latents, dim=1)
            temporal_loss = F.mse_loss(pred_diff, target_diff)
        else:
            temporal_loss = torch.tensor(0.0, device=predicted_latents.device)
        
        # Perceptual loss (cosine similarity)
        pred_flat = predicted_latents.flatten(2).mean(dim=2)
        target_flat = target_latents.flatten(2).mean(dim=2)
        cosine_sim = F.cosine_similarity(pred_flat, target_flat, dim=1).mean()
        perceptual_loss = 1.0 - cosine_sim
        
        # Combined loss
        total_loss = (
            self.mse_weight * mse_loss +
            self.temporal_weight * temporal_loss +
            self.perceptual_weight * perceptual_loss
        )
        
        return {
            'total': total_loss,
            'mse': mse_loss,
            'temporal': temporal_loss,
            'perceptual': perceptual_loss
        }
    
    def calculate_accuracy(self, predicted_latents: torch.Tensor, target_latents: torch.Tensor) -> float:
        """Calculate prediction accuracy"""
        # Threshold-based accuracy
        threshold = 0.1
        diff = torch.abs(predicted_latents - target_latents)
        accuracy = (diff < threshold).float().mean().item()
        return accuracy
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        epoch_losses = {
            'total': 0,
            'mse': 0,
            'temporal': 0,
            'perceptual': 0,
            'accuracy': 0
        }
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}/{self.config['num_epochs']}")
        
        for batch_idx, batch in enumerate(pbar):
            input_frames = batch['input_frames'].to(self.device)
            target_frames = batch['target_frames'].to(self.device)
            
            # Create full sequence for model input
            full_sequence = torch.cat([input_frames, target_frames], dim=1)
            
            # Mixed precision training
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                # Forward pass
                predicted_latents, target_latents = self.model(full_sequence)
                
                # Calculate losses
                losses = self.compute_loss(predicted_latents, target_latents)
                total_loss = losses['total']
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            
            self.scheduler.step()
            
            # Calculate accuracy
            accuracy = self.calculate_accuracy(predicted_latents, target_latents)
            
            # Update metrics
            epoch_losses['total'] += total_loss.item()
            epoch_losses['mse'] += losses['mse'].item()
            epoch_losses['temporal'] += losses['temporal'].item()
            epoch_losses['perceptual'] += losses['perceptual'].item()
            epoch_losses['accuracy'] += accuracy
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'acc': f"{accuracy:.3f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Log to tensorboard
            if self.global_step % self.config['log_interval'] == 0:
                for key, value in losses.items():
                    self.writer.add_scalar(f'train/{key}_loss', value.item(), self.global_step)
                self.writer.add_scalar('train/accuracy', accuracy, self.global_step)
                self.writer.add_scalar('train/lr', self.scheduler.get_last_lr()[0], self.global_step)
            
            # Save sample predictions
            if self.global_step % self.config['sample_interval'] == 0:
                self.save_prediction_samples(input_frames[:2], predicted_latents[:2], target_latents[:2], 'train')
            
            self.global_step += 1
        
        # Average losses
        num_batches = len(self.train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        
        val_losses = {
            'total': 0,
            'mse': 0,
            'temporal': 0,
            'perceptual': 0,
            'accuracy': 0
        }
        
        num_batches = 0
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            input_frames = batch['input_frames'].to(self.device)
            target_frames = batch['target_frames'].to(self.device)
            
            # Create full sequence
            full_sequence = torch.cat([input_frames, target_frames], dim=1)
            
            # Forward pass
            predicted_latents, target_latents = self.model(full_sequence)
            
            # Calculate losses
            losses = self.compute_loss(predicted_latents, target_latents)
            
            # Calculate accuracy
            accuracy = self.calculate_accuracy(predicted_latents, target_latents)
            
            # Update metrics
            val_losses['total'] += losses['total'].item()
            val_losses['mse'] += losses['mse'].item()
            val_losses['temporal'] += losses['temporal'].item()
            val_losses['perceptual'] += losses['perceptual'].item()
            val_losses['accuracy'] += accuracy
            
            num_batches += 1
        
        # Average losses
        for key in val_losses:
            val_losses[key] /= num_batches
        
        # Log to tensorboard
        for key, value in val_losses.items():
            self.writer.add_scalar(f'val/{key}', value, self.current_epoch)
        
        # Save sample predictions
        self.save_prediction_samples(input_frames[:4], predicted_latents[:4], target_latents[:4], 'val')
        
        return val_losses
    
    def save_prediction_samples(self, input_frames: torch.Tensor, 
                              predicted_latents: torch.Tensor, 
                              target_latents: torch.Tensor, split: str):
        """Save sample predictions for visualization"""
        
        # Decode latents to frames
        try:
            # Decode predicted frames
            predicted_frames = self.model.decode_latent_sequence(predicted_latents)
            
            # Get target frames (decode targets)
            target_frames = self.model.decode_latent_sequence(target_latents)
            
            # Denormalize frames
            input_frames = input_frames * 0.5 + 0.5
            predicted_frames = predicted_frames * 0.5 + 0.5
            target_frames = target_frames * 0.5 + 0.5
            
            # Create visualization
            batch_size = min(2, input_frames.shape[0])
            seq_len = input_frames.shape[1]
            pred_len = predicted_frames.shape[1]
            
            fig, axes = plt.subplots(batch_size, seq_len + pred_len + pred_len, 
                                   figsize=((seq_len + pred_len * 2) * 2, batch_size * 2))
            
            if batch_size == 1:
                axes = axes.reshape(1, -1)
            
            for b in range(batch_size):
                col_idx = 0
                
                # Input frames
                for t in range(seq_len):
                    frame = input_frames[b, t].cpu().permute(1, 2, 0).numpy()
                    axes[b, col_idx].imshow(np.clip(frame, 0, 1))
                    axes[b, col_idx].set_title(f'Input {t+1}', fontsize=8)
                    axes[b, col_idx].axis('off')
                    col_idx += 1
                
                # Predicted frames
                for t in range(pred_len):
                    frame = predicted_frames[b, t].cpu().permute(1, 2, 0).numpy()
                    axes[b, col_idx].imshow(np.clip(frame, 0, 1))
                    axes[b, col_idx].set_title(f'Pred {t+1}', fontsize=8)
                    axes[b, col_idx].axis('off')
                    col_idx += 1
                
                # Target frames
                for t in range(pred_len):
                    frame = target_frames[b, t].cpu().permute(1, 2, 0).numpy()
                    axes[b, col_idx].imshow(np.clip(frame, 0, 1))
                    axes[b, col_idx].set_title(f'Target {t+1}', fontsize=8)
                    axes[b, col_idx].axis('off')
                    col_idx += 1
            
            plt.suptitle(f'Temporal Prediction - {split.upper()}', fontsize=12)
            plt.tight_layout()
            
            # Save figure
            save_path = self.log_dir / f'{split}_predictions_epoch_{self.current_epoch:03d}.png'
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            # Log to tensorboard
            self.writer.add_figure(f'{split}/predictions', fig, self.current_epoch)
            
        except Exception as e:
            print(f"Warning: Could not save prediction samples: {e}")
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_accuracy': self.best_val_accuracy,
            'config': self.config
        }
        
        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / 'latest.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best.pt'
            torch.save(checkpoint, best_path)
            print(f"💾 Saved best model with accuracy: {self.best_val_accuracy:.3f}")
        
        # Save periodic checkpoint
        if self.current_epoch % self.config['save_interval'] == 0:
            epoch_path = self.checkpoint_dir / f'epoch_{self.current_epoch:03d}.pt'
            torch.save(checkpoint, epoch_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_accuracy = checkpoint['best_val_accuracy']
        
        print(f"📂 Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self):
        """Main training loop"""
        print(f"🚀 Starting temporal prediction training on {self.device}")
        print(f"📊 Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e6:.2f}M")
        
        if self.device.type == 'cuda':
            print(f"💾 Initial GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        for epoch in range(self.current_epoch, self.config['num_epochs']):
            self.current_epoch = epoch
            
            # Train
            train_losses = self.train_epoch()
            
            # Validate
            val_losses = self.validate()
            
            # Check if best model
            is_best = val_losses['accuracy'] > self.best_val_accuracy
            if is_best:
                self.best_val_accuracy = val_losses['accuracy']
            
            # Save checkpoint
            self.save_checkpoint(is_best)
            
            # Print epoch summary
            print(f"\n📈 Epoch {epoch+1}/{self.config['num_epochs']} Summary:")
            print(f"  Train Loss: {train_losses['total']:.4f} (MSE: {train_losses['mse']:.4f}, Temporal: {train_losses['temporal']:.4f})")
            print(f"  Train Accuracy: {train_losses['accuracy']:.3f}")
            print(f"  Val Loss: {val_losses['total']:.4f} (MSE: {val_losses['mse']:.4f}, Temporal: {val_losses['temporal']:.4f})")
            print(f"  Val Accuracy: {val_losses['accuracy']:.3f} {'🎯 (New Best!)' if is_best else ''}")
            print(f"  Learning Rate: {self.scheduler.get_last_lr()[0]:.6f}")
            
            if self.device.type == 'cuda':
                print(f"  GPU Memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
            
            # Check milestone
            if val_losses['accuracy'] >= 0.8:
                print("\n🎉 MILESTONE 2 ACHIEVED! Prediction accuracy > 80%")
                print("✅ Next-Frame Prediction capability unlocked!")
                self.save_milestone_report(val_losses)
                
                if val_losses['accuracy'] >= 0.85:
                    print("🔥 Exceeding target! Ready for next milestone.")
                    break
        
        print("\n✨ Training complete!")
        self.writer.close()
    
    def save_milestone_report(self, metrics: Dict[str, float]):
        """Save milestone achievement report"""
        report = {
            'milestone': 'Next-Frame Prediction',
            'achieved': True,
            'epoch': self.current_epoch,
            'metrics': {
                'accuracy': metrics['accuracy'],
                'mse_loss': metrics['mse'],
                'temporal_consistency': metrics['temporal'],
                'perceptual_loss': metrics['perceptual']
            },
            'trainable_params': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'gpu_memory_gb': torch.cuda.max_memory_allocated() / 1024**3 if self.device.type == 'cuda' else 0,
            'timestamp': datetime.now().isoformat()
        }
        
        report_path = self.checkpoint_dir / 'milestone_2_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📝 Milestone report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Train Temporal Prediction Model')
    parser.add_argument('--data_dir', type=str, default='datasets/temporal', help='Path to temporal data')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/temporal', help='Path to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs/temporal', help='Path to save logs')
    parser.add_argument('--vqvae_checkpoint', type=str, default='checkpoints/vqvae/best.pt', help='Path to VQ-VAE checkpoint')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--sequence_length', type=int, default=8, help='Input sequence length')
    parser.add_argument('--prediction_horizon', type=int, default=1, help='Number of frames to predict')
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--use_amp', action='store_true', help='Use mixed precision training')
    parser.add_argument('--freeze_vqvae', action='store_true', default=True, help='Freeze VQ-VAE weights')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--log_interval', type=int, default=10, help='Log interval')
    parser.add_argument('--sample_interval', type=int, default=100, help='Sample interval')
    parser.add_argument('--save_interval', type=int, default=5, help='Save interval')
    
    args = parser.parse_args()
    
    # Create config
    config = vars(args)
    
    # Create dataloaders
    print("📊 Loading temporal data...")
    train_loader, val_loader, test_loader = create_temporal_dataloaders(
        data_dir=args.data_dir,
        sequence_length=args.sequence_length,
        prediction_horizon=args.prediction_horizon,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Create model
    print("🔨 Building temporal prediction model...")
    model = WorldModelWithPrediction(
        in_channels=3,
        latent_dim=256,
        num_embeddings=512,
        vqvae_hidden_dims=[64, 128, 256, 512],
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_sequence_length=args.sequence_length * 2,
        freeze_vqvae=args.freeze_vqvae
    )
    
    # Load VQ-VAE checkpoint if available
    if Path(args.vqvae_checkpoint).exists():
        model.load_vqvae_checkpoint(args.vqvae_checkpoint)
    else:
        print("⚠️ No VQ-VAE checkpoint found, using random initialization")
    
    # Create trainer
    trainer = TemporalTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()
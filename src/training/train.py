import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import wandb
from einops import rearrange
import numpy as np
from typing import Optional, Dict, Any

from ..models.improved_vqvae import ImprovedVQVAE
from ..models.dynamics import DynamicsModel, WorldModel


class VQVAETrainer(pl.LightningModule):
    def __init__(self, 
                 model_config: Dict[str, Any],
                 learning_rate: float = 1e-4,
                 beta: float = 0.25):
        super().__init__()
        self.save_hyperparameters()
        
        self.vqvae = ImprovedVQVAE(**model_config)
        self.beta = beta
        self.learning_rate = learning_rate
        
    def forward(self, x):
        return self.vqvae(x)
    
    def training_step(self, batch, batch_idx):
        images = batch['images']
        recon, vq_loss, perplexity = self(images)
        
        recon_loss = F.mse_loss(recon, images)
        loss = recon_loss + self.beta * vq_loss
        
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/recon_loss', recon_loss)
        self.log('train/vq_loss', vq_loss)
        self.log('train/perplexity', perplexity)
        
        if batch_idx % 100 == 0:
            self.log_images(images, recon, 'train')
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images = batch['images']
        recon, vq_loss, perplexity = self(images)
        
        recon_loss = F.mse_loss(recon, images)
        loss = recon_loss + self.beta * vq_loss
        
        self.log('val/loss', loss, prog_bar=True)
        self.log('val/recon_loss', recon_loss)
        self.log('val/vq_loss', vq_loss)
        self.log('val/perplexity', perplexity)
        
        if batch_idx == 0:
            self.log_images(images, recon, 'val')
        
        return loss
    
    def log_images(self, original, reconstructed, split):
        n_images = min(4, original.shape[0])
        
        orig_grid = rearrange(original[:n_images], 'b c h w -> (b h) w c').cpu().numpy()
        recon_grid = rearrange(reconstructed[:n_images], 'b c h w -> (b h) w c').cpu().numpy()
        
        comparison = np.concatenate([orig_grid, recon_grid], axis=1)
        
        self.logger.log_image(
            f'{split}/reconstruction',
            [wandb.Image(comparison, caption="Original (left) vs Reconstructed (right)")]
        )
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return [optimizer], [scheduler]


class WorldModelTrainer(pl.LightningModule):
    def __init__(self,
                 vqvae_checkpoint: str,
                 dynamics_config: Dict[str, Any],
                 learning_rate: float = 1e-4,
                 gradient_clip: float = 1.0):
        super().__init__()
        self.save_hyperparameters()
        
        # Load VQ-VAE weights from a standard checkpoint
        try:
            ckpt = torch.load(vqvae_checkpoint, map_location='cpu')
            state_dict = ckpt.get('model_state_dict', ckpt)
            self.vqvae = ImprovedVQVAE()
            self.vqvae.load_state_dict(state_dict, strict=False)
            print(f"Loaded VQ-VAE weights from {vqvae_checkpoint}")
        except Exception as e:
            print(f"Warning: Failed to load VQ-VAE checkpoint '{vqvae_checkpoint}': {e}. Using randomly initialized VQ-VAE.")
            self.vqvae = ImprovedVQVAE()
        self.vqvae.eval()
        for param in self.vqvae.parameters():
            param.requires_grad = False
        
        self.dynamics = DynamicsModel(**dynamics_config)
        self.world_model = WorldModel(self.vqvae, self.dynamics)
        
        self.learning_rate = learning_rate
        self.gradient_clip = gradient_clip
        self.scaler = GradScaler()
        
    def forward(self, frames, actions=None):
        return self.world_model(frames, actions)
    
    def training_step(self, batch, batch_idx):
        frames = batch['frames']
        actions = batch.get('actions', None)
        
        with autocast():
            predicted_latents, target_latents = self(frames, actions)
            loss = F.mse_loss(predicted_latents, target_latents)
        
        self.log('train/loss', loss, prog_bar=True)
        
        if batch_idx % 100 == 0:
            self.log_generation_quality(frames, actions, 'train')
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        frames = batch['frames']
        actions = batch.get('actions', None)
        
        with torch.no_grad():
            predicted_latents, target_latents = self(frames, actions)
            loss = F.mse_loss(predicted_latents, target_latents)
        
        self.log('val/loss', loss, prog_bar=True)
        
        if batch_idx == 0:
            self.log_generation_quality(frames, actions, 'val')
        
        return loss
    
    def log_generation_quality(self, frames, actions, split):
        with torch.no_grad():
            initial_frame = frames[:1, 0]
            num_steps = min(16, frames.shape[1] - 1)
            
            if actions is not None:
                action_seq = actions[:1, :num_steps]
            else:
                action_seq = None
            
            generated = self.world_model.generate_trajectory(
                initial_frame, action_seq, num_steps
            )
            
            real_seq = frames[:1, 1:num_steps+1]
            
            comparison_frames = []
            for t in range(num_steps):
                real_frame = real_seq[0, t].cpu().numpy().transpose(1, 2, 0)
                gen_frame = generated[0, t].cpu().numpy().transpose(1, 2, 0)
                comparison = np.concatenate([real_frame, gen_frame], axis=1)
                comparison_frames.append(comparison)
            
            self.logger.log_image(
                f'{split}/generation',
                [wandb.Image(f, caption=f"Step {i}: Real (left) vs Generated (right)") 
                 for i, f in enumerate(comparison_frames[:4])]
            )
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.dynamics.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.95)
        )
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.05
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            }
        }
    
    def on_before_optimizer_step(self, optimizer):
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.dynamics.parameters(),
                self.gradient_clip
            )


def train_vqvae(config):
    model = VQVAETrainer(
        model_config=config['model'],
        learning_rate=config['training']['learning_rate'],
        beta=config['training']['beta']
    )
    
    wandb_logger = WandbLogger(
        project="lightweight-world-model",
        name="vqvae-training"
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/vqvae',
        filename='vqvae-{epoch:02d}-{val_loss:.4f}',
        save_top_k=3,
        monitor='val/loss',
        mode='min'
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val/loss',
        patience=10,
        mode='min'
    )
    
    trainer = pl.Trainer(
        max_epochs=config['training']['epochs'],
        accelerator='gpu',
        devices=1,
        precision='16-mixed',
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=wandb_logger,
        gradient_clip_val=1.0
    )
    
    trainer.fit(model)
    
    return checkpoint_callback.best_model_path


def train_dynamics(config, vqvae_checkpoint):
    model = WorldModelTrainer(
        vqvae_checkpoint=vqvae_checkpoint,
        dynamics_config=config['dynamics'],
        learning_rate=config['training']['learning_rate'],
        gradient_clip=config['training']['gradient_clip']
    )
    
    wandb_logger = WandbLogger(
        project="lightweight-world-model",
        name="dynamics-training"
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/dynamics',
        filename='dynamics-{epoch:02d}-{val_loss:.4f}',
        save_top_k=3,
        monitor='val/loss',
        mode='min'
    )
    
    trainer = pl.Trainer(
        max_epochs=config['training']['epochs'],
        accelerator='gpu',
        devices=1,
        precision='16-mixed',
        callbacks=[checkpoint_callback],
        logger=wandb_logger,
        accumulate_grad_batches=4
    )
    
    trainer.fit(model)
    
    return checkpoint_callback.best_model_path

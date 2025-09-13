import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from .train import VQVAETrainer, WorldModelTrainer
from ..data.game_dataset import create_dataloaders as create_image_dataloaders
from ..data.temporal_dataset import create_temporal_dataloaders


def train_vqvae_cli(args: argparse.Namespace) -> None:
    # Dataloaders
    train_loader, val_loader, _ = create_image_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=(args.image_size, args.image_size),
        sequence_length=1,
    )

    # Lightning module
    model = VQVAETrainer(
        model_config={
            'in_channels': 3,
            'latent_dim': args.latent_dim,
            'num_embeddings': args.num_embeddings,
            'hidden_dims': [64, 128, 256, 512],
            'use_ema': True,
            'use_attention': True,
        },
        learning_rate=args.learning_rate,
        beta=args.vq_weight,
    )

    # Logger + callbacks
    logger = WandbLogger(project="lightweight-world-model", name="vqvae-training") if args.wandb else None
    checkpoint = ModelCheckpoint(
        dirpath='checkpoints/vqvae',
        filename='vqvae-{epoch:02d}-{val_loss:.4f}',
        save_top_k=3,
        monitor='val/loss',
        mode='min',
    )
    early_stop = EarlyStopping(monitor='val/loss', patience=10, mode='min')

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='gpu' if (args.device == 'cuda' and torch.cuda.is_available()) else 'cpu',
        devices=1,
        precision='16-mixed' if args.amp else 32,
        callbacks=[checkpoint, early_stop],
        logger=logger,
        gradient_clip_val=1.0,
    )

    # Fit
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


def train_dynamics_cli(args: argparse.Namespace) -> None:
    # Dataloaders
    train_loader, val_loader, _ = create_temporal_dataloaders(
        data_dir=args.data_dir,
        sequence_length=args.sequence_length,
        prediction_horizon=args.prediction_horizon,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        frame_skip=args.frame_skip,
    )

    # Lightning module
    model = WorldModelTrainer(
        vqvae_checkpoint=args.vqvae_checkpoint,
        dynamics_config={
            'latent_dim': args.latent_dim,
            'hidden_dim': args.hidden_dim,
            'num_layers': args.num_layers,
            'action_dim': 32,
            'num_actions': 256,
            'context_length': args.sequence_length,
            'dropout': 0.1,
        },
        learning_rate=args.learning_rate,
        gradient_clip=1.0,
    )

    # Logger + callbacks
    logger = WandbLogger(project="lightweight-world-model", name="dynamics-training") if args.wandb else None
    checkpoint = ModelCheckpoint(
        dirpath='checkpoints/dynamics',
        filename='dynamics-{epoch:02d}-{val_loss:.4f}',
        save_top_k=3,
        monitor='val/loss',
        mode='min',
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='gpu' if (args.device == 'cuda' and torch.cuda.is_available()) else 'cpu',
        devices=1,
        precision='16-mixed' if args.amp else 32,
        callbacks=[checkpoint],
        logger=logger,
        accumulate_grad_batches=4,
    )

    # Fit
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


def main():
    parser = argparse.ArgumentParser(description="Lightning training CLI")
    sub = parser.add_subparsers(dest='cmd', required=True)

    # VQ-VAE
    p_vq = sub.add_parser('vqvae', help='Train ImprovedVQVAE with Lightning')
    p_vq.add_argument('--data_dir', type=str, default='datasets/raw')
    p_vq.add_argument('--epochs', type=int, default=50)
    p_vq.add_argument('--batch_size', type=int, default=32)
    p_vq.add_argument('--num_workers', type=int, default=4)
    p_vq.add_argument('--image_size', type=int, default=256)
    p_vq.add_argument('--learning_rate', type=float, default=1e-4)
    p_vq.add_argument('--vq_weight', type=float, default=0.25)
    p_vq.add_argument('--num_embeddings', type=int, default=512)
    p_vq.add_argument('--latent_dim', type=int, default=256)
    p_vq.add_argument('--amp', action='store_true')
    p_vq.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    p_vq.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
    p_vq.set_defaults(func=train_vqvae_cli)

    # Dynamics
    p_dyn = sub.add_parser('dynamics', help='Train world model dynamics with Lightning')
    p_dyn.add_argument('--data_dir', type=str, default='datasets/temporal')
    p_dyn.add_argument('--epochs', type=int, default=30)
    p_dyn.add_argument('--batch_size', type=int, default=8)
    p_dyn.add_argument('--num_workers', type=int, default=4)
    p_dyn.add_argument('--sequence_length', type=int, default=16)
    p_dyn.add_argument('--prediction_horizon', type=int, default=1)
    p_dyn.add_argument('--frame_skip', type=int, default=1)
    p_dyn.add_argument('--learning_rate', type=float, default=1e-4)
    p_dyn.add_argument('--latent_dim', type=int, default=256*16)  # placeholder; will be adapted in model
    p_dyn.add_argument('--hidden_dim', type=int, default=768)
    p_dyn.add_argument('--num_layers', type=int, default=12)
    p_dyn.add_argument('--vqvae_checkpoint', type=str, required=True)
    p_dyn.add_argument('--amp', action='store_true')
    p_dyn.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    p_dyn.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
    p_dyn.set_defaults(func=train_dynamics_cli)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()


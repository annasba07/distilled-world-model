#!/usr/bin/env python3
"""
Train Continuous Dynamics Model - EXP-2

Key differences from EXP-1 (discrete):
1. Predicts continuous VQ-VAE latents instead of discrete tokens
2. Uses MSE loss instead of cross-entropy
3. Much stronger learning signal: dense gradients vs sparse classification

Expected improvement: 10-100× better learning efficiency

Usage:
    python scripts/train_dynamics_mario_exp2.py --epochs 20 --batch_size 4
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.tokenizers import CosmosInspiredTokenizer
from models.actions.continuous_dynamics_model import ContinuousDynamicsModel


class ContinuousGameplayDataset(Dataset):
    """Dataset that extracts continuous VQ-VAE latents (before quantization)."""

    def __init__(self, data_dir, tokenizer, device='cpu', max_sequences=None, use_cache=True, batch_size=16):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.device = device

        # Load all .npz files
        npz_files = sorted(list(self.data_dir.glob('*.npz')))

        if max_sequences is not None:
            npz_files = npz_files[:max_sequences]

        print(f"Found {len(npz_files)} sequence files")

        # Check for cached latents
        import pickle
        import hashlib

        cache_key = f"{data_dir}_{len(npz_files)}_continuous"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
        cache_file = Path(f"data/latent_cache_{cache_hash}.pkl")

        if use_cache and cache_file.exists():
            print(f"⚡ Loading cached continuous latents from {cache_file}...")
            with open(cache_file, 'rb') as f:
                self.latent_sequences = pickle.load(f)
            print(f"✅ Loaded {len(self.latent_sequences)} sequences with continuous latents from cache")
            if len(self.latent_sequences) > 0:
                print(f"   Frames per sequence: {self.latent_sequences[0]['latents'].shape[0]}")
                print(f"   Latent dim: {self.latent_sequences[0]['latents'].shape[1]}")
            return

        # Extract continuous latents with streaming batching
        print(f"Extracting continuous latents with batch_size={batch_size}...")
        print("(This will be cached for future runs)")
        self.latent_sequences = []

        num_batches = (len(npz_files) + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(num_batches), desc="Extracting latents"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(npz_files))
            batch_files = npz_files[start_idx:end_idx]

            # Load batch
            batch_sequences = []
            for npz_file in batch_files:
                try:
                    data = np.load(npz_file, allow_pickle=True)
                    frames = data['frames']  # [T, H, W, 3]
                    batch_sequences.append(frames)
                except Exception as e:
                    print(f"Failed to load {npz_file.name}: {e}")
                    continue

            try:
                with torch.no_grad():
                    # Stack sequences: List of [T, H, W, 3] -> [B, C, T, H, W]
                    batch_tensors = []
                    for frames in batch_sequences:
                        frames_tensor = torch.from_numpy(frames).float()
                        frames_tensor = frames_tensor.permute(3, 0, 1, 2)  # [T, H, W, 3] -> [3, T, H, W]
                        if frames_tensor.max() > 1.0:
                            frames_tensor = frames_tensor / 255.0
                        batch_tensors.append(frames_tensor)

                    batch_tensor = torch.stack(batch_tensors, dim=0).to(device)  # [B, 3, T, H, W]

                    # Extract continuous latents BEFORE quantization
                    latents = self._extract_continuous_latents(batch_tensor)
                    # latents: [B, T, H', W', latent_dim]

                    # Process each sequence
                    for i in range(latents.shape[0]):
                        seq_latents = latents[i]  # [T, H', W', latent_dim]
                        T, H, W, D = seq_latents.shape

                        # Flatten spatial dims: [T, H', W', D] -> [T, H'*W', D]
                        latents_flat = seq_latents.reshape(T, H * W, D)  # [T, seq_len, latent_dim]

                        # Create dummy actions
                        num_actions = 8
                        actions = torch.randint(0, num_actions, (T,))

                        # Store sequence
                        self.latent_sequences.append({
                            'latents': latents_flat.cpu(),  # [T, seq_len, latent_dim]
                            'actions': actions  # [T]
                        })

            except Exception as e:
                print(f"Failed to process batch {batch_idx}: {e}")
                # Fall back to sequential
                for npz_file in batch_files:
                    try:
                        data = np.load(npz_file, allow_pickle=True)
                        frames = data['frames']
                        with torch.no_grad():
                            frames_tensor = torch.from_numpy(frames).float().to(device)
                            frames_tensor = frames_tensor.permute(3, 0, 1, 2).unsqueeze(0)  # [1, 3, T, H, W]
                            if frames_tensor.max() > 1.0:
                                frames_tensor = frames_tensor / 255.0

                            latents = self._extract_continuous_latents(frames_tensor)
                            T, H, W, D = latents.shape[1:]
                            latents_flat = latents[0].reshape(T, H * W, D)

                            num_actions = 8
                            actions = torch.randint(0, num_actions, (T,))
                            self.latent_sequences.append({
                                'latents': latents_flat.cpu(),
                                'actions': actions
                            })
                    except Exception as e2:
                        print(f"Failed to process {npz_file.name}: {e2}")
                        continue

            # Free memory
            del batch_sequences
            if device.type == 'mps' or device.type == 'cuda':
                torch.mps.empty_cache() if device.type == 'mps' else torch.cuda.empty_cache()

        print(f"✅ Extracted {len(self.latent_sequences)} sequences")
        if len(self.latent_sequences) > 0:
            print(f"   Frames per sequence: {self.latent_sequences[0]['latents'].shape[0]}")
            print(f"   Latent dim: {self.latent_sequences[0]['latents'].shape[-1]}")

        # Save to cache
        print(f"💾 Saving latents to cache: {cache_file}")
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(self.latent_sequences, f)
        print(f"✅ Cache saved")

    def _extract_continuous_latents(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract continuous latents from VQ-VAE encoder (before quantization).

        Args:
            x: Input video [B, C, T, H, W]

        Returns:
            latents: Continuous latents [B, T, H', W', latent_dim]
        """
        # Encode through VQ-VAE encoder
        h = self.tokenizer.encoder(x)  # [B, C_enc, T', H', W']
        h = self.tokenizer.pre_quant_conv(h)  # [B, latent_dim, T', H', W']

        # Rearrange: [B, latent_dim, T', H', W'] -> [B, T', H', W', latent_dim]
        B, D, T, H, W = h.shape
        latents = h.permute(0, 2, 3, 4, 1).contiguous()  # [B, T', H', W', D]

        return latents

    def __len__(self):
        return len(self.latent_sequences)

    def __getitem__(self, idx):
        seq = self.latent_sequences[idx]
        latents = seq['latents']  # [T, seq_len, latent_dim]
        actions = seq['actions']  # [T]

        # Randomly sample a frame pair
        T = latents.shape[0]
        if T < 2:
            t = 0
        else:
            t = np.random.randint(0, T - 1)

        latent_t = latents[t]  # [seq_len, latent_dim]
        latent_t1 = latents[t + 1]  # [seq_len, latent_dim]
        action = actions[t]  # scalar

        return latent_t, action, latent_t1


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    total_mse = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc="Training")
    for latent_t, action, latent_t1 in pbar:
        # Move to device
        latent_t = latent_t.to(device)
        action = action.unsqueeze(1).to(device)  # [B] -> [B, 1]
        latent_t1 = latent_t1.to(device)

        # Forward pass
        optimizer.zero_grad()
        output = model(latent_t, action, latent_t1)

        loss = output['loss']
        mse = output['mse']

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Accumulate metrics
        total_loss += loss.item()
        total_mse += mse.item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.6f}",
            'mse': f"{mse.item():.6f}"
        })

    avg_loss = total_loss / num_batches
    avg_mse = total_mse / num_batches

    return avg_loss, avg_mse


@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_mse = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc="Validating", leave=False)
    for latent_t, action, latent_t1 in pbar:
        # Move to device
        latent_t = latent_t.to(device)
        action = action.unsqueeze(1).to(device)
        latent_t1 = latent_t1.to(device)

        # Forward pass
        output = model(latent_t, action, latent_t1)

        loss = output['loss']
        mse = output['mse']

        # Accumulate metrics
        total_loss += loss.item()
        total_mse += mse.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    avg_mse = total_mse / num_batches

    return avg_loss, avg_mse


def main():
    parser = argparse.ArgumentParser(description='Train Continuous Dynamics Model (EXP-2)')
    parser.add_argument('--data_dir', type=str, default='./data/mario_sequences_100x',
                        help='Directory containing .npz sequence files')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/dynamics_mario_exp2',
                        help='Directory to save checkpoints')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--max_sequences', type=int, default=None,
                        help='Maximum number of sequences to use')
    parser.add_argument('--device', type=str, default='mps',
                        help='Device to use (cpu, cuda, mps)')
    args = parser.parse_args()

    # Setup
    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'mps' else 'cpu')
    print(f"Using device: {device}")

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Initialize tokenizer (for latent extraction only)
    print("\n" + "="*80)
    print("Initializing Tokenizer (for latent extraction)...")
    print("="*80)
    tokenizer = CosmosInspiredTokenizer(
        codebook_size=512,
        latent_dim=64,
        resolution=(256, 256)
    ).to(device)
    tokenizer.eval()

    # Load dataset
    print("\n" + "="*80)
    print("Loading Dataset with Continuous Latents...")
    print("="*80)
    dataset = ContinuousGameplayDataset(
        args.data_dir,
        tokenizer,
        device=device,
        max_sequences=args.max_sequences
    )

    # Train/val split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"\nDataset split:")
    print(f"  Train: {len(train_dataset)} sequences")
    print(f"  Val:   {len(val_dataset)} sequences")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    # Initialize continuous dynamics model
    print("\n" + "="*80)
    print("Initializing Continuous Dynamics Model (EXP-2)...")
    print("="*80)

    # Get latent dim from dataset
    latent_t, action, latent_t1 = dataset[0]
    latent_dim = latent_t.shape[-1]  # Should be 64
    print(f"Latent dimension: {latent_dim}")

    # Use same architecture as EXP-1 for fair comparison
    model = ContinuousDynamicsModel(
        latent_dim=latent_dim,
        action_vocab_size=8,
        d_model=512,  # Same as EXP-1
        nhead=16,     # Same as EXP-1
        num_layers=12  # Same as EXP-1
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Training setup
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    print("\n" + "="*80)
    print(f"Training EXP-2: Continuous Latent Prediction")
    print(f"Baseline (EXP-1 discrete): 0.217% accuracy")
    print(f"Expected: 10-100× better learning efficiency")
    print("="*80 + "\n")

    best_val_mse = float('inf')
    training_history = []

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 80)

        # Train
        train_loss, train_mse = train_epoch(model, train_loader, optimizer, device)

        # Validate
        val_loss, val_mse = validate(model, val_loader, device)

        # Update scheduler
        scheduler.step()

        # Log
        print(f"Train Loss: {train_loss:.6f} | Train MSE: {train_mse:.6f}")
        print(f"Val Loss:   {val_loss:.6f} | Val MSE:   {val_mse:.6f}")

        # Note: MSE is not directly comparable to accuracy
        # Lower MSE = better (closer to ground truth latents)
        print(f"Note: Lower MSE = better continuous prediction")

        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_mse': train_mse,
            'val_loss': val_loss,
            'val_mse': val_mse,
            'lr': optimizer.param_groups[0]['lr']
        })

        # Save best model
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mse': val_mse,
                'val_loss': val_loss,
            }, checkpoint_dir / 'best_model.pt')
            print(f"✓ Saved best model (Val MSE: {val_mse:.6f})")

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mse': val_mse,
                'val_loss': val_loss,
            }, checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt')

    # Save training history
    with open(checkpoint_dir / 'training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)

    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print(f"Best validation MSE: {best_val_mse:.6f}")
    print(f"\nCheckpoints saved to: {checkpoint_dir}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Train Dynamics Model on Real Mario Data - Week 7B

Train on real Super Mario Bros gameplay footage collected from YouTube.

Usage:
    python scripts/train_dynamics_mario.py --epochs 20 --batch_size 4
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
from models.actions import DynamicsModel


class RealGameplayDataset(Dataset):
    """Dataset that loads real gameplay sequences from .npz files."""

    def __init__(self, data_dir, tokenizer, device='cpu', max_sequences=None, use_cache=True, batch_size=16):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.device = device

        # Load all .npz files
        npz_files = sorted(list(self.data_dir.glob('*.npz')))

        if max_sequences is not None:
            npz_files = npz_files[:max_sequences]

        print(f"Found {len(npz_files)} sequence files")

        # Check for cached tokens
        import pickle
        import hashlib

        # Create cache filename based on data_dir and number of files
        cache_key = f"{data_dir}_{len(npz_files)}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
        cache_file = Path(f"data/token_cache_{cache_hash}.pkl")

        if use_cache and cache_file.exists():
            print(f"⚡ Loading cached tokens from {cache_file}...")
            with open(cache_file, 'rb') as f:
                self.tokenized_sequences = pickle.load(f)
            print(f"✅ Loaded {len(self.tokenized_sequences)} tokenized sequences from cache")
            if len(self.tokenized_sequences) > 0:
                print(f"   Frames per sequence: {self.tokenized_sequences[0]['tokens'].shape[0]}")
                print(f"   Tokens per frame: {self.tokenized_sequences[0]['tokens'].shape[1]}")
            return

        # Pre-tokenize all sequences with STREAMING batching (memory efficient)
        print(f"Pre-tokenizing sequences with batch_size={batch_size} (this will be cached for future runs)...")
        self.tokenized_sequences = []

        # Stream tokenization in batches (don't load all into memory at once!)
        num_batches = (len(npz_files) + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(num_batches), desc="Tokenizing batches"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(npz_files))
            batch_files = npz_files[start_idx:end_idx]

            # Load only this batch into memory
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
                        # [T, H, W, 3] -> [3, T, H, W]
                        frames_tensor = frames_tensor.permute(3, 0, 1, 2)
                        # Normalize to [0, 1] if needed
                        if frames_tensor.max() > 1.0:
                            frames_tensor = frames_tensor / 255.0
                        batch_tensors.append(frames_tensor)

                    # Stack: [B, 3, T, H, W]
                    batch_tensor = torch.stack(batch_tensors, dim=0).to(device)

                    # Encode batch
                    z_quantized, indices = self.tokenizer.encode(batch_tensor)
                    # indices: [B, T, H, W, 1]

                    # Process each sequence in batch
                    for i in range(indices.shape[0]):
                        seq_indices = indices[i]  # [T, H, W, 1]
                        T, H, W, _ = seq_indices.shape
                        tokens = seq_indices.squeeze(-1).reshape(T, H * W)  # [T, H*W]

                        # Create dummy actions
                        num_actions = 8
                        actions = torch.randint(0, num_actions, (T,))

                        # Store tokenized sequence
                        self.tokenized_sequences.append({
                            'tokens': tokens.cpu(),  # [T, seq_len]
                            'actions': actions  # [T]
                        })

            except Exception as e:
                print(f"Failed to tokenize batch {batch_idx}: {e}")
                # Fall back to sequential for this batch
                for npz_file in batch_files:
                    try:
                        data = np.load(npz_file, allow_pickle=True)
                        frames = data['frames']
                        with torch.no_grad():
                            frames_tensor = torch.from_numpy(frames).float().to(device)
                            frames_tensor = frames_tensor.permute(3, 0, 1, 2).unsqueeze(0)
                            if frames_tensor.max() > 1.0:
                                frames_tensor = frames_tensor / 255.0
                            z_quantized, indices = self.tokenizer.encode(frames_tensor)
                            B, T, H, W, _ = indices.shape
                            tokens = indices.squeeze(0).squeeze(-1).reshape(T, H * W)
                            num_actions = 8
                            actions = torch.randint(0, num_actions, (T,))
                            self.tokenized_sequences.append({
                                'tokens': tokens.cpu(),
                                'actions': actions
                            })
                    except Exception as e2:
                        print(f"Failed to tokenize {npz_file.name}: {e2}")
                        continue

            # Free memory after each batch
            del batch_sequences
            if device.type == 'mps' or device.type == 'cuda':
                torch.mps.empty_cache() if device.type == 'mps' else torch.cuda.empty_cache()

        print(f"✅ Tokenized {len(self.tokenized_sequences)} sequences")
        if len(self.tokenized_sequences) > 0:
            print(f"   Frames per sequence: {self.tokenized_sequences[0]['tokens'].shape[0]}")
            print(f"   Tokens per frame: {self.tokenized_sequences[0]['tokens'].shape[1]}")

        # Save to cache
        print(f"💾 Saving tokens to cache: {cache_file}")
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(self.tokenized_sequences, f)
        print(f"✅ Cache saved")

    def __len__(self):
        return len(self.tokenized_sequences)

    def __getitem__(self, idx):
        seq = self.tokenized_sequences[idx]
        tokens = seq['tokens']  # [T, seq_len]
        actions = seq['actions']  # [T]

        # Randomly sample a frame pair from this sequence
        T = tokens.shape[0]
        if T < 2:
            # Edge case: sequence too short
            t = 0
        else:
            t = np.random.randint(0, T - 1)

        frame_t = tokens[t]  # [seq_len]
        frame_t1 = tokens[t + 1]  # [seq_len]
        action = actions[t]  # scalar

        return frame_t, action, frame_t1


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_accuracy = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc="Training")
    for frame_t, action, frame_t1 in pbar:
        # Move to device
        frame_t = frame_t.to(device)
        action = action.unsqueeze(1).to(device)  # [B] -> [B, 1]
        frame_t1 = frame_t1.to(device)

        # Forward pass
        optimizer.zero_grad()
        output = model(frame_t, action, frame_t1)

        loss = output['loss']
        accuracy = output['accuracy']

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Accumulate metrics
        total_loss += loss.item()
        total_accuracy += accuracy.item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{accuracy.item():.2%}"
        })

    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches

    return avg_loss, avg_accuracy


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_accuracy = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc="Validating", leave=False)
    for frame_t, action, frame_t1 in pbar:
        # Move to device
        frame_t = frame_t.to(device)
        action = action.unsqueeze(1).to(device)  # [B] -> [B, 1]
        frame_t1 = frame_t1.to(device)

        # Forward pass
        output = model(frame_t, action, frame_t1)

        loss = output['loss']
        accuracy = output['accuracy']

        # Accumulate metrics
        total_loss += loss.item()
        total_accuracy += accuracy.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches

    return avg_loss, avg_accuracy


def main():
    parser = argparse.ArgumentParser(description='Train Dynamics Model on Real Mario Data')
    parser.add_argument('--data_dir', type=str, default='./data/mario_sequences_real',
                        help='Directory containing .npz sequence files')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/dynamics_mario',
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

    # Initialize tokenizer
    print("\n" + "="*80)
    print("Initializing Tokenizer...")
    print("="*80)
    tokenizer = CosmosInspiredTokenizer(
        codebook_size=512,
        latent_dim=64,
        resolution=(256, 256)
    ).to(device)
    tokenizer.eval()

    # Load dataset
    print("\n" + "="*80)
    print("Loading Real Mario Dataset...")
    print("="*80)
    dataset = RealGameplayDataset(
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

    # Initialize model
    print("\n" + "="*80)
    print("Initializing Dynamics Model...")
    print("="*80)

    # Get sequence length from dataset
    frame_t, action, frame_t1 = dataset[0]
    seq_len = frame_t.shape[0]  # Get token sequence length from frame

    # EXP-1: Scaled-up model (50M parameters)
    # Baseline was: d_model=256, nhead=8, num_layers=6 (9.7M params, 0.217% val acc)
    # Scaling: d_model=512, nhead=16, num_layers=12 (~50M params)
    model = DynamicsModel(
        frame_vocab_size=tokenizer.codebook_size,
        action_vocab_size=8,  # NES has 8 buttons typically
        d_model=512,  # 256 → 512 (2×)
        nhead=16,     # 8 → 16 (2×)
        num_layers=12  # 6 → 12 (2×)
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    print("\n" + "="*80)
    print(f"Training on Real Mario Data - {len(train_dataset)} sequences")
    print(f"Baseline (Synthetic): 0.05% accuracy")
    print("="*80 + "\n")

    best_val_acc = 0.0
    training_history = []

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 80)

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step()

        # Log
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.2f}%")

        # Compare to baseline
        improvement = (val_acc - 0.0005) * 100  # 0.0005 = 0.05%
        print(f"Improvement over synthetic: {improvement:+.2f}%")

        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': optimizer.param_groups[0]['lr']
        })

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, checkpoint_dir / 'best_model.pt')
            print(f"✓ Saved best model (Val Acc: {val_acc:.2f}%)")

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt')

    # Save training history
    with open(checkpoint_dir / 'training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)

    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Improvement over synthetic (0.05%): {best_val_acc - 0.05:+.2f}%")
    print(f"\nCheckpoints saved to: {checkpoint_dir}")


if __name__ == '__main__':
    main()

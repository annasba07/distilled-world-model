#!/usr/bin/env python3
"""
Quick training script for World Model using available data
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from pathlib import Path
import torchvision.transforms as transforms
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.models.improved_vqvae import ImprovedVQVAE
from src.models.dynamics import DynamicsModel, WorldModel

class SimpleFrameDataset(Dataset):
    """Simple dataset for video frames"""

    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform

        # Find all PNG files
        self.frame_paths = list(self.root_dir.glob("**/*.png"))
        print(f"Found {len(self.frame_paths)} frames")

        if len(self.frame_paths) == 0:
            raise ValueError(f"No PNG files found in {root_dir}")

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.frame_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image

def train_vqvae_simple(data_dir="datasets/youtube/frames",
                       epochs=10,
                       batch_size=4,
                       learning_rate=1e-4,
                       device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Train VQ-VAE on available frames"""

    print(f"Training on device: {device}")

    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Create dataset and dataloader
    dataset = SimpleFrameDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create model
    model = ImprovedVQVAE(
        in_channels=3,
        latent_dim=256,
        num_embeddings=512,
        hidden_dims=[64, 128, 256],  # Reduced for small dataset
        use_ema=True,
        use_attention=False  # Disable attention for faster training on CPU
    ).to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    print(f"Starting training for {epochs} epochs...")
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        total_recon_loss = 0
        total_vq_loss = 0

        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, images in enumerate(progress):
            images = images.to(device)

            # Forward pass
            recon, vq_stats = model(images)

            # Calculate losses
            recon_loss = nn.functional.mse_loss(recon, images)
            vq_loss = vq_stats.get('loss', torch.tensor(0.0).to(device))
            loss = recon_loss + 0.25 * vq_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update stats
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()

            # Update progress bar
            progress.set_postfix({
                'loss': f"{loss.item():.4f}",
                'recon': f"{recon_loss.item():.4f}",
                'vq': f"{vq_loss.item():.4f}"
            })

        # Print epoch stats
        avg_loss = total_loss / len(dataloader)
        avg_recon = total_recon_loss / len(dataloader)
        avg_vq = total_vq_loss / len(dataloader)

        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Recon={avg_recon:.4f}, VQ={avg_vq:.4f}")

    # Save checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = "checkpoints/vqvae_quick.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epochs,
    }, checkpoint_path)
    print(f"Model saved to {checkpoint_path}")

    return model

def train_dynamics_simple(vqvae_model,
                         data_dir="datasets/youtube/frames",
                         epochs=5,
                         batch_size=2,
                         learning_rate=1e-4,
                         device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Train dynamics model using encoded frames"""

    print("Training dynamics model...")

    # First, get the actual encoded dimension
    vqvae_model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 256, 256).to(device)
        dummy_encoded = vqvae_model.encode(dummy_input)
        if isinstance(dummy_encoded, tuple):
            dummy_encoded = dummy_encoded[0]
        actual_latent_dim = dummy_encoded.view(1, -1).shape[1]
        print(f"Actual flattened latent dimension: {actual_latent_dim}")

    # Create dynamics model
    dynamics_model = DynamicsModel(
        latent_dim=actual_latent_dim,  # Use actual flattened dim
        action_dim=8,  # 8 possible actions
        hidden_dim=256,
        num_layers=4
    ).to(device)

    # For simplicity, we'll train with random actions
    # In production, you'd have actual action sequences

    optimizer = optim.Adam(dynamics_model.parameters(), lr=learning_rate)

    # Simple training with random actions
    print(f"Training dynamics for {epochs} epochs...")
    dynamics_model.train()
    vqvae_model.eval()

    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    dataset = SimpleFrameDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0

        progress = tqdm(dataloader, desc=f"Dynamics Epoch {epoch+1}/{epochs}")
        for batch_idx, images in enumerate(progress):
            if images.shape[0] < 2:
                continue  # Skip if batch too small

            images = images.to(device)

            # Encode frames with VQ-VAE
            with torch.no_grad():
                encoded = vqvae_model.encode(images)
                if isinstance(encoded, tuple):
                    encoded = encoded[0]
                # Flatten spatial dimensions for dynamics model
                batch_size = encoded.shape[0]
                encoded = encoded.view(batch_size, -1)  # Flatten to (batch, features)

            # Create random actions
            actions = torch.randint(0, 8, (images.shape[0],)).to(device)

            # Use current frame to predict next (simplified)
            if batch_idx > 0:
                # Add sequence dimension for dynamics model
                encoded_seq = encoded[:-1].unsqueeze(1)  # (batch-1, 1, features)
                actions_seq = actions[:-1].unsqueeze(1)  # (batch-1, 1)

                # Predict next latent
                predicted = dynamics_model(encoded_seq, actions_seq)
                predicted = predicted.squeeze(1)  # Remove sequence dim
                target = encoded[1:]

                loss = nn.functional.mse_loss(predicted, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                progress.set_postfix({'loss': f"{loss.item():.4f}"})

        if batch_idx > 0:
            avg_loss = total_loss / max(1, batch_idx)
            print(f"Dynamics Epoch {epoch+1}: Loss={avg_loss:.4f}")

    # Save dynamics model
    checkpoint_path = "checkpoints/dynamics_quick.pt"
    torch.save({
        'model_state_dict': dynamics_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epochs,
    }, checkpoint_path)
    print(f"Dynamics model saved to {checkpoint_path}")

    return dynamics_model

def create_world_model(vqvae_model, dynamics_model, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Create complete world model"""

    world_model = WorldModel(
        vqvae=vqvae_model,
        dynamics=dynamics_model
    ).to(device)

    # Save complete model
    checkpoint_path = "checkpoints/world_model_final.pt"
    torch.save({
        'vqvae_state': vqvae_model.state_dict(),
        'dynamics_state': dynamics_model.state_dict(),
    }, checkpoint_path)
    print(f"Complete world model saved to {checkpoint_path}")

    return world_model

def main():
    """Main training function"""
    print("=" * 50)
    print("World Model Quick Training")
    print("=" * 50)

    # Check for data
    data_dir = "datasets/youtube/frames"
    if not Path(data_dir).exists():
        print(f"Error: Data directory {data_dir} not found")
        print("Please collect data first using youtube_collector.py")
        return

    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("WARNING: Training on CPU will be slow!")
        print("Consider using a GPU for faster training")

    # Train VQ-VAE
    print("\n" + "="*50)
    print("Stage 1: Training VQ-VAE")
    print("="*50)
    vqvae = train_vqvae_simple(
        data_dir=data_dir,
        epochs=5,  # Quick training for demo
        batch_size=2 if device == 'cpu' else 4,
        device=device
    )

    # Train Dynamics
    print("\n" + "="*50)
    print("Stage 2: Training Dynamics Model")
    print("="*50)
    dynamics = train_dynamics_simple(
        vqvae,
        data_dir=data_dir,
        epochs=3,  # Quick training for demo
        batch_size=2 if device == 'cpu' else 4,
        device=device
    )

    # Create World Model
    print("\n" + "="*50)
    print("Stage 3: Creating World Model")
    print("="*50)
    world_model = create_world_model(vqvae, dynamics, device)

    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    print("\nModels saved to checkpoints/")
    print("- vqvae_quick.pt: VQ-VAE model")
    print("- dynamics_quick.pt: Dynamics model")
    print("- world_model_final.pt: Complete world model")
    print("\nYou can now restart the enhanced server to use the trained model!")

if __name__ == "__main__":
    main()
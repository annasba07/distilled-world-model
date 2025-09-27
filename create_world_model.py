#!/usr/bin/env python3
"""
Create world model checkpoint from trained VQ-VAE
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.models.improved_vqvae import ImprovedVQVAE
from src.models.dynamics import DynamicsModel

def create_minimal_world_model():
    """Create a minimal world model with trained VQ-VAE and simple dynamics"""

    print("Creating world model checkpoint...")

    # Load trained VQ-VAE
    vqvae_checkpoint = torch.load("checkpoints/vqvae_quick.pt", map_location='cpu')

    # Create VQ-VAE model with same config as training
    vqvae = ImprovedVQVAE(
        in_channels=3,
        latent_dim=256,
        num_embeddings=512,
        hidden_dims=[64, 128, 256],
        use_ema=True,
        use_attention=False
    )

    # Load VQ-VAE weights
    vqvae.load_state_dict(vqvae_checkpoint['model_state_dict'])
    print("[OK] VQ-VAE loaded successfully")

    # Create a simple dynamics model (won't be trained, just placeholder)
    dynamics = DynamicsModel(
        latent_dim=256,  # Will work with pooled features
        action_dim=8,
        hidden_dim=256,
        num_layers=2  # Reduced layers for simplicity
    )
    print("[OK] Dynamics model created (untrained placeholder)")

    # Save as world_model_final.pt in the format the server expects
    checkpoint = {
        'vqvae_state': vqvae.state_dict(),
        'dynamics_state': dynamics.state_dict(),
        'config': {
            'vqvae': {
                'in_channels': 3,
                'latent_dim': 256,
                'num_embeddings': 512,
                'hidden_dims': [64, 128, 256],
                'use_ema': True,
                'use_attention': False
            },
            'dynamics': {
                'latent_dim': 256,
                'action_dim': 8,
                'hidden_dim': 256,
                'num_layers': 2
            }
        },
        'training_info': {
            'vqvae_trained': True,
            'dynamics_trained': False,
            'vqvae_loss': 0.84,
            'dataset': 'youtube_frames',
            'num_frames': 60
        }
    }

    save_path = "checkpoints/world_model_final.pt"
    torch.save(checkpoint, save_path)
    print(f"[OK] World model saved to {save_path}")

    # Verify the checkpoint
    test_load = torch.load(save_path, map_location='cpu')
    print(f"[OK] Checkpoint verified - contains {len(test_load['vqvae_state'])} VQ-VAE parameters")

    print("\n" + "="*50)
    print("SUCCESS! World Model Created")
    print("="*50)
    print("\nThe world model checkpoint has been created with:")
    print("- Trained VQ-VAE (can reconstruct images)")
    print("- Placeholder dynamics model (for compatibility)")
    print("\nYou can now restart the enhanced server to use this model!")
    print("\nNote: The model was trained on only 60 frames, so quality will be limited.")
    print("For better results, collect more data using youtube_collector.py")

if __name__ == "__main__":
    create_minimal_world_model()
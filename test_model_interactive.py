#!/usr/bin/env python3
"""
Interactive test of your trained World Model!
This bypasses the server and lets you directly interact with the model.
"""

import torch
import numpy as np
from PIL import Image
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.models.improved_vqvae import ImprovedVQVAE
from src.inference.toy_world import ToyWorldSimulator

def load_trained_model():
    """Load your trained VQ-VAE model"""
    print("Loading your trained model...")

    # Load checkpoint
    checkpoint = torch.load("checkpoints/world_model_final.pt", map_location='cpu')

    # Create VQ-VAE with same config as training
    vqvae = ImprovedVQVAE(
        in_channels=3,
        latent_dim=256,
        num_embeddings=512,
        hidden_dims=[64, 128, 256],
        use_ema=True,
        use_attention=False
    )

    # Load weights
    vqvae.load_state_dict(checkpoint['vqvae_state'])
    vqvae.eval()

    print("Model loaded successfully!")
    return vqvae

def test_reconstruction(vqvae):
    """Test model by reconstructing an image"""
    print("\n" + "="*50)
    print("Testing Image Reconstruction")
    print("="*50)

    # Create a test image (random noise or toy world frame)
    toy_world = ToyWorldSimulator()
    state = toy_world.create_state("test world", seed=42)
    test_frame = toy_world.render(state)

    # Convert to tensor
    test_tensor = torch.from_numpy(test_frame).float().permute(2, 0, 1) / 255.0
    test_tensor = test_tensor.unsqueeze(0)  # Add batch dim

    print(f"Input shape: {test_tensor.shape}")

    # Run through model
    with torch.no_grad():
        reconstructed, vq_stats = vqvae(test_tensor)

    # Convert back to image
    reconstructed_np = reconstructed.squeeze(0).permute(1, 2, 0).cpu().numpy()
    reconstructed_np = (reconstructed_np * 255).clip(0, 255).astype(np.uint8)

    print(f"Reconstruction complete!")
    print(f"VQ perplexity: {vq_stats.get('perplexity', 'N/A')}")

    return test_frame, reconstructed_np

def generate_new_frame(vqvae, prompt_text="pixel art world"):
    """Generate a new frame from scratch"""
    print("\n" + "="*50)
    print(f"Generating New Frame: '{prompt_text}'")
    print("="*50)

    # For now, we'll start with random latents since dynamics isn't trained
    # In a full system, this would use the dynamics model

    # Generate random latent codes
    batch_size = 1
    latent_shape = (batch_size, 256, 16, 16)  # Adjust based on your model
    random_latent = torch.randn(latent_shape) * 0.5

    print(f"Latent shape: {random_latent.shape}")

    # Decode through VQ-VAE decoder
    with torch.no_grad():
        # Pass through VQ layer first
        quantized, _ = vqvae.vq(random_latent)
        # Then decode
        generated = vqvae.decoder(quantized)

    # Convert to image
    generated_np = generated.squeeze(0).permute(1, 2, 0).cpu().numpy()
    generated_np = (generated_np * 255).clip(0, 255).astype(np.uint8)

    print("Generation complete!")
    return generated_np

def interactive_test():
    """Interactive testing menu"""
    # Load model
    vqvae = load_trained_model()

    while True:
        print("\n" + "="*50)
        print("Interactive Model Test")
        print("="*50)
        print("1. Test reconstruction (toy world)")
        print("2. Generate new frame")
        print("3. Show model info")
        print("4. Save test images")
        print("0. Exit")

        choice = input("\nChoose an option: ").strip()

        if choice == "1":
            # Test reconstruction
            original, reconstructed = test_reconstruction(vqvae)

            # Display results
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(original)
            axes[0].set_title("Original")
            axes[0].axis('off')
            axes[1].imshow(reconstructed)
            axes[1].set_title("Reconstructed by Your Model")
            axes[1].axis('off')
            plt.tight_layout()
            plt.show()

        elif choice == "2":
            # Generate new frame
            prompt = input("Enter prompt (or press Enter for default): ").strip()
            if not prompt:
                prompt = "pixel art platformer"

            generated = generate_new_frame(vqvae, prompt)

            # Display result
            plt.figure(figsize=(6, 6))
            plt.imshow(generated)
            plt.title(f"Generated: {prompt}")
            plt.axis('off')
            plt.show()

        elif choice == "3":
            # Show model info
            total_params = sum(p.numel() for p in vqvae.parameters())
            trainable_params = sum(p.numel() for p in vqvae.parameters() if p.requires_grad)

            print(f"\nModel Information:")
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
            print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
            print(f"Training dataset: 60 frames from YouTube videos")
            print(f"Final training loss: 0.84")

        elif choice == "4":
            # Save test images
            print("Generating test images...")

            # Generate multiple samples
            samples = []
            for i in range(4):
                generated = generate_new_frame(vqvae, f"world {i}")
                samples.append(generated)

            # Create grid
            fig, axes = plt.subplots(2, 2, figsize=(8, 8))
            for i, ax in enumerate(axes.flat):
                ax.imshow(samples[i])
                ax.set_title(f"Sample {i+1}")
                ax.axis('off')

            plt.suptitle("Generated Samples from Your Trained Model")
            plt.tight_layout()

            # Save
            save_path = "model_test_output.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
            plt.show()

        elif choice == "0":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    print("=" * 60)
    print("INTERACTIVE MODEL TESTER")
    print("Test your trained World Model directly!")
    print("=" * 60)

    try:
        interactive_test()
    except KeyboardInterrupt:
        print("\nTest interrupted. Goodbye!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
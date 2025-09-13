"""
Demo for Milestone 1: Static World Reconstruction
Shows that we can compress and reconstruct game images with PSNR > 30dB
"""

import argparse
import time
from pathlib import Path
from typing import Optional, Tuple, List

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

from .models.improved_vqvae import ImprovedVQVAE, calculate_psnr


class ReconstructionDemo:
    def __init__(self, checkpoint_path: Optional[str] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(checkpoint_path)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.denormalize = transforms.Compose([
            transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])
        ])

    def load_model(self, checkpoint_path: Optional[str]) -> ImprovedVQVAE:
        model = ImprovedVQVAE(
            in_channels=3,
            latent_dim=256,
            num_embeddings=512,
            hidden_dims=[64, 128, 256, 512],
            use_ema=True,
            use_attention=True
        ).to(self.device)
        if checkpoint_path and Path(checkpoint_path).exists():
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint, strict=False)
            print(f"Loaded model from epoch {checkpoint.get('epoch','?')} with PSNR: {checkpoint.get('best_val_psnr', 0):.2f} dB")
        else:
            print("No checkpoint found, using random initialization")
        return model

    @torch.no_grad()
    def reconstruct_image(self, image: Image.Image) -> Tuple[Image.Image, dict]:
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        start_time = time.time()
        quantized, vq_dict = self.model.encode(img_tensor)
        reconstructed = self.model.decode(quantized)
        inference_time = (time.time() - start_time) * 1000
        psnr = calculate_psnr(img_tensor, reconstructed)
        original_size = img_tensor.numel() * 4
        latent_size = quantized.numel() * 4
        compression_ratio = original_size / latent_size
        reconstructed = self.denormalize(reconstructed.squeeze(0)).clamp(0, 1)
        reconstructed_pil = transforms.ToPILImage()(reconstructed.cpu())
        metrics = {
            'psnr': psnr.item(),
            'compression_ratio': compression_ratio,
            'inference_time': inference_time,
            'perplexity': vq_dict['perplexity'].item(),
            'latent_shape': list(quantized.shape),
            'gpu_memory': torch.cuda.max_memory_allocated() / 1024**2 if self.device.type == 'cuda' else 0
        }
        return reconstructed_pil, metrics

    def create_comparison_plot(self, original: Image.Image, reconstructed: Image.Image, metrics: dict) -> Image.Image:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(original)
        axes[0].set_title('Original', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        axes[1].imshow(reconstructed)
        axes[1].set_title('Reconstructed', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        orig_array = np.array(original.resize((256, 256)))
        recon_array = np.array(reconstructed)
        diff = np.abs(orig_array.astype(float) - recon_array.astype(float)).mean(axis=2)
        im = axes[2].imshow(diff, cmap='hot', vmin=0, vmax=50)
        axes[2].set_title('Difference Map', fontsize=14, fontweight='bold')
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2], fraction=0.046)
        fig.text(0.02, 0.5, f"PSNR: {metrics['psnr']:.2f} dB\nCompression: {metrics['compression_ratio']:.1f}x\nInference: {metrics['inference_time']:.1f} ms\nPerplexity: {metrics['perplexity']:.1f}", fontsize=12, va='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        plt.suptitle('VQ-VAE Reconstruction Demo - Milestone 1', fontsize=16, fontweight='bold')
        plt.tight_layout()
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        return Image.fromarray(img_array)

    def generate_sample_images(self) -> List[Image.Image]:
        samples = []
        for i in range(4):
            img = Image.new('RGB', (256, 256))
            pixels = img.load()
            for y in range(128):
                color = int(135 + y * 0.5)
                for x in range(256):
                    pixels[x, y] = (color, color + 20, 255)
            for y in range(128, 256):
                for x in range(256):
                    pixels[x, y] = (34, 139, 34)
            platform_y = 140 + i * 20
            for x in range(50 + i * 30, 150 + i * 30):
                y = platform_y
                if 0 <= x < 256 and 0 <= y+9 < 256:
                    for yy in range(y, y + 10):
                        pixels[x, yy] = (139, 69, 19)
            char_x = 100 + i * 20
            char_y = 120
            for x in range(char_x, min(char_x + 16, 256)):
                for y in range(char_y, char_y + 16):
                    pixels[x, y] = (255, 0, 0)
            samples.append(img)
        return samples

    def run_cli_demo(self, image_path: str):
        print("\n" + "=" * 60)
        print("VQ-VAE Reconstruction Demo - Milestone 1")
        print("=" * 60)
        if Path(image_path).exists():
            image = Image.open(image_path).convert('RGB')
        else:
            print(f"Image not found at {image_path}, generating sample...")
            image = self.generate_sample_images()[0]
        print(f"\nProcessing image: {image_path}")
        print(f"   Original size: {image.size}")
        reconstructed, metrics = self.reconstruct_image(image)
        print("\nReconstruction Metrics:")
        print(f"   PSNR: {metrics['psnr']:.2f} dB {'OK' if metrics['psnr'] > 30 else 'LOW'}")
        print(f"   Compression Ratio: {metrics['compression_ratio']:.1f}x")
        print(f"   Inference Time: {metrics['inference_time']:.1f} ms")
        print(f"   Codebook Perplexity: {metrics['perplexity']:.1f}")
        print(f"   Latent Shape: {metrics['latent_shape']}")
        if self.device.type == 'cuda':
            print(f"   GPU Memory: {metrics['gpu_memory']:.1f} MB")
        output_dir = Path("outputs/reconstruction")
        output_dir.mkdir(parents=True, exist_ok=True)
        recon_path = output_dir / "reconstructed.png"
        reconstructed.save(recon_path)
        print(f"\nSaved reconstructed image to {recon_path}")
        comparison = self.create_comparison_plot(image, reconstructed, metrics)
        comp_path = output_dir / "comparison.png"
        comparison.save(comp_path)
        print(f"Saved comparison to {comp_path}")
        print("\n" + "=" * 60)
        if metrics['psnr'] > 30:
            print("MILESTONE 1 ACHIEVED!")
            print("Static World Reconstruction capability unlocked!")
            print("   Next: Temporal Prediction (Milestone 2)")
        else:
            print("Milestone 1 not yet achieved")
            print(f"   Current PSNR: {metrics['psnr']:.2f} dB")
            print("   Target PSNR: > 30.0 dB")
            print(f"   Gap: {30.0 - metrics['psnr']:.2f} dB")
        print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='VQ-VAE Reconstruction Demo')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/vqvae/best.pt', help='Path to model checkpoint')
    parser.add_argument('--image', type=str, default=None, help='Path to image for CLI demo')
    parser.add_argument('--mode', type=str, default='gradio', choices=['gradio', 'cli'], help='Demo mode')
    args = parser.parse_args()
    demo = ReconstructionDemo(args.checkpoint)
    if args.mode == 'gradio':
        # Deferred import to avoid hard dependency
        try:
            import gradio as gr
        except Exception:
            print("Gradio not installed. Run: pip install gradio")
            return
        def process_image(image):
            if image is None:
                return None, "Please upload an image"
            reconstructed, metrics = demo.reconstruct_image(image)
            comparison = demo.create_comparison_plot(image, reconstructed, metrics)
            metrics_text = (
                f"PSNR: {metrics['psnr']:.2f} dB\n"
                f"Compression Ratio: {metrics['compression_ratio']:.1f}x\n"
                f"Inference Time: {metrics['inference_time']:.1f} ms\n"
                f"Codebook Perplexity: {metrics['perplexity']:.1f}\n"
                f"Latent Shape: {metrics['latent_shape']}"
            )
            return comparison, metrics_text
        with gr.Blocks() as app:
            gr.Markdown("# VQ-VAE Reconstruction Demo - Milestone 1")
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(label="Upload Image", type="pil")
                    process_btn = gr.Button("Reconstruct", variant="primary")
                with gr.Column():
                    output_comparison = gr.Image(label="Reconstruction Result")
                    metrics_display = gr.Markdown()
            process_btn.click(process_image, inputs=[input_image], outputs=[output_comparison, metrics_display])
        app.launch(share=True)
    else:
        if args.image:
            demo.run_cli_demo(args.image)
        else:
            print("No image specified, testing with generated sample...")
            Path("outputs").mkdir(exist_ok=True)
            sample_path = "outputs/sample_game.png"
            demo.generate_sample_images()[0].save(sample_path)
            demo.run_cli_demo(sample_path)


if __name__ == "__main__":
    main()

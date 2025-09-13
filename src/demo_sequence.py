#!/usr/bin/env python3
"""
Demo Interface for Milestone 3: Short Sequence Generation
Interactive demo for generating coherent 30-frame sequences
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import io
import base64
from typing import List, Dict, Tuple, Optional
import time
import argparse

# Gradio for web interface (optional)
try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    print("WARNING: Gradio not available. Install with: pip install gradio")
    GRADIO_AVAILABLE = False

# Import our models
from models.sequence_generator import SequenceGenerator
from data.sequence_dataset import LongSequenceDataset


class SequenceGenerationDemo:
    """Interactive demo for sequence generation (Milestone 3)"""
    
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = 'auto',
        image_size: Tuple[int, int] = (256, 256)
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() and device != 'cpu' else 'cpu')
        self.image_size = image_size
        
        # Initialize model
        self.model = SequenceGenerator(
            in_channels=3,
            latent_dim=256,
            num_embeddings=512,
            d_model=512,
            num_layers=8,
            num_heads=8,
            max_sequence_length=64,
            freeze_vqvae=True
        ).to(self.device)
        
        # Load checkpoint if provided
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        else:
            print("⚠️  No checkpoint loaded - using untrained model")
        
        # Create sample generator for initial frames
        self.sample_generator = LongSequenceDataset(
            sequence_length=1,  # Just need initial frames
            num_sequences=100,
            image_size=image_size
        )
        
        print(f"🎬 SequenceGenerationDemo initialized")
        print(f"   Device: {self.device}")
        print(f"   Image size: {image_size}")
        print(f"   Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"📂 Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
            else:
                self.model.load_state_dict(checkpoint)
                print(f"📂 Loaded model state dict")
            
            self.model.eval()
            
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
    
    def generate_initial_frame(self, scene_type: str = 'random') -> Image.Image:
        """Generate or select an initial frame for sequence generation"""
        
        if scene_type == 'random':
            # Get random initial frame from dataset
            idx = np.random.randint(0, len(self.sample_generator))
            frames, metadata = self.sample_generator[idx]
            initial_frame = frames[0]  # First frame of sequence
            
            # Convert tensor to PIL Image
            frame_np = (initial_frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            return Image.fromarray(frame_np)
        
        elif scene_type == 'custom':
            # Create a custom scene
            img = Image.new('RGB', self.image_size, (40, 60, 80))
            draw = ImageDraw.Draw(img)
            
            # Add some objects
            # Character
            char_x, char_y = self.image_size[0] // 4, self.image_size[1] // 2
            draw.ellipse([char_x-15, char_y-15, char_x+15, char_y+15], fill=(255, 100, 100))
            
            # Platform
            platform_y = self.image_size[1] - 50
            draw.rectangle([50, platform_y, self.image_size[0]-50, platform_y+20], fill=(100, 255, 100))
            
            # Decorative elements
            draw.ellipse([200, 100, 220, 120], fill=(255, 255, 100))  # Sun
            
            return img
        
        elif scene_type == 'bouncing_ball':
            # Create a bouncing ball scene
            img = Image.new('RGB', self.image_size, (30, 30, 50))
            draw = ImageDraw.Draw(img)
            
            # Ball
            ball_x, ball_y = self.image_size[0] // 2, 50
            draw.ellipse([ball_x-20, ball_y-20, ball_x+20, ball_y+20], fill=(255, 150, 150))
            
            # Ground
            ground_y = self.image_size[1] - 30
            draw.rectangle([0, ground_y, self.image_size[0], self.image_size[1]], fill=(100, 80, 60))
            
            return img
        
        else:
            # Simple test scene
            img = Image.new('RGB', self.image_size, (50, 50, 50))
            draw = ImageDraw.Draw(img)
            
            # Central object
            center_x, center_y = self.image_size[0] // 2, self.image_size[1] // 2
            draw.rectangle([center_x-30, center_y-30, center_x+30, center_y+30], fill=(200, 100, 100))
            
            return img
    
    def generate_diverse_sequences(
        self,
        initial_frame: Image.Image,
        num_sequences: int = 4,
        sequence_length: int = 30,
        temperature: float = 1.0
    ) -> Tuple[List[List[Image.Image]], Dict]:
        """Generate multiple diverse sequences from initial frame"""
        
        print(f"🎬 Generating {num_sequences} sequences of length {sequence_length}")
        
        # Convert PIL to tensor
        frame_np = np.array(initial_frame).astype(np.float32) / 255.0
        frame_tensor = torch.from_numpy(frame_np).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(self.device)
        
        start_time = time.time()
        
        try:
            with torch.no_grad():
                # Generate sequences
                generated_sequences = self.model.generate_diverse_sequences(
                    frame_tensor,
                    num_sequences=num_sequences,
                    sequence_length=sequence_length,
                    temperature=temperature,
                    diversity_boost=0.1
                )
            
            generation_time = time.time() - start_time
            
            # Convert back to PIL Images
            pil_sequences = []
            for seq_tensor in generated_sequences:
                pil_frames = []
                for frame_idx in range(seq_tensor.size(1)):
                    frame = seq_tensor[0, frame_idx]  # Remove batch dimension
                    frame_np = (frame.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                    pil_frames.append(Image.fromarray(frame_np))
                pil_sequences.append(pil_frames)
            
            # Calculate metrics
            metrics = {
                'generation_time': generation_time,
                'sequences_generated': len(pil_sequences),
                'sequence_length': sequence_length,
                'fps': sequence_length / generation_time if generation_time > 0 else 0,
                'temperature': temperature
            }
            
            print(f"   ✅ Generated in {generation_time:.2f}s ({metrics['fps']:.1f} FPS)")
            
            return pil_sequences, metrics
            
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            return [], {'error': str(e)}
    
    def create_sequence_comparison(
        self,
        sequences: List[List[Image.Image]],
        max_frames_display: int = 10
    ) -> Image.Image:
        """Create a visual comparison of multiple sequences"""
        
        if not sequences:
            # Create error image
            error_img = Image.new('RGB', (400, 200), (50, 50, 50))
            draw = ImageDraw.Draw(error_img)
            draw.text((50, 90), "No sequences generated", fill=(255, 255, 255))
            return error_img
        
        num_sequences = len(sequences)
        frames_to_show = min(max_frames_display, len(sequences[0]))
        
        # Calculate grid dimensions
        frame_width, frame_height = sequences[0][0].size
        thumbnail_size = 64
        
        grid_width = frames_to_show * thumbnail_size
        grid_height = num_sequences * thumbnail_size
        
        # Create comparison image
        comparison_img = Image.new('RGB', (grid_width + 50, grid_height + 50), (30, 30, 30))
        
        # Add sequence thumbnails
        for seq_idx, sequence in enumerate(sequences):
            for frame_idx in range(frames_to_show):
                if frame_idx < len(sequence):
                    # Resize frame to thumbnail
                    thumbnail = sequence[frame_idx].resize((thumbnail_size, thumbnail_size))
                    
                    # Paste into grid
                    x = frame_idx * thumbnail_size + 25
                    y = seq_idx * thumbnail_size + 25
                    comparison_img.paste(thumbnail, (x, y))
        
        # Add labels
        draw = ImageDraw.Draw(comparison_img)
        
        # Sequence labels
        for seq_idx in range(num_sequences):
            y = seq_idx * thumbnail_size + 35
            draw.text((5, y), f"S{seq_idx+1}", fill=(255, 255, 255))
        
        # Frame labels
        for frame_idx in range(min(5, frames_to_show)):  # Only label first 5 frames
            x = frame_idx * thumbnail_size + 35
            draw.text((x, 5), f"{frame_idx}", fill=(255, 255, 255))
        
        return comparison_img
    
    def create_sequence_video(
        self,
        sequence: List[Image.Image],
        output_path: str = "generated_sequence.gif",
        fps: int = 10
    ) -> str:
        """Create animated GIF from sequence"""
        
        try:
            # Save as animated GIF
            if sequence:
                sequence[0].save(
                    output_path,
                    save_all=True,
                    append_images=sequence[1:],
                    duration=int(1000 / fps),  # Duration in milliseconds
                    loop=0
                )
                print(f"💾 Saved animation to {output_path}")
                return output_path
            else:
                return None
                
        except Exception as e:
            print(f"❌ Failed to create animation: {e}")
            return None
    
    def run_cli_demo(self):
        """Run command-line interactive demo"""
        print("\n" + "="*60)
        print("🎬 MILESTONE 3 DEMO: Short Sequence Generation")
        print("="*60)
        
        while True:
            print("\nOptions:")
            print("1. Generate from random initial frame")
            print("2. Generate from custom scene")
            print("3. Generate bouncing ball sequence")
            print("4. Generate comparison grid")
            print("5. Exit")
            
            choice = input("\nEnter choice (1-5): ").strip()
            
            if choice == '5':
                print("👋 Goodbye!")
                break
            
            elif choice in ['1', '2', '3']:
                # Get parameters
                try:
                    num_sequences = int(input("Number of sequences (1-8, default=4): ") or "4")
                    num_sequences = max(1, min(8, num_sequences))
                    
                    sequence_length = int(input("Sequence length (5-50, default=20): ") or "20")
                    sequence_length = max(5, min(50, sequence_length))
                    
                    temperature = float(input("Temperature (0.1-2.0, default=1.0): ") or "1.0")
                    temperature = max(0.1, min(2.0, temperature))
                    
                except ValueError:
                    print("❌ Invalid input, using defaults")
                    num_sequences, sequence_length, temperature = 4, 20, 1.0
                
                # Generate initial frame
                scene_types = {'1': 'random', '2': 'custom', '3': 'bouncing_ball'}
                scene_type = scene_types[choice]
                
                print(f"\n🎬 Generating {num_sequences} sequences...")
                initial_frame = self.generate_initial_frame(scene_type)
                
                # Generate sequences
                sequences, metrics = self.generate_diverse_sequences(
                    initial_frame,
                    num_sequences=num_sequences,
                    sequence_length=sequence_length,
                    temperature=temperature
                )
                
                if sequences:
                    print(f"   ✅ Success! Generated {len(sequences)} sequences")
                    print(f"   📊 Time: {metrics['generation_time']:.2f}s")
                    print(f"   📊 Speed: {metrics['fps']:.1f} frames/second")
                    
                    # Save first sequence as GIF
                    gif_path = f"sequence_{int(time.time())}.gif"
                    self.create_sequence_video(sequences[0], gif_path, fps=8)
                    
                    # Create comparison
                    comparison = self.create_sequence_comparison(sequences)
                    comparison_path = f"comparison_{int(time.time())}.png"
                    comparison.save(comparison_path)
                    print(f"   💾 Saved comparison to {comparison_path}")
                    
                else:
                    print("   ❌ Generation failed")
            
            elif choice == '4':
                # Generate comparison with different settings
                print("\n🎬 Generating comparison grid...")
                
                initial_frame = self.generate_initial_frame('bouncing_ball')
                
                # Generate with different temperatures
                all_sequences = []
                temperatures = [0.5, 1.0, 1.5]
                
                for temp in temperatures:
                    sequences, _ = self.generate_diverse_sequences(
                        initial_frame,
                        num_sequences=2,
                        sequence_length=15,
                        temperature=temp
                    )
                    all_sequences.extend(sequences)
                
                if all_sequences:
                    comparison = self.create_sequence_comparison(all_sequences, max_frames_display=8)
                    comparison_path = f"temperature_comparison_{int(time.time())}.png"
                    comparison.save(comparison_path)
                    print(f"   💾 Saved temperature comparison to {comparison_path}")
                else:
                    print("   ❌ Comparison generation failed")
            
            else:
                print("❌ Invalid choice")
    
    def create_gradio_interface(self):
        """Create Gradio web interface"""
        
        if not GRADIO_AVAILABLE:
            print("❌ Gradio not available. Run CLI demo instead.")
            return None
        
        def gradio_generate(
            scene_type: str,
            num_sequences: int,
            sequence_length: int,
            temperature: float
        ):
            """Gradio wrapper for sequence generation"""
            
            # Generate initial frame
            initial_frame = self.generate_initial_frame(scene_type.lower().replace(' ', '_'))
            
            # Generate sequences
            sequences, metrics = self.generate_diverse_sequences(
                initial_frame,
                num_sequences=num_sequences,
                sequence_length=sequence_length,
                temperature=temperature
            )
            
            if not sequences:
                return None, "Generation failed", {}
            
            # Create comparison image
            comparison = self.create_sequence_comparison(sequences)
            
            # Create animation for first sequence
            gif_path = f"temp_sequence_{int(time.time())}.gif"
            self.create_sequence_video(sequences[0], gif_path, fps=8)
            
            # Metrics string
            metrics_str = f"""
            Generation Time: {metrics.get('generation_time', 0):.2f}s
            Sequences: {metrics.get('sequences_generated', 0)}
            Length: {metrics.get('sequence_length', 0)} frames
            Speed: {metrics.get('fps', 0):.1f} frames/sec
            Temperature: {metrics.get('temperature', 0):.1f}
            """
            
            return comparison, gif_path, metrics_str
        
        # Create Gradio interface
        interface = gr.Interface(
            fn=gradio_generate,
            inputs=[
                gr.Dropdown(
                    choices=["Random", "Custom", "Bouncing Ball", "Simple"],
                    value="Random",
                    label="Scene Type"
                ),
                gr.Slider(1, 6, value=4, step=1, label="Number of Sequences"),
                gr.Slider(10, 40, value=20, step=5, label="Sequence Length"),
                gr.Slider(0.1, 2.0, value=1.0, step=0.1, label="Temperature")
            ],
            outputs=[
                gr.Image(label="Sequence Comparison"),
                gr.File(label="Animation (GIF)"),
                gr.Textbox(label="Generation Metrics")
            ],
            title="🎬 Milestone 3: Short Sequence Generation",
            description="Generate coherent 30-frame sequences from a single initial frame. Adjust parameters to explore different generation styles.",
            examples=[
                ["Random", 4, 20, 1.0],
                ["Bouncing Ball", 3, 25, 0.8],
                ["Custom", 2, 15, 1.5]
            ]
        )
        
        return interface


def test_sequence_demo():
    """Test the sequence generation demo"""
    print("🧪 Testing Sequence Generation Demo")
    
    demo = SequenceGenerationDemo()
    
    # Test initial frame generation
    initial_frame = demo.generate_initial_frame('bouncing_ball')
    print(f"   ✅ Generated initial frame: {initial_frame.size}")
    
    # Test sequence generation
    sequences, metrics = demo.generate_diverse_sequences(
        initial_frame,
        num_sequences=2,
        sequence_length=8,
        temperature=1.0
    )
    
    if sequences:
        print(f"   ✅ Generated {len(sequences)} sequences")
        print(f"   📊 First sequence: {len(sequences[0])} frames")
        print(f"   📊 Generation time: {metrics['generation_time']:.2f}s")
        
        # Test comparison creation
        comparison = demo.create_sequence_comparison(sequences)
        print(f"   ✅ Created comparison: {comparison.size}")
        
        # Test video creation
        gif_path = demo.create_sequence_video(sequences[0], "test_sequence.gif", fps=8)
        if gif_path:
            print(f"   ✅ Created animation: {gif_path}")
        
        print("   ✅ Demo test completed!")
    else:
        print("   ❌ Sequence generation failed")


def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description='Sequence Generation Demo (Milestone 3)')
    parser.add_argument('--checkpoint', type=str, default=None, help='Model checkpoint path')
    parser.add_argument('--mode', type=str, choices=['cli', 'web', 'test'], default='cli', help='Demo mode')
    parser.add_argument('--port', type=int, default=7860, help='Web interface port')
    parser.add_argument('--image_size', type=int, default=256, help='Image size')
    
    args = parser.parse_args()
    
    if args.mode == 'test':
        test_sequence_demo()
        return
    
    # Create demo
    demo = SequenceGenerationDemo(
        checkpoint_path=args.checkpoint,
        device='auto',
        image_size=(args.image_size, args.image_size)
    )
    
    if args.mode == 'cli':
        demo.run_cli_demo()
    elif args.mode == 'web':
        interface = demo.create_gradio_interface()
        if interface:
            interface.launch(server_port=args.port, share=True)
        else:
            print("❌ Web interface not available. Running CLI demo...")
            demo.run_cli_demo()


if __name__ == "__main__":
    main()

"""
Sequence Generator for Milestone 3: Short Sequence Generation
Extends temporal prediction to generate coherent 30-frame sequences
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import math
from einops import rearrange, repeat

from .temporal_predictor import WorldModelWithPrediction, TemporalDynamicsModel
from .improved_vqvae import ImprovedVQVAE


class EnhancedTemporalModel(TemporalDynamicsModel):
    """Enhanced temporal model with improved long-range coherence"""
    
    def __init__(
        self,
        latent_dim: int = 256,
        d_model: int = 512,
        num_layers: int = 8,  # More layers for better long-range modeling
        num_heads: int = 8,
        max_sequence_length: int = 64,  # Longer sequences
        dropout: float = 0.1,
        use_positional_bias: bool = True,
        use_memory_bank: bool = True
    ):
        super().__init__(
            latent_dim=latent_dim,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            max_sequence_length=max_sequence_length,
            dropout=dropout
        )
        
        self.use_positional_bias = use_positional_bias
        self.use_memory_bank = use_memory_bank
        
        # Enhanced positional encoding with learned bias
        if use_positional_bias:
            self.pos_bias = nn.Parameter(torch.zeros(1, max_sequence_length, d_model))
            nn.init.normal_(self.pos_bias, std=0.02)
        
        # Memory bank for long-range consistency
        if use_memory_bank:
            self.memory_bank_size = 16
            self.memory_proj = nn.Linear(d_model, d_model)
            self.memory_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
    
    def forward(self, latent_sequence: torch.Tensor, memory_bank: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Enhanced forward pass with memory bank and positional bias
        
        Args:
            latent_sequence: (batch, sequence_length, latent_dim)
            memory_bank: Optional memory from previous sequences
            
        Returns:
            next_frame_prediction: (batch, sequence_length, latent_dim)
        """
        batch_size, seq_len, _ = latent_sequence.shape
        
        # Project to model dimension
        x = self.latent_proj(latent_sequence)
        
        # Add positional encoding and bias
        x = self.pos_encoding(x)
        if self.use_positional_bias:
            x = x + self.pos_bias[:, :seq_len]
        
        # Memory bank attention (if enabled)
        if self.use_memory_bank and memory_bank is not None:
            memory_features = self.memory_proj(memory_bank)
            x_for_memory = x.transpose(0, 1)  # (seq, batch, dim)
            memory_features = memory_features.transpose(0, 1)  # (mem_len, batch, dim)
            
            # Cross-attention to memory
            attended_x, _ = self.memory_attention(x_for_memory, memory_features, memory_features)
            x = x + attended_x.transpose(0, 1)  # Residual connection
        
        # Create causal mask
        mask = self.create_causal_mask(seq_len, x.device)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, mask)
        
        # Final normalization
        x = self.norm(x)
        
        # Project back to latent space
        output = self.output_proj(x)
        
        return output
    
    def generate_long_sequence(
        self, 
        initial_frames: torch.Tensor,
        target_length: int = 30,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate long coherent sequences in continuous latent space
        
        Args:
            initial_frames: Starting frames (batch, init_len, latent_dim)
            target_length: Total sequence length to generate
            temperature: Noise level for diversity (higher = more random)
            
        Returns:
            generated_sequence: Full generated sequence
            memory_states: Memory bank states for each step
        """
        self.eval()
        device = initial_frames.device
        batch_size = initial_frames.size(0)
        
        # Initialize generation
        current_sequence = initial_frames.clone()
        memory_bank = None
        memory_states = []
        
        with torch.no_grad():
            for step in range(target_length - initial_frames.size(1)):
                # Keep only recent frames for efficiency
                if current_sequence.size(1) > self.max_sequence_length:
                    # Update memory bank with older frames
                    if self.use_memory_bank:
                        old_frames = current_sequence[:, :-self.max_sequence_length//2]
                        old_features = self.latent_proj(old_frames.mean(dim=1, keepdim=True))
                        
                        if memory_bank is None:
                            memory_bank = old_features
                        else:
                            memory_bank = torch.cat([memory_bank, old_features], dim=1)
                            if memory_bank.size(1) > self.memory_bank_size:
                                memory_bank = memory_bank[:, -self.memory_bank_size:]
                    
                    # Keep recent frames
                    current_sequence = current_sequence[:, -self.max_sequence_length//2:]
                
                # Predict next frame
                predictions = self.forward(current_sequence, memory_bank)
                next_frame = predictions[:, -1:, :]  # Last prediction
                
                # Add noise for diversity
                if temperature > 0:
                    noise = torch.randn_like(next_frame) * (0.01 * temperature)
                    next_frame = next_frame + noise
                
                # Append to sequence
                current_sequence = torch.cat([current_sequence, next_frame], dim=1)
                
                # Store memory state
                memory_states.append(memory_bank.clone() if memory_bank is not None else None)
        
        return current_sequence, memory_states


class SequenceGenerator(nn.Module):
    """Complete sequence generation system for Milestone 3"""
    
    def __init__(
        self,
        # VQ-VAE parameters
        in_channels: int = 3,
        latent_dim: int = 256,
        num_embeddings: int = 512,
        vqvae_hidden_dims: List[int] = None,
        
        # Enhanced temporal model parameters
        d_model: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        max_sequence_length: int = 64,
        
        # Generation parameters
        use_pretrained_vqvae: bool = True,
        freeze_vqvae: bool = True
    ):
        super().__init__()
        
        if vqvae_hidden_dims is None:
            vqvae_hidden_dims = [64, 128, 256, 512]
        
        # VQ-VAE for spatial encoding/decoding (same as before)
        self.vqvae = ImprovedVQVAE(
            in_channels=in_channels,
            latent_dim=latent_dim,
            num_embeddings=num_embeddings,
            hidden_dims=vqvae_hidden_dims,
            use_ema=True,
            use_attention=True
        )
        
        # Freeze VQ-VAE if using pretrained
        if freeze_vqvae:
            for param in self.vqvae.parameters():
                param.requires_grad = False
        
        # Get latent spatial dimensions
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, 256, 256)
            dummy_latent = self.vqvae.encoder(dummy_input)
            self.latent_spatial_shape = dummy_latent.shape[2:]  # (H, W)
            self.spatial_latent_dim = latent_dim * np.prod(self.latent_spatial_shape)
        
        # Enhanced temporal model for sequence generation
        self.sequence_model = EnhancedTemporalModel(
            latent_dim=self.spatial_latent_dim,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            max_sequence_length=max_sequence_length,
            use_positional_bias=True,
            use_memory_bank=True
        )
        
        # Consistency regularization
        self.consistency_weight = 0.1
        
    def encode_frame_sequence(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode a sequence of frames to latent space"""
        batch_size, seq_len, channels, height, width = frames.shape
        
        # Flatten sequence dimension
        frames_flat = rearrange(frames, 'b t c h w -> (b t) c h w')
        
        # Encode each frame
        with torch.no_grad():
            latents, _ = self.vqvae.encode(frames_flat)
        
        # Reshape back to sequence
        latents = rearrange(latents, '(b t) c h w -> b t (c h w)', b=batch_size)
        
        return latents
    
    def decode_latent_sequence(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode a sequence of latents back to frames"""
        batch_size, seq_len, flat_latent_dim = latents.shape
        
        # Reshape to spatial latents
        latent_shape = (self.vqvae.encoder.conv_out.out_channels,) + self.latent_spatial_shape
        latents_spatial = latents.reshape(batch_size * seq_len, *latent_shape)
        
        # Decode each frame
        with torch.no_grad():
            frames = self.vqvae.decode(latents_spatial)
        
        # Reshape back to sequence
        frames = rearrange(frames, '(b t) c h w -> b t c h w', b=batch_size)
        
        return frames
    
    @torch.no_grad()
    def generate_diverse_sequences(
        self, 
        initial_frame: torch.Tensor,
        num_sequences: int = 4,
        sequence_length: int = 30,
        temperature: float = 1.0,
        diversity_boost: float = 0.1
    ) -> List[torch.Tensor]:
        """
        Generate multiple diverse sequences from a single initial frame
        
        Args:
            initial_frame: Starting frame (batch, 1, channels, height, width)
            num_sequences: Number of different sequences to generate
            sequence_length: Length of each sequence
            temperature: Sampling temperature
            diversity_boost: Additional randomness for diversity
            
        Returns:
            List of generated sequences (each is batch, seq_len, channels, height, width)
        """
        # Encode initial frame
        initial_latent = self.encode_frame_sequence(initial_frame)
        
        generated_sequences = []
        
        for seq_idx in range(num_sequences):
            # Add slight variation to initial latent for diversity
            varied_initial = initial_latent + torch.randn_like(initial_latent) * diversity_boost
            
            # Use different temperature for each sequence
            seq_temperature = temperature * (0.8 + 0.4 * seq_idx / max(1, num_sequences - 1))
            
            # Generate sequence
            latent_sequence, _ = self.sequence_model.generate_long_sequence(
                varied_initial,
                target_length=sequence_length,
                temperature=seq_temperature
            )
            
            # Decode to frames
            frame_sequence = self.decode_latent_sequence(latent_sequence)
            generated_sequences.append(frame_sequence)
        
        return generated_sequences
    
    def forward(self, frame_sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Forward pass for training
        
        Returns:
            predicted_latents: Predicted latent sequence
            target_latents: Ground truth latent sequence  
            additional_losses: Dictionary of additional loss components
        """
        # Encode frames to latent space
        latents = self.encode_frame_sequence(frame_sequence)
        
        # Predict sequence
        predicted_latents = self.sequence_model(latents[:, :-1])
        target_latents = latents[:, 1:]
        
        # Calculate additional losses for coherence
        additional_losses = {}
        
        # Temporal consistency loss
        if predicted_latents.size(1) > 1:
            pred_diff = torch.diff(predicted_latents, dim=1)
            target_diff = torch.diff(target_latents, dim=1)
            additional_losses['temporal_consistency'] = F.mse_loss(pred_diff, target_diff)
        
        # Smoothness regularization
        if predicted_latents.size(1) > 2:
            pred_second_diff = torch.diff(pred_diff, dim=1)
            target_second_diff = torch.diff(target_diff, dim=1)
            additional_losses['smoothness'] = F.mse_loss(pred_second_diff, target_second_diff) * 0.1
        
        return predicted_latents, target_latents, additional_losses
    
    def load_vqvae_checkpoint(self, checkpoint_path: str):
        """Load pretrained VQ-VAE weights"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            self.vqvae.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.vqvae.load_state_dict(checkpoint)
        
        print(f"✅ Loaded VQ-VAE checkpoint from {checkpoint_path}")


def calculate_sequence_metrics(predicted_sequence: torch.Tensor, target_sequence: torch.Tensor) -> Dict[str, float]:
    """Calculate comprehensive metrics for sequence generation quality"""
    
    # Basic reconstruction metrics
    mse = F.mse_loss(predicted_sequence, target_sequence)
    psnr = -10 * torch.log10(mse)
    
    # Temporal consistency (frame-to-frame changes)
    pred_temporal_diff = torch.diff(predicted_sequence, dim=1)
    target_temporal_diff = torch.diff(target_sequence, dim=1)
    temporal_consistency = F.mse_loss(pred_temporal_diff, target_temporal_diff)
    
    # Long-range consistency (compare frames far apart)
    if predicted_sequence.size(1) >= 10:
        stride = predicted_sequence.size(1) // 5
        long_range_pred = predicted_sequence[:, ::stride]
        long_range_target = target_sequence[:, ::stride]
        long_range_consistency = F.mse_loss(long_range_pred, long_range_target)
    else:
        long_range_consistency = torch.tensor(0.0)
    
    # Perceptual similarity
    cosine_sim = F.cosine_similarity(
        predicted_sequence.flatten(2).mean(dim=2),
        target_sequence.flatten(2).mean(dim=2),
        dim=1
    ).mean()
    
    # Motion consistency (optical flow approximation)
    motion_pred = torch.diff(predicted_sequence, dim=1)
    motion_target = torch.diff(target_sequence, dim=1)
    motion_consistency = F.cosine_similarity(
        motion_pred.flatten(2),
        motion_target.flatten(2),
        dim=-1
    ).mean()
    
    return {
        'mse': mse.item(),
        'psnr': psnr.item(),
        'temporal_consistency': temporal_consistency.item(),
        'long_range_consistency': long_range_consistency.item(),
        'cosine_similarity': cosine_sim.item(),
        'motion_consistency': motion_consistency.item(),
        'sequence_length': predicted_sequence.size(1)
    }


def test_sequence_generator():
    """Test the sequence generation model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = SequenceGenerator(
        in_channels=3,
        latent_dim=256,
        num_embeddings=512,
        d_model=512,
        num_layers=6,  # Smaller for testing
        num_heads=8,
        max_sequence_length=48,
        freeze_vqvae=False  # Don't freeze for testing
    ).to(device)
    
    print(f"[TEST] Sequence Generator")
    print(f"   Device: {device}")
    
    # Test input: single initial frame
    batch_size = 1
    initial_frame = torch.randn(batch_size, 1, 3, 256, 256).to(device)
    
    print(f"   Input shape: {initial_frame.shape}")
    
    try:
        # Test diverse sequence generation
        sequences = model.generate_diverse_sequences(
            initial_frame, 
            num_sequences=3,
            sequence_length=10,  # Shorter for testing
            temperature=1.0
        )
        
        print(f"   [OK] Generated {len(sequences)} diverse sequences")
        for i, seq in enumerate(sequences):
            print(f"   [INFO] Sequence {i+1}: {seq.shape}")
        
        # Test training forward pass
        test_sequence = torch.randn(batch_size, 8, 3, 256, 256).to(device)
        predicted, target, additional_losses = model(test_sequence)
        
        print(f"   [OK] Training forward pass successful")
        print(f"   [INFO] Predicted: {predicted.shape}")
        print(f"   [INFO] Target: {target.shape}")
        print(f"   [INFO] Additional losses: {list(additional_losses.keys())}")
        
        # Calculate metrics
        metrics = calculate_sequence_metrics(predicted, target)
        print(f"   [METRIC] Sequence PSNR: {metrics['psnr']:.2f} dB")
        print(f"   [METRIC] Temporal consistency: {metrics['temporal_consistency']:.6f}")
        print(f"   [METRIC] Motion consistency: {metrics['motion_consistency']:.4f}")
        
        print(f"   [SUCCESS] Sequence generator test passed!")
        return True
        
    except Exception as e:
        print(f"   [ERROR] Test failed: {e}")
        return False


if __name__ == "__main__":
    test_sequence_generator()
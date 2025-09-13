"""
Temporal Predictor for Milestone 2: Next-Frame Prediction
Combines VQ-VAE with temporal dynamics to predict future frames
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math
from typing import Optional, Tuple, Dict, List
import numpy as np

from .improved_vqvae import ImprovedVQVAE


class PositionalEncoding(nn.Module):
    """Positional encoding for temporal sequences"""
    
    def __init__(self, d_model: int, max_seq_length: int = 1000):
        super().__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)].detach()


class TemporalAttention(nn.Module):
    """Multi-head attention for temporal modeling"""
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous().reshape(
            batch_size, seq_len, self.d_model
        )
        out = self.out_proj(out)
        
        return out


class TemporalBlock(nn.Module):
    """Transformer block with temporal attention"""
    
    def __init__(self, d_model: int, num_heads: int = 8, dim_feedforward: int = 2048, 
                 dropout: float = 0.1):
        super().__init__()
        
        self.self_attention = TemporalAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual
        attn_out = self.self_attention(x, mask)
        x = self.norm1(x + attn_out)
        
        # Feedforward with residual
        ff_out = self.feedforward(x)
        x = self.norm2(x + ff_out)
        
        return x


class TemporalDynamicsModel(nn.Module):
    """Temporal dynamics model for next-frame prediction"""
    
    def __init__(
        self,
        latent_dim: int = 256,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        max_sequence_length: int = 32,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.max_sequence_length = max_sequence_length
        
        # Project latents to model dimension
        self.latent_proj = nn.Linear(latent_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_sequence_length)
        
        # Temporal transformer layers
        self.layers = nn.ModuleList([
            TemporalBlock(d_model, num_heads, d_model * 4, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
        # Output projection back to latent space
        self.output_proj = nn.Linear(d_model, latent_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal mask to prevent looking at future frames"""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions
    
    def forward(self, latent_sequence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent_sequence: (batch, sequence_length, latent_dim)
        
        Returns:
            next_frame_prediction: (batch, sequence_length, latent_dim)
        """
        batch_size, seq_len, _ = latent_sequence.shape
        
        # Project to model dimension
        x = self.latent_proj(latent_sequence)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
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
    
    def predict_next(self, latent_sequence: torch.Tensor, num_predictions: int = 1) -> torch.Tensor:
        """Predict multiple future frames"""
        self.eval()
        with torch.no_grad():
            predictions = []
            current_sequence = latent_sequence
            
            for _ in range(num_predictions):
                # Predict next frame
                output = self.forward(current_sequence)
                next_frame = output[:, -1:, :]  # Take last prediction
                
                predictions.append(next_frame)
                
                # Append prediction to sequence for next iteration
                current_sequence = torch.cat([current_sequence, next_frame], dim=1)
                
                # Keep only the last max_sequence_length frames
                if current_sequence.shape[1] > self.max_sequence_length:
                    current_sequence = current_sequence[:, -self.max_sequence_length:, :]
            
            return torch.cat(predictions, dim=1)


class WorldModelWithPrediction(nn.Module):
    """Complete world model with VQ-VAE and temporal prediction"""
    
    def __init__(
        self,
        # VQ-VAE parameters
        in_channels: int = 3,
        latent_dim: int = 256,
        num_embeddings: int = 512,
        vqvae_hidden_dims: List[int] = None,
        
        # Temporal model parameters  
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        max_sequence_length: int = 32,
        
        # Training parameters
        use_pretrained_vqvae: bool = True,
        freeze_vqvae: bool = True
    ):
        super().__init__()
        
        if vqvae_hidden_dims is None:
            vqvae_hidden_dims = [64, 128, 256, 512]
        
        # VQ-VAE for spatial encoding/decoding
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
        
        # Temporal dynamics model
        self.temporal_model = TemporalDynamicsModel(
            latent_dim=self.spatial_latent_dim,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            max_sequence_length=max_sequence_length
        )
        
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
    
    def forward(self, frame_sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            frame_sequence: (batch, sequence_length, channels, height, width)
        
        Returns:
            predicted_frames: Next frames in the sequence
            target_latents: Ground truth latents for loss computation
        """
        # Encode frames to latent space
        latents = self.encode_frame_sequence(frame_sequence)
        
        # Predict next frame latents
        predicted_latents = self.temporal_model(latents[:, :-1])  # Don't use last frame as input
        
        # Get target latents (shifted by 1)
        target_latents = latents[:, 1:]  # Don't include first frame as target
        
        return predicted_latents, target_latents
    
    @torch.no_grad()
    def predict_future(self, initial_frames: torch.Tensor, num_future_frames: int = 5) -> torch.Tensor:
        """Predict future frames given initial sequence"""
        # Encode initial frames
        initial_latents = self.encode_frame_sequence(initial_frames)
        
        # Predict future latents
        future_latents = self.temporal_model.predict_next(initial_latents, num_future_frames)
        
        # Decode to frames
        future_frames = self.decode_latent_sequence(future_latents)
        
        return future_frames
    
    def load_vqvae_checkpoint(self, checkpoint_path: str):
        """Load pretrained VQ-VAE weights"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            # PyTorch Lightning checkpoint
            self.vqvae.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Direct state dict
            self.vqvae.load_state_dict(checkpoint)
        
        print(f"✅ Loaded VQ-VAE checkpoint from {checkpoint_path}")


def calculate_temporal_metrics(predicted: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """Calculate metrics for temporal prediction quality"""
    
    # MSE loss
    mse = F.mse_loss(predicted, target)
    
    # Temporal consistency (difference between consecutive frames)
    pred_diff = torch.diff(predicted, dim=1)
    target_diff = torch.diff(target, dim=1)
    temporal_consistency = F.mse_loss(pred_diff, target_diff)
    
    # Perceptual metrics (simplified)
    cosine_similarity = F.cosine_similarity(
        predicted.flatten(2).mean(dim=2),
        target.flatten(2).mean(dim=2),
        dim=1
    ).mean()
    
    return {
        'mse': mse.item(),
        'temporal_consistency': temporal_consistency.item(),
        'cosine_similarity': cosine_similarity.item(),
        'psnr': -10 * torch.log10(mse).item()
    }


def test_temporal_predictor():
    """Test the temporal prediction model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = WorldModelWithPrediction(
        in_channels=3,
        latent_dim=256,
        num_embeddings=512,
        d_model=512,
        num_layers=4,  # Smaller for testing
        num_heads=8,
        max_sequence_length=16,
        freeze_vqvae=False  # Don't freeze for testing
    ).to(device)
    
    # Test input: batch of frame sequences
    batch_size = 2
    seq_len = 8
    test_frames = torch.randn(batch_size, seq_len, 3, 256, 256).to(device)
    
    print(f"📱 Testing Temporal Predictor")
    print(f"   Input shape: {test_frames.shape}")
    
    # Forward pass
    try:
        predicted_latents, target_latents = model(test_frames)
        
        print(f"   ✅ Forward pass successful")
        print(f"   📊 Predicted latents: {predicted_latents.shape}")
        print(f"   📊 Target latents: {target_latents.shape}")
        
        # Calculate metrics
        metrics = calculate_temporal_metrics(predicted_latents, target_latents)
        print(f"   📈 MSE: {metrics['mse']:.4f}")
        print(f"   📈 Temporal consistency: {metrics['temporal_consistency']:.4f}")
        print(f"   📈 Cosine similarity: {metrics['cosine_similarity']:.4f}")
        
        # Test prediction
        initial_frames = test_frames[:, :4]  # First 4 frames
        future_frames = model.predict_future(initial_frames, num_future_frames=3)
        print(f"   🔮 Future prediction: {future_frames.shape}")
        
        # Check memory usage
        if device.type == 'cuda':
            memory_used = torch.cuda.max_memory_allocated() / 1024**3
            print(f"   💾 GPU Memory: {memory_used:.2f} GB")
        
        print(f"   ✅ Temporal predictor test passed!")
        return True
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    test_temporal_predictor()
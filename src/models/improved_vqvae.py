"""
Improved VQ-VAE implementation for Milestone 1: Static World Reconstruction
Target: PSNR > 30dB, <4GB VRAM for encoding/decoding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np
from typing import Tuple, Dict, Optional


class ImprovedVectorQuantizer(nn.Module):
    """
    Improved Vector Quantizer with EMA updates and better initialization
    """
    def __init__(
        self,
        num_embeddings: int = 512,
        embedding_dim: int = 256,
        commitment_cost: float = 0.25,
        use_ema: bool = True,
        ema_decay: float = 0.99,
        epsilon: float = 1e-5
    ):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.epsilon = epsilon
        
        # Initialize embeddings
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.normal_()
        
        if use_ema:
            self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
            self.register_buffer('ema_embed_avg', self.embedding.weight.data.clone())
            
    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        # Flatten input
        input_shape = inputs.shape
        flat_input = inputs.reshape(-1, self.embedding_dim)
        
        # Calculate distances
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self.embedding.weight.t())
        )
        
        # Get nearest embedding indices
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize
        quantized = torch.matmul(encodings, self.embedding.weight).reshape(input_shape)
        
        # Update embeddings with EMA
        if self.use_ema and self.training:
            with torch.no_grad():
                # Update cluster size
                encodings_sum = encodings.sum(0)
                self.ema_cluster_size = self.ema_cluster_size * self.ema_decay + encodings_sum * (1 - self.ema_decay)
                
                # Update embedding average
                embed_sum = torch.matmul(encodings.t(), flat_input)
                self.ema_embed_avg = self.ema_embed_avg * self.ema_decay + embed_sum * (1 - self.ema_decay)
                
                # Normalize
                n = self.ema_cluster_size.sum()
                cluster_size = (self.ema_cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n
                embed_normalized = self.ema_embed_avg / cluster_size.unsqueeze(1)
                self.embedding.weight.data.copy_(embed_normalized)
        
        # Calculate losses
        if not self.use_ema:
            e_latent_loss = F.mse_loss(quantized.detach(), inputs)
            q_latent_loss = F.mse_loss(quantized, inputs.detach())
            loss = q_latent_loss + self.commitment_cost * e_latent_loss
        else:
            e_latent_loss = F.mse_loss(quantized.detach(), inputs)
            loss = self.commitment_cost * e_latent_loss
        
        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()
        
        # Calculate perplexity
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return quantized, {
            'loss': loss,
            'perplexity': perplexity,
            'encodings': encodings,
            'encoding_indices': encoding_indices,
            'distances': distances
        }


class ResidualBlock(nn.Module):
    """Improved Residual Block with GroupNorm for better stability"""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.GroupNorm(8, out_channels)
            )
        
        self.activation = nn.SiLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out += self.shortcut(x)
        out = self.activation(out)
        return out


class AttentionBlock(nn.Module):
    """Self-attention block for capturing long-range dependencies"""
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)
        self.scale = channels ** -0.5
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)
        
        # Reshape for attention
        b, c, h, w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        v = rearrange(v, 'b c h w -> b (h w) c')
        
        # Compute attention
        attn = torch.bmm(q, k) * self.scale
        attn = F.softmax(attn, dim=2)
        
        # Apply attention
        out = torch.bmm(attn, v)
        out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)
        out = self.proj_out(out)
        
        return x + out


class Encoder(nn.Module):
    """Improved encoder with attention and residual connections"""
    def __init__(
        self,
        in_channels: int = 3,
        hidden_dims: list = None,
        latent_dim: int = 256,
        use_attention: bool = True
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [64, 128, 256, 512]
        
        self.use_attention = use_attention
        
        # Initial convolution
        self.input_conv = nn.Conv2d(in_channels, hidden_dims[0], 3, 1, 1)
        
        # Build encoder layers
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    ResidualBlock(hidden_dims[i], hidden_dims[i]),
                    ResidualBlock(hidden_dims[i], hidden_dims[i+1], stride=2),
                )
            )
            
            # Add attention at higher resolutions
            if use_attention and i >= len(hidden_dims) - 3:
                modules.append(AttentionBlock(hidden_dims[i+1]))
        
        self.encoder = nn.Sequential(*modules)
        
        # Final layers
        self.final_block = ResidualBlock(hidden_dims[-1], hidden_dims[-1])
        self.norm_out = nn.GroupNorm(8, hidden_dims[-1])
        self.conv_out = nn.Conv2d(hidden_dims[-1], latent_dim, 1)
        self.activation = nn.SiLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_conv(x)
        x = self.encoder(x)
        x = self.final_block(x)
        x = self.activation(self.norm_out(x))
        x = self.conv_out(x)
        return x


class Decoder(nn.Module):
    """Improved decoder with attention and residual connections"""
    def __init__(
        self,
        latent_dim: int = 256,
        hidden_dims: list = None,
        out_channels: int = 3,
        use_attention: bool = True
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256, 128, 64]
        
        self.use_attention = use_attention
        
        # Initial convolution
        self.input_conv = nn.Conv2d(latent_dim, hidden_dims[0], 1)
        
        # Build decoder layers
        modules = []
        for i in range(len(hidden_dims) - 1):
            # Add attention at lower resolutions
            if use_attention and i < 2:
                modules.append(AttentionBlock(hidden_dims[i]))
            
            modules.append(
                nn.Sequential(
                    ResidualBlock(hidden_dims[i], hidden_dims[i]),
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i+1], 4, 2, 1),
                    ResidualBlock(hidden_dims[i+1], hidden_dims[i+1]),
                )
            )
        
        self.decoder = nn.Sequential(*modules)
        
        # Final layers
        self.final_block = ResidualBlock(hidden_dims[-1], hidden_dims[-1])
        self.norm_out = nn.GroupNorm(8, hidden_dims[-1])
        self.conv_out = nn.Conv2d(hidden_dims[-1], out_channels, 3, 1, 1)
        self.activation = nn.SiLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_conv(x)
        x = self.decoder(x)
        x = self.final_block(x)
        x = self.activation(self.norm_out(x))
        x = self.conv_out(x)
        return x


class ImprovedVQVAE(nn.Module):
    """
    Complete Improved VQ-VAE for Milestone 1
    Target: PSNR > 30dB on game images
    """
    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 256,
        num_embeddings: int = 512,
        hidden_dims: list = None,
        use_ema: bool = True,
        use_attention: bool = True,
    ):
        super().__init__()
        
        self.encoder = Encoder(in_channels, hidden_dims, latent_dim, use_attention)
        self.decoder = Decoder(latent_dim, hidden_dims[::-1] if hidden_dims else None, in_channels, use_attention)
        self.vq = ImprovedVectorQuantizer(num_embeddings, latent_dim, use_ema=use_ema)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.GroupNorm):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Encode images to quantized latent representations.

        Args:
            x: Input images (batch, channels, height, width)

        Returns:
            Tuple of (quantized latents, VQ statistics dict)
        """
        z = self.encoder(x)
        z = rearrange(z, 'b c h w -> b h w c')
        quantized, vq_dict = self.vq(z)
        quantized = rearrange(quantized, 'b h w c -> b c h w')
        return quantized, vq_dict

    def decode(self, quantized: torch.Tensor) -> torch.Tensor:
        """
        Decode quantized latents to images.

        Args:
            quantized: Quantized latent representations

        Returns:
            Reconstructed images (batch, channels, height, width)
        """
        return self.decoder(quantized)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass: encode, quantize, and decode.

        Args:
            x: Input images (batch, channels, height, width)

        Returns:
            Tuple of (reconstructed images, VQ statistics dict)
        """
        quantized, vq_dict = self.encode(x)
        x_recon = self.decode(quantized)

        return x_recon, vq_dict

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct input images without gradients.

        Args:
            x: Input images (batch, channels, height, width)

        Returns:
            Reconstructed images
        """
        quantized, _ = self.encode(x)
        return self.decode(quantized)

    def get_codebook_usage(self, dataloader: Any, device: str = 'cuda') -> torch.Tensor:
        """
        Analyze codebook utilization across a dataset.

        Args:
            dataloader: PyTorch DataLoader
            device: Device to run on ('cuda' or 'cpu')

        Returns:
            Normalized usage count per codebook entry
        """
        usage_count = torch.zeros(self.vq.num_embeddings, device=device)
        
        for batch in dataloader:
            if isinstance(batch, dict):
                x = batch['images'].to(device)
            else:
                x = batch[0].to(device)
            
            with torch.no_grad():
                _, vq_dict = self.encode(x)
                indices = vq_dict['encoding_indices'].flatten()
                usage_count.scatter_add_(0, indices, torch.ones_like(indices, dtype=torch.float))
        
        return usage_count / usage_count.sum()


def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Calculate PSNR between two images"""
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def test_vqvae():
    """Test function to verify the model works"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create model
    model = ImprovedVQVAE(
        in_channels=3,
        latent_dim=256,
        num_embeddings=512,
        hidden_dims=[64, 128, 256, 512],
        use_ema=True,
        use_attention=True
    ).to(device)
    
    # Test input
    x = torch.randn(2, 3, 256, 256).to(device)
    
    # Forward pass
    x_recon, vq_dict = model(x)
    
    # Calculate metrics
    psnr = calculate_psnr(x, x_recon)
    
    print(f"✅ Model initialized successfully!")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {x_recon.shape}")
    print(f"VQ Loss: {vq_dict['loss'].item():.4f}")
    print(f"Perplexity: {vq_dict['perplexity'].item():.2f}")
    print(f"Initial PSNR: {psnr:.2f} dB")
    
    # Check memory usage
    if device == 'cuda':
        memory_used = torch.cuda.max_memory_allocated() / 1024**3
        print(f"GPU Memory used: {memory_used:.2f} GB")
    
    return model


if __name__ == "__main__":
    test_vqvae()
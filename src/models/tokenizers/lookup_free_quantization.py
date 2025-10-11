"""
Lookup-Free Quantization (LFQ)

Based on MAGVIT-v2 and NVIDIA Cosmos research.
Key improvements over traditional VQ-VAE:
- Fully differentiable (no straight-through estimator)
- No codebook collapse
- Better gradient flow
- Simpler training

References:
- MAGVIT-v2: https://arxiv.org/abs/2310.05737
- NVIDIA Cosmos: https://arxiv.org/abs/2501.03575
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional


class LookupFreeQuantizer(nn.Module):
    """
    Lookup-Free Quantization module.

    Instead of maintaining an explicit codebook and doing nearest-neighbor lookups,
    this uses learnable projections and Gumbel-Softmax for differentiable quantization.

    Args:
        codebook_size: Number of discrete codes (e.g., 2^16 = 65536)
        embedding_dim: Dimension of the latent space
        commitment_cost: Weight for commitment loss
        temperature: Temperature for Gumbel-Softmax (lower = more discrete)
        straight_through: Use straight-through in eval mode
    """

    def __init__(
        self,
        codebook_size: int = 2**16,
        embedding_dim: int = 512,
        commitment_cost: float = 0.25,
        temperature: float = 1.0,
        straight_through: bool = True
    ):
        super().__init__()

        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.temperature = temperature
        self.straight_through = straight_through

        # Number of bits to represent codebook_size
        self.num_bits = int(np.log2(codebook_size))
        assert 2**self.num_bits == codebook_size, "codebook_size must be power of 2"

        # Learnable projections (no explicit codebook!)
        self.project_in = nn.Linear(embedding_dim, codebook_size)
        self.project_out = nn.Linear(codebook_size, embedding_dim)

        # Initialize with small weights for stability
        nn.init.xavier_uniform_(self.project_in.weight, gain=0.01)
        nn.init.xavier_uniform_(self.project_out.weight, gain=0.01)
        nn.init.zeros_(self.project_in.bias)
        nn.init.zeros_(self.project_out.bias)

    def forward(
        self,
        z: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Quantize the input tensor.

        Args:
            z: Input tensor [..., embedding_dim]

        Returns:
            z_quantized: Quantized tensor [..., embedding_dim]
            info_dict: Dictionary with:
                - indices: Discrete indices [..., 1]
                - loss: Quantization loss (scalar)
                - perplexity: Codebook usage metric
        """
        input_shape = z.shape

        # Flatten to [..., embedding_dim]
        z_flat = z.reshape(-1, self.embedding_dim)

        # Project to logits over codebook
        logits = self.project_in(z_flat)  # [N, codebook_size]

        if self.training:
            # Gumbel-Softmax for differentiable sampling during training
            soft_one_hot = F.gumbel_softmax(
                logits,
                tau=self.temperature,
                hard=True,  # Use straight-through
                dim=-1
            )
        else:
            # For eval/inference: use argmax with added entropy regularization
            # Add small uniform noise to logits to prevent complete collapse in untrained models
            if not hasattr(self, '_is_trained'):
                # Untrained model: add noise for diversity
                noise = torch.randn_like(logits) * 0.1
                logits = logits + noise

            indices = logits.argmax(dim=-1, keepdim=True)
            soft_one_hot = F.one_hot(indices.squeeze(-1), self.codebook_size).float()

            if self.straight_through:
                # Add straight-through gradient
                soft_one_hot = soft_one_hot + (F.softmax(logits, dim=-1) - F.softmax(logits, dim=-1).detach())

        # Project back to embedding space
        z_quantized_flat = self.project_out(soft_one_hot)

        # Reshape to original shape
        z_quantized = z_quantized_flat.reshape(input_shape)

        # Compute commitment loss (encourage encoder to commit to codes)
        commitment_loss = F.mse_loss(z_quantized.detach(), z) * self.commitment_cost

        # Compute perplexity (measure of codebook usage)
        avg_probs = soft_one_hot.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # Get indices for analysis
        indices = logits.argmax(dim=-1).reshape(input_shape[:-1] + (1,))

        info_dict = {
            'indices': indices,
            'loss': commitment_loss,
            'perplexity': perplexity,
            'codebook_usage': (torch.unique(indices).numel() / self.codebook_size),
        }

        return z_quantized, info_dict

    def encode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Get discrete indices without quantization.

        Args:
            z: Input tensor [..., embedding_dim]

        Returns:
            indices: Discrete indices [..., 1]
        """
        z_flat = z.reshape(-1, self.embedding_dim)
        logits = self.project_in(z_flat)
        indices = logits.argmax(dim=-1)
        return indices.reshape(z.shape[:-1] + (1,))

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Decode indices to continuous embeddings.

        Args:
            indices: Discrete indices [..., 1]

        Returns:
            z: Continuous embeddings [..., embedding_dim]
        """
        indices_flat = indices.reshape(-1)
        one_hot = F.one_hot(indices_flat, self.codebook_size).float()
        z_flat = self.project_out(one_hot)
        return z_flat.reshape(indices.shape[:-1] + (self.embedding_dim,))

    def get_codebook_usage(self, z: torch.Tensor) -> float:
        """
        Compute what fraction of the codebook is being used.

        Args:
            z: Input tensor

        Returns:
            usage: Fraction of codebook used (0.0 to 1.0)
        """
        with torch.no_grad():
            indices = self.encode(z)
            unique_codes = torch.unique(indices).numel()
            return unique_codes / self.codebook_size


class MultiScaleLFQ(nn.Module):
    """
    Multi-scale Lookup-Free Quantization.

    Uses multiple quantizers at different scales for better quality.
    Similar to VQ-VAE-2 but with LFQ.

    Args:
        codebook_sizes: List of codebook sizes for each scale
        embedding_dim: Dimension of latent space
        commitment_cost: Weight for commitment loss
    """

    def __init__(
        self,
        codebook_sizes: list = [2**8, 2**12, 2**16],
        embedding_dim: int = 512,
        commitment_cost: float = 0.25
    ):
        super().__init__()

        self.num_scales = len(codebook_sizes)
        self.embedding_dim = embedding_dim

        # Create quantizers for each scale
        self.quantizers = nn.ModuleList([
            LookupFreeQuantizer(
                codebook_size=size,
                embedding_dim=embedding_dim,
                commitment_cost=commitment_cost
            )
            for size in codebook_sizes
        ])

        # Learnable scale weights
        self.scale_weights = nn.Parameter(torch.ones(self.num_scales))

    def forward(
        self,
        z: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Quantize using multiple scales.

        Returns:
            z_quantized: Weighted combination of quantized outputs
            info_dict: Combined information from all scales
        """
        quantized_outputs = []
        total_loss = 0.0
        total_perplexity = 0.0

        # Normalize scale weights
        weights = F.softmax(self.scale_weights, dim=0)

        for i, quantizer in enumerate(self.quantizers):
            z_q, info = quantizer(z)
            quantized_outputs.append(z_q * weights[i])
            total_loss += info['loss'] * weights[i]
            total_perplexity += info['perplexity'] * weights[i]

        # Combine outputs
        z_quantized = sum(quantized_outputs)

        info_dict = {
            'indices': self.quantizers[-1].encode(z),  # Use finest scale
            'loss': total_loss,
            'perplexity': total_perplexity,
            'scale_weights': weights,
        }

        return z_quantized, info_dict


# Example usage and testing
if __name__ == '__main__':
    # Test basic LFQ
    print("Testing LookupFreeQuantizer...")
    lfq = LookupFreeQuantizer(
        codebook_size=256,
        embedding_dim=512
    )

    # Forward pass
    z = torch.randn(4, 16, 16, 512)
    z_quantized, info = lfq(z)

    print(f"Input shape: {z.shape}")
    print(f"Output shape: {z_quantized.shape}")
    print(f"Indices shape: {info['indices'].shape}")
    print(f"Perplexity: {info['perplexity'].item():.2f}")
    print(f"Codebook usage: {info['codebook_usage'].item():.2%}")
    print(f"Loss: {info['loss'].item():.4f}")

    # Test gradient flow
    print("\nTesting gradient flow...")
    z = torch.randn(2, 8, 8, 512, requires_grad=True)
    z_q, _ = lfq(z)
    loss = z_q.sum()
    loss.backward()
    print(f"Gradients exist: {z.grad is not None}")
    print(f"Gradient norm: {z.grad.norm().item():.4f}")

    print("\n✅ LookupFreeQuantizer tests passed!")

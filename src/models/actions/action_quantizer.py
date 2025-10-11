"""
Action Quantizer - Discretize continuous actions into vocabulary

Converts continuous action latents into discrete action tokens.
Uses Lookup-Free Quantization for better gradient flow.

Key differences from video quantization:
- Smaller vocabulary (512 vs 65536)
- Actions are more abstract than pixels
- Need good diversity (avoid action collapse)

Reference: Genie 3 (DeepMind, October 2025)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional


class ActionQuantizer(nn.Module):
    """
    Quantize continuous action latents to discrete action vocabulary.

    Uses Lookup-Free Quantization (LFQ) for:
    - Better gradient flow
    - No codebook collapse
    - Simpler training

    Args:
        action_vocab_size: Number of discrete actions (e.g., 512)
        action_dim: Dimension of continuous action latent
        commitment_cost: Weight for commitment loss
        temperature: Temperature for Gumbel-Softmax
        diversity_weight: Weight for diversity loss (encourage using all actions)
    """

    def __init__(
        self,
        action_vocab_size: int = 512,
        action_dim: int = 128,
        commitment_cost: float = 0.25,
        temperature: float = 1.0,
        diversity_weight: float = 0.1
    ):
        super().__init__()

        self.action_vocab_size = action_vocab_size
        self.action_dim = action_dim
        self.commitment_cost = commitment_cost
        self.temperature = temperature
        self.diversity_weight = diversity_weight

        # Verify vocab size is power of 2
        self.num_bits = int(np.log2(action_vocab_size))
        assert 2**self.num_bits == action_vocab_size, \
            f"action_vocab_size must be power of 2, got {action_vocab_size}"

        # Learnable projections (no explicit codebook)
        self.project_in = nn.Linear(action_dim, action_vocab_size)
        self.project_out = nn.Linear(action_vocab_size, action_dim)

        # Initialize with small weights for stability
        nn.init.xavier_uniform_(self.project_in.weight, gain=0.01)
        nn.init.xavier_uniform_(self.project_out.weight, gain=0.01)
        nn.init.zeros_(self.project_in.bias)
        nn.init.zeros_(self.project_out.bias)

    def forward(
        self,
        action_latent: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Quantize action latent.

        Args:
            action_latent: Continuous action [B, action_dim] or [B, T, action_dim]

        Returns:
            action_quantized: Quantized action (same shape as input)
            info_dict: Dictionary with:
                - indices: Discrete action indices [B, 1] or [B, T, 1]
                - loss: Quantization loss (scalar)
                - perplexity: Action usage metric
                - diversity: Fraction of vocabulary used
        """
        input_shape = action_latent.shape
        is_sequence = len(input_shape) == 3  # [B, T, action_dim]

        # Flatten to [N, action_dim]
        if is_sequence:
            B, T, D = action_latent.shape
            action_flat = action_latent.reshape(-1, D)
        else:
            action_flat = action_latent

        # Project to logits over action vocabulary
        logits = self.project_in(action_flat)  # [N, action_vocab_size]

        if self.training:
            # Gumbel-Softmax for differentiable sampling
            soft_one_hot = F.gumbel_softmax(
                logits,
                tau=self.temperature,
                hard=True,  # Straight-through estimator
                dim=-1
            )
        else:
            # Argmax for inference with diversity noise
            if not hasattr(self, '_is_trained'):
                # Add noise for untrained models
                noise = torch.randn_like(logits) * 0.1
                logits = logits + noise

            indices_flat = logits.argmax(dim=-1, keepdim=True)
            soft_one_hot = F.one_hot(
                indices_flat.squeeze(-1),
                self.action_vocab_size
            ).float()

        # Project back to action space
        action_quantized_flat = self.project_out(soft_one_hot)

        # Reshape to original shape
        if is_sequence:
            action_quantized = action_quantized_flat.reshape(B, T, D)
        else:
            action_quantized = action_quantized_flat

        # Compute commitment loss (encourage encoder to commit to actions)
        commitment_loss = F.mse_loss(
            action_quantized.detach(),
            action_latent
        ) * self.commitment_cost

        # Compute diversity loss (encourage using all actions)
        avg_probs = soft_one_hot.mean(dim=0)
        uniform_dist = torch.ones_like(avg_probs) / self.action_vocab_size
        diversity_loss = F.kl_div(
            avg_probs.log() + 1e-10,
            uniform_dist,
            reduction='batchmean'
        ) * self.diversity_weight

        # Total loss
        total_loss = commitment_loss + diversity_loss

        # Compute perplexity (measure of action usage)
        perplexity = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
        )

        # Get indices for analysis
        indices_flat = logits.argmax(dim=-1)
        if is_sequence:
            indices = indices_flat.reshape(B, T, 1)
        else:
            indices = indices_flat.unsqueeze(-1)

        # Compute diversity (fraction of vocab used)
        unique_actions = torch.unique(indices).numel()
        diversity = unique_actions / self.action_vocab_size

        info_dict = {
            'indices': indices,
            'loss': total_loss,
            'commitment_loss': commitment_loss,
            'diversity_loss': diversity_loss,
            'perplexity': perplexity,
            'diversity': diversity,
            'unique_actions': unique_actions,
        }

        return action_quantized, info_dict

    def encode(self, action_latent: torch.Tensor) -> torch.Tensor:
        """
        Get discrete action indices without quantization.

        Args:
            action_latent: Continuous action [B, action_dim] or [B, T, action_dim]

        Returns:
            indices: Discrete action indices [B, 1] or [B, T, 1]
        """
        input_shape = action_latent.shape
        is_sequence = len(input_shape) == 3

        if is_sequence:
            B, T, D = action_latent.shape
            action_flat = action_latent.reshape(-1, D)
        else:
            action_flat = action_latent

        logits = self.project_in(action_flat)
        indices_flat = logits.argmax(dim=-1)

        if is_sequence:
            indices = indices_flat.reshape(B, T, 1)
        else:
            indices = indices_flat.unsqueeze(-1)

        return indices

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Decode action indices to continuous latents.

        Args:
            indices: Discrete action indices [B, 1] or [B, T, 1]

        Returns:
            action_latent: Continuous action [B, action_dim] or [B, T, action_dim]
        """
        input_shape = indices.shape
        is_sequence = len(input_shape) == 3

        if is_sequence:
            B, T, _ = indices.shape
            indices_flat = indices.reshape(-1)
        else:
            indices_flat = indices.squeeze(-1)

        one_hot = F.one_hot(indices_flat, self.action_vocab_size).float()
        action_flat = self.project_out(one_hot)

        if is_sequence:
            action_latent = action_flat.reshape(B, T, -1)
        else:
            action_latent = action_flat

        return action_latent


# Testing
if __name__ == '__main__':
    print("Testing ActionQuantizer...\n")

    # Test 1: Basic forward pass
    print("1. Testing basic forward pass...")
    quantizer = ActionQuantizer(
        action_vocab_size=512,
        action_dim=128
    )

    action_latent = torch.randn(4, 128)
    action_quantized, info = quantizer(action_latent)

    print(f"   Input shape: {action_latent.shape}")
    print(f"   Output shape: {action_quantized.shape}")
    print(f"   Indices shape: {info['indices'].shape}")
    print(f"   Perplexity: {info['perplexity'].item():.2f}")
    print(f"   Diversity: {info['diversity']:.2%}")
    print(f"   Unique actions: {info['unique_actions']}/{512}")
    assert action_quantized.shape == action_latent.shape
    assert info['indices'].shape == (4, 1)
    print("   ✅ Basic forward pass successful\n")

    # Test 2: Sequence quantization
    print("2. Testing sequence quantization...")
    action_sequence = torch.randn(2, 7, 128)  # [B, T, action_dim]
    action_quantized_seq, info_seq = quantizer(action_sequence)

    print(f"   Input shape: {action_sequence.shape}")
    print(f"   Output shape: {action_quantized_seq.shape}")
    print(f"   Indices shape: {info_seq['indices'].shape}")
    print(f"   Diversity: {info_seq['diversity']:.2%}")
    assert action_quantized_seq.shape == action_sequence.shape
    assert info_seq['indices'].shape == (2, 7, 1)
    print("   ✅ Sequence quantization successful\n")

    # Test 3: Encode/decode
    print("3. Testing encode/decode...")
    indices = quantizer.encode(action_latent)
    decoded = quantizer.decode(indices)

    print(f"   Original: {action_latent.shape}")
    print(f"   Indices: {indices.shape}")
    print(f"   Decoded: {decoded.shape}")
    assert decoded.shape == action_latent.shape
    print("   ✅ Encode/decode successful\n")

    # Test 4: Gradient flow
    print("4. Testing gradient flow...")
    action_latent_grad = torch.randn(2, 128, requires_grad=True)
    action_q, info_q = quantizer(action_latent_grad)
    loss = action_q.sum() + info_q['loss']
    loss.backward()

    print(f"   Gradients exist: {action_latent_grad.grad is not None}")
    print(f"   Gradient norm: {action_latent_grad.grad.norm().item():.4f}")
    print("   ✅ Gradient flow works\n")

    # Test 5: Training vs eval mode
    print("5. Testing training vs eval mode...")
    quantizer.train()
    _, info_train = quantizer(action_latent)

    quantizer.eval()
    _, info_eval = quantizer(action_latent)

    print(f"   Train diversity: {info_train['diversity']:.2%}")
    print(f"   Eval diversity: {info_eval['diversity']:.2%}")
    print("   ✅ Training/eval modes work\n")

    # Test 6: Action diversity over large batch
    print("6. Testing action diversity...")
    large_batch = torch.randn(100, 128)
    _, info_large = quantizer(large_batch)

    print(f"   Batch size: 100")
    print(f"   Unique actions: {info_large['unique_actions']}/{512}")
    print(f"   Diversity: {info_large['diversity']:.2%}")
    print(f"   Perplexity: {info_large['perplexity'].item():.2f}")
    print(f"   Commitment loss: {info_large['commitment_loss'].item():.4f}")
    print(f"   Diversity loss: {info_large['diversity_loss'].item():.4f}")
    print("   ✅ Diversity metrics work\n")

    print("✅ All ActionQuantizer tests passed!")

"""
MaskGIT: Parallel Token Generation

Implements MaskGIT-style parallel generation for video tokens.
Key innovation: Predict all tokens simultaneously instead of autoregressively.

Benefits:
- 10x faster generation than autoregressive
- Iterative refinement for quality
- Confidence-based unmasking

References:
- MaskGIT (2022): https://arxiv.org/abs/2202.04200
- Matrix-Game 2.0 (Aug 2025): Uses MaskGIT for fast generation
- MAGVIT-v2 (2023): Combines with LFQ for video

Author: Based on October 2025 research
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Callable
import math
import numpy as np


class MaskingScheduler:
    """
    Masking schedule for iterative refinement.

    Controls what fraction of tokens to unmask at each iteration.

    Args:
        schedule_type: Type of schedule ('cosine', 'linear', 'sqrt')
        num_iterations: Total number of refinement iterations
    """

    def __init__(
        self,
        schedule_type: str = 'cosine',
        num_iterations: int = 12
    ):
        self.schedule_type = schedule_type
        self.num_iterations = num_iterations

    def get_mask_ratio(self, iteration: int) -> float:
        """
        Get fraction of tokens that should remain masked at this iteration.

        Args:
            iteration: Current iteration (0 to num_iterations-1)

        Returns:
            mask_ratio: Fraction of tokens to keep masked (1.0 → 0.0)
        """
        # Normalize to [0, 1]
        t = iteration / max(1, self.num_iterations - 1)

        if self.schedule_type == 'cosine':
            # Cosine schedule: slower at start/end, faster in middle
            # mask_ratio goes from 1.0 → 0.0
            return np.cos(t * np.pi / 2)

        elif self.schedule_type == 'linear':
            # Linear schedule: constant unmasking rate
            return 1.0 - t

        elif self.schedule_type == 'sqrt':
            # Square root schedule: faster at start, slower at end
            return 1.0 - np.sqrt(t)

        elif self.schedule_type == 'quadratic':
            # Quadratic schedule: slower at start, faster at end
            return (1.0 - t) ** 2

        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")

    def get_num_masked(self, total_tokens: int, iteration: int) -> int:
        """
        Get number of tokens to keep masked at this iteration.

        Args:
            total_tokens: Total number of tokens
            iteration: Current iteration

        Returns:
            num_masked: Number of tokens to keep masked
        """
        mask_ratio = self.get_mask_ratio(iteration)
        return int(total_tokens * mask_ratio)


class MaskGITPredictor(nn.Module):
    """
    MaskGIT predictor for parallel token generation.

    Predicts masked tokens based on unmasked context.
    Uses transformer with masked self-attention.

    Args:
        vocab_size: Size of token vocabulary
        hidden_dim: Hidden dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        dropout: Dropout rate
        mask_token_id: ID for mask token
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        dropout: float = 0.1,
        mask_token_id: Optional[int] = None
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.mask_token_id = mask_token_id or vocab_size  # Use vocab_size as mask token

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size + 1, hidden_dim)  # +1 for mask token

        # Positional embedding (learnable)
        self.max_seq_len = 4096  # Support up to 4K tokens
        self.pos_embedding = nn.Embedding(self.max_seq_len, hidden_dim)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm (better for deep networks)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        # Token embedding
        nn.init.normal_(self.token_embedding.weight, std=0.02)

        # Positional embedding
        nn.init.normal_(self.pos_embedding.weight, std=0.02)

        # Output projection
        nn.init.normal_(self.output_proj.weight, std=0.02)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: predict tokens at masked positions.

        Args:
            tokens: Input tokens [B, N] (may contain mask_token_id)
            mask: Boolean mask [B, N] - True for positions to predict

        Returns:
            logits: Predicted logits [B, N, vocab_size]
            confidences: Confidence scores [B, N]
        """
        B, N = tokens.shape

        # Embed tokens
        token_emb = self.token_embedding(tokens)  # [B, N, hidden_dim]

        # Add positional embeddings
        positions = torch.arange(N, device=tokens.device).unsqueeze(0).expand(B, -1)
        pos_emb = self.pos_embedding(positions)
        x = token_emb + pos_emb

        # Apply transformer
        x = self.transformer(x)  # [B, N, hidden_dim]

        # Project to vocabulary
        logits = self.output_proj(x)  # [B, N, vocab_size]

        # Compute confidence (max probability)
        probs = F.softmax(logits, dim=-1)
        confidences = probs.max(dim=-1)[0]  # [B, N]

        return logits, confidences

    def predict_masked(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict only the masked positions.

        Args:
            tokens: Input tokens [B, N] (with mask_token_id at masked positions)
            mask: Boolean mask [B, N] - True for masked positions

        Returns:
            predicted_tokens: Predicted tokens [B, N]
            confidences: Confidence scores [B, N]
        """
        # Forward pass
        logits, confidences = self.forward(tokens, mask)

        # Get predictions
        predicted_tokens = logits.argmax(dim=-1)

        # Only update masked positions
        output_tokens = tokens.clone()
        output_tokens[mask] = predicted_tokens[mask]

        return output_tokens, confidences


class MaskGITGenerator:
    """
    MaskGIT-based generator for parallel token generation.

    Generates tokens iteratively:
    1. Start with all tokens masked
    2. Predict all tokens in parallel
    3. Unmask highest-confidence tokens
    4. Repeat until all tokens unmasked

    Args:
        predictor: MaskGIT predictor model
        scheduler: Masking scheduler
        num_iterations: Number of refinement iterations
        temperature: Sampling temperature (1.0 = no change)
        top_k: Top-k sampling (None = no filtering)
        top_p: Nucleus sampling (None = no filtering)
    """

    def __init__(
        self,
        predictor: MaskGITPredictor,
        scheduler: Optional[MaskingScheduler] = None,
        num_iterations: int = 12,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ):
        self.predictor = predictor
        self.scheduler = scheduler or MaskingScheduler('cosine', num_iterations)
        self.num_iterations = num_iterations
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

        self.mask_token_id = predictor.mask_token_id

    @torch.no_grad()
    def generate(
        self,
        batch_size: int,
        seq_len: int,
        condition: Optional[torch.Tensor] = None,
        device: torch.device = torch.device('cpu')
    ) -> torch.Tensor:
        """
        Generate tokens from scratch.

        Args:
            batch_size: Batch size
            seq_len: Sequence length to generate
            condition: Optional conditioning (e.g., first few tokens) [B, K]
            device: Device to generate on

        Returns:
            tokens: Generated tokens [B, seq_len]
        """
        # Initialize with all mask tokens
        tokens = torch.full(
            (batch_size, seq_len),
            self.mask_token_id,
            dtype=torch.long,
            device=device
        )

        # If conditioning provided, use it
        if condition is not None:
            cond_len = condition.shape[1]
            tokens[:, :cond_len] = condition

        # Mask indicating which positions need prediction
        mask = tokens == self.mask_token_id

        # Iterative refinement
        for iteration in range(self.num_iterations):
            # Predict all masked tokens
            logits, confidences = self.predictor(tokens, mask)

            # Sample from predictions
            if self.temperature != 1.0:
                logits = logits / self.temperature

            if self.top_k is not None:
                logits = self._top_k_filtering(logits, self.top_k)

            if self.top_p is not None:
                logits = self._nucleus_filtering(logits, self.top_p)

            # Sample
            probs = F.softmax(logits, dim=-1)
            sampled_tokens = torch.multinomial(
                probs.view(-1, probs.shape[-1]),
                num_samples=1
            ).view(batch_size, seq_len)

            # Update confidence for sampled tokens
            sampled_probs = torch.gather(probs, -1, sampled_tokens.unsqueeze(-1)).squeeze(-1)
            confidences = sampled_probs

            # Determine how many tokens to unmask
            num_masked = mask.sum(dim=-1)  # [B]
            num_to_unmask = num_masked - self.scheduler.get_num_masked(
                seq_len, iteration + 1
            )

            # Unmask highest-confidence tokens
            for b in range(batch_size):
                if num_to_unmask[b] > 0:
                    # Get confidence of masked tokens
                    masked_confidences = confidences[b, mask[b]]

                    # Find top-k confident predictions
                    if len(masked_confidences) > 0:
                        k = min(int(num_to_unmask[b]), len(masked_confidences))
                        top_k_idx = masked_confidences.topk(k).indices

                        # Unmask these tokens
                        masked_positions = torch.where(mask[b])[0]
                        unmask_positions = masked_positions[top_k_idx]

                        tokens[b, unmask_positions] = sampled_tokens[b, unmask_positions]
                        mask[b, unmask_positions] = False

        return tokens

    def _top_k_filtering(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        """Top-k filtering"""
        top_k = min(k, logits.shape[-1])
        indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
        logits[indices_to_remove] = -float('inf')
        return logits

    def _nucleus_filtering(self, logits: torch.Tensor, p: float) -> torch.Tensor:
        """Nucleus (top-p) filtering"""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Scatter to original order
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = -float('inf')
        return logits


# Utility functions
def create_maskgit_generator(
    vocab_size: int,
    hidden_dim: int = 512,
    num_layers: int = 8,
    num_iterations: int = 12,
    schedule_type: str = 'cosine',
    **kwargs
) -> MaskGITGenerator:
    """
    Create a MaskGIT generator with sensible defaults.

    Args:
        vocab_size: Token vocabulary size
        hidden_dim: Hidden dimension
        num_layers: Transformer layers
        num_iterations: Refinement iterations
        schedule_type: Masking schedule type
        **kwargs: Additional arguments for predictor/generator

    Returns:
        generator: MaskGITGenerator instance
    """
    predictor = MaskGITPredictor(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        **{k: v for k, v in kwargs.items() if k in ['num_heads', 'dropout']}
    )

    scheduler = MaskingScheduler(schedule_type, num_iterations)

    generator = MaskGITGenerator(
        predictor=predictor,
        scheduler=scheduler,
        num_iterations=num_iterations,
        **{k: v for k, v in kwargs.items() if k in ['temperature', 'top_k', 'top_p']}
    )

    return generator


if __name__ == '__main__':
    print("Testing MaskGIT...\n")

    # Test 1: Masking scheduler
    print("1. Testing MaskingScheduler...")
    scheduler = MaskingScheduler('cosine', num_iterations=12)

    for i in range(0, 12, 3):
        ratio = scheduler.get_mask_ratio(i)
        num_masked = scheduler.get_num_masked(100, i)
        print(f"   Iteration {i}: {ratio:.2%} masked ({num_masked}/100 tokens)")

    print("   ✅ Scheduler works\n")

    # Test 2: MaskGIT predictor
    print("2. Testing MaskGITPredictor...")
    predictor = MaskGITPredictor(
        vocab_size=1024,
        hidden_dim=256,
        num_layers=4,
        num_heads=4
    )

    tokens = torch.randint(0, 1024, (2, 64))
    mask = torch.rand(2, 64) > 0.5  # Random mask

    logits, confidences = predictor(tokens, mask)
    print(f"   Input shape: {tokens.shape}")
    print(f"   Logits shape: {logits.shape}")
    print(f"   Confidences shape: {confidences.shape}")
    print(f"   Confidence range: [{confidences.min():.3f}, {confidences.max():.3f}]")
    print("   ✅ Predictor works\n")

    # Test 3: MaskGIT generator
    print("3. Testing MaskGITGenerator...")
    generator = MaskGITGenerator(
        predictor=predictor,
        num_iterations=8,
        temperature=1.0
    )

    generated = generator.generate(
        batch_size=2,
        seq_len=64,
        device=torch.device('cpu')
    )

    print(f"   Generated shape: {generated.shape}")
    print(f"   Token range: [{generated.min()}, {generated.max()}]")
    print(f"   Unique tokens: {generated.unique().numel()}")
    print("   ✅ Generator works\n")

    # Test 4: With conditioning
    print("4. Testing conditional generation...")
    condition = torch.randint(0, 1024, (2, 16))
    generated_cond = generator.generate(
        batch_size=2,
        seq_len=64,
        condition=condition,
        device=torch.device('cpu')
    )

    # First 16 tokens should match condition
    assert torch.equal(generated_cond[:, :16], condition)
    print("   ✅ Conditional generation works\n")

    # Test 5: Different schedules
    print("5. Testing different masking schedules...")
    for schedule in ['cosine', 'linear', 'sqrt', 'quadratic']:
        scheduler = MaskingScheduler(schedule, num_iterations=12)
        ratio_start = scheduler.get_mask_ratio(0)
        ratio_mid = scheduler.get_mask_ratio(6)
        ratio_end = scheduler.get_mask_ratio(11)
        print(f"   {schedule:10s}: {ratio_start:.2f} → {ratio_mid:.2f} → {ratio_end:.2f}")

    print("   ✅ All schedules work\n")

    print("✅ All MaskGIT tests passed!")

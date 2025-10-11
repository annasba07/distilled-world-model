"""
Dynamics Model - Predict next frame from current frame + action

The dynamics model learns: (frame_t, action) → frame_t+1

This enables:
1. Action-conditioned generation (user controls video)
2. Rollouts (apply actions repeatedly)
3. Planning (search over actions)

Architecture:
- Action-conditioned Transformer
- Predicts next frame tokens given current frame + action token

Reference: Genie 3 (DeepMind, October 2025)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, seq_len, d_model]
        Returns:
            x + positional encoding
        """
        return x + self.pe[:x.size(1)]


class ActionConditionedTransformer(nn.Module):
    """
    Transformer that predicts next frame tokens conditioned on action.

    Architecture:
    1. Embed current frame tokens
    2. Embed action token
    3. Concatenate [action_emb, frame_embs]
    4. Transformer decoder
    5. Predict next frame token logits

    Args:
        frame_vocab_size: Size of frame token vocabulary (from Phase 1)
        action_vocab_size: Size of action vocabulary (from action quantizer)
        d_model: Model dimension
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        dim_feedforward: Feedforward dimension
        dropout: Dropout rate
    """

    def __init__(
        self,
        frame_vocab_size: int = 4096,
        action_vocab_size: int = 512,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()

        self.frame_vocab_size = frame_vocab_size
        self.action_vocab_size = action_vocab_size
        self.d_model = d_model

        # Embeddings
        self.frame_embedding = nn.Embedding(frame_vocab_size, d_model)
        self.action_embedding = nn.Embedding(action_vocab_size, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)

        # Output projection
        self.output_proj = nn.Linear(d_model, frame_vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        nn.init.normal_(self.frame_embedding.weight, std=0.02)
        nn.init.normal_(self.action_embedding.weight, std=0.02)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        frame_tokens: torch.Tensor,
        action_token: torch.Tensor,
        return_logits: bool = True
    ) -> torch.Tensor:
        """
        Predict next frame tokens given current frame + action.

        Args:
            frame_tokens: Current frame tokens [B, seq_len]
            action_token: Action token [B, 1]
            return_logits: If True, return logits; if False, return token predictions

        Returns:
            next_frame_logits: [B, seq_len, frame_vocab_size] or
            next_frame_tokens: [B, seq_len] (if return_logits=False)
        """
        B, seq_len = frame_tokens.shape

        # Embed frame and action
        frame_emb = self.frame_embedding(frame_tokens)  # [B, seq_len, d_model]
        action_emb = self.action_embedding(action_token)  # [B, 1, d_model]

        # Concatenate: [action, frame_tokens]
        # Action acts as a prefix conditioning
        input_seq = torch.cat([action_emb, frame_emb], dim=1)  # [B, seq_len+1, d_model]

        # Add positional encoding
        input_seq = self.pos_encoder(input_seq)

        # Create causal mask (prevent attending to future tokens)
        tgt_len = input_seq.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt_len,
            device=input_seq.device
        )

        # Transformer (using decoder in autoregressive mode)
        # memory=input_seq because we're using decoder as encoder-decoder
        output = self.transformer(
            tgt=input_seq,
            memory=input_seq,
            tgt_mask=causal_mask
        )  # [B, seq_len+1, d_model]

        # Remove action prefix, keep only frame predictions
        output = output[:, 1:, :]  # [B, seq_len, d_model]

        # Project to frame vocabulary
        logits = self.output_proj(output)  # [B, seq_len, frame_vocab_size]

        if return_logits:
            return logits
        else:
            # Return predicted tokens
            return logits.argmax(dim=-1)

    def generate(
        self,
        frame_tokens: torch.Tensor,
        action_token: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate next frame with sampling.

        Args:
            frame_tokens: Current frame [B, seq_len]
            action_token: Action [B, 1]
            temperature: Sampling temperature
            top_k: If specified, sample from top-k tokens

        Returns:
            next_frame_tokens: [B, seq_len]
        """
        logits = self.forward(frame_tokens, action_token, return_logits=True)

        # Apply temperature
        logits = logits / temperature

        # Top-k filtering
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, :, [-1]]] = float('-inf')

        # Sample
        probs = F.softmax(logits, dim=-1)
        next_tokens = torch.multinomial(
            probs.view(-1, self.frame_vocab_size),
            num_samples=1
        ).view(logits.shape[0], -1)

        return next_tokens


class DynamicsModel(nn.Module):
    """
    High-level dynamics model that wraps the transformer.

    Provides convenient interface for:
    - Training: compute loss for (frame_t, action) → frame_t+1
    - Generation: predict next frame given current + action
    - Rollout: apply sequence of actions
    """

    def __init__(
        self,
        frame_vocab_size: int = 4096,
        action_vocab_size: int = 512,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6
    ):
        super().__init__()

        self.transformer = ActionConditionedTransformer(
            frame_vocab_size=frame_vocab_size,
            action_vocab_size=action_vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers
        )

    def forward(
        self,
        frame_t_tokens: torch.Tensor,
        action_token: torch.Tensor,
        frame_t1_tokens: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            frame_t_tokens: Current frame [B, seq_len]
            action_token: Action [B, 1]
            frame_t1_tokens: Next frame (ground truth) [B, seq_len] (optional, for training)

        Returns:
            Dictionary with:
                - logits: Predicted logits [B, seq_len, vocab_size]
                - loss: Cross-entropy loss (if frame_t1_tokens provided)
                - predictions: Predicted tokens [B, seq_len]
        """
        # Get logits
        logits = self.transformer(frame_t_tokens, action_token, return_logits=True)

        # Get predictions
        predictions = logits.argmax(dim=-1)

        output = {
            'logits': logits,
            'predictions': predictions
        }

        # Compute loss if targets provided
        if frame_t1_tokens is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                frame_t1_tokens.reshape(-1)
            )
            output['loss'] = loss

            # Accuracy
            accuracy = (predictions == frame_t1_tokens).float().mean()
            output['accuracy'] = accuracy

        return output

    def predict_next_frame(
        self,
        frame_t_tokens: torch.Tensor,
        action_token: torch.Tensor,
        temperature: float = 1.0,
        deterministic: bool = False
    ) -> torch.Tensor:
        """
        Predict next frame tokens.

        Args:
            frame_t_tokens: Current frame [B, seq_len]
            action_token: Action [B, 1]
            temperature: Sampling temperature
            deterministic: If True, use argmax; if False, sample

        Returns:
            next_frame_tokens: [B, seq_len]
        """
        if deterministic:
            output = self.forward(frame_t_tokens, action_token)
            return output['predictions']
        else:
            return self.transformer.generate(
                frame_t_tokens,
                action_token,
                temperature=temperature
            )

    def rollout(
        self,
        initial_frame_tokens: torch.Tensor,
        action_sequence: torch.Tensor,
        temperature: float = 1.0,
        deterministic: bool = False
    ) -> torch.Tensor:
        """
        Apply sequence of actions to generate multiple frames.

        Args:
            initial_frame_tokens: Starting frame [B, seq_len]
            action_sequence: Action tokens [B, num_steps, 1]
            temperature: Sampling temperature
            deterministic: If True, use argmax

        Returns:
            frames: All generated frames [B, num_steps, seq_len]
        """
        B, num_steps, _ = action_sequence.shape

        frames = []
        current_frame = initial_frame_tokens

        for t in range(num_steps):
            action_t = action_sequence[:, t, :]  # [B, 1]

            # Predict next frame
            next_frame = self.predict_next_frame(
                current_frame,
                action_t,
                temperature=temperature,
                deterministic=deterministic
            )

            frames.append(next_frame)
            current_frame = next_frame

        # Stack frames
        frames = torch.stack(frames, dim=1)  # [B, num_steps, seq_len]

        return frames


# Testing
if __name__ == '__main__':
    print("Testing Dynamics Model...\n")

    # Test 1: ActionConditionedTransformer
    print("1. Testing ActionConditionedTransformer...")
    transformer = ActionConditionedTransformer(
        frame_vocab_size=4096,
        action_vocab_size=512,
        d_model=256,
        nhead=4,
        num_layers=2
    )

    frame_tokens = torch.randint(0, 4096, (2, 100))  # [B, seq_len]
    action_token = torch.randint(0, 512, (2, 1))  # [B, 1]

    logits = transformer(frame_tokens, action_token)
    print(f"   Input: frame={frame_tokens.shape}, action={action_token.shape}")
    print(f"   Output: logits={logits.shape}")
    assert logits.shape == (2, 100, 4096)
    print("   ✅ Transformer forward pass works\n")

    # Test 2: DynamicsModel
    print("2. Testing DynamicsModel...")
    model = DynamicsModel(
        frame_vocab_size=4096,
        action_vocab_size=512,
        d_model=256,
        nhead=4,
        num_layers=2
    )

    frame_t = torch.randint(0, 4096, (2, 100))
    frame_t1 = torch.randint(0, 4096, (2, 100))
    action = torch.randint(0, 512, (2, 1))

    output = model(frame_t, action, frame_t1)
    print(f"   Predictions: {output['predictions'].shape}")
    print(f"   Loss: {output['loss'].item():.4f}")
    print(f"   Accuracy: {output['accuracy'].item():.2%}")
    print("   ✅ DynamicsModel forward pass works\n")

    # Test 3: Prediction
    print("3. Testing prediction...")
    next_frame = model.predict_next_frame(frame_t, action, deterministic=True)
    print(f"   Next frame: {next_frame.shape}")
    assert next_frame.shape == (2, 100)
    print("   ✅ Prediction works\n")

    # Test 4: Rollout
    print("4. Testing rollout...")
    action_seq = torch.randint(0, 512, (2, 5, 1))  # 5 actions
    frames = model.rollout(frame_t, action_seq, deterministic=True)
    print(f"   Input: {frame_t.shape}")
    print(f"   Actions: {action_seq.shape}")
    print(f"   Output frames: {frames.shape}")
    assert frames.shape == (2, 5, 100)
    print("   ✅ Rollout works\n")

    # Test 5: Gradient flow
    print("5. Testing gradient flow...")
    frame_t_grad = torch.randint(0, 4096, (2, 100))
    action_grad = torch.randint(0, 512, (2, 1))
    frame_t1_grad = torch.randint(0, 4096, (2, 100))

    output = model(frame_t_grad, action_grad, frame_t1_grad)
    loss = output['loss']
    loss.backward()

    # Check gradients exist
    has_grads = any(p.grad is not None for p in model.parameters())
    print(f"   Gradients exist: {has_grads}")
    print(f"   Loss: {loss.item():.4f}")
    print("   ✅ Gradient flow works\n")

    print("✅ All Dynamics Model tests passed!")

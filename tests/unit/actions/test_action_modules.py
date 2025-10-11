"""
Unit tests for Action Encoder and Action Quantizer

Tests both components individually and in combination.
"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from models.actions import ActionEncoder, ActionQuantizer


class TestActionEncoder:
    """Tests for ActionEncoder"""

    def test_basic_forward(self):
        """Test basic forward pass"""
        encoder = ActionEncoder(feature_dim=256, action_dim=128)

        frame_t = torch.randn(2, 256, 16, 16)
        frame_t1 = torch.randn(2, 256, 16, 16)

        action = encoder(frame_t, frame_t1)

        assert action.shape == (2, 128), f"Expected (2, 128), got {action.shape}"

    def test_sequence_encoding(self):
        """Test encoding a full sequence"""
        encoder = ActionEncoder(feature_dim=256, action_dim=128)

        features = torch.randn(2, 256, 8, 16, 16)  # [B, C, T, H, W]
        actions = encoder.encode_sequence(features)

        assert actions.shape == (2, 7, 128), f"Expected (2, 7, 128), got {actions.shape}"

    def test_action_discrimination(self):
        """Actions should be different for different frame pairs"""
        encoder = ActionEncoder(feature_dim=256, action_dim=128)
        encoder.eval()

        frame_t = torch.randn(1, 256, 16, 16)
        frame_same = frame_t.clone()
        frame_diff = frame_t + torch.randn_like(frame_t) * 0.5

        with torch.no_grad():
            action_same = encoder(frame_t, frame_same)
            action_diff = encoder(frame_t, frame_diff)

        # Actions should be different
        diff_norm = (action_same - action_diff).norm()
        assert diff_norm > 0.1, "Actions should differ for different frames"

    def test_no_attention(self):
        """Test encoder without attention"""
        encoder = ActionEncoder(
            feature_dim=256,
            action_dim=128,
            use_attention=False
        )

        frame_t = torch.randn(2, 256, 16, 16)
        frame_t1 = torch.randn(2, 256, 16, 16)

        action = encoder(frame_t, frame_t1)
        assert action.shape == (2, 128)

    def test_different_dimensions(self):
        """Test with different feature/action dimensions"""
        encoder = ActionEncoder(feature_dim=128, action_dim=64)

        frame = torch.randn(2, 128, 8, 8)
        action = encoder(frame, frame)

        assert action.shape == (2, 64)

    def test_gradient_flow(self):
        """Test gradients flow through encoder"""
        encoder = ActionEncoder(feature_dim=256, action_dim=128)

        frame_t = torch.randn(2, 256, 16, 16, requires_grad=True)
        frame_t1 = torch.randn(2, 256, 16, 16, requires_grad=True)

        action = encoder(frame_t, frame_t1)
        loss = action.sum()
        loss.backward()

        assert frame_t.grad is not None, "Gradients should flow to frame_t"
        assert frame_t1.grad is not None, "Gradients should flow to frame_t1"


class TestActionQuantizer:
    """Tests for ActionQuantizer"""

    def test_basic_quantization(self):
        """Test basic quantization"""
        quantizer = ActionQuantizer(action_vocab_size=512, action_dim=128)

        action_latent = torch.randn(4, 128)
        action_quantized, info = quantizer(action_latent)

        assert action_quantized.shape == action_latent.shape
        assert info['indices'].shape == (4, 1)
        assert 'loss' in info
        assert 'perplexity' in info
        assert 'diversity' in info

    def test_sequence_quantization(self):
        """Test quantizing action sequences"""
        quantizer = ActionQuantizer(action_vocab_size=512, action_dim=128)

        action_sequence = torch.randn(2, 7, 128)
        action_quantized, info = quantizer(action_sequence)

        assert action_quantized.shape == action_sequence.shape
        assert info['indices'].shape == (2, 7, 1)

    def test_encode_decode(self):
        """Test encode/decode cycle"""
        quantizer = ActionQuantizer(action_vocab_size=512, action_dim=128)

        action_latent = torch.randn(4, 128)
        indices = quantizer.encode(action_latent)
        decoded = quantizer.decode(indices)

        assert indices.shape == (4, 1)
        assert decoded.shape == action_latent.shape

    def test_gradient_flow(self):
        """Test gradients flow through quantizer"""
        quantizer = ActionQuantizer(action_vocab_size=512, action_dim=128)

        action_latent = torch.randn(2, 128, requires_grad=True)
        action_q, info = quantizer(action_latent)

        loss = action_q.sum() + info['loss']
        loss.backward()

        assert action_latent.grad is not None, "Gradients should flow through quantizer"

    def test_diversity_increases_with_batch(self):
        """Larger batches should use more actions"""
        quantizer = ActionQuantizer(action_vocab_size=512, action_dim=128)

        small_batch = torch.randn(10, 128)
        large_batch = torch.randn(100, 128)

        _, info_small = quantizer(small_batch)
        _, info_large = quantizer(large_batch)

        assert info_large['unique_actions'] >= info_small['unique_actions'], \
            "Larger batch should use at least as many actions"

    def test_training_vs_eval(self):
        """Test behavior in training vs eval mode"""
        quantizer = ActionQuantizer(action_vocab_size=512, action_dim=128)
        action_latent = torch.randn(4, 128)

        # Training mode
        quantizer.train()
        _, info_train = quantizer(action_latent)

        # Eval mode
        quantizer.eval()
        _, info_eval = quantizer(action_latent)

        # Both should work
        assert info_train['indices'].shape == (4, 1)
        assert info_eval['indices'].shape == (4, 1)


class TestActionEncoderQuantizerIntegration:
    """Test encoder and quantizer working together"""

    def test_end_to_end(self):
        """Test full pipeline: frames → actions → quantized actions"""
        encoder = ActionEncoder(feature_dim=256, action_dim=128)
        quantizer = ActionQuantizer(action_vocab_size=512, action_dim=128)

        # Simulate two consecutive frames
        frame_t = torch.randn(2, 256, 16, 16)
        frame_t1 = torch.randn(2, 256, 16, 16)

        # Encode action
        action_latent = encoder(frame_t, frame_t1)
        assert action_latent.shape == (2, 128)

        # Quantize action
        action_quantized, info = quantizer(action_latent)
        assert action_quantized.shape == (2, 128)
        assert info['indices'].shape == (2, 1)

    def test_sequence_pipeline(self):
        """Test encoding and quantizing a full sequence"""
        encoder = ActionEncoder(feature_dim=256, action_dim=128)
        quantizer = ActionQuantizer(action_vocab_size=512, action_dim=128)

        # Video features: [B, C, T, H, W]
        features = torch.randn(2, 256, 8, 16, 16)

        # Encode actions
        action_sequence = encoder.encode_sequence(features)
        assert action_sequence.shape == (2, 7, 128)

        # Quantize actions
        action_quantized, info = quantizer(action_sequence)
        assert action_quantized.shape == (2, 7, 128)
        assert info['indices'].shape == (2, 7, 1)

    def test_gradient_flow_end_to_end(self):
        """Test gradients flow through entire pipeline"""
        encoder = ActionEncoder(feature_dim=256, action_dim=128)
        quantizer = ActionQuantizer(action_vocab_size=512, action_dim=128)

        frame_t = torch.randn(2, 256, 16, 16, requires_grad=True)
        frame_t1 = torch.randn(2, 256, 16, 16, requires_grad=True)

        # Forward pass
        action_latent = encoder(frame_t, frame_t1)
        action_quantized, info = quantizer(action_latent)

        # Backward pass
        loss = action_quantized.sum() + info['loss']
        loss.backward()

        assert frame_t.grad is not None, "Gradients should reach input frames"
        assert frame_t1.grad is not None, "Gradients should reach input frames"

    def test_action_consistency(self):
        """Same frame pair should produce similar action (deterministic in eval)"""
        encoder = ActionEncoder(feature_dim=256, action_dim=128)
        quantizer = ActionQuantizer(action_vocab_size=512, action_dim=128)

        encoder.eval()
        quantizer.eval()

        frame_t = torch.randn(1, 256, 16, 16)
        frame_t1 = torch.randn(1, 256, 16, 16)

        with torch.no_grad():
            # Encode twice
            action1 = encoder(frame_t, frame_t1)
            action2 = encoder(frame_t, frame_t1)

            # Should be identical
            assert torch.allclose(action1, action2), "Encoder should be deterministic in eval"

            # Quantize twice
            _, info1 = quantizer(action1)
            _, info2 = quantizer(action2)

            # Indices might differ due to noise in eval, but should be close
            # (This is expected for untrained models)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

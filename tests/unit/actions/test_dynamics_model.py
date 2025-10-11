"""
Unit tests for Dynamics Model

Tests the action-conditioned transformer that predicts next frames from current frames + actions.
"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from models.actions import DynamicsModel


class TestPositionalEncoding:
    """Tests for PositionalEncoding component"""

    def test_basic_encoding(self):
        """Test basic positional encoding"""
        from models.actions.dynamics_model import PositionalEncoding

        pe = PositionalEncoding(d_model=512, max_len=100)
        x = torch.randn(2, 50, 512)
        output = pe(x)

        assert output.shape == x.shape
        assert not torch.allclose(output, x), "Should add positional information"

    def test_different_sequence_lengths(self):
        """Test with different sequence lengths"""
        from models.actions.dynamics_model import PositionalEncoding

        pe = PositionalEncoding(d_model=256, max_len=200)

        x_short = torch.randn(1, 10, 256)
        x_long = torch.randn(1, 100, 256)

        out_short = pe(x_short)
        out_long = pe(x_long)

        assert out_short.shape == x_short.shape
        assert out_long.shape == x_long.shape


class TestActionConditionedTransformer:
    """Tests for ActionConditionedTransformer"""

    def test_basic_forward(self):
        """Test basic forward pass"""
        from models.actions.dynamics_model import ActionConditionedTransformer

        model = ActionConditionedTransformer(
            frame_vocab_size=4096,
            action_vocab_size=512,
            d_model=256,
            nhead=4,
            num_layers=2
        )

        frame_tokens = torch.randint(0, 4096, (2, 100))
        action_token = torch.randint(0, 512, (2, 1))

        logits = model(frame_tokens, action_token)

        assert logits.shape == (2, 100, 4096)

    def test_return_predictions(self):
        """Test returning predictions instead of logits"""
        from models.actions.dynamics_model import ActionConditionedTransformer

        model = ActionConditionedTransformer(
            frame_vocab_size=4096,
            action_vocab_size=512,
            d_model=128,
            nhead=4,
            num_layers=2
        )

        frame_tokens = torch.randint(0, 4096, (2, 50))
        action_token = torch.randint(0, 512, (2, 1))

        predictions = model(frame_tokens, action_token, return_logits=False)

        assert predictions.shape == (2, 50)
        assert predictions.dtype == torch.long

    def test_generate_with_temperature(self):
        """Test generation with temperature sampling"""
        from models.actions.dynamics_model import ActionConditionedTransformer

        model = ActionConditionedTransformer(
            frame_vocab_size=512,
            action_vocab_size=128,
            d_model=128,
            nhead=4,
            num_layers=2
        )
        model.eval()

        frame_tokens = torch.randint(0, 512, (1, 50))
        action_token = torch.randint(0, 128, (1, 1))

        with torch.no_grad():
            # Temperature should affect diversity
            tokens_low_temp = model.generate(frame_tokens, action_token, temperature=0.5)
            tokens_high_temp = model.generate(frame_tokens, action_token, temperature=2.0)

        assert tokens_low_temp.shape == (1, 50)
        assert tokens_high_temp.shape == (1, 50)

    def test_top_k_filtering(self):
        """Test top-k sampling"""
        from models.actions.dynamics_model import ActionConditionedTransformer

        model = ActionConditionedTransformer(
            frame_vocab_size=512,
            action_vocab_size=128,
            d_model=128,
            nhead=4,
            num_layers=2
        )
        model.eval()

        frame_tokens = torch.randint(0, 512, (1, 20))
        action_token = torch.randint(0, 128, (1, 1))

        with torch.no_grad():
            tokens = model.generate(frame_tokens, action_token, temperature=1.0, top_k=50)

        assert tokens.shape == (1, 20)

    def test_different_vocab_sizes(self):
        """Test with different vocabulary sizes"""
        from models.actions.dynamics_model import ActionConditionedTransformer

        model = ActionConditionedTransformer(
            frame_vocab_size=8192,
            action_vocab_size=1024,
            d_model=256,
            nhead=8,
            num_layers=3
        )

        frame_tokens = torch.randint(0, 8192, (2, 200))
        action_token = torch.randint(0, 1024, (2, 1))

        logits = model(frame_tokens, action_token)
        assert logits.shape == (2, 200, 8192)

    def test_gradient_flow(self):
        """Test gradients flow through transformer"""
        from models.actions.dynamics_model import ActionConditionedTransformer

        model = ActionConditionedTransformer(
            frame_vocab_size=512,
            action_vocab_size=128,
            d_model=128,
            nhead=4,
            num_layers=2
        )

        frame_tokens = torch.randint(0, 512, (2, 50))
        action_token = torch.randint(0, 128, (2, 1))
        target = torch.randint(0, 512, (2, 50))

        logits = model(frame_tokens, action_token)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, 512),
            target.reshape(-1)
        )
        loss.backward()

        # Check gradients exist
        has_grads = any(p.grad is not None for p in model.parameters())
        assert has_grads, "Gradients should flow through transformer"


class TestDynamicsModel:
    """Tests for high-level DynamicsModel wrapper"""

    def test_forward_with_targets(self):
        """Test forward pass with target frames (training mode)"""
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

        assert 'logits' in output
        assert 'predictions' in output
        assert 'loss' in output
        assert 'accuracy' in output

        assert output['logits'].shape == (2, 100, 4096)
        assert output['predictions'].shape == (2, 100)
        assert output['loss'].ndim == 0  # Scalar
        assert output['accuracy'].ndim == 0  # Scalar

    def test_forward_without_targets(self):
        """Test forward pass without targets (inference mode)"""
        model = DynamicsModel(
            frame_vocab_size=4096,
            action_vocab_size=512,
            d_model=256,
            nhead=4,
            num_layers=2
        )

        frame_t = torch.randint(0, 4096, (2, 100))
        action = torch.randint(0, 512, (2, 1))

        output = model(frame_t, action)

        assert 'logits' in output
        assert 'predictions' in output
        assert 'loss' not in output  # No loss without targets
        assert 'accuracy' not in output

    def test_predict_next_frame_deterministic(self):
        """Test deterministic prediction"""
        model = DynamicsModel(
            frame_vocab_size=512,
            action_vocab_size=128,
            d_model=128,
            nhead=4,
            num_layers=2
        )
        model.eval()

        frame_t = torch.randint(0, 512, (2, 50))
        action = torch.randint(0, 128, (2, 1))

        with torch.no_grad():
            next_frame = model.predict_next_frame(frame_t, action, deterministic=True)

        assert next_frame.shape == (2, 50)
        assert next_frame.dtype == torch.long

    def test_predict_next_frame_stochastic(self):
        """Test stochastic prediction with temperature"""
        model = DynamicsModel(
            frame_vocab_size=512,
            action_vocab_size=128,
            d_model=128,
            nhead=4,
            num_layers=2
        )
        model.eval()

        frame_t = torch.randint(0, 512, (2, 50))
        action = torch.randint(0, 128, (2, 1))

        with torch.no_grad():
            next_frame = model.predict_next_frame(
                frame_t, action,
                temperature=1.0,
                deterministic=False
            )

        assert next_frame.shape == (2, 50)

    def test_rollout(self):
        """Test rollout with action sequence"""
        model = DynamicsModel(
            frame_vocab_size=512,
            action_vocab_size=128,
            d_model=128,
            nhead=4,
            num_layers=2
        )
        model.eval()

        initial_frame = torch.randint(0, 512, (2, 50))
        action_sequence = torch.randint(0, 128, (2, 10, 1))  # 10 actions

        with torch.no_grad():
            frames = model.rollout(initial_frame, action_sequence, deterministic=True)

        assert frames.shape == (2, 10, 50)
        assert frames.dtype == torch.long

    def test_rollout_consistency(self):
        """Test rollout produces consistent results in eval mode"""
        model = DynamicsModel(
            frame_vocab_size=256,
            action_vocab_size=64,
            d_model=128,
            nhead=4,
            num_layers=2
        )
        model.eval()

        initial_frame = torch.randint(0, 256, (1, 30))
        action_sequence = torch.randint(0, 64, (1, 5, 1))

        with torch.no_grad():
            frames1 = model.rollout(initial_frame, action_sequence, deterministic=True)
            frames2 = model.rollout(initial_frame, action_sequence, deterministic=True)

        # Should be identical in deterministic mode
        assert torch.allclose(frames1.float(), frames2.float())

    def test_gradient_flow_end_to_end(self):
        """Test gradients flow through entire model"""
        model = DynamicsModel(
            frame_vocab_size=512,
            action_vocab_size=128,
            d_model=128,
            nhead=4,
            num_layers=2
        )

        frame_t = torch.randint(0, 512, (2, 50))
        frame_t1 = torch.randint(0, 512, (2, 50))
        action = torch.randint(0, 128, (2, 1))

        output = model(frame_t, action, frame_t1)
        loss = output['loss']
        loss.backward()

        # Check gradients exist
        has_grads = any(p.grad is not None for p in model.parameters())
        assert has_grads, "Gradients should flow through model"

    def test_batch_processing(self):
        """Test model handles different batch sizes"""
        model = DynamicsModel(
            frame_vocab_size=512,
            action_vocab_size=128,
            d_model=128,
            nhead=4,
            num_layers=2
        )
        model.eval()

        for batch_size in [1, 2, 4, 8]:
            frame = torch.randint(0, 512, (batch_size, 50))
            action = torch.randint(0, 128, (batch_size, 1))

            with torch.no_grad():
                output = model(frame, action)

            assert output['predictions'].shape == (batch_size, 50)


class TestDynamicsModelIntegration:
    """Integration tests with action encoder and quantizer"""

    def test_integration_with_action_modules(self):
        """Test dynamics model works with action encoder output"""
        from models.actions import ActionEncoder, ActionQuantizer

        # Create components
        action_encoder = ActionEncoder(feature_dim=256, action_dim=128)
        action_quantizer = ActionQuantizer(action_vocab_size=512, action_dim=128)
        dynamics_model = DynamicsModel(
            frame_vocab_size=4096,
            action_vocab_size=512,
            d_model=256,
            nhead=4,
            num_layers=2
        )

        # Simulate Phase 1 features
        features = torch.randn(2, 256, 4, 16, 16)  # [B, C, T, H, W]

        # Extract actions
        action_latents = action_encoder.encode_sequence(features)
        action_quantized, info = action_quantizer(action_latents)
        action_indices = info['indices']  # [B, T-1, 1]

        # Simulate frame tokens
        frame_tokens = torch.randint(0, 4096, (2, 100))

        # Use first action
        action = action_indices[:, 0, :]  # [B, 1]

        # Predict next frame
        output = dynamics_model(frame_tokens, action)

        assert output['predictions'].shape == (2, 100)

    def test_complete_pipeline(self):
        """Test complete pipeline: features → actions → dynamics prediction"""
        from models.actions import ActionEncoder, ActionQuantizer

        encoder = ActionEncoder(feature_dim=256, action_dim=128)
        quantizer = ActionQuantizer(action_vocab_size=512, action_dim=128)
        dynamics = DynamicsModel(
            frame_vocab_size=4096,
            action_vocab_size=512,
            d_model=256,
            nhead=4,
            num_layers=2
        )

        encoder.eval()
        quantizer.eval()
        dynamics.eval()

        # Generate features
        features = torch.randn(1, 256, 8, 16, 16)

        # Extract and quantize actions
        with torch.no_grad():
            actions = encoder.encode_sequence(features)
            _, info = quantizer(actions)
            action_sequence = info['indices']  # [1, 7, 1]

        # Simulate initial frame tokens
        initial_tokens = torch.randint(0, 4096, (1, 100))

        # Rollout
        with torch.no_grad():
            predicted_frames = dynamics.rollout(
                initial_tokens,
                action_sequence,
                deterministic=True
            )

        assert predicted_frames.shape == (1, 7, 100)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

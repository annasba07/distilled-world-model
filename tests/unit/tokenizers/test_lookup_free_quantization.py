"""
Unit tests for Lookup-Free Quantization (LFQ)

Tests cover:
- Forward pass correctness
- Gradient flow verification
- Codebook usage tracking
- No collapse behavior
- Encode/decode consistency
"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from models.tokenizers.lookup_free_quantization import LookupFreeQuantizer, MultiScaleLFQ


class TestLookupFreeQuantizer:
    """Test suite for LookupFreeQuantizer"""

    @pytest.fixture
    def quantizer(self):
        """Create a standard quantizer for testing"""
        return LookupFreeQuantizer(
            codebook_size=256,
            embedding_dim=128,
            commitment_cost=0.25,
            temperature=1.0
        )

    @pytest.fixture
    def large_quantizer(self):
        """Create a large-scale quantizer"""
        return LookupFreeQuantizer(
            codebook_size=2**16,
            embedding_dim=512,
            commitment_cost=0.25
        )

    def test_initialization(self, quantizer):
        """Test that quantizer initializes correctly"""
        assert quantizer.codebook_size == 256
        assert quantizer.embedding_dim == 128
        assert quantizer.num_bits == 8  # log2(256)
        assert isinstance(quantizer.project_in, nn.Linear)
        assert isinstance(quantizer.project_out, nn.Linear)

    def test_forward_pass_shape(self, quantizer):
        """Test that forward pass maintains correct shapes"""
        # Test with 4D input (typical for images)
        z = torch.randn(4, 16, 16, 128)
        z_quantized, info = quantizer(z)

        assert z_quantized.shape == z.shape, "Output shape should match input"
        assert info['indices'].shape == (4, 16, 16, 1), "Indices shape incorrect"
        assert 'loss' in info, "Loss should be in output"
        assert 'perplexity' in info, "Perplexity should be in output"
        assert 'codebook_usage' in info, "Codebook usage should be in output"

    def test_forward_pass_3d(self, quantizer):
        """Test forward pass with 3D input (spatial-only)"""
        z = torch.randn(2, 32, 128)
        z_quantized, info = quantizer(z)

        assert z_quantized.shape == z.shape
        assert info['indices'].shape == (2, 32, 1)

    def test_forward_pass_5d(self, quantizer):
        """Test forward pass with 5D input (video)"""
        z = torch.randn(2, 8, 16, 16, 128)  # [B, T, H, W, D]
        z_quantized, info = quantizer(z)

        assert z_quantized.shape == z.shape
        assert info['indices'].shape == (2, 8, 16, 16, 1)

    def test_gradient_flow(self, quantizer):
        """Test that gradients flow through quantization"""
        z = torch.randn(2, 8, 8, 128, requires_grad=True)
        z_quantized, info = quantizer(z)

        # Backpropagate
        loss = z_quantized.sum()
        loss.backward()

        assert z.grad is not None, "Gradients should exist"
        assert z.grad.norm() > 0, "Gradients should be non-zero"
        assert not torch.isnan(z.grad).any(), "No NaN gradients"

    def test_commitment_loss(self, quantizer):
        """Test that commitment loss is computed correctly"""
        z = torch.randn(4, 16, 16, 128)
        _, info = quantizer(z)

        loss = info['loss']
        assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
        assert loss.dim() == 0, "Loss should be a scalar"
        assert loss >= 0, "Commitment loss should be non-negative"

    def test_perplexity(self, quantizer):
        """Test perplexity computation"""
        z = torch.randn(4, 16, 16, 128)
        _, info = quantizer(z)

        perplexity = info['perplexity']
        assert perplexity > 0, "Perplexity should be positive"
        assert perplexity <= quantizer.codebook_size, "Perplexity bounded by codebook size"

    def test_codebook_usage(self, quantizer):
        """Test that codebook usage is tracked correctly"""
        z = torch.randn(4, 32, 32, 128)
        _, info = quantizer(z)

        usage = info['codebook_usage']
        assert 0 <= usage <= 1, "Usage should be between 0 and 1"

    def test_no_codebook_collapse(self, quantizer):
        """Test that codebook doesn't collapse (uses multiple codes)"""
        # Train for a few steps
        quantizer.train()
        optimizer = torch.optim.Adam(quantizer.parameters(), lr=1e-3)

        for _ in range(10):
            z = torch.randn(8, 16, 16, 128)
            z_quantized, info = quantizer(z)
            loss = info['loss']

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Check final usage
        z = torch.randn(8, 16, 16, 128)
        _, info = quantizer(z)

        # Should use at least 10% of codebook after training
        assert info['codebook_usage'] > 0.1, "Codebook collapse detected!"

    def test_encode_decode_consistency(self, quantizer):
        """Test that encode/decode round-trip is consistent"""
        quantizer.eval()

        z = torch.randn(2, 8, 8, 128)

        # Encode to indices
        indices = quantizer.encode(z)
        assert indices.shape == (2, 8, 8, 1)

        # Decode back
        z_decoded = quantizer.decode(indices)
        assert z_decoded.shape == z.shape

        # Should be deterministic
        indices2 = quantizer.encode(z)
        assert torch.equal(indices, indices2), "Encoding should be deterministic"

    def test_training_vs_eval_mode(self, quantizer):
        """Test behavior difference between training and eval"""
        z = torch.randn(4, 16, 16, 128)

        # Training mode
        quantizer.train()
        z_train, info_train = quantizer(z)

        # Eval mode
        quantizer.eval()
        with torch.no_grad():
            z_eval, info_eval = quantizer(z)

        # Outputs might differ due to Gumbel noise
        # But shapes should match
        assert z_train.shape == z_eval.shape
        assert info_train['indices'].shape == info_eval['indices'].shape

    def test_large_scale_quantizer(self, large_quantizer):
        """Test with large codebook (2^16 codes)"""
        z = torch.randn(2, 16, 16, 512)
        z_quantized, info = large_quantizer(z)

        assert z_quantized.shape == z.shape
        assert large_quantizer.codebook_size == 65536
        assert large_quantizer.num_bits == 16

    def test_batch_independence(self, quantizer):
        """Test that batch items are processed independently"""
        z1 = torch.randn(1, 8, 8, 128)
        z2 = torch.randn(1, 8, 8, 128)
        z_batch = torch.cat([z1, z2], dim=0)

        quantizer.eval()
        with torch.no_grad():
            _, info1 = quantizer(z1)
            _, info2 = quantizer(z2)
            _, info_batch = quantizer(z_batch)

        # Indices should match when processed separately vs together
        assert torch.equal(info_batch['indices'][0], info1['indices'][0])
        assert torch.equal(info_batch['indices'][1], info2['indices'][0])


class TestMultiScaleLFQ:
    """Test suite for Multi-Scale LFQ"""

    @pytest.fixture
    def multi_scale_quantizer(self):
        """Create multi-scale quantizer"""
        return MultiScaleLFQ(
            codebook_sizes=[2**8, 2**12, 2**16],
            embedding_dim=256,
            commitment_cost=0.25
        )

    def test_initialization(self, multi_scale_quantizer):
        """Test multi-scale quantizer initialization"""
        assert multi_scale_quantizer.num_scales == 3
        assert len(multi_scale_quantizer.quantizers) == 3
        assert multi_scale_quantizer.scale_weights.shape == (3,)

    def test_forward_pass(self, multi_scale_quantizer):
        """Test multi-scale forward pass"""
        z = torch.randn(4, 16, 16, 256)
        z_quantized, info = multi_scale_quantizer(z)

        assert z_quantized.shape == z.shape
        assert 'indices' in info
        assert 'loss' in info
        assert 'perplexity' in info
        assert 'scale_weights' in info

    def test_scale_weights_normalized(self, multi_scale_quantizer):
        """Test that scale weights are normalized"""
        z = torch.randn(2, 8, 8, 256)
        _, info = multi_scale_quantizer(z)

        weights = info['scale_weights']
        assert torch.allclose(weights.sum(), torch.tensor(1.0)), "Weights should sum to 1"
        assert (weights >= 0).all(), "Weights should be non-negative"

    def test_gradient_flow(self, multi_scale_quantizer):
        """Test gradient flow through multi-scale quantizer"""
        z = torch.randn(2, 8, 8, 256, requires_grad=True)
        z_quantized, _ = multi_scale_quantizer(z)

        loss = z_quantized.sum()
        loss.backward()

        assert z.grad is not None
        assert not torch.isnan(z.grad).any()


class TestLFQProperties:
    """Test mathematical properties of LFQ"""

    def test_commitment_loss_decreases_distance(self):
        """Test that commitment loss encourages z to match quantized output"""
        quantizer = LookupFreeQuantizer(codebook_size=256, embedding_dim=128)
        quantizer.train()

        z = torch.randn(4, 16, 16, 128, requires_grad=True)
        optimizer = torch.optim.Adam([z], lr=0.1)

        # Initial distance
        z_q, _ = quantizer(z)
        initial_dist = (z - z_q.detach()).pow(2).mean()

        # Optimize z only (not quantizer)
        for _ in range(5):
            optimizer.zero_grad()
            z_q, info = quantizer(z)
            loss = info['loss']
            loss.backward()
            optimizer.step()

        # Final distance
        z_q, _ = quantizer(z)
        final_dist = (z - z_q.detach()).pow(2).mean()

        assert final_dist < initial_dist, "Commitment loss should reduce distance"

    def test_indices_in_valid_range(self):
        """Test that all indices are valid (within codebook size)"""
        quantizer = LookupFreeQuantizer(codebook_size=256, embedding_dim=128)

        z = torch.randn(8, 32, 32, 128)
        _, info = quantizer(z)

        indices = info['indices']
        assert (indices >= 0).all(), "Indices should be non-negative"
        assert (indices < 256).all(), "Indices should be < codebook_size"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])

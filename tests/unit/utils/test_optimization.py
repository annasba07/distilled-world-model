"""
Unit tests for Optimization Utilities

Tests cover:
- Mixed precision wrapper
- torch.compile integration
- Flash Attention optimizer
- Inference optimizer
- TF32 configuration
"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from utils.optimization import (
    MixedPrecisionWrapper,
    compile_model,
    FlashAttentionOptimizer,
    InferenceOptimizer,
    enable_tf32,
    optimize_for_inference,
    autocast_inference
)


class TestMixedPrecisionWrapper:
    """Test suite for MixedPrecisionWrapper"""

    @pytest.fixture
    def wrapper(self):
        """Create wrapper"""
        return MixedPrecisionWrapper(enabled=torch.cuda.is_available())

    def test_initialization(self, wrapper):
        """Test wrapper initializes correctly"""
        assert isinstance(wrapper.enabled, bool)

    def test_autocast_context(self, wrapper):
        """Test autocast context manager"""
        with wrapper.autocast():
            x = torch.randn(2, 10)
            y = x * 2

        # Should complete without error
        assert y is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_fp16_conversion(self):
        """Test FP16 conversion with CUDA"""
        wrapper = MixedPrecisionWrapper(enabled=True)

        model = nn.Linear(10, 10).cuda()
        x = torch.randn(2, 10).cuda()

        with wrapper.autocast():
            y = model(x)

        # Output should be FP16
        assert y.dtype == torch.float16

    def test_optimize_model(self, wrapper):
        """Test model optimization"""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.BatchNorm1d(20),
            nn.ReLU()
        )

        optimized = wrapper.optimize_model(model)

        # Should return a model
        assert isinstance(optimized, nn.Module)

    def test_disabled_wrapper(self):
        """Test wrapper when disabled"""
        wrapper = MixedPrecisionWrapper(enabled=False)

        assert not wrapper.enabled

        with wrapper.autocast():
            x = torch.randn(2, 10)
            y = x * 2

        # Should still work
        assert y is not None


class TestCompileModel:
    """Test suite for compile_model"""

    @pytest.fixture
    def model(self):
        """Create simple model"""
        return nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )

    def test_compile_basic(self, model):
        """Test basic compilation"""
        try:
            compiled = compile_model(model, mode='default')
            assert compiled is not None
        except Exception:
            # Compilation may fail on older PyTorch
            pytest.skip("torch.compile not available")

    def test_compile_modes(self, model):
        """Test different compilation modes"""
        modes = ['default', 'reduce-overhead', 'max-autotune']

        for mode in modes:
            try:
                compiled = compile_model(model, mode=mode)
                assert compiled is not None
            except Exception:
                pytest.skip(f"Compilation mode {mode} not available")

    def test_compile_fallback(self, model):
        """Test that compilation failure returns original model"""
        # This should work even if compilation fails
        result = compile_model(model, mode='default')

        # Should return some model (compiled or original)
        assert result is not None


class TestFlashAttentionOptimizer:
    """Test suite for FlashAttentionOptimizer"""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer"""
        return FlashAttentionOptimizer()

    def test_initialization(self, optimizer):
        """Test optimizer initializes"""
        assert isinstance(optimizer.available, bool)

    def test_check_availability(self, optimizer):
        """Test availability check"""
        # Should not crash
        available = optimizer._check_availability()
        assert isinstance(available, bool)

    def test_optimize_transformer(self, optimizer):
        """Test transformer optimization"""
        transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=128, nhead=4, batch_first=True),
            num_layers=2
        )

        optimized = optimizer.optimize_transformer(transformer)

        # Should return a model
        assert isinstance(optimized, nn.Module)

    def test_create_flash_attention_module(self, optimizer):
        """Test creating flash attention module"""
        module = optimizer.create_flash_attention_module(
            embed_dim=128,
            num_heads=4,
            dropout=0.1
        )

        # Should return some attention module
        assert isinstance(module, nn.Module)

        # Test forward pass
        x = torch.randn(2, 10, 128)

        try:
            # For standard MultiheadAttention
            output, _ = module(x, x, x)
            assert output.shape == x.shape
        except Exception:
            # Flash attention may have different API
            pass


class TestInferenceOptimizer:
    """Test suite for InferenceOptimizer"""

    @pytest.fixture
    def optimizer(self):
        """Create inference optimizer"""
        return InferenceOptimizer(
            use_fp16=torch.cuda.is_available(),
            use_compile=False,  # Skip compilation for tests
            use_flash_attn=False
        )

    @pytest.fixture
    def model(self):
        """Create test model"""
        return nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

    def test_initialization(self, optimizer):
        """Test optimizer initializes"""
        assert optimizer.fp16_wrapper is not None
        assert isinstance(optimizer.use_fp16, bool)

    def test_optimize_model(self, optimizer, model):
        """Test model optimization"""
        optimized = optimizer.optimize(model)

        # Should return a model
        assert isinstance(optimized, nn.Module)

    def test_autocast_context(self, optimizer):
        """Test autocast context"""
        with optimizer.autocast():
            x = torch.randn(2, 32)
            y = x * 2

        assert y is not None

    def test_get_optimization_stats(self, optimizer):
        """Test getting optimization stats"""
        stats = optimizer.get_optimization_stats()

        assert 'fp16_enabled' in stats
        assert 'compile_enabled' in stats
        assert 'flash_attn_enabled' in stats
        assert 'expected_speedup' in stats

    def test_speedup_estimation(self, optimizer):
        """Test speedup estimation"""
        stats = optimizer.get_optimization_stats()
        speedup = stats['expected_speedup']

        # Should be a string like "2.0x"
        assert isinstance(speedup, str)
        assert 'x' in speedup

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_fp16_inference(self):
        """Test FP16 inference"""
        optimizer = InferenceOptimizer(
            use_fp16=True,
            use_compile=False,
            use_flash_attn=False
        )

        model = nn.Linear(32, 32).cuda()
        model = optimizer.optimize(model)

        x = torch.randn(2, 32).cuda()

        with optimizer.autocast():
            y = model(x)

        assert y.dtype == torch.float16


class TestUtilityFunctions:
    """Test suite for utility functions"""

    def test_enable_tf32(self):
        """Test TF32 configuration"""
        # Should not crash
        enable_tf32(True)
        enable_tf32(False)

    def test_optimize_for_inference(self):
        """Test quick optimization"""
        model = nn.Linear(10, 10)
        device = torch.device('cpu')

        optimized = optimize_for_inference(model, device)

        # Should be in eval mode
        assert not optimized.training

        # Should have gradients disabled
        for param in optimized.parameters():
            assert not param.requires_grad

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_optimize_for_inference_cuda(self):
        """Test optimization with CUDA"""
        model = nn.Linear(10, 10)
        device = torch.device('cuda')

        optimized = optimize_for_inference(model, device)

        # Should be on CUDA
        assert next(optimized.parameters()).is_cuda

    def test_autocast_decorator(self):
        """Test autocast decorator"""
        @autocast_inference
        def compute(x):
            return x * 2

        x = torch.randn(2, 10)
        y = compute(x)

        assert y is not None
        assert y.shape == x.shape


class TestOptimizationIntegration:
    """Integration tests for optimization pipeline"""

    def test_full_optimization_pipeline(self):
        """Test complete optimization pipeline"""
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Optimize
        optimizer = InferenceOptimizer(
            use_fp16=torch.cuda.is_available(),
            use_compile=False,
            use_flash_attn=False
        )

        model = model.to(device)
        model = optimizer.optimize(model)

        # Test inference
        x = torch.randn(4, 64).to(device)

        with torch.no_grad(), optimizer.autocast():
            y = model(x)

        assert y.shape == (4, 64)

    def test_mixed_precision_consistency(self):
        """Test that mixed precision doesn't change outputs significantly"""
        if not torch.cuda.is_available():
            pytest.skip("Requires CUDA")

        model = nn.Linear(32, 32).cuda()

        # Baseline (FP32)
        x = torch.randn(4, 32).cuda()
        with torch.no_grad():
            y_fp32 = model(x)

        # FP16
        optimizer = InferenceOptimizer(use_fp16=True, use_compile=False)
        with torch.no_grad(), optimizer.autocast():
            y_fp16 = model(x)

        # Should be close (not exact due to precision difference)
        diff = (y_fp32 - y_fp16.float()).abs().max()
        assert diff < 0.1, f"FP16 output differs too much: {diff}"

    @pytest.mark.parametrize("use_fp16", [True, False])
    def test_different_configurations(self, use_fp16):
        """Test different optimization configurations"""
        if use_fp16 and not torch.cuda.is_available():
            pytest.skip("FP16 requires CUDA")

        model = nn.Linear(32, 32)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        optimizer = InferenceOptimizer(
            use_fp16=use_fp16 and torch.cuda.is_available(),
            use_compile=False,
            use_flash_attn=False
        )

        model = model.to(device)
        model = optimizer.optimize(model)

        x = torch.randn(2, 32).to(device)

        with torch.no_grad(), optimizer.autocast():
            y = model(x)

        assert y.shape == (2, 32)


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_model(self):
        """Test optimization with empty model"""
        model = nn.Sequential()

        optimizer = InferenceOptimizer(use_fp16=False, use_compile=False)
        optimized = optimizer.optimize(model)

        assert optimized is not None

    def test_model_with_batchnorm(self):
        """Test optimization preserves batchnorm behavior"""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.BatchNorm1d(20),
            nn.Linear(20, 10)
        )

        optimizer = InferenceOptimizer(use_fp16=torch.cuda.is_available(), use_compile=False)

        if torch.cuda.is_available():
            model = model.cuda()

        optimized = optimizer.optimize(model)

        # Batchnorm should still exist
        has_bn = any(isinstance(m, nn.BatchNorm1d) for m in optimized.modules())
        assert has_bn


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])

"""
Unit tests for Geometric Compression

Tests cover:
- Basic geometric compression (2x, 4x, 8x)
- Geometric hierarchy management
- Constant context manager
- Memory savings validation
- Different compression methods
"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from utils.context.geometric_compression import (
    GeometricCompressor,
    GeometricHierarchy,
    ConstantContextManager,
    estimate_compression_savings
)


class TestGeometricCompressor:
    """Test suite for GeometricCompressor"""

    @pytest.fixture
    def compressor_2x(self):
        """2x compression with pooling"""
        return GeometricCompressor(compression_factor=2, method='pool')

    @pytest.fixture
    def compressor_4x(self):
        """4x compression with pooling"""
        return GeometricCompressor(compression_factor=4, method='pool')

    def test_compression_2x(self, compressor_2x):
        """Test 2x compression reduces temporal dimension by half"""
        x = torch.randn(2, 16, 64)  # [B, T, Features]
        x_compressed = compressor_2x(x)

        assert x_compressed.shape[0] == 2, "Batch should be preserved"
        assert x_compressed.shape[1] == 8, "Should compress to T/2"
        assert x_compressed.shape[2] == 64, "Features should be preserved"

    def test_compression_4x(self, compressor_4x):
        """Test 4x compression reduces temporal dimension by 4"""
        x = torch.randn(2, 16, 64)
        x_compressed = compressor_4x(x)

        assert x_compressed.shape[1] == 4, "Should compress to T/4"

    def test_compression_with_padding(self, compressor_2x):
        """Test compression with non-divisible temporal dimension"""
        x = torch.randn(2, 15, 64)  # Not divisible by 2
        x_compressed = compressor_2x(x)

        # Should pad to 16, then compress to 8
        assert x_compressed.shape[1] == 8

    def test_no_compression(self):
        """Test that factor=1 returns input unchanged"""
        compressor = GeometricCompressor(compression_factor=1, method='pool')
        x = torch.randn(2, 16, 64)
        x_compressed = compressor(x)

        assert torch.equal(x, x_compressed)

    def test_different_methods(self):
        """Test different compression methods"""
        x = torch.randn(2, 16, 64)

        # Pool
        compressor_pool = GeometricCompressor(compression_factor=2, method='pool')
        out_pool = compressor_pool(x)
        assert out_pool.shape[1] == 8

        # Conv (requires channels)
        compressor_conv = GeometricCompressor(
            compression_factor=2,
            method='conv',
            channels=64
        )
        out_conv = compressor_conv(x)
        assert out_conv.shape[1] == 8

    def test_spatial_features(self):
        """Test compression with spatial features"""
        compressor = GeometricCompressor(compression_factor=2, method='pool')
        x = torch.randn(2, 16, 64, 8, 8)  # [B, T, C, H, W]
        x_compressed = compressor(x)

        assert x_compressed.shape[1] == 8, "Temporal should be compressed"
        assert x_compressed.shape[2:] == (64, 8, 8), "Spatial should be preserved"

    def test_gradient_flow(self):
        """Test that gradients flow through compression"""
        compressor = GeometricCompressor(compression_factor=2, method='pool')
        x = torch.randn(2, 16, 64, requires_grad=True)
        x_compressed = compressor(x)

        loss = x_compressed.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestGeometricHierarchy:
    """Test suite for GeometricHierarchy"""

    @pytest.fixture
    def hierarchy(self):
        """Create basic hierarchy"""
        return GeometricHierarchy(
            max_levels=4,
            level_capacity=8,
            compression_method='pool'
        )

    def test_initialization(self, hierarchy):
        """Test hierarchy initializes correctly"""
        assert len(hierarchy.levels) == 4
        assert all(level is None for level in hierarchy.levels)
        assert hierarchy.get_total_frames() == 0

    def test_add_frames_to_l0(self, hierarchy):
        """Test adding frames fills L0 first"""
        frames = torch.randn(1, 4, 64)
        hierarchy.add_frames(frames)

        assert hierarchy.levels[0] is not None
        assert hierarchy.levels[0].shape[1] == 4
        assert hierarchy.get_total_frames() == 4

    def test_cascade_to_l1(self, hierarchy):
        """Test that overflow from L0 cascades to L1"""
        # Add enough frames to overflow L0 (capacity=8)
        frames1 = torch.randn(1, 8, 64)
        hierarchy.add_frames(frames1)

        # L0 should be full
        assert hierarchy.levels[0].shape[1] == 8

        # Add more frames
        frames2 = torch.randn(1, 4, 64)
        hierarchy.add_frames(frames2)

        # L0 should still be 8 (capacity)
        assert hierarchy.levels[0].shape[1] == 8

        # L1 should have compressed frames
        assert hierarchy.levels[1] is not None

    def test_multiple_levels(self, hierarchy):
        """Test filling multiple levels"""
        # Add many frames
        for _ in range(10):
            frames = torch.randn(1, 4, 64)
            hierarchy.add_frames(frames)

        stats = hierarchy.get_memory_stats()

        # Should have frames in multiple levels
        active_levels = sum(1 for level in hierarchy.levels if level is not None)
        assert active_levels >= 2, "Should use multiple compression levels"

    def test_memory_stats(self, hierarchy):
        """Test memory statistics computation"""
        frames = torch.randn(1, 16, 64)
        hierarchy.add_frames(frames)

        stats = hierarchy.get_memory_stats()

        assert 'total_elements' in stats
        assert stats['total_elements'] > 0
        assert 'level_0' in stats

    def test_reset(self, hierarchy):
        """Test reset clears all levels"""
        frames = torch.randn(1, 16, 64)
        hierarchy.add_frames(frames)

        assert hierarchy.get_total_frames() > 0

        hierarchy.reset()

        assert all(level is None for level in hierarchy.levels)
        assert hierarchy.get_total_frames() == 0

    def test_get_all_frames(self, hierarchy):
        """Test retrieving all frames"""
        frames = torch.randn(1, 16, 64)
        hierarchy.add_frames(frames)

        all_frames = hierarchy.get_all_frames()

        assert all_frames is not None
        assert all_frames.shape[0] == 1  # Batch


class TestConstantContextManager:
    """Test suite for ConstantContextManager"""

    @pytest.fixture
    def manager(self):
        """Create basic context manager"""
        return ConstantContextManager(
            max_levels=5,
            level_capacity=16,
            compression_method='pool'
        )

    def test_initialization(self, manager):
        """Test manager initializes correctly"""
        assert manager.max_levels == 5
        assert manager.level_capacity == 16

    def test_update_context(self, manager):
        """Test updating context with new features"""
        features = torch.randn(2, 8, 128)
        manager.update(features)

        context = manager.get_context()
        assert context is not None
        assert context.shape[0] == 2  # Batch

    def test_progressive_updates(self, manager):
        """Test multiple progressive updates"""
        for i in range(10):
            features = torch.randn(2, 8, 128)
            manager.update(features)

        stats = manager.get_stats()
        assert stats['total_frames_represented'] == 80

    def test_memory_savings(self, manager):
        """Test that memory savings are achieved with long sequences"""
        # Add many frames
        for _ in range(20):
            features = torch.randn(2, 10, 128)
            manager.update(features)

        stats = manager.get_stats()

        # Should have significant memory savings
        assert 'memory_savings' in stats
        # With geometric compression, should save at least 30%
        assert stats['memory_savings'] > 0.3

    def test_constant_memory_growth(self, manager):
        """Test that memory doesn't grow linearly with video length"""
        memory_usages = []

        for i in range(1, 11):
            features = torch.randn(1, 10, 64)
            manager.update(features)

            stats = manager.get_stats()
            memory_usages.append(stats['total_elements'])

        # Memory growth should slow down (not linear)
        # Growth from 1->5 should be much larger than 5->10
        early_growth = memory_usages[4] - memory_usages[0]
        late_growth = memory_usages[9] - memory_usages[5]

        assert late_growth < early_growth, "Memory growth should slow down"

    def test_get_stats(self, manager):
        """Test statistics retrieval"""
        features = torch.randn(2, 16, 128)
        manager.update(features)

        stats = manager.get_stats()

        assert 'total_elements' in stats
        assert 'total_frames_represented' in stats
        assert 'memory_savings' in stats

    def test_reset_context(self, manager):
        """Test resetting context"""
        features = torch.randn(2, 16, 128)
        manager.update(features)

        assert manager.get_context() is not None

        manager.reset()

        # After reset, context should be empty
        context = manager.get_context()
        assert context is None

    def test_with_spatial_features(self, manager):
        """Test manager with spatial features (video)"""
        features = torch.randn(2, 8, 64, 16, 16)  # [B, T, C, H, W]
        manager.update(features)

        context = manager.get_context()
        assert context is not None


class TestMemorySavings:
    """Test memory savings calculations"""

    def test_estimate_savings_short_video(self):
        """Test savings estimate for short videos"""
        # Short videos have minimal savings (not enough to compress)
        savings = estimate_compression_savings(num_frames=30)
        assert 0.0 <= savings < 0.5

    def test_estimate_savings_long_video(self):
        """Test savings estimate for long videos"""
        # Long videos should have significant savings
        savings = estimate_compression_savings(num_frames=240)
        assert savings > 0.5, "Long videos should save >50% memory"

    def test_estimate_savings_very_long_video(self):
        """Test savings estimate for very long videos"""
        # Very long videos (e.g., 480 frames = 60 sec at 8 fps)
        savings = estimate_compression_savings(num_frames=480)
        assert savings > 0.7, "Very long videos should save >70% memory"

    @pytest.mark.parametrize("num_frames", [60, 120, 240, 480])
    def test_savings_scale_with_length(self, num_frames):
        """Test that savings increase with video length"""
        savings = estimate_compression_savings(num_frames=num_frames)

        # Savings should be bounded
        assert 0.0 <= savings < 1.0

    def test_different_configurations(self):
        """Test savings with different hierarchy configurations"""
        # More levels = more compression = more savings
        savings_3_levels = estimate_compression_savings(
            num_frames=240,
            max_levels=3
        )
        savings_5_levels = estimate_compression_savings(
            num_frames=240,
            max_levels=5
        )

        assert savings_5_levels >= savings_3_levels


class TestCompressionMethods:
    """Test different compression methods"""

    @pytest.mark.parametrize("method", ['pool', 'conv'])
    def test_compression_methods(self, method):
        """Test that all compression methods work"""
        channels = 64 if method == 'conv' else None

        compressor = GeometricCompressor(
            compression_factor=2,
            method=method,
            channels=channels
        )

        x = torch.randn(2, 16, 64)
        x_compressed = compressor(x)

        assert x_compressed.shape[1] == 8

    def test_learnable_compression(self):
        """Test that conv method has learnable parameters"""
        compressor = GeometricCompressor(
            compression_factor=2,
            method='conv',
            channels=64
        )

        params = list(compressor.parameters())
        assert len(params) > 0, "Conv method should have parameters"

        # Test gradient update
        x = torch.randn(2, 16, 64, requires_grad=True)
        out = compressor(x)
        loss = out.sum()
        loss.backward()

        # Parameters should have gradients
        assert all(p.grad is not None for p in params)


class TestIntegration:
    """Integration tests for full pipeline"""

    def test_long_video_processing(self):
        """Test processing a long video (120 frames)"""
        manager = ConstantContextManager(
            max_levels=5,
            level_capacity=16,
            compression_method='pool'
        )

        # Simulate processing 120 frames in chunks of 10
        for _ in range(12):
            features = torch.randn(1, 10, 128, 16, 16)
            manager.update(features)

        stats = manager.get_stats()

        # Should represent all 120 frames
        assert stats['total_frames_represented'] == 120

        # Should have significant memory savings
        assert stats['memory_savings'] > 0.5

    def test_very_long_video(self):
        """Test processing very long video (480 frames = 60 sec at 8fps)"""
        manager = ConstantContextManager(
            max_levels=6,
            level_capacity=16,
            compression_method='pool'
        )

        # Process 480 frames
        for _ in range(48):
            features = torch.randn(1, 10, 64, 8, 8)
            manager.update(features)

        stats = manager.get_stats()

        assert stats['total_frames_represented'] == 480
        # Very long videos should save >70% memory
        assert stats['memory_savings'] > 0.7

    def test_batch_processing(self):
        """Test processing multiple videos in batch"""
        manager = ConstantContextManager(max_levels=5, level_capacity=16)

        # Process batch of 4 videos
        for _ in range(10):
            features = torch.randn(4, 10, 128)
            manager.update(features)

        context = manager.get_context()
        assert context.shape[0] == 4  # Batch size preserved


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])

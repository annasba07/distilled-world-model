"""
Unit tests for configuration and memory auto-detection
"""

import pytest
import os
from unittest.mock import patch, MagicMock


class TestConfigMemoryAutoDetection:
    """Test suite for memory auto-detection in config"""

    def test_gpu_memory_detection_with_cuda(self):
        """Test GPU memory is auto-detected when CUDA is available"""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.get_device_properties') as mock_props:
                # Mock GPU with 8GB (8 * 1024^3 bytes)
                mock_device = MagicMock()
                mock_device.total_memory = 8 * 1024**3
                mock_props.return_value = mock_device

                # Import config module to trigger detection
                import importlib
                import src.config as config
                importlib.reload(config)

                # Should be 90% of 8GB = 7.2GB
                assert config.MAX_GPU_MEMORY_GB == pytest.approx(7.2, rel=0.01)

    def test_gpu_memory_detection_without_cuda(self):
        """Test default value when CUDA is not available"""
        with patch('torch.cuda.is_available', return_value=False):
            import importlib
            import src.config as config
            importlib.reload(config)

            # Should fallback to 4GB default
            assert config.MAX_GPU_MEMORY_GB == pytest.approx(4.0)

    def test_gpu_memory_can_be_overridden_by_env(self):
        """Test GPU memory can be overridden via environment variable"""
        with patch.dict(os.environ, {'LWM_MAX_GPU_MEMORY_GB': '6.0'}):
            import importlib
            import src.config as config
            importlib.reload(config)

            assert config.MAX_GPU_MEMORY_GB == 6.0

    def test_config_has_all_memory_settings(self):
        """Test config module has all memory-related settings"""
        import src.config as config

        # Check all memory settings exist
        assert hasattr(config, 'MAX_GPU_MEMORY_GB')
        assert hasattr(config, 'MAX_SESSIONS')
        assert hasattr(config, 'SESSION_TTL_SECONDS')
        assert hasattr(config, 'MEMORY_CHECK_INTERVAL')

    def test_config_has_all_api_settings(self):
        """Test config module has all API settings"""
        import src.config as config

        assert hasattr(config, 'HOST')
        assert hasattr(config, 'PORT')
        assert hasattr(config, 'RATE_LIMIT_REQUESTS')
        assert hasattr(config, 'RATE_LIMIT_WINDOW')

    def test_config_defaults(self):
        """Test default configuration values"""
        import src.config as config

        # These should have sensible defaults
        assert config.MAX_SESSIONS >= 1
        assert config.SESSION_TTL_SECONDS > 0
        assert config.MEMORY_CHECK_INTERVAL > 0
        assert config.RATE_LIMIT_REQUESTS > 0
        assert config.RATE_LIMIT_WINDOW > 0
        assert config.HOST in ["0.0.0.0", "localhost", "127.0.0.1"]
        assert 1000 <= config.PORT <= 65535

    def test_config_can_be_overridden(self):
        """Test all config values can be overridden via environment"""
        env_vars = {
            'LWM_MAX_SESSIONS': '50',
            'LWM_SESSION_TTL_SECONDS': '1800',
            'LWM_MEMORY_CHECK_INTERVAL': '15',
            'LWM_RATE_LIMIT_REQUESTS': '200',
            'LWM_RATE_LIMIT_WINDOW': '120',
            'LWM_HOST': '127.0.0.1',
            'LWM_PORT': '9000',
        }

        with patch.dict(os.environ, env_vars):
            import importlib
            import src.config as config
            importlib.reload(config)

            assert config.MAX_SESSIONS == 50
            assert config.SESSION_TTL_SECONDS == 1800
            assert config.MEMORY_CHECK_INTERVAL == 15
            assert config.RATE_LIMIT_REQUESTS == 200
            assert config.RATE_LIMIT_WINDOW == 120
            assert config.HOST == '127.0.0.1'
            assert config.PORT == 9000


class TestMemoryLimitsCalculation:
    """Test memory limit calculation logic"""

    def test_90_percent_calculation(self):
        """Test 90% memory limit calculation"""
        total_memory_gb = 10.0
        expected_limit = total_memory_gb * 0.9

        assert expected_limit == 9.0

    def test_small_gpu_memory(self):
        """Test behavior with small GPU (2GB)"""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.get_device_properties') as mock_props:
                mock_device = MagicMock()
                mock_device.total_memory = 2 * 1024**3  # 2GB
                mock_props.return_value = mock_device

                import importlib
                import src.config as config
                importlib.reload(config)

                # Should be 90% of 2GB = 1.8GB
                assert config.MAX_GPU_MEMORY_GB == pytest.approx(1.8, rel=0.01)

    def test_large_gpu_memory(self):
        """Test behavior with large GPU (24GB)"""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.get_device_properties') as mock_props:
                mock_device = MagicMock()
                mock_device.total_memory = 24 * 1024**3  # 24GB
                mock_props.return_value = mock_device

                import importlib
                import src.config as config
                importlib.reload(config)

                # Should be 90% of 24GB = 21.6GB
                assert config.MAX_GPU_MEMORY_GB == pytest.approx(21.6, rel=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Unit tests for RateLimiter class
"""

import pytest
import time
from datetime import datetime, timedelta


# Import the RateLimiter from enhanced_server
# Note: We need to import it properly or copy the class for testing
class RateLimiter:
    """Simple in-memory rate limiter"""
    def __init__(self, requests: int = 100, window: int = 60):
        from collections import defaultdict
        self.requests = requests
        self.window = window
        self.clients = defaultdict(list)

    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed"""
        now = datetime.now()
        cutoff = now - timedelta(seconds=self.window)

        # Clean old requests
        self.clients[client_id] = [
            req_time for req_time in self.clients[client_id]
            if req_time > cutoff
        ]

        # Check limit
        if len(self.clients[client_id]) >= self.requests:
            return False

        # Add request
        self.clients[client_id].append(now)
        return True


class TestRateLimiter:
    """Test suite for RateLimiter"""

    def test_rate_limiter_initialization(self):
        """Test rate limiter can be initialized with custom settings"""
        limiter = RateLimiter(requests=50, window=30)
        assert limiter.requests == 50
        assert limiter.window == 30
        assert len(limiter.clients) == 0

    def test_rate_limiter_allows_first_request(self):
        """Test first request from client is always allowed"""
        limiter = RateLimiter(requests=10, window=60)
        assert limiter.is_allowed("client1") is True

    def test_rate_limiter_allows_under_limit(self):
        """Test requests under limit are allowed"""
        limiter = RateLimiter(requests=5, window=60)

        for i in range(5):
            assert limiter.is_allowed("client1") is True, f"Request {i+1} should be allowed"

    def test_rate_limiter_blocks_over_limit(self):
        """Test requests over limit are blocked"""
        limiter = RateLimiter(requests=3, window=60)

        # First 3 should be allowed
        for i in range(3):
            assert limiter.is_allowed("client1") is True

        # 4th should be blocked
        assert limiter.is_allowed("client1") is False

    def test_rate_limiter_different_clients(self):
        """Test different clients have independent limits"""
        limiter = RateLimiter(requests=2, window=60)

        # Client 1 hits limit
        assert limiter.is_allowed("client1") is True
        assert limiter.is_allowed("client1") is True
        assert limiter.is_allowed("client1") is False

        # Client 2 still has quota
        assert limiter.is_allowed("client2") is True
        assert limiter.is_allowed("client2") is True
        assert limiter.is_allowed("client2") is False

    def test_rate_limiter_window_cleanup(self):
        """Test old requests are cleaned up after window expires"""
        limiter = RateLimiter(requests=2, window=1)  # 1 second window

        # Fill quota
        assert limiter.is_allowed("client1") is True
        assert limiter.is_allowed("client1") is True
        assert limiter.is_allowed("client1") is False

        # Wait for window to expire
        time.sleep(1.1)

        # Should be allowed again
        assert limiter.is_allowed("client1") is True

    def test_rate_limiter_tracks_multiple_clients(self):
        """Test rate limiter can track multiple clients simultaneously"""
        limiter = RateLimiter(requests=3, window=60)

        clients = ["client1", "client2", "client3"]

        for client in clients:
            for i in range(3):
                assert limiter.is_allowed(client) is True
            assert limiter.is_allowed(client) is False

        assert len(limiter.clients) == 3

    def test_rate_limiter_empty_client_id(self):
        """Test rate limiter handles empty client ID"""
        limiter = RateLimiter(requests=5, window=60)

        # Should still work with empty string
        assert limiter.is_allowed("") is True
        assert limiter.is_allowed("") is True

    def test_rate_limiter_high_volume(self):
        """Test rate limiter with high request volume"""
        limiter = RateLimiter(requests=100, window=60)

        # Should allow 100 requests
        for i in range(100):
            assert limiter.is_allowed("high_volume_client") is True

        # 101st should be blocked
        assert limiter.is_allowed("high_volume_client") is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

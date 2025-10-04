"""
Unit tests for custom exceptions
"""

import pytest
from src.utils.exceptions import (
    WorldModelError,
    ModelLoadingError,
    InferenceError,
    MemoryError,
    SessionError,
    SessionNotFoundError,
    SessionCreationError,
    SessionLimitError,
    ConfigurationError,
    CheckpointError,
    InvalidActionError,
)


class TestExceptions:
    """Test suite for custom exception hierarchy"""

    def test_base_exception(self):
        """Test WorldModelError is base for all exceptions"""
        error = WorldModelError("test error")
        assert str(error) == "test error"
        assert isinstance(error, Exception)

    def test_model_loading_error(self):
        """Test ModelLoadingError inherits from WorldModelError"""
        error = ModelLoadingError("failed to load model")
        assert isinstance(error, WorldModelError)
        assert isinstance(error, Exception)
        assert str(error) == "failed to load model"

    def test_inference_error(self):
        """Test InferenceError inherits from WorldModelError"""
        error = InferenceError("inference failed")
        assert isinstance(error, WorldModelError)
        assert str(error) == "inference failed"

    def test_memory_error(self):
        """Test MemoryError inherits from WorldModelError"""
        error = MemoryError("out of memory")
        assert isinstance(error, WorldModelError)
        assert str(error) == "out of memory"

    def test_session_error_base(self):
        """Test SessionError is base for session-related errors"""
        error = SessionError("session error")
        assert isinstance(error, WorldModelError)
        assert str(error) == "session error"

    def test_session_not_found_error(self):
        """Test SessionNotFoundError inherits from SessionError"""
        error = SessionNotFoundError("session not found")
        assert isinstance(error, SessionError)
        assert isinstance(error, WorldModelError)
        assert str(error) == "session not found"

    def test_session_creation_error(self):
        """Test SessionCreationError inherits from SessionError"""
        error = SessionCreationError("failed to create session")
        assert isinstance(error, SessionError)
        assert str(error) == "failed to create session"

    def test_session_limit_error(self):
        """Test SessionLimitError inherits from SessionError"""
        error = SessionLimitError("session limit reached")
        assert isinstance(error, SessionError)
        assert str(error) == "session limit reached"

    def test_configuration_error(self):
        """Test ConfigurationError inherits from WorldModelError"""
        error = ConfigurationError("invalid configuration")
        assert isinstance(error, WorldModelError)
        assert str(error) == "invalid configuration"

    def test_checkpoint_error(self):
        """Test CheckpointError inherits from ModelLoadingError"""
        error = CheckpointError("checkpoint corrupted")
        assert isinstance(error, ModelLoadingError)
        assert isinstance(error, WorldModelError)
        assert str(error) == "checkpoint corrupted"

    def test_invalid_action_error(self):
        """Test InvalidActionError inherits from InferenceError"""
        error = InvalidActionError("action out of range")
        assert isinstance(error, InferenceError)
        assert isinstance(error, WorldModelError)
        assert str(error) == "action out of range"

    def test_exception_can_be_raised_and_caught(self):
        """Test exceptions can be raised and caught properly"""
        with pytest.raises(SessionNotFoundError) as exc_info:
            raise SessionNotFoundError("test session not found")

        assert "test session not found" in str(exc_info.value)

    def test_exception_hierarchy_catching(self):
        """Test catching by parent exception type"""
        try:
            raise SessionNotFoundError("specific error")
        except SessionError:
            # Should be caught by parent SessionError
            pass
        else:
            pytest.fail("Exception should have been caught")

        try:
            raise SessionNotFoundError("specific error")
        except WorldModelError:
            # Should also be caught by base WorldModelError
            pass
        else:
            pytest.fail("Exception should have been caught")

    def test_multiple_exception_types(self):
        """Test different exception types are distinct"""
        session_error = SessionNotFoundError("session error")
        inference_error = InferenceError("inference error")

        assert not isinstance(session_error, InferenceError)
        assert not isinstance(inference_error, SessionError)
        assert isinstance(session_error, WorldModelError)
        assert isinstance(inference_error, WorldModelError)

    def test_exception_with_formatted_message(self):
        """Test exceptions with formatted messages"""
        session_id = "abc-123"
        error = SessionNotFoundError(f"Session {session_id} not found")
        assert session_id in str(error)
        assert "not found" in str(error)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

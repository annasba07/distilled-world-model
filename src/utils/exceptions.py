"""
Custom exceptions for the Lightweight World Model project
"""


class WorldModelError(Exception):
    """Base exception for all world model errors"""
    pass


class ModelLoadingError(WorldModelError):
    """Raised when model loading fails"""
    pass


class InferenceError(WorldModelError):
    """Raised when inference fails"""
    pass


class MemoryError(WorldModelError):
    """Raised when memory limits are exceeded"""
    pass


class SessionError(WorldModelError):
    """Base exception for session-related errors"""
    pass


class SessionNotFoundError(SessionError):
    """Raised when session is not found"""
    pass


class SessionCreationError(SessionError):
    """Raised when session creation fails"""
    pass


class SessionLimitError(SessionError):
    """Raised when session limit is reached"""
    pass


class ConfigurationError(WorldModelError):
    """Raised when configuration is invalid"""
    pass


class CheckpointError(ModelLoadingError):
    """Raised when checkpoint operations fail"""
    pass


class InvalidActionError(InferenceError):
    """Raised when an invalid action is provided"""
    pass

"""
Context Management Utilities

Implements FramePack-style geometric compression for constant context windows.
Enables unlimited video length processing with constant memory.
"""

from .geometric_compression import (
    GeometricCompressor,
    GeometricHierarchy,
    ConstantContextManager,
    estimate_compression_savings,
)

__all__ = [
    'GeometricCompressor',
    'GeometricHierarchy',
    'ConstantContextManager',
    'estimate_compression_savings',
]

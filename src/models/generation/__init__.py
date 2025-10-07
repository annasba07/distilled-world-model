"""
Generation modules for video synthesis

Implements October 2025 SOTA generation techniques:
- MaskGIT: Parallel token generation (10x faster)
"""

from .maskgit import (
    MaskGITPredictor,
    MaskGITGenerator,
    MaskingScheduler,
    create_maskgit_generator,
)

__all__ = [
    'MaskGITPredictor',
    'MaskGITGenerator',
    'MaskingScheduler',
    'create_maskgit_generator',
]

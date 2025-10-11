"""
Latent Action Learning - Phase 2

Learn controllable actions from video without labels.
Inspired by Genie 3 (DeepMind, October 2025).

Components:
- ActionEncoder: Infer actions from consecutive frames
- ActionQuantizer: Discretize actions into vocabulary
- DynamicsModel: Predict next frame given current + action
- LatentActionPipeline: Full integrated pipeline
"""

from .action_encoder import ActionEncoder
from .action_quantizer import ActionQuantizer

__all__ = [
    'ActionEncoder',
    'ActionQuantizer',
]

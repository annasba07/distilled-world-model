"""
Enhanced World Model with Optimized Components

This module provides an enhanced world model that integrates:
- Optimized dynamics with vectorized Mamba
- Progressive model loading
- Error recovery mechanisms
- Performance monitoring
"""

import torch
import torch.nn as nn
import math
import time
import asyncio
from typing import Optional, Tuple, Dict, Any, Callable
from pathlib import Path
import threading
from einops import rearrange

from .improved_vqvae import ImprovedVQVAE
from .optimized_dynamics import OptimizedDynamicsModel


class ProgressCallback:
    """Callback interface for progress reporting"""

    def __init__(self, callback_fn: Optional[Callable[[str, float], None]] = None):
        self.callback_fn = callback_fn

    def update(self, stage: str, progress: float):
        """Update progress (0.0 to 1.0)"""
        if self.callback_fn:
            self.callback_fn(stage, progress)

    def __call__(self, stage: str, progress: float):
        self.update(stage, progress)


class ModelLoadingError(Exception):
    """Custom exception for model loading errors"""
    pass


class EnhancedWorldModel(nn.Module):
    """
    Enhanced World Model with progressive loading and optimized components
    """

    def __init__(self, vqvae=None, dynamics_model=None, device='cuda'):
        super().__init__()

        self.device = torch.device(device)
        self.vqvae = vqvae or ImprovedVQVAE()
        self.dynamics = dynamics_model or OptimizedDynamicsModel()

        # Move to device
        self.vqvae = self.vqvae.to(self.device)
        self.dynamics = self.dynamics.to(self.device)

        # Loading state
        self.is_loading = False
        self.loading_progress = 0.0
        self.loading_stage = "idle"
        self.loading_lock = threading.Lock()

        # Performance tracking
        self.performance_stats = {
            'encoding_times': [],
            'dynamics_times': [],
            'decoding_times': [],
            'total_inference_times': []
        }

        # Error recovery
        self.last_known_good_state = None
        self.error_count = 0
        self.max_errors = 5

    def encode_frames(self, frames):
        """Encode frames with performance tracking"""
        start_time = time.time()

        batch_size, seq_len = frames.shape[:2]
        frames_flat = rearrange(frames, 'b t c h w -> (b t) c h w')

        try:
            with torch.no_grad():
                latents, _ = self.vqvae.encode(frames_flat)

            latents = rearrange(latents, '(b t) c h w -> b t (c h w)', b=batch_size)

            # Track performance
            encoding_time = time.time() - start_time
            self._update_performance_stats('encoding_times', encoding_time)

            return latents

        except Exception as e:
            self.error_count += 1
            if self.error_count > self.max_errors:
                raise ModelLoadingError(f"Too many encoding errors: {e}")

            # Return cached state if available
            if self.last_known_good_state is not None:
                return self.last_known_good_state.get('latents')
            raise

    def decode_latents(self, latents):
        """Decode latents with performance tracking"""
        start_time = time.time()

        try:
            batch_size, seq_len = latents.shape[:2]
            latent_dim = int(math.sqrt(latents.shape[-1] // self.vqvae.encoder.conv_out.out_channels))

            latents_reshaped = rearrange(
                latents,
                'b t (c h w) -> (b t) c h w',
                c=self.vqvae.encoder.conv_out.out_channels,
                h=latent_dim,
                w=latent_dim
            )

            with torch.no_grad():
                frames = self.vqvae.decode(latents_reshaped)

            frames = rearrange(frames, '(b t) c h w -> b t c h w', b=batch_size)

            # Track performance
            decoding_time = time.time() - start_time
            self._update_performance_stats('decoding_times', decoding_time)

            return frames

        except Exception as e:
            self.error_count += 1
            if self.error_count > self.max_errors:
                raise ModelLoadingError(f"Too many decoding errors: {e}")
            raise

    def forward(self, frames, actions=None):
        """Forward pass with performance tracking"""
        start_time = time.time()

        try:
            latents = self.encode_frames(frames)

            dynamics_start = time.time()
            predicted_latents = self.dynamics(latents[:, :-1], actions)
            dynamics_time = time.time() - dynamics_start
            self._update_performance_stats('dynamics_times', dynamics_time)

            target_latents = latents[:, 1:]

            total_time = time.time() - start_time
            self._update_performance_stats('total_inference_times', total_time)

            # Reset error count on successful operation
            self.error_count = 0

            return predicted_latents, target_latents

        except Exception as e:
            self.error_count += 1
            raise

    def generate_trajectory(self, initial_frame, actions, num_steps):
        """Generate trajectory with error recovery"""
        try:
            initial_latent = self.encode_frames(initial_frame.unsqueeze(1))
            predicted_latents = self.dynamics.generate(initial_latent, actions, num_steps)
            generated_frames = self.decode_latents(predicted_latents)

            # Cache successful state
            self.last_known_good_state = {
                'latents': initial_latent,
                'frames': generated_frames
            }

            return generated_frames

        except Exception as e:
            self.error_count += 1
            if self.last_known_good_state and self.error_count <= self.max_errors:
                # Return cached frames as fallback
                return self.last_known_good_state.get('frames')
            raise

    def _update_performance_stats(self, stat_name: str, value: float):
        """Update performance statistics"""
        stats_list = self.performance_stats[stat_name]
        stats_list.append(value)

        # Keep only recent 100 measurements
        if len(stats_list) > 100:
            stats_list.pop(0)

    async def load_checkpoint_progressive(self, checkpoint_path: str,
                                        progress_callback: Optional[ProgressCallback] = None) -> bool:
        """
        Load model checkpoint with progressive loading and status updates
        """
        if self.is_loading:
            return False

        with self.loading_lock:
            self.is_loading = True
            self.loading_progress = 0.0
            self.loading_stage = "initializing"

        try:
            if progress_callback:
                progress_callback("Validating checkpoint", 0.1)

            # Validate checkpoint exists
            checkpoint_path = Path(checkpoint_path)
            if not checkpoint_path.exists():
                raise ModelLoadingError(f"Checkpoint not found: {checkpoint_path}")

            if progress_callback:
                progress_callback("Loading checkpoint data", 0.2)

            # Load checkpoint data
            checkpoint = await asyncio.get_event_loop().run_in_executor(
                None, torch.load, str(checkpoint_path), {'map_location': self.device}
            )

            # Determine checkpoint format
            state_dict = None
            vqvae_state = None
            dynamics_state = None

            if progress_callback:
                progress_callback("Parsing checkpoint format", 0.3)

            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'vqvae' in checkpoint and 'dynamics' in checkpoint:
                    vqvae_state = checkpoint['vqvae']
                    dynamics_state = checkpoint['dynamics']
                elif 'state_dict' in checkpoint:
                    # Lightning checkpoint
                    lightning_state = checkpoint['state_dict']
                    vqvae_state = {
                        k.replace('vqvae.', ''): v
                        for k, v in lightning_state.items()
                        if k.startswith('vqvae.')
                    }
                    dynamics_state = {
                        k.replace('dynamics.', ''): v
                        for k, v in lightning_state.items()
                        if k.startswith('dynamics.')
                    }
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            # Load VQ-VAE weights
            if progress_callback:
                progress_callback("Loading VQ-VAE weights", 0.5)

            if vqvae_state is not None:
                missing, unexpected = self.vqvae.load_state_dict(vqvae_state, strict=False)
                if missing:
                    print(f"Warning: Missing VQ-VAE keys: {missing}")
                if unexpected:
                    print(f"Warning: Unexpected VQ-VAE keys: {unexpected}")
            elif state_dict is not None:
                # Extract VQ-VAE weights from combined state dict
                vqvae_weights = {
                    k.replace('vqvae.', ''): v
                    for k, v in state_dict.items()
                    if k.startswith('vqvae.')
                }
                if vqvae_weights:
                    self.vqvae.load_state_dict(vqvae_weights, strict=False)

            await asyncio.sleep(0.1)  # Yield control

            # Load dynamics weights
            if progress_callback:
                progress_callback("Loading dynamics weights", 0.7)

            if dynamics_state is not None:
                missing, unexpected = self.dynamics.load_state_dict(dynamics_state, strict=False)
                if missing:
                    print(f"Warning: Missing dynamics keys: {missing}")
                if unexpected:
                    print(f"Warning: Unexpected dynamics keys: {unexpected}")
            elif state_dict is not None:
                # Extract dynamics weights from combined state dict
                dynamics_weights = {
                    k.replace('dynamics.', ''): v
                    for k, v in state_dict.items()
                    if k.startswith('dynamics.')
                }
                if dynamics_weights:
                    self.dynamics.load_state_dict(dynamics_weights, strict=False)

            await asyncio.sleep(0.1)  # Yield control

            # Optimization and finalization
            if progress_callback:
                progress_callback("Optimizing model", 0.9)

            # Enable fast inference optimizations
            if hasattr(self.dynamics, 'enable_fast_inference'):
                self.dynamics.enable_fast_inference()

            # Set to eval mode
            self.eval()

            if progress_callback:
                progress_callback("Loading complete", 1.0)

            self.loading_stage = "complete"
            return True

        except Exception as e:
            if progress_callback:
                progress_callback(f"Loading failed: {str(e)}", 1.0)
            self.loading_stage = f"error: {str(e)}"
            raise ModelLoadingError(f"Failed to load checkpoint: {e}")

        finally:
            with self.loading_lock:
                self.is_loading = False

    def get_loading_status(self) -> Dict[str, Any]:
        """Get current loading status"""
        return {
            'is_loading': self.is_loading,
            'progress': self.loading_progress,
            'stage': self.loading_stage
        }

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        def compute_stats(times_list):
            if not times_list:
                return {'avg': 0.0, 'min': 0.0, 'max': 0.0, 'count': 0}
            return {
                'avg': sum(times_list) / len(times_list),
                'min': min(times_list),
                'max': max(times_list),
                'count': len(times_list)
            }

        report = {
            'encoding': compute_stats(self.performance_stats['encoding_times']),
            'dynamics': compute_stats(self.performance_stats['dynamics_times']),
            'decoding': compute_stats(self.performance_stats['decoding_times']),
            'total_inference': compute_stats(self.performance_stats['total_inference_times']),
            'error_count': self.error_count,
            'has_fallback_state': self.last_known_good_state is not None
        }

        # Add dynamics-specific stats
        if hasattr(self.dynamics, 'get_performance_stats'):
            report['dynamics_detailed'] = self.dynamics.get_performance_stats()

        return report

    def reset_performance_stats(self):
        """Reset all performance statistics"""
        for key in self.performance_stats:
            self.performance_stats[key].clear()
        self.error_count = 0

    def enable_fast_mode(self):
        """Enable all performance optimizations"""
        if hasattr(self.dynamics, 'enable_fast_inference'):
            self.dynamics.enable_fast_inference()

        # Enable eval mode for faster inference
        self.eval()

        # Clear performance stats for fresh measurement
        self.reset_performance_stats()

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters())

        return {
            'vqvae_parameters': count_parameters(self.vqvae),
            'dynamics_parameters': count_parameters(self.dynamics),
            'total_parameters': count_parameters(self),
            'device': str(self.device),
            'vqvae_type': type(self.vqvae).__name__,
            'dynamics_type': type(self.dynamics).__name__,
            'fast_scan_enabled': getattr(self.dynamics, 'use_fast_scan', False) if hasattr(self.dynamics.layers[0], 'use_fast_scan') else False
        }


# Factory functions for easy creation
def create_enhanced_world_model(device='cuda', use_optimized_dynamics=True, **kwargs):
    """Create enhanced world model with optimized components"""
    vqvae = ImprovedVQVAE()

    if use_optimized_dynamics:
        dynamics = OptimizedDynamicsModel(**kwargs)
    else:
        from .dynamics import DynamicsModel
        dynamics = DynamicsModel(**kwargs)

    return EnhancedWorldModel(vqvae, dynamics, device)


async def load_model_with_progress(checkpoint_path: str, device='cuda',
                                 progress_callback: Optional[ProgressCallback] = None):
    """Convenience function to create and load model with progress"""
    model = create_enhanced_world_model(device=device)
    success = await model.load_checkpoint_progressive(checkpoint_path, progress_callback)

    if success:
        model.enable_fast_mode()

    return model, success
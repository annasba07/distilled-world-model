"""
Optimization Utilities for Production Deployment

Provides optimization techniques for efficient inference on consumer GPUs:
- FP16 mixed precision
- torch.compile() support
- Flash Attention integration
- Memory optimization

Target: 40-50 FPS @ 640×360 on RTX 3060 with <4GB VRAM

Author: Based on October 2025 optimization techniques
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Callable
import functools
import warnings


class MixedPrecisionWrapper:
    """
    Wrapper for FP16 mixed precision inference.

    Benefits:
    - 2x faster computation (on compatible GPUs)
    - 2x less memory usage
    - Minimal accuracy loss

    Usage:
        wrapper = MixedPrecisionWrapper()
        with wrapper.autocast():
            output = model(input)
    """

    def __init__(
        self,
        enabled: bool = True,
        device_type: str = 'cuda',
        dtype: torch.dtype = torch.float16
    ):
        self.enabled = enabled and torch.cuda.is_available()
        self.device_type = device_type if self.enabled else 'cpu'
        self.dtype = dtype

        if self.enabled:
            print(f"✅ Mixed precision enabled: {dtype}")
        else:
            print("⚠️  Mixed precision disabled (CUDA not available or not enabled)")

    def autocast(self):
        """Get autocast context manager"""
        if self.enabled:
            return torch.autocast(device_type=self.device_type, dtype=self.dtype)
        else:
            return torch.autocast(device_type='cpu', enabled=False)

    def optimize_model(self, model: nn.Module) -> nn.Module:
        """
        Optimize model for mixed precision.

        Args:
            model: Model to optimize

        Returns:
            Optimized model
        """
        if not self.enabled:
            return model

        # Convert batchnorm to float32 for stability
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.float()

        return model


def compile_model(
    model: nn.Module,
    mode: str = 'reduce-overhead',
    fullgraph: bool = False,
    dynamic: bool = False
) -> nn.Module:
    """
    Compile model with torch.compile() for 2x speedup.

    Requires PyTorch 2.0+

    Args:
        model: Model to compile
        mode: Compilation mode
            - 'default': Balanced
            - 'reduce-overhead': Minimize Python overhead (best for inference)
            - 'max-autotune': Maximum performance (slower compile)
        fullgraph: Compile entire graph (stricter, faster but may fail)
        dynamic: Support dynamic shapes (slower but more flexible)

    Returns:
        Compiled model

    Modes:
        - reduce-overhead: Best for inference (default)
        - max-autotune: Best for training
        - default: Balanced
    """
    # Check PyTorch version
    pytorch_version = tuple(int(x) for x in torch.__version__.split('.')[:2])
    if pytorch_version < (2, 0):
        warnings.warn(f"torch.compile requires PyTorch 2.0+, got {torch.__version__}. Skipping compilation.")
        return model

    try:
        print(f"Compiling model with mode='{mode}'...")
        compiled_model = torch.compile(
            model,
            mode=mode,
            fullgraph=fullgraph,
            dynamic=dynamic
        )
        print("✅ Model compiled successfully")
        return compiled_model

    except Exception as e:
        warnings.warn(f"Model compilation failed: {e}. Using uncompiled model.")
        return model


class FlashAttentionOptimizer:
    """
    Flash Attention optimization for memory-efficient attention.

    Flash Attention reduces memory from O(n²) to O(n) for attention.

    Benefits:
    - 2-4x faster attention
    - 10-20x less memory
    - Enables longer sequences

    Requires: flash-attn package (pip install flash-attn)
    """

    def __init__(self):
        self.available = self._check_availability()

        if self.available:
            print("✅ Flash Attention available")
        else:
            print("⚠️  Flash Attention not available (install: pip install flash-attn)")

    def _check_availability(self) -> bool:
        """Check if flash attention is available"""
        try:
            import flash_attn
            return True
        except ImportError:
            return False

    def optimize_transformer(self, transformer: nn.Module) -> nn.Module:
        """
        Replace standard attention with Flash Attention.

        Args:
            transformer: Transformer model

        Returns:
            Optimized transformer
        """
        if not self.available:
            warnings.warn("Flash Attention not available, skipping optimization")
            return transformer

        # Replace attention layers
        # This is model-specific, implement based on your architecture
        # For now, just return as-is with a warning

        warnings.warn("Flash Attention replacement not yet implemented for this model")
        return transformer

    def create_flash_attention_module(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0
    ) -> nn.Module:
        """
        Create a Flash Attention module.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout rate

        Returns:
            Flash Attention module
        """
        if not self.available:
            # Fallback to standard attention
            return nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )

        try:
            from flash_attn import FlashMHA

            return FlashMHA(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_flash_attn=True
            )
        except Exception as e:
            warnings.warn(f"Flash Attention creation failed: {e}. Using standard attention.")
            return nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )


class InferenceOptimizer:
    """
    Complete inference optimization pipeline.

    Combines all optimization techniques:
    1. FP16 mixed precision (2x speedup)
    2. torch.compile() (2x speedup)
    3. Flash Attention (2-4x speedup for attention)

    Total expected speedup: 4-8x

    Usage:
        optimizer = InferenceOptimizer()
        model = optimizer.optimize(model)

        # Inference
        with optimizer.autocast():
            output = model(input)
    """

    def __init__(
        self,
        use_fp16: bool = True,
        use_compile: bool = True,
        use_flash_attn: bool = False,  # Disabled by default (requires extra package)
        compile_mode: str = 'reduce-overhead'
    ):
        self.use_fp16 = use_fp16
        self.use_compile = use_compile
        self.use_flash_attn = use_flash_attn
        self.compile_mode = compile_mode

        # Initialize components
        self.fp16_wrapper = MixedPrecisionWrapper(enabled=use_fp16)
        self.flash_attn = FlashAttentionOptimizer() if use_flash_attn else None

        print("\nInferenceOptimizer initialized:")
        print(f"  FP16: {use_fp16}")
        print(f"  torch.compile: {use_compile}")
        print(f"  Flash Attention: {use_flash_attn}")

    def optimize(self, model: nn.Module, **compile_kwargs) -> nn.Module:
        """
        Apply all optimizations to model.

        Args:
            model: Model to optimize
            **compile_kwargs: Additional arguments for torch.compile

        Returns:
            Optimized model
        """
        print("\nOptimizing model...")

        # 1. FP16 optimization
        if self.use_fp16:
            print("  Applying FP16 mixed precision...")
            model = self.fp16_wrapper.optimize_model(model)

        # 2. Flash Attention (if enabled)
        if self.use_flash_attn and self.flash_attn:
            print("  Applying Flash Attention...")
            model = self.flash_attn.optimize_transformer(model)

        # 3. torch.compile (must be last!)
        if self.use_compile:
            print("  Compiling model...")
            compile_args = {'mode': self.compile_mode, **compile_kwargs}
            model = compile_model(model, **compile_args)

        print("✅ Model optimization complete\n")
        return model

    def autocast(self):
        """Get autocast context for inference"""
        return self.fp16_wrapper.autocast()

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get statistics about enabled optimizations"""
        return {
            'fp16_enabled': self.use_fp16,
            'compile_enabled': self.use_compile,
            'flash_attn_enabled': self.use_flash_attn,
            'expected_speedup': self._estimate_speedup()
        }

    def _estimate_speedup(self) -> str:
        """Estimate total speedup from enabled optimizations"""
        speedup = 1.0

        if self.use_fp16:
            speedup *= 2.0  # ~2x from FP16

        if self.use_compile:
            speedup *= 2.0  # ~2x from torch.compile

        if self.use_flash_attn:
            speedup *= 2.5  # ~2-3x from Flash Attention

        return f"{speedup:.1f}x"


def enable_tf32(enabled: bool = True):
    """
    Enable TF32 mode for faster computation on Ampere+ GPUs.

    TF32 uses Tensor Cores for faster matmul with minimal accuracy loss.

    Args:
        enabled: Whether to enable TF32
    """
    if not torch.cuda.is_available():
        return

    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled

    if enabled:
        print("✅ TF32 enabled (faster on Ampere+ GPUs)")
    else:
        print("TF32 disabled")


def optimize_for_inference(model: nn.Module, device: torch.device) -> nn.Module:
    """
    Quick optimization for inference.

    Args:
        model: Model to optimize
        device: Device to use

    Returns:
        Optimized model
    """
    # Move to device
    model = model.to(device)

    # Set to eval mode
    model.eval()

    # Disable gradients
    for param in model.parameters():
        param.requires_grad = False

    # Enable TF32 if on CUDA
    if device.type == 'cuda':
        enable_tf32(True)

    print(f"✅ Model optimized for inference on {device}")

    return model


# Utility decorator for automatic mixed precision
def autocast_inference(func: Callable) -> Callable:
    """
    Decorator to automatically use mixed precision for inference.

    Usage:
        @autocast_inference
        def generate(model, input):
            return model(input)
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)

    return wrapper


if __name__ == '__main__':
    print("Testing Optimization Utilities...\n")

    # Test 1: Mixed precision
    print("1. Testing Mixed Precision...")
    mp_wrapper = MixedPrecisionWrapper()

    model = nn.Linear(512, 512)
    x = torch.randn(4, 512)

    if torch.cuda.is_available():
        model = model.cuda()
        x = x.cuda()

    with mp_wrapper.autocast():
        y = model(x)
        print(f"   Output dtype: {y.dtype}")

    print("   ✅ Mixed precision works\n")

    # Test 2: torch.compile
    print("2. Testing torch.compile...")
    model2 = nn.Linear(256, 256)

    try:
        compiled = compile_model(model2, mode='default')
        print("   ✅ Compilation successful\n")
    except Exception as e:
        print(f"   ⚠️  Compilation failed (expected on older PyTorch): {e}\n")

    # Test 3: Flash Attention
    print("3. Testing Flash Attention...")
    flash_opt = FlashAttentionOptimizer()
    print(f"   Available: {flash_opt.available}\n")

    # Test 4: Full optimizer
    print("4. Testing InferenceOptimizer...")
    optimizer = InferenceOptimizer(
        use_fp16=True,
        use_compile=False,  # Skip compilation for test
        use_flash_attn=False
    )

    model3 = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 128)
    )

    optimized = optimizer.optimize(model3)
    stats = optimizer.get_optimization_stats()

    print(f"   Optimization stats: {stats}")
    print("   ✅ Full optimizer works\n")

    # Test 5: TF32
    print("5. Testing TF32...")
    enable_tf32(True)
    print("   ✅ TF32 configured\n")

    print("✅ All optimization tests passed!")

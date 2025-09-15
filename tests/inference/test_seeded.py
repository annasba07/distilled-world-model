import numpy as np
from src.inference.engine import OptimizedInferenceEngine


def test_seeded_initial_frame_is_deterministic():
    e = OptimizedInferenceEngine(model_path=None, device='cpu', use_tensorrt=False, use_fp16=False)
    f1 = e.generate_interactive("test", seed=42)
    # Reset internal state and generate again
    e._init_buffers()
    f2 = e.generate_interactive("test", seed=42)
    assert f1.shape == (256, 256, 3)
    assert np.array_equal(f1, f2)


import numpy as np

from src.inference.engine import BatchedInferenceEngine, OptimizedInferenceEngine


def test_seeded_initial_frame_is_deterministic():
    e = OptimizedInferenceEngine(model_path=None, device='cpu', use_tensorrt=False, use_fp16=False)
    f1 = e.generate_interactive("test", seed=42)
    # Reset internal state and generate again
    e._init_buffers()
    f2 = e.generate_interactive("test", seed=42)
    assert f1.shape == (256, 256, 3)
    assert np.array_equal(f1, f2)


def test_multi_session_state_isolated():
    engine = BatchedInferenceEngine(model_path=None, device='cpu', use_tensorrt=False, use_fp16=False)

    frame_a = engine.start_session("a", seed=123)
    frame_b = engine.start_session("b", seed=456)

    assert frame_a.shape == (256, 256, 3)
    assert frame_b.shape == (256, 256, 3)
    assert engine.is_session_running("a")
    assert engine.is_session_running("b")

    state_b_pre = engine.get_session_state("b").last_frame.copy()

    next_a, metrics = engine.step_session("a", action=0)
    assert isinstance(metrics, dict)
    assert next_a.shape == (256, 256, 3)

    state_b_post = engine.get_session_state("b").last_frame
    assert np.array_equal(state_b_pre, state_b_post)
    assert engine.is_session_running("b")

    engine.stop_session("a")
    assert not engine.has_session("a")


def test_toy_world_deterministic_control():
    engine = BatchedInferenceEngine(model_path=None, device='cpu', use_tensorrt=False, use_fp16=False)

    frame1 = engine.start_session("toy", seed=7)
    engine.stop_session("toy")
    frame2 = engine.start_session("toy", seed=7)

    assert np.array_equal(frame1, frame2)

    frame3 = engine.start_session("move", seed=21)
    moved_frame, metrics = engine.step_session("move", action=4)  # move right

    assert moved_frame.shape == (256, 256, 3)
    assert metrics.get("moved") is True
    assert metrics.get("tick") == 1
    assert not np.array_equal(frame3, moved_frame)

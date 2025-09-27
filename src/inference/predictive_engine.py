"""
Predictive Inference Engine for Zero-Latency User Experience

This module implements frame prediction with lookahead to eliminate perceived
latency during real-time interaction. Users get instant visual feedback while
the actual inference happens in the background.
"""

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor

from .engine import BatchedInferenceEngine
try:
    from .engine import SessionState
except ImportError:
    # Fallback if SessionState is not available in the original engine
    from dataclasses import dataclass
    from typing import Optional
    import numpy as np
    import torch
    from collections import deque

    @dataclass
    class SessionState:
        latent: Optional[torch.Tensor]
        last_frame: Optional[np.ndarray]
        frame_buffer: deque
        prompt: Optional[str]
        seed: Optional[int]
        running: bool = False
        world: Optional[Any] = None


@dataclass
class PredictedFrame:
    """Container for a predicted frame with metadata"""
    frame: np.ndarray
    action: int
    confidence: float
    timestamp: float


@dataclass
class PredictionContext:
    """Context for frame prediction including likely actions"""
    session_id: str
    current_frame: np.ndarray
    current_latent: Optional[torch.Tensor]
    likely_actions: List[int]
    prediction_horizon: int = 3


class FramePredictor:
    """Handles frame prediction logic and caching"""

    def __init__(self, engine: BatchedInferenceEngine, prediction_horizon: int = 3):
        self.engine = engine
        self.prediction_horizon = prediction_horizon
        self.prediction_cache: Dict[str, Dict[int, PredictedFrame]] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)

    async def predict_frames_async(self, context: PredictionContext) -> Dict[int, PredictedFrame]:
        """Predict multiple frames for different actions asynchronously"""
        loop = asyncio.get_event_loop()

        # Predict frames for likely actions in parallel
        prediction_tasks = []
        for action in context.likely_actions:
            task = loop.run_in_executor(
                self.executor,
                self._predict_single_frame,
                context.session_id,
                context.current_frame,
                context.current_latent,
                action
            )
            prediction_tasks.append((action, task))

        # Collect results
        predicted_frames = {}
        for action, task in prediction_tasks:
            try:
                frame, confidence = await task
                predicted_frames[action] = PredictedFrame(
                    frame=frame,
                    action=action,
                    confidence=confidence,
                    timestamp=time.time()
                )
            except Exception as e:
                print(f"Prediction failed for action {action}: {e}")
                # Use current frame as fallback
                predicted_frames[action] = PredictedFrame(
                    frame=context.current_frame,
                    action=action,
                    confidence=0.0,
                    timestamp=time.time()
                )

        # Cache predictions
        self.prediction_cache[context.session_id] = predicted_frames
        return predicted_frames

    def _predict_single_frame(self, session_id: str, current_frame: np.ndarray,
                             current_latent: Optional[torch.Tensor], action: int) -> Tuple[np.ndarray, float]:
        """Predict a single frame for given action"""
        try:
            if self.engine.test_mode:
                # Fast deterministic prediction for testing
                predicted_frame = current_frame.copy()
                predicted_frame[:, :, 1] = min(max(action, 0), 255)  # Green channel = action
                return predicted_frame, 1.0

            # Use the engine's stateless processing
            predicted_frame, _ = self.engine.process_frame_with_state(
                current_frame, action, current_latent
            )

            # Calculate confidence based on frame stability
            confidence = self._calculate_confidence(current_frame, predicted_frame)

            return predicted_frame, confidence

        except Exception as e:
            print(f"Frame prediction error: {e}")
            return current_frame, 0.0

    def _calculate_confidence(self, current_frame: np.ndarray, predicted_frame: np.ndarray) -> float:
        """Calculate prediction confidence based on frame stability"""
        try:
            # Simple confidence metric based on frame difference
            diff = np.mean(np.abs(predicted_frame.astype(float) - current_frame.astype(float)))
            # Normalize to 0-1 range (lower diff = higher confidence)
            confidence = max(0.0, min(1.0, 1.0 - diff / 255.0))
            return confidence
        except:
            return 0.5  # Default confidence

    def get_cached_prediction(self, session_id: str, action: int) -> Optional[PredictedFrame]:
        """Get cached prediction for session and action"""
        session_cache = self.prediction_cache.get(session_id, {})
        prediction = session_cache.get(action)

        # Check if prediction is still fresh (within 500ms)
        if prediction and (time.time() - prediction.timestamp) < 0.5:
            return prediction
        return None

    def clear_cache(self, session_id: str):
        """Clear prediction cache for session"""
        self.prediction_cache.pop(session_id, None)


class ActionPredictor:
    """Predicts likely next actions based on user behavior patterns"""

    def __init__(self, history_size: int = 50):
        self.action_history: Dict[str, deque] = {}
        self.history_size = history_size

    def update_history(self, session_id: str, action: int):
        """Update action history for session"""
        if session_id not in self.action_history:
            self.action_history[session_id] = deque(maxlen=self.history_size)
        self.action_history[session_id].append(action)

    def predict_likely_actions(self, session_id: str, top_k: int = 3) -> List[int]:
        """Predict most likely next actions based on history"""
        history = self.action_history.get(session_id, deque())

        if len(history) < 3:
            # Default actions for new sessions
            return [0, 1, 2, 3, 4][:top_k]  # stay, up, down, left, right

        # Simple frequency-based prediction
        action_counts = {}
        for action in history:
            action_counts[action] = action_counts.get(action, 0) + 1

        # Weight recent actions more heavily
        recent_actions = list(history)[-10:]
        for action in recent_actions:
            action_counts[action] = action_counts.get(action, 0) + 2

        # Sort by frequency and return top_k
        sorted_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)
        return [action for action, _ in sorted_actions[:top_k]]

    def clear_history(self, session_id: str):
        """Clear action history for session"""
        self.action_history.pop(session_id, None)


class PredictiveInferenceEngine:
    """Enhanced inference engine with zero-latency frame prediction"""

    def __init__(self, base_engine: BatchedInferenceEngine, prediction_horizon: int = 3):
        self.base_engine = base_engine
        self.frame_predictor = FramePredictor(base_engine, prediction_horizon)
        self.action_predictor = ActionPredictor()
        self.prediction_tasks: Dict[str, asyncio.Task] = {}

    async def start_session_predictive(self, session_id: str, prompt: Optional[str] = None,
                                     seed: Optional[int] = None) -> np.ndarray:
        """Start session with predictive capabilities"""
        # Start the session normally
        initial_frame = self.base_engine.start_session(session_id, prompt, seed)

        # Initialize prediction for this session
        await self._start_prediction_for_session(session_id)

        return initial_frame

    async def step_session_predictive(self, session_id: str, action: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Step session with zero-latency using predictions"""
        start_time = time.time()

        # Check if we have a cached prediction for this action
        cached_prediction = self.frame_predictor.get_cached_prediction(session_id, action)

        if cached_prediction and cached_prediction.confidence > 0.7:
            # Use cached prediction for instant response
            instant_frame = cached_prediction.frame

            # Start actual inference in background
            asyncio.create_task(self._update_ground_truth(session_id, action))

            # Update action history
            self.action_predictor.update_history(session_id, action)

            # Restart prediction for next frame
            asyncio.create_task(self._start_prediction_for_session(session_id))

            instant_time = time.time() - start_time
            metrics = {
                'fps': 1.0 / instant_time if instant_time > 0 else 1000.0,
                'inference_time': instant_time,
                'predicted': True,
                'confidence': cached_prediction.confidence
            }

            return instant_frame, metrics

        else:
            # Fall back to normal inference if no good prediction available
            frame, metrics = self.base_engine.step_session(session_id, action)

            # Update action history
            self.action_predictor.update_history(session_id, action)

            # Restart prediction for next frame
            asyncio.create_task(self._start_prediction_for_session(session_id))

            metrics['predicted'] = False
            return frame, metrics

    async def _start_prediction_for_session(self, session_id: str):
        """Start background prediction for session"""
        try:
            # Cancel existing prediction task
            if session_id in self.prediction_tasks:
                self.prediction_tasks[session_id].cancel()

            # Get current session state
            state = self.base_engine.get_session_state(session_id)
            if not state.running or state.last_frame is None:
                return

            # Predict likely actions
            likely_actions = self.action_predictor.predict_likely_actions(session_id, top_k=5)

            # Create prediction context
            context = PredictionContext(
                session_id=session_id,
                current_frame=state.last_frame,
                current_latent=state.latent,
                likely_actions=likely_actions,
                prediction_horizon=3
            )

            # Start prediction task
            self.prediction_tasks[session_id] = asyncio.create_task(
                self.frame_predictor.predict_frames_async(context)
            )

        except Exception as e:
            print(f"Prediction setup error for session {session_id}: {e}")

    async def _update_ground_truth(self, session_id: str, action: int):
        """Update with actual inference result in background"""
        try:
            # Perform actual inference
            self.base_engine.step_session(session_id, action)
            # Ground truth will be used for next frame prediction
        except Exception as e:
            print(f"Ground truth update error: {e}")

    def stop_session_predictive(self, session_id: str):
        """Stop session and cleanup prediction resources"""
        # Cancel prediction task
        if session_id in self.prediction_tasks:
            self.prediction_tasks[session_id].cancel()
            del self.prediction_tasks[session_id]

        # Clear caches
        self.frame_predictor.clear_cache(session_id)
        self.action_predictor.clear_history(session_id)

        # Stop base session
        self.base_engine.stop_session(session_id)

    def get_prediction_stats(self, session_id: str) -> Dict[str, Any]:
        """Get prediction performance statistics"""
        cache = self.frame_predictor.prediction_cache.get(session_id, {})
        if not cache:
            return {'cache_size': 0, 'avg_confidence': 0.0}

        confidences = [pred.confidence for pred in cache.values()]
        return {
            'cache_size': len(cache),
            'avg_confidence': sum(confidences) / len(confidences) if confidences else 0.0,
            'actions_cached': list(cache.keys())
        }


# Integration helper for existing API
def create_predictive_engine(*args, **kwargs) -> PredictiveInferenceEngine:
    """Factory function to create predictive engine"""
    # Extract prediction-specific arguments
    prediction_horizon = kwargs.pop('prediction_horizon', 3)

    # Create base engine with remaining arguments
    base_engine = BatchedInferenceEngine(*args, **kwargs)
    return PredictiveInferenceEngine(base_engine, prediction_horizon)
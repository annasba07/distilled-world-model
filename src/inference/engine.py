import os
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ..models.dynamics import DynamicsModel, WorldModel
from ..models.improved_vqvae import ImprovedVQVAE
from .toy_world import ToyWorldSimulator

if TYPE_CHECKING:
    from .toy_world import ToyWorldState


@dataclass
class SessionState:
    """Container for interactive session state managed by BatchedInferenceEngine."""

    latent: Optional[torch.Tensor]
    last_frame: Optional[np.ndarray]
    frame_buffer: deque
    prompt: Optional[str]
    seed: Optional[int]
    running: bool = False
    world: Optional["ToyWorldState"] = None


class OptimizedInferenceEngine:
    def __init__(self,
                 model_path: Optional[str] = None,
                 device: str = "cuda",
                 use_tensorrt: bool = True,
                 use_fp16: bool = True,
                 frame_buffer_size: int = 32,
                 batch_size: int = 1):
        
        self.test_mode = os.getenv("LWM_TEST_FAST", "0").lower() in {"1", "true", "yes"}
        self.device = torch.device("cpu" if self.test_mode else device)
        self.use_fp16 = use_fp16
        self.frame_buffer_size = frame_buffer_size
        self.batch_size = batch_size

        self.model_path = model_path
        if self.test_mode:
            self.model = None
            self.latent_shape = (1, 1, 1)
        else:
            self.model = self._load_model(model_path)
            self.latent_shape = self._infer_latent_shape()

        if not self.test_mode and use_tensorrt and device == "cuda":
            self._optimize_with_tensorrt()

        self.frame_buffer = deque(maxlen=frame_buffer_size)
        self.latent_cache = {}

        self._init_buffers()

    def seed_everything(self, seed: int):
        """Seed torch and numpy for deterministic behavior within this engine scope."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _load_model(self, model_path: Optional[str]) -> WorldModel:
        """
        Build a WorldModel and optionally load weights from a checkpoint.
        The expected checkpoint formats are either a plain state_dict for the
        composite model, or a dict with keys like 'vqvae' and 'dynamics', or a
        training checkpoint with 'model_state_dict'.
        """
        # Construct default components
        vqvae = ImprovedVQVAE()
        # Infer flattened latent dimension from encoder output shape
        latent_c, latent_h, latent_w = self._infer_latent_shape_static(vqvae, (3, 256, 256))
        flattened_latent_dim = int(latent_c * latent_h * latent_w)
        dynamics = DynamicsModel(latent_dim=flattened_latent_dim)
        model = WorldModel(vqvae, dynamics)
        
        # Try to load weights if provided and exists
        if model_path and os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                state_dict = None
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    elif 'vqvae' in checkpoint or 'dynamics' in checkpoint:
                        # Load submodules if present
                        if 'vqvae' in checkpoint:
                            model.vqvae.load_state_dict(checkpoint['vqvae'], strict=False)
                        if 'dynamics' in checkpoint:
                            model.dynamics.load_state_dict(checkpoint['dynamics'], strict=False)
                    else:
                        state_dict = checkpoint
                if state_dict is not None:
                    model.load_state_dict(state_dict, strict=False)
                print(f"Loaded world model checkpoint from {model_path}")
            except Exception as e:
                print(f"Warning: Failed to load checkpoint '{model_path}': {e}. Using randomly initialized weights.")
        else:
            if model_path:
                print(f"Warning: Checkpoint '{model_path}' not found. Using randomly initialized weights.")
        
        model.to(self.device)
        model.eval()
        if self.use_fp16:
            model = model.half()
        return model

    def _ensure_model_available(self) -> None:
        if self.test_mode and self.model is None:
            raise RuntimeError("Model is unavailable in test fast mode")

    def reload_model(self, model_path: str):
        """Reload model weights from a new checkpoint path."""
        self.model_path = model_path
        self.model = self._load_model(model_path)
        self.latent_shape = self._infer_latent_shape()
        self._init_buffers()

    def _infer_latent_shape_static(self, vqvae: ImprovedVQVAE, input_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        with torch.no_grad():
            x = torch.zeros(1, *input_shape)
            z = vqvae.encoder(x)
            return (z.shape[1], z.shape[2], z.shape[3])

    def _infer_latent_shape(self) -> Tuple[int, int, int]:
        if self.test_mode:
            return (1, 1, 1)
        with torch.no_grad():
            x = torch.zeros(1, 3, 256, 256, device=self.device)
            if self.use_fp16:
                x = x.half()
            z = self.model.vqvae.encoder(x)
            return (z.shape[1], z.shape[2], z.shape[3])
    
    def _optimize_with_tensorrt(self):
        try:
            import torch_tensorrt
            
            example_input = torch.randn(1, 3, 256, 256).to(self.device)
            if self.use_fp16:
                example_input = example_input.half()
            
            self.model.vqvae = torch_tensorrt.compile(
                self.model.vqvae,
                inputs=[example_input],
                enabled_precisions={torch.float16} if self.use_fp16 else {torch.float32}
            )
            
            print("TensorRT optimization successful")
        except Exception as e:
            print(f"TensorRT optimization failed: {e}")
    
    def _init_buffers(self):
        self.current_latent = None
        self.generation_thread = None
        self.is_running = False
        
    @torch.no_grad()
    def process_frame(self, frame: np.ndarray, action: Optional[int] = None):
        """Backward-compatible single-session processing using internal state."""
        generated_frame, new_latent = self.process_frame_with_state(frame, action, self.current_latent)
        self.current_latent = new_latent
        self.frame_buffer.append(generated_frame)
        return generated_frame

    @torch.no_grad()
    def process_frame_with_state(self, frame: np.ndarray, action: Optional[int] = None,
                                 latent: Optional[torch.Tensor] = None) -> Tuple[np.ndarray, Optional[torch.Tensor]]:
        """
        Stateless processing: accept an optional latent state and return the updated state
        alongside the generated frame. Use this for multi-session scenarios.
        """
        if self.test_mode:
            # In fast test mode, bypass heavy model usage and return a deterministic variant of the input.
            altered = frame.copy()
            altered[:, :, 0] = (altered[:, :, 0] // 2)  # simple visual change for determinism
            return altered, None

        self._ensure_model_available()
        frame_tensor = torch.from_numpy(frame).to(self.device)
        frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)
        frame_tensor = frame_tensor.float() / 255.0

        if self.use_fp16:
            frame_tensor = frame_tensor.half()
        
        current_latent = latent
        if current_latent is None:
            latent_feat, _ = self.model.vqvae.encode(frame_tensor)
            current_latent = latent_feat.flatten(1)
        
        if action is not None:
            try:
                action_tensor = torch.tensor([action], device=self.device)
                
                context = current_latent.unsqueeze(1)
                next_latent = self.model.dynamics(context, action_tensor.unsqueeze(1))
                next_latent = next_latent[:, -1, :]
                
                new_latent = next_latent
                
                next_latent_reshaped = new_latent.reshape(1, *self.latent_shape)
                
                generated_frame = self.model.vqvae.decode(next_latent_reshaped)
                
                generated_frame = generated_frame[0].cpu()
                if self.use_fp16:
                    generated_frame = generated_frame.float()
                
                generated_frame = (generated_frame * 255).clamp(0, 255).byte()
                generated_frame = generated_frame.permute(1, 2, 0).numpy()
                return generated_frame, new_latent
            except Exception as e:
                # Fallback: return original frame and keep latent state on failure
                print(f"Warning: dynamics step failed ({e}). Returning input frame.")
                return frame, current_latent
        
        return frame, current_latent
    
    def _generate_initial_frame(self, prompt: Optional[str], seed: Optional[int]) -> np.ndarray:
        if prompt:
            return self._generate_from_prompt(prompt, seed=seed)
        rng = np.random.default_rng(seed)
        return rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)

    @torch.no_grad()
    def _initialize_state_from_frame(
        self,
        frame: np.ndarray,
        seed: Optional[int],
    ) -> Tuple[np.ndarray, Optional[torch.Tensor]]:
        if self.test_mode:
            return frame, None
        if seed is not None:
            self.seed_everything(seed)
        generated_frame, latent = self.process_frame_with_state(frame, action=None, latent=None)
        return generated_frame, latent

    def _prepare_initial_inference(
        self,
        prompt: Optional[str],
        seed: Optional[int],
    ) -> Tuple[np.ndarray, Optional[torch.Tensor]]:
        initial_frame = self._generate_initial_frame(prompt, seed)
        return self._initialize_state_from_frame(initial_frame, seed)

    def generate_interactive(self, initial_prompt: str = None, seed: Optional[int] = None):
        self.is_running = True

        current_frame, current_latent = self._prepare_initial_inference(initial_prompt, seed)

        self.current_latent = current_latent
        self.frame_buffer.clear()
        self.frame_buffer.append(current_frame)

        return current_frame
    
    def _generate_from_prompt(self, prompt: str, seed: Optional[int] = None):
        # Placeholder prompt-to-initialization: deterministic gray background with seeded dots
        img = np.ones((256, 256, 3), dtype=np.uint8) * 128
        if seed is not None:
            rng = np.random.default_rng(seed)
            for _ in range(200):
                x, y = rng.integers(0, 256, size=2)
                color = rng.integers(80, 200, size=3, dtype=np.uint8)
                img[y, x] = color
        return img
    
    def step(self, action: int):
        if not self.is_running:
            raise RuntimeError("Engine not running. Call generate_interactive first.")

        start_time = time.time()

        if self.test_mode:
            base = np.zeros((256, 256, 3), dtype=np.uint8)
            base[:, :, 1] = min(max(action, 0), 255)
            frame = base
            self.current_latent = None
        else:
            frame, self.current_latent = self.process_frame_with_state(
                np.zeros((256, 256, 3), dtype=np.uint8),
                action,
                self.current_latent
            )

        inference_time = time.time() - start_time
        fps = 1.0 / inference_time if inference_time > 0 else 0
        
        self.frame_buffer.append(frame)
        
        return frame, {'fps': fps, 'inference_time': inference_time}
    
    def stop(self):
        self.is_running = False
        self.current_latent = None
        self.frame_buffer.clear()
    
    def export_onnx(self, output_path: str):
        dummy_input = torch.randn(1, 3, 256, 256).to(self.device)
        if self.use_fp16:
            dummy_input = dummy_input.half()
        
        torch.onnx.export(
            self.model.vqvae,
            dummy_input,
            f"{output_path}/vqvae.onnx",
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
            opset_version=11
        )
        
        flattened = int(np.prod(self.latent_shape))
        dummy_latent = torch.randn(1, 1, flattened).to(self.device)
        dummy_action = torch.tensor([[0]]).to(self.device)
        
        torch.onnx.export(
            self.model.dynamics,
            (dummy_latent, dummy_action),
            f"{output_path}/dynamics.onnx",
            input_names=['latent', 'action'],
            output_names=['next_latent'],
            dynamic_axes={
                'latent': {0: 'batch_size', 1: 'sequence'},
                'action': {0: 'batch_size', 1: 'sequence'},
                'next_latent': {0: 'batch_size', 1: 'sequence'}
            },
            opset_version=11
        )
        
        print(f"Models exported to ONNX at {output_path}")


class BatchedInferenceEngine(OptimizedInferenceEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.user_states: Dict[str, SessionState] = {}
        flag = os.getenv("LWM_ENABLE_TOY_WORLD", "auto").lower()
        if self.test_mode:
            self.use_toy_world = True
        elif flag == "auto":
            self.use_toy_world = not (self.model_path and os.path.exists(self.model_path))
        else:
            self.use_toy_world = flag in {"1", "true", "yes", "on"}

        self._toy_world = ToyWorldSimulator()
        
    def create_session(self, session_id: str) -> SessionState:
        state = SessionState(
            latent=None,
            last_frame=None,
            frame_buffer=deque(maxlen=self.frame_buffer_size),
            prompt=None,
            seed=None,
            running=False,
            world=None,
        )
        self.user_states[session_id] = state
        return state

    def has_session(self, session_id: str) -> bool:
        return session_id in self.user_states

    def is_session_running(self, session_id: str) -> bool:
        state = self.user_states.get(session_id)
        return bool(state and state.running)

    def get_session_state(self, session_id: str) -> SessionState:
        if session_id not in self.user_states:
            raise KeyError(f"Unknown session_id: {session_id}")
        return self.user_states[session_id]

    def start_session(self, session_id: str, prompt: Optional[str] = None,
                      seed: Optional[int] = None) -> np.ndarray:
        state = self.user_states.get(session_id) or self.create_session(session_id)

        state.latent = None
        state.last_frame = None
        state.frame_buffer.clear()
        state.prompt = prompt
        state.seed = seed

        if self.use_toy_world:
            world_state = self._toy_world.create_state(prompt, seed)
            generated_frame = self._toy_world.render(world_state)
            state.world = world_state
            state.latent = None
        else:
            generated_frame, latent = self._prepare_initial_inference(prompt, seed)
            state.latent = latent
            state.world = None

        state.last_frame = generated_frame
        state.frame_buffer.append(generated_frame)
        state.running = True

        return generated_frame

    def step_session(self, session_id: str, action: int) -> Tuple[np.ndarray, Dict[str, float]]:
        if session_id not in self.user_states:
            raise KeyError(f"Unknown session_id: {session_id}")

        state = self.user_states[session_id]
        if not state.running:
            raise RuntimeError("Session not initialized. Call start_session first.")

        start_time = time.time()

        if self.use_toy_world and state.world is not None:
            new_world, frame_output, world_metrics = self._toy_world.step(
                state.world,
                int(action) if action is not None else 0,
            )
            state.world = new_world
            state.latent = None
            inference_time = time.time() - start_time
            fps = 1.0 / inference_time if inference_time > 0 else 0.0
            metrics = {
                'fps': fps,
                'inference_time': inference_time,
            }
            metrics.update(world_metrics)
        else:
            frame_input = state.last_frame
            if frame_input is None:
                frame_input = np.ones((256, 256, 3), dtype=np.uint8) * 128

            frame_output, new_latent = self.process_frame_with_state(
                frame_input,
                action=int(action) if action is not None else None,
                latent=state.latent
            )

            inference_time = time.time() - start_time
            fps = 1.0 / inference_time if inference_time > 0 else 0.0

            state.latent = new_latent
            metrics = {'fps': fps, 'inference_time': inference_time}

        state.last_frame = frame_output
        state.frame_buffer.append(frame_output)
        state.running = True

        return frame_output, metrics

    def stop_session(self, session_id: str):
        state = self.user_states.pop(session_id, None)
        if state:
            state.frame_buffer.clear()
            state.latent = None
            state.last_frame = None
            state.running = False
            state.world = None

    def process_batch(self, requests: List[Dict[str, Any]]):
        batch_actions = []
        session_ids = []

        for req in requests:
            session_id = req['session_id']
            action = req.get('action', None)

            if session_id not in self.user_states:
                self.create_session(session_id)

            session_ids.append(session_id)
            batch_actions.append(action)

        results = []
        for i, session_id in enumerate(session_ids):
            state = self.user_states[session_id]

            # If we don't have a last frame yet, initialize with gray noise
            if self.use_toy_world and state.world is not None:
                new_world, frame, world_metrics = self._toy_world.step(
                    state.world,
                    int(batch_actions[i]) if batch_actions[i] is not None else 0,
                )
                state.world = new_world
                state.latent = None
                state.running = True
                state.frame_buffer.append(frame)
            else:
                if state.last_frame is None:
                    state.last_frame = np.ones((256, 256, 3), dtype=np.uint8) * 128

                frame, new_latent = self.process_frame_with_state(
                    state.last_frame,
                    batch_actions[i],
                    state.latent
                )
                state.latent = new_latent
                state.frame_buffer.append(frame)
                state.running = True

            state.last_frame = frame
            results.append({
                'session_id': session_id,
                'frame': frame
            })

        return results

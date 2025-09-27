from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import RedirectResponse
from pydantic import BaseModel
import numpy as np
import base64
import io
from PIL import Image
import json
from typing import Optional, List, Dict, Any
import uuid
from datetime import datetime
import torch

from ..inference.engine import BatchedInferenceEngine
from ..utils.logging import get_logger
from .. import config as cfg


engine: Optional[BatchedInferenceEngine] = None
sessions: Dict[str, Dict[str, Any]] = {}
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine

    engine = BatchedInferenceEngine(
        model_path=cfg.DEFAULT_WORLD_MODEL_CKPT,
        device=("cuda" if torch.cuda.is_available() else "cpu") if cfg.DEVICE == "cuda" else cfg.DEVICE,
        use_tensorrt=cfg.USE_TENSORRT and torch.cuda.is_available(),
        use_fp16=cfg.USE_FP16 and torch.cuda.is_available(),
        batch_size=cfg.BATCH_SIZE,
    )
    logger.info("World Model Engine loaded successfully")

    try:
        app.mount(
            "/demo",
            StaticFiles(directory=str((cfg.PROJECT_ROOT / "demo").resolve()), html=True),
            name="demo",
        )
        logger.info("Mounted static demo at /demo")
    except Exception as exc:
        logger.warning("Could not mount demo static files: %s", exc)

    try:
        yield
    finally:
        if engine is not None:
            for session_id in list(sessions.keys()):
                engine.stop_session(session_id)
            engine = None


app = FastAPI(title="Lightweight World Model API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class InitRequest(BaseModel):
    prompt: Optional[str] = None
    seed: Optional[int] = None


class ActionRequest(BaseModel):
    session_id: str
    action: int


class SessionInfo(BaseModel):
    session_id: str
    created_at: str
    frames_generated: int
    current_fps: float


@app.get("/")
async def root():
    # Redirect to demo UI if available
    try:
        return RedirectResponse(url="/demo/")
    except Exception:
        return {"message": "Lightweight World Model API", "status": "running"}


@app.post("/session/create")
async def create_session(request: InitRequest):
    session_id = str(uuid.uuid4())

    try:
        initial_frame = engine.start_session(session_id, prompt=request.prompt, seed=request.seed)
    except Exception as e:
        logger.exception("Failed to start session %s: %s", session_id, e)
        raise HTTPException(status_code=500, detail="Engine initialization failed")
    
    sessions[session_id] = {
        "created_at": datetime.now().isoformat(),
        "frames_generated": 1,
        "current_fps": 0,
        "prompt": request.prompt,
        "seed": request.seed,
        "actions": []
    }
    frame_base64 = encode_frame_current(initial_frame)

    return {
        "session_id": session_id,
        "initial_frame": frame_base64,
        "status": "created"
    }


@app.post("/session/step")
async def step_session(request: ActionRequest):
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    # Validate action range (0-255)
    if not isinstance(request.action, int) or request.action < 0 or request.action > 255:
        raise HTTPException(status_code=400, detail="Invalid action value; expected 0-255 integer")
    
    try:
        frame, metrics = engine.step_session(request.session_id, request.action)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not initialized")
    except Exception as e:
        logger.exception("Error stepping session: %s", e)
        raise HTTPException(status_code=500, detail="Engine step failed")
    
    sessions[request.session_id]["frames_generated"] += 1
    sessions[request.session_id]["current_fps"] = metrics["fps"]
    sessions[request.session_id]["actions"].append({
        "t": datetime.now().isoformat(),
        "action": request.action
    })
    
    frame_base64 = encode_frame_current(frame)
    
    return {
        "frame": frame_base64,
        "metrics": metrics,
        "frame_number": sessions[request.session_id]["frames_generated"]
    }


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    del sessions[session_id]
    engine.stop_session(session_id)
    
    return {"status": "deleted", "session_id": session_id}


@app.get("/session/{session_id}/info")
async def get_session_info(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return SessionInfo(
        session_id=session_id,
        created_at=sessions[session_id]["created_at"],
        frames_generated=sessions[session_id]["frames_generated"],
        current_fps=sessions[session_id]["current_fps"]
    )


@app.get("/sessions")
async def list_sessions():
    return {
        "total": len(sessions),
        "sessions": list(sessions.keys())
    }


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    
    if session_id not in sessions:
        sessions[session_id] = {
            "created_at": datetime.now().isoformat(),
            "frames_generated": 0,
            "current_fps": 0,
            "prompt": None,
            "seed": None,
            "actions": []
        }
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "action":
                action = message["action"]
                if not engine.is_session_running(session_id):
                    session_meta = sessions.get(session_id, {})
                    engine.start_session(
                        session_id,
                        prompt=session_meta.get("prompt"),
                        seed=session_meta.get("seed"),
                    )

                frame, metrics = engine.step_session(session_id, action)

                sessions[session_id]["frames_generated"] += 1
                sessions[session_id]["current_fps"] = metrics["fps"]
                sessions[session_id]["actions"].append({
                    "t": datetime.now().isoformat(),
                    "action": action
                })
                
                frame_base64 = encode_frame_current(frame)
                
                await websocket.send_json({
                    "type": "frame",
                    "data": frame_base64,
                    "metrics": metrics,
                    "frame_number": sessions[session_id]["frames_generated"]
                })
                
            elif message["type"] == "reset":
                prompt = message.get("prompt", None)
                seed = message.get("seed", None)
                initial_frame = engine.start_session(session_id, prompt=prompt, seed=seed)
                
                sessions[session_id]["frames_generated"] = 1
                sessions[session_id]["prompt"] = prompt
                sessions[session_id]["seed"] = seed
                sessions[session_id]["actions"] = []
                
                frame_base64 = encode_frame_current(initial_frame)
                
                await websocket.send_json({
                    "type": "reset",
                    "data": frame_base64
                })
                
    except WebSocketDisconnect:
        print(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        print(f"Error in WebSocket for session {session_id}: {e}")
        await websocket.close()


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "engine_loaded": engine is not None,
        "active_sessions": len(sessions),
        "cuda_available": torch.cuda.is_available()
    }


# Models endpoints
@app.get("/models")
async def list_models():
    ckpt_dir = cfg.CHECKPOINTS_DIR
    models: List[Dict[str, Any]] = []
    try:
        for p in ckpt_dir.glob("**/*.pt"):
            try:
                size = p.stat().st_size
                mtime = p.stat().st_mtime
            except Exception:
                size = None; mtime = None
            models.append({
                "id": p.name,
                "path": str(p.resolve()),
                "size": size,
                "modified": mtime
            })
    except Exception as e:
        logger.warning("Failed to list models: %s", e)
    return {
        "models": models,
        "current_model": engine.model_path
    }


class LoadModelRequest(BaseModel):
    id: str  # filename under checkpoints dir


@app.post("/models/load")
async def load_model(req: LoadModelRequest):
    target = (cfg.CHECKPOINTS_DIR / req.id).resolve()
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"Model not found: {target}")
    try:
        engine.reload_model(str(target))
        return {"status": "ok", "current_model": engine.model_path}
    except Exception as e:
        logger.exception("Failed to load model: %s", e)
        raise HTTPException(status_code=500, detail="Failed to load model")


@app.get("/models/status")
async def model_status():
    return {"current_model": engine.model_path, "ready": engine is not None}


# Snapshot & Replay
class SnapshotRequest(BaseModel):
    session_id: str


@app.post("/session/snapshot")
async def snapshot_session(req: SnapshotRequest):
    if req.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    state = sessions[req.session_id]
    snapshot = {
        "session_id": req.session_id,
        "prompt": state.get("prompt"),
        "seed": state.get("seed"),
        "actions": state.get("actions", []),
        "settings": {
            "resolution": 256,
        },
        "model": engine.model_path,
        "created_at": datetime.now().isoformat(),
    }
    return snapshot


class ReplayRequest(BaseModel):
    prompt: Optional[str] = None
    seed: Optional[int] = None
    actions: List[Dict[str, Any]]
    model: Optional[str] = None
    settings: Optional[Dict[str, Any]] = None
    verify: Optional[bool] = False
    verify_mode: Optional[str] = "byte"  # 'byte' or 'ssim'
    tolerance: Optional[float] = 0.98


@app.post("/session/replay")
async def replay_session(req: ReplayRequest):
    # Build a temporary engine to avoid altering global session state
    tmp_engine = BatchedInferenceEngine(
        model_path=req.model or engine.model_path,
        device="cpu",  # favor determinism
        use_tensorrt=False,
        use_fp16=False,
        batch_size=1,
    )
    # Seed and generate initial frame
    replay_session_id = "replay-primary"
    initial = tmp_engine.start_session(replay_session_id, prompt=req.prompt, seed=req.seed)
    frames_np: List[np.ndarray] = []
    resized0 = resize_frame_current(initial)
    frames_np.append(resized0)
    frames = [frame_to_base64(resized0)]
    for act in req.actions:
        a = int(act.get("action", 0))
        frame, _ = tmp_engine.step_session(replay_session_id, a)
        r = resize_frame_current(frame)
        frames.append(frame_to_base64(r))
        frames_np.append(r)
    tmp_engine.stop_session(replay_session_id)
    result = {"frames": frames, "count": len(frames)}
    if req.verify:
        # Rerun to check determinism (byte equality)
        tmp2 = BatchedInferenceEngine(
            model_path=req.model or engine.model_path,
            device="cpu",
            use_tensorrt=False,
            use_fp16=False,
            batch_size=1,
        )
        replay_secondary_id = "replay-verify"
        initial2 = tmp2.start_session(replay_secondary_id, prompt=req.prompt, seed=req.seed)
        frames2_np: List[np.ndarray] = []
        resized02 = resize_frame_current(initial2)
        frames2_np.append(resized02)
        frames2 = [frame_to_base64(resized02)]
        for act in req.actions:
            a = int(act.get("action", 0))
            f2, _ = tmp2.step_session(replay_secondary_id, a)
            r2 = resize_frame_current(f2)
            frames2.append(frame_to_base64(r2))
            frames2_np.append(r2)
        tmp2.stop_session(replay_secondary_id)
        if (req.verify_mode or "byte").lower() == "ssim":
            # Compute global SSIM per frame
            vals = [compute_ssim(a, b) for a, b in zip(frames_np, frames2_np)]
            min_ssim = float(min(vals)) if vals else 1.0
            avg_ssim = float(sum(vals) / len(vals)) if vals else 1.0
            tol = req.tolerance or 0.98
            result["verification"] = {
                "mode": "ssim",
                "min_ssim": min_ssim,
                "avg_ssim": avg_ssim,
                "tolerance": tol,
                "identical": min_ssim >= tol,
            }
        else:
            equal = len(frames2) == len(frames) and all(f1 == f2 for f1, f2 in zip(frames, frames2))
            result["verification"] = {"mode": "byte", "identical": equal}
    return result


# Settings (minimal placeholder)
class Settings(BaseModel):
    resolution: int = 256
    fps_target: Optional[int] = None
    overlay: bool = True


_settings = Settings()


@app.get("/settings")
async def get_settings():
    return _settings.dict()


@app.post("/settings")
async def set_settings(s: Settings):
    global _settings
    _settings = s
    return {"status": "ok"}


@app.post("/batch/process")
async def batch_process(requests: list[ActionRequest]):
    def ensure_session_meta(session_id: str) -> None:
        if session_id not in sessions:
            sessions[session_id] = {
                "created_at": datetime.now().isoformat(),
                "frames_generated": 0,
                "current_fps": 0,
                "prompt": None,
                "seed": None,
                "actions": [],
            }

    batch_requests = [
        {"session_id": req.session_id, "action": req.action}
        for req in requests
    ]
    for req in requests:
        ensure_session_meta(req.session_id)

    results = engine.process_batch(batch_requests)

    response = []
    for idx, result in enumerate(results):
        session_id = result["session_id"]
        frame = result["frame"]

        sessions[session_id]["frames_generated"] += 1
        if idx < len(requests):
            sessions[session_id]["actions"].append({
                "t": datetime.now().isoformat(),
                "action": requests[idx].action,
            })

        response.append({
            "session_id": session_id,
            "frame": encode_frame_current(frame),
            "frame_number": sessions[session_id]["frames_generated"]
        })
    
    return response


def frame_to_base64(frame: np.ndarray) -> str:
    img = Image.fromarray(frame.astype('uint8'))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode()


def base64_to_frame(base64_str: str) -> np.ndarray:
    img_data = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_data))
    return np.array(img)

def encode_frame_current(frame: np.ndarray) -> str:
    """Apply current settings (e.g., resolution) before encoding."""
    try:
        target = _settings.resolution if isinstance(_settings.resolution, int) else 256
    except Exception:
        target = 256
    if frame.shape[0] != target or frame.shape[1] != target:
        img = Image.fromarray(frame.astype('uint8')).resize((target, target), Image.BILINEAR)
        return frame_to_base64(np.array(img))
    return frame_to_base64(frame)

def resize_frame_current(frame: np.ndarray) -> np.ndarray:
    try:
        target = _settings.resolution if isinstance(_settings.resolution, int) else 256
    except Exception:
        target = 256
    if frame.shape[0] != target or frame.shape[1] != target:
        img = Image.fromarray(frame.astype('uint8')).resize((target, target), Image.BILINEAR)
        return np.array(img)
    return frame

def compute_ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute a simple global SSIM between two RGB uint8 images scaled to [0,1]."""
    import numpy as _np
    def to_gray(x: _np.ndarray) -> _np.ndarray:
        if x.ndim == 3 and x.shape[2] >= 3:
            r, g, bl = x[..., 0], x[..., 1], x[..., 2]
            y = 0.299 * r + 0.587 * g + 0.114 * bl
            return y
        return x.astype(_np.float32)
    x = to_gray(a.astype(_np.float32)) / 255.0
    y = to_gray(b.astype(_np.float32)) / 255.0
    x_mean = _np.mean(x)
    y_mean = _np.mean(y)
    x_var = _np.var(x)
    y_var = _np.var(y)
    cov = _np.mean((x - x_mean) * (y - y_mean))
    C1 = (0.01 ** 2)
    C2 = (0.03 ** 2)
    numerator = (2 * x_mean * y_mean + C1) * (2 * cov + C2)
    denominator = (x_mean ** 2 + y_mean ** 2 + C1) * (x_var + y_var + C2)
    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0
    val = float(numerator / denominator)
    # Clamp to [0,1]
    return max(0.0, min(1.0, val))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app=app,
        host=cfg.HOST,
        port=cfg.PORT,
        reload=True,
        log_level="info",
    )

"""
Enhanced API Server with Memory Management and Predictive Inference

This is an improved version of the API server that includes:
- Memory-aware session management with automatic cleanup
- Predictive inference for zero-latency user experience
- Progressive model loading with status updates
- Comprehensive error recovery and health monitoring
"""

from contextlib import asynccontextmanager
import asyncio
import time
from typing import Optional, List, Dict, Any
from collections import defaultdict
from datetime import datetime, timedelta

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import RedirectResponse, JSONResponse
from pydantic import BaseModel, Field
import numpy as np
import base64
import io
from PIL import Image
import json
import uuid
import torch

from ..inference.engine import BatchedInferenceEngine
from ..inference.predictive_engine import PredictiveInferenceEngine, create_predictive_engine
from ..utils.memory_manager import MemoryAwareSessionManager
from ..utils.logging import get_logger
from ..utils.exceptions import (
    SessionNotFoundError,
    SessionCreationError,
    SessionLimitError,
    MemoryError as WMMemoryError,
    InvalidActionError,
    InferenceError
)
from .model_loading import ProgressiveModelLoader, LoadModelRequest, LoadModelResponse, JobStatusResponse
from .. import config as cfg


# Global instances
predictive_engine: Optional[PredictiveInferenceEngine] = None
memory_manager: Optional[MemoryAwareSessionManager] = None
progressive_loader = None
logger = get_logger(__name__)


# Simple rate limiter
class RateLimiter:
    """Simple in-memory rate limiter"""
    def __init__(self, requests: int = 100, window: int = 60):
        self.requests = requests
        self.window = window
        self.clients: Dict[str, List[datetime]] = defaultdict(list)

    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed"""
        now = datetime.now()
        cutoff = now - timedelta(seconds=self.window)

        # Clean old requests
        self.clients[client_id] = [
            req_time for req_time in self.clients[client_id]
            if req_time > cutoff
        ]

        # Check limit
        if len(self.clients[client_id]) >= self.requests:
            return False

        # Add request
        self.clients[client_id].append(now)
        return True

rate_limiter = RateLimiter(requests=cfg.RATE_LIMIT_REQUESTS, window=cfg.RATE_LIMIT_WINDOW)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictive_engine, memory_manager, progressive_loader

    try:
        # Initialize progressive loader
        device = ("cuda" if torch.cuda.is_available() else "cpu") if cfg.DEVICE == "cuda" else cfg.DEVICE
        progressive_loader = ProgressiveModelLoader(device=device)
        logger.info("Progressive model loader initialized")

        # Initialize memory manager
        memory_manager = MemoryAwareSessionManager(
            max_sessions=cfg.MAX_SESSIONS,
            ttl_seconds=cfg.SESSION_TTL_SECONDS,
            memory_check_interval=cfg.MEMORY_CHECK_INTERVAL,
            max_memory_gb=cfg.MAX_GPU_MEMORY_GB
        )

        # Start memory monitoring
        await memory_manager.start_monitoring()
        logger.info("Memory manager initialized and monitoring started")

        # Initialize predictive engine
        predictive_engine = create_predictive_engine(
            model_path=cfg.DEFAULT_WORLD_MODEL_CKPT,
            device=device,
            use_tensorrt=cfg.USE_TENSORRT and torch.cuda.is_available(),
            use_fp16=cfg.USE_FP16 and torch.cuda.is_available(),
            batch_size=cfg.BATCH_SIZE,
            prediction_horizon=3
        )
        logger.info("Predictive World Model Engine loaded successfully")

        # Mount demo static files
        try:
            app.mount(
                "/demo",
                StaticFiles(directory=str((cfg.PROJECT_ROOT / "demo").resolve()), html=True),
                name="demo",
            )
            logger.info("Mounted static demo at /demo")
        except Exception as exc:
            logger.warning("Could not mount demo static files: %s", exc)

        yield

    finally:
        # Cleanup on shutdown
        if memory_manager:
            await memory_manager.stop_monitoring()
            logger.info("Memory manager monitoring stopped")

        if predictive_engine:
            # Stop all sessions
            for session_id in list(memory_manager.cache.sessions.keys()):
                predictive_engine.stop_session_predictive(session_id)
            logger.info("All sessions stopped")


app = FastAPI(
    title="Enhanced Lightweight World Model API",
    version="0.2.0",
    lifespan=lifespan,
    description="Real-time world model with predictive inference and memory management"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Restrict to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Apply rate limiting to API requests"""
    # Skip rate limiting for static files and health checks
    if request.url.path.startswith("/demo") or request.url.path == "/":
        return await call_next(request)

    # Get client identifier (IP address)
    client_ip = request.client.host if request.client else "unknown"

    # Check rate limit
    if not rate_limiter.is_allowed(client_ip):
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Please try again later."}
        )

    return await call_next(request)


# Enhanced request models
class InitRequest(BaseModel):
    prompt: Optional[str] = None
    seed: Optional[int] = None
    priority: Optional[float] = Field(default=1.0, ge=0.1, le=2.0)


class ActionRequest(BaseModel):
    session_id: str = Field(..., pattern=r'^[a-f0-9-]{36}$')  # UUID format
    action: int = Field(..., ge=0, le=255)


class SessionInfo(BaseModel):
    session_id: str
    created_at: str
    frames_generated: int
    current_fps: float
    memory_usage_mb: float
    priority: float
    prediction_stats: Dict[str, Any]


# Helper functions
def calculate_session_priority(
    prompt: Optional[str],
    frames_generated: int,
    recent_activity: bool
) -> float:
    """
    Calculate session priority based on various factors.

    Args:
        prompt: User-provided prompt (if any)
        frames_generated: Number of frames generated so far
        recent_activity: Whether session had recent activity

    Returns:
        Priority score between 0.1 and 2.0
    """
    base_priority = 1.0

    # Boost priority for sessions with prompts (more intentional)
    if prompt:
        base_priority += 0.3

    # Boost priority for active sessions
    if recent_activity:
        base_priority += 0.2

    # Slightly boost priority for sessions with more progress
    if frames_generated > 50:
        base_priority += 0.1

    return min(base_priority, 2.0)  # Cap at 2.0


async def estimate_session_memory_usage(session_id: str) -> float:
    """
    Estimate memory usage for a session in MB.

    Args:
        session_id: Session identifier

    Returns:
        Estimated memory usage in MB (50-114 MB typical range)
    """
    try:
        # Base memory per session (rough estimate)
        base_memory = 50.0  # MB

        # Get session data
        session_data = memory_manager.get_session(session_id)
        if session_data:
            frames_generated = session_data.get('frames_generated', 0)
            # Assume ~2MB per frame buffer entry
            buffer_memory = min(frames_generated * 2.0, 64.0)  # Cap at 64MB
            return base_memory + buffer_memory

        return base_memory
    except Exception:
        return 50.0  # Default estimate


@app.get("/")
async def root() -> Dict[str, Any]:
    """
    Root endpoint with system status.

    Returns:
        System status including memory and session information
    """
    try:
        memory_report = memory_manager.get_memory_report() if memory_manager else {}
        return {
            "message": "Enhanced Lightweight World Model API",
            "status": "running",
            "version": "0.2.0",
            "features": ["predictive_inference", "memory_management", "auto_cleanup"],
            "memory_status": memory_report.get('memory', {}).get('status', 'unknown'),
            "active_sessions": memory_report.get('sessions', {}).get('session_count', 0)
        }
    except Exception:
        return {"message": "Enhanced Lightweight World Model API", "status": "running"}


@app.post("/session/create")
async def create_session(
    request: InitRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Create new session with memory management.

    Args:
        request: Session initialization request
        background_tasks: FastAPI background tasks

    Returns:
        Session ID, initial frame, and metadata

    Raises:
        HTTPException: If memory limit exceeded or session creation fails
    """
    session_id = str(uuid.uuid4())

    # Check memory pressure before creating session
    if memory_manager:
        memory_report = memory_manager.get_memory_report()
        pressure = memory_report.get('pressure', 0.0)

        if pressure > 0.99:  # Only block at 99% to allow high-baseline systems
            raise WMMemoryError(
                f"Service temporarily unavailable due to high memory usage ({pressure:.2f})"
            )

    try:
        # Start predictive session
        initial_frame = await predictive_engine.start_session_predictive(
            session_id,
            prompt=request.prompt,
            seed=request.seed
        )

        # Calculate priority
        priority = request.priority or calculate_session_priority(
            request.prompt, 0, True
        )

        # Create session metadata
        session_metadata = {
            "created_at": datetime.now().isoformat(),
            "frames_generated": 1,
            "current_fps": 0,
            "prompt": request.prompt,
            "seed": request.seed,
            "priority": priority,
            "actions": []
        }

        # Add to memory manager
        if memory_manager:
            success = memory_manager.create_session(session_id, session_metadata)
            if not success:
                # Cleanup and raise error
                predictive_engine.stop_session_predictive(session_id)
                raise SessionLimitError(
                    "Cannot create session due to resource constraints"
                )

            # Schedule memory usage estimation
            background_tasks.add_task(update_session_memory_usage, session_id)

        frame_base64 = encode_frame_current(initial_frame)

        return {
            "session_id": session_id,
            "initial_frame": frame_base64,
            "status": "created",
            "priority": priority
        }

    except (WMMemoryError, SessionLimitError) as e:
        # Cleanup on resource constraint failures
        if predictive_engine:
            predictive_engine.stop_session_predictive(session_id)
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Failed to start session %s: %s", session_id, e)
        # Cleanup on failure
        if predictive_engine:
            predictive_engine.stop_session_predictive(session_id)
        raise HTTPException(status_code=500, detail=f"Session creation failed: {str(e)}")


async def update_session_memory_usage(session_id: str) -> None:
    """
    Background task to update session memory usage.

    Args:
        session_id: Session identifier to update
    """
    try:
        memory_usage = await estimate_session_memory_usage(session_id)
        session_data = memory_manager.get_session(session_id)
        if session_data and memory_manager:
            memory_manager.update_session(
                session_id,
                session_data,
                memory_usage_mb=memory_usage,
                priority=session_data.get('priority', 1.0)
            )
    except Exception as e:
        logger.warning(f"Failed to update memory usage for session {session_id}: {e}")


@app.post("/session/step")
async def step_session(
    request: ActionRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Step session with predictive inference and memory tracking.

    Args:
        request: Action request with session ID and action value
        background_tasks: FastAPI background tasks

    Returns:
        Next frame and performance metrics

    Raises:
        SessionNotFoundError: If session doesn't exist
        InvalidActionError: If action value is invalid
    """
    session_data = memory_manager.get_session(request.session_id) if memory_manager else None
    if not session_data:
        raise SessionNotFoundError(f"Session {request.session_id} not found")

    # Validate action range
    if request.action < 0 or request.action > 255:
        raise InvalidActionError(f"Invalid action value {request.action}; expected 0-255 integer")

    try:
        # Use predictive inference for instant response
        frame, metrics = await predictive_engine.step_session_predictive(
            request.session_id,
            request.action
        )

        # Update session metadata
        session_data["frames_generated"] += 1
        session_data["current_fps"] = metrics["fps"]
        session_data["actions"].append({
            "t": datetime.now().isoformat(),
            "action": request.action
        })

        # Calculate updated priority
        recent_activity = True  # Just performed an action
        updated_priority = calculate_session_priority(
            session_data.get("prompt"),
            session_data["frames_generated"],
            recent_activity
        )
        session_data["priority"] = updated_priority

        # Update in memory manager
        if memory_manager:
            memory_manager.update_session(
                request.session_id,
                session_data,
                priority=updated_priority
            )
            # Schedule memory usage update
            background_tasks.add_task(update_session_memory_usage, request.session_id)

        frame_base64 = encode_frame_current(frame)

        return {
            "frame": frame_base64,
            "metrics": metrics,
            "frame_number": session_data["frames_generated"],
            "predicted": metrics.get("predicted", False)
        }

    except KeyError:
        raise HTTPException(status_code=404, detail="Session not initialized")
    except Exception as e:
        logger.exception("Error stepping session: %s", e)
        raise HTTPException(status_code=500, detail="Engine step failed")


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete session and cleanup resources"""
    session_data = memory_manager.get_session(session_id) if memory_manager else None
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")

    # Stop predictive session
    if predictive_engine:
        predictive_engine.stop_session_predictive(session_id)

    # Remove from memory manager
    if memory_manager:
        memory_manager.remove_session(session_id)

    return {"status": "deleted", "session_id": session_id}


@app.get("/session/{session_id}/info")
async def get_session_info(session_id: str):
    """Get detailed session information"""
    session_data = memory_manager.get_session(session_id) if memory_manager else None
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")

    # Get prediction stats
    prediction_stats = {}
    if predictive_engine:
        prediction_stats = predictive_engine.get_prediction_stats(session_id)

    # Estimate memory usage
    memory_usage = await estimate_session_memory_usage(session_id)

    return SessionInfo(
        session_id=session_id,
        created_at=session_data["created_at"],
        frames_generated=session_data["frames_generated"],
        current_fps=session_data["current_fps"],
        memory_usage_mb=memory_usage,
        priority=session_data.get("priority", 1.0),
        prediction_stats=prediction_stats
    )


@app.get("/sessions")
async def list_sessions():
    """List all active sessions with stats"""
    if not memory_manager:
        return {"total": 0, "sessions": []}

    session_list = []
    for session_id in memory_manager.cache.sessions.keys():
        try:
            session_data = memory_manager.get_session(session_id)
            if session_data:
                session_list.append({
                    "session_id": session_id,
                    "frames_generated": session_data.get("frames_generated", 0),
                    "priority": session_data.get("priority", 1.0),
                    "created_at": session_data.get("created_at", "unknown")
                })
        except Exception:
            continue  # Skip problematic sessions

    return {
        "total": len(session_list),
        "sessions": session_list
    }


@app.get("/health")
async def health_check():
    """Comprehensive health check with memory and performance metrics"""
    memory_report = memory_manager.get_memory_report() if memory_manager else {}

    gpu_available = torch.cuda.is_available()
    gpu_memory = {}
    if gpu_available:
        gpu_memory = {
            "allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "reserved_gb": torch.cuda.memory_reserved() / 1e9,
            "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9
        }

    return {
        "status": "healthy",
        "engine_loaded": predictive_engine is not None,
        "memory_manager_active": memory_manager is not None and memory_manager.is_monitoring,
        "memory_report": memory_report,
        "gpu_available": gpu_available,
        "gpu_memory": gpu_memory,
        "features": {
            "predictive_inference": True,
            "memory_management": True,
            "auto_cleanup": True,
            "session_recovery": False  # TODO: Implement
        }
    }


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time session interaction"""
    await websocket.accept()

    # Get or create session in memory manager
    session_data = memory_manager.get_session(session_id) if memory_manager else None
    if not session_data:
        # Create new session
        if memory_manager:
            session_data = {
                "created_at": datetime.now().isoformat(),
                "frames_generated": 0,
                "current_fps": 0,
                "prompt": None,
                "seed": None,
                "actions": []
            }
            if not memory_manager.create_session(session_id, session_data):
                await websocket.close(code=1013, reason="Server overloaded")
                return

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message["type"] == "action":
                action = message["action"]

                # Ensure session is started
                if not predictive_engine.is_session_running(session_id):
                    session_meta = memory_manager.get_session(session_id) if memory_manager else {}
                    predictive_engine.start_session(
                        session_id,
                        prompt=session_meta.get("prompt"),
                        seed=session_meta.get("seed"),
                    )

                # Step the session
                frame, metrics = await predictive_engine.step_session_predictive(session_id, action)

                # Update session data
                if memory_manager and session_data:
                    session_data["frames_generated"] += 1
                    session_data["actions"].append(action)
                    memory_usage = await estimate_session_memory_usage(session_id)
                    priority = calculate_session_priority(
                        prompt=session_data.get("prompt"),
                        frames_generated=session_data["frames_generated"],
                        recent_activity=True
                    )
                    memory_manager.update_session(session_id, session_data, memory_usage, priority)

                # Encode frame to base64 for frontend display
                frame_base64 = encode_frame_current(frame)

                # Send response with frame in 'data' field as expected by frontend
                response = {
                    "type": "frame",
                    "data": frame_base64,  # Frontend expects 'data' field, not 'frame'
                    "metrics": metrics,
                    "session_id": session_id,
                    "frame_number": session_data["frames_generated"] if session_data else 1
                }
                await websocket.send_text(json.dumps(response))

    except WebSocketDisconnect:
        print(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        print(f"Error in WebSocket for session {session_id}: {e}")
        await websocket.close()


@app.post("/settings")
async def update_settings(settings: Dict[str, Any]):
    """Update application settings"""
    # For now, just acknowledge the settings
    # In the future, this could update actual rendering settings
    return {
        "status": "success",
        "message": "Settings updated",
        "settings": settings
    }


@app.get("/settings")
async def get_settings():
    """Get current application settings"""
    return {
        "resolution": "256x256",
        "fps_target": 30,
        "show_overlay": True
    }


@app.get("/admin/memory")
async def get_memory_status():
    """Admin endpoint for detailed memory status"""
    if not memory_manager:
        raise HTTPException(status_code=503, detail="Memory manager not available")

    return memory_manager.get_memory_report()


@app.post("/admin/cleanup")
async def force_cleanup(target_sessions: int = 10):
    """Admin endpoint to force session cleanup"""
    if not memory_manager:
        raise HTTPException(status_code=503, detail="Memory manager not available")

    removed_count = memory_manager.force_cleanup(target_sessions)

    return {
        "status": "completed",
        "sessions_removed": removed_count,
        "remaining_sessions": len(memory_manager.cache.sessions)
    }


# Progressive Model Loading Endpoints
@app.post("/models/load-progressive", response_model=LoadModelResponse)
async def start_progressive_loading(request: LoadModelRequest):
    """Start progressive model loading with real-time updates"""
    if not progressive_loader:
        raise HTTPException(status_code=503, detail="Progressive loader not available")

    try:
        job_id = await progressive_loader.load_model_progressive(request.model_path)
        websocket_url = f"/ws/loading/{job_id}"

        return LoadModelResponse(
            job_id=job_id,
            status="started",
            websocket_url=websocket_url
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Failed to start progressive loading")
        raise HTTPException(status_code=500, detail=f"Failed to start loading: {str(e)}")


@app.get("/models/loading/{job_id}", response_model=JobStatusResponse)
async def get_loading_status(job_id: str):
    """Get current loading status for a job"""
    if not progressive_loader:
        raise HTTPException(status_code=503, detail="Progressive loader not available")

    status = progressive_loader.get_job_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Loading job not found")

    return JobStatusResponse(**status)


@app.get("/models/loading")
async def list_loading_jobs():
    """List all loading jobs"""
    if not progressive_loader:
        raise HTTPException(status_code=503, detail="Progressive loader not available")

    return progressive_loader.list_jobs()


@app.delete("/models/loading/{job_id}")
async def cancel_loading(job_id: str):
    """Cancel a loading job"""
    if not progressive_loader:
        raise HTTPException(status_code=503, detail="Progressive loader not available")

    success = progressive_loader.cancel_loading(job_id)
    if not success:
        raise HTTPException(status_code=404, detail="Loading job not found or already completed")

    return {"status": "cancelled", "job_id": job_id}


@app.websocket("/ws/loading/{job_id}")
async def loading_websocket(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for real-time loading updates"""
    if not progressive_loader:
        await websocket.close(code=1011, reason="Progressive loader not available")
        return

    await progressive_loader.websocket_handler(websocket, job_id)


@app.get("/models/current")
async def get_current_model_info():
    """Get information about currently loaded model"""
    if not progressive_loader:
        raise HTTPException(status_code=503, detail="Progressive loader not available")

    current_model = progressive_loader.get_current_model()
    if not current_model:
        return {"status": "no_model_loaded", "model": None}

    model_info = current_model.get_model_info()
    performance_report = current_model.get_performance_report()

    return {
        "status": "model_loaded",
        "model": model_info,
        "performance": performance_report
    }


# Utility functions (same as original with minor enhancements)
def frame_to_base64(frame: np.ndarray) -> str:
    """Convert frame to base64 string"""
    img = Image.fromarray(frame.astype('uint8'))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode()


def encode_frame_current(frame: np.ndarray) -> str:
    """Apply current settings and encode frame"""
    try:
        # Use fixed 256x256 for now - TODO: Make configurable
        target = 256
        if frame.shape[0] != target or frame.shape[1] != target:
            img = Image.fromarray(frame.astype('uint8')).resize((target, target), Image.BILINEAR)
            return frame_to_base64(np.array(img))
        return frame_to_base64(frame)
    except Exception as e:
        logger.warning(f"Frame encoding error: {e}")
        # Return a placeholder frame on error
        placeholder = np.zeros((256, 256, 3), dtype=np.uint8)
        return frame_to_base64(placeholder)


# Model Management Endpoints (UI compatibility)
@app.get("/models")
async def list_models():
    """List available models for UI compatibility"""
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

    # Get current model info
    current_model = None
    if predictive_engine and hasattr(predictive_engine, 'base_engine'):
        current_model = getattr(predictive_engine.base_engine, 'model_path', None)

    return {
        "models": models,
        "current_model": current_model
    }


class LoadSimpleModelRequest(BaseModel):
    id: str  # filename under checkpoints dir


@app.post("/models/load")
async def load_model(req: LoadSimpleModelRequest):
    """Load model for UI compatibility"""
    target = (cfg.CHECKPOINTS_DIR / req.id).resolve()
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"Model not found: {target}")

    try:
        # Use progressive loader if available, otherwise fallback
        if progressive_loader:
            job_id = await progressive_loader.load_model_progressive(str(target))
            # Wait for completion (simplified for UI)
            import time
            for _ in range(30):  # 30 second timeout
                status = progressive_loader.get_job_status(job_id)
                if status and status.get("status") == "completed":
                    break
                await asyncio.sleep(1)

        current_model = str(target)
        return {"status": "ok", "current_model": current_model}
    except Exception as e:
        logger.exception("Failed to load model: %s", e)
        raise HTTPException(status_code=500, detail="Failed to load model")


@app.get("/models/status")
async def model_status():
    """Get current model status for UI compatibility"""
    current_model = None
    if predictive_engine and hasattr(predictive_engine, 'base_engine'):
        current_model = getattr(predictive_engine.base_engine, 'model_path', None)

    return {
        "current_model": current_model,
        "ready": predictive_engine is not None
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app=app,
        host=cfg.HOST,
        port=cfg.PORT,
        reload=False,  # Disable reload for better performance
        log_level="info",
        access_log=True
    )
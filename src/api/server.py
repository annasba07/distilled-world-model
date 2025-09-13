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
from typing import Optional
import uuid
from datetime import datetime
import torch

from ..inference.engine import BatchedInferenceEngine
from ..utils.logging import get_logger
from .. import config as cfg


app = FastAPI(title="Lightweight World Model API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = None
sessions = {}
logger = get_logger(__name__)


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


@app.on_event("startup")
async def startup_event():
    global engine
    engine = BatchedInferenceEngine(
        model_path=cfg.DEFAULT_WORLD_MODEL_CKPT,
        device=("cuda" if torch.cuda.is_available() else "cpu") if cfg.DEVICE == "cuda" else cfg.DEVICE,
        use_tensorrt=cfg.USE_TENSORRT and torch.cuda.is_available(),
        use_fp16=cfg.USE_FP16 and torch.cuda.is_available(),
        batch_size=cfg.BATCH_SIZE,
    )
    logger.info("World Model Engine loaded successfully")
    # Mount static demo at /demo (best-effort)
    try:
        app.mount("/demo", StaticFiles(directory=str((cfg.PROJECT_ROOT / "demo").resolve()), html=True), name="demo")
        logger.info("Mounted static demo at /demo")
    except Exception as e:
        logger.warning("Could not mount demo static files: %s", e)


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
    
    engine.create_session(session_id)
    
    initial_frame = engine.generate_interactive(request.prompt)
    
    sessions[session_id] = {
        "created_at": datetime.now().isoformat(),
        "frames_generated": 1,
        "current_fps": 0,
        "prompt": request.prompt
    }
    
    frame_base64 = frame_to_base64(initial_frame)
    
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
        frame, metrics = engine.step(request.action)
    except Exception as e:
        logger.exception("Error stepping session: %s", e)
        raise HTTPException(status_code=500, detail="Engine step failed")
    
    sessions[request.session_id]["frames_generated"] += 1
    sessions[request.session_id]["current_fps"] = metrics["fps"]
    
    frame_base64 = frame_to_base64(frame)
    
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
    engine.stop()
    
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
        engine.create_session(session_id)
        sessions[session_id] = {
            "created_at": datetime.now().isoformat(),
            "frames_generated": 0,
            "current_fps": 0,
            "prompt": None
        }
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "action":
                action = message["action"]
                
                frame, metrics = engine.step(action)
                
                sessions[session_id]["frames_generated"] += 1
                sessions[session_id]["current_fps"] = metrics["fps"]
                
                frame_base64 = frame_to_base64(frame)
                
                await websocket.send_json({
                    "type": "frame",
                    "data": frame_base64,
                    "metrics": metrics,
                    "frame_number": sessions[session_id]["frames_generated"]
                })
                
            elif message["type"] == "reset":
                prompt = message.get("prompt", None)
                initial_frame = engine.generate_interactive(prompt)
                
                sessions[session_id]["frames_generated"] = 1
                sessions[session_id]["prompt"] = prompt
                
                frame_base64 = frame_to_base64(initial_frame)
                
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


@app.post("/batch/process")
async def batch_process(requests: list[ActionRequest]):
    batch_requests = [
        {"session_id": req.session_id, "action": req.action}
        for req in requests
    ]
    
    results = engine.process_batch(batch_requests)
    
    response = []
    for result in results:
        session_id = result["session_id"]
        frame = result["frame"]
        
        sessions[session_id]["frames_generated"] += 1
        
        response.append({
            "session_id": session_id,
            "frame": frame_to_base64(frame),
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app=app,
        host=cfg.HOST,
        port=cfg.PORT,
        reload=True,
        log_level="info",
    )

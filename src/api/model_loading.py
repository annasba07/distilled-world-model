"""
Progressive Model Loading System with Real-time Status Updates

This module provides progressive model loading capabilities with WebSocket-based
status updates, ensuring users get real-time feedback during model switching.
"""

import asyncio
import json
import time
from typing import Dict, Any, Optional, Set
from pathlib import Path
import uuid
from dataclasses import dataclass
from enum import Enum

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from ..models.enhanced_world_model import EnhancedWorldModel, ProgressCallback, load_model_with_progress
from ..utils.logging import get_logger


logger = get_logger(__name__)


class LoadingStatus(Enum):
    """Loading status enumeration"""
    IDLE = "idle"
    STARTING = "starting"
    LOADING = "loading"
    OPTIMIZING = "optimizing"
    COMPLETE = "complete"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class LoadingJob:
    """Represents a model loading job"""
    job_id: str
    model_path: str
    status: LoadingStatus
    progress: float
    stage: str
    start_time: float
    error_message: Optional[str] = None
    estimated_completion: Optional[float] = None


class LoadingJobManager:
    """Manages model loading jobs and status tracking"""

    def __init__(self):
        self.active_jobs: Dict[str, LoadingJob] = {}
        self.completed_jobs: Dict[str, LoadingJob] = {}
        self.max_completed_jobs = 50  # Keep last 50 completed jobs
        self.subscribers: Dict[str, Set[WebSocket]] = {}

    def create_job(self, model_path: str) -> str:
        """Create a new loading job"""
        job_id = str(uuid.uuid4())

        job = LoadingJob(
            job_id=job_id,
            model_path=model_path,
            status=LoadingStatus.IDLE,
            progress=0.0,
            stage="initialized",
            start_time=time.time()
        )

        self.active_jobs[job_id] = job
        return job_id

    def update_job(self, job_id: str, status: Optional[LoadingStatus] = None,
                  progress: Optional[float] = None, stage: Optional[str] = None,
                  error_message: Optional[str] = None):
        """Update job status and notify subscribers"""
        if job_id not in self.active_jobs:
            return

        job = self.active_jobs[job_id]

        if status is not None:
            job.status = status
        if progress is not None:
            job.progress = min(max(progress, 0.0), 1.0)
        if stage is not None:
            job.stage = stage
        if error_message is not None:
            job.error_message = error_message

        # Estimate completion time
        if job.progress > 0 and job.status == LoadingStatus.LOADING:
            elapsed = time.time() - job.start_time
            total_estimated = elapsed / job.progress
            job.estimated_completion = job.start_time + total_estimated

        # Notify subscribers
        asyncio.create_task(self._notify_subscribers(job_id))

        # Move to completed if finished
        if status in [LoadingStatus.COMPLETE, LoadingStatus.ERROR, LoadingStatus.CANCELLED]:
            self._complete_job(job_id)

    def _complete_job(self, job_id: str):
        """Move job from active to completed"""
        if job_id in self.active_jobs:
            job = self.active_jobs.pop(job_id)
            self.completed_jobs[job_id] = job

            # Maintain completed jobs limit
            if len(self.completed_jobs) > self.max_completed_jobs:
                oldest_job_id = min(self.completed_jobs.keys(),
                                  key=lambda x: self.completed_jobs[x].start_time)
                del self.completed_jobs[oldest_job_id]

    async def _notify_subscribers(self, job_id: str):
        """Notify all subscribers about job update"""
        job = self.active_jobs.get(job_id) or self.completed_jobs.get(job_id)
        if not job:
            return

        message = {
            "type": "loading_update",
            "job_id": job_id,
            "status": job.status.value,
            "progress": job.progress,
            "stage": job.stage,
            "error_message": job.error_message,
            "estimated_completion": job.estimated_completion
        }

        # Notify job-specific subscribers
        if job_id in self.subscribers:
            disconnected_clients = []
            for websocket in self.subscribers[job_id].copy():
                try:
                    await websocket.send_json(message)
                except Exception:
                    disconnected_clients.append(websocket)

            # Clean up disconnected clients
            for websocket in disconnected_clients:
                self.subscribers[job_id].discard(websocket)

    def subscribe(self, job_id: str, websocket: WebSocket):
        """Subscribe to job updates"""
        if job_id not in self.subscribers:
            self.subscribers[job_id] = set()
        self.subscribers[job_id].add(websocket)

    def unsubscribe(self, job_id: str, websocket: WebSocket):
        """Unsubscribe from job updates"""
        if job_id in self.subscribers:
            self.subscribers[job_id].discard(websocket)
            if not self.subscribers[job_id]:
                del self.subscribers[job_id]

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get current job status"""
        job = self.active_jobs.get(job_id) or self.completed_jobs.get(job_id)
        if not job:
            return None

        return {
            "job_id": job.job_id,
            "model_path": job.model_path,
            "status": job.status.value,
            "progress": job.progress,
            "stage": job.stage,
            "start_time": job.start_time,
            "error_message": job.error_message,
            "estimated_completion": job.estimated_completion
        }

    def list_jobs(self) -> Dict[str, Any]:
        """List all jobs"""
        return {
            "active": [self.get_job_status(job_id) for job_id in self.active_jobs.keys()],
            "completed": [self.get_job_status(job_id) for job_id in list(self.completed_jobs.keys())[-10:]]  # Last 10
        }

    def cancel_job(self, job_id: str) -> bool:
        """Cancel an active job"""
        if job_id in self.active_jobs:
            self.update_job(job_id, status=LoadingStatus.CANCELLED, stage="cancelled by user")
            return True
        return False


class ProgressiveModelLoader:
    """Progressive model loader with real-time updates"""

    def __init__(self, device='cuda'):
        self.device = device
        self.job_manager = LoadingJobManager()
        self.current_model: Optional[EnhancedWorldModel] = None
        self.loading_lock = asyncio.Lock()

    async def load_model_progressive(self, model_path: str) -> str:
        """Start progressive model loading and return job ID"""
        # Validate model path
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Create loading job
        job_id = self.job_manager.create_job(model_path)

        # Start loading in background
        asyncio.create_task(self._load_model_background(job_id, model_path))

        return job_id

    async def _load_model_background(self, job_id: str, model_path: str):
        """Background task for model loading"""
        async with self.loading_lock:
            try:
                # Update status to starting
                self.job_manager.update_job(
                    job_id,
                    status=LoadingStatus.STARTING,
                    stage="Preparing model loading"
                )

                await asyncio.sleep(0.1)  # Brief pause for UI update

                # Create progress callback
                def progress_callback(stage: str, progress: float):
                    """Callback for progress updates"""
                    self.job_manager.update_job(
                        job_id,
                        status=LoadingStatus.LOADING,
                        progress=progress,
                        stage=stage
                    )

                progress_cb = ProgressCallback(progress_callback)

                # Load model with progress
                model, success = await load_model_with_progress(
                    model_path,
                    device=self.device,
                    progress_callback=progress_cb
                )

                if success:
                    # Optimization phase
                    self.job_manager.update_job(
                        job_id,
                        status=LoadingStatus.OPTIMIZING,
                        progress=0.95,
                        stage="Optimizing model for inference"
                    )

                    await asyncio.sleep(0.1)  # Brief pause

                    # Enable optimizations
                    model.enable_fast_mode()

                    # Update current model
                    self.current_model = model

                    # Complete
                    self.job_manager.update_job(
                        job_id,
                        status=LoadingStatus.COMPLETE,
                        progress=1.0,
                        stage="Model loaded successfully"
                    )

                    logger.info(f"Model loaded successfully: {model_path}")

                else:
                    self.job_manager.update_job(
                        job_id,
                        status=LoadingStatus.ERROR,
                        stage="Model loading failed",
                        error_message="Unknown loading error"
                    )

            except Exception as e:
                error_msg = str(e)
                logger.error(f"Model loading error for {model_path}: {error_msg}")

                self.job_manager.update_job(
                    job_id,
                    status=LoadingStatus.ERROR,
                    stage="Error occurred during loading",
                    error_message=error_msg
                )

    def get_current_model(self) -> Optional[EnhancedWorldModel]:
        """Get currently loaded model"""
        return self.current_model

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status"""
        return self.job_manager.get_job_status(job_id)

    def list_jobs(self) -> Dict[str, Any]:
        """List all loading jobs"""
        return self.job_manager.list_jobs()

    def cancel_loading(self, job_id: str) -> bool:
        """Cancel a loading job"""
        return self.job_manager.cancel_job(job_id)

    async def websocket_handler(self, websocket: WebSocket, job_id: str):
        """WebSocket handler for real-time loading updates"""
        await websocket.accept()

        # Subscribe to job updates
        self.job_manager.subscribe(job_id, websocket)

        try:
            # Send initial status
            initial_status = self.get_job_status(job_id)
            if initial_status:
                await websocket.send_json({
                    "type": "initial_status",
                    **initial_status
                })

            # Keep connection alive and handle client messages
            while True:
                try:
                    data = await websocket.receive_text()
                    message = json.loads(data)

                    if message.get("type") == "cancel":
                        self.cancel_loading(job_id)
                        await websocket.send_json({
                            "type": "cancelled",
                            "job_id": job_id
                        })

                    elif message.get("type") == "ping":
                        await websocket.send_json({
                            "type": "pong",
                            "timestamp": time.time()
                        })

                except WebSocketDisconnect:
                    break
                except Exception as e:
                    logger.warning(f"WebSocket error: {e}")
                    break

        except Exception as e:
            logger.error(f"WebSocket handler error: {e}")
        finally:
            # Unsubscribe on disconnect
            self.job_manager.unsubscribe(job_id, websocket)


# Request/Response models
class LoadModelRequest(BaseModel):
    model_path: str
    device: Optional[str] = "cuda"


class LoadModelResponse(BaseModel):
    job_id: str
    status: str
    websocket_url: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: float
    stage: str
    error_message: Optional[str] = None
    estimated_completion: Optional[float] = None


# Global instance
progressive_loader: Optional[ProgressiveModelLoader] = None


def get_progressive_loader() -> ProgressiveModelLoader:
    """Get or create progressive loader instance"""
    global progressive_loader
    if progressive_loader is None:
        progressive_loader = ProgressiveModelLoader()
    return progressive_loader
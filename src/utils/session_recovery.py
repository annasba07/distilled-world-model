"""
Session State Recovery System for Crash-Resilient User Experience

This module implements automatic session state persistence and recovery
to ensure users never lose progress due to crashes or network interruptions.
"""

import asyncio
import json
import pickle
import time
import threading
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional, List, Set
import hashlib
import uuid
from datetime import datetime, timedelta
import numpy as np
import torch

from ..utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class SessionCheckpoint:
    """Represents a session state checkpoint"""
    session_id: str
    checkpoint_id: str
    timestamp: datetime
    frame_count: int

    # Core session data
    prompt: Optional[str]
    seed: Optional[int]
    actions: List[Dict[str, Any]]

    # Model state
    latent_state: Optional[bytes]  # Serialized tensor
    last_frame: Optional[bytes]    # Serialized frame

    # Metadata
    model_hash: str
    settings: Dict[str, Any]
    performance_stats: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionCheckpoint':
        """Create from dictionary"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class StateSerializer:
    """Handles serialization of complex state objects"""

    @staticmethod
    def serialize_tensor(tensor: Optional[torch.Tensor]) -> Optional[bytes]:
        """Serialize PyTorch tensor to bytes"""
        if tensor is None:
            return None

        try:
            # Move to CPU and serialize
            cpu_tensor = tensor.cpu() if tensor.is_cuda else tensor
            return pickle.dumps(cpu_tensor)
        except Exception as e:
            logger.error(f"Failed to serialize tensor: {e}")
            return None

    @staticmethod
    def deserialize_tensor(data: Optional[bytes], device: str = 'cuda') -> Optional[torch.Tensor]:
        """Deserialize bytes to PyTorch tensor"""
        if data is None:
            return None

        try:
            tensor = pickle.loads(data)
            if isinstance(tensor, torch.Tensor):
                return tensor.to(device) if torch.cuda.is_available() and device == 'cuda' else tensor
        except Exception as e:
            logger.error(f"Failed to deserialize tensor: {e}")

        return None

    @staticmethod
    def serialize_frame(frame: Optional[np.ndarray]) -> Optional[bytes]:
        """Serialize numpy frame to bytes"""
        if frame is None:
            return None

        try:
            return pickle.dumps(frame)
        except Exception as e:
            logger.error(f"Failed to serialize frame: {e}")
            return None

    @staticmethod
    def deserialize_frame(data: Optional[bytes]) -> Optional[np.ndarray]:
        """Deserialize bytes to numpy frame"""
        if data is None:
            return None

        try:
            frame = pickle.loads(data)
            if isinstance(frame, np.ndarray):
                return frame
        except Exception as e:
            logger.error(f"Failed to deserialize frame: {e}")

        return None

    @staticmethod
    def compute_model_hash(model) -> str:
        """Compute hash of model state for versioning"""
        try:
            # Create a hash based on model parameters
            hasher = hashlib.sha256()

            if hasattr(model, 'state_dict'):
                for param in model.state_dict().values():
                    if isinstance(param, torch.Tensor):
                        hasher.update(param.cpu().numpy().tobytes())

            return hasher.hexdigest()[:16]  # First 16 characters
        except Exception:
            return "unknown"


class SessionStorage:
    """Handles persistent storage of session checkpoints"""

    def __init__(self, storage_dir: str = "session_checkpoints"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.serializer = StateSerializer()

    def save_checkpoint(self, checkpoint: SessionCheckpoint) -> bool:
        """Save checkpoint to persistent storage"""
        try:
            checkpoint_file = self.storage_dir / f"{checkpoint.session_id}_{checkpoint.checkpoint_id}.json"

            # Save metadata as JSON
            metadata = checkpoint.to_dict()

            # Save binary data separately
            if checkpoint.latent_state:
                latent_file = self.storage_dir / f"{checkpoint.session_id}_{checkpoint.checkpoint_id}_latent.pkl"
                with open(latent_file, 'wb') as f:
                    f.write(checkpoint.latent_state)
                metadata['latent_file'] = str(latent_file)

            if checkpoint.last_frame:
                frame_file = self.storage_dir / f"{checkpoint.session_id}_{checkpoint.checkpoint_id}_frame.pkl"
                with open(frame_file, 'wb') as f:
                    f.write(checkpoint.last_frame)
                metadata['frame_file'] = str(frame_file)

            # Remove binary data from JSON
            metadata.pop('latent_state', None)
            metadata.pop('last_frame', None)

            # Save metadata
            with open(checkpoint_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.debug(f"Saved checkpoint {checkpoint.checkpoint_id} for session {checkpoint.session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return False

    def load_checkpoint(self, session_id: str, checkpoint_id: Optional[str] = None) -> Optional[SessionCheckpoint]:
        """Load checkpoint from persistent storage"""
        try:
            # Find checkpoint file
            if checkpoint_id:
                checkpoint_file = self.storage_dir / f"{session_id}_{checkpoint_id}.json"
            else:
                # Find latest checkpoint for session
                pattern = f"{session_id}_*.json"
                files = list(self.storage_dir.glob(pattern))
                if not files:
                    return None
                checkpoint_file = max(files, key=lambda f: f.stat().st_mtime)

            if not checkpoint_file.exists():
                return None

            # Load metadata
            with open(checkpoint_file, 'r') as f:
                metadata = json.load(f)

            # Load binary data
            latent_state = None
            if 'latent_file' in metadata:
                latent_file = Path(metadata['latent_file'])
                if latent_file.exists():
                    with open(latent_file, 'rb') as f:
                        latent_state = f.read()

            last_frame = None
            if 'frame_file' in metadata:
                frame_file = Path(metadata['frame_file'])
                if frame_file.exists():
                    with open(frame_file, 'rb') as f:
                        last_frame = f.read()

            # Reconstruct checkpoint
            metadata['latent_state'] = latent_state
            metadata['last_frame'] = last_frame
            metadata.pop('latent_file', None)
            metadata.pop('frame_file', None)

            checkpoint = SessionCheckpoint.from_dict(metadata)
            logger.debug(f"Loaded checkpoint {checkpoint.checkpoint_id} for session {session_id}")
            return checkpoint

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

    def list_checkpoints(self, session_id: str) -> List[str]:
        """List all checkpoint IDs for a session"""
        try:
            pattern = f"{session_id}_*.json"
            files = list(self.storage_dir.glob(pattern))

            checkpoint_ids = []
            for file in files:
                # Extract checkpoint ID from filename
                parts = file.stem.split('_')
                if len(parts) >= 2:
                    checkpoint_ids.append('_'.join(parts[1:]))

            return sorted(checkpoint_ids)
        except Exception as e:
            logger.error(f"Failed to list checkpoints: {e}")
            return []

    def cleanup_old_checkpoints(self, max_age_days: int = 7, max_per_session: int = 10):
        """Clean up old checkpoints"""
        try:
            cutoff_time = datetime.now() - timedelta(days=max_age_days)

            # Group files by session
            session_files: Dict[str, List[Path]] = {}

            for file in self.storage_dir.glob("*.json"):
                session_id = file.stem.split('_')[0]
                if session_id not in session_files:
                    session_files[session_id] = []
                session_files[session_id].append(file)

            # Clean up each session
            for session_id, files in session_files.items():
                # Sort by modification time
                files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

                files_to_remove = []

                # Remove old files
                for file in files:
                    file_time = datetime.fromtimestamp(file.stat().st_mtime)
                    if file_time < cutoff_time:
                        files_to_remove.append(file)

                # Remove excess files (keep max_per_session most recent)
                if len(files) > max_per_session:
                    files_to_remove.extend(files[max_per_session:])

                # Actually remove files
                for file in files_to_remove:
                    try:
                        # Remove associated binary files
                        checkpoint_id = '_'.join(file.stem.split('_')[1:])
                        latent_file = self.storage_dir / f"{session_id}_{checkpoint_id}_latent.pkl"
                        frame_file = self.storage_dir / f"{session_id}_{checkpoint_id}_frame.pkl"

                        for f in [file, latent_file, frame_file]:
                            if f.exists():
                                f.unlink()

                        logger.debug(f"Removed old checkpoint: {file.name}")
                    except Exception as e:
                        logger.warning(f"Failed to remove checkpoint file {file}: {e}")

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")


class SessionRecoveryManager:
    """Manages session state recovery and checkpointing"""

    def __init__(self, storage_dir: str = "session_checkpoints",
                 checkpoint_interval: int = 10,  # Save every 10 frames
                 auto_cleanup: bool = True):

        self.storage = SessionStorage(storage_dir)
        self.checkpoint_interval = checkpoint_interval
        self.auto_cleanup = auto_cleanup

        # Active sessions and their checkpoint counters
        self.active_sessions: Dict[str, int] = {}  # session_id -> frame_count
        self.recovery_enabled_sessions: Set[str] = set()

        # Background tasks
        self.cleanup_task: Optional[asyncio.Task] = None
        self.is_monitoring = False

    async def start_monitoring(self):
        """Start background monitoring and cleanup"""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        if self.auto_cleanup:
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Session recovery monitoring started")

    async def stop_monitoring(self):
        """Stop background monitoring"""
        self.is_monitoring = False
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("Session recovery monitoring stopped")

    def enable_recovery(self, session_id: str):
        """Enable recovery for a session"""
        self.recovery_enabled_sessions.add(session_id)
        self.active_sessions[session_id] = 0

    def disable_recovery(self, session_id: str):
        """Disable recovery for a session"""
        self.recovery_enabled_sessions.discard(session_id)
        self.active_sessions.pop(session_id, None)

    def should_checkpoint(self, session_id: str) -> bool:
        """Check if session should be checkpointed"""
        if session_id not in self.recovery_enabled_sessions:
            return False

        frame_count = self.active_sessions.get(session_id, 0)
        return frame_count % self.checkpoint_interval == 0

    async def create_checkpoint(self, session_id: str, session_data: Dict[str, Any],
                              model_state: Optional[Any] = None) -> bool:
        """Create a checkpoint for a session"""
        try:
            if session_id not in self.recovery_enabled_sessions:
                return False

            # Increment frame count
            self.active_sessions[session_id] = self.active_sessions.get(session_id, 0) + 1
            frame_count = self.active_sessions[session_id]

            # Create checkpoint ID
            checkpoint_id = f"frame_{frame_count}_{int(time.time())}"

            # Serialize model state
            latent_state = None
            last_frame = None
            model_hash = "unknown"

            if model_state:
                if hasattr(model_state, 'latent'):
                    latent_state = self.storage.serializer.serialize_tensor(model_state.latent)
                if hasattr(model_state, 'last_frame'):
                    last_frame = self.storage.serializer.serialize_frame(model_state.last_frame)
                if hasattr(model_state, 'model'):
                    model_hash = self.storage.serializer.compute_model_hash(model_state.model)

            # Create checkpoint
            checkpoint = SessionCheckpoint(
                session_id=session_id,
                checkpoint_id=checkpoint_id,
                timestamp=datetime.now(),
                frame_count=frame_count,
                prompt=session_data.get('prompt'),
                seed=session_data.get('seed'),
                actions=session_data.get('actions', []),
                latent_state=latent_state,
                last_frame=last_frame,
                model_hash=model_hash,
                settings=session_data.get('settings', {}),
                performance_stats=session_data.get('performance_stats', {})
            )

            # Save checkpoint
            success = self.storage.save_checkpoint(checkpoint)
            if success:
                logger.debug(f"Created checkpoint for session {session_id} at frame {frame_count}")

            return success

        except Exception as e:
            logger.error(f"Failed to create checkpoint: {e}")
            return False

    async def recover_session(self, session_id: str, checkpoint_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Recover a session from checkpoint"""
        try:
            checkpoint = self.storage.load_checkpoint(session_id, checkpoint_id)
            if not checkpoint:
                logger.warning(f"No checkpoint found for session {session_id}")
                return None

            # Deserialize state
            recovered_state = {
                'session_id': checkpoint.session_id,
                'prompt': checkpoint.prompt,
                'seed': checkpoint.seed,
                'actions': checkpoint.actions,
                'frame_count': checkpoint.frame_count,
                'settings': checkpoint.settings,
                'performance_stats': checkpoint.performance_stats,
                'checkpoint_info': {
                    'checkpoint_id': checkpoint.checkpoint_id,
                    'timestamp': checkpoint.timestamp.isoformat(),
                    'model_hash': checkpoint.model_hash
                }
            }

            # Deserialize model state if available
            if checkpoint.latent_state:
                recovered_state['latent_tensor'] = self.storage.serializer.deserialize_tensor(
                    checkpoint.latent_state
                )

            if checkpoint.last_frame:
                recovered_state['last_frame'] = self.storage.serializer.deserialize_frame(
                    checkpoint.last_frame
                )

            # Re-enable recovery for this session
            self.enable_recovery(session_id)
            self.active_sessions[session_id] = checkpoint.frame_count

            logger.info(f"Recovered session {session_id} from checkpoint {checkpoint.checkpoint_id}")
            return recovered_state

        except Exception as e:
            logger.error(f"Failed to recover session: {e}")
            return None

    def list_recoverable_sessions(self) -> Dict[str, List[Dict[str, Any]]]:
        """List all sessions that can be recovered"""
        try:
            sessions = {}

            # Find all checkpoint files
            for file in self.storage.storage_dir.glob("*.json"):
                session_id = file.stem.split('_')[0]

                if session_id not in sessions:
                    sessions[session_id] = []

                try:
                    with open(file, 'r') as f:
                        metadata = json.load(f)

                    sessions[session_id].append({
                        'checkpoint_id': metadata.get('checkpoint_id'),
                        'timestamp': metadata.get('timestamp'),
                        'frame_count': metadata.get('frame_count', 0),
                        'prompt': metadata.get('prompt'),
                        'actions_count': len(metadata.get('actions', []))
                    })
                except Exception as e:
                    logger.warning(f"Failed to read checkpoint metadata: {e}")

            # Sort checkpoints by timestamp
            for session_id in sessions:
                sessions[session_id].sort(
                    key=lambda x: x['timestamp'],
                    reverse=True
                )

            return sessions

        except Exception as e:
            logger.error(f"Failed to list recoverable sessions: {e}")
            return {}

    async def _cleanup_loop(self):
        """Background cleanup loop"""
        while self.is_monitoring:
            try:
                await asyncio.sleep(3600)  # Run every hour
                self.storage.cleanup_old_checkpoints()
                logger.debug("Completed checkpoint cleanup")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {e}")

    def get_recovery_stats(self) -> Dict[str, Any]:
        """Get recovery system statistics"""
        try:
            recoverable_sessions = self.list_recoverable_sessions()
            total_checkpoints = sum(len(checkpoints) for checkpoints in recoverable_sessions.values())

            return {
                'active_sessions': len(self.active_sessions),
                'recovery_enabled_sessions': len(self.recovery_enabled_sessions),
                'recoverable_sessions': len(recoverable_sessions),
                'total_checkpoints': total_checkpoints,
                'checkpoint_interval': self.checkpoint_interval,
                'storage_dir': str(self.storage.storage_dir),
                'monitoring_active': self.is_monitoring
            }
        except Exception as e:
            logger.error(f"Failed to get recovery stats: {e}")
            return {"error": str(e)}


# Global instance
global_recovery_manager = SessionRecoveryManager()


def get_recovery_manager() -> SessionRecoveryManager:
    """Get global recovery manager instance"""
    return global_recovery_manager


# Decorator for automatic checkpointing
def with_auto_checkpoint(recovery_manager: SessionRecoveryManager):
    """Decorator to add automatic checkpointing to session operations"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)

            # Extract session ID and create checkpoint if needed
            session_id = kwargs.get('session_id')
            if session_id and recovery_manager.should_checkpoint(session_id):
                # Try to extract session data and model state
                session_data = kwargs.get('session_data', {})
                model_state = kwargs.get('model_state')

                await recovery_manager.create_checkpoint(
                    session_id, session_data, model_state
                )

            return result
        return wrapper
    return decorator
from pathlib import Path
import os
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Paths
CHECKPOINTS_DIR = Path(os.getenv("LWM_CHECKPOINTS_DIR", PROJECT_ROOT / "checkpoints"))
DEFAULT_WORLD_MODEL_CKPT = os.getenv(
    "LWM_WORLD_MODEL_CKPT", str(CHECKPOINTS_DIR / "world_model_final.pt")
)

# Inference
DEVICE = os.getenv("LWM_DEVICE", "cuda")
USE_TENSORRT = os.getenv("LWM_TENSORRT", "true").lower() in {"1", "true", "yes"}
USE_FP16 = os.getenv("LWM_FP16", "true").lower() in {"1", "true", "yes"}
BATCH_SIZE = int(os.getenv("LWM_BATCH_SIZE", "4"))

# Memory Management
def _get_default_gpu_memory_limit():
    """Auto-detect GPU memory and set conservative limit"""
    if torch.cuda.is_available():
        total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        # Use 90% of available GPU memory as limit
        return total_memory_gb * 0.9
    return 4.0  # Default 4GB for CPU/unknown

MAX_GPU_MEMORY_GB = float(os.getenv("LWM_MAX_GPU_MEMORY_GB", str(_get_default_gpu_memory_limit())))
MAX_SESSIONS = int(os.getenv("LWM_MAX_SESSIONS", "100"))
SESSION_TTL_SECONDS = int(os.getenv("LWM_SESSION_TTL_SECONDS", "3600"))
MEMORY_CHECK_INTERVAL = int(os.getenv("LWM_MEMORY_CHECK_INTERVAL", "30"))

# API
HOST = os.getenv("LWM_HOST", "0.0.0.0")
PORT = int(os.getenv("LWM_PORT", "8000"))
RATE_LIMIT_REQUESTS = int(os.getenv("LWM_RATE_LIMIT_REQUESTS", "100"))
RATE_LIMIT_WINDOW = int(os.getenv("LWM_RATE_LIMIT_WINDOW", "60"))  # seconds


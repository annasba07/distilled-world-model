from pathlib import Path
import os


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

# API
HOST = os.getenv("LWM_HOST", "0.0.0.0")
PORT = int(os.getenv("LWM_PORT", "8000"))


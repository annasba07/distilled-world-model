# Lightweight Interactive World Model for Consumer GPUs

A lightweight world model that generates interactive 2D game environments in real time on consumer hardware.

## Features

- **Real-time generation** on consumer GPUs (28 FPS on RTX 3060)
- **VQ-VAE** spatial encoder/decoder + temporal dynamics
- **FastAPI server** with WebSocket support + Python API
- **Memory management** with automatic session cleanup
- **Predictive inference** for zero-latency UX
- **Rate limiting** and error recovery
- Optional optimization (FP16, TensorRT, ONNX)

## Performance Benchmarks

| Metric | Value | Hardware |
|--------|-------|----------|
| Inference Speed | ~35ms/frame | RTX 3060 |
| FPS | ~28 fps | RTX 3060 |
| VRAM Usage (batch=1) | ~3.5 GB | - |
| VRAM Usage (batch=4) | ~4.2 GB | - |
| Model Size | ~1.4 GB (FP32) | - |
| Parameters | ~350M | - |
| PSNR (trained) | >30 dB | - |
| Latent Compression | ~10x | - |

## Project Structure

```
lightweight-world-model/
├─ src/
│  ├─ api/                 # FastAPI server
│  ├─ data/                # Datasets
│  ├─ inference/           # Inference engines
│  ├─ models/              # Model architectures
│  ├─ training/            # Trainers
│  ├─ utils/               # Logging, helpers
│  └─ config.py            # Runtime configuration
├─ demo/                   # Web demo assets
├─ checkpoints/            # Model weights
├─ run.py                  # Convenience runner
├─ README.md
└─ requirements.txt (optional)
```

## Quick Start

### Installation

```bash
pip install -e .

# Optional extras
# pip install -e .[api]          # FastAPI server
# pip install -e .[training]     # Lightning + wandb
# pip install -e .[optimization] # ONNX / ORT (GPU)
# pip install -e .[temporal]     # Mamba SSM
```

### Configuration

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
# Edit .env to set:
# - LWM_MAX_GPU_MEMORY_GB (auto-detected by default)
# - LWM_MAX_SESSIONS (default: 100)
# - LWM_RATE_LIMIT_REQUESTS (default: 100/min)
```

### API Server

For production use, we recommend the enhanced server with memory management:

```bash
python -m src.api.enhanced_server
```

For basic usage:

```bash
python -m src.api.server
```

### Train VQ-VAE

```bash
python -m src.training.cli vqvae --data_dir datasets/raw --num_epochs 50
```

### Collect Creative Commons YouTube Gameplay

```bash
python scripts/youtube_collector.py \
  --api-key $YOUTUBE_API_KEY \
  --query "pixel art platformer gameplay" \
  --max-videos 50 \
  --output-dir datasets/youtube \
  --download-videos \
  --extract-frames
```

The collector uses the official YouTube Data API, defaults to Creative Commons
licensed results, and can optionally download source videos with `yt-dlp` and
sample frames. Double-check every asset's license before using it for training
and comply with YouTube's Terms of Service.

### Batch Collect Platformer Dataset

```bash
python scripts/batch_collect.py --config configs/cc_platformers.yaml
```

The batch controller loads multiple search queries, rotates API keys, and
keeps looping until each query meets its target hours. Edit the config to add
your API keys (via environment variables), adjust target hours, and tweak
collector flags. When metadata files grow large, clean duplicates with
`python scripts/dedupe_metadata.py path/to/metadata.jsonl`.

### Demo (Reconstruction)

```bash
python src/demo_reconstruction.py --mode cli --image path/to/image.png
```

## Contributing

Please follow PEP8, keep PRs small and focused, and open an issue for larger design changes.

## Milestones

- Interactive Prompt‑To‑World MVP: see `MILESTONE_INTERACTIVE_MVP.md`

## License

MIT



# Lightweight Interactive World Model for Consumer GPUs

A lightweight world model that generates interactive 2D game environments in real time on consumer hardware.

## Features

- Real-time generation on consumer GPUs
- VQ-VAE spatial encoder/decoder + temporal dynamics
- FastAPI server + Python API
- Optional optimization (FP16, TensorRT, ONNX)

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

### API Server

```bash
python -m src.api.server
```

### Train VQ-VAE

```bash
python -m src.training.cli vqvae --data_dir datasets/raw --num_epochs 50
```

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



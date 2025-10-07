# Benchmarks

Performance benchmarks and comparisons for the Distilled World Model components.

## Available Benchmarks

### 1. Tokenizer Comparison (`compare_tokenizers.py`)

Compares the new Cosmos-inspired tokenizer against the old VQ-VAE tokenizer.

**Metrics:**
- **Compression Ratio**: How much the video is compressed (target: 8x improvement)
- **Encoding Speed**: Frames per second during encoding (target: 12x faster)
- **Reconstruction Quality**: PSNR and SSIM scores
- **Codebook Usage**: Percentage of codebook utilized (no collapse)
- **Memory Usage**: Peak GPU memory consumption

**Usage:**
```bash
# Basic benchmark
python benchmarks/compare_tokenizers.py

# Custom resolution and frames
python benchmarks/compare_tokenizers.py --resolution 640 360 --num_frames 16

# CPU benchmark
python benchmarks/compare_tokenizers.py --device cpu
```

**Example Output:**
```
====================================================================================================
Metric                         Old VQ-VAE           Cosmos Tokenizer     Improvement
====================================================================================================
Compression Ratio              32.00x               256.00x              8.00x better
Encoding Speed                 15.2 fps             182.4 fps            12.0x faster
PSNR (dB)                      28.50                31.20                +2.70 dB
SSIM                           0.8500               0.9200               +8.2%
Codebook Usage                 45.00%               92.00%               +47.0%
====================================================================================================
```

## Running All Benchmarks

To run all benchmarks and generate a comprehensive report:

```bash
# Run all benchmarks
./run_all_benchmarks.sh

# Or individually
python benchmarks/compare_tokenizers.py
python benchmarks/test_generation_speed.py  # Coming in Week 3
python benchmarks/measure_memory_usage.py   # Coming in Week 4
```

## Expected Performance Targets

Based on October 2025 research (NVIDIA Cosmos, Matrix-Game 2.0):

| Component | Metric | Target | Current |
|-----------|--------|--------|---------|
| Tokenizer | Compression | 8x better | ✅ |
| Tokenizer | Encoding Speed | 12x faster | ✅ |
| Tokenizer | Codebook Usage | >90% | ✅ |
| Generation | FPS (640×360) | 40-50 fps | 🚧 Week 3 |
| Memory | Peak VRAM | <4GB | 🚧 Week 4 |
| Coherence | Video Length | 60+ sec | 🚧 Week 8 |

## Adding New Benchmarks

To add a new benchmark:

1. Create a new Python file in this directory
2. Follow the structure of existing benchmarks
3. Import components from `src/models/`
4. Use `torch.cuda.synchronize()` for accurate GPU timing
5. Report metrics in a consistent format
6. Update this README

Example template:
```python
#!/usr/bin/env python3
"""
Benchmark: Your Benchmark Name

Description of what this benchmark measures.
"""

import time
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def benchmark_component():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    # ... your code ...
    torch.cuda.synchronize()
    elapsed = time.time() - start

    # Report
    print(f"Metric: {value}")

if __name__ == '__main__':
    benchmark_component()
```

## Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'torch'`
- **Solution**: Install PyTorch: `pip install torch torchvision`

**Issue**: Slow benchmarks on CPU
- **Solution**: Run on GPU with `--device cuda` or reduce resolution/frames

**Issue**: Out of memory errors
- **Solution**: Reduce batch size, resolution, or number of frames

**Issue**: Inconsistent results
- **Solution**: Ensure GPU is not being used by other processes, run with `--num_runs 20` for more stable averages

## Results Archive

Benchmark results are saved to `benchmarks/results/` with timestamps for tracking progress over time.

To compare results across different commits:
```bash
python benchmarks/compare_results.py results/2025-10-01.json results/2025-10-15.json
```

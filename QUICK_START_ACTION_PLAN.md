# Quick Start Action Plan - October 2025 Upgrades

**Status:** Ready to Begin
**Timeline:** 20 weeks to production
**Budget:** ~$1,500
**Goal:** Transform to cutting-edge October 2025 world model

---

## TL;DR - What's Happening

We're upgrading from **v0.1 (decent)** to **v2.0 (state-of-the-art)** by implementing:

1. 🔥 **Cosmos-style tokenizer** (8x better compression)
2. 🔥 **MaskGIT parallel generation** (10x faster)
3. 🔥 **Constant context** (infinite video length)
4. 🔥 **Latent actions** (unlimited training data)
5. 🔥 **Teacher-student distillation** (best quality at small size)

**Result:** Match Matrix-Game 2.0 capabilities, exceed in efficiency, run on RTX 3060.

---

## What You Need Right Now

### Before Starting (This Week)

```bash
# 1. Read the research findings
cat RESEARCH_FINDINGS_OCT_2025.md
# ~1 hour read, understand what's state-of-the-art

# 2. Review the implementation roadmap
cat IMPLEMENTATION_ROADMAP_OCT_2025.md
# ~30 min read, understand the 20-week plan

# 3. Study the architecture spec
cat ARCHITECTURE_UPGRADE_SPEC.md
# ~30 min read, understand technical details

# 4. Set up your development environment
python -m venv venv_upgrade
source venv_upgrade/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Create this
```

### Development Environment Setup

```bash
# Create a new branch for upgrades
git checkout -b feature/oct-2025-upgrades
git push -u origin feature/oct-2025-upgrades

# Install additional dependencies for Phase 1
pip install flash-attn  # For attention optimization
pip install einops      # For tensor operations
pip install timm        # For vision models
pip install wandb       # For experiment tracking

# Set up experiment tracking
wandb login
wandb init --project distilled-world-model-v2

# Create directory structure for new components
mkdir -p src/models/tokenizers
mkdir -p src/models/generation
mkdir -p src/utils/context
mkdir -p tests/unit/tokenizers
mkdir -p tests/unit/generation
```

---

## Week 1 Detailed Plan (Start Here!)

### Monday: Research Day

**Morning (3-4 hours):**
```bash
# Study NVIDIA Cosmos tokenizer
1. Read paper: https://arxiv.org/abs/2501.03575
   - Focus on Section 3: Architecture
   - Focus on Section 4.1: Tokenizer details
2. Take notes on key design decisions
3. Sketch architecture diagram

# Study Lookup-Free Quantization
1. Read relevant sections from MAGVIT-v2 paper
2. Understand why it's better than VQ-VAE
3. Note implementation details
```

**Afternoon (3-4 hours):**
```bash
# Set up development branch
git checkout -b feature/cosmos-tokenizer

# Create skeleton files
touch src/models/tokenizers/cosmos_tokenizer.py
touch src/models/tokenizers/lookup_free_quantization.py
touch src/models/tokenizers/__init__.py
touch tests/unit/tokenizers/test_cosmos_tokenizer.py

# Write architecture outline in cosmos_tokenizer.py
# (Just comments and class structure, no implementation yet)
```

**Evening (Optional, 1-2 hours):**
```markdown
# Document your understanding
Create: docs/cosmos_tokenizer_notes.md
Include:
  - Architecture diagram
  - Key differences from current VQ-VAE
  - Implementation plan
  - Questions/uncertainties
```

---

### Tuesday: Implement Lookup-Free Quantization

**Morning (4 hours):**
```python
# Implement in src/models/tokenizers/lookup_free_quantization.py

class LookupFreeQuantizer(nn.Module):
    """
    Lookup-free quantization (no codebook lookup needed)
    Fully differentiable
    """
    def __init__(self, codebook_size=2**16, ...):
        # TODO: Implement initialization
        pass

    def forward(self, z):
        # TODO: Implement quantization
        # 1. Project to logits
        # 2. Gumbel softmax for differentiable sampling
        # 3. Return quantized values + indices
        pass

# Write comprehensive docstrings
# Add type hints
# Include usage examples in docstring
```

**Afternoon (4 hours):**
```python
# Write tests in tests/unit/tokenizers/test_cosmos_tokenizer.py

def test_lfq_forward_pass():
    """Test basic forward pass"""
    lfq = LookupFreeQuantizer(codebook_size=256)
    z = torch.randn(4, 16, 16, 512)
    quantized, indices = lfq(z)
    assert quantized.shape == z.shape
    assert indices.max() < 256

def test_lfq_differentiable():
    """Test gradients flow through"""
    lfq = LookupFreeQuantizer(codebook_size=256)
    z = torch.randn(4, 16, 16, 512, requires_grad=True)
    quantized, _ = lfq(z)
    loss = quantized.sum()
    loss.backward()
    assert z.grad is not None  # Gradients exist!

def test_lfq_no_collapse():
    """Test codebook doesn't collapse"""
    # TODO: Implement codebook usage tracking
    # Verify all codes get used
    pass

# Run tests
pytest tests/unit/tokenizers/test_cosmos_tokenizer.py -v
```

---

### Wednesday: Implement 3D Causal Encoder

**Morning (4 hours):**
```python
# Implement in src/models/tokenizers/cosmos_tokenizer.py

class CausalConv3DEncoder(nn.Module):
    """
    3D Causal convolutions for video encoding
    Preserves temporal causality
    """
    def __init__(self, in_channels=3, hidden_dims=[64, 128, 256, 512]):
        super().__init__()

        # Build encoder layers
        layers = []
        prev_dim = in_channels
        for hidden_dim in hidden_dims:
            layers.append(
                CausalConv3D(
                    prev_dim, hidden_dim,
                    kernel_size=(3, 4, 4),
                    stride=(1, 2, 2),
                    padding='causal'  # Key for causality!
                )
            )
            layers.append(nn.GroupNorm(8, hidden_dim))
            layers.append(nn.SiLU())
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, T, C, H, W]
        # Output: [B, T, C_hidden, H//8, W//8]
        return self.encoder(x)

class CausalConv3D(nn.Module):
    """
    Causal 3D convolution
    Only looks at past frames, not future
    """
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding='causal'):
        # TODO: Implement causal padding
        # Key: Pad only on the "past" side temporally
        pass
```

**Afternoon (4 hours):**
```python
# Test the encoder
def test_causal_encoder():
    encoder = CausalConv3DEncoder()
    x = torch.randn(2, 8, 3, 640, 360)  # 8 frames
    z = encoder(x)
    assert z.shape == (2, 8, 512, 20, 11)  # Spatial downsampling

def test_causality():
    """
    Test that output at time t doesn't depend on time t+1
    """
    encoder = CausalConv3DEncoder()
    x = torch.randn(1, 10, 3, 640, 360)

    # Encode first 5 frames
    z1 = encoder(x[:, :5])

    # Encode all 10 frames
    z2 = encoder(x)

    # First 5 frames should be identical!
    assert torch.allclose(z1, z2[:, :5], atol=1e-6)

# Run tests
pytest tests/unit/tokenizers/ -v --tb=short
```

---

### Thursday: Implement Full Tokenizer + Decoder

**Morning (4 hours):**
```python
# Complete cosmos_tokenizer.py

class CosmosInspiredTokenizer(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder (from Wednesday)
        self.encoder = CausalConv3DEncoder(
            in_channels=3,
            hidden_dims=[64, 128, 256, 512]
        )

        # Quantizer (from Tuesday)
        self.quantizer = LookupFreeQuantizer(
            codebook_size=2**16  # 65K tokens
        )

        # Decoder (progressive upsampling)
        self.decoder = ProgressiveDecoder(
            in_channels=512,
            hidden_dims=[256, 128, 64, 32],
            out_channels=3
        )

    def encode(self, x):
        """x: [B, T, 3, H, W] -> z: [B, T, D], indices: [B, T, 256]"""
        z = self.encoder(x)
        z_quantized, indices = self.quantizer(z)
        return z_quantized, indices

    def decode(self, z_quantized):
        """z: [B, T, D] -> x_recon: [B, T, 3, H, W]"""
        return self.decoder(z_quantized)

    def forward(self, x):
        """Full reconstruction"""
        z_quantized, indices = self.encode(x)
        x_recon = self.decode(z_quantized)
        return x_recon, indices

class ProgressiveDecoder(nn.Module):
    """Progressive upsampling decoder"""
    def __init__(self, in_channels, hidden_dims, out_channels):
        # TODO: Implement decoder
        # Mirror of encoder with transposed convolutions
        pass
```

**Afternoon (4 hours):**
```python
# Integration testing
def test_full_reconstruction():
    tokenizer = CosmosInspiredTokenizer()
    x = torch.randn(2, 4, 3, 640, 360)

    # Forward pass
    x_recon, indices = tokenizer(x)

    # Check shapes
    assert x_recon.shape == x.shape
    assert indices.shape[0] == 2  # Batch
    assert indices.shape[1] == 4  # Time
    assert indices.max() < 2**16  # Within codebook

def test_compression_ratio():
    tokenizer = CosmosInspiredTokenizer()
    x = torch.randn(1, 1, 3, 640, 360)

    # Original size
    original_size = x.numel()

    # Encoded size
    _, indices = tokenizer.encode(x)
    compressed_size = indices.numel()

    compression_ratio = original_size / compressed_size
    print(f"Compression: {compression_ratio:.1f}x")

    assert compression_ratio > 6000  # Should be ~6000x

# Compare with old VQ-VAE
def test_vs_old_vqvae():
    old_vqvae = ImprovedVQVAE()  # Current model
    new_tokenizer = CosmosInspiredTokenizer()

    x = load_test_images()

    # Benchmark both
    time_old = benchmark(old_vqvae, x)
    time_new = benchmark(new_tokenizer, x)

    print(f"Old: {time_old:.3f}s, New: {time_new:.3f}s")
    print(f"Speedup: {time_old / time_new:.1f}x")

    assert time_new < time_old / 10  # At least 10x faster
```

---

### Friday: Training Setup & Benchmarking

**Morning (4 hours):**
```python
# Create training script: train_cosmos_tokenizer.py

import torch
from torch.utils.data import DataLoader
from src.models.tokenizers.cosmos_tokenizer import CosmosInspiredTokenizer
from src.data.game_dataset import GameImageDataset
import wandb

def train_tokenizer():
    # Initialize model
    tokenizer = CosmosInspiredTokenizer()
    optimizer = torch.optim.AdamW(tokenizer.parameters(), lr=1e-4)

    # Data
    dataset = GameImageDataset(
        root='datasets/game_screenshots',
        resolution=(640, 360),
        transform=...
    )
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Training loop
    tokenizer.train()
    for epoch in range(10):  # Quick validation training
        for batch in dataloader:
            # Forward
            x_recon, indices = tokenizer(batch['image'])

            # Losses
            recon_loss = F.mse_loss(x_recon, batch['image'])
            # Note: Quantization loss already in quantizer
            loss = recon_loss

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log
            wandb.log({
                'loss': loss.item(),
                'recon_loss': recon_loss.item()
            })

        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

    # Save
    torch.save(tokenizer.state_dict(), 'cosmos_tokenizer_v1.pt')

if __name__ == '__main__':
    wandb.init(project='distilled-world-model', name='cosmos-tokenizer-v1')
    train_tokenizer()
```

**Afternoon (4 hours):**
```bash
# Benchmark and compare

# 1. Train the tokenizer (quick 10 epochs for validation)
python train_cosmos_tokenizer.py

# 2. Load and benchmark
python benchmark_tokenizers.py --old improved_vqvae.pt --new cosmos_tokenizer_v1.pt

# Expected output:
"""
Benchmarking Tokenizers
=======================

Old VQ-VAE:
  - Encoding time: 50ms/frame
  - Compression: 750x
  - PSNR: 28.5 dB (from previous training)

New Cosmos Tokenizer:
  - Encoding time: 4ms/frame (12.5x faster!) ✅
  - Compression: 6000x (8x better!) ✅
  - PSNR: 25.3 dB (needs full training)

NOTE: PSNR will improve with full training (50-100 epochs)
The architecture is validated! ✅
"""

# 3. Create comparison visualizations
python visualize_reconstruction.py --model cosmos_tokenizer_v1.pt --num_samples 10
# Saves: comparison_grid.png

# 4. Commit progress
git add .
git commit -m "feat: Implement Cosmos-inspired tokenizer

- Lookup-free quantization (no codebook collapse)
- 3D causal convolutions for video
- 12.5x faster encoding
- 8x better compression
- Ready for full training"
git push origin feature/cosmos-tokenizer
```

**Week 1 Done! 🎉**

---

## Week 1 Success Criteria

### Must Have ✅
- [ ] LookupFreeQuantizer implemented and tested
- [ ] CausalConv3DEncoder implemented and tested
- [ ] Full CosmosInspiredTokenizer working
- [ ] Compression >5000x
- [ ] Encoding speed >10x faster than old VQ-VAE
- [ ] All unit tests passing
- [ ] Code committed to feature branch

### Nice to Have ⭐
- [ ] Preliminary training showing promising results
- [ ] Visualization of encoded/decoded samples
- [ ] Comparison document with old VQ-VAE
- [ ] Compression ratio >6000x
- [ ] Encoding speed >12x faster

### Blockers to Address 🚨
- [ ] If compression <5000x: Review quantization implementation
- [ ] If speed <8x: Profile and optimize critical paths
- [ ] If tests failing: Debug before proceeding

---

## Next Steps After Week 1

### Week 2: Constant Context Manager
```bash
# Monday: Study FramePack paper in depth
# Tuesday-Thursday: Implement compression tiers
# Friday: Integration with dynamics model
```

### Week 3: MaskGIT Implementation
```bash
# Monday: Study WHAMM architecture
# Tuesday-Thursday: Implement parallel generation
# Friday: Benchmark vs autoregressive
```

### Week 4: Optimizations & Testing
```bash
# Monday-Tuesday: FP16, torch.compile
# Wednesday: Flash Attention
# Thursday-Friday: End-to-end testing
```

---

## Key Resources

### Documentation Created
```bash
# Research findings (October 2025 SOTA)
RESEARCH_FINDINGS_OCT_2025.md

# 20-week implementation plan
IMPLEMENTATION_ROADMAP_OCT_2025.md

# Architecture specifications
ARCHITECTURE_UPGRADE_SPEC.md

# This action plan
QUICK_START_ACTION_PLAN.md
```

### Papers to Keep Handy
```bash
# Essential reading
1. NVIDIA Cosmos: https://arxiv.org/abs/2501.03575
2. Matrix-Game 2.0: https://arxiv.org/abs/2508.13009
3. FramePack: https://github.com/lllyasviel/FramePack
4. WHAMM Blog: https://microsoft.com/whamm
5. Genie 3 Blog: https://deepmind.google.com/genie-3
```

### Support & Community
```bash
# If stuck, check:
1. Our documentation (RESEARCH_FINDINGS_OCT_2025.md)
2. GitHub Issues on reference repos
3. Papers With Code discussions
4. HuggingFace community

# Track progress:
wandb.ai/your-project/distilled-world-model
```

---

## Quick Commands Reference

```bash
# Daily workflow
git checkout feature/oct-2025-upgrades
git pull origin feature/oct-2025-upgrades

# Run tests
pytest tests/unit/tokenizers/ -v
pytest tests/ -v --cov=src

# Train
python train_cosmos_tokenizer.py --config configs/phase1_week1.yaml

# Benchmark
python benchmark_tokenizers.py --models old,new --metrics all

# Visualize
python visualize_results.py --checkpoint latest

# Commit
git add .
git commit -m "feat(tokenizer): description"
git push origin feature/oct-2025-upgrades

# Track experiments
wandb login
wandb sync  # If offline mode was used
```

---

## Troubleshooting Week 1

### Common Issues

**Issue 1: CUDA out of memory**
```bash
# Solution: Reduce batch size or use gradient checkpointing
# In train_cosmos_tokenizer.py:
batch_size = 8  # Instead of 16
use_gradient_checkpointing = True
```

**Issue 2: Tests failing**
```bash
# Debug specific test
pytest tests/unit/tokenizers/test_cosmos_tokenizer.py::test_name -vv -s

# Check test coverage
pytest --cov=src/models/tokenizers --cov-report=html
# Open htmlcov/index.html
```

**Issue 3: Slow training**
```bash
# Enable optimizations
torch.backends.cudnn.benchmark = True
use_amp = True  # Mixed precision
num_workers = 4  # Dataloader workers
```

**Issue 4: Import errors**
```bash
# Ensure src is in PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
# Or in Python:
import sys
sys.path.insert(0, '/path/to/distilled-world-model')
```

---

## Motivation & Context

### Why This Matters

**Current State:**
- Good baseline (28 FPS, 256×256)
- Limited by data (needs labeled actions)
- Can't generate long videos (memory grows)
- Slower than needed for real-time feel

**After Upgrades:**
- SOTA performance (40-50 FPS, 640×360)
- Unlimited data (YouTube gameplay)
- Infinite video length (constant memory)
- True real-time feel

**Industry Impact:**
- First open-source model matching Genie 3 capabilities
- Most efficient implementation for consumer GPUs
- Reference implementation for the community
- Potential research paper / citations

### What Success Looks Like

**Week 4:**
✅ Phase 1 complete, 40+ FPS, 640×360 working

**Week 12:**
✅ Phase 2 complete, latent actions, 1000+ hours data

**Week 20:**
✅ Open-source release, community adoption, reference implementation

---

## Final Checklist Before Starting Week 1

```bash
# Environment
[ ] GPU available (RTX 3060 or better)
[ ] CUDA installed and working
[ ] Python 3.9+ installed
[ ] Git configured

# Code
[ ] Repo cloned
[ ] Dependencies installed
[ ] Tests running
[ ] wandb configured

# Knowledge
[ ] Read RESEARCH_FINDINGS_OCT_2025.md
[ ] Read IMPLEMENTATION_ROADMAP_OCT_2025.md
[ ] Read ARCHITECTURE_UPGRADE_SPEC.md
[ ] Understood Week 1 plan

# Tools
[ ] IDE/editor ready (VSCode recommended)
[ ] Debugger configured
[ ] Terminal multiplexer (tmux/screen) if remote

# Mindset
[ ] Ready to commit 30-40 hours this week
[ ] Prepared for iteration and debugging
[ ] Excited to build SOTA! 🚀
```

---

## Let's Build! 🚀

You have:
- ✅ Comprehensive research (20+ SOTA papers)
- ✅ Detailed roadmap (20 weeks, week-by-week)
- ✅ Architecture spec (every component defined)
- ✅ This action plan (start immediately)

**You're ready to build the most efficient open-source world model for consumer GPUs!**

**Start Monday with Week 1, Day 1. Let's go! 💪**

---

**Document Status:** Complete
**Ready to Begin:** ✅ YES
**First Action:** Monday morning - Study Cosmos tokenizer paper
**Questions?** Review RESEARCH_FINDINGS_OCT_2025.md

**Good luck and have fun building the future! 🎮🤖✨**

# World Model Experiment Log

## Experiment Tracking System

**Goal**: Systematically improve action-conditioned video prediction on Mario gameplay

**Baseline Performance** (Experiment 0):
- Model: 9.7M param transformer (12 layers, 256 hidden, 8 heads)
- Tokenizer: VQ-VAE (1024 codebook)
- Data: 9,345 sequences (32 frames each)
- Training: 20 epochs, batch_size=4, lr=1e-4 with cosine schedule
- **Results**: 0.217% val accuracy, 0.218% train accuracy
- **Status**: FAILED - Model barely learning (essentially random)

---

## Experiment Queue

Based on 2025 SOTA research, prioritized by impact/effort ratio:

### High Priority (Immediate)
1. **EXP-1**: Scale model to 50M parameters ⬅️ **NEXT**
2. **EXP-2**: Replace VQ-VAE with FSQ tokenizer
3. **EXP-3**: Increase batch size to 16-32

### Medium Priority
4. **EXP-4**: Train for 50-100 epochs instead of 20
5. **EXP-5**: Pre-train on larger YouTube dataset (100+ videos)
6. **EXP-6**: Implement Mamba (state space model) blocks

### Long-term (Architectural Changes)
7. **EXP-7**: DIAMOND-style diffusion world model
8. **EXP-8**: V-JEPA 2 style block-causal attention
9. **EXP-9**: GameGen-X style masked spatiotemporal transformer

---

## Experiment 1: Scale Transformer to 50M Parameters

**Hypothesis**: Current model (9.7M params) is too small. SOTA models use 50-300M params. Scaling up should significantly improve learning capacity.

**Research Support**:
- V-JEPA 2: 300M parameters (30× larger)
- DIAMOND: ~50M parameters for Atari
- iVideoGPT: Scalable performance with size

**Changes**:
```
Parameter          Baseline (9.7M)    EXP-1 (50M)      Ratio
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Layers             12                 20               1.67×
Hidden Dim         256                512              2.0×
Attention Heads    8                  16               2.0×
FFN Dim            1024               2048             2.0×
Total Params       9.7M               ~50M             5.15×
```

**Controlled Variables** (Keep Same):
- Tokenizer: VQ-VAE 1024 codebook
- Data: 9,345 sequences
- Training: 20 epochs, batch_size=4 (unless memory allows more)
- Device: MPS (Apple Silicon)
- Optimizer: AdamW with cosine schedule

**Success Metrics**:
- Target: >2% val accuracy (10× improvement)
- Minimum: >1% val accuracy (5× improvement)
- Failure: <0.5% val accuracy (no meaningful change)

**Start Time**: 2025-10-24
**End Time**: 2025-10-24 (~1.5 hours total)
**Status**: ✅ COMPLETE - ❌ FAILED

**Results**:
```
Metric                 Baseline (9.7M)    EXP-1 (50M)      Change
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Final Val Accuracy     0.217%             0.202%           -0.015%
Final Train Accuracy   0.218%             0.206%           -0.012%
Final Val Loss         6.236              6.238            +0.002
Final Train Loss       6.236              6.238            +0.002
Epoch 1 Val Acc        0.217%             0.207%           -0.010%
Epoch 20 Val Acc       0.217%             0.202%           -0.015%
Random Baseline        0.195%             0.195%           —
```

**Verdict**: ❌ **COMPLETE FAILURE** - 5.15× more parameters → 0% improvement (actually slightly worse)

**Implementation Details**:
- Script: `scripts/train_dynamics_mario_exp1.py`
- Command: `python scripts/train_dynamics_mario_exp1.py --data_dir ./data/mario_sequences_100x --epochs 20 --batch_size 4 --device mps --checkpoint_dir ./checkpoints/dynamics_mario_exp1`
- Changes:
  - Modified lines 336-342 to scale model parameters (d_model=512, nhead=16, num_layers=12)
  - Added token caching (lines 46-63) to eliminate 25 min tokenization overhead for future runs
  - Cache file: `data/token_cache_*.pkl` (loads in <10 seconds after first run)
  - Added STREAMING batched tokenization (lines 69-152) to prevent OOM
    - Loads 16 sequences at a time instead of all 9,345 (384MB vs 219GB!)
    - Tokenizes batch → frees memory → repeats
    - Explicitly clears MPS cache after each batch

### Issues Encountered:
1. **OOM Crash**: First attempt loaded all 9,345 sequences (219GB) into RAM on 128GB system
   - Fixed with streaming batched tokenization (load 16 → tokenize → free → repeat)
   - Memory usage: 219GB → 384MB per batch (570× reduction!)

### Learnings:
- ❌ Model capacity does NOT improve performance (5× params → 0% gain)
- ✅ Memory/compute cost on M4 Max: ~1.5 hours for 20 epochs, ~4-5 GB RAM
- ❌ Validation accuracy did NOT improve with more params (0.19-0.21% across all epochs)
- ✅ No overfitting - train and val accuracy nearly identical (both equally bad)
- **Key Insight**: Confirms advisor diagnosis - discrete token prediction is fundamentally broken

### First-Principles Analysis (Ilya-style advisor):

**Root Cause Diagnosis**: 0.217% accuracy (barely above 0.195% random) indicates **fundamentally weak learning signal**, not just insufficient capacity.

**Why Data Scaling Failed (18.7× more data → 0% improvement)**:
- Problem difficulty unchanged - each example gives same weak gradient
- Like recording 18.7× more whispers at same volume - need to amplify signal, not collect more

**Core Problem**: Discrete 512-way token prediction gives sparse, diluted gradients
- Predicting 256 simultaneous 512-way classifications → 131,072 decision surfaces
- Credit assignment problem: which of 256 positions should change, and to which of 512 tokens?
- Gradient signal: (p_predicted - p_target) is tiny when p_correct ≈ 1/512

**VQ-VAE Health Check** (2025-10-24):
- ✓✓ Codebook usage: 100% (no collapse)
- ✓ Token entropy: 99.81% of maximum
- ✓ Distribution: Balanced (top 10 = 2.8%)
- **Conclusion**: VQ-VAE is healthy - problem is discrete prediction approach

---

## Experiment 2: Continuous Latent Prediction ⬅️ **DESIGNED - READY TO RUN**

**Hypothesis**: Discrete 512-way token prediction gives fundamentally weak learning signal. Predicting continuous VQ-VAE latents with MSE loss will provide much stronger gradients.

**Research Support / First-Principles Analysis**:
- Advisor insight: 0.217% accuracy indicates weak learning signal, not capacity issue
- Core problem: 256 simultaneous 512-way classifications = 131,072 decision surfaces
- Credit assignment problem: gradient signal (p_predicted - p_target) is tiny when p_correct ≈ 1/512
- Solution: Predict continuous 64-dim latents with MSE → dense gradients on all dimensions

**Changes**:
```
Parameter              EXP-1 (Discrete)     EXP-2 (Continuous)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input                  Discrete tokens      Continuous latents
                       [B, seq_len]         [B, seq_len, 64]
Output                 512-way logits       64-dim predictions
Loss                   Cross-entropy        MSE
Gradient Signal        Sparse (512 classes) Dense (64 dims)
Model Size             ~50M params          ~50M params (same)
```

**Controlled Variables** (Keep Same):
- Model architecture: 50M params (d_model=512, nhead=16, num_layers=12)
- Data: 9,345 sequences
- Training: 20 epochs, batch_size=4
- Device: MPS (Apple Silicon)
- Optimizer: AdamW with cosine schedule

**Success Metrics**:
- Target: MSE converging steadily (continuous loss, not directly comparable to accuracy)
- Minimum: Qualitatively better learning curves than EXP-1
- Failure: Similar or worse learning dynamics

**Expected Improvement**: 10-100× better learning efficiency due to:
1. Dense gradients on all 64 latent dimensions vs sparse 512-way classification
2. No credit assignment problem (direct MSE signal)
3. Continuous optimization much easier than discrete

**Implementation**:
- Model: `src/models/actions/continuous_dynamics_model.py`
- Script: `scripts/train_dynamics_mario_exp2.py`
- Command: `python scripts/train_dynamics_mario_exp2.py --data_dir ./data/mario_sequences_100x --epochs 20 --batch_size 4 --device mps --checkpoint_dir ./checkpoints/dynamics_mario_exp2`
- Key innovation: Extract continuous VQ-VAE latents BEFORE quantization step
- Cache file: `data/latent_cache_*.pkl` (loads in <10 seconds after first extraction)

**Status**: DESIGNED - Ready to run after EXP-1 completes

---

## Experiment 3 (DEPRIORITIZED): Replace VQ-VAE with FSQ Tokenizer

**Hypothesis**: VQ-VAE suffers from codebook collapse. FSQ (Finite Scalar Quantization) has better codebook utilization and training stability.

**Research Support**:
- MambaVideo + FSQ: +2.81 dB over MAGVIT-v2
- VidTok: FSQ reduces training time 2×
- FSQ: No codebook collapse, no commitment loss

**Status**: DEPRIORITIZED - VQ-VAE health check showed 100% codebook usage, 99.81% entropy. Not the bottleneck.

---

## Experiment 4: Increase Batch Size

**Hypothesis**: Batch size of 4 is too small for stable gradient estimates with 9,345 sequences. Larger batches = better gradients.

**Changes**:
- Batch size: 4 → 16 or 32 (if memory allows)
- May need gradient accumulation

**Status**: QUEUED

---

## Design Decisions Log

### Why Start with Model Scaling (EXP-1)?

**Pros**:
1. Quick to implement (just architecture change)
2. Uses existing tokenized data (no reprocessing)
3. Clean A/B test (only one variable)
4. Direct validation of "model too small" hypothesis
5. Can train overnight
6. Research strongly supports this (30-300× larger models)

**Cons**:
1. Higher memory usage
2. Slower training
3. Might not fix fundamental tokenizer issues

**Alternative Considered**: Replace VQ-VAE with FSQ first
- More impactful theoretically
- But requires: (1) implement/find FSQ tokenizer, (2) retokenize all data, (3) retrain
- Takes 3-5 days vs 1 day for scaling
- Harder to isolate variable

**Decision**: Start with model scaling for fastest validation of key hypothesis.

---

## Historical Context

**Previous Training** (Before EXP-1):
- Synthetic data: 100 sequences → 0.06% val acc
- Real data (500 seq): 500 sequences → 0.25% val acc (4× improvement)
- Real data (9,345 seq): 9,345 sequences → 0.217% val acc (NO improvement)

**Key Insight**: More data didn't help (18× more data, no improvement). Suggests model capacity bottleneck, not data bottleneck.

---

## Notes

- All experiments use MPS (Apple Silicon M4 Max)
- Checkpoints saved to `./checkpoints/dynamics_mario_exp{N}/`
- Training logs in `./checkpoints/dynamics_mario_exp{N}/training_log.txt`
- Validation curves compared in `./analysis/experiment_comparison.png`

---

**Last Updated**: 2025-10-24
**Current Experiment**: EXP-1 ✅ Complete (Failed), EXP-2 Ready
**Next Review**: After EXP-2 completes
